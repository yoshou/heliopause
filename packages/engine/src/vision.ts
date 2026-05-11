import type {
  GgmlTypeName,
} from "./gguf";
import {
  buildGemma4VisionManifest,
  type Gemma4VisionManifest,
} from "./model";
import {
  gqaAttention,
  rmsNorm,
} from "./ops";
import {
  dequantizeRow,
} from "./quant";
import {
  matMulQuantizedWasm,
} from "./runner/cpu/wasm-kernels";
import {
  GgufTensorReader,
  tensorByteLength,
} from "./tensor-reader";

export type Gemma4VisionPixelValues = {
  values: Float32Array;
  width: number;
  height: number;
};

export type Gemma4VisionResize = {
  width: number;
  height: number;
  outputTokenCount: number;
};

export type Gemma4VisionEncodeResult = {
  hidden: Float32Array;
  tokenCount: number;
  width: number;
  height: number;
};

export class Gemma4VisionSession {
  readonly tensorReader: GgufTensorReader;
  readonly manifest: Gemma4VisionManifest;
  readonly epsilon: number;

  private readonly f32TensorCache = new Map<string, Float32Array>();
  private readonly weightBytesCache = new Map<string, Uint8Array>();

  constructor(tensorReader: GgufTensorReader) {
    this.tensorReader = tensorReader;
    this.manifest = buildGemma4VisionManifest(tensorReader.metadata);
    this.epsilon = this.manifest.layerNormEpsilon;
  }

  getTensor(name: string) {
    return this.tensorReader.getTensor(name);
  }

  hasTensor(name: string): boolean {
    return this.tensorReader.metadata.tensors.some((tensor) => tensor.name === name);
  }

  async readF32Tensor(name: string): Promise<Float32Array> {
    const cached = this.f32TensorCache.get(name);
    if (cached) {
      return cached;
    }
    const tensor = this.getTensor(name);
    if (tensor.type !== "F32") {
      throw new Error(`${name} must be F32, got ${tensor.type}`);
    }
    const bytes = await this.tensorReader.readTensorBytes(name);
    const value = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
    this.f32TensorCache.set(name, value);
    return value;
  }

  async readWeightBytes(name: string): Promise<Uint8Array> {
    const cached = this.weightBytesCache.get(name);
    if (cached) {
      return cached;
    }
    const bytes = await this.tensorReader.readTensorBytes(name);
    this.weightBytesCache.set(name, bytes);
    return bytes;
  }
}

export function createGemma4VisionSession(tensorReader: GgufTensorReader): Gemma4VisionSession {
  return new Gemma4VisionSession(tensorReader);
}

export function calculateGemma4VisionResize(
  manifest: Pick<Gemma4VisionManifest, "patchSize" | "spatialMergeSize" | "imageMinTokens" | "imageMaxTokens">,
  sourceWidth: number,
  sourceHeight: number,
): Gemma4VisionResize {
  const alignSize = manifest.patchSize * manifest.spatialMergeSize;
  const minPixels = manifest.imageMinTokens * alignSize * alignSize;
  const maxPixels = manifest.imageMaxTokens * alignSize * alignSize;
  const width = Math.max(1, Math.floor(sourceWidth));
  const height = Math.max(1, Math.floor(sourceHeight));

  let resizedHeight = Math.max(alignSize, roundByFactor(height, alignSize));
  let resizedWidth = Math.max(alignSize, roundByFactor(width, alignSize));
  if (resizedHeight * resizedWidth > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    resizedHeight = Math.max(alignSize, floorByFactor(height / beta, alignSize));
    resizedWidth = Math.max(alignSize, floorByFactor(width / beta, alignSize));
  } else if (resizedHeight * resizedWidth < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    resizedHeight = ceilByFactor(height * beta, alignSize);
    resizedWidth = ceilByFactor(width * beta, alignSize);
  }

  return {
    width: resizedWidth,
    height: resizedHeight,
    outputTokenCount: (resizedWidth / alignSize) * (resizedHeight / alignSize),
  };
}

export async function preprocessGemma4VisionImageFile(
  file: Blob,
  manifest: Gemma4VisionManifest,
): Promise<Gemma4VisionPixelValues> {
  const bitmap = await createImageBitmap(file);
  try {
    const rgba = imageBitmapToRgba(bitmap);
    const resize = calculateGemma4VisionResize(manifest, bitmap.width, bitmap.height);
    const resized = resizeRgbaBilinear(rgba, bitmap.width, bitmap.height, resize.width, resize.height);
    const values = new Float32Array(resize.width * resize.height * 3);
    for (let pixel = 0; pixel < resize.width * resize.height; pixel += 1) {
      for (let channel = 0; channel < 3; channel += 1) {
        values[pixel * 3 + channel] =
          ((resized[pixel * 4 + channel] ?? 0) / 255 - (manifest.imageMean[channel] ?? 0)) /
          (manifest.imageStd[channel] ?? 1);
      }
    }
    return { values, width: resize.width, height: resize.height };
  } finally {
    bitmap.close();
  }
}

export async function runGemma4VisionEncoder(
  session: Gemma4VisionSession,
  pixels: Gemma4VisionPixelValues,
): Promise<Gemma4VisionEncodeResult> {
  const manifest = session.manifest;
  const patchGridX = pixels.width / manifest.patchSize;
  const patchGridY = pixels.height / manifest.patchSize;
  if (!Number.isInteger(patchGridX) || !Number.isInteger(patchGridY)) {
    throw new Error(`Vision image size must be patch-aligned, got ${pixels.width}x${pixels.height}`);
  }
  if (patchGridX % manifest.spatialMergeSize !== 0 || patchGridY % manifest.spatialMergeSize !== 0) {
    throw new Error(`Vision image size must be merge-aligned, got ${pixels.width}x${pixels.height}`);
  }

  const nPatches = patchGridX * patchGridY;
  let hidden = await patchEmbed(session, pixels, patchGridX, patchGridY);
  hidden = await addPositionEmbeddings(session, hidden, patchGridX, nPatches);
  for (let layer = 0; layer < manifest.blockCount; layer += 1) {
    hidden = await forwardVisionLayer(session, hidden, layer, patchGridX, nPatches);
  }

  hidden = averagePoolVisionTokens(
    hidden,
    patchGridX,
    patchGridY,
    manifest.embeddingLength,
    manifest.spatialMergeSize,
  );
  const pooledScale = Math.sqrt(manifest.embeddingLength);
  for (let index = 0; index < hidden.length; index += 1) {
    hidden[index] = Math.fround((hidden[index] ?? 0) * pooledScale);
  }

  if (session.hasTensor("v.std_bias") && session.hasTensor("v.std_scale")) {
    const bias = await session.readF32Tensor("v.std_bias");
    const scale = await session.readF32Tensor("v.std_scale");
    for (let token = 0; token < hidden.length / manifest.embeddingLength; token += 1) {
      const offset = token * manifest.embeddingLength;
      for (let index = 0; index < manifest.embeddingLength; index += 1) {
        hidden[offset + index] = Math.fround(
          Math.fround((hidden[offset + index] ?? 0) - (bias[index] ?? 0)) * (scale[index] ?? 1),
        );
      }
    }
  }

  hidden = await matMulVisionWeight(session, "mm.input_projection.weight", hidden);
  hidden = rmsNormRowsNoWeight(hidden, manifest.projectionDim, manifest.layerNormEpsilon);

  return {
    hidden,
    tokenCount: hidden.length / manifest.projectionDim,
    width: pixels.width,
    height: pixels.height,
  };
}

async function patchEmbed(
  session: Gemma4VisionSession,
  pixels: Gemma4VisionPixelValues,
  patchGridX: number,
  patchGridY: number,
): Promise<Float32Array> {
  const manifest = session.manifest;
  const weights = await session.readF32Tensor("v.patch_embd.weight");
  const patchSize = manifest.patchSize;
  const nPatches = patchGridX * patchGridY;
  const hidden = new Float32Array(nPatches * manifest.embeddingLength);

  for (let patchY = 0; patchY < patchGridY; patchY += 1) {
    for (let patchX = 0; patchX < patchGridX; patchX += 1) {
      const patch = patchY * patchGridX + patchX;
      const outputOffset = patch * manifest.embeddingLength;
      for (let emb = 0; emb < manifest.embeddingLength; emb += 1) {
        let sum = 0;
        for (let ky = 0; ky < patchSize; ky += 1) {
          const y = patchY * patchSize + ky;
          for (let kx = 0; kx < patchSize; kx += 1) {
            const x = patchX * patchSize + kx;
            const pixelOffset = (y * pixels.width + x) * 3;
            for (let channel = 0; channel < 3; channel += 1) {
              const weightOffset = kx + patchSize * (ky + patchSize * (channel + 3 * emb));
              const scaledPixel = Math.fround(Math.fround((pixels.values[pixelOffset + channel] ?? 0) * 2) - 1);
              sum = Math.fround(sum + Math.fround((weights[weightOffset] ?? 0) * scaledPixel));
            }
          }
        }
        hidden[outputOffset + emb] = sum;
      }
    }
  }
  return hidden;
}

async function addPositionEmbeddings(
  session: Gemma4VisionSession,
  hidden: Float32Array,
  patchGridX: number,
  nPatches: number,
): Promise<Float32Array> {
  const manifest = session.manifest;
  const positions = await session.readF32Tensor("v.position_embd.weight");
  const tensor = session.getTensor("v.position_embd.weight");
  const tableSize = tensor.dimensions[1] ?? 0;
  const output = new Float32Array(hidden);

  for (let patch = 0; patch < nPatches; patch += 1) {
    const x = patch % patchGridX;
    const y = Math.floor(patch / patchGridX);
    if (x >= tableSize || y >= tableSize) {
      throw new Error(`Vision position ${x},${y} exceeds table size ${tableSize}`);
    }
    const outputOffset = patch * manifest.embeddingLength;
    const xOffset = x * manifest.embeddingLength;
    const yOffset = (tableSize + y) * manifest.embeddingLength;
    for (let index = 0; index < manifest.embeddingLength; index += 1) {
      output[outputOffset + index] = Math.fround(
        Math.fround((output[outputOffset + index] ?? 0) + (positions[xOffset + index] ?? 0)) +
          (positions[yOffset + index] ?? 0),
      );
    }
  }
  return output;
}

async function forwardVisionLayer(
  session: Gemma4VisionSession,
  input: Float32Array,
  layer: number,
  patchGridX: number,
  tokenCount: number,
): Promise<Float32Array> {
  const manifest = session.manifest;
  const headSize = manifest.embeddingLength / manifest.headCount;
  const norm = rmsNormRows(input, await session.readF32Tensor(`v.blk.${layer}.ln1.weight`), session.epsilon);
  const [qProjection, kProjection, vProjection] = await Promise.all([
    matMulVisionWeight(session, `v.blk.${layer}.attn_q.weight`, norm),
    matMulVisionWeight(session, `v.blk.${layer}.attn_k.weight`, norm),
    matMulVisionWeight(session, `v.blk.${layer}.attn_v.weight`, norm),
  ]);
  const q = rope2dNeox(
    normHeads(qProjection, await session.readF32Tensor(`v.blk.${layer}.attn_q_norm.weight`), session.epsilon),
    patchGridX,
    headSize,
    manifest.headCount,
    100,
  );
  const k = rope2dNeox(
    normHeads(kProjection, await session.readF32Tensor(`v.blk.${layer}.attn_k_norm.weight`), session.epsilon),
    patchGridX,
    headSize,
    manifest.headCount,
    100,
  );
  const v = rmsNormRowsNoWeight(vProjection, headSize, session.epsilon);
  const attention = gqaAttention(q, k, v, {
    headSize,
    queryHeadCount: manifest.headCount,
    keyValueHeadCount: manifest.headCount,
    tokenCount,
    keyValueTokenCount: tokenCount,
    scale: 1,
    causal: false,
  });
  const attentionOutput = await matMulVisionWeight(session, `v.blk.${layer}.attn_out.weight`, attention);
  const attentionNorm = rmsNormRows(
    attentionOutput,
    await session.readF32Tensor(`v.blk.${layer}.attn_post_norm.weight`),
    session.epsilon,
  );
  const attentionResidual = residualAdd(input, attentionNorm);
  const ffnInput = rmsNormRows(
    attentionResidual,
    await session.readF32Tensor(`v.blk.${layer}.ln2.weight`),
    session.epsilon,
  );
  const [gate, up] = await Promise.all([
    matMulVisionWeight(session, `v.blk.${layer}.ffn_gate.weight`, ffnInput),
    matMulVisionWeight(session, `v.blk.${layer}.ffn_up.weight`, ffnInput),
  ]);
  for (let index = 0; index < gate.length; index += 1) {
    gate[index] = Math.fround(gelu(gate[index] ?? 0) * (up[index] ?? 0));
  }
  const ffnOutput = await matMulVisionWeight(session, `v.blk.${layer}.ffn_down.weight`, gate);
  const ffnNorm = rmsNormRows(
    ffnOutput,
    await session.readF32Tensor(`v.blk.${layer}.ffn_post_norm.weight`),
    session.epsilon,
  );
  return residualAdd(attentionResidual, ffnNorm);
}

async function matMulVisionWeight(
  session: Gemma4VisionSession,
  weightName: string,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const clampedInput = await clampLinearInput(session, weightName, inputColumns);
  const output = tensor.type === "F32" || tensor.type === "F16" || tensor.type === "BF16"
    ? await matMulDenseRows(session, weightName, clampedInput)
    : await matMulQuantizedRows(session, weightName, tensor.type, clampedInput);
  return clampLinearOutput(session, weightName, output);
}

async function matMulDenseRows(
  session: Gemma4VisionSession,
  weightName: string,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  const bytes = await session.readWeightBytes(weightName);
  const rowByteLength = tensorByteLength({ ...tensor, dimensions: [inputSize] });
  const output = new Float32Array(rowCount * columnCount);

  for (let row = 0; row < rowCount; row += 1) {
    const weight = dequantizeRow(tensor.type, bytes.subarray(row * rowByteLength, (row + 1) * rowByteLength), inputSize);
    for (let column = 0; column < columnCount; column += 1) {
      const inputOffset = column * inputSize;
      let sum = 0;
      for (let index = 0; index < inputSize; index += 1) {
        sum = Math.fround(sum + Math.fround((weight[index] ?? 0) * (inputColumns[inputOffset + index] ?? 0)));
      }
      output[column * rowCount + row] = sum;
    }
  }
  return output;
}

async function matMulQuantizedRows(
  session: Gemma4VisionSession,
  weightName: string,
  type: GgmlTypeName,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  if (type !== "Q8_0") {
    throw new Error(`${weightName} has unsupported vision matmul type ${type}`);
  }
  const tensor = session.getTensor(weightName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  const weightBytes = await session.readWeightBytes(weightName);
  const wasm = await matMulQuantizedWasm(type, weightBytes, inputColumns, inputSize, rowCount, columnCount);
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(rowCount * columnCount);
  const rowByteLength = tensorByteLength({ ...tensor, dimensions: [inputSize] });
  for (let row = 0; row < rowCount; row += 1) {
    const weight = dequantizeRow(type, weightBytes.subarray(row * rowByteLength, (row + 1) * rowByteLength), inputSize);
    for (let column = 0; column < columnCount; column += 1) {
      const inputOffset = column * inputSize;
      let sum = 0;
      for (let index = 0; index < inputSize; index += 1) {
        sum = Math.fround(sum + Math.fround((weight[index] ?? 0) * (inputColumns[inputOffset + index] ?? 0)));
      }
      output[column * rowCount + row] = sum;
    }
  }
  return output;
}

async function clampLinearInput(
  session: Gemma4VisionSession,
  weightName: string,
  input: Float32Array,
): Promise<Float32Array> {
  return clampTensor(session, input, weightName.replace(/\.weight$/, ".input_min"), weightName.replace(/\.weight$/, ".input_max"));
}

async function clampLinearOutput(
  session: Gemma4VisionSession,
  weightName: string,
  input: Float32Array,
): Promise<Float32Array> {
  return clampTensor(session, input, weightName.replace(/\.weight$/, ".output_min"), weightName.replace(/\.weight$/, ".output_max"));
}

async function clampTensor(
  session: Gemma4VisionSession,
  input: Float32Array,
  minTensorName: string,
  maxTensorName: string,
): Promise<Float32Array> {
  if (!session.hasTensor(minTensorName) || !session.hasTensor(maxTensorName)) {
    return input;
  }
  const min = (await session.readF32Tensor(minTensorName))[0] ?? -Infinity;
  const max = (await session.readF32Tensor(maxTensorName))[0] ?? Infinity;
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = Math.min(max, Math.max(min, input[index] ?? 0));
  }
  return output;
}

function averagePoolVisionTokens(
  input: Float32Array,
  patchGridX: number,
  patchGridY: number,
  embeddingLength: number,
  kernelSize: number,
): Float32Array {
  const outX = patchGridX / kernelSize;
  const outY = patchGridY / kernelSize;
  const output = new Float32Array(outX * outY * embeddingLength);
  const scale = 1 / (kernelSize * kernelSize);
  for (let oy = 0; oy < outY; oy += 1) {
    for (let ox = 0; ox < outX; ox += 1) {
      const outToken = oy * outX + ox;
      for (let ky = 0; ky < kernelSize; ky += 1) {
        for (let kx = 0; kx < kernelSize; kx += 1) {
          const inToken = (oy * kernelSize + ky) * patchGridX + ox * kernelSize + kx;
          for (let emb = 0; emb < embeddingLength; emb += 1) {
            output[outToken * embeddingLength + emb] = Math.fround(
              (output[outToken * embeddingLength + emb] ?? 0) +
                Math.fround((input[inToken * embeddingLength + emb] ?? 0) * scale),
            );
          }
        }
      }
    }
  }
  return output;
}

function rope2dNeox(
  input: Float32Array,
  patchGridX: number,
  headSize: number,
  headCount: number,
  freqBase: number,
): Float32Array {
  const tokenCount = input.length / (headSize * headCount);
  const output = new Float32Array(input);
  applyRopeSlice(output, input, patchGridX, headSize, headCount, tokenCount, 0, headSize / 2, freqBase, "x");
  applyRopeSlice(output, input, patchGridX, headSize, headCount, tokenCount, headSize / 2, headSize / 2, freqBase, "y");
  return output;
}

function applyRopeSlice(
  output: Float32Array,
  input: Float32Array,
  patchGridX: number,
  headSize: number,
  headCount: number,
  tokenCount: number,
  sliceOffset: number,
  sliceLength: number,
  freqBase: number,
  axis: "x" | "y",
): void {
  const thetaScale = Math.pow(freqBase, -2 / sliceLength);
  for (let token = 0; token < tokenCount; token += 1) {
    const position = axis === "x" ? token % patchGridX : Math.floor(token / patchGridX);
    for (let head = 0; head < headCount; head += 1) {
      const rowOffset = (token * headCount + head) * headSize + sliceOffset;
      let theta = position;
      for (let i0 = 0; i0 < sliceLength; i0 += 2) {
        const index = i0 / 2;
        const x0 = input[rowOffset + index] ?? 0;
        const x1 = input[rowOffset + sliceLength / 2 + index] ?? 0;
        const cosTheta = Math.cos(theta);
        const sinTheta = Math.sin(theta);
        output[rowOffset + index] = Math.fround(Math.fround(x0 * cosTheta) - Math.fround(x1 * sinTheta));
        output[rowOffset + sliceLength / 2 + index] = Math.fround(Math.fround(x0 * sinTheta) + Math.fround(x1 * cosTheta));
        theta = Math.fround(theta * thetaScale);
      }
    }
  }
}

function rmsNormRows(input: Float32Array, weight: Float32Array, epsilon: number): Float32Array {
  const output = new Float32Array(input.length);
  for (let row = 0; row < input.length / weight.length; row += 1) {
    const offset = row * weight.length;
    output.set(rmsNorm(input.slice(offset, offset + weight.length), weight, epsilon), offset);
  }
  return output;
}

function rmsNormRowsNoWeight(input: Float32Array, rowSize: number, epsilon: number): Float32Array {
  const output = new Float32Array(input.length);
  for (let row = 0; row < input.length / rowSize; row += 1) {
    const offset = row * rowSize;
    let sumSquares = 0;
    for (let index = 0; index < rowSize; index += 1) {
      const value = input[offset + index] ?? 0;
      sumSquares += value * value;
    }
    const scale = 1 / Math.sqrt(sumSquares / rowSize + epsilon);
    for (let index = 0; index < rowSize; index += 1) {
      output[offset + index] = Math.fround((input[offset + index] ?? 0) * scale);
    }
  }
  return output;
}

function normHeads(input: Float32Array, weight: Float32Array, epsilon: number): Float32Array {
  const output = new Float32Array(input.length);
  for (let row = 0; row < input.length / weight.length; row += 1) {
    const offset = row * weight.length;
    output.set(rmsNorm(input.slice(offset, offset + weight.length), weight, epsilon), offset);
  }
  return output;
}

function residualAdd(left: Float32Array, right: Float32Array): Float32Array {
  if (left.length !== right.length) {
    throw new Error(`Residual shape mismatch: left=${left.length} right=${right.length}`);
  }
  const output = new Float32Array(left.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Math.fround((left[index] ?? 0) + (right[index] ?? 0));
  }
  return output;
}

function gelu(value: number): number {
  if (value <= -10) {
    return 0;
  }
  if (value >= 10) {
    return value;
  }
  const inner = Math.fround(Math.fround(Math.sqrt(2 / Math.PI) * value) * Math.fround(1 + Math.fround(0.044715 * value * value)));
  return Math.fround(Math.fround(0.5 * value) * Math.fround(1 + Math.tanh(inner)));
}

function imageBitmapToRgba(bitmap: ImageBitmap): Uint8ClampedArray {
  const canvas = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(bitmap.width, bitmap.height)
    : document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Could not create image decode canvas.");
  }
  context.drawImage(bitmap, 0, 0);
  return context.getImageData(0, 0, bitmap.width, bitmap.height).data;
}

function resizeRgbaBilinear(
  src: Uint8ClampedArray,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): Uint8ClampedArray {
  const output = new Uint8ClampedArray(targetWidth * targetHeight * 4);
  const xRatio = targetWidth > 1 ? (sourceWidth - 1) / (targetWidth - 1) : 0;
  const yRatio = targetHeight > 1 ? (sourceHeight - 1) / (targetHeight - 1) : 0;
  for (let y = 0; y < targetHeight; y += 1) {
    const py = y * yRatio;
    const y0 = Math.min(Math.trunc(py), sourceHeight - 1);
    const y1 = Math.min(y0 + 1, sourceHeight - 1);
    const yf = py - y0;
    for (let x = 0; x < targetWidth; x += 1) {
      const px = x * xRatio;
      const x0 = Math.min(Math.trunc(px), sourceWidth - 1);
      const x1 = Math.min(x0 + 1, sourceWidth - 1);
      const xf = px - x0;
      const dst = (y * targetWidth + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        const top = lerp(
          src[(y0 * sourceWidth + x0) * 4 + channel] ?? 0,
          src[(y0 * sourceWidth + x1) * 4 + channel] ?? 0,
          xf,
        );
        const bottom = lerp(
          src[(y1 * sourceWidth + x0) * 4 + channel] ?? 0,
          src[(y1 * sourceWidth + x1) * 4 + channel] ?? 0,
          xf,
        );
        output[dst + channel] = Math.trunc(lerp(top, bottom, yf));
      }
      output[dst + 3] = 255;
    }
  }
  return output;
}

function roundByFactor(value: number, factor: number): number {
  return Math.round(value / factor) * factor;
}

function ceilByFactor(value: number, factor: number): number {
  return Math.ceil(value / factor) * factor;
}

function floorByFactor(value: number, factor: number): number {
  return Math.floor(value / factor) * factor;
}

function lerp(left: number, right: number, amount: number): number {
  return left + (right - left) * amount;
}
