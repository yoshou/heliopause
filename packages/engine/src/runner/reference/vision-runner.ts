import type {
  GgmlTypeName,
} from "../../gguf";
import {
  dequantizeRow,
} from "../../quant";
import {
  tensorByteLength,
} from "../../tensor-reader";
import type {
  VisionEncodeResult,
  VisionPixelValues,
  VisionSession,
} from "../../vision";
import type {
  VisionEncoderRunner,
  VisionPreprocessRunner,
} from "../vision-runner";
import {
  gqaAttention,
  rmsNorm,
} from "./kernels";

export const referenceVisionPreprocessRunner: VisionPreprocessRunner = {
  provider: "reference",
  run: async (session, input, visionPreprocess) =>
    visionPreprocess(input.rgba, input.sourceWidth, input.sourceHeight, session.manifest),
};

export const referenceVisionEncoderRunner: VisionEncoderRunner = {
  provider: "reference",
  run: (session, pixels) => runReferenceVisionEncoder(session, pixels),
};

export async function runReferenceVisionEncoder(
  session: VisionSession,
  pixels: VisionPixelValues,
): Promise<VisionEncodeResult> {
  requireReferenceProvider(session);
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
    Math.sqrt(manifest.embeddingLength),
  );

  if (session.hasTensor("v.std_bias") && session.hasTensor("v.std_scale")) {
    hidden = stdNormalizeVisionTokens(
      hidden,
      await session.readF32Tensor("v.std_bias"),
      await session.readF32Tensor("v.std_scale"),
      manifest.embeddingLength,
    );
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

function requireReferenceProvider(session: VisionSession): void {
  if (!session.hasProvider("reference")) {
    throw new Error("Vision reference execution requires an enabled reference provider.");
  }
}

async function patchEmbed(
  session: VisionSession,
  pixels: VisionPixelValues,
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
  session: VisionSession,
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
  session: VisionSession,
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
    rmsNormRows(qProjection, await session.readF32Tensor(`v.blk.${layer}.attn_q_norm.weight`), session.epsilon),
    patchGridX,
    headSize,
    manifest.headCount,
    100,
  );
  const k = rope2dNeox(
    rmsNormRows(kProjection, await session.readF32Tensor(`v.blk.${layer}.attn_k_norm.weight`), session.epsilon),
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
  const activated = geluMul(gate, up);
  const ffnOutput = await matMulVisionWeight(session, `v.blk.${layer}.ffn_down.weight`, activated);
  const ffnNorm = rmsNormRows(
    ffnOutput,
    await session.readF32Tensor(`v.blk.${layer}.ffn_post_norm.weight`),
    session.epsilon,
  );
  return residualAdd(attentionResidual, ffnNorm);
}

async function matMulVisionWeight(
  session: VisionSession,
  weightName: string,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const clampedInput = await clampLinearInput(session, weightName, inputColumns);
  const output = await matMulRows(session, weightName, tensor.type, clampedInput);
  return clampLinearOutput(session, weightName, output);
}

async function matMulRows(
  session: VisionSession,
  weightName: string,
  type: GgmlTypeName,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  if (!Number.isInteger(columnCount) || inputSize <= 0) {
    throw new Error(`${weightName} input shape mismatch: ${inputColumns.length}`);
  }
  if (type !== "F32" && type !== "F16" && type !== "BF16" && type !== "Q8_0") {
    throw new Error(`${weightName} has unsupported vision matmul type ${type}`);
  }
  const bytes = await session.readWeightBytes(weightName);
  const rowByteLength = tensorByteLength({ ...tensor, dimensions: [inputSize] });
  const output = new Float32Array(rowCount * columnCount);

  for (let row = 0; row < rowCount; row += 1) {
    const weight = dequantizeRow(type, bytes.subarray(row * rowByteLength, (row + 1) * rowByteLength), inputSize);
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
  session: VisionSession,
  weightName: string,
  input: Float32Array,
): Promise<Float32Array> {
  return clampTensor(session, input, weightName.replace(/\.weight$/, ".input_min"), weightName.replace(/\.weight$/, ".input_max"));
}

async function clampLinearOutput(
  session: VisionSession,
  weightName: string,
  input: Float32Array,
): Promise<Float32Array> {
  return clampTensor(session, input, weightName.replace(/\.weight$/, ".output_min"), weightName.replace(/\.weight$/, ".output_max"));
}

async function clampTensor(
  session: VisionSession,
  input: Float32Array,
  minTensorName: string,
  maxTensorName: string,
): Promise<Float32Array> {
  const clamp = await readClamp(session, minTensorName, maxTensorName);
  if (!clamp) {
    return input;
  }
  return clampTensorValues(input, clamp.min, clamp.max);
}

async function readClamp(
  session: VisionSession,
  minTensorName: string,
  maxTensorName: string,
): Promise<{ min: number; max: number } | undefined> {
  if (!session.hasTensor(minTensorName) || !session.hasTensor(maxTensorName)) {
    return undefined;
  }
  return {
    min: (await session.readF32Tensor(minTensorName))[0] ?? -Infinity,
    max: (await session.readF32Tensor(maxTensorName))[0] ?? Infinity,
  };
}

function clampTensorValues(input: Float32Array, min: number, max: number): Float32Array {
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
  outputScale: number,
): Float32Array {
  const outX = patchGridX / kernelSize;
  const outY = patchGridY / kernelSize;
  const output = new Float32Array(outX * outY * embeddingLength);
  const scale = outputScale / (kernelSize * kernelSize);
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
  if (input.length % weight.length !== 0) {
    throw new Error(`Vision RMSNorm row shape mismatch: input=${input.length} weight=${weight.length}`);
  }
  const output = new Float32Array(input.length);
  for (let row = 0; row < input.length / weight.length; row += 1) {
    const offset = row * weight.length;
    output.set(rmsNorm(input.slice(offset, offset + weight.length), weight, epsilon), offset);
  }
  return output;
}

function rmsNormRowsNoWeight(input: Float32Array, rowSize: number, epsilon: number): Float32Array {
  if (input.length % rowSize !== 0) {
    throw new Error(`Vision RMSNorm row shape mismatch: input=${input.length} row=${rowSize}`);
  }
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

function geluMul(gate: Float32Array, up: Float32Array): Float32Array {
  if (gate.length !== up.length) {
    throw new Error(`GELU multiply shape mismatch: gate=${gate.length} up=${up.length}`);
  }
  const output = new Float32Array(gate);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Math.fround(gelu(output[index] ?? 0) * (up[index] ?? 0));
  }
  return output;
}

function stdNormalizeVisionTokens(
  input: Float32Array,
  bias: Float32Array,
  scale: Float32Array,
  rowSize: number,
): Float32Array {
  const output = new Float32Array(input);
  for (let token = 0; token < output.length / rowSize; token += 1) {
    const offset = token * rowSize;
    for (let index = 0; index < rowSize; index += 1) {
      output[offset + index] = Math.fround(
        Math.fround((output[offset + index] ?? 0) - (bias[index] ?? 0)) * (scale[index] ?? 1),
      );
    }
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
