import type {
  GgmlTypeName,
} from "../../gguf";
import {
  gqaAttention,
  rmsNorm,
} from "../../ops";
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
import {
  createWasmQuantizedWeightHandle,
  gqaAttentionWasm,
  matMulQuantizedMultiWasm,
  matMulQuantizedWasm,
  matMulQuantizedWasmResident,
  matMulQuantizedWasmResidentMulti,
  releaseWasmQuantizedWeightHandle,
  type QuantizedMatMulInput,
  type WasmQuantizedWeightHandle,
  visionAddPositionEmbeddingsWasm,
  visionAveragePoolScaleWasm,
  visionClampWasm,
  visionGeluMulWasm,
  visionPatchEmbedWasm,
  visionResidualAddWasm,
  visionRmsNormWasm,
  visionRope2dNeoxWasm,
  visionStdNormalizeWasm,
} from "./wasm-kernels";

type VisionWasmWeightCache = {
  handles: Map<string, WasmQuantizedWeightHandle>;
  bytes: number;
  hits: number;
  misses: number;
};

const residentWeightCaches = new WeakMap<VisionSession, VisionWasmWeightCache>();

export async function runCpuVisionEncoder(
  session: VisionSession,
  pixels: VisionPixelValues,
): Promise<VisionEncodeResult> {
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

  hidden = await averagePoolVisionTokens(
    hidden,
    patchGridX,
    patchGridY,
    manifest.embeddingLength,
    manifest.spatialMergeSize,
    Math.sqrt(manifest.embeddingLength),
    session,
  );

  if (session.hasTensor("v.std_bias") && session.hasTensor("v.std_scale")) {
    const bias = await session.readF32Tensor("v.std_bias");
    const scale = await session.readF32Tensor("v.std_scale");
    hidden = await stdNormalizeVisionTokens(hidden, bias, scale, manifest.embeddingLength, session);
  }

  hidden = await matMulVisionWeight(session, "mm.input_projection.weight", hidden);
  hidden = await rmsNormRowsNoWeight(hidden, manifest.projectionDim, manifest.layerNormEpsilon);

  return {
    hidden,
    tokenCount: hidden.length / manifest.projectionDim,
    width: pixels.width,
    height: pixels.height,
  };
}

export function releaseCpuVisionEncoder(session: VisionSession): void {
  const cache = residentWeightCaches.get(session);
  if (!cache) {
    return;
  }
  for (const handle of cache.handles.values()) {
    releaseWasmQuantizedWeightHandle(handle);
  }
  residentWeightCaches.delete(session);
}

function visionResidentWeightCacheEnabled(session: VisionSession): boolean {
  const provider = session.executionProvider("cpu");
  return provider?.options?.residentWeightCache === true;
}

function visionProjectionBatchingEnabled(session: VisionSession): boolean {
  const provider = session.executionProvider("cpu");
  return provider?.options?.projectionBatching !== false;
}

function visionWasmKernelsEnabled(session: VisionSession): boolean {
  const provider = session.executionProvider("cpu");
  return provider?.options?.wasmKernels !== false;
}

function residentWeightCache(session: VisionSession): VisionWasmWeightCache {
  let cache = residentWeightCaches.get(session);
  if (!cache) {
    cache = {
      handles: new Map(),
      bytes: 0,
      hits: 0,
      misses: 0,
    };
    residentWeightCaches.set(session, cache);
    session.addDisposeCallback(() => releaseCpuVisionEncoder(session));
    const statsCache = cache;
    session.setExecutionProviderStatsProvider(() => ({
      cpuVisionResidentWeightCacheEnabled: visionResidentWeightCacheEnabled(session),
      cpuVisionResidentWeightCacheCount: statsCache.handles.size,
      cpuVisionResidentWeightCacheBytes: statsCache.bytes,
      cpuVisionResidentWeightCacheHits: statsCache.hits,
      cpuVisionResidentWeightCacheMisses: statsCache.misses,
    }), "cpu-vision");
  }
  return cache;
}

async function readWasmWeightHandle(
  session: VisionSession,
  weightName: string,
  type: "Q8_0",
  inputSize: number,
  rowCount: number,
): Promise<WasmQuantizedWeightHandle | undefined> {
  const cache = residentWeightCache(session);
  const cached = cache.handles.get(weightName);
  if (cached) {
    cache.hits += 1;
    return cached;
  }
  cache.misses += 1;
  const bytes = await session.readWeightBytes(weightName);
  const handle = await createWasmQuantizedWeightHandle(type, bytes, inputSize, rowCount);
  if (!handle) {
    return undefined;
  }
  cache.handles.set(weightName, handle);
  cache.bytes += handle.byteLength + handle.scaleByteLength;
  return handle;
}

async function patchEmbed(
  session: VisionSession,
  pixels: VisionPixelValues,
  patchGridX: number,
  patchGridY: number,
): Promise<Float32Array> {
  const manifest = session.manifest;
  const weights = await session.readF32Tensor("v.patch_embd.weight");
  const wasm = visionWasmKernelsEnabled(session)
    ? await visionPatchEmbedWasm(pixels.values, weights, pixels.width, manifest.patchSize, patchGridX, patchGridY, manifest.embeddingLength)
    : undefined;
  if (wasm) {
    return wasm;
  }

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
  const wasm = visionWasmKernelsEnabled(session)
    ? await visionAddPositionEmbeddingsWasm(hidden, positions, patchGridX, nPatches, manifest.embeddingLength, tableSize)
    : undefined;
  if (wasm) {
    return wasm;
  }
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
  const norm = await rmsNormRows(input, await session.readF32Tensor(`v.blk.${layer}.ln1.weight`), session.epsilon, session);
  const qkv = await matMulVisionWeightBatch(session, [
    `v.blk.${layer}.attn_q.weight`,
    `v.blk.${layer}.attn_k.weight`,
    `v.blk.${layer}.attn_v.weight`,
  ], norm);
  const [qProjection, kProjection, vProjection] = qkv ?? await Promise.all([
    matMulVisionWeight(session, `v.blk.${layer}.attn_q.weight`, norm),
    matMulVisionWeight(session, `v.blk.${layer}.attn_k.weight`, norm),
    matMulVisionWeight(session, `v.blk.${layer}.attn_v.weight`, norm),
  ]);
  const q = await rope2dNeox(
    await normHeads(qProjection!, await session.readF32Tensor(`v.blk.${layer}.attn_q_norm.weight`), session.epsilon, session),
    patchGridX,
    headSize,
    manifest.headCount,
    100,
    session,
  );
  const k = await rope2dNeox(
    await normHeads(kProjection!, await session.readF32Tensor(`v.blk.${layer}.attn_k_norm.weight`), session.epsilon, session),
    patchGridX,
    headSize,
    manifest.headCount,
    100,
    session,
  );
  const v = await rmsNormRowsNoWeight(vProjection!, headSize, session.epsilon, session);
  const attention = (visionWasmKernelsEnabled(session)
    ? await gqaAttentionWasm(q, k, v, {
      headSize,
      queryHeadCount: manifest.headCount,
      keyValueHeadCount: manifest.headCount,
      tokenCount,
      keyValueTokenCount: tokenCount,
      scale: 1,
      causal: false,
    })
    : undefined) ?? gqaAttention(q, k, v, {
    headSize,
    queryHeadCount: manifest.headCount,
    keyValueHeadCount: manifest.headCount,
    tokenCount,
    keyValueTokenCount: tokenCount,
    scale: 1,
    causal: false,
  });
  const attentionOutput = await matMulVisionWeight(session, `v.blk.${layer}.attn_out.weight`, attention);
  const attentionNorm = await rmsNormRows(
    attentionOutput,
    await session.readF32Tensor(`v.blk.${layer}.attn_post_norm.weight`),
    session.epsilon,
    session,
  );
  const attentionResidual = await residualAdd(input, attentionNorm, session);
  const ffnInput = await rmsNormRows(
    attentionResidual,
    await session.readF32Tensor(`v.blk.${layer}.ln2.weight`),
    session.epsilon,
    session,
  );
  const gateUp = await matMulVisionWeightBatch(session, [
    `v.blk.${layer}.ffn_gate.weight`,
    `v.blk.${layer}.ffn_up.weight`,
  ], ffnInput);
  const [gate, up] = gateUp ?? await Promise.all([
    matMulVisionWeight(session, `v.blk.${layer}.ffn_gate.weight`, ffnInput),
    matMulVisionWeight(session, `v.blk.${layer}.ffn_up.weight`, ffnInput),
  ]);
  const activated = await geluMul(gate!, up!, session);
  const ffnOutput = await matMulVisionWeight(session, `v.blk.${layer}.ffn_down.weight`, activated);
  const ffnNorm = await rmsNormRows(
    ffnOutput,
    await session.readF32Tensor(`v.blk.${layer}.ffn_post_norm.weight`),
    session.epsilon,
    session,
  );
  return residualAdd(attentionResidual, ffnNorm, session);
}

async function matMulVisionWeight(
  session: VisionSession,
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

async function matMulVisionWeightBatch(
  session: VisionSession,
  weightNames: readonly string[],
  inputColumns: Float32Array,
): Promise<Float32Array[] | undefined> {
  if (!visionProjectionBatchingEnabled(session) || weightNames.length < 2 || weightNames.length > 4) {
    return undefined;
  }
  const tensors = weightNames.map((name) => session.getTensor(name));
  const inputSize = tensors[0]?.dimensions[0] ?? 0;
  if (inputSize <= 0 || inputColumns.length % inputSize !== 0) {
    return undefined;
  }
  const columnCount = inputColumns.length / inputSize;
  if (tensors.some((tensor) => tensor.dimensions[0] !== inputSize || tensor.type !== "Q8_0")) {
    return undefined;
  }
  const clamp = await sharedInputClamp(session, weightNames);
  if (!clamp) {
    return undefined;
  }
  const clampedInput = await clampTensorValues(inputColumns, clamp.min, clamp.max, session);

  let outputs: Float32Array[] | undefined;
  if (visionResidentWeightCacheEnabled(session)) {
    const handles = await Promise.all(weightNames.map((name, index) => {
      const tensor = tensors[index];
      return readWasmWeightHandle(session, name, "Q8_0", inputSize, tensor?.dimensions[1] ?? 0);
    }));
    if (handles.every((handle): handle is WasmQuantizedWeightHandle => Boolean(handle))) {
      outputs = await matMulQuantizedWasmResidentMulti(handles, clampedInput, inputSize, columnCount);
      if (!outputs) {
        outputs = await Promise.all(handles.map((handle) =>
          matMulQuantizedWasmResident(handle, clampedInput, inputSize, handle.rowCount, columnCount),
        )) as Float32Array[];
      }
    }
  }

  if (!outputs && visionWasmKernelsEnabled(session)) {
    const weights: QuantizedMatMulInput[] = [];
    for (let index = 0; index < weightNames.length; index += 1) {
      const name = weightNames[index];
      const tensor = tensors[index];
      if (!name || !tensor || tensor.type !== "Q8_0") {
        return undefined;
      }
      weights.push({
        type: "Q8_0",
        weightBytes: await session.readWeightBytes(name),
        rowCount: tensor.dimensions[1] ?? 0,
      });
    }
    outputs = await matMulQuantizedMultiWasm(weights, clampedInput, inputSize, columnCount);
  }
  if (!outputs) {
    return undefined;
  }
  return Promise.all(outputs.map((output, index) =>
    clampLinearOutput(session, weightNames[index] ?? "", output),
  ));
}

async function sharedInputClamp(
  session: VisionSession,
  weightNames: readonly string[],
): Promise<{ min: number; max: number } | undefined> {
  let shared: { min: number; max: number } | undefined;
  for (const weightName of weightNames) {
    const minName = weightName.replace(/\.weight$/, ".input_min");
    const maxName = weightName.replace(/\.weight$/, ".input_max");
    const clamp = await readClamp(session, minName, maxName);
    if (!clamp) {
      return undefined;
    }
    if (!shared) {
      shared = clamp;
    } else if (shared.min !== clamp.min || shared.max !== clamp.max) {
      return undefined;
    }
  }
  return shared;
}

async function matMulDenseRows(
  session: VisionSession,
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
  session: VisionSession,
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
  const handle = visionResidentWeightCacheEnabled(session)
    ? await readWasmWeightHandle(session, weightName, type, inputSize, rowCount)
    : undefined;
  const resident = handle
    ? await matMulQuantizedWasmResident(handle, inputColumns, inputSize, rowCount, columnCount)
    : undefined;
  if (resident) {
    return resident;
  }
  const weightBytes = await session.readWeightBytes(weightName);
  const wasm = visionWasmKernelsEnabled(session)
    ? await matMulQuantizedWasm(type, weightBytes, inputColumns, inputSize, rowCount, columnCount)
    : undefined;
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
  return clampTensorValues(input, clamp.min, clamp.max, session);
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

async function clampTensorValues(
  input: Float32Array,
  min: number,
  max: number,
  session: VisionSession,
): Promise<Float32Array> {
  const wasm = visionWasmKernelsEnabled(session) ? await visionClampWasm(input, min, max) : undefined;
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = Math.min(max, Math.max(min, input[index] ?? 0));
  }
  return output;
}

async function averagePoolVisionTokens(
  input: Float32Array,
  patchGridX: number,
  patchGridY: number,
  embeddingLength: number,
  kernelSize: number,
  outputScale: number,
  session: VisionSession,
): Promise<Float32Array> {
  const wasm = visionWasmKernelsEnabled(session)
    ? await visionAveragePoolScaleWasm(input, patchGridX, patchGridY, embeddingLength, kernelSize, outputScale)
    : undefined;
  if (wasm) {
    return wasm;
  }
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

async function rope2dNeox(
  input: Float32Array,
  patchGridX: number,
  headSize: number,
  headCount: number,
  freqBase: number,
  session: VisionSession,
): Promise<Float32Array> {
  const tokenCount = input.length / (headSize * headCount);
  const wasm = visionWasmKernelsEnabled(session)
    ? await visionRope2dNeoxWasm(input, patchGridX, headSize, headCount, tokenCount, freqBase)
    : undefined;
  if (wasm) {
    return wasm;
  }
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

async function rmsNormRows(
  input: Float32Array,
  weight: Float32Array,
  epsilon: number,
  session: VisionSession,
): Promise<Float32Array> {
  const wasm = visionWasmKernelsEnabled(session)
    ? await visionRmsNormWasm(input, weight.length, epsilon, weight)
    : undefined;
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(input.length);
  for (let row = 0; row < input.length / weight.length; row += 1) {
    const offset = row * weight.length;
    output.set(rmsNorm(input.slice(offset, offset + weight.length), weight, epsilon), offset);
  }
  return output;
}

async function rmsNormRowsNoWeight(
  input: Float32Array,
  rowSize: number,
  epsilon: number,
  session?: VisionSession,
): Promise<Float32Array> {
  const wasm = session && visionWasmKernelsEnabled(session)
    ? await visionRmsNormWasm(input, rowSize, epsilon)
    : undefined;
  if (wasm) {
    return wasm;
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

function normHeads(
  input: Float32Array,
  weight: Float32Array,
  epsilon: number,
  session: VisionSession,
): Promise<Float32Array> {
  return rmsNormRows(input, weight, epsilon, session);
}

async function residualAdd(
  left: Float32Array,
  right: Float32Array,
  session: VisionSession,
): Promise<Float32Array> {
  if (left.length !== right.length) {
    throw new Error(`Residual shape mismatch: left=${left.length} right=${right.length}`);
  }
  const wasm = visionWasmKernelsEnabled(session) ? await visionResidualAddWasm(left, right) : undefined;
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(left.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Math.fround((left[index] ?? 0) + (right[index] ?? 0));
  }
  return output;
}

async function geluMul(
  gate: Float32Array,
  up: Float32Array,
  session: VisionSession,
): Promise<Float32Array> {
  if (gate.length !== up.length) {
    throw new Error(`GELU multiply shape mismatch: gate=${gate.length} up=${up.length}`);
  }
  const wasm = visionWasmKernelsEnabled(session) ? await visionGeluMulWasm(gate, up) : undefined;
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(gate);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Math.fround(gelu(output[index] ?? 0) * (up[index] ?? 0));
  }
  return output;
}

async function stdNormalizeVisionTokens(
  input: Float32Array,
  bias: Float32Array,
  scale: Float32Array,
  rowSize: number,
  session: VisionSession,
): Promise<Float32Array> {
  const wasm = visionWasmKernelsEnabled(session) ? await visionStdNormalizeWasm(input, bias, scale, rowSize) : undefined;
  if (wasm) {
    return wasm;
  }
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
