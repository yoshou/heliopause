import type {
  GgmlTypeName,
} from "../../gguf";
import type {
  MtpAssistantRunResult,
  MtpAssistantSession,
} from "../../mtp-assistant";
import {
  dequantizeRow,
} from "../../quant";
import {
  topK,
} from "../../runtime";
import {
  tensorByteLength,
} from "../../tensor-reader";
import type {
  MtpAssistantRunInput,
  MtpAssistantRunner,
  MtpAssistantRunners,
  MtpTargetKvLayerView,
} from "../mtp-assistant-runner";
import {
  gqaAttentionWasm,
  matMulDequantizedWasm,
} from "./wasm-kernels";

export function createWasmMtpAssistantRunners(): MtpAssistantRunners {
  return { runner: wasmMtpAssistantRunner };
}

const wasmMtpAssistantRunner: MtpAssistantRunner = {
  provider: "wasm",
  async run(session, input, options) {
    incrementWasmAssistantRuns(session);
    if (options.signal?.aborted) {
      throw new DOMException("MTP assistant execution was aborted.", "AbortError");
    }
    return runWasmMtpAssistant(session, input, options);
  },
};

async function runWasmMtpAssistant(
  session: MtpAssistantSession,
  input: MtpAssistantRunInput,
  options: { signal?: AbortSignal },
): Promise<MtpAssistantRunResult> {
  validateInput(session, input);
  const projectionInput = new Float32Array(session.manifest.backboneEmbeddingLength * 2);
  projectionInput.set(input.targetInputEmbedding, 0);
  projectionInput.set(input.targetCurrentHidden, session.manifest.backboneEmbeddingLength);
  let layerHidden = await matMulAssistantWeight(session, "mtp.pre_projection.weight", projectionInput);
  for (let layer = 0; layer < session.manifest.blockCount; layer += 1) {
    if (options.signal?.aborted) {
      throw new DOMException("MTP assistant execution was aborted.", "AbortError");
    }
    layerHidden = await forwardAssistantLayer(session, input, layer, layerHidden);
  }
  const normalizedHidden = rmsNorm(layerHidden, await session.readF32Tensor("output_norm.weight"), session.epsilon);
  const postProjection = await matMulAssistantWeight(session, "mtp.post_projection.weight", normalizedHidden);
  const centroidLogits = await matMulAssistantWeight(session, "mtp.centroids.weight", normalizedHidden);
  return {
    backboneHidden: postProjection,
    topTokens: await maskedEmbeddingTopTokens(session, normalizedHidden, centroidLogits, input.topK),
  };
}

async function forwardAssistantLayer(
  session: MtpAssistantSession,
  input: MtpAssistantRunInput,
  layer: number,
  hidden: Float32Array,
): Promise<Float32Array> {
  const manifest = session.manifest;
  const kind = manifest.layerKinds[layer] ?? "sliding-attention";
  const headSize = manifest.layerKeyLengths[layer] ?? manifest.keyLength;
  const valueSize = manifest.layerValueLengths[layer] ?? manifest.valueLength;
  const targetKv = input.targetKv.layers[layer];
  if (!targetKv) {
    throw new Error(`Missing target KV view for assistant layer ${layer}`);
  }
  validateTargetKvLayer(targetKv, manifest.headCountKv, headSize, valueSize, input.position, layer);
  const attnNorm = rmsNorm(hidden, await session.readF32Tensor(`blk.${layer}.attn_norm.weight`), session.epsilon);
  const qNorm = normHeads(
    await matMulAssistantWeight(session, `blk.${layer}.attn_q.weight`, attnNorm),
    await session.readF32Tensor(`blk.${layer}.attn_q_norm.weight`),
    session.epsilon,
  );
  const qRope = ropeNeox(qNorm, {
    headSize,
    headCount: manifest.headCount,
    position: input.position,
    nDims: kind === "sliding-attention" ? manifest.rope.slidingDimensionCount : manifest.rope.fullDimensionCount,
    freqBase: kind === "sliding-attention" ? manifest.rope.slidingFreqBase : manifest.rope.fullFreqBase,
    freqFactors: await readRopeFreqFactors(session, kind),
    activePairCount: kind === "full-attention" ? Math.floor(manifest.rope.fullDimensionCount * 0.25 / 2) : undefined,
  });
  const keyValueTokenCount = targetKvKeyValueTokenCount(targetKv, input.position);
  const attention = await gqaAttentionWasm(
    qRope,
    compactKey(targetKv, keyValueTokenCount),
    compactValue(targetKv, keyValueTokenCount),
    {
      headSize,
      queryHeadCount: manifest.headCount,
      keyValueHeadCount: manifest.headCountKv,
      tokenCount: 1,
      keyValueTokenCount,
      scale: 1,
      causal: true,
      mask: attentionMask(
        input.position,
        targetKv.logicalStart,
        keyValueTokenCount,
        kind === "sliding-attention" ? manifest.slidingWindow : undefined,
      ),
      valueLayout: "dim-head-token",
    },
  );
  if (!attention) {
    throw new Error("WASM GQA attention is unavailable.");
  }
  const attentionOutput = await matMulAssistantWeight(session, `blk.${layer}.attn_output.weight`, attention);
  const attentionResidual = residualAdd(hidden, rmsNorm(
    attentionOutput,
    await session.readF32Tensor(`blk.${layer}.post_attention_norm.weight`),
    session.epsilon,
  ));
  const ffn = await forwardAssistantFfn(session, layer, attentionResidual);
  const scale = (await session.readF32Tensor(`blk.${layer}.layer_output_scale.weight`))[0] ?? 1;
  const output = new Float32Array(ffn.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Math.fround((ffn[index] ?? 0) * scale);
  }
  return output;
}

async function forwardAssistantFfn(
  session: MtpAssistantSession,
  layer: number,
  residual: Float32Array,
): Promise<Float32Array> {
  const ffnNorm = rmsNorm(residual, await session.readF32Tensor(`blk.${layer}.ffn_norm.weight`), session.epsilon);
  const [gate, up] = await Promise.all([
    matMulAssistantWeight(session, `blk.${layer}.ffn_gate.weight`, ffnNorm),
    matMulAssistantWeight(session, `blk.${layer}.ffn_up.weight`, ffnNorm),
  ]);
  const gated = new Float32Array(gate.length);
  for (let index = 0; index < gated.length; index += 1) {
    gated[index] = Math.fround(gelu(gate[index] ?? 0) * (up[index] ?? 0));
  }
  const ffnOut = await matMulAssistantWeight(session, `blk.${layer}.ffn_down.weight`, gated);
  return residualAdd(residual, rmsNorm(ffnOut, await session.readF32Tensor(`blk.${layer}.post_ffw_norm.weight`), session.epsilon));
}

async function matMulAssistantWeight(
  session: MtpAssistantSession,
  weightName: string,
  input: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const outputSize = tensor.dimensions[1] ?? 0;
  if (inputSize <= 0 || outputSize <= 0) {
    throw new Error(`${weightName} must be a matrix tensor`);
  }
  if (input.length % inputSize !== 0) {
    throw new Error(`${weightName} input size mismatch: ${input.length} is not divisible by ${inputSize}`);
  }
  const columnCount = input.length / inputSize;
  const weightBytes = await session.readWeightBytes(weightName);
  if (isWasmQuantizedType(tensor.type)) {
    const output = await matMulDequantizedWasm(tensor.type, weightBytes, input, inputSize, outputSize, columnCount);
    if (!output) {
      throw new Error(`WASM dequantized matmul is unavailable for ${weightName}`);
    }
    return output;
  }
  return matMulDequantizedRows(tensor.type, weightBytes, input, inputSize, outputSize, columnCount);
}

function matMulDequantizedRows(
  type: GgmlTypeName,
  weightBytes: Uint8Array,
  input: Float32Array,
  inputSize: number,
  outputSize: number,
  columnCount: number,
): Float32Array {
  const output = new Float32Array(outputSize * columnCount);
  const rowByteLength = tensorByteLength({ name: "", dimensions: [inputSize], type, typeId: 0, offset: 0n, dataOffset: 0n });
  for (let row = 0; row < outputSize; row += 1) {
    const weightRow = dequantizeRow(type, weightBytes.subarray(row * rowByteLength, (row + 1) * rowByteLength), inputSize);
    for (let column = 0; column < columnCount; column += 1) {
      let sum = 0;
      const inputOffset = column * inputSize;
      for (let index = 0; index < inputSize; index += 1) {
        sum = Math.fround(sum + Math.fround((weightRow[index] ?? 0) * (input[inputOffset + index] ?? 0)));
      }
      output[column * outputSize + row] = sum;
    }
  }
  return output;
}

function isWasmQuantizedType(type: GgmlTypeName): type is "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0" {
  return type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "IQ4_XS" || type === "Q8_0";
}

async function maskedEmbeddingTopTokens(
  session: MtpAssistantSession,
  hidden: Float32Array,
  centroidLogits: Float32Array,
  topTokenCount: number,
) {
  const topCentroids = topK(centroidLogits, Math.min(session.manifest.centroidTopK, session.manifest.nCentroids));
  const tokenOrdering = await session.readI32Tensor("mtp.token_ordering.weight");
  const tokensPerCentroid = Math.floor(tokenOrdering.length / session.manifest.nCentroids);
  const selectedIds: number[] = [];
  const seen = new Set<number>();
  for (const centroid of topCentroids) {
    const end = Math.min((centroid.id + 1) * tokensPerCentroid, tokenOrdering.length);
    for (let index = centroid.id * tokensPerCentroid; index < end; index += 1) {
      const tokenId = tokenOrdering[index] ?? -1;
      if (tokenId >= 0 && tokenId < tokenOrdering.length && !seen.has(tokenId)) {
        seen.add(tokenId);
        selectedIds.push(tokenId);
      }
    }
  }
  const rows = await session.readEmbeddingRows(selectedIds);
  const candidates: Array<{ id: number; value: number }> = [];
  for (let row = 0; row < selectedIds.length; row += 1) {
    let sum = 0;
    const offset = row * session.manifest.embeddingLength;
    for (let index = 0; index < session.manifest.embeddingLength; index += 1) {
      sum = Math.fround(sum + Math.fround((rows[offset + index] ?? 0) * (hidden[index] ?? 0)));
    }
    candidates.push({ id: selectedIds[row] ?? 0, value: sum });
  }
  return candidates.sort((a, b) => b.value - a.value || a.id - b.id).slice(0, topTokenCount);
}

function validateInput(session: MtpAssistantSession, input: MtpAssistantRunInput): void {
  if (input.targetInputEmbedding.length !== session.manifest.backboneEmbeddingLength) {
    throw new Error(`targetInputEmbedding shape mismatch: ${input.targetInputEmbedding.length}`);
  }
  if (input.targetCurrentHidden.length !== session.manifest.backboneEmbeddingLength) {
    throw new Error(`targetCurrentHidden shape mismatch: ${input.targetCurrentHidden.length}`);
  }
  if (input.position < 0 || input.position >= session.manifest.contextLength) {
    throw new Error(`Position ${input.position} is outside context length ${session.manifest.contextLength}`);
  }
}

function validateTargetKvLayer(
  layer: MtpTargetKvLayerView,
  expectedHeadCountKv: number,
  expectedKeySize: number,
  expectedValueSize: number,
  position: number,
  layerIndex: number,
): void {
  if (layer.headCountKv !== expectedHeadCountKv) {
    throw new Error(`Target KV head count mismatch for assistant layer ${layerIndex}: ${layer.headCountKv}`);
  }
  if (layer.keyLength !== expectedKeySize || layer.valueLength !== expectedValueSize) {
    throw new Error(`Target KV head size mismatch for assistant layer ${layerIndex}: ${layer.keyLength}/${layer.valueLength}`);
  }
  if (layer.tokenCount <= position || position < layer.logicalStart) {
    throw new Error(`Target KV for assistant layer ${layerIndex} does not cover position ${position}`);
  }
}

function rmsNorm(input: Float32Array, weight: Float32Array, epsilon: number): Float32Array {
  if (input.length !== weight.length) {
    throw new Error(`RMS norm shape mismatch: input=${input.length}, weight=${weight.length}`);
  }
  let sumSquares = 0;
  for (let index = 0; index < input.length; index += 1) {
    const value = input[index] ?? 0;
    sumSquares = Math.fround(sumSquares + Math.fround(value * value));
  }
  const scale = 1 / Math.sqrt(sumSquares / input.length + epsilon);
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = Math.fround(Math.fround((input[index] ?? 0) * scale) * (weight[index] ?? 1));
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

function compactKey(layer: MtpTargetKvLayerView, keyValueTokenCount: number): Float32Array {
  const length = keyValueTokenCount * layer.headCountKv * layer.keyLength;
  return layer.key.length === length ? layer.key : layer.key.slice(0, length);
}

function compactValue(layer: MtpTargetKvLayerView, keyValueTokenCount: number): Float32Array {
  const output = new Float32Array(layer.valueLength * layer.headCountKv * keyValueTokenCount);
  if (layer.value.length === output.length) {
    return layer.value;
  }
  for (let dim = 0; dim < layer.valueLength; dim += 1) {
    for (let head = 0; head < layer.headCountKv; head += 1) {
      for (let token = 0; token < keyValueTokenCount; token += 1) {
        output[(dim * layer.headCountKv + head) * keyValueTokenCount + token] =
          layer.value[(dim * layer.headCountKv + head) * layer.contextLength + token] ?? 0;
      }
    }
  }
  return output;
}

function targetKvKeyValueTokenCount(layer: MtpTargetKvLayerView, position: number): number {
  return Math.min(layer.tokenCount, layer.contextLength, position - layer.logicalStart + 1);
}

function attentionMask(position: number, logicalStart: number, keyValueTokenCount: number, slidingWindow?: number): Float32Array {
  const output = new Float32Array(keyValueTokenCount);
  const minPosition = slidingWindow === undefined ? 0 : Math.max(0, position - slidingWindow + 1);
  for (let keyToken = 0; keyToken < keyValueTokenCount; keyToken += 1) {
    const logicalPosition = logicalStart + keyToken;
    output[keyToken] = logicalPosition <= position && logicalPosition >= minPosition ? 0 : -Infinity;
  }
  return output;
}

function residualAdd(left: Float32Array, right: Float32Array): Float32Array {
  const output = new Float32Array(left.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Math.fround((left[index] ?? 0) + (right[index] ?? 0));
  }
  return output;
}

function ropeNeox(
  input: Float32Array,
  options: {
    headSize: number;
    headCount: number;
    position: number;
    nDims: number;
    freqBase: number;
    freqFactors?: Float32Array;
    activePairCount?: number;
  },
): Float32Array {
  const output = new Float32Array(input);
  const thetaScale = Math.pow(options.freqBase, -2 / options.nDims);
  for (let head = 0; head < options.headCount; head += 1) {
    const rowOffset = head * options.headSize;
    let theta = options.position;
    const pairCount = options.activePairCount ?? options.nDims / 2;
    for (let i0 = 0; i0 < pairCount * 2; i0 += 2) {
      const index = i0 / 2;
      const x0 = input[rowOffset + index] ?? 0;
      const x1 = input[rowOffset + options.nDims / 2 + index] ?? 0;
      const t = theta / (options.freqFactors?.[index] ?? 1);
      output[rowOffset + index] = Math.fround(Math.fround(x0 * Math.cos(t)) - Math.fround(x1 * Math.sin(t)));
      output[rowOffset + options.nDims / 2 + index] = Math.fround(Math.fround(x0 * Math.sin(t)) + Math.fround(x1 * Math.cos(t)));
      theta = Math.fround(theta * thetaScale);
    }
  }
  return output;
}

async function readRopeFreqFactors(
  session: MtpAssistantSession,
  kind: "sliding-attention" | "full-attention",
) {
  return kind === "full-attention" && session.hasTensor("rope_freqs.weight")
    ? session.readF32Tensor("rope_freqs.weight")
    : undefined;
}

function gelu(value: number): number {
  if (value <= -10) {
    return 0;
  }
  if (value >= 10) {
    return value;
  }
  const x = value;
  const inner = Math.fround(
    Math.fround(Math.sqrt(2 / Math.PI) * x) *
      Math.fround(1 + Math.fround(0.044715 * Math.fround(x * x))),
  );
  return Math.fround(Math.fround(0.5 * x) * Math.fround(1 + Math.tanh(inner)));
}

function incrementWasmAssistantRuns(session: MtpAssistantSession): void {
  const previous = session.cacheStats().executionProviderStats.wasmMtpAssistantRuns;
  const runs = typeof previous === "number" ? previous + 1 : 1;
  session.setExecutionProviderStatsProvider(() => ({ wasmMtpAssistantRuns: runs }), "wasm-mtp-assistant");
}
