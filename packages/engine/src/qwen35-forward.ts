import {
  type GgmlTypeName,
} from "./gguf";
import {
  gatedDeltaNet,
  gqaAttention,
  l2NormRows,
  matMulRows,
  rmsNorm,
  ropeMultiMropeNeox,
  sigmoid,
  silu,
  softplus,
  ssmConv1d,
} from "./ops";
import {
  dequantizeRow,
  quantizeQ8_0,
  quantizeQ8_K,
  vecDotIQ4_XS_Q8_K,
  vecDotQ4_K_Q8_K,
  vecDotQ5_K_Q8_K,
  vecDotQ6_K_Q8_K,
  vecDotQ8_0_Q8_0,
} from "./quant";
import {
  type GgufTensorReader,
  tensorByteLength,
} from "./tensor-reader";
import {
  buildQwen35Manifest,
  type Qwen35ModelManifest,
} from "./qwen35";
import {
  gatedDeltaNetWasm,
  gqaAttentionWasm,
  matMulQuantizedWasm,
  ssmConv1dWasm,
} from "./prefill-wasm";

export type Qwen35FullAttentionCache = {
  key: Float32Array;
  value: Float32Array;
};

export type Qwen35RecurrentCache = {
  conv: Float32Array;
  state: Float32Array;
};

export type Qwen35InferenceState = {
  recurrent: Map<number, Qwen35RecurrentCache>;
  fullAttention: Map<number, Qwen35FullAttentionCache>;
  nextPosition: number;
};

export type Qwen35PrefillOptions = {
  positions?: Int32Array | number[];
  state?: Qwen35InferenceState;
  computeLogits?: boolean;
  logitsTopK?: number;
};

export type Qwen35PrefillResult = {
  hidden: Float32Array;
  state: Qwen35InferenceState;
  logits?: Float32Array;
  topTokens?: Array<{ id: number; value: number }>;
};

export type Qwen35DecodeOptions = {
  position?: number;
  state?: Qwen35InferenceState;
  logitsTopK?: number;
};

export type Qwen35DecodeResult = {
  hidden: Float32Array;
  state: Qwen35InferenceState;
  logits: Float32Array;
  topTokens: Array<{ id: number; value: number }>;
};

export type Qwen35ModelSessionOptions = {
  maxWeightCacheBytes?: number;
};

export class Qwen35ModelSession {
  readonly tensorReader: GgufTensorReader;
  readonly manifest: Qwen35ModelManifest;
  readonly epsilon: number;

  private readonly maxWeightCacheBytes: number;
  private readonly f32TensorCache = new Map<string, Float32Array>();
  private readonly weightBytesCache = new Map<string, Uint8Array>();
  private readonly embeddingRowCache = new Map<number, Float32Array>();
  private weightCacheBytes = 0;

  constructor(
    tensorReader: GgufTensorReader,
    options: Qwen35ModelSessionOptions = {},
  ) {
    this.tensorReader = tensorReader;
    this.manifest = buildQwen35Manifest(tensorReader.metadata);
    this.epsilon = requiredMetadataNumber(
      tensorReader,
      "qwen35.attention.layer_norm_rms_epsilon",
    );
    this.maxWeightCacheBytes = options.maxWeightCacheBytes ?? 256 * 1024 * 1024;
  }

  createInferenceState(): Qwen35InferenceState {
    return createQwen35InferenceState(this.manifest);
  }

  getTensor(name: string) {
    return this.tensorReader.getTensor(name);
  }

  async readF32Tensor(name: string): Promise<Float32Array> {
    const cached = this.f32TensorCache.get(name);
    if (cached) {
      return cached;
    }

    const tensor = this.tensorReader.getTensor(name);
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
      this.weightBytesCache.delete(name);
      this.weightBytesCache.set(name, cached);
      return cached;
    }

    const bytes = await this.tensorReader.readTensorBytes(name);
    if (bytes.byteLength <= this.maxWeightCacheBytes) {
      this.weightBytesCache.set(name, bytes);
      this.weightCacheBytes += bytes.byteLength;
      this.evictWeightBytes();
    }
    return bytes;
  }

  async readEmbeddingRows(tokenIds: readonly number[]): Promise<Float32Array> {
    const tokenEmbedding = this.tensorReader.getTensor("token_embd.weight");
    const rowElements = tokenEmbedding.dimensions[0] ?? 0;
    const rowByteLength = tensorByteLength({
      ...tokenEmbedding,
      dimensions: [rowElements],
    });
    const rows = new Float32Array(rowElements * tokenIds.length);

    for (let index = 0; index < tokenIds.length; index += 1) {
      const tokenId = tokenIds[index] ?? 0;
      let row = this.embeddingRowCache.get(tokenId);
      if (!row) {
        const rowBytes = await this.tensorReader.readTensorRange({
          tensor: tokenEmbedding,
          offset: BigInt(rowByteLength * tokenId),
          length: rowByteLength,
        });
        row = dequantizeRow(tokenEmbedding.type, rowBytes, rowElements);
        this.embeddingRowCache.set(tokenId, row);
      }
      rows.set(row, index * rowElements);
    }

    return rows;
  }

  cacheStats() {
    return {
      f32TensorCount: this.f32TensorCache.size,
      weightTensorCount: this.weightBytesCache.size,
      weightCacheBytes: this.weightCacheBytes,
      embeddingRowCount: this.embeddingRowCache.size,
    };
  }

  private evictWeightBytes(): void {
    while (this.weightCacheBytes > this.maxWeightCacheBytes) {
      const oldest = this.weightBytesCache.entries().next().value as [string, Uint8Array] | undefined;
      if (!oldest) {
        this.weightCacheBytes = 0;
        return;
      }
      this.weightBytesCache.delete(oldest[0]);
      this.weightCacheBytes -= oldest[1].byteLength;
    }
  }
}

export function createQwen35ModelSession(
  tensorReader: GgufTensorReader,
  options: Qwen35ModelSessionOptions = {},
): Qwen35ModelSession {
  return new Qwen35ModelSession(tensorReader, options);
}

export type Qwen35ModelInput = GgufTensorReader | Qwen35ModelSession;

export function createQwen35InferenceState(
  manifest: Qwen35ModelManifest,
): Qwen35InferenceState {
  const recurrent = new Map<number, Qwen35RecurrentCache>();
  const fullAttention = new Map<number, Qwen35FullAttentionCache>();
  const convDim =
    manifest.ssm.stateSize * manifest.ssm.groupCount * 2 +
    manifest.ssm.stateSize * manifest.ssm.timeStepRank;
  const recurrentStateSize =
    manifest.ssm.stateSize * manifest.ssm.stateSize * manifest.ssm.timeStepRank;
  const fullCacheSize = manifest.contextLength * manifest.headCountKv * manifest.keyLength;

  for (const layer of manifest.recurrentLayers) {
    recurrent.set(layer, {
      conv: new Float32Array((manifest.ssm.convKernel - 1) * convDim),
      state: new Float32Array(recurrentStateSize),
    });
  }

  for (const layer of manifest.fullAttentionLayers) {
    fullAttention.set(layer, {
      key: new Float32Array(fullCacheSize),
      value: new Float32Array(fullCacheSize),
    });
  }

  return { recurrent, fullAttention, nextPosition: 0 };
}

export async function prefillQwen35(
  model: Qwen35ModelInput,
  tokenIds: readonly number[],
  options: Qwen35PrefillOptions = {},
): Promise<Qwen35PrefillResult> {
  const session = modelSession(model);
  const { manifest } = session;
  const state = options.state ?? createQwen35InferenceState(manifest);
  const positions = normalizePositions(options.positions, tokenIds.length);
  const epsilon = session.epsilon;

  let hidden = await session.readEmbeddingRows(tokenIds);
  for (let layer = 0; layer < manifest.blockCount; layer += 1) {
    hidden = manifest.fullAttentionLayers.includes(layer)
      ? await forwardQwen35FullAttentionLayer(session, manifest, state, layer, hidden, positions, epsilon)
      : await forwardQwen35RecurrentLayer(session, manifest, state, layer, hidden, epsilon);
  }
  updateNextPosition(state, positions, tokenIds.length);

  const result: Qwen35PrefillResult = { hidden, state };
  if (options.computeLogits) {
    const norm = rmsNorm(
      hidden,
      await readF32ModelTensor(session, "output_norm.weight"),
      epsilon,
    );
    const logits = await matMulQwen35Weight(session, "output.weight", norm);
    result.logits = logits;
    result.topTokens = topK(logits, options.logitsTopK ?? 10);
  }

  return result;
}

export async function decodeQwen35(
  model: Qwen35ModelInput,
  tokenId: number,
  options: Qwen35DecodeOptions = {},
): Promise<Qwen35DecodeResult> {
  const session = modelSession(model);
  const { manifest } = session;
  const state = options.state ?? createQwen35InferenceState(manifest);
  const position = options.position ?? state.nextPosition;
  const positions = new Int32Array([position]);
  const epsilon = session.epsilon;

  let hidden = await session.readEmbeddingRows([tokenId]);
  for (let layer = 0; layer < manifest.blockCount; layer += 1) {
    hidden = manifest.fullAttentionLayers.includes(layer)
      ? await forwardQwen35FullAttentionLayer(session, manifest, state, layer, hidden, positions, epsilon)
      : await forwardQwen35RecurrentLayer(session, manifest, state, layer, hidden, epsilon);
  }
  state.nextPosition = Math.max(state.nextPosition, position + 1);

  const norm = rmsNorm(
    hidden,
    await readF32ModelTensor(session, "output_norm.weight"),
    epsilon,
  );
  const logits = await matMulQwen35Weight(session, "output.weight", norm);
  return {
    hidden,
    state,
    logits,
    topTokens: topK(logits, options.logitsTopK ?? 10),
  };
}

export async function forwardQwen35RecurrentLayer(
  model: Qwen35ModelInput,
  manifest: Qwen35ModelManifest,
  state: Qwen35InferenceState,
  layer: number,
  input: Float32Array,
  epsilon = modelSession(model).epsilon,
): Promise<Float32Array> {
  const session = modelSession(model);
  const cache = requiredRecurrentCache(state, layer);
  const tokenCount = input.length / manifest.embeddingLength;
  const convDim =
    manifest.ssm.stateSize * manifest.ssm.groupCount * 2 +
    manifest.ssm.stateSize * manifest.ssm.timeStepRank;
  const valueDim = manifest.ssm.stateSize * manifest.ssm.timeStepRank;

  const attnNorm = rmsNormRows(
    input,
    await readF32ModelTensor(session, `blk.${layer}.attn_norm.weight`),
    epsilon,
  );
  const qkv = await matMulQwen35Weight(session, `blk.${layer}.attn_qkv.weight`, attnNorm);
  const alpha = await matMulQwen35Weight(session, `blk.${layer}.ssm_alpha.weight`, attnNorm);
  const beta = await matMulQwen35Weight(session, `blk.${layer}.ssm_beta.weight`, attnNorm);
  const z = await matMulQwen35Weight(session, `blk.${layer}.attn_gate.weight`, attnNorm);
  const convInput = composeConvInput(cache.conv, qkv, convDim, tokenCount, manifest.ssm.convKernel);
  const convKernel = await readF32ModelTensor(session, `blk.${layer}.ssm_conv1d.weight`);
  const convRaw = (await ssmConv1dWasm(
    convInput,
    convKernel,
    convDim,
    tokenCount,
    manifest.ssm.convKernel,
  )) ?? ssmConv1d(
    convInput,
    convKernel,
    convDim,
    tokenCount,
    manifest.ssm.convKernel,
  );
  updateConvState(cache, convInput, convDim, tokenCount, manifest.ssm.convKernel);
  const convSilu = silu(convRaw);
  const qConv = l2NormRows(
    sliceConvChannels(convSilu, convDim, tokenCount, 0, manifest.ssm.stateSize * manifest.ssm.groupCount),
    manifest.ssm.stateSize,
    1e-6,
  );
  const kConv = l2NormRows(
    sliceConvChannels(
      convSilu,
      convDim,
      tokenCount,
      manifest.ssm.stateSize * manifest.ssm.groupCount,
      manifest.ssm.stateSize * manifest.ssm.groupCount,
    ),
    manifest.ssm.stateSize,
    1e-6,
  );
  const vConv = sliceConvChannels(
    convSilu,
    convDim,
    tokenCount,
    manifest.ssm.stateSize * manifest.ssm.groupCount * 2,
    valueDim,
  );
  const gate = recurrentDeltaGate(
    alpha,
    await readF32ModelTensor(session, `blk.${layer}.ssm_dt.bias`),
    await readF32ModelTensor(session, `blk.${layer}.ssm_a`),
  );
  const betaSigmoid = sigmoid(beta);
  const delta = (await gatedDeltaNetWasm(
    qConv,
    kConv,
    vConv,
    gate,
    betaSigmoid,
    cache.state,
    {
      stateSize: manifest.ssm.stateSize,
      keyHeadCount: manifest.ssm.groupCount,
      valueHeadCount: manifest.ssm.timeStepRank,
      tokenCount,
    },
  )) ?? gatedDeltaNet(
    qConv,
    kConv,
    vConv,
    gate,
    betaSigmoid,
    cache.state,
    {
      stateSize: manifest.ssm.stateSize,
      keyHeadCount: manifest.ssm.groupCount,
      valueHeadCount: manifest.ssm.timeStepRank,
      tokenCount,
    },
  );
  cache.state = delta.newState;

  const ssmNormWeight = await readF32ModelTensor(session, `blk.${layer}.ssm_norm.weight`);
  const finalOutput = new Float32Array(delta.output.length);
  for (let row = 0; row < delta.output.length / ssmNormWeight.length; row += 1) {
    const offset = row * ssmNormWeight.length;
    const normalized = rmsNorm(delta.output.slice(offset, offset + ssmNormWeight.length), ssmNormWeight, epsilon);
    for (let index = 0; index < ssmNormWeight.length; index += 1) {
      const gateValue = z[offset + index] ?? 0;
      finalOutput[offset + index] = normalized[index] * (gateValue / (1 + Math.exp(-gateValue)));
    }
  }

  const attention = await matMulQ8_0Weight(session, `blk.${layer}.ssm_out.weight`, finalOutput);
  return forwardQwen35Ffn(session, layer, residualAdd(input, attention), epsilon);
}

export async function forwardQwen35FullAttentionLayer(
  model: Qwen35ModelInput,
  manifest: Qwen35ModelManifest,
  state: Qwen35InferenceState,
  layer: number,
  input: Float32Array,
  positions: Int32Array,
  epsilon = modelSession(model).epsilon,
): Promise<Float32Array> {
  const session = modelSession(model);
  const cache = requiredFullAttentionCache(state, layer);
  const tokenCount = input.length / manifest.embeddingLength;
  const tokenPositions = tokenPositionsFromMrope(positions, tokenCount);
  const attnNorm = rmsNormRows(
    input,
    await readF32ModelTensor(session, `blk.${layer}.attn_norm.weight`),
    epsilon,
  );
  const qFull = await matMulQwen35Weight(session, `blk.${layer}.attn_q.weight`, attnNorm);
  const q = sliceFullAttentionQ(qFull, manifest.headCount, manifest.keyLength, tokenCount);
  const gate = sliceFullAttentionGate(qFull, manifest.headCount, manifest.keyLength, tokenCount);
  const kProjection = await matMulQwen35Weight(session, `blk.${layer}.attn_k.weight`, attnNorm);
  const vProjection = await matMulQwen35Weight(session, `blk.${layer}.attn_v.weight`, attnNorm);
  const qNorm = normHeads(
    q,
    await readF32ModelTensor(session, `blk.${layer}.attn_q_norm.weight`),
    epsilon,
  );
  const kNorm = normHeads(
    kProjection,
    await readF32ModelTensor(session, `blk.${layer}.attn_k_norm.weight`),
    epsilon,
  );
  const ropeCommon = {
    headSize: manifest.keyLength,
    tokenCount,
    positions: mropePositions(positions, tokenCount),
    nDims: manifest.rope.dimensionCount,
    sections: manifest.rope.dimensionSections,
    freqBase: manifest.rope.freqBase,
    nCtxOrig: manifest.contextLength,
  };
  const qRope = ropeMultiMropeNeox(qNorm, {
    ...ropeCommon,
    headCount: manifest.headCount,
  });
  const kRope = ropeMultiMropeNeox(kNorm, {
    ...ropeCommon,
    headCount: manifest.headCountKv,
  });
  updateFullAttentionCache(cache, kRope, vProjection, tokenPositions, manifest);
  const keyValueTokenCount = Math.min(
    manifest.contextLength,
    Math.max(...Array.from(tokenPositions)) + 1,
  );
  const attentionOptions = {
    headSize: manifest.keyLength,
    queryHeadCount: manifest.headCount,
    keyValueHeadCount: manifest.headCountKv,
    tokenCount,
    keyValueTokenCount,
    scale: 1 / Math.sqrt(manifest.keyLength),
    mask: causalMask(tokenPositions, keyValueTokenCount),
    valueLayout: "dim-head-token" as const,
    quantizeQueryForScore: "f16" as const,
  };
  const compactValue = compactValueCache(cache.value, keyValueTokenCount, manifest);
  const attention = (await gqaAttentionWasm(
    qRope,
    cache.key.subarray(0, keyValueTokenCount * manifest.headCountKv * manifest.keyLength),
    compactValue,
    attentionOptions,
  )) ?? gqaAttention(
    qRope,
    cache.key.subarray(0, keyValueTokenCount * manifest.headCountKv * manifest.keyLength),
    compactValue,
    attentionOptions,
  );
  const gateSigmoid = sigmoid(gate);
  const gated = new Float32Array(attention.length);
  for (let index = 0; index < gated.length; index += 1) {
    gated[index] = (attention[index] ?? 0) * (gateSigmoid[index] ?? 0);
  }
  const output = await matMulQwen35Weight(session, `blk.${layer}.attn_output.weight`, gated);
  const residual = layer === manifest.blockCount - 1
    ? residualAdd(input, output).slice(input.length - manifest.embeddingLength)
    : residualAdd(input, output);
  return forwardQwen35Ffn(session, layer, residual, epsilon);
}

async function forwardQwen35Ffn(
  session: Qwen35ModelSession,
  layer: number,
  residual: Float32Array,
  epsilon: number,
): Promise<Float32Array> {
  const postNorm = rmsNormRows(
    residual,
    await readF32ModelTensor(session, `blk.${layer}.post_attention_norm.weight`),
    epsilon,
  );
  const gate = await matMulQwen35Weight(session, `blk.${layer}.ffn_gate.weight`, postNorm);
  const up = await matMulQwen35Weight(session, `blk.${layer}.ffn_up.weight`, postNorm);
  const swiglu = new Float32Array(gate.length);
  for (let index = 0; index < swiglu.length; index += 1) {
    const gateValue = gate[index] ?? 0;
    swiglu[index] = (gateValue / (1 + Math.exp(-gateValue))) * (up[index] ?? 0);
  }
  const ffnOut = await matMulQwen35Weight(session, `blk.${layer}.ffn_down.weight`, swiglu);
  return residualAdd(residual, ffnOut);
}

async function readF32ModelTensor(
  session: Qwen35ModelSession,
  name: string,
): Promise<Float32Array> {
  return session.readF32Tensor(name);
}

async function matMulQwen35Weight(
  session: Qwen35ModelSession,
  weightName: string,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  if (tensor.type === "F32") {
    return matMulF32Rows(session, weightName, inputColumns);
  }
  if (tensor.type === "Q4_K" || tensor.type === "Q5_K" || tensor.type === "Q6_K" || tensor.type === "IQ4_XS") {
    return matMulKQ8K(session, weightName, inputColumns, tensor.type);
  }
  throw new Error(`${weightName} has unsupported matmul type ${tensor.type}`);
}

async function matMulF32Rows(
  session: Qwen35ModelSession,
  weightName: string,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  const weight = await readF32ModelTensor(session, weightName);
  const rows: Float32Array[] = [];

  for (let row = 0; row < rowCount; row += 1) {
    rows.push(weight.slice(row * inputSize, (row + 1) * inputSize));
  }

  return matMulRows(rows, inputColumns, inputSize, columnCount);
}

async function matMulKQ8K(
  session: Qwen35ModelSession,
  weightName: string,
  inputColumns: Float32Array,
  type: Extract<GgmlTypeName, "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS">,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  const rowByteLength = tensorByteLength({
    ...tensor,
    dimensions: [inputSize],
  });
  const weightBytes = await session.readWeightBytes(weightName);
  const wasm = await matMulQuantizedWasm(
    type,
    weightBytes,
    inputColumns,
    inputSize,
    rowCount,
    columnCount,
  );
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(rowCount * columnCount);

  for (let column = 0; column < columnCount; column += 1) {
    const q8 = quantizeQ8_K(inputColumns.slice(column * inputSize, (column + 1) * inputSize));
    for (let row = 0; row < rowCount; row += 1) {
      const rowOffset = row * rowByteLength;
      const rowBytes = weightBytes.subarray(rowOffset, rowOffset + rowByteLength);
      if (type === "Q4_K") {
        output[column * rowCount + row] = vecDotQ4_K_Q8_K(rowBytes, q8);
      } else if (type === "Q5_K") {
        output[column * rowCount + row] = vecDotQ5_K_Q8_K(rowBytes, q8);
      } else if (type === "Q6_K") {
        output[column * rowCount + row] = vecDotQ6_K_Q8_K(rowBytes, q8);
      } else {
        output[column * rowCount + row] = vecDotIQ4_XS_Q8_K(rowBytes, q8);
      }
    }
  }

  return output;
}

async function matMulQ8_0Weight(
  session: Qwen35ModelSession,
  weightName: string,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  if (tensor.type !== "Q8_0") {
    throw new Error(`${weightName} must be Q8_0, got ${tensor.type}`);
  }
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  const rowByteLength = tensorByteLength({
    ...tensor,
    dimensions: [inputSize],
  });
  const weightBytes = await session.readWeightBytes(weightName);
  const wasm = await matMulQuantizedWasm(
    "Q8_0",
    weightBytes,
    inputColumns,
    inputSize,
    rowCount,
    columnCount,
  );
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(rowCount * columnCount);

  for (let column = 0; column < columnCount; column += 1) {
    const q8 = quantizeQ8_0(inputColumns.slice(column * inputSize, (column + 1) * inputSize));
    for (let row = 0; row < rowCount; row += 1) {
      const rowOffset = row * rowByteLength;
      output[column * rowCount + row] = vecDotQ8_0_Q8_0(
        weightBytes.subarray(rowOffset, rowOffset + rowByteLength),
        q8,
      );
    }
  }

  return output;
}

function rmsNormRows(input: Float32Array, weight: Float32Array, epsilon: number): Float32Array {
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

function composeConvInput(
  convState: Float32Array,
  qkv: Float32Array,
  channelCount: number,
  tokenCount: number,
  kernelSize: number,
): Float32Array {
  const history = kernelSize - 1;
  const inputWindow = history + tokenCount;
  if (convState.length !== history * channelCount) {
    throw new Error(`Conv state shape mismatch: ${convState.length}`);
  }
  if (qkv.length !== tokenCount * channelCount) {
    throw new Error(`Conv QKV shape mismatch: ${qkv.length}`);
  }
  const output = new Float32Array(channelCount * inputWindow);
  for (let channel = 0; channel < channelCount; channel += 1) {
    const outputOffset = channel * inputWindow;
    const stateOffset = channel * history;
    for (let index = 0; index < history; index += 1) {
      output[outputOffset + index] = convState[stateOffset + index] ?? 0;
    }
    for (let token = 0; token < tokenCount; token += 1) {
      output[outputOffset + history + token] = qkv[token * channelCount + channel] ?? 0;
    }
  }
  return output;
}

function updateConvState(
  cache: Qwen35RecurrentCache,
  convInput: Float32Array,
  channelCount: number,
  tokenCount: number,
  kernelSize: number,
): void {
  const history = kernelSize - 1;
  const inputWindow = history + tokenCount;
  for (let channel = 0; channel < channelCount; channel += 1) {
    const source = channel * inputWindow + tokenCount;
    const target = channel * history;
    cache.conv.set(convInput.subarray(source, source + history), target);
  }
}

function sliceConvChannels(
  conv: Float32Array,
  channelCount: number,
  tokenCount: number,
  channelOffset: number,
  length: number,
): Float32Array {
  const output = new Float32Array(length * tokenCount);
  for (let token = 0; token < tokenCount; token += 1) {
    output.set(
      conv.slice(token * channelCount + channelOffset, token * channelCount + channelOffset + length),
      token * length,
    );
  }
  return output;
}

function recurrentDeltaGate(alpha: Float32Array, dtBias: Float32Array, ssmA: Float32Array): Float32Array {
  const alphaBiased = new Float32Array(alpha.length);
  for (let token = 0; token < alpha.length / dtBias.length; token += 1) {
    for (let index = 0; index < dtBias.length; index += 1) {
      alphaBiased[token * dtBias.length + index] =
        (alpha[token * dtBias.length + index] ?? 0) + (dtBias[index] ?? 0);
    }
  }
  const alphaSoftplus = softplus(alphaBiased);
  const gate = new Float32Array(alpha.length);
  for (let token = 0; token < alphaSoftplus.length / ssmA.length; token += 1) {
    for (let index = 0; index < ssmA.length; index += 1) {
      gate[token * ssmA.length + index] =
        (alphaSoftplus[token * ssmA.length + index] ?? 0) * (ssmA[index] ?? 0);
    }
  }
  return gate;
}

function sliceFullAttentionQ(
  qFull: Float32Array,
  headCount: number,
  headSize: number,
  tokenCount: number,
): Float32Array {
  const rowSize = headCount * headSize * 2;
  const qSize = headCount * headSize;
  const output = new Float32Array(qSize * tokenCount);
  for (let token = 0; token < tokenCount; token += 1) {
    for (let head = 0; head < headCount; head += 1) {
      const source = token * rowSize + head * headSize * 2;
      const target = token * qSize + head * headSize;
      output.set(qFull.slice(source, source + headSize), target);
    }
  }
  return output;
}

function sliceFullAttentionGate(
  qFull: Float32Array,
  headCount: number,
  headSize: number,
  tokenCount: number,
): Float32Array {
  const rowSize = headCount * headSize * 2;
  const gateSize = headCount * headSize;
  const output = new Float32Array(gateSize * tokenCount);
  for (let token = 0; token < tokenCount; token += 1) {
    for (let head = 0; head < headCount; head += 1) {
      const source = token * rowSize + head * headSize * 2 + headSize;
      const target = token * gateSize + head * headSize;
      output.set(qFull.slice(source, source + headSize), target);
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

function updateFullAttentionCache(
  cache: Qwen35FullAttentionCache,
  key: Float32Array,
  value: Float32Array,
  positions: Int32Array,
  manifest: Qwen35ModelManifest,
): void {
  const tokenCount = positions.length;
  for (let token = 0; token < tokenCount; token += 1) {
    const position = positions[token] ?? 0;
    if (position < 0 || position >= manifest.contextLength) {
      throw new Error(`Position ${position} is outside context length ${manifest.contextLength}`);
    }
    for (let head = 0; head < manifest.headCountKv; head += 1) {
      for (let dim = 0; dim < manifest.keyLength; dim += 1) {
        const currentOffset = (token * manifest.headCountKv + head) * manifest.keyLength + dim;
        cache.key[(position * manifest.headCountKv + head) * manifest.keyLength + dim] =
          key[currentOffset] ?? 0;
        cache.value[(dim * manifest.headCountKv + head) * manifest.contextLength + position] =
          value[currentOffset] ?? 0;
      }
    }
  }
}

function causalMask(positions: Int32Array, keyValueTokenCount: number): Float32Array {
  const output = new Float32Array(positions.length * keyValueTokenCount);
  for (let token = 0; token < positions.length; token += 1) {
    const position = positions[token] ?? 0;
    for (let keyToken = 0; keyToken < keyValueTokenCount; keyToken += 1) {
      output[token * keyValueTokenCount + keyToken] = keyToken <= position ? 0 : -Infinity;
    }
  }
  return output;
}

function mropePositions(positions: Int32Array, tokenCount: number): Int32Array {
  if (positions.length === tokenCount * 4) {
    return positions;
  }
  if (positions.length !== tokenCount) {
    throw new Error(`Expected ${tokenCount} or ${tokenCount * 4} positions, got ${positions.length}`);
  }
  const output = new Int32Array(tokenCount * 4);
  for (let section = 0; section < 4; section += 1) {
    output.set(positions, section * tokenCount);
  }
  return output;
}

function tokenPositionsFromMrope(positions: Int32Array, tokenCount: number): Int32Array {
  if (positions.length === tokenCount) {
    return positions;
  }
  if (positions.length === tokenCount * 4) {
    return positions.slice(0, tokenCount);
  }
  throw new Error(`Expected ${tokenCount} or ${tokenCount * 4} positions, got ${positions.length}`);
}

function compactValueCache(
  value: Float32Array,
  keyValueTokenCount: number,
  manifest: Qwen35ModelManifest,
): Float32Array {
  const output = new Float32Array(manifest.keyLength * manifest.headCountKv * keyValueTokenCount);
  for (let dim = 0; dim < manifest.keyLength; dim += 1) {
    for (let head = 0; head < manifest.headCountKv; head += 1) {
      for (let token = 0; token < keyValueTokenCount; token += 1) {
        output[(dim * manifest.headCountKv + head) * keyValueTokenCount + token] =
          value[(dim * manifest.headCountKv + head) * manifest.contextLength + token] ?? 0;
      }
    }
  }
  return output;
}

function normalizePositions(positions: Qwen35PrefillOptions["positions"], tokenCount: number): Int32Array {
  if (!positions) {
    return Int32Array.from({ length: tokenCount }, (_, index) => index);
  }
  const output = positions instanceof Int32Array ? positions : Int32Array.from(positions);
  if (output.length !== tokenCount && output.length !== tokenCount * 4) {
    throw new Error(`Expected ${tokenCount} or ${tokenCount * 4} positions, got ${output.length}`);
  }
  return output;
}

function updateNextPosition(
  state: Qwen35InferenceState,
  positions: Int32Array,
  tokenCount: number,
): void {
  if (tokenCount === 0) {
    return;
  }
  const tokenPositions = tokenPositionsFromMrope(positions, tokenCount);
  let nextPosition = state.nextPosition;
  for (const position of tokenPositions) {
    nextPosition = Math.max(nextPosition, position + 1);
  }
  state.nextPosition = nextPosition;
}

function modelSession(model: Qwen35ModelInput): Qwen35ModelSession {
  return model instanceof Qwen35ModelSession
    ? model
    : new Qwen35ModelSession(model);
}

function requiredMetadataNumber(tensorReader: GgufTensorReader, key: string): number {
  const value = tensorReader.metadata.metadata[key];
  if (typeof value !== "number") {
    throw new Error(`Missing numeric metadata ${key}`);
  }
  return value;
}

function requiredRecurrentCache(state: Qwen35InferenceState, layer: number): Qwen35RecurrentCache {
  const cache = state.recurrent.get(layer);
  if (!cache) {
    throw new Error(`Missing recurrent cache for layer ${layer}`);
  }
  return cache;
}

function requiredFullAttentionCache(state: Qwen35InferenceState, layer: number): Qwen35FullAttentionCache {
  const cache = state.fullAttention.get(layer);
  if (!cache) {
    throw new Error(`Missing full-attention cache for layer ${layer}`);
  }
  return cache;
}

function topK(values: Float32Array, k: number): Array<{ id: number; value: number }> {
  const best: Array<{ id: number; value: number }> = [];
  for (let id = 0; id < values.length; id += 1) {
    const value = values[id] ?? 0;
    if (best.length < k || value > (best[best.length - 1]?.value ?? -Infinity)) {
      best.push({ id, value });
      best.sort((left, right) => right.value - left.value);
      if (best.length > k) {
        best.pop();
      }
    }
  }
  return best;
}
