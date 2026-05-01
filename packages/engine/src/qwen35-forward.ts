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
  type Qwen35LayerKind,
  type Qwen35ModelManifest,
} from "./qwen35";
import {
  gatedDeltaNetWasm,
  gqaAttentionWasm,
  matMulQuantizedMultiWasm,
  matMulQuantizedWasmResidentMulti,
  matMulQuantizedWasmResident,
  matMulQuantizedWasm,
  createWasmQuantizedWeightHandle,
  type WasmQuantizedWeightHandle,
  type QuantizedMatMulInput,
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
  contextLength: number;
  nextPosition: number;
};

export type Qwen35PrefillOptions = {
  positions?: Int32Array | number[];
  state?: Qwen35InferenceState;
  computeLogits?: boolean;
  logitsTopK?: number;
  onTiming?: Qwen35TimingSink;
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
  onTiming?: Qwen35TimingSink;
};

export type Qwen35DecodeResult = {
  hidden: Float32Array;
  state: Qwen35InferenceState;
  logits: Float32Array;
  topTokens: Array<{ id: number; value: number }>;
};

export type Qwen35ModelSessionOptions = {
  maxContextLength?: number;
  maxWeightCacheBytes?: number;
  enableProjectionBatching?: boolean;
  enableWasmWeightCache?: boolean;
};

export type Qwen35TimingPhase = "prefill" | "decode";

export type Qwen35TimingEvent = {
  phase: Qwen35TimingPhase;
  section: string;
  durationMs: number;
  layer?: number;
  layerKind?: Qwen35LayerKind;
  weightName?: string;
  tokenIndex?: number;
};

export type Qwen35TimingSink = (event: Qwen35TimingEvent) => void;

export type Qwen35ForwardTrace = {
  phase: Qwen35TimingPhase;
  onTiming: Qwen35TimingSink;
};

export type Qwen35CacheStats = {
  f32TensorCount: number;
  weightTensorCount: number;
  weightCacheBytes: number;
  maxWeightCacheBytes: number;
  weightCacheHits: number;
  weightCacheMisses: number;
  weightCacheEvictions: number;
  wasmWeightCacheEnabled: boolean;
  wasmWeightCacheCount: number;
  wasmWeightCacheBytes: number;
  wasmWeightCacheHits: number;
  wasmWeightCacheMisses: number;
  embeddingRowCount: number;
};

export class Qwen35ModelSession {
  readonly tensorReader: GgufTensorReader;
  readonly manifest: Qwen35ModelManifest;
  readonly epsilon: number;

  private readonly maxContextLength?: number;
  private readonly maxWeightCacheBytes: number;
  readonly enableProjectionBatching: boolean;
  readonly enableWasmWeightCache: boolean;
  private readonly f32TensorCache = new Map<string, Float32Array>();
  private readonly weightBytesCache = new Map<string, Uint8Array>();
  private readonly wasmWeightHandleCache = new Map<string, WasmQuantizedWeightHandle>();
  private readonly embeddingRowCache = new Map<number, Float32Array>();
  private weightCacheBytes = 0;
  private weightCacheHits = 0;
  private weightCacheMisses = 0;
  private weightCacheEvictions = 0;
  private wasmWeightCacheBytes = 0;
  private wasmWeightCacheHits = 0;
  private wasmWeightCacheMisses = 0;

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
    this.maxContextLength = options.maxContextLength;
    this.maxWeightCacheBytes = options.maxWeightCacheBytes ?? 256 * 1024 * 1024;
    this.enableProjectionBatching = options.enableProjectionBatching ?? false;
    this.enableWasmWeightCache = options.enableWasmWeightCache ?? false;
  }

  createInferenceState(): Qwen35InferenceState {
    return createQwen35InferenceState(this.manifest, {
      contextLength: this.maxContextLength === undefined
        ? this.manifest.contextLength
        : Math.min(this.manifest.contextLength, this.maxContextLength),
    });
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
      this.weightCacheHits += 1;
      this.weightBytesCache.delete(name);
      this.weightBytesCache.set(name, cached);
      return cached;
    }

    this.weightCacheMisses += 1;
    const bytes = await this.tensorReader.readTensorBytes(name);
    if (bytes.byteLength <= this.maxWeightCacheBytes) {
      this.weightBytesCache.set(name, bytes);
      this.weightCacheBytes += bytes.byteLength;
      this.evictWeightBytes();
    }
    return bytes;
  }

  async readWasmWeightHandle(
    name: string,
    type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0",
    inputSize: number,
    rowCount: number,
  ): Promise<WasmQuantizedWeightHandle | undefined> {
    if (!this.enableWasmWeightCache) {
      return undefined;
    }
    const cached = this.wasmWeightHandleCache.get(name);
    if (cached) {
      this.wasmWeightCacheHits += 1;
      return cached;
    }

    this.wasmWeightCacheMisses += 1;
    const bytes = await this.tensorReader.readTensorBytes(name);
    const handle = await createWasmQuantizedWeightHandle(type, bytes, inputSize, rowCount);
    if (!handle) {
      return undefined;
    }
    this.wasmWeightHandleCache.set(name, handle);
    this.wasmWeightCacheBytes += handle.byteLength;
    return handle;
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

  cacheStats(): Qwen35CacheStats {
    return {
      f32TensorCount: this.f32TensorCache.size,
      weightTensorCount: this.weightBytesCache.size,
      weightCacheBytes: this.weightCacheBytes,
      maxWeightCacheBytes: this.maxWeightCacheBytes,
      weightCacheHits: this.weightCacheHits,
      weightCacheMisses: this.weightCacheMisses,
      weightCacheEvictions: this.weightCacheEvictions,
      wasmWeightCacheEnabled: this.enableWasmWeightCache,
      wasmWeightCacheCount: this.wasmWeightHandleCache.size,
      wasmWeightCacheBytes: this.wasmWeightCacheBytes,
      wasmWeightCacheHits: this.wasmWeightCacheHits,
      wasmWeightCacheMisses: this.wasmWeightCacheMisses,
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
      this.weightCacheEvictions += 1;
    }
  }
}

export function estimateQwen35WeightCacheBytes(tensorReader: GgufTensorReader): number {
  let total = 0;
  for (const tensor of tensorReader.metadata.tensors) {
    if (tensor.name === "token_embd.weight") {
      continue;
    }
    if (!isCachedMatmulWeightType(tensor.type)) {
      continue;
    }
    total += tensorByteLength(tensor);
  }
  return total;
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
  options: { contextLength?: number } = {},
): Qwen35InferenceState {
  const recurrent = new Map<number, Qwen35RecurrentCache>();
  const fullAttention = new Map<number, Qwen35FullAttentionCache>();
  const contextLength = options.contextLength ?? manifest.contextLength;
  const convDim =
    manifest.ssm.stateSize * manifest.ssm.groupCount * 2 +
    manifest.ssm.stateSize * manifest.ssm.timeStepRank;
  const recurrentStateSize =
    manifest.ssm.stateSize * manifest.ssm.stateSize * manifest.ssm.timeStepRank;
  const fullCacheSize = contextLength * manifest.headCountKv * manifest.keyLength;

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

  return { recurrent, fullAttention, contextLength, nextPosition: 0 };
}

export async function prefillQwen35(
  model: Qwen35ModelInput,
  tokenIds: readonly number[],
  options: Qwen35PrefillOptions = {},
): Promise<Qwen35PrefillResult> {
  const session = modelSession(model);
  const { manifest } = session;
  const state = options.state ?? session.createInferenceState();
  const positions = normalizePositions(options.positions, tokenIds.length);
  const epsilon = session.epsilon;
  const trace = createForwardTrace("prefill", options.onTiming);

  let hidden = await timedAsync(trace, "embedding read", () => session.readEmbeddingRows(tokenIds));
  for (let layer = 0; layer < manifest.blockCount; layer += 1) {
    const isFullAttention = manifest.fullAttentionLayers.includes(layer);
    hidden = await timedAsync(
      trace,
      "layer total",
      () => isFullAttention
        ? forwardQwen35FullAttentionLayer(session, manifest, state, layer, hidden, positions, epsilon, trace)
        : forwardQwen35RecurrentLayer(session, manifest, state, layer, hidden, epsilon, trace),
      { layer, layerKind: isFullAttention ? "full-attention" : "recurrent" },
    );
  }
  updateNextPosition(state, positions, tokenIds.length);

  const result: Qwen35PrefillResult = { hidden, state };
  if (options.computeLogits) {
    const norm = await timedAsync(
      trace,
      "final norm",
      async () => rmsNorm(
        hidden,
        await readF32ModelTensor(session, "output_norm.weight"),
        epsilon,
      ),
    );
    const logits = await timedAsync(
      trace,
      "output logits",
      () => matMulQwen35Weight(session, "output.weight", norm, trace),
      { weightName: "output.weight" },
    );
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
  const state = options.state ?? session.createInferenceState();
  const position = options.position ?? state.nextPosition;
  const positions = new Int32Array([position]);
  const epsilon = session.epsilon;
  const trace = createForwardTrace("decode", options.onTiming);

  let hidden = await timedAsync(trace, "embedding read", () => session.readEmbeddingRows([tokenId]));
  for (let layer = 0; layer < manifest.blockCount; layer += 1) {
    const isFullAttention = manifest.fullAttentionLayers.includes(layer);
    hidden = await timedAsync(
      trace,
      "layer total",
      () => isFullAttention
        ? forwardQwen35FullAttentionLayer(session, manifest, state, layer, hidden, positions, epsilon, trace)
        : forwardQwen35RecurrentLayer(session, manifest, state, layer, hidden, epsilon, trace),
      { layer, layerKind: isFullAttention ? "full-attention" : "recurrent" },
    );
  }
  state.nextPosition = Math.max(state.nextPosition, position + 1);

  const norm = await timedAsync(
    trace,
    "final norm",
    async () => rmsNorm(
      hidden,
      await readF32ModelTensor(session, "output_norm.weight"),
      epsilon,
    ),
  );
  const logits = await timedAsync(
    trace,
    "output logits",
    () => matMulQwen35Weight(session, "output.weight", norm, trace),
    { weightName: "output.weight" },
  );
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
  trace?: Qwen35ForwardTrace,
): Promise<Float32Array> {
  const session = modelSession(model);
  const cache = requiredRecurrentCache(state, layer);
  const tokenCount = input.length / manifest.embeddingLength;
  const convDim =
    manifest.ssm.stateSize * manifest.ssm.groupCount * 2 +
    manifest.ssm.stateSize * manifest.ssm.timeStepRank;
  const valueDim = manifest.ssm.stateSize * manifest.ssm.timeStepRank;

  const attnNorm = await timedAsync(
    trace,
    "recurrent norm",
    async () => rmsNormRows(
      input,
      await readF32ModelTensor(session, `blk.${layer}.attn_norm.weight`),
      epsilon,
    ),
    { layer, layerKind: "recurrent" },
  );
  const recurrentProjectionNames = [
    `blk.${layer}.attn_qkv.weight`,
    `blk.${layer}.ssm_alpha.weight`,
    `blk.${layer}.ssm_beta.weight`,
    `blk.${layer}.attn_gate.weight`,
  ] as const;
  const recurrentProjectionBatch = await timedAsync(
    trace,
    "recurrent projection batch",
    () => matMulQwen35WeightBatch(session, recurrentProjectionNames, attnNorm, trace),
    { layer, layerKind: "recurrent" },
  );
  const qkv = recurrentProjectionBatch?.[0] ?? await timedAsync(
    trace,
    "recurrent projection qkv",
    () => matMulQwen35Weight(session, recurrentProjectionNames[0], attnNorm, trace),
    { layer, layerKind: "recurrent", weightName: recurrentProjectionNames[0] },
  );
  const alpha = recurrentProjectionBatch?.[1] ?? await timedAsync(
    trace,
    "recurrent projection alpha",
    () => matMulQwen35Weight(session, recurrentProjectionNames[1], attnNorm, trace),
    { layer, layerKind: "recurrent", weightName: recurrentProjectionNames[1] },
  );
  const beta = recurrentProjectionBatch?.[2] ?? await timedAsync(
    trace,
    "recurrent projection beta",
    () => matMulQwen35Weight(session, recurrentProjectionNames[2], attnNorm, trace),
    { layer, layerKind: "recurrent", weightName: recurrentProjectionNames[2] },
  );
  const z = recurrentProjectionBatch?.[3] ?? await timedAsync(
    trace,
    "recurrent projection z",
    () => matMulQwen35Weight(session, recurrentProjectionNames[3], attnNorm, trace),
    { layer, layerKind: "recurrent", weightName: recurrentProjectionNames[3] },
  );
  const convInput = timedSync(
    trace,
    "conv input compose",
    () => composeConvInput(cache.conv, qkv, convDim, tokenCount, manifest.ssm.convKernel),
    { layer, layerKind: "recurrent" },
  );
  const convKernel = await readF32ModelTensor(session, `blk.${layer}.ssm_conv1d.weight`);
  const convRaw = await timedAsync(
    trace,
    "conv kernel",
    async () => (await ssmConv1dWasm(
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
    ),
    { layer, layerKind: "recurrent" },
  );
  timedSync(
    trace,
    "conv state update",
    () => updateConvState(cache, convInput, convDim, tokenCount, manifest.ssm.convKernel),
    { layer, layerKind: "recurrent" },
  );
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
  const delta = await timedAsync(
    trace,
    "Gated DeltaNet",
    async () => (await gatedDeltaNetWasm(
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
    ),
    { layer, layerKind: "recurrent" },
  );
  cache.state = delta.newState;

  const ssmNormWeight = await readF32ModelTensor(session, `blk.${layer}.ssm_norm.weight`);
  const finalOutput = timedSync(trace, "SSM norm/gate", () => {
    const output = new Float32Array(delta.output.length);
    for (let row = 0; row < delta.output.length / ssmNormWeight.length; row += 1) {
      const offset = row * ssmNormWeight.length;
      const normalized = rmsNorm(delta.output.slice(offset, offset + ssmNormWeight.length), ssmNormWeight, epsilon);
      for (let index = 0; index < ssmNormWeight.length; index += 1) {
        const gateValue = z[offset + index] ?? 0;
        output[offset + index] = normalized[index] * (gateValue / (1 + Math.exp(-gateValue)));
      }
    }
    return output;
  }, { layer, layerKind: "recurrent" });

  const attention = await timedAsync(
    trace,
    "SSM out projection",
    () => matMulQ8_0Weight(session, `blk.${layer}.ssm_out.weight`, finalOutput, trace),
    { layer, layerKind: "recurrent", weightName: `blk.${layer}.ssm_out.weight` },
  );
  return timedAsync(
    trace,
    "FFN",
    () => forwardQwen35Ffn(session, layer, residualAdd(input, attention), epsilon, trace),
    { layer, layerKind: "recurrent" },
  );
}

export async function forwardQwen35FullAttentionLayer(
  model: Qwen35ModelInput,
  manifest: Qwen35ModelManifest,
  state: Qwen35InferenceState,
  layer: number,
  input: Float32Array,
  positions: Int32Array,
  epsilon = modelSession(model).epsilon,
  trace?: Qwen35ForwardTrace,
): Promise<Float32Array> {
  const session = modelSession(model);
  const cache = requiredFullAttentionCache(state, layer);
  const tokenCount = input.length / manifest.embeddingLength;
  const tokenPositions = tokenPositionsFromMrope(positions, tokenCount);
  const attnNorm = await timedAsync(
    trace,
    "full attention norm",
    async () => rmsNormRows(
      input,
      await readF32ModelTensor(session, `blk.${layer}.attn_norm.weight`),
      epsilon,
    ),
    { layer, layerKind: "full-attention" },
  );
  const fullProjectionNames = [
    `blk.${layer}.attn_q.weight`,
    `blk.${layer}.attn_k.weight`,
    `blk.${layer}.attn_v.weight`,
  ] as const;
  const fullProjectionBatch = await timedAsync(
    trace,
    "full attention projection batch",
    () => matMulQwen35WeightBatch(session, fullProjectionNames, attnNorm, trace),
    { layer, layerKind: "full-attention" },
  );
  const qFull = fullProjectionBatch?.[0] ?? await timedAsync(
    trace,
    "full attention projection q",
    () => matMulQwen35Weight(session, fullProjectionNames[0], attnNorm, trace),
    { layer, layerKind: "full-attention", weightName: fullProjectionNames[0] },
  );
  const q = sliceFullAttentionQ(qFull, manifest.headCount, manifest.keyLength, tokenCount);
  const gate = sliceFullAttentionGate(qFull, manifest.headCount, manifest.keyLength, tokenCount);
  const kProjection = fullProjectionBatch?.[1] ?? await timedAsync(
    trace,
    "full attention projection k",
    () => matMulQwen35Weight(session, fullProjectionNames[1], attnNorm, trace),
    { layer, layerKind: "full-attention", weightName: fullProjectionNames[1] },
  );
  const vProjection = fullProjectionBatch?.[2] ?? await timedAsync(
    trace,
    "full attention projection v",
    () => matMulQwen35Weight(session, fullProjectionNames[2], attnNorm, trace),
    { layer, layerKind: "full-attention", weightName: fullProjectionNames[2] },
  );
  const qNorm = await timedAsync(
    trace,
    "q norm",
    async () => normHeads(
      q,
      await readF32ModelTensor(session, `blk.${layer}.attn_q_norm.weight`),
      epsilon,
    ),
    { layer, layerKind: "full-attention" },
  );
  const kNorm = await timedAsync(
    trace,
    "k norm",
    async () => normHeads(
      kProjection,
      await readF32ModelTensor(session, `blk.${layer}.attn_k_norm.weight`),
      epsilon,
    ),
    { layer, layerKind: "full-attention" },
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
  const qRope = timedSync(trace, "RoPE q", () => ropeMultiMropeNeox(qNorm, {
    ...ropeCommon,
    headCount: manifest.headCount,
  }), { layer, layerKind: "full-attention" });
  const kRope = timedSync(trace, "RoPE k", () => ropeMultiMropeNeox(kNorm, {
    ...ropeCommon,
    headCount: manifest.headCountKv,
  }), { layer, layerKind: "full-attention" });
  timedSync(
    trace,
    "KV cache update",
    () => updateFullAttentionCache(cache, kRope, vProjection, tokenPositions, manifest, state.contextLength),
    { layer, layerKind: "full-attention" },
  );
  const keyValueTokenCount = Math.min(
    state.contextLength,
    Math.max(...Array.from(tokenPositions)) + 1,
  );
  const mask = timedSync(
    trace,
    "attention mask",
    () => causalMask(tokenPositions, keyValueTokenCount),
    { layer, layerKind: "full-attention" },
  );
  const attentionOptions = {
    headSize: manifest.keyLength,
    queryHeadCount: manifest.headCount,
    keyValueHeadCount: manifest.headCountKv,
    tokenCount,
    keyValueTokenCount,
    scale: 1 / Math.sqrt(manifest.keyLength),
    mask,
    valueLayout: "dim-head-token" as const,
    quantizeQueryForScore: "f16" as const,
  };
  const compactValue = timedSync(
    trace,
    "value compact",
    () => compactValueCache(cache.value, keyValueTokenCount, manifest, state.contextLength),
    { layer, layerKind: "full-attention" },
  );
  const attention = await timedAsync(
    trace,
    "GQA attention",
    async () => (await gqaAttentionWasm(
      qRope,
      cache.key.subarray(0, keyValueTokenCount * manifest.headCountKv * manifest.keyLength),
      compactValue,
      attentionOptions,
    )) ?? gqaAttention(
      qRope,
      cache.key.subarray(0, keyValueTokenCount * manifest.headCountKv * manifest.keyLength),
      compactValue,
      attentionOptions,
    ),
    { layer, layerKind: "full-attention" },
  );
  const gated = timedSync(trace, "attention gate", () => {
    const gateSigmoid = sigmoid(gate);
    const output = new Float32Array(attention.length);
    for (let index = 0; index < output.length; index += 1) {
      output[index] = (attention[index] ?? 0) * (gateSigmoid[index] ?? 0);
    }
    return output;
  }, { layer, layerKind: "full-attention" });
  const output = await timedAsync(
    trace,
    "full attention output projection",
    () => matMulQwen35Weight(session, `blk.${layer}.attn_output.weight`, gated, trace),
    { layer, layerKind: "full-attention", weightName: `blk.${layer}.attn_output.weight` },
  );
  const residual = layer === manifest.blockCount - 1
    ? residualAdd(input, output).slice(input.length - manifest.embeddingLength)
    : residualAdd(input, output);
  return timedAsync(
    trace,
    "FFN",
    () => forwardQwen35Ffn(session, layer, residual, epsilon, trace),
    { layer, layerKind: "full-attention" },
  );
}

async function forwardQwen35Ffn(
  session: Qwen35ModelSession,
  layer: number,
  residual: Float32Array,
  epsilon: number,
  trace?: Qwen35ForwardTrace,
): Promise<Float32Array> {
  const postNorm = await timedAsync(
    trace,
    "FFN post norm",
    async () => rmsNormRows(
      residual,
      await readF32ModelTensor(session, `blk.${layer}.post_attention_norm.weight`),
      epsilon,
    ),
    { layer },
  );
  const gateUpBatch = await timedAsync(
    trace,
    "FFN gate/up projection batch",
    () => matMulQwen35WeightBatch(session, [`blk.${layer}.ffn_gate.weight`, `blk.${layer}.ffn_up.weight`], postNorm, trace),
    { layer },
  );
  const gate = gateUpBatch?.[0] ?? await timedAsync(
    trace,
    "FFN gate projection",
    () => matMulQwen35Weight(session, `blk.${layer}.ffn_gate.weight`, postNorm, trace),
    { layer, weightName: `blk.${layer}.ffn_gate.weight` },
  );
  const up = gateUpBatch?.[1] ?? await timedAsync(
    trace,
    "FFN up projection",
    () => matMulQwen35Weight(session, `blk.${layer}.ffn_up.weight`, postNorm, trace),
    { layer, weightName: `blk.${layer}.ffn_up.weight` },
  );
  const swiglu = timedSync(trace, "FFN SwiGLU", () => {
    const output = new Float32Array(gate.length);
    for (let index = 0; index < output.length; index += 1) {
      const gateValue = gate[index] ?? 0;
      output[index] = (gateValue / (1 + Math.exp(-gateValue))) * (up[index] ?? 0);
    }
    return output;
  }, { layer });
  const ffnOut = await timedAsync(
    trace,
    "FFN down projection",
    () => matMulQwen35Weight(session, `blk.${layer}.ffn_down.weight`, swiglu, trace),
    { layer, weightName: `blk.${layer}.ffn_down.weight` },
  );
  return timedSync(trace, "FFN residual add", () => residualAdd(residual, ffnOut), { layer });
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
  trace?: Qwen35ForwardTrace,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  if (tensor.type === "F32") {
    return matMulF32Rows(session, weightName, inputColumns);
  }
  if (tensor.type === "Q4_K" || tensor.type === "Q5_K" || tensor.type === "Q6_K" || tensor.type === "IQ4_XS") {
    return matMulKQ8K(session, weightName, inputColumns, tensor.type, trace);
  }
  throw new Error(`${weightName} has unsupported matmul type ${tensor.type}`);
}

async function matMulQwen35WeightBatch(
  session: Qwen35ModelSession,
  weightNames: readonly string[],
  inputColumns: Float32Array,
  trace?: Qwen35ForwardTrace,
): Promise<Float32Array[] | undefined> {
  if (!session.enableProjectionBatching && !session.enableWasmWeightCache) {
    return undefined;
  }
  if (weightNames.length < 2 || weightNames.length > 4) {
    return undefined;
  }

  const tensors = weightNames.map((name) => session.getTensor(name));
  const inputSize = tensors[0]?.dimensions[0] ?? 0;
  if (inputSize <= 0 || inputColumns.length % inputSize !== 0) {
    return undefined;
  }
  const columnCount = inputColumns.length / inputSize;

  for (const tensor of tensors) {
    if (tensor.dimensions[0] !== inputSize || !isQuantizedMatmulWasmType(tensor.type)) {
      return undefined;
    }
  }

  if (session.enableWasmWeightCache) {
    const handles: WasmQuantizedWeightHandle[] = [];
    for (let index = 0; index < weightNames.length; index += 1) {
      const name = weightNames[index];
      const tensor = tensors[index];
      if (!name || !tensor || !isQuantizedMatmulWasmType(tensor.type)) {
        return undefined;
      }
      const handle = await session.readWasmWeightHandle(name, tensor.type, inputSize, tensor.dimensions[1] ?? 0);
      if (!handle) {
        return undefined;
      }
      handles.push(handle);
    }

    const outputs: Array<Float32Array | undefined> = new Array(handles.length);
    const groups = new Map<number, number[]>();
    for (let index = 0; index < handles.length; index += 1) {
      const handle = handles[index];
      if (!handle) {
        return undefined;
      }
      const group = groups.get(handle.instanceId) ?? [];
      group.push(index);
      groups.set(handle.instanceId, group);
    }

    for (const indices of groups.values()) {
      if (indices.length >= 2) {
        const groupHandles = indices.map((index) => handles[index]).filter((handle): handle is WasmQuantizedWeightHandle => Boolean(handle));
        const groupOutputs = await timedAsync(
          trace,
          "WASM resident matmul batch wrapper",
          () => matMulQuantizedWasmResidentMulti(groupHandles, inputColumns, inputSize, columnCount),
        );
        if (!groupOutputs || groupOutputs.length !== groupHandles.length) {
          throw new Error("Resident matmul batch did not return the expected outputs");
        }
        for (let outputIndex = 0; outputIndex < groupOutputs.length; outputIndex += 1) {
          outputs[indices[outputIndex] ?? 0] = groupOutputs[outputIndex];
        }
      } else {
        const index = indices[0] ?? 0;
        const handle = handles[index];
        if (!handle) {
          return undefined;
        }
        outputs[index] = await timedAsync(
          trace,
          "WASM resident matmul wrapper",
          () => matMulQuantizedWasmResident(
            handle,
            inputColumns,
            inputSize,
            handle.rowCount,
            columnCount,
          ),
        );
      }
    }

    if (outputs.every((output): output is Float32Array => output !== undefined)) {
      return outputs as Float32Array[];
    }
  }

  if (!session.enableProjectionBatching) {
    return undefined;
  }

  const weights: QuantizedMatMulInput[] = [];
  for (let index = 0; index < weightNames.length; index += 1) {
    const name = weightNames[index];
    const tensor = tensors[index];
    if (!name || !tensor || !isQuantizedMatmulWasmType(tensor.type)) {
      return undefined;
    }
    weights.push({
      type: tensor.type,
      weightBytes: await session.readWeightBytes(name),
      rowCount: tensor.dimensions[1] ?? 0,
    });
  }

  return timedAsync(
    trace,
    "WASM matmul batch wrapper",
    () => matMulQuantizedMultiWasm(weights, inputColumns, inputSize, columnCount),
  );
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
  trace?: Qwen35ForwardTrace,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  const rowByteLength = tensorByteLength({
    ...tensor,
    dimensions: [inputSize],
  });
  const wasmHandle = await session.readWasmWeightHandle(weightName, type, inputSize, rowCount);
  if (wasmHandle) {
    const wasm = await timedAsync(
      trace,
      "WASM resident matmul wrapper",
      () => matMulQuantizedWasmResident(
        wasmHandle,
        inputColumns,
        inputSize,
        rowCount,
        columnCount,
      ),
      { weightName },
    );
    if (wasm) {
      return wasm;
    }
  }
  const weightBytes = await session.readWeightBytes(weightName);
  const wasm = await timedAsync(
    trace,
    "WASM matmul wrapper",
    () => matMulQuantizedWasm(
      type,
      weightBytes,
      inputColumns,
      inputSize,
      rowCount,
      columnCount,
    ),
    { weightName },
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
  trace?: Qwen35ForwardTrace,
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
  const wasmHandle = await session.readWasmWeightHandle(weightName, "Q8_0", inputSize, rowCount);
  if (wasmHandle) {
    const wasm = await timedAsync(
      trace,
      "WASM resident matmul wrapper",
      () => matMulQuantizedWasmResident(
        wasmHandle,
        inputColumns,
        inputSize,
        rowCount,
        columnCount,
      ),
      { weightName },
    );
    if (wasm) {
      return wasm;
    }
  }
  const weightBytes = await session.readWeightBytes(weightName);
  const wasm = await timedAsync(
    trace,
    "WASM matmul wrapper",
    () => matMulQuantizedWasm(
      "Q8_0",
      weightBytes,
      inputColumns,
      inputSize,
      rowCount,
      columnCount,
    ),
    { weightName },
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
  contextLength: number,
): void {
  const tokenCount = positions.length;
  for (let token = 0; token < tokenCount; token += 1) {
    const position = positions[token] ?? 0;
    if (position < 0 || position >= contextLength) {
      throw new Error(`Position ${position} is outside context length ${contextLength}`);
    }
    for (let head = 0; head < manifest.headCountKv; head += 1) {
      for (let dim = 0; dim < manifest.keyLength; dim += 1) {
        const currentOffset = (token * manifest.headCountKv + head) * manifest.keyLength + dim;
        cache.key[(position * manifest.headCountKv + head) * manifest.keyLength + dim] =
          key[currentOffset] ?? 0;
        cache.value[(dim * manifest.headCountKv + head) * contextLength + position] =
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
  contextLength: number,
): Float32Array {
  const output = new Float32Array(manifest.keyLength * manifest.headCountKv * keyValueTokenCount);
  for (let dim = 0; dim < manifest.keyLength; dim += 1) {
    for (let head = 0; head < manifest.headCountKv; head += 1) {
      for (let token = 0; token < keyValueTokenCount; token += 1) {
        output[(dim * manifest.headCountKv + head) * keyValueTokenCount + token] =
          value[(dim * manifest.headCountKv + head) * contextLength + token] ?? 0;
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

function createForwardTrace(
  phase: Qwen35TimingPhase,
  onTiming: Qwen35TimingSink | undefined,
): Qwen35ForwardTrace | undefined {
  return onTiming ? { phase, onTiming } : undefined;
}

async function timedAsync<T>(
  trace: Qwen35ForwardTrace | undefined,
  section: string,
  run: () => Promise<T> | T,
  details: Omit<Qwen35TimingEvent, "phase" | "section" | "durationMs"> = {},
): Promise<T> {
  if (!trace) {
    return run();
  }
  const start = nowMs();
  try {
    return await run();
  } finally {
    trace.onTiming({
      phase: trace.phase,
      section,
      durationMs: nowMs() - start,
      ...details,
    });
  }
}

function timedSync<T>(
  trace: Qwen35ForwardTrace | undefined,
  section: string,
  run: () => T,
  details: Omit<Qwen35TimingEvent, "phase" | "section" | "durationMs"> = {},
): T {
  if (!trace) {
    return run();
  }
  const start = nowMs();
  try {
    return run();
  } finally {
    trace.onTiming({
      phase: trace.phase,
      section,
      durationMs: nowMs() - start,
      ...details,
    });
  }
}

function isCachedMatmulWeightType(type: GgmlTypeName): boolean {
  return type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "IQ4_XS" || type === "Q8_0";
}

function isQuantizedMatmulWasmType(
  type: GgmlTypeName,
): type is Extract<GgmlTypeName, "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0"> {
  return type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "IQ4_XS" || type === "Q8_0";
}

function nowMs(): number {
  return globalThis.performance?.now() ?? Date.now();
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
