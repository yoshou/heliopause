import {
  type GgmlTypeName,
} from "./gguf";
import {
  dequantizeRow,
} from "./quant";
import {
  type GgufTensorReader,
  tensorByteLength,
} from "./tensor-reader";
import {
  buildQwen35Manifest,
  type Qwen35LayerKind,
  type Qwen35ModelManifest,
} from "./model";

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

export type OutputResult = {
  logits: Float32Array;
  topTokens: Array<{ id: number; value: number }>;
};

export type Qwen35ModelSessionOptions = {
  maxContextLength?: number;
  maxWeightCacheBytes?: number;
  executionProviders?: readonly ExecutionProviderConfig[];
};

export type ExecutionProviderConfig = {
  name: string;
  options?: Readonly<Record<string, unknown>>;
};

export type TimingPhase = "prefill" | "decode";

export type TimingEvent = {
  phase: TimingPhase;
  section: string;
  durationMs: number;
  layer?: number;
  layerKind?: Qwen35LayerKind;
  weightName?: string;
  tokenIndex?: number;
};

export type TimingSink = (event: TimingEvent) => void;

export type ForwardTrace = {
  phase: TimingPhase;
  onTiming: TimingSink;
};

export type CacheStats = {
  f32TensorCount: number;
  weightTensorCount: number;
  weightCacheBytes: number;
  maxWeightCacheBytes: number;
  weightCacheHits: number;
  weightCacheMisses: number;
  weightCacheEvictions: number;
  embeddingRowCount: number;
  executionProviderStats: ExecutionProviderStats;
};

export type ExecutionProviderStats = Record<string, number | boolean | string>;

export class Qwen35ModelSession {
  readonly tensorReader: GgufTensorReader;
  readonly manifest: Qwen35ModelManifest;
  readonly epsilon: number;

  private readonly maxContextLength?: number;
  private readonly maxWeightCacheBytes: number;
  readonly executionProviders: readonly ExecutionProviderConfig[];
  private readonly f32TensorCache = new Map<string, Float32Array>();
  private readonly weightBytesCache = new Map<string, Uint8Array>();
  private readonly embeddingRowCache = new Map<number, Float32Array>();
  private executionProviderStatsProvider?: () => ExecutionProviderStats;
  private weightCacheBytes = 0;
  private weightCacheHits = 0;
  private weightCacheMisses = 0;
  private weightCacheEvictions = 0;

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
    this.executionProviders = (options.executionProviders ?? []).map((provider) => ({
      name: provider.name,
      options: provider.options ? { ...provider.options } : undefined,
    }));
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

  cacheStats(): CacheStats {
    return {
      f32TensorCount: this.f32TensorCache.size,
      weightTensorCount: this.weightBytesCache.size,
      weightCacheBytes: this.weightCacheBytes,
      maxWeightCacheBytes: this.maxWeightCacheBytes,
      weightCacheHits: this.weightCacheHits,
      weightCacheMisses: this.weightCacheMisses,
      weightCacheEvictions: this.weightCacheEvictions,
      embeddingRowCount: this.embeddingRowCache.size,
      executionProviderStats: this.executionProviderStatsProvider?.() ?? {},
    };
  }

  setExecutionProviderStatsProvider(provider: (() => ExecutionProviderStats) | undefined): void {
    this.executionProviderStatsProvider = provider;
  }

  executionProvider(name: string): ExecutionProviderConfig | undefined {
    return this.executionProviders.find((provider) => provider.name === name);
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

export function estimateWeightCacheBytes(tensorReader: GgufTensorReader): number {
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

export function cloneQwen35InferenceState(state: Qwen35InferenceState): Qwen35InferenceState {
  const recurrent = new Map<number, Qwen35RecurrentCache>();
  const fullAttention = new Map<number, Qwen35FullAttentionCache>();

  for (const [layer, cache] of state.recurrent) {
    recurrent.set(layer, {
      conv: cache.conv.slice(),
      state: cache.state.slice(),
    });
  }

  for (const [layer, cache] of state.fullAttention) {
    fullAttention.set(layer, {
      key: cache.key.slice(),
      value: cache.value.slice(),
    });
  }

  return {
    recurrent,
    fullAttention,
    contextLength: state.contextLength,
    nextPosition: state.nextPosition,
  };
}

export function modelSession(model: Qwen35ModelInput): Qwen35ModelSession {
  return model instanceof Qwen35ModelSession
    ? model
    : new Qwen35ModelSession(model);
}

export function requiredMetadataNumber(tensorReader: GgufTensorReader, key: string): number {
  const value = tensorReader.metadata.metadata[key];
  if (typeof value !== "number") {
    throw new Error(`Missing numeric metadata ${key}`);
  }
  return value;
}

export function requiredRecurrentCache(state: Qwen35InferenceState, layer: number): Qwen35RecurrentCache {
  const cache = state.recurrent.get(layer);
  if (!cache) {
    throw new Error(`Missing recurrent cache for layer ${layer}`);
  }
  return cache;
}

export function requiredFullAttentionCache(state: Qwen35InferenceState, layer: number): Qwen35FullAttentionCache {
  const cache = state.fullAttention.get(layer);
  if (!cache) {
    throw new Error(`Missing full-attention cache for layer ${layer}`);
  }
  return cache;
}

export function createForwardTrace(
  phase: TimingPhase,
  onTiming: TimingSink | undefined,
): ForwardTrace | undefined {
  return onTiming ? { phase, onTiming } : undefined;
}

export async function timedAsync<T>(
  trace: ForwardTrace | undefined,
  section: string,
  run: () => Promise<T> | T,
  details: Omit<TimingEvent, "phase" | "section" | "durationMs"> = {},
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

export function timedSync<T>(
  trace: ForwardTrace | undefined,
  section: string,
  run: () => T,
  details: Omit<TimingEvent, "phase" | "section" | "durationMs"> = {},
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

function nowMs(): number {
  return globalThis.performance?.now() ?? Date.now();
}

export function topK(values: Float32Array, k: number): Array<{ id: number; value: number }> {
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
