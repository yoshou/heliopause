import {
  buildMtpAssistantManifest,
  type MtpAssistantManifest,
} from "./model";
import {
  dequantizeRow,
} from "./quant";
import type {
  CacheStats,
  ExecutionProviderStats,
} from "./runtime";
import {
  runMtpAssistant as dispatchMtpAssistant,
} from "./runner/mtp-assistant";
import type {
  MtpAssistantRunInput,
  MtpAssistantRunners,
} from "./runner/mtp-assistant-runner";
import type {
  MtpAssistantRunnerProvider,
  RunnerProvider,
} from "./runner/provider";
import {
  validateProviderList,
} from "./runner/provider";
import type {
  SegmentRunnerProvider,
} from "./runner/segment-runner";
import {
  GgufTensorReader,
  tensorByteLength,
} from "./tensor-reader";

export type {
  MtpAssistantRunInput,
  MtpAssistantRunner,
  MtpAssistantRunners,
  MtpTargetKvLayerView,
  MtpTargetKvView,
} from "./runner/mtp-assistant-runner";

export type MtpAssistantForwardIntermediates = {
  preProjection: Float32Array;
  layerOutputs: Float32Array[];
  normalizedHidden: Float32Array;
  postProjection: Float32Array;
  centroidLogits: Float32Array;
};

export type MtpAssistantRunResult = {
  hidden?: Float32Array;
  backboneHidden: Float32Array;
  topTokens: Array<{ id: number; value: number }>;
  intermediates?: MtpAssistantForwardIntermediates;
};

export type MtpAssistantSessionOptions = {
  maxWeightCacheBytes?: number;
  providers: readonly MtpAssistantRunnerProvider[];
};

export class MtpAssistantSession {
  readonly tensorReader: GgufTensorReader;
  readonly manifest: MtpAssistantManifest;
  readonly epsilon: number;
  readonly providers: readonly MtpAssistantRunnerProvider[];
  readonly assistantRunners: readonly MtpAssistantRunners[];

  private readonly maxWeightCacheBytes: number;
  private readonly f32TensorCache = new Map<string, Float32Array>();
  private readonly i32TensorCache = new Map<string, Int32Array>();
  private readonly weightBytesCache = new Map<string, Uint8Array>();
  private readonly embeddingRowCache = new Map<number, Float32Array>();
  private readonly executionProviderStatsProviders = new Map<string, () => ExecutionProviderStats>();
  private readonly disposeCallbacks = new Set<() => void>();
  private weightCacheBytes = 0;
  private weightCacheHits = 0;
  private weightCacheMisses = 0;
  private weightCacheEvictions = 0;

  constructor(tensorReader: GgufTensorReader, options: MtpAssistantSessionOptions) {
    this.tensorReader = tensorReader;
    this.manifest = buildMtpAssistantManifest(tensorReader.metadata);
    this.epsilon = this.manifest.layerNormEpsilon;
    this.maxWeightCacheBytes = options.maxWeightCacheBytes ?? 256 * 1024 * 1024;
    this.providers = validateProviderList(options.providers, "createMtpAssistantRunners");
    this.assistantRunners = this.providers.map((provider) => provider.createMtpAssistantRunners());
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

  async readI32Tensor(name: string): Promise<Int32Array> {
    const cached = this.i32TensorCache.get(name);
    if (cached) {
      return cached;
    }
    const tensor = this.getTensor(name);
    if (tensor.type !== "I32") {
      throw new Error(`${name} must be I32, got ${tensor.type}`);
    }
    const bytes = await this.tensorReader.readTensorBytes(name);
    const value = new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
    this.i32TensorCache.set(name, value);
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
    const rowCount = tokenEmbedding.dimensions[1] ?? 0;
    const rowByteLength = tensorByteLength({ ...tokenEmbedding, dimensions: [rowElements] });
    const rows = new Float32Array(rowElements * tokenIds.length);

    for (let index = 0; index < tokenIds.length; index += 1) {
      const tokenId = tokenIds[index] ?? 0;
      if (tokenId < 0 || tokenId >= rowCount) {
        throw new Error(`token_embd.weight row ${tokenId} is outside ${rowCount}`);
      }
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
      f32TensorCount: this.f32TensorCache.size + this.i32TensorCache.size,
      weightTensorCount: this.weightBytesCache.size,
      weightCacheBytes: this.weightCacheBytes,
      maxWeightCacheBytes: this.maxWeightCacheBytes,
      weightCacheHits: this.weightCacheHits,
      weightCacheMisses: this.weightCacheMisses,
      weightCacheEvictions: this.weightCacheEvictions,
      embeddingRowCount: this.embeddingRowCache.size,
      executionProviderStats: this.executionProviderStats(),
    };
  }

  provider<TProvider extends RunnerProvider = RunnerProvider>(name: SegmentRunnerProvider): TProvider | undefined {
    return this.providers.find((provider) => provider.name === name) as TProvider | undefined;
  }

  hasProvider(name: SegmentRunnerProvider): boolean {
    return this.provider(name) !== undefined;
  }

  setExecutionProviderStatsProvider(
    provider: (() => ExecutionProviderStats) | undefined,
    name = "default",
  ): void {
    if (!provider) {
      this.executionProviderStatsProviders.delete(name);
      return;
    }
    this.executionProviderStatsProviders.set(name, provider);
  }

  addDisposeCallback(callback: () => void): void {
    this.disposeCallbacks.add(callback);
  }

  dispose(): void {
    for (const callback of this.disposeCallbacks) {
      callback();
    }
    this.disposeCallbacks.clear();
    this.executionProviderStatsProviders.clear();
    this.f32TensorCache.clear();
    this.i32TensorCache.clear();
    this.weightBytesCache.clear();
    this.embeddingRowCache.clear();
    this.weightCacheBytes = 0;
  }

  private executionProviderStats(): ExecutionProviderStats {
    const stats: ExecutionProviderStats = {};
    for (const provider of this.executionProviderStatsProviders.values()) {
      Object.assign(stats, provider());
    }
    return stats;
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

export function createMtpAssistantSession(
  tensorReader: GgufTensorReader,
  options: MtpAssistantSessionOptions,
): MtpAssistantSession {
  return new MtpAssistantSession(tensorReader, options);
}

export function runMtpAssistant(
  session: MtpAssistantSession,
  input: MtpAssistantRunInput,
  options: { signal?: AbortSignal } = {},
): Promise<MtpAssistantRunResult> {
  return dispatchMtpAssistant(session, input, options);
}
