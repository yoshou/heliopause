import type { GgmlTypeName } from "../../gguf";
import { type GgufTensorReader, type TensorByteRange } from "../../tensor-reader";
import type { ModelManifest } from "../../model";
import { addInferenceStateDisposeCallback, kvCacheCapacity, type InferenceState } from "../../runtime";
import { dequantizeRow, float16ToFloat32 } from "../../quant";
import type { WebGpuOptimizationLevel } from "./execution-provider";
import type {
  MtpTargetKvLayerView,
  MtpTargetKvView,
} from "../mtp-assistant-runner";
import { GPU_COPY_DST, GPU_COPY_SRC, GPU_MAP_READ, GPU_QUERY_RESOLVE, GPU_SHADER_STAGE_COMPUTE, GPU_STORAGE, GPU_UNIFORM, WEBGPU_MEMORY_LIMIT_BYTES } from "./gpu-constants";
import { bindBuffer, storageEntry } from "./gpu-bindings";
import { webGpuAdapterLimits, webGpuDevice } from "./gpu-device";
import {
  GpuMemoryArena,
  scratchF32,
  scratchQ8_0,
  scratchQ8K,
  type F32Handle,
  type GpuResource,
  type QuantizedHandle,
} from "./arena";
import {
  dispatchBatchedFullAttentionRollingTile,
  dispatchBatchedFullKvUpdate,
  dispatchBatchedFullQuery,
  dispatchBatchedGegluSlice,
  dispatchBatchedRmsNormQ8_0Quantize,
  dispatchBatchedRmsNormQ8KQuantize,
  dispatchBatchedRmsNormResidualAdd,
  dispatchBatchedRmsNormResidualAddScale,
  dispatchDualQ4KMatMul,
  dispatchDualQ4_0MatMul,
  dispatchF32GatherRowsScale,
  dispatchF32MatMul,
  dispatchFullAttentionApply,
  dispatchFullAttentionScore,
  dispatchFullKvUpdate,
  dispatchFullQuery,
  dispatchGeglu,
  dispatchGegluSlice,
  dispatchHeadRmsNorm,
  dispatchHeadRmsNormNoWeight,
  dispatchKeyCacheRope,
  dispatchKMatMul,
  dispatchQ8_0MatMul,
  dispatchQ8_0Quantize,
  dispatchQ8KQuantize,
  dispatchPreparePerLayerInputs,
  dispatchQuantizedGatherRowsScale,
  dispatchResidualAdd,
  dispatchRope,
  dispatchRmsNorm,
  dispatchRmsNormQ8KQuantize,
  dispatchRmsNormResidualAdd,
  dispatchRmsNormResidualAddScale,
  dispatchScale,
  dispatchSelectTop1Candidate,
  dispatchTokenSlice,
  dispatchTop1Chunks,
  dispatchTopK,
  dispatchValueCacheWrite,
  TOP1_CHUNK_SIZE,
} from "./dispatch";
import {
  loadF32Handle,
  loadGpuLayer,
  loadOutputStripes,
  loadQuantizedHandle,
  type GpuLayer,
  type OutputStripe,
} from "./segment-layer-loader";
import { createQuantizedHandleFromBytes, webGpuMatMulType, webGpuQuantizedWeightLayout } from "./quantized-handles";
import {
  diffWebGpuRuntimeResourceStats,
  installWebGpuRuntimeResourceCache,
  runtimeResourceCreateMs,
  type WebGpuRuntimeResourceCache,
} from "./runtime-resources";
import type { WebGpuBufferLike, WebGpuCommandEncoderLike, WebGpuComputePassLike, WebGpuDeviceLike, WebGpuQuerySetLike, WebGpuTopToken } from "./gpu-types";
import type { SegmentRunner } from "../segment-runner";

export type WebGpuSegmentRunnerOptions = {
  tensorReader: GgufTensorReader;
  manifest: ModelManifest;
  epsilon: number;
  contextLength: number;
  memoryLimitBytes?: number;
  prefillChunkSize?: number;
  optimizationLevel?: WebGpuOptimizationLevel;
  gpuProfiling?: boolean;
  trackBufferAllocations?: boolean;
  segmentStartLayer: number;
  segmentEndLayerExclusive?: number;
  loadOutput?: boolean;
};

export type WebGpuStateLike = {
  contextLength: number;
  nextPosition: number;
  fullAttention?: InferenceState["fullAttention"];
};

export type WebGpuTokenResult = {
  selectedTokenId?: number;
  selectedTokenIds?: number[];
  topTokens?: WebGpuTopToken[];
  topTokensByPosition?: WebGpuTopToken[][];
};

export type WebGpuHiddenResult = {
  hidden: Float32Array;
  selectedTokenId?: number;
  selectedTokenIds?: number[];
  topTokens?: WebGpuTopToken[];
  topTokensByPosition?: WebGpuTopToken[][];
};

export type WebGpuRunOptions = {
  computeSelectedToken?: boolean;
  computeSelectedTokens?: boolean;
  computeTopK?: boolean;
  computeTopKTokens?: boolean;
  topK?: number;
  readAllHidden?: boolean;
  perLayerInputs?: Float32Array;
  perLayerInputsBuffer?: WebGpuBufferLike;
  attentionCausal?: boolean;
};

type WebGpuInternalRunOptions = WebGpuRunOptions & {
  sourceTokenCount: number;
  sourceTokenIndex: number;
  keyValueTokenCount?: number;
  skipKvUpdate?: boolean;
};

export type WebGpuRuntimeStats = {
  webgpuLazyLoadMs: number;
  webgpuRunnerCreateMs: number;
  webgpuRuntimeInitMs: number;
  webgpuRuntimeResizeMs: number;
  webgpuFirstRunTotalMs: number;
  webgpuSteadyRunMs: number;
  webgpuSteadyRunCount: number;
  webgpuResidentBytes: number;
  webgpuLastRunPeakResidentBytes: number;
  webgpuLastRunAttentionTempBytes: number;
  webgpuReadbackBytes: number;
  webgpuReadbackCount: number;
  webgpuSelectedTokenReadbacks: number;
  webgpuSubmitCount: number;
  webgpuBlockingWaitCount: number;
  webgpuDeferredCleanupCount: number;
  webgpuBoundaryUploads: number;
  webgpuTokenIdInputBatches: number;
  webgpuTokenIdInputTokens: number;
  webgpuInputPreparationSupported: boolean;
  webgpuInputPreparationUnsupportedReason: string;
  webgpuShaderModuleCacheHits: number;
  webgpuShaderModuleCacheMisses: number;
  webgpuBindGroupLayoutCacheHits: number;
  webgpuBindGroupLayoutCacheMisses: number;
  webgpuPipelineLayoutCacheHits: number;
  webgpuPipelineLayoutCacheMisses: number;
  webgpuComputePipelineCacheHits: number;
  webgpuComputePipelineCacheMisses: number;
  webgpuBindGroupCacheHits: number;
  webgpuBindGroupCacheMisses: number;
  webgpuBindGroupCreates: number;
  webgpuBufferCreates: number;
  webgpuLastRunDurationMs: number;
  webgpuLastRunSubmitCount: number;
  webgpuLastRunReadbackCount: number;
  webgpuLastRunBindGroupCreates: number;
  webgpuLastRunBindGroupCreateMs: number;
  webgpuLastRunBufferCreates: number;
  webgpuLastRunBufferCreateMs: number;
  webgpuLastRunBufferAllocationsByLabel: string;
  webgpuLastRunEncodeMs: number;
  webgpuLastRunSubmitMs: number;
  webgpuLastRunReadbackWaitMs: number;
  webgpuLastRunReadbackWaitMinusGpuPassMs: number;
  webgpuLastRunTimestampReadbackWaitMs: number;
  webgpuLastRunGpuPassMs: number;
  webgpuLastRunGpuSections: string;
  webgpuLastRunGpuTimingStatus: string;
  webgpuLastRunReadbackBytes: number;
  webgpuLastRunDispatchCount: number;
  webgpuLastRunSelectedTokenId: number;
  webgpuLastRunShaderModuleHits: number;
  webgpuLastRunShaderModuleMisses: number;
  webgpuLastRunBindGroupLayoutHits: number;
  webgpuLastRunBindGroupLayoutMisses: number;
  webgpuLastRunPipelineLayoutHits: number;
  webgpuLastRunPipelineLayoutMisses: number;
  webgpuLastRunComputePipelineHits: number;
  webgpuLastRunComputePipelineMisses: number;
  webgpuLastRunBindGroupHits: number;
  webgpuLastRunBindGroupMisses: number;
};

type FullAttentionGpuLayerState = {
  key: WebGpuBufferLike;
  value: WebGpuBufferLike;
  capacity: number;
  headCountKv: number;
};

type KvResizeResource = {
  pipeline: unknown;
  bindGroup: unknown;
  paramsBuffer: WebGpuBufferLike;
};

type GpuState = {
  fullAttention: Map<number, FullAttentionGpuLayerState>;
};

const F16_BYTE_LENGTH = 2;

type GpuInputResources = {
  tokenEmbedding?: F32Handle | QuantizedHandle;
  perLayerTokenEmbedding?: F32Handle | QuantizedHandle;
  perLayerModelProjection?: QuantizedHandle | F32Handle;
  perLayerProjectionNorm?: F32Handle;
};

type GpuEmbeddingRowChunk = {
  handle: F32Handle | QuantizedHandle;
  rowIds: Uint32Array;
  outputTokenOffset: number;
  tokenCount: number;
  transientHandle: boolean;
};

type PreparedGpuInput = {
  tokenCount: number;
  hidden: WebGpuBufferLike;
  perLayerInputs?: WebGpuBufferLike;
  destroy: () => void;
};

export type WebGpuPreparedInput = PreparedGpuInput;

type TimestampProfiler = {
  querySet: WebGpuQuerySetLike;
  resolveBuffer: WebGpuBufferLike;
  readbackBuffer: WebGpuBufferLike;
  maxPasses: number;
};

type TimestampPass = {
  queryIndex: number;
  destinationOffset: number;
  label: string;
};

type ActiveComputePass = {
  pass: WebGpuComputePassLike;
  timestampPass?: TimestampPass;
  profiling: boolean;
};

const TIMESTAMP_QUERY_PAIR_BYTES = 2 * BigUint64Array.BYTES_PER_ELEMENT;
const TIMESTAMP_RESOLVE_STRIDE_BYTES = 256;
const TIMESTAMP_MAX_PASSES = 256;
const WEBGPU_LAYER_LOAD_CONCURRENCY = 2;

export class WebGpuSegmentRunner implements SegmentRunner {
  readonly provider = "webgpu" as const;
  readonly segmentStartLayer: number;
  readonly segmentEndLayerExclusive: number;

  private readonly states = new WeakMap<object, GpuState>();
  private readonly arena: GpuMemoryArena;
  private readonly manifest: ModelManifest;
  private readonly epsilon: number;
  private readonly layers: GpuLayer[];
  private readonly outputNorm?: F32Handle;
  private readonly outputStripes?: OutputStripe[];
  private readonly ropeFreqFactors: F32Handle;
  private readonly hasRopeFreqFactors: boolean;
  private readonly tensorReader: GgufTensorReader;
  private readonly prefillChunkSize: number;
  private readonly optimizationLevel: WebGpuOptimizationLevel;
  private readonly gpuProfiling: boolean;
  private readonly trackBufferAllocations: boolean;
  private lazyLoadMs: number;
  private inputResourcesPromise?: Promise<GpuInputResources>;
  private readbackBytes = 0;
  private selectedTokenReadbacks = 0;
  private boundaryUploads = 0;
  private tokenIdInputBatches = 0;
  private tokenIdInputTokens = 0;
  private runtimeResourceCache?: WebGpuRuntimeResourceCache;
  private timestampProfiler?: TimestampProfiler;
  private timestampProfilerStatus = "not-requested";
  private runtimeResourcesInitialized = false;
  private runtimeInitMs = 0;
  private runtimeResizeMs = 0;
  private firstRunTotalMs = 0;
  private steadyRunMs = 0;
  private steadyRunCount = 0;
  private hasRecordedFirstRun = false;
  private submitCount = 0;
  private blockingWaitCount = 0;
  private readbackCount = 0;
  private deferredCleanupCount = 0;
  private lastRunStats = {
    durationMs: 0,
    submitCount: 0,
    readbackCount: 0,
    readbackBytes: 0,
    dispatchCount: 0,
    encodeMs: 0,
    submitMs: 0,
    readbackWaitMs: 0,
    timestampReadbackWaitMs: 0,
    gpuPassMs: 0,
    gpuSections: "",
    gpuTimingStatus: "disabled",
    selectedTokenId: -1,
    peakResidentBytes: 0,
    attentionTempBytes: 0,
    resourceStats: undefined as ReturnType<WebGpuRuntimeResourceCache["stats"]> | undefined,
  };
  private activeRunEncodeMs = 0;
  private activeRunSubmitMs = 0;
  private activeRunReadbackWaitMs = 0;
  private activeRunTimestampReadbackWaitMs = 0;
  private activeRunGpuPassMs = 0;
  private activeRunReadbackBytes = 0;
  private activeRunDispatchCount = 0;
  private activeRunAttentionTempBytes = 0;
  private activeRunTimestampPassCount = 0;
  private activeRunTimestampPassLabels: string[] = [];
  private activeRunGpuSections = "";
  private activeRunTimestampStatus = "not-requested";
  private activeRunSelectedTokenId = -1;

  private constructor(
    arena: GpuMemoryArena,
    manifest: ModelManifest,
    tensorReader: GgufTensorReader,
    epsilon: number,
    layers: GpuLayer[],
    ropeFreqFactors: F32Handle,
    hasRopeFreqFactors: boolean,
    outputNorm: F32Handle | undefined,
    outputStripes: OutputStripe[] | undefined,
    segmentStartLayer: number,
    segmentEndLayerExclusive: number,
    prefillChunkSize: number,
    optimizationLevel: WebGpuOptimizationLevel,
    gpuProfiling: boolean,
    trackBufferAllocations: boolean,
    lazyLoadMs: number,
  ) {
    this.arena = arena;
    this.manifest = manifest;
    this.tensorReader = tensorReader;
    this.epsilon = epsilon;
    this.layers = layers;
    this.ropeFreqFactors = ropeFreqFactors;
    this.hasRopeFreqFactors = hasRopeFreqFactors;
    this.outputNorm = outputNorm;
    this.outputStripes = outputStripes;
    this.segmentStartLayer = segmentStartLayer;
    this.segmentEndLayerExclusive = segmentEndLayerExclusive;
    this.prefillChunkSize = prefillChunkSize;
    this.optimizationLevel = optimizationLevel;
    this.gpuProfiling = gpuProfiling;
    this.trackBufferAllocations = trackBufferAllocations;
    this.lazyLoadMs = lazyLoadMs;
  }

  static async create(options: WebGpuSegmentRunnerOptions): Promise<WebGpuSegmentRunner> {
    const startMs = nowMs();
    const optimizationLevel = normalizeOptimizationLevel(options.optimizationLevel);
    const gpuProfiling = options.gpuProfiling === true;
    const trackBufferAllocations = options.trackBufferAllocations === true;
    const device = await webGpuDevice({ gpuProfiling });
    if (!device) {
      throw new Error("WebGPU is not available for  segment execution.");
    }
    const segmentStartLayer = options.segmentStartLayer;
    const segmentEndLayerExclusive = options.segmentEndLayerExclusive ?? options.manifest.blockCount;
    if (
      !Number.isInteger(segmentStartLayer) ||
      !Number.isInteger(segmentEndLayerExclusive) ||
      segmentStartLayer < 0 ||
      segmentEndLayerExclusive <= segmentStartLayer ||
      segmentEndLayerExclusive > options.manifest.blockCount
    ) {
      throw new Error(`Invalid WebGPU layer segment: ${segmentStartLayer}..${segmentEndLayerExclusive}`);
    }

    for (let layer = segmentStartLayer; layer < segmentEndLayerExclusive; layer += 1) {
      const source = options.manifest.kvSourceLayers[layer] ?? layer;
      if (options.manifest.layerHasKv[layer] !== true && source < segmentStartLayer) {
        throw new Error(
          `WebGPU layer ${layer} reuses KV from layer ${source}, which is outside the WebGPU segment.`,
        );
      }
    }

    const arena = new GpuMemoryArena(device, options.memoryLimitBytes ?? WEBGPU_MEMORY_LIMIT_BYTES);
    const layerIndexes = Array.from(
      { length: segmentEndLayerExclusive - segmentStartLayer },
      (_, index) => segmentStartLayer + index,
    );
    const layers = await mapWithConcurrency(
      layerIndexes,
      WEBGPU_LAYER_LOAD_CONCURRENCY,
      (layer) => loadGpuLayer(arena, options.tensorReader, options.manifest, layer),
    );
    const { handle: ropeFreqFactors, present: hasRopeFreqFactors } = await loadRopeFreqFactors(arena, options.tensorReader);
    const loadOutput = options.loadOutput ?? true;
    const outputNorm = loadOutput
      ? await loadF32Handle(arena, options.tensorReader, "output_norm.weight")
      : undefined;
    const outputStripes = loadOutput
      ? await loadOutputStripes(arena, options.tensorReader, options.manifest)
      : undefined;

    return new WebGpuSegmentRunner(
      arena,
      options.manifest,
      options.tensorReader,
      options.epsilon,
      layers,
      ropeFreqFactors,
      hasRopeFreqFactors,
      outputNorm,
      outputStripes,
      segmentStartLayer,
      segmentEndLayerExclusive,
      normalizePrefillChunkSize(options.prefillChunkSize),
      optimizationLevel,
      gpuProfiling,
      trackBufferAllocations,
      nowMs() - startMs,
    );
  }

  get residentBytes(): number {
    return this.arena.residentBytes;
  }

  runtimeStats(): WebGpuRuntimeStats {
    const resourceStats = this.runtimeResourceCache?.stats();
    return {
      webgpuLazyLoadMs: this.lazyLoadMs,
      webgpuRunnerCreateMs: this.lazyLoadMs,
      webgpuRuntimeInitMs: this.runtimeInitMs,
      webgpuRuntimeResizeMs: this.runtimeResizeMs,
      webgpuFirstRunTotalMs: this.firstRunTotalMs,
      webgpuSteadyRunMs: this.steadyRunMs,
      webgpuSteadyRunCount: this.steadyRunCount,
      webgpuResidentBytes: this.residentBytes,
      webgpuLastRunPeakResidentBytes: this.lastRunStats.peakResidentBytes,
      webgpuLastRunAttentionTempBytes: this.lastRunStats.attentionTempBytes,
      webgpuReadbackBytes: this.readbackBytes,
      webgpuReadbackCount: this.readbackCount,
      webgpuSelectedTokenReadbacks: this.selectedTokenReadbacks,
      webgpuSubmitCount: this.submitCount,
      webgpuBlockingWaitCount: this.blockingWaitCount,
      webgpuDeferredCleanupCount: this.deferredCleanupCount,
      webgpuBoundaryUploads: this.boundaryUploads,
      webgpuTokenIdInputBatches: this.tokenIdInputBatches,
      webgpuTokenIdInputTokens: this.tokenIdInputTokens,
      webgpuInputPreparationSupported: this.gpuInputPreparationSupport().supported,
      webgpuInputPreparationUnsupportedReason: this.gpuInputPreparationSupport().reason,
      webgpuShaderModuleCacheHits: resourceStats?.shaderModuleHits ?? 0,
      webgpuShaderModuleCacheMisses: resourceStats?.shaderModuleMisses ?? 0,
      webgpuBindGroupLayoutCacheHits: resourceStats?.bindGroupLayoutHits ?? 0,
      webgpuBindGroupLayoutCacheMisses: resourceStats?.bindGroupLayoutMisses ?? 0,
      webgpuPipelineLayoutCacheHits: resourceStats?.pipelineLayoutHits ?? 0,
      webgpuPipelineLayoutCacheMisses: resourceStats?.pipelineLayoutMisses ?? 0,
      webgpuComputePipelineCacheHits: resourceStats?.computePipelineHits ?? 0,
      webgpuComputePipelineCacheMisses: resourceStats?.computePipelineMisses ?? 0,
      webgpuBindGroupCacheHits: resourceStats?.bindGroupHits ?? 0,
      webgpuBindGroupCacheMisses: resourceStats?.bindGroupMisses ?? 0,
      webgpuBindGroupCreates: resourceStats?.bindGroupCreates ?? 0,
      webgpuBufferCreates: resourceStats?.bufferCreates ?? 0,
      webgpuLastRunDurationMs: this.lastRunStats.durationMs,
      webgpuLastRunSubmitCount: this.lastRunStats.submitCount,
      webgpuLastRunReadbackCount: this.lastRunStats.readbackCount,
      webgpuLastRunBindGroupCreates: this.lastRunStats.resourceStats?.bindGroupCreates ?? 0,
      webgpuLastRunBindGroupCreateMs: this.lastRunStats.resourceStats?.bindGroupCreateMs ?? 0,
      webgpuLastRunBufferCreates: this.lastRunStats.resourceStats?.bufferCreates ?? 0,
      webgpuLastRunBufferCreateMs: this.lastRunStats.resourceStats?.bufferCreateMs ?? 0,
      webgpuLastRunBufferAllocationsByLabel: this.lastRunStats.resourceStats?.bufferAllocationsByLabel ?? "",
      webgpuLastRunEncodeMs: this.lastRunStats.encodeMs,
      webgpuLastRunSubmitMs: this.lastRunStats.submitMs,
      webgpuLastRunReadbackWaitMs: this.lastRunStats.readbackWaitMs,
      webgpuLastRunReadbackWaitMinusGpuPassMs: Math.max(0, this.lastRunStats.readbackWaitMs - this.lastRunStats.gpuPassMs),
      webgpuLastRunTimestampReadbackWaitMs: this.lastRunStats.timestampReadbackWaitMs,
      webgpuLastRunGpuPassMs: this.lastRunStats.gpuPassMs,
      webgpuLastRunGpuSections: this.lastRunStats.gpuSections,
      webgpuLastRunGpuTimingStatus: this.lastRunStats.gpuTimingStatus,
      webgpuLastRunReadbackBytes: this.lastRunStats.readbackBytes,
      webgpuLastRunDispatchCount: this.lastRunStats.dispatchCount,
      webgpuLastRunSelectedTokenId: this.lastRunStats.selectedTokenId,
      webgpuLastRunShaderModuleHits: this.lastRunStats.resourceStats?.shaderModuleHits ?? 0,
      webgpuLastRunShaderModuleMisses: this.lastRunStats.resourceStats?.shaderModuleMisses ?? 0,
      webgpuLastRunBindGroupLayoutHits: this.lastRunStats.resourceStats?.bindGroupLayoutHits ?? 0,
      webgpuLastRunBindGroupLayoutMisses: this.lastRunStats.resourceStats?.bindGroupLayoutMisses ?? 0,
      webgpuLastRunPipelineLayoutHits: this.lastRunStats.resourceStats?.pipelineLayoutHits ?? 0,
      webgpuLastRunPipelineLayoutMisses: this.lastRunStats.resourceStats?.pipelineLayoutMisses ?? 0,
      webgpuLastRunComputePipelineHits: this.lastRunStats.resourceStats?.computePipelineHits ?? 0,
      webgpuLastRunComputePipelineMisses: this.lastRunStats.resourceStats?.computePipelineMisses ?? 0,
      webgpuLastRunBindGroupHits: this.lastRunStats.resourceStats?.bindGroupHits ?? 0,
      webgpuLastRunBindGroupMisses: this.lastRunStats.resourceStats?.bindGroupMisses ?? 0,
    };
  }

  ensureRuntimeResources(): void {
    if (this.runtimeResourcesInitialized) {
      return;
    }
    const startMs = nowMs();
    const { device, cache } = installWebGpuRuntimeResourceCache(this.arena.device, {
      trackBufferAllocations: this.trackBufferAllocations,
    });
    this.arena.device = device;
    this.runtimeResourceCache = cache;
    this.runtimeInitMs += nowMs() - startMs;
    this.runtimeResourcesInitialized = true;
  }

  private beginRuntimeRun(): {
    startMs: number;
    firstRun: boolean;
    submitCount: number;
    readbackCount: number;
    resourceStats?: ReturnType<WebGpuRuntimeResourceCache["stats"]>;
  } {
    this.arena.destroyScratchBuffers();
    this.arena.resetPeakResidentBytes();
    this.activeRunEncodeMs = 0;
    this.activeRunSubmitMs = 0;
    this.activeRunReadbackWaitMs = 0;
    this.activeRunTimestampReadbackWaitMs = 0;
    this.activeRunGpuPassMs = 0;
    this.activeRunReadbackBytes = 0;
    this.activeRunDispatchCount = 0;
    this.activeRunAttentionTempBytes = 0;
    this.activeRunTimestampPassCount = 0;
    this.activeRunTimestampPassLabels = [];
    this.activeRunGpuSections = "";
    this.activeRunTimestampStatus = this.gpuProfiling ? "not-requested" : "disabled";
    this.activeRunSelectedTokenId = -1;
    return {
      startMs: nowMs(),
      firstRun: !this.hasRecordedFirstRun,
      submitCount: this.submitCount,
      readbackCount: this.readbackCount,
      resourceStats: this.runtimeResourceCache?.stats(),
    };
  }

  private endRuntimeRun(run: {
    startMs: number;
    firstRun: boolean;
    submitCount: number;
    readbackCount: number;
    resourceStats?: ReturnType<WebGpuRuntimeResourceCache["stats"]>;
  }): void {
    const durationMs = nowMs() - run.startMs;
    const after = this.runtimeResourceCache?.stats();
    const resourceDelta = after && run.resourceStats
      ? diffWebGpuRuntimeResourceStats(after, run.resourceStats)
      : undefined;
    this.lastRunStats = {
      durationMs,
      submitCount: this.submitCount - run.submitCount,
      readbackCount: this.readbackCount - run.readbackCount,
      readbackBytes: this.activeRunReadbackBytes,
      dispatchCount: this.activeRunDispatchCount,
      encodeMs: this.activeRunEncodeMs,
      submitMs: this.activeRunSubmitMs,
      readbackWaitMs: this.activeRunReadbackWaitMs,
      timestampReadbackWaitMs: this.activeRunTimestampReadbackWaitMs,
      gpuPassMs: this.activeRunGpuPassMs,
      gpuSections: this.activeRunGpuSections,
      gpuTimingStatus: this.activeRunTimestampStatus,
      selectedTokenId: this.activeRunSelectedTokenId,
      peakResidentBytes: this.arena.peakResidentBytes,
      attentionTempBytes: this.activeRunAttentionTempBytes,
      resourceStats: resourceDelta,
    };
    if (run.firstRun) {
      if (resourceDelta) {
        this.runtimeInitMs += runtimeResourceCreateMs(resourceDelta);
      }
      this.firstRunTotalMs = durationMs;
      this.hasRecordedFirstRun = true;
      this.arena.destroyScratchBuffers();
      return;
    }
    this.steadyRunMs = durationMs;
    this.steadyRunCount += 1;
    this.arena.destroyScratchBuffers();
  }

  private submitCommandBuffer(commandBuffer: unknown): void {
    const startMs = nowMs();
    try {
      this.arena.device.queue.submit([commandBuffer]);
      this.submitCount += 1;
    } finally {
      this.activeRunSubmitMs += nowMs() - startMs;
    }
  }

  private async waitForSubmittedWorkDone(): Promise<void> {
    const done = this.arena.device.queue.onSubmittedWorkDone?.();
    if (done) {
      await done;
    }
  }

  private async mapReadback(buffer: WebGpuBufferLike): Promise<void> {
    const startMs = nowMs();
    try {
      await buffer.mapAsync(GPU_MAP_READ);
    } finally {
      this.activeRunReadbackWaitMs += nowMs() - startMs;
    }
  }

  private countedPass(pass: WebGpuComputePassLike): WebGpuComputePassLike {
    return {
      setPipeline: (pipeline) => pass.setPipeline(pipeline),
      setBindGroup: (index, bindGroup) => pass.setBindGroup(index, bindGroup),
      dispatchWorkgroups: (x, y, z) => {
        this.activeRunDispatchCount += 1;
        pass.dispatchWorkgroups(x, y, z);
      },
      end: () => pass.end(),
    };
  }

  private dispatchRmsNormToQ8K(
    pass: WebGpuComputePassLike,
    resources: Array<{ destroy: () => void }>,
    input: WebGpuBufferLike,
    weight: WebGpuBufferLike,
    q8: ReturnType<typeof scratchQ8K>,
    length: number,
    cleanup: GpuResource[],
    label: string,
  ): void {
    if (this.usesStandardOptimizations()) {
      dispatchRmsNormQ8KQuantize(this.arena.device, pass, resources, input, weight, q8, length, this.epsilon);
      return;
    }
    const normalized = scratchF32(this.arena, length, cleanup, label);
    dispatchRmsNorm(this.arena.device, pass, resources, input, weight, normalized, length, this.epsilon);
    dispatchQ8KQuantize(this.arena.device, pass, resources, normalized, q8, length, 1);
  }

  private dispatchRmsNormToQ8_0(
    pass: WebGpuComputePassLike,
    resources: Array<{ destroy: () => void }>,
    input: WebGpuBufferLike,
    weight: WebGpuBufferLike,
    q8: ReturnType<typeof scratchQ8_0>,
    length: number,
  ): void {
    dispatchBatchedRmsNormQ8_0Quantize(this.arena.device, pass, resources, input, weight, q8, {
      length,
      tokenCount: 1,
      epsilon: this.epsilon,
    });
  }

  private dispatchRmsNormThenResidualAdd(
    pass: WebGpuComputePassLike,
    resources: Array<{ destroy: () => void }>,
    input: WebGpuBufferLike,
    weight: WebGpuBufferLike,
    residual: WebGpuBufferLike,
    output: WebGpuBufferLike,
    length: number,
    cleanup: GpuResource[],
    label: string,
  ): void {
    if (this.usesStandardOptimizations()) {
      dispatchRmsNormResidualAdd(this.arena.device, pass, resources, input, weight, residual, output, length, this.epsilon);
      return;
    }
    const normalized = scratchF32(this.arena, length, cleanup, label);
    dispatchRmsNorm(this.arena.device, pass, resources, input, weight, normalized, length, this.epsilon);
    dispatchResidualAdd(this.arena.device, pass, resources, normalized, residual, output, length);
  }

  private dispatchRmsNormThenResidualAddScale(
    pass: WebGpuComputePassLike,
    resources: Array<{ destroy: () => void }>,
    input: WebGpuBufferLike,
    weight: WebGpuBufferLike,
    residual: WebGpuBufferLike,
    scale: WebGpuBufferLike,
    output: WebGpuBufferLike,
    length: number,
    cleanup: GpuResource[],
    label: string,
  ): void {
    if (this.usesStandardOptimizations()) {
      dispatchRmsNormResidualAddScale(this.arena.device, pass, resources, input, weight, residual, scale, output, length, this.epsilon);
      return;
    }
    const normalized = scratchF32(this.arena, length, cleanup, label);
    const residualAdded = scratchF32(this.arena, length, cleanup, `${label}.residual`);
    dispatchRmsNorm(this.arena.device, pass, resources, input, weight, normalized, length, this.epsilon);
    dispatchResidualAdd(this.arena.device, pass, resources, normalized, residual, residualAdded, length);
    dispatchScale(this.arena.device, pass, resources, residualAdded, scale, output, length);
  }

  private beginComputePass(
    encoder: WebGpuCommandEncoderLike,
    label = "gpu",
  ): ActiveComputePass {
    const timestampPass = this.gpuProfiling ? this.allocateTimestampPass(label) : undefined;
    if (this.gpuProfiling && !timestampPass) {
      throw new Error(`WebGPU GPU profiling is unavailable: ${this.activeRunTimestampStatus}`);
    }
    const descriptor = timestampPass
      ? {
          timestampWrites: {
            querySet: this.timestampProfiler?.querySet,
            beginningOfPassWriteIndex: timestampPass.queryIndex,
            endOfPassWriteIndex: timestampPass.queryIndex + 1,
          },
        }
      : undefined;
    return {
      pass: this.countedPass(encoder.beginComputePass(descriptor)),
      timestampPass,
      profiling: timestampPass !== undefined,
    };
  }

  private endComputePass(encoder: WebGpuCommandEncoderLike, computePass: ActiveComputePass): void {
    computePass.pass.end();
    this.finishTimestampPass(encoder, computePass.timestampPass);
  }

  private restartComputePass(
    encoder: WebGpuCommandEncoderLike,
    computePass: ActiveComputePass,
    label: string,
  ): ActiveComputePass {
    if (!computePass.profiling) {
      return computePass;
    }
    this.endComputePass(encoder, computePass);
    return this.beginComputePass(encoder, label);
  }

  private finishTimestampPass(encoder: WebGpuCommandEncoderLike, timestampPass: TimestampPass | undefined): void {
    if (!timestampPass || !encoder.resolveQuerySet || !this.timestampProfiler) {
      return;
    }
    encoder.resolveQuerySet(
      this.timestampProfiler.querySet,
      timestampPass.queryIndex,
      2,
      this.timestampProfiler.resolveBuffer,
      timestampPass.destinationOffset,
    );
    encoder.copyBufferToBuffer(
      this.timestampProfiler.resolveBuffer,
      timestampPass.destinationOffset,
      this.timestampProfiler.readbackBuffer,
      timestampPass.destinationOffset,
      TIMESTAMP_QUERY_PAIR_BYTES,
    );
  }

  private allocateTimestampPass(label: string): TimestampPass | undefined {
    const profiler = this.ensureTimestampProfiler();
    if (!profiler) {
      this.activeRunTimestampStatus = this.timestampProfilerStatus;
      return undefined;
    }
    if (this.activeRunTimestampPassCount >= profiler.maxPasses) {
      this.activeRunTimestampStatus = "max-pass-count-exceeded";
      return undefined;
    }
    const passIndex = this.activeRunTimestampPassCount;
    this.activeRunTimestampPassCount += 1;
    this.activeRunTimestampPassLabels[passIndex] = label;
    this.activeRunTimestampStatus = "pending";
    return {
      queryIndex: passIndex * 2,
      destinationOffset: passIndex * TIMESTAMP_RESOLVE_STRIDE_BYTES,
      label,
    };
  }

  private ensureTimestampProfiler(): TimestampProfiler | undefined {
    if (this.timestampProfiler) {
      return this.timestampProfiler;
    }
    const device = this.arena.device;
    if (!device.features?.has("timestamp-query")) {
      this.timestampProfilerStatus = "timestamp-query-unavailable";
      return undefined;
    }
    if (!device.createQuerySet) {
      this.timestampProfilerStatus = "create-query-set-unavailable";
      return undefined;
    }
    const encoder = device.createCommandEncoder();
    if (!encoder.resolveQuerySet) {
      this.timestampProfilerStatus = "resolve-query-set-unavailable";
      return undefined;
    }
    const maxPasses = Math.max(TIMESTAMP_MAX_PASSES, this.layers.length * 24 + 64);
    const byteLength = maxPasses * TIMESTAMP_RESOLVE_STRIDE_BYTES;
    let querySet: WebGpuQuerySetLike | undefined;
    let resolveBuffer: WebGpuBufferLike | undefined;
    let readbackBuffer: WebGpuBufferLike | undefined;
    try {
      querySet = device.createQuerySet({ type: "timestamp", count: maxPasses * 2 });
      resolveBuffer = device.createBuffer({
        size: byteLength,
        usage: GPU_QUERY_RESOLVE | GPU_COPY_SRC,
      });
      readbackBuffer = device.createBuffer({
        size: byteLength,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
      this.timestampProfiler = {
        querySet,
        resolveBuffer,
        readbackBuffer,
        maxPasses,
      };
    } catch {
      querySet?.destroy?.();
      resolveBuffer?.destroy?.();
      readbackBuffer?.destroy?.();
      this.timestampProfilerStatus = "timestamp-profiler-create-failed";
      return undefined;
    }
    this.timestampProfilerStatus = "available";
    return this.timestampProfiler;
  }

  private async readTimestampProfiler(): Promise<void> {
    const profiler = this.timestampProfiler;
    if (!profiler || this.activeRunTimestampPassCount === 0) {
      return;
    }
    const startMs = nowMs();
    try {
      await profiler.readbackBuffer.mapAsync(GPU_MAP_READ);
    } catch (error) {
      this.activeRunTimestampStatus = "timestamp-readback-failed";
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`WebGPU GPU profiling timestamp readback failed: ${message}`);
    } finally {
      this.activeRunTimestampReadbackWaitMs += nowMs() - startMs;
    }
    try {
      const mapped = profiler.readbackBuffer.getMappedRange();
      let totalNs = 0n;
      const sectionMs = new Map<string, number>();
      for (let index = 0; index < this.activeRunTimestampPassCount; index += 1) {
        const timestamps = new BigUint64Array(mapped, index * TIMESTAMP_RESOLVE_STRIDE_BYTES, 2);
        const passStart = timestamps[0] ?? 0n;
        const passEnd = timestamps[1] ?? passStart;
        if (passEnd >= passStart) {
          const passNs = passEnd - passStart;
          const passMs = Number(passNs) / 1_000_000;
          const label = this.activeRunTimestampPassLabels[index] ?? `pass.${index}`;
          totalNs += passNs;
          sectionMs.set(label, (sectionMs.get(label) ?? 0) + passMs);
        }
      }
      this.activeRunGpuPassMs = Number(totalNs) / 1_000_000;
      this.activeRunGpuSections = formatGpuSectionMs(sectionMs);
      this.activeRunTimestampStatus = this.activeRunTimestampStatus === "max-pass-count-exceeded"
        ? "partial"
        : "ok";
    } finally {
      profiler.readbackBuffer.unmap();
    }
  }

  private deferResourceCleanup(...groups: ReadonlyArray<ReadonlyArray<GpuResource>>): void {
    const items = groups.flat().filter((item) => item.destroy);
    if (items.length === 0) {
      return;
    }
    this.deferredCleanupCount += 1;
    const destroyItems = () => {
      for (let index = items.length - 1; index >= 0; index -= 1) {
        items[index]?.destroy?.();
      }
    };
    const done = this.arena.device.queue.onSubmittedWorkDone?.();
    if (!done) {
      destroyItems();
      return;
    }
    void done.then(destroyItems, destroyItems);
  }

  supportsGpuInputPreparation(): boolean {
    return this.gpuInputPreparationSupport().supported;
  }

  supportsTokenIdInput(): boolean {
    return this.supportsGpuInputPreparation();
  }

  private usesStandardOptimizations(): boolean {
    return this.optimizationLevel === "standard";
  }

  async prepareTokenIds(tokenIds: readonly number[]): Promise<WebGpuPreparedInput> {
    this.ensureRuntimeResources();
    const runtimeRun = this.beginRuntimeRun();
    try {
      const prepared = await this.prepareGpuInput(tokenIds);
      this.tokenIdInputBatches += 1;
      this.tokenIdInputTokens += tokenIds.length;
      return prepared;
    } finally {
      this.endRuntimeRun(runtimeRun);
    }
  }

  async readPreparedInputHidden(input: WebGpuPreparedInput): Promise<Float32Array> {
    const byteLength = input.tokenCount * this.manifest.embeddingLength * Float32Array.BYTES_PER_ELEMENT;
    const readback = this.arena.device.createBuffer({
      label: "input.hidden.readback",
      size: byteLength,
      usage: GPU_COPY_DST | GPU_MAP_READ,
    });
    try {
      const encoder = this.arena.device.createCommandEncoder();
      encoder.copyBufferToBuffer(input.hidden, 0, readback, 0, byteLength);
      this.submitCommandBuffer(encoder.finish());
      this.readbackCount += 1;
      await this.mapReadback(readback);
      const hidden = new Float32Array(readback.getMappedRange()).slice();
      readback.unmap();
      this.readbackBytes += byteLength;
      return hidden;
    } finally {
      readback.destroy?.();
    }
  }

  async readMtpTargetKvView(
    state: WebGpuStateLike,
    layerIndexes: readonly number[],
    tokenCount: number,
  ): Promise<MtpTargetKvView> {
    if (!Number.isInteger(tokenCount) || tokenCount < 0 || tokenCount > state.contextLength) {
      throw new Error(`Invalid MTP KV token count ${tokenCount} for context length ${state.contextLength}.`);
    }
    const gpuState = this.ensureGpuState(state);
    const layers: MtpTargetKvLayerView[] = [];
    for (const layerIndex of layerIndexes) {
      const layer = this.layers.find((candidate) => candidate.layer === layerIndex);
      if (!layer) {
        throw new Error(`WebGPU MTP KV readback requires loaded target layer ${layerIndex}.`);
      }
      const layerState = gpuState.fullAttention.get(layerIndex);
      if (!layerState) {
        throw new Error(`Missing WebGPU MTP KV state for target layer ${layerIndex}.`);
      }
      layers.push(await this.readMtpTargetKvLayer(layer, layerState, tokenCount));
    }
    return { layers };
  }

  async runPreparedInputHidden(
    input: WebGpuPreparedInput,
    positions: Int32Array,
    state: WebGpuStateLike,
    options: WebGpuRunOptions = {},
  ): Promise<WebGpuHiddenResult> {
    this.ensureRuntimeResources();
    const runtimeRun = this.beginRuntimeRun();
    try {
      if (input.tokenCount > 1) {
        return await this.runBatchedPrefillFromBoundary(input.hidden, positions, state, {
          ...options,
          perLayerInputsBuffer: input.perLayerInputs,
          sourceTokenCount: input.tokenCount,
          sourceTokenIndex: 0,
        }, true);
      }
      return await this.runTokenFromBoundaryHidden(input.hidden, 0, tokenPositionsFromBatch(positions, 0, 1), state, {
        ...options,
        perLayerInputsBuffer: input.perLayerInputs,
        sourceTokenCount: 1,
        sourceTokenIndex: 0,
      });
    } finally {
      this.endRuntimeRun(runtimeRun);
    }
  }

  async runPreparedInput(
    input: WebGpuPreparedInput,
    positions: Int32Array,
    state: WebGpuStateLike,
    options: WebGpuRunOptions = {},
  ): Promise<WebGpuTokenResult> {
    this.ensureRuntimeResources();
    const runtimeRun = this.beginRuntimeRun();
    try {
      if (input.tokenCount > 1) {
        return await this.runBatchedPrefillFromBoundary(input.hidden, positions, state, {
          ...options,
          perLayerInputsBuffer: input.perLayerInputs,
          sourceTokenCount: input.tokenCount,
          sourceTokenIndex: 0,
        }, false);
      }
      return await this.runTokenFromBoundary(input.hidden, 0, tokenPositionsFromBatch(positions, 0, 1), state, {
        ...options,
        perLayerInputsBuffer: input.perLayerInputs,
        sourceTokenCount: 1,
        sourceTokenIndex: 0,
      });
    } finally {
      this.endRuntimeRun(runtimeRun);
    }
  }

  private gpuInputPreparationSupport(): { supported: boolean; reason: string } {
    if (this.segmentStartLayer !== 0) {
      return { supported: false, reason: `segment starts at layer ${this.segmentStartLayer}` };
    }
    const tokenEmbeddingType = this.tensorReader.getTensor("token_embd.weight").type;
    if (!isSupportedEmbeddingGatherType(tokenEmbeddingType)) {
      return { supported: false, reason: `token_embd.weight type ${tokenEmbeddingType}` };
    }
    if (this.manifest.perLayerEmbeddingLength <= 0) {
      return { supported: true, reason: "supported" };
    }
    const perLayerToken = this.tensorReader.getTensor("per_layer_token_embd.weight");
    const modelProjection = this.tensorReader.getTensor("per_layer_model_proj.weight");
    const projectionNorm = this.tensorReader.getTensor("per_layer_proj_norm.weight");
    if (!isSupportedEmbeddingGatherType(perLayerToken.type)) {
      return { supported: false, reason: `per_layer_token_embd.weight type ${perLayerToken.type}` };
    }
    if (projectionNorm.type !== "F32") {
      return { supported: false, reason: `per_layer_proj_norm.weight type ${projectionNorm.type}` };
    }
    if (!isSupportedProjectionType(modelProjection.type)) {
      return { supported: false, reason: `per_layer_model_proj.weight type ${modelProjection.type}` };
    }
    return { supported: true, reason: "supported" };
  }

  async runTokenIds(
    tokenIds: readonly number[],
    positions: Int32Array,
    state: WebGpuStateLike,
    options: WebGpuRunOptions = {},
  ): Promise<WebGpuTokenResult> {
    this.ensureRuntimeResources();
    const runtimeRun = this.beginRuntimeRun();
    try {
      const tokenCount = this.assertTokenIdBatch(tokenIds, positions);
      const prepared = await this.prepareGpuInput(tokenIds);
      this.tokenIdInputBatches += 1;
      this.tokenIdInputTokens += tokenCount;
      if (tokenCount > 1) {
        try {
          return await this.runBatchedPrefillFromBoundary(prepared.hidden, positions, state, {
            ...options,
            perLayerInputsBuffer: prepared.perLayerInputs,
            sourceTokenCount: tokenCount,
            sourceTokenIndex: 0,
          }, false);
        } finally {
          if (options.computeTopK === true || options.computeSelectedToken === true) {
            prepared.destroy();
          } else {
            this.deferResourceCleanup([{ destroy: prepared.destroy }]);
          }
        }
      }
      let topTokens: WebGpuTopToken[] | undefined;
      let selectedTokenId: number | undefined;
      try {
        if (tokenCount === 1) {
          const result = await this.runTokenFromBoundary(prepared.hidden, 0, tokenPositionsFromBatch(positions, 0, 1), state, {
            ...options,
            perLayerInputsBuffer: prepared.perLayerInputs,
            sourceTokenCount: 1,
            sourceTokenIndex: 0,
          });
          selectedTokenId = result.selectedTokenId;
          topTokens = result.topTokens;
        }
      } finally {
        if (options.computeTopK === true || options.computeSelectedToken === true) {
          prepared.destroy();
        } else {
          this.deferResourceCleanup([{ destroy: prepared.destroy }]);
        }
      }
      return { selectedTokenId, topTokens };
    } finally {
      this.endRuntimeRun(runtimeRun);
    }
  }

  async runToken(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: WebGpuStateLike,
    options: WebGpuRunOptions = {},
  ): Promise<WebGpuTokenResult> {
    this.ensureRuntimeResources();
    const runtimeRun = this.beginRuntimeRun();
    try {
      if (inputHidden.length !== this.manifest.embeddingLength) {
        throw new Error(`WebGPU segment input shape mismatch: ${inputHidden.length}`);
      }
      const boundary = this.arena.createBuffer(
        "segment boundary hidden",
        inputHidden.byteLength,
        GPU_STORAGE | GPU_COPY_DST,
      );
      this.arena.device.queue.writeBuffer(boundary, 0, inputHidden);
      this.boundaryUploads += 1;
      try {
        return await this.runTokenFromBoundary(boundary, 0, positions, state, {
          ...options,
          sourceTokenCount: 1,
          sourceTokenIndex: 0,
        });
      } finally {
        if (options.computeTopK === true || options.computeSelectedToken === true) {
          boundary.destroy?.();
        } else {
          this.deferResourceCleanup([boundary]);
        }
      }
    } finally {
      this.endRuntimeRun(runtimeRun);
    }
  }

  async runTokenHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: WebGpuStateLike,
    options: WebGpuRunOptions = {},
  ): Promise<WebGpuHiddenResult> {
    this.ensureRuntimeResources();
    const runtimeRun = this.beginRuntimeRun();
    try {
      if (inputHidden.length !== this.manifest.embeddingLength) {
        throw new Error(`WebGPU segment input shape mismatch: ${inputHidden.length}`);
      }
      const boundary = this.arena.createBuffer(
        "segment boundary hidden",
        inputHidden.byteLength,
        GPU_STORAGE | GPU_COPY_DST,
      );
      this.arena.device.queue.writeBuffer(boundary, 0, inputHidden);
      this.boundaryUploads += 1;
      try {
        return await this.runTokenFromBoundaryHidden(boundary, 0, positions, state, {
          ...options,
          sourceTokenCount: 1,
          sourceTokenIndex: 0,
        });
      } finally {
        boundary.destroy?.();
      }
    } finally {
      this.endRuntimeRun(runtimeRun);
    }
  }

  async runTokens(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: WebGpuStateLike,
    options: WebGpuRunOptions = {},
  ): Promise<WebGpuTokenResult> {
    this.ensureRuntimeResources();
    const runtimeRun = this.beginRuntimeRun();
    try {
      const tokenCount = this.assertBatchedHidden(inputHidden, positions);
      const boundary = this.arena.createBuffer(
        "segment boundary hidden batch",
        inputHidden.byteLength,
        GPU_STORAGE | GPU_COPY_DST | GPU_COPY_SRC,
      );
      this.arena.device.queue.writeBuffer(boundary, 0, inputHidden);
      this.boundaryUploads += 1;
      if (tokenCount > 1) {
        try {
          return await this.runBatchedPrefillFromBoundary(boundary, positions, state, {
            ...options,
            sourceTokenCount: tokenCount,
            sourceTokenIndex: 0,
          }, false);
        } finally {
          if (options.computeTopK === true || options.computeSelectedToken === true) {
            boundary.destroy?.();
          } else {
            this.deferResourceCleanup([boundary]);
          }
        }
      }
      let topTokens: WebGpuTopToken[] | undefined;
      let selectedTokenId: number | undefined;
      try {
        if (tokenCount === 1) {
          const result = await this.runTokenFromBoundary(boundary, 0, tokenPositionsFromBatch(positions, 0, 1), state, {
            ...options,
            sourceTokenCount: 1,
            sourceTokenIndex: 0,
          });
          selectedTokenId = result.selectedTokenId;
          topTokens = result.topTokens;
        }
      } finally {
        if (options.computeTopK === true || options.computeSelectedToken === true) {
          boundary.destroy?.();
        } else {
          this.deferResourceCleanup([boundary]);
        }
      }
      return { selectedTokenId, topTokens };
    } finally {
      this.endRuntimeRun(runtimeRun);
    }
  }

  async runTokensHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: WebGpuStateLike,
    options: WebGpuRunOptions = {},
  ): Promise<WebGpuHiddenResult> {
    this.ensureRuntimeResources();
    const runtimeRun = this.beginRuntimeRun();
    try {
      const tokenCount = this.assertBatchedHidden(inputHidden, positions);
      const boundary = this.arena.createBuffer(
        "segment boundary hidden batch",
        inputHidden.byteLength,
        GPU_STORAGE | GPU_COPY_DST | GPU_COPY_SRC,
      );
      this.arena.device.queue.writeBuffer(boundary, 0, inputHidden);
      this.boundaryUploads += 1;
      if (tokenCount > 1) {
        try {
          return await this.runBatchedPrefillFromBoundary(boundary, positions, state, {
            ...options,
            sourceTokenCount: tokenCount,
            sourceTokenIndex: 0,
          }, true);
        } finally {
          boundary.destroy?.();
        }
      }
      try {
        return await this.runTokenFromBoundaryHidden(boundary, 0, tokenPositionsFromBatch(positions, 0, 1), state, {
          ...options,
          sourceTokenCount: 1,
          sourceTokenIndex: 0,
        });
      } finally {
        boundary.destroy?.();
      }
    } finally {
      this.endRuntimeRun(runtimeRun);
    }
  }

  private async runBatchedPrefillFromBoundary(
    boundary: WebGpuBufferLike,
    positions: Int32Array,
    state: WebGpuStateLike,
    options: WebGpuInternalRunOptions,
    readHidden: boolean,
  ): Promise<WebGpuHiddenResult> {
    const tokenCount = options.sourceTokenCount;
    const tokenPositions = tokenPositionsFromBatchedMrope(positions, tokenCount);
    for (const position of tokenPositions) {
      if (position < 0 || position >= state.contextLength) {
        throw new Error(`Position ${position} is outside context length ${state.contextLength}`);
      }
    }

    const gpuState = this.ensureGpuState(state);
    const persistentCleanup: GpuResource[] = [];
    const hiddenSize = this.manifest.embeddingLength;
    const hiddenByteLength = hiddenSize * Float32Array.BYTES_PER_ELEMENT;
    const outputTokenCount = readHidden
      ? this.segmentEndLayerExclusive === this.manifest.blockCount && options.readAllHidden !== true
        ? 1
        : tokenCount
      : 0;
    const outputByteLength = outputTokenCount * hiddenByteLength;
    let hiddenReadback: WebGpuBufferLike | undefined;
    let topReadback: WebGpuBufferLike | undefined;
    let topTokensReadback: WebGpuBufferLike | undefined;
    let selectedTokenReadback: WebGpuBufferLike | undefined;
    let selectedTokensReadback: WebGpuBufferLike | undefined;
    const perLayerInputsBuffer = this.prepareBatchedPerLayerInputBuffer(options, persistentCleanup);

    const encodeStartMs = nowMs();
    if (readHidden) {
      hiddenReadback = this.arena.device.createBuffer({
        size: outputByteLength,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
    }
    if (options.computeSelectedTokens === true) {
      selectedTokensReadback = this.arena.device.createBuffer({
        size: tokenCount * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
    }

    let candidateCount = 0;
    let candidateByteLength = 0;
    let tokenCandidateCount = 0;
    let tokenCandidateByteLength = 0;
    const outputStripeCount = this.segmentEndLayerExclusive === this.manifest.blockCount
      ? this.requireOutputStripes().length
      : 0;
    if (options.computeTopKTokens === true) {
      tokenCandidateCount = Math.max(1, options.topK ?? 1);
      tokenCandidateByteLength = outputStripeCount * tokenCandidateCount * 2 * Float32Array.BYTES_PER_ELEMENT;
      topTokensReadback = this.arena.device.createBuffer({
        size: tokenCount * tokenCandidateByteLength,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
    }

    try {
      // Match llama.cpp: non-causal attention is evaluated as one physical batch.
      const prefillChunkSize = options.attentionCausal === false ? tokenCount : this.prefillChunkSize;
      for (let chunkStart = 0; chunkStart < tokenCount;) {
        const cleanup: GpuResource[] = [];
        const resources: Array<{ destroy: () => void }> = [];
        const encoder = this.arena.device.createCommandEncoder();
        const chunkTokenCount = this.prefillChunkTokenCount(
          tokenPositions,
          chunkStart,
          tokenCount,
          prefillChunkSize,
          state,
          options.attentionCausal !== false,
        );
        const chunkByteLength = chunkTokenCount * hiddenByteLength;
        try {
          const chunkPositions = tokenPositions.slice(chunkStart, chunkStart + chunkTokenCount);
          const positionsBuffer = this.createPositionsBuffer(chunkPositions, cleanup, `prefill.positions.${chunkStart}`);
          let currentBatch = this.arena.createScratchBuffer(
            `prefill.chunk.${chunkStart}.input`,
            chunkByteLength,
            GPU_STORAGE | GPU_COPY_SRC | GPU_COPY_DST,
          );
          cleanup.push(currentBatch);
          encoder.copyBufferToBuffer(boundary, chunkStart * hiddenByteLength, currentBatch, 0, chunkByteLength);

          let compute = this.beginComputePass(encoder, `prefill.chunk.${chunkStart}`);
          for (const layer of this.layers) {
            compute = this.restartComputePass(encoder, compute, `layer.${layer.layer}.prefill`);
            currentBatch = this.dispatchBatchedLayer(
              compute.pass,
              layer,
              gpuState,
              currentBatch,
              positionsBuffer,
              chunkPositions,
              state.contextLength,
              {
                ...options,
                perLayerInputsBuffer,
                sourceTokenIndex: chunkStart,
                sourceTokenCount: tokenCount,
              },
              cleanup,
              resources,
            );
          }

          const isLastChunk = chunkStart + chunkTokenCount >= tokenCount;
          let topBuffer: WebGpuBufferLike | undefined;
          let selectedTokenBuffer: WebGpuBufferLike | undefined;
          const selectedTokenBuffers: WebGpuBufferLike[] = [];
          const topTokenBuffers: WebGpuBufferLike[] = [];
          const lastHiddenOffset = (chunkTokenCount - 1) * hiddenByteLength;
          const selectedHidden = isLastChunk &&
              this.segmentEndLayerExclusive === this.manifest.blockCount &&
              chunkTokenCount > 1 &&
              (options.computeTopK === true || options.computeSelectedToken === true)
            ? this.createLastTokenView(compute.pass, resources, currentBatch, cleanup, chunkTokenCount)
            : currentBatch;

          if (isLastChunk && options.computeTopK === true) {
            compute = this.restartComputePass(encoder, compute, "output.topk");
            candidateCount = Math.max(1, options.topK ?? 1);
            const outputStripes = this.requireOutputStripes();
            candidateByteLength = outputStripes.length * candidateCount * 2 * Float32Array.BYTES_PER_ELEMENT;
            topBuffer = this.dispatchOutputTopK(compute.pass, selectedHidden, candidateCount, cleanup, resources);
            topReadback = this.arena.device.createBuffer({
              size: candidateByteLength,
              usage: GPU_MAP_READ | GPU_COPY_DST,
            });
          }
          if (isLastChunk && options.computeSelectedToken === true) {
            compute = this.restartComputePass(encoder, compute, "output.selected");
            selectedTokenBuffer = this.dispatchOutputSelectedToken(compute.pass, selectedHidden, cleanup, resources);
            selectedTokenReadback = this.arena.device.createBuffer({
              size: Uint32Array.BYTES_PER_ELEMENT,
              usage: GPU_MAP_READ | GPU_COPY_DST,
            });
          }
          if (options.computeSelectedTokens === true) {
            compute = this.restartComputePass(encoder, compute, "output.selected_tokens");
            for (let tokenIndex = 0; tokenIndex < chunkTokenCount; tokenIndex += 1) {
              const tokenHidden = chunkTokenCount === 1
                ? currentBatch
                : this.createTokenView(compute.pass, resources, currentBatch, cleanup, tokenIndex);
              selectedTokenBuffers.push(this.dispatchOutputSelectedToken(compute.pass, tokenHidden, cleanup, resources));
            }
          }
          if (options.computeTopKTokens === true) {
            compute = this.restartComputePass(encoder, compute, "output.topk_tokens");
            for (let tokenIndex = 0; tokenIndex < chunkTokenCount; tokenIndex += 1) {
              const tokenHidden = chunkTokenCount === 1
                ? currentBatch
                : this.createTokenView(compute.pass, resources, currentBatch, cleanup, tokenIndex);
              topTokenBuffers.push(this.dispatchOutputTopK(compute.pass, tokenHidden, tokenCandidateCount, cleanup, resources));
            }
          }

          this.endComputePass(encoder, compute);
          if (hiddenReadback) {
            if (this.segmentEndLayerExclusive === this.manifest.blockCount && options.readAllHidden !== true) {
              if (isLastChunk) {
                encoder.copyBufferToBuffer(currentBatch, lastHiddenOffset, hiddenReadback, 0, hiddenByteLength);
              }
            } else {
              encoder.copyBufferToBuffer(currentBatch, 0, hiddenReadback, chunkStart * hiddenByteLength, chunkByteLength);
            }
          }
          if (topReadback && topBuffer) {
            encoder.copyBufferToBuffer(topBuffer, 0, topReadback, 0, candidateByteLength);
          }
          if (selectedTokenReadback && selectedTokenBuffer) {
            encoder.copyBufferToBuffer(selectedTokenBuffer, 0, selectedTokenReadback, 0, Uint32Array.BYTES_PER_ELEMENT);
          }
          if (selectedTokensReadback) {
            for (let tokenIndex = 0; tokenIndex < selectedTokenBuffers.length; tokenIndex += 1) {
              encoder.copyBufferToBuffer(
                selectedTokenBuffers[tokenIndex] as WebGpuBufferLike,
                0,
                selectedTokensReadback,
                (chunkStart + tokenIndex) * Uint32Array.BYTES_PER_ELEMENT,
                Uint32Array.BYTES_PER_ELEMENT,
              );
            }
          }
          if (topTokensReadback) {
            for (let tokenIndex = 0; tokenIndex < topTokenBuffers.length; tokenIndex += 1) {
              encoder.copyBufferToBuffer(
                topTokenBuffers[tokenIndex] as WebGpuBufferLike,
                0,
                topTokensReadback,
                (chunkStart + tokenIndex) * tokenCandidateByteLength,
                tokenCandidateByteLength,
              );
            }
          }

          this.submitCommandBuffer(encoder.finish());
          await this.waitForSubmittedWorkDone();
        } finally {
          for (const resource of resources) {
            resource.destroy();
          }
          for (const item of cleanup.reverse()) {
            item.destroy?.();
          }
        }
        chunkStart += chunkTokenCount;
      }

      this.activeRunEncodeMs += nowMs() - encodeStartMs;

      if (!hiddenReadback && !topReadback && !topTokensReadback && !selectedTokenReadback && !selectedTokensReadback) {
        await this.readTimestampProfiler();
        return {
          hidden: new Float32Array(),
          selectedTokenId: undefined,
          selectedTokenIds: undefined,
          topTokens: undefined,
          topTokensByPosition: undefined,
        };
      }

      const hidden = readHidden ? new Float32Array(outputTokenCount * hiddenSize) : new Float32Array();
      if (hiddenReadback) {
        this.readbackCount += 1;
        await this.mapReadback(hiddenReadback);
        hidden.set(new Float32Array(hiddenReadback.getMappedRange()).slice());
        hiddenReadback.unmap();
        hiddenReadback.destroy?.();
        hiddenReadback = undefined;
        this.readbackBytes += outputByteLength;
        this.activeRunReadbackBytes += outputByteLength;
      }

      let topTokens: WebGpuTopToken[] | undefined;
      if (topReadback) {
        this.readbackCount += 1;
        await this.mapReadback(topReadback);
        const values = new Float32Array(topReadback.getMappedRange()).slice();
        topReadback.unmap();
        topReadback.destroy?.();
        topReadback = undefined;
        this.readbackBytes += candidateByteLength;
        this.activeRunReadbackBytes += candidateByteLength;
        topTokens = mergeTopCandidates(values, candidateCount, this.manifest.finalLogitSoftcap);
      }

      let topTokensByPosition: WebGpuTopToken[][] | undefined;
      if (topTokensReadback) {
        this.readbackCount += 1;
        await this.mapReadback(topTokensReadback);
        const values = new Float32Array(topTokensReadback.getMappedRange()).slice();
        topTokensReadback.unmap();
        topTokensReadback.destroy?.();
        topTokensReadback = undefined;
        const byteLength = tokenCount * tokenCandidateByteLength;
        this.readbackBytes += byteLength;
        this.activeRunReadbackBytes += byteLength;
        const floatsPerToken = tokenCandidateByteLength / Float32Array.BYTES_PER_ELEMENT;
        topTokensByPosition = Array.from({ length: tokenCount }, (_, tokenIndex) =>
          mergeTopCandidates(
            values.subarray(tokenIndex * floatsPerToken, (tokenIndex + 1) * floatsPerToken),
            tokenCandidateCount,
            this.manifest.finalLogitSoftcap,
          )
        );
      }

      let selectedTokenId: number | undefined;
      if (selectedTokenReadback) {
        this.readbackCount += 1;
        await this.mapReadback(selectedTokenReadback);
        selectedTokenId = new Uint32Array(selectedTokenReadback.getMappedRange()).slice()[0] ?? 0;
        this.activeRunSelectedTokenId = selectedTokenId;
        selectedTokenReadback.unmap();
        selectedTokenReadback.destroy?.();
        selectedTokenReadback = undefined;
        this.readbackBytes += Uint32Array.BYTES_PER_ELEMENT;
        this.activeRunReadbackBytes += Uint32Array.BYTES_PER_ELEMENT;
        this.selectedTokenReadbacks += 1;
      }

      let selectedTokenIds: number[] | undefined;
      if (selectedTokensReadback) {
        this.readbackCount += 1;
        await this.mapReadback(selectedTokensReadback);
        selectedTokenIds = Array.from(new Uint32Array(selectedTokensReadback.getMappedRange()).slice(0, tokenCount));
        selectedTokensReadback.unmap();
        selectedTokensReadback.destroy?.();
        selectedTokensReadback = undefined;
        const byteLength = tokenCount * Uint32Array.BYTES_PER_ELEMENT;
        this.readbackBytes += byteLength;
        this.activeRunReadbackBytes += byteLength;
        this.selectedTokenReadbacks += tokenCount;
        this.activeRunSelectedTokenId = selectedTokenIds.at(-1) ?? -1;
      }

      await this.readTimestampProfiler();
      return { hidden, selectedTokenId, selectedTokenIds, topTokens, topTokensByPosition };
    } finally {
      hiddenReadback?.destroy?.();
      topReadback?.destroy?.();
      topTokensReadback?.destroy?.();
      selectedTokenReadback?.destroy?.();
      selectedTokensReadback?.destroy?.();
      for (const item of persistentCleanup.reverse()) {
        item.destroy?.();
      }
    }
  }

  private dispatchBatchedLayer(
    pass: WebGpuComputePassLike,
    layer: GpuLayer,
    gpuState: GpuState,
    input: WebGpuBufferLike,
    positionsBuffer: WebGpuBufferLike,
    tokenPositions: Int32Array,
    contextLength: number,
    options: WebGpuInternalRunOptions,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const hiddenSize = this.manifest.embeddingLength;
    const tokenCount = tokenPositions.length;
    const queryDim = this.manifest.headCount * layer.headSize;
    const valueDim = this.manifest.headCount * layer.valueSize;
    const kvDim = layer.headCountKv * layer.headSize;
    const kvValueDim = layer.headCountKv * layer.valueSize;

    const qProjection = scratchF32(this.arena, queryDim * tokenCount, cleanup, `blk.${layer.layer}.prefill.q`);
    const attnQ8_0 = layer.q.type === "Q4_0"
      ? scratchQ8_0(this.arena, hiddenSize, tokenCount, hiddenSize / 32, cleanup, `blk.${layer.layer}.prefill.attn_norm.q8_0`)
      : undefined;
    const attnQ8K = attnQ8_0
      ? undefined
      : scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.prefill.attn_norm.q8k`);
    if (attnQ8_0) {
      dispatchBatchedRmsNormQ8_0Quantize(this.arena.device, pass, resources, input, layer.attnNorm.buffer, attnQ8_0, {
        length: hiddenSize,
        tokenCount,
        epsilon: this.epsilon,
      });
      dispatchQ8_0MatMul(pass, resources, layer.q, attnQ8_0, qProjection, tokenCount);
    } else if (attnQ8K) {
      dispatchBatchedRmsNormQ8KQuantize(this.arena.device, pass, resources, input, layer.attnNorm.buffer, attnQ8K, {
        length: hiddenSize,
        tokenCount,
        epsilon: this.epsilon,
      });
      dispatchKMatMul(pass, resources, layer.q, attnQ8K, qProjection, tokenCount);
    }
    const query = scratchF32(this.arena, queryDim * tokenCount, cleanup, `blk.${layer.layer}.prefill.q_rope`);
    dispatchBatchedFullQuery(
      this.arena.device,
      pass,
      resources,
      qProjection,
      layer.qNorm.buffer,
      this.ropeFreqFactors.buffer,
      positionsBuffer,
      query,
      {
        headCount: this.manifest.headCount,
        headSize: layer.headSize,
        ropeDims: ropeDimensionCount(this.manifest, layer.kind),
        epsilon: this.epsilon,
        freqBase: ropeFreqBase(this.manifest, layer.kind),
        tokenCount,
        hasFreqFactors: this.hasRopeFreqFactors && layer.kind === "full-attention",
      },
    );

    if (layer.hasKv) {
      const layerState = gpuState.fullAttention.get(layer.layer);
      if (!layerState || !layer.k || !layer.kNorm || (layer.valueProjectionMode !== "shared-with-key" && !layer.v)) {
        throw new Error(`Missing WebGPU KV state or weights for layer ${layer.layer}`);
      }
      const kProjection = scratchF32(this.arena, kvDim * tokenCount, cleanup, `blk.${layer.layer}.prefill.k`);
      const vProjection = layer.valueProjectionMode === "shared-with-key"
        ? kProjection
        : scratchF32(this.arena, kvValueDim * tokenCount, cleanup, `blk.${layer.layer}.prefill.v`);
      const dispatchedDualKv = layer.valueProjectionMode !== "shared-with-key" && this.usesStandardOptimizations() && (
        attnQ8_0
          ? dispatchDualQ4_0MatMul(pass, resources, layer.k, layer.v!, attnQ8_0, kProjection, vProjection, tokenCount)
          : dispatchDualQ4KMatMul(pass, resources, layer.k, layer.v!, attnQ8K!, kProjection, vProjection, tokenCount)
      );
      if (!dispatchedDualKv) {
        if (attnQ8_0) {
          dispatchQ8_0MatMul(pass, resources, layer.k, attnQ8_0, kProjection, tokenCount);
        } else {
          dispatchKMatMul(pass, resources, layer.k, attnQ8K!, kProjection, tokenCount);
        }
        if (layer.valueProjectionMode !== "shared-with-key") {
          if (attnQ8_0) {
            dispatchQ8_0MatMul(pass, resources, layer.v!, attnQ8_0, vProjection, tokenCount);
          } else {
            dispatchKMatMul(pass, resources, layer.v!, attnQ8K!, vProjection, tokenCount);
          }
        }
      }
      dispatchBatchedFullKvUpdate(
        this.arena.device,
        pass,
        resources,
        kProjection,
        vProjection,
        layer.kNorm.buffer,
        this.ropeFreqFactors.buffer,
        positionsBuffer,
        layerState.key,
        layerState.value,
        {
          headCount: layer.headCountKv,
          headSize: layer.headSize,
          valueSize: layer.valueSize,
          ropeDims: ropeDimensionCount(this.manifest, layer.kind),
          epsilon: this.epsilon,
          freqBase: ropeFreqBase(this.manifest, layer.kind),
          tokenCount,
          contextLength: layerState.capacity,
          hasFreqFactors: this.hasRopeFreqFactors && layer.kind === "full-attention",
        },
      );
    }

    const state = gpuState.fullAttention.get(layer.hasKv ? layer.layer : layer.kvSourceLayer);
    if (!state) {
      throw new Error(`Missing WebGPU KV state for layer ${layer.layer}`);
    }
    const attention = scratchF32(this.arena, tokenCount * valueDim, cleanup, `blk.${layer.layer}.prefill.attention`);
    const slidingWindow = options.attentionCausal === false
      ? undefined
      : layer.kind === "sliding-attention" ? this.manifest.slidingWindow : undefined;
    const keyValueRange = attentionLogicalRange(maxInt32(tokenPositions), state.capacity, contextLength, slidingWindow);
    const attentionOptions = {
      headSize: layer.headSize,
      valueSize: layer.valueSize,
      queryHeadCount: this.manifest.headCount,
      keyValueHeadCount: state.headCountKv,
      keyValueTokenCount: keyValueRange.count,
      contextLength: state.capacity,
      keyValueStart: keyValueRange.start,
      slidingWindow,
      tokenCount,
      scale: 1,
      causal: options.attentionCausal !== false,
    };
    const attentionHeadTokenCount = tokenCount * this.manifest.headCount;
    const attentionTileSize = 512;
    const attentionProbabilityTile = scratchF32(
      this.arena,
      attentionHeadTokenCount * attentionTileSize,
      cleanup,
      `blk.${layer.layer}.prefill.attention_rolling.probability`,
    );
    const attentionRowMax = scratchF32(this.arena, attentionHeadTokenCount, cleanup, `blk.${layer.layer}.prefill.attention_rolling.row_max`);
    const attentionRowSum = scratchF32(this.arena, attentionHeadTokenCount, cleanup, `blk.${layer.layer}.prefill.attention_rolling.row_sum`);
    const attentionTileMax = scratchF32(this.arena, attentionHeadTokenCount, cleanup, `blk.${layer.layer}.prefill.attention_rolling.tile_max`);
    const attentionTileSum = scratchF32(this.arena, attentionHeadTokenCount, cleanup, `blk.${layer.layer}.prefill.attention_rolling.tile_sum`);
    const attentionOldScale = scratchF32(this.arena, attentionHeadTokenCount, cleanup, `blk.${layer.layer}.prefill.attention_rolling.old_scale`);
    const attentionTileScale = scratchF32(this.arena, attentionHeadTokenCount, cleanup, `blk.${layer.layer}.prefill.attention_rolling.tile_scale`);
    this.activeRunAttentionTempBytes = Math.max(
      this.activeRunAttentionTempBytes,
      (
        attentionHeadTokenCount * attentionTileSize +
        attentionHeadTokenCount * 6
      ) * Float32Array.BYTES_PER_ELEMENT,
    );
    dispatchBatchedFullAttentionRollingTile(
      this.arena.device,
      pass,
      resources,
      query,
      state.key,
      state.value,
      positionsBuffer,
      attentionProbabilityTile,
      attentionRowMax,
      attentionRowSum,
      attentionTileMax,
      attentionTileSum,
      attentionOldScale,
      attentionTileScale,
      attention,
      { ...attentionOptions, tileSize: attentionTileSize },
    );

    const attentionOut = this.dispatchQuantizedMatMul(pass, resources, layer.attnOut, attention, tokenCount, cleanup, `blk.${layer.layer}.prefill.attention_out`);
    const attentionResidual = scratchF32(this.arena, hiddenSize * tokenCount, cleanup, `blk.${layer.layer}.prefill.attention_residual`);
    dispatchBatchedRmsNormResidualAdd(this.arena.device, pass, resources, attentionOut, layer.postAttentionNorm.buffer, input, attentionResidual, {
      length: hiddenSize,
      tokenCount,
      epsilon: this.epsilon,
    });

    const ffn = this.dispatchBatchedFfn(pass, layer, attentionResidual, tokenCount, cleanup, resources);
    return this.dispatchBatchedPerLayerInput(pass, layer, ffn, tokenCount, options, cleanup, resources);
  }

  private dispatchBatchedFfn(
    pass: WebGpuComputePassLike,
    layer: GpuLayer,
    residual: WebGpuBufferLike,
    tokenCount: number,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const hiddenSize = this.manifest.embeddingLength;
    const gate = scratchF32(this.arena, this.manifest.feedForwardLength * tokenCount, cleanup, `blk.${layer.layer}.prefill.ffn_gate`);
    const up = scratchF32(this.arena, this.manifest.feedForwardLength * tokenCount, cleanup, `blk.${layer.layer}.prefill.ffn_up`);
    const geglu = scratchF32(this.arena, this.manifest.feedForwardLength * tokenCount, cleanup, `blk.${layer.layer}.prefill.ffn_geglu`);
    if (layer.ffnGate.type === "Q4_0" && layer.ffnUp.type === "Q4_0") {
      const ffnQ8 = scratchQ8_0(this.arena, hiddenSize, tokenCount, hiddenSize / 32, cleanup, `blk.${layer.layer}.prefill.ffn_norm.q8_0`);
      dispatchBatchedRmsNormQ8_0Quantize(this.arena.device, pass, resources, residual, layer.ffnNorm.buffer, ffnQ8, {
        length: hiddenSize,
        tokenCount,
        epsilon: this.epsilon,
      });
      if (!dispatchDualQ4_0MatMul(pass, resources, layer.ffnGate, layer.ffnUp, ffnQ8, gate, up, tokenCount)) {
        throw new Error(`WebGPU Q4_0 FFN dual matmul failed for layer ${layer.layer}`);
      }
    } else {
      const ffnQ8 = scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.prefill.ffn_norm.q8k`);
      dispatchBatchedRmsNormQ8KQuantize(this.arena.device, pass, resources, residual, layer.ffnNorm.buffer, ffnQ8, {
        length: hiddenSize,
        tokenCount,
        epsilon: this.epsilon,
      });
      dispatchKMatMul(pass, resources, layer.ffnGate, ffnQ8, gate, tokenCount);
      dispatchKMatMul(pass, resources, layer.ffnUp, ffnQ8, up, tokenCount);
    }
    dispatchGeglu(this.arena.device, pass, resources, gate, up, geglu, this.manifest.feedForwardLength * tokenCount);
    const ffnOut = this.dispatchQuantizedMatMul(pass, resources, layer.ffnDown, geglu, tokenCount, cleanup, `blk.${layer.layer}.prefill.ffn_out`);
    const output = scratchF32(this.arena, hiddenSize * tokenCount, cleanup, `blk.${layer.layer}.prefill.ffn_residual`);
    dispatchBatchedRmsNormResidualAdd(this.arena.device, pass, resources, ffnOut, layer.postFfwNorm.buffer, residual, output, {
      length: hiddenSize,
      tokenCount,
      epsilon: this.epsilon,
    });
    return output;
  }

  private dispatchBatchedPerLayerInput(
    pass: WebGpuComputePassLike,
    layer: GpuLayer,
    input: WebGpuBufferLike,
    tokenCount: number,
    options: WebGpuInternalRunOptions,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const hiddenSize = this.manifest.embeddingLength;
    if (this.manifest.perLayerEmbeddingLength <= 0) {
      const scaled = scratchF32(this.arena, hiddenSize * tokenCount, cleanup, `blk.${layer.layer}.prefill.scaled`);
      dispatchScale(this.arena.device, pass, resources, input, layer.layerOutputScale.buffer, scaled, hiddenSize * tokenCount);
      return scaled;
    }
    if (!options.perLayerInputsBuffer || !layer.perLayerInputGate || !layer.perLayerProjection || !layer.postNorm) {
      throw new Error("WebGPU batched per-layer input requires prepared per-layer inputs and weights.");
    }
    const perLayerLength = this.manifest.perLayerEmbeddingLength;
    const perLayerOffset = (layer.layer * options.sourceTokenCount + options.sourceTokenIndex) * perLayerLength;
    const gate = scratchF32(this.arena, perLayerLength * tokenCount, cleanup, `blk.${layer.layer}.prefill.inp_gate`);
    const mixed = scratchF32(this.arena, perLayerLength * tokenCount, cleanup, `blk.${layer.layer}.prefill.inp_mixed`);
    dispatchF32MatMul(this.arena.device, pass, resources, layer.perLayerInputGate.buffer, input, gate, hiddenSize, perLayerLength, tokenCount);
    dispatchBatchedGegluSlice(this.arena.device, pass, resources, gate, options.perLayerInputsBuffer, mixed, {
      length: perLayerLength,
      tokenCount,
      rightOffset: perLayerOffset,
      rightStride: perLayerLength,
    });
    const projected = scratchF32(this.arena, hiddenSize * tokenCount, cleanup, `blk.${layer.layer}.prefill.inp_projected`);
    dispatchF32MatMul(this.arena.device, pass, resources, layer.perLayerProjection.buffer, mixed, projected, perLayerLength, hiddenSize, tokenCount);
    const output = scratchF32(this.arena, hiddenSize * tokenCount, cleanup, `blk.${layer.layer}.prefill.scaled`);
    dispatchBatchedRmsNormResidualAddScale(
      this.arena.device,
      pass,
      resources,
      projected,
      layer.postNorm.buffer,
      input,
      layer.layerOutputScale.buffer,
      output,
      {
        length: hiddenSize,
        tokenCount,
        epsilon: this.epsilon,
      },
    );
    return output;
  }

  private prepareBatchedPerLayerInputBuffer(
    options: WebGpuInternalRunOptions,
    cleanup: GpuResource[],
  ): WebGpuBufferLike | undefined {
    if (this.manifest.perLayerEmbeddingLength <= 0) {
      return undefined;
    }
    if (options.perLayerInputsBuffer) {
      return options.perLayerInputsBuffer;
    }
    if (!options.perLayerInputs) {
      throw new Error("WebGPU batched prefill requires prepared per-layer inputs.");
    }
    const buffer = this.arena.createBuffer(
      "prefill.per_layer_inputs",
      options.perLayerInputs.byteLength,
      GPU_STORAGE | GPU_COPY_DST,
    );
    this.arena.device.queue.writeBuffer(buffer, 0, options.perLayerInputs);
    cleanup.push(buffer);
    return buffer;
  }

  private createPositionsBuffer(
    positions: Int32Array,
    cleanup: GpuResource[],
    label: string,
  ): WebGpuBufferLike {
    const values = Uint32Array.from(positions);
    const buffer = this.arena.createScratchBuffer(label, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
    this.arena.device.queue.writeBuffer(buffer, 0, values);
    cleanup.push(buffer);
    return buffer;
  }

  private createLastTokenView(
    pass: WebGpuComputePassLike,
    resources: Array<{ destroy: () => void }>,
    input: WebGpuBufferLike,
    cleanup: GpuResource[],
    tokenCount: number,
  ): WebGpuBufferLike {
    return this.createTokenView(pass, resources, input, cleanup, tokenCount - 1);
  }

  private createTokenView(
    pass: WebGpuComputePassLike,
    resources: Array<{ destroy: () => void }>,
    input: WebGpuBufferLike,
    cleanup: GpuResource[],
    tokenIndex: number,
  ): WebGpuBufferLike {
    const output = scratchF32(this.arena, this.manifest.embeddingLength, cleanup, "prefill.last_hidden");
    dispatchTokenSlice(this.arena.device, pass, resources, input, output, {
      rowSize: this.manifest.embeddingLength,
      rowIndex: tokenIndex,
    });
    return output;
  }

  private async readMtpTargetKvLayer(
    layer: GpuLayer,
    layerState: FullAttentionGpuLayerState,
    tokenCount: number,
  ): Promise<MtpTargetKvLayerView> {
    const viewTokenCount = mtpTargetKvViewTokenCount(layer, layerState.capacity, this.manifest.slidingWindow, tokenCount);
    const logicalStart = Math.max(0, tokenCount - viewTokenCount);
    const keyCapacityElementCount = layerState.capacity * layer.headCountKv * layer.headSize;
    const valueCapacityElementCount = layerState.capacity * layer.headCountKv * layer.valueSize;
    const keyCapacityBytes = keyCapacityElementCount * F16_BYTE_LENGTH;
    const valueCapacityBytes = valueCapacityElementCount * F16_BYTE_LENGTH;
    const keyCopyBytes = alignBufferCopyBytes(keyCapacityBytes);
    const valueCopyBytes = alignBufferCopyBytes(valueCapacityBytes);
    const keyReadback = this.arena.device.createBuffer({
      label: `blk.${layer.layer}.mtp.key.readback`,
      size: Math.max(1, keyCopyBytes),
      usage: GPU_COPY_DST | GPU_MAP_READ,
    });
    const valueReadback = this.arena.device.createBuffer({
      label: `blk.${layer.layer}.mtp.value.readback`,
      size: Math.max(1, valueCopyBytes),
      usage: GPU_COPY_DST | GPU_MAP_READ,
    });
    try {
      const encoder = this.arena.device.createCommandEncoder();
      if (keyCapacityBytes > 0) {
        encoder.copyBufferToBuffer(layerState.key, 0, keyReadback, 0, keyCopyBytes);
      }
      if (valueCapacityBytes > 0) {
        encoder.copyBufferToBuffer(layerState.value, 0, valueReadback, 0, valueCopyBytes);
      }
      this.submitCommandBuffer(encoder.finish());

      const keyCapacity = await this.readF16BufferAsF32(keyReadback, keyCapacityElementCount, keyCapacityBytes);
      const valueCapacity = await this.readF16BufferAsF32(valueReadback, valueCapacityElementCount, valueCapacityBytes);
      const key = compactMtpTargetKeyView(keyCapacity, {
        logicalStart,
        tokenCount: viewTokenCount,
        capacity: layerState.capacity,
        headCountKv: layer.headCountKv,
        headSize: layer.headSize,
      });
      const value = compactMtpTargetValueView(valueCapacity, {
        logicalStart,
        tokenCount: viewTokenCount,
        capacity: layerState.capacity,
        headCountKv: layer.headCountKv,
        valueSize: layer.valueSize,
      });
      const keyBytes = key.length * F16_BYTE_LENGTH;
      const valueBytes = value.length * F16_BYTE_LENGTH;
      this.readbackBytes += keyBytes + valueBytes;
      this.activeRunReadbackBytes += keyBytes + valueBytes;
      return {
        key,
        value,
        keyLength: layer.headSize,
        valueLength: layer.valueSize,
        headCountKv: layer.headCountKv,
        contextLength: viewTokenCount,
        tokenCount,
        logicalStart,
      };
    } finally {
      keyReadback.destroy?.();
      valueReadback.destroy?.();
    }
  }

  private async readF16BufferAsF32(
    buffer: WebGpuBufferLike,
    elementCount: number,
    byteLength: number,
  ): Promise<Float32Array> {
    if (byteLength <= 0) {
      return new Float32Array();
    }
    this.readbackCount += 1;
    await this.mapReadback(buffer);
    try {
      const values = new Uint16Array(buffer.getMappedRange()).slice(0, elementCount);
      const output = new Float32Array(elementCount);
      for (let index = 0; index < values.length; index += 1) {
        output[index] = float16ToFloat32(values[index] ?? 0);
      }
      return output;
    } finally {
      buffer.unmap();
    }
  }

  private async runTokenFromBoundary(
    boundary: WebGpuBufferLike,
    tokenIndex: number,
    positions: Int32Array,
    state: WebGpuStateLike,
    options: WebGpuInternalRunOptions,
  ): Promise<WebGpuTokenResult> {
    const result = await this.runTokenFromBoundaryInternal(boundary, tokenIndex, positions, state, options, false);
    return { selectedTokenId: result.selectedTokenId, topTokens: result.topTokens };
  }

  private async runTokenFromBoundaryHidden(
    boundary: WebGpuBufferLike,
    tokenIndex: number,
    positions: Int32Array,
    state: WebGpuStateLike,
    options: WebGpuInternalRunOptions,
  ): Promise<WebGpuHiddenResult> {
    return this.runTokenFromBoundaryInternal(boundary, tokenIndex, positions, state, options, true);
  }

  private async prepareGpuInput(tokenIds: readonly number[]): Promise<PreparedGpuInput> {
    if (!this.supportsGpuInputPreparation()) {
      throw new Error("WebGPU token-id input preparation is not supported for this segment or tensor layout.");
    }
    const tokenCount = tokenIds.length;
    if (tokenCount <= 0) {
      throw new Error("WebGPU token-id input preparation requires at least one token.");
    }
    const cleanup: GpuResource[] = [];
    const resources: Array<{ destroy: () => void }> = [];
    try {
      const input = await this.loadGpuInputResources();
      const tokenEmbeddingChunks = await this.buildEmbeddingRowChunks("token_embd.weight", tokenIds, input.tokenEmbedding);
      for (const chunk of tokenEmbeddingChunks) {
        if (chunk.transientHandle) {
          cleanup.push(chunk.handle);
        }
      }
      const perLayerTokenEmbeddingChunks = this.manifest.perLayerEmbeddingLength > 0
        ? await this.buildEmbeddingRowChunks("per_layer_token_embd.weight", tokenIds, input.perLayerTokenEmbedding)
        : undefined;
      if (perLayerTokenEmbeddingChunks) {
        for (const chunk of perLayerTokenEmbeddingChunks) {
          if (chunk.transientHandle) {
            cleanup.push(chunk.handle);
          }
        }
      }

      const hidden = this.arena.createScratchBuffer(
        "input.hidden.gpu",
        tokenCount * this.manifest.embeddingLength * Float32Array.BYTES_PER_ELEMENT,
        GPU_STORAGE | GPU_COPY_SRC,
      );
      cleanup.push(hidden);
      let perLayerInputs: WebGpuBufferLike | undefined;

      const encodeStartMs = nowMs();
      const encoder = this.arena.device.createCommandEncoder();
      const compute = this.beginComputePass(encoder, "input.prepare");
      this.dispatchEmbeddingRowChunks(compute.pass, resources, cleanup, tokenEmbeddingChunks, hidden, {
        rowSize: this.manifest.embeddingLength,
        scale: Math.sqrt(this.manifest.embeddingLength),
      });

      if (this.manifest.perLayerEmbeddingLength > 0) {
        if (!input.perLayerModelProjection || !input.perLayerProjectionNorm) {
          throw new Error("WebGPU per-layer input resources are missing.");
        }
        if (!perLayerTokenEmbeddingChunks) {
          throw new Error("WebGPU per-layer token embedding rows are missing.");
        }
        const perLayerLength = this.manifest.perLayerEmbeddingLength;
        const totalPerLayerLength = perLayerLength * this.manifest.blockCount;
        const tokenRows = this.arena.createScratchBuffer(
          "input.per_layer_token_rows",
          tokenCount * totalPerLayerLength * Float32Array.BYTES_PER_ELEMENT,
          GPU_STORAGE,
        );
        const projected = this.arena.createScratchBuffer(
          "input.per_layer_projected",
          tokenCount * totalPerLayerLength * Float32Array.BYTES_PER_ELEMENT,
          GPU_STORAGE,
        );
        perLayerInputs = this.arena.createScratchBuffer(
          "input.per_layer_inputs",
          tokenCount * totalPerLayerLength * Float32Array.BYTES_PER_ELEMENT,
          GPU_STORAGE | GPU_COPY_SRC,
        );
        cleanup.push(tokenRows, projected, perLayerInputs);

        this.dispatchEmbeddingRowChunks(compute.pass, resources, cleanup, perLayerTokenEmbeddingChunks, tokenRows, {
          rowSize: totalPerLayerLength,
          scale: Math.sqrt(perLayerLength),
        });

        if (isF32Handle(input.perLayerModelProjection)) {
          dispatchF32MatMul(
            this.arena.device,
            compute.pass,
            resources,
            input.perLayerModelProjection.buffer,
            hidden,
            projected,
            this.manifest.embeddingLength,
            totalPerLayerLength,
            tokenCount,
          );
        } else {
          if (input.perLayerModelProjection.type === "Q4_0" || input.perLayerModelProjection.type === "Q8_0") {
            const q8 = scratchQ8_0(
              this.arena,
              this.manifest.embeddingLength,
              tokenCount,
              this.manifest.embeddingLength / 32,
              cleanup,
              "input.hidden.q8_0",
            );
            dispatchQ8_0Quantize(
              this.arena.device,
              compute.pass,
              resources,
              hidden,
              q8,
              this.manifest.embeddingLength,
              tokenCount,
              this.manifest.embeddingLength / 32,
            );
            dispatchQ8_0MatMul(compute.pass, resources, input.perLayerModelProjection, q8, projected, tokenCount);
          } else {
            const q8 = scratchQ8K(
              this.arena,
              this.manifest.embeddingLength,
              tokenCount,
              cleanup,
              "input.hidden.q8k",
            );
            dispatchQ8KQuantize(
              this.arena.device,
              compute.pass,
              resources,
              hidden,
              q8,
              this.manifest.embeddingLength,
              tokenCount,
            );
            dispatchKMatMul(compute.pass, resources, input.perLayerModelProjection, q8, projected, tokenCount);
          }
        }

        dispatchPreparePerLayerInputs(
          this.arena.device,
          compute.pass,
          resources,
          tokenRows,
          projected,
          input.perLayerProjectionNorm.buffer,
          perLayerInputs,
          {
            perLayerLength,
            totalPerLayerLength,
            tokenCount,
            blockCount: this.manifest.blockCount,
            projectionScale: 1 / Math.sqrt(this.manifest.embeddingLength),
            epsilon: this.epsilon,
          },
        );
      }

      this.endComputePass(encoder, compute);
      this.activeRunEncodeMs += nowMs() - encodeStartMs;
      this.submitCommandBuffer(encoder.finish());
      await this.readTimestampProfiler();
      this.deferResourceCleanup(resources);

      return {
        tokenCount,
        hidden,
        perLayerInputs,
        destroy: () => {
          for (const item of cleanup.reverse()) {
            item.destroy?.();
          }
        },
      };
    } catch (error) {
      for (const resource of resources.reverse()) {
        resource.destroy?.();
      }
      for (const item of cleanup.reverse()) {
        item.destroy?.();
      }
      throw error;
    }
  }

  private dispatchLayerKvUpdate(
    pass: WebGpuComputePassLike,
    layer: GpuLayer,
    gpuState: GpuState,
    input: WebGpuBufferLike,
    positions: Int32Array,
    tokenPosition: number,
    contextLength: number,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): void {
    if (!layer.hasKv) {
      return;
    }
    const layerState = gpuState.fullAttention.get(layer.layer);
    if (!layerState || !layer.k || !layer.kNorm || (layer.valueProjectionMode !== "shared-with-key" && !layer.v)) {
      throw new Error(`Missing WebGPU KV state or weights for layer ${layer.layer}`);
    }
    const hiddenSize = this.manifest.embeddingLength;
    const tokenCount = 1;
    const kvDim = layer.headCountKv * layer.headSize;
    const kvValueDim = layer.headCountKv * layer.valueSize;
    const attnQ8_0 = layer.k?.type === "Q4_0"
      ? scratchQ8_0(this.arena, hiddenSize, tokenCount, hiddenSize / 32, cleanup, `blk.${layer.layer}.prefill_kv.attn_norm.q8_0`)
      : undefined;
    const attnQ8K = attnQ8_0
      ? undefined
      : scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.prefill_kv.attn_norm.q8k`);
    if (attnQ8_0) {
      this.dispatchRmsNormToQ8_0(pass, resources, input, layer.attnNorm.buffer, attnQ8_0, hiddenSize);
    } else if (attnQ8K) {
      this.dispatchRmsNormToQ8K(pass, resources, input, layer.attnNorm.buffer, attnQ8K, hiddenSize, cleanup, `blk.${layer.layer}.prefill_kv.attn_norm`);
    }
    const kProjection = scratchF32(this.arena, kvDim, cleanup, `blk.${layer.layer}.prefill_kv.k`);
    const vProjection = layer.valueProjectionMode === "shared-with-key"
      ? kProjection
      : scratchF32(this.arena, kvValueDim, cleanup, `blk.${layer.layer}.prefill_kv.v`);
    const dispatchedDualKv = layer.valueProjectionMode !== "shared-with-key" && this.usesStandardOptimizations() && (
      attnQ8_0
        ? dispatchDualQ4_0MatMul(pass, resources, layer.k, layer.v!, attnQ8_0, kProjection, vProjection, tokenCount)
        : dispatchDualQ4KMatMul(pass, resources, layer.k, layer.v!, attnQ8K!, kProjection, vProjection, tokenCount)
    );
    if (!dispatchedDualKv) {
      if (attnQ8_0) {
        dispatchQ8_0MatMul(pass, resources, layer.k, attnQ8_0, kProjection, tokenCount);
      } else {
        dispatchKMatMul(pass, resources, layer.k, attnQ8K!, kProjection, tokenCount);
      }
      if (layer.valueProjectionMode !== "shared-with-key") {
        if (attnQ8_0) {
          dispatchQ8_0MatMul(pass, resources, layer.v!, attnQ8_0, vProjection, tokenCount);
        } else {
          dispatchKMatMul(pass, resources, layer.v!, attnQ8K!, vProjection, tokenCount);
        }
      }
    }
    if (this.usesStandardOptimizations()) {
      dispatchFullKvUpdate(
        this.arena.device,
        pass,
        resources,
        kProjection,
        vProjection,
        layer.kNorm.buffer,
        this.ropeFreqFactors.buffer,
        layerState.key,
        layerState.value,
        {
          headCount: layer.headCountKv,
          headSize: layer.headSize,
          valueSize: layer.valueSize,
          ropeDims: ropeDimensionCount(this.manifest, layer.kind),
          epsilon: this.epsilon,
          freqBase: ropeFreqBase(this.manifest, layer.kind),
          position: mropeTextPosition(positions, tokenPosition),
          tokenPosition,
          contextLength: layerState.capacity,
          hasFreqFactors: this.hasRopeFreqFactors && layer.kind === "full-attention",
        },
      );
      return;
    }
    const kNormed = scratchF32(this.arena, kvDim, cleanup, `blk.${layer.layer}.prefill_kv.k_norm`);
    dispatchHeadRmsNorm(
      this.arena.device,
      pass,
      resources,
      kProjection,
      layer.kNorm.buffer,
      kNormed,
      {
        headCount: layer.headCountKv,
        headSize: layer.headSize,
        epsilon: this.epsilon,
      },
    );
    dispatchKeyCacheRope(
      this.arena.device,
      pass,
      resources,
      kNormed,
      this.ropeFreqFactors.buffer,
      layerState.key,
      {
        headCount: layer.headCountKv,
        headSize: layer.headSize,
        ropeDims: ropeDimensionCount(this.manifest, layer.kind),
        freqBase: ropeFreqBase(this.manifest, layer.kind),
        position: mropeTextPosition(positions, tokenPosition),
        tokenPosition: tokenPosition % layerState.capacity,
        hasFreqFactors: this.hasRopeFreqFactors && layer.kind === "full-attention",
      },
    );
    const vNormed = scratchF32(this.arena, kvValueDim, cleanup, `blk.${layer.layer}.prefill_kv.v_norm`);
    dispatchHeadRmsNormNoWeight(
      this.arena.device,
      pass,
      resources,
      vProjection,
      vNormed,
      {
        headCount: layer.headCountKv,
        headSize: layer.valueSize,
        epsilon: this.epsilon,
      },
    );
    dispatchValueCacheWrite(
      this.arena.device,
      pass,
      resources,
      vNormed,
      layerState.value,
      {
        headCount: layer.headCountKv,
        valueSize: layer.valueSize,
        tokenPosition: tokenPosition % layerState.capacity,
        contextLength: layerState.capacity,
      },
    );
  }

  private async loadGpuInputResources(): Promise<GpuInputResources> {
    this.inputResourcesPromise ??= this.loadGpuInputResourcesUncached();
    return this.inputResourcesPromise;
  }

  private async loadGpuInputResourcesUncached(): Promise<GpuInputResources> {
    const startMs = nowMs();
    const tokenEmbedding = await this.loadFullEmbeddingGatherHandleIfBindable("token_embd.weight");
    if (this.manifest.perLayerEmbeddingLength <= 0) {
      this.lazyLoadMs += nowMs() - startMs;
      return { tokenEmbedding };
    }
    const projectionTensor = this.tensorReader.getTensor("per_layer_model_proj.weight");
    const resources = {
      tokenEmbedding,
      perLayerTokenEmbedding: await this.loadFullEmbeddingGatherHandleIfBindable("per_layer_token_embd.weight"),
      perLayerModelProjection: isF32CompatibleType(projectionTensor.type)
        ? await this.loadF32CompatibleHandle("per_layer_model_proj.weight")
        : await loadQuantizedHandle(this.arena, this.tensorReader, "per_layer_model_proj.weight"),
      perLayerProjectionNorm: await loadF32Handle(this.arena, this.tensorReader, "per_layer_proj_norm.weight"),
    };
    this.lazyLoadMs += nowMs() - startMs;
    return resources;
  }

  private async loadFullEmbeddingGatherHandleIfBindable(name: string): Promise<F32Handle | QuantizedHandle | undefined> {
    const tensor = this.tensorReader.getTensor(name);
    const rowElements = tensor.dimensions[0] ?? 0;
    const rowCount = tensor.dimensions[1] ?? 0;
    if (rowElements <= 0 || rowCount <= 0) {
      throw new Error(`${name} has invalid WebGPU embedding shape [${tensor.dimensions.join(", ")}]`);
    }
    const fullByteLength = this.embeddingUploadByteLength(tensor.type, rowElements, rowCount);
    const limits = await webGpuAdapterLimits();
    const bindingLimit = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
    return fullByteLength <= bindingLimit ? await this.loadEmbeddingGatherHandle(name) : undefined;
  }

  private async buildEmbeddingRowChunks(
    name: string,
    tokenIds: readonly number[],
    fullHandle: F32Handle | QuantizedHandle | undefined,
  ): Promise<GpuEmbeddingRowChunk[]> {
    const tensor = this.tensorReader.getTensor(name);
    const rowElements = tensor.dimensions[0] ?? 0;
    const rowCount = tensor.dimensions[1] ?? 0;
    if (rowElements <= 0 || rowCount <= 0) {
      throw new Error(`${name} has invalid WebGPU embedding shape [${tensor.dimensions.join(", ")}]`);
    }
    this.assertEmbeddingTokenIds(name, tokenIds, rowCount);
    if (fullHandle) {
      return [{
        handle: fullHandle,
        rowIds: Uint32Array.from(tokenIds),
        outputTokenOffset: 0,
        tokenCount: tokenIds.length,
        transientHandle: false,
      }];
    }

    const rowByteLength = this.embeddingRowByteLength(tensor.type, rowElements);
    const limits = await webGpuAdapterLimits();
    const bindingLimit = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
    // Quantized uploads are padded to a 4-byte boundary, so reserve the maximum 3-byte padding.
    const uploadPaddingHeadroom = tensor.type === "F32" ? 0 : 3;
    const maxRowsPerChunk = Math.floor(Math.max(0, bindingLimit - uploadPaddingHeadroom) / rowByteLength);
    if (maxRowsPerChunk < 1) {
      throw new Error(`${name} row byte length ${rowByteLength} exceeds maxStorageBufferBindingSize ${bindingLimit}`);
    }

    const chunks: GpuEmbeddingRowChunk[] = [];
    try {
      for (let offset = 0; offset < tokenIds.length; offset += maxRowsPerChunk) {
        const chunkTokenIds = tokenIds.slice(offset, offset + maxRowsPerChunk);
        const compactRows = await this.readCompactEmbeddingRows(name, chunkTokenIds, rowElements, rowCount);
        chunks.push({
          handle: this.createCompactEmbeddingHandle(name, tensor.type, rowElements, chunkTokenIds.length, compactRows),
          rowIds: Uint32Array.from(chunkTokenIds, (_tokenId, index) => index),
          outputTokenOffset: offset,
          tokenCount: chunkTokenIds.length,
          transientHandle: true,
        });
      }
    } catch (error) {
      for (const chunk of chunks.reverse()) {
        chunk.handle.destroy?.();
      }
      throw error;
    }
    return chunks;
  }

  private createCompactEmbeddingHandle(
    name: string,
    tensorType: GgmlTypeName,
    rowElements: number,
    rowCount: number,
    compactRows: Uint8Array,
  ): F32Handle | QuantizedHandle {
    if (tensorType === "F32") {
      const buffer = this.arena.createBuffer(`${name}.compact`, compactRows.byteLength, GPU_STORAGE | GPU_COPY_DST);
      this.arena.device.queue.writeBuffer(buffer, 0, compactRows);
      return {
        length: compactRows.byteLength / Float32Array.BYTES_PER_ELEMENT,
        byteLength: compactRows.byteLength,
        device: this.arena.device,
        buffer,
        destroy: () => buffer.destroy?.(),
      };
    }
    if (!isSupportedEmbeddingGatherType(tensorType)) {
      throw new Error(`${name} has unsupported WebGPU gather type ${tensorType}`);
    }
    return createQuantizedHandleFromBytes(
      this.arena,
      `${name}.compact`,
      webGpuMatMulType(tensorType, name),
      rowElements,
      rowCount,
      compactRows,
    );
  }

  private async readCompactEmbeddingRows(
    name: string,
    tokenIds: readonly number[],
    rowElements: number,
    rowCount: number,
  ): Promise<Uint8Array> {
    const tensor = this.tensorReader.getTensor(name);
    const rowByteLength = this.embeddingRowByteLength(tensor.type, rowElements);
    const ranges: TensorByteRange[] = tokenIds.map((tokenId) => {
      if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= rowCount) {
        throw new Error(`${name} token id ${tokenId} is outside vocab row count ${rowCount}`);
      }
      return {
        tensor,
        offset: BigInt(rowByteLength * tokenId),
        length: rowByteLength,
      };
    });
    const rows = await this.tensorReader.readTensorRangesCoalesced(ranges, {
      maxGapBytes: 1024 * 1024,
      maxReadBytes: 256 * 1024 * 1024,
      copyResults: false,
    });
    const compact = new Uint8Array(rowByteLength * tokenIds.length);
    for (let index = 0; index < rows.length; index += 1) {
      compact.set(rows[index] ?? new Uint8Array(0), index * rowByteLength);
    }
    return compact;
  }

  private assertEmbeddingTokenIds(name: string, tokenIds: readonly number[], rowCount: number): void {
    for (const tokenId of tokenIds) {
      if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= rowCount) {
        throw new Error(`${name} token id ${tokenId} is outside vocab row count ${rowCount}`);
      }
    }
  }

  private embeddingRowByteLength(tensorType: GgmlTypeName, rowElements: number): number {
    if (tensorType === "F32") {
      return rowElements * Float32Array.BYTES_PER_ELEMENT;
    }
    const type = webGpuMatMulType(tensorType, "embedding");
    return webGpuQuantizedWeightLayout(type, rowElements).rowByteLength;
  }

  private embeddingUploadByteLength(tensorType: GgmlTypeName, rowElements: number, rowCount: number): number {
    const byteLength = this.embeddingRowByteLength(tensorType, rowElements) * rowCount;
    return tensorType === "F32" ? byteLength : Math.ceil(byteLength / 4) * 4;
  }

  private async loadEmbeddingGatherHandle(name: string): Promise<F32Handle | QuantizedHandle> {
    const tensor = this.tensorReader.getTensor(name);
    if (tensor.type === "F32") {
      return loadF32Handle(this.arena, this.tensorReader, name);
    }
    if (isSupportedEmbeddingGatherType(tensor.type)) {
      return loadQuantizedHandle(this.arena, this.tensorReader, name);
    }
    throw new Error(`${name} has unsupported WebGPU gather type ${tensor.type}`);
  }

  private async loadF32CompatibleHandle(name: string): Promise<F32Handle> {
    const tensor = this.tensorReader.getTensor(name);
    if (tensor.type === "F32") {
      return loadF32Handle(this.arena, this.tensorReader, name);
    }
    if (!isF32CompatibleType(tensor.type)) {
      throw new Error(`${name} must be F32-compatible for WebGPU input preparation, got ${tensor.type}`);
    }
    const elementCount = tensor.dimensions.reduce((product, dimension) => product * dimension, 1);
    const source = await this.tensorReader.readTensorBytes(name);
    const values = dequantizeRow(tensor.type, source, elementCount);
    const buffer = this.arena.createBuffer(name, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
    this.arena.device.queue.writeBuffer(buffer, 0, values);
    return {
      length: elementCount,
      byteLength: values.byteLength,
      device: this.arena.device,
      buffer,
      destroy: () => buffer.destroy?.(),
    };
  }

  private dispatchGatherRowsScale(
    pass: WebGpuComputePassLike,
    resources: Array<{ destroy: () => void }>,
    handle: F32Handle | QuantizedHandle,
    tokenIdBuffer: WebGpuBufferLike,
    output: WebGpuBufferLike,
    options: {
      rowSize: number;
      tokenCount: number;
      scale: number;
      outputTokenOffset?: number;
    },
  ): void {
    if (isF32Handle(handle)) {
      dispatchF32GatherRowsScale(this.arena.device, pass, resources, handle.buffer, tokenIdBuffer, output, options);
      return;
    }
    dispatchQuantizedGatherRowsScale(
      pass,
      resources,
      handle,
      tokenIdBuffer,
      output,
      options,
    );
  }

  private dispatchEmbeddingRowChunks(
    pass: WebGpuComputePassLike,
    resources: Array<{ destroy: () => void }>,
    cleanup: GpuResource[],
    chunks: readonly GpuEmbeddingRowChunk[],
    output: WebGpuBufferLike,
    options: {
      rowSize: number;
      scale: number;
    },
  ): void {
    for (const chunk of chunks) {
      const tokenIdBuffer = this.arena.createScratchBuffer(
        "input.token_ids.chunk",
        chunk.rowIds.byteLength,
        GPU_STORAGE | GPU_COPY_DST,
      );
      this.arena.device.queue.writeBuffer(tokenIdBuffer, 0, chunk.rowIds);
      cleanup.push(tokenIdBuffer);
      this.dispatchGatherRowsScale(pass, resources, chunk.handle, tokenIdBuffer, output, {
        rowSize: options.rowSize,
        tokenCount: chunk.tokenCount,
        scale: options.scale,
        outputTokenOffset: chunk.outputTokenOffset,
      });
    }
  }

  private async runTokenFromBoundaryInternal(
    boundary: WebGpuBufferLike,
    tokenIndex: number,
    positions: Int32Array,
    state: WebGpuStateLike,
    options: WebGpuInternalRunOptions,
    readHidden: boolean,
  ): Promise<WebGpuHiddenResult> {
    const tokenPosition = tokenPositionFromSingleMrope(positions);
    if (tokenPosition < 0 || tokenPosition >= state.contextLength) {
      throw new Error(`Position ${tokenPosition} is outside context length ${state.contextLength}`);
    }
    const gpuState = this.ensureGpuState(state);
    const cleanup: GpuResource[] = [];
    const resources: Array<{ destroy: () => void }> = [];
    const encodeStartMs = nowMs();
    const encoder = this.arena.device.createCommandEncoder();
    let compute = this.beginComputePass(encoder, "decode");
    let hiddenReadback: WebGpuBufferLike | undefined;
    let topReadback: WebGpuBufferLike | undefined;
    let selectedTokenReadback: WebGpuBufferLike | undefined;

    try {
      let current = scratchF32(this.arena, this.manifest.embeddingLength, cleanup, "segment.hidden.token");
      dispatchTokenSlice(this.arena.device, compute.pass, resources, boundary, current, {
        rowSize: this.manifest.embeddingLength,
        rowIndex: tokenIndex,
      });

      for (const layer of this.layers) {
        const layerResult = this.dispatchLayer(encoder, compute, layer, gpuState, current, positions, tokenPosition, state.contextLength, options, cleanup, resources);
        current = layerResult.output;
        compute = layerResult.compute;
      }

      const hiddenByteLength = this.manifest.embeddingLength * Float32Array.BYTES_PER_ELEMENT;
      if (readHidden) {
        hiddenReadback = this.arena.device.createBuffer({
          size: hiddenByteLength,
          usage: GPU_MAP_READ | GPU_COPY_DST,
        });
      }

      let candidateCount = 0;
      let candidateByteLength = 0;
      let topBuffer: WebGpuBufferLike | undefined;
      let selectedTokenBuffer: WebGpuBufferLike | undefined;
      if (options.computeTopK === true) {
        compute = this.restartComputePass(encoder, compute, "output.topk");
        candidateCount = Math.max(1, options.topK ?? 1);
        const outputStripes = this.requireOutputStripes();
        candidateByteLength = outputStripes.length * candidateCount * 2 * Float32Array.BYTES_PER_ELEMENT;
        topBuffer = this.dispatchOutputTopK(compute.pass, current, candidateCount, cleanup, resources);
        topReadback = this.arena.device.createBuffer({
          size: candidateByteLength,
          usage: GPU_MAP_READ | GPU_COPY_DST,
        });
      }
      if (options.computeSelectedToken === true) {
        compute = this.restartComputePass(encoder, compute, "output.selected.norm");
        const selectedTokenResult = this.dispatchOutputSelectedTokenProfiled(encoder, compute, current, cleanup, resources);
        selectedTokenBuffer = selectedTokenResult.selectedToken;
        compute = selectedTokenResult.compute;
        selectedTokenReadback = this.arena.device.createBuffer({
          size: Uint32Array.BYTES_PER_ELEMENT,
          usage: GPU_MAP_READ | GPU_COPY_DST,
        });
      }

      this.endComputePass(encoder, compute);
      if (hiddenReadback) {
        encoder.copyBufferToBuffer(current, 0, hiddenReadback, 0, hiddenByteLength);
      }
      if (topReadback && topBuffer) {
        encoder.copyBufferToBuffer(topBuffer, 0, topReadback, 0, candidateByteLength);
      }
      if (selectedTokenReadback && selectedTokenBuffer) {
        encoder.copyBufferToBuffer(
          selectedTokenBuffer,
          0,
          selectedTokenReadback,
          0,
          Uint32Array.BYTES_PER_ELEMENT,
        );
      }
      this.activeRunEncodeMs += nowMs() - encodeStartMs;
      this.submitCommandBuffer(encoder.finish());

      if (!hiddenReadback && !topReadback && !selectedTokenReadback) {
        await this.readTimestampProfiler();
        this.deferResourceCleanup(resources, cleanup);
        resources.length = 0;
        cleanup.length = 0;
        return {
          hidden: new Float32Array(),
          selectedTokenId: undefined,
          topTokens: undefined,
        };
      }

      const hidden = readHidden ? new Float32Array(this.manifest.embeddingLength) : new Float32Array();
      if (hiddenReadback) {
        this.readbackCount += 1;
        await this.mapReadback(hiddenReadback);
        hidden.set(new Float32Array(hiddenReadback.getMappedRange()).slice());
        hiddenReadback.unmap();
        hiddenReadback.destroy?.();
        hiddenReadback = undefined;
        this.readbackBytes += hiddenByteLength;
        this.activeRunReadbackBytes += hiddenByteLength;
      }

      let topTokens: WebGpuTopToken[] | undefined;
      if (topReadback) {
        this.readbackCount += 1;
        await this.mapReadback(topReadback);
        const values = new Float32Array(topReadback.getMappedRange()).slice();
        topReadback.unmap();
        topReadback.destroy?.();
        topReadback = undefined;
        this.readbackBytes += candidateByteLength;
        this.activeRunReadbackBytes += candidateByteLength;
        topTokens = mergeTopCandidates(values, candidateCount, this.manifest.finalLogitSoftcap);
      }

      let selectedTokenId: number | undefined;
      if (selectedTokenReadback) {
        this.readbackCount += 1;
        await this.mapReadback(selectedTokenReadback);
        selectedTokenId = new Uint32Array(selectedTokenReadback.getMappedRange()).slice()[0] ?? 0;
        this.activeRunSelectedTokenId = selectedTokenId;
        selectedTokenReadback.unmap();
        selectedTokenReadback.destroy?.();
        selectedTokenReadback = undefined;
        this.readbackBytes += Uint32Array.BYTES_PER_ELEMENT;
        this.activeRunReadbackBytes += Uint32Array.BYTES_PER_ELEMENT;
        this.selectedTokenReadbacks += 1;
      }

      await this.readTimestampProfiler();
      return { hidden, selectedTokenId, topTokens };
    } finally {
      hiddenReadback?.destroy?.();
      topReadback?.destroy?.();
      selectedTokenReadback?.destroy?.();
      for (const resource of resources) {
        resource.destroy();
      }
      for (const item of cleanup.reverse()) {
        item.destroy?.();
      }
    }
  }

  private dispatchLayer(
    encoder: WebGpuCommandEncoderLike,
    computePass: ActiveComputePass,
    layer: GpuLayer,
    gpuState: GpuState,
    input: WebGpuBufferLike,
    positions: Int32Array,
    tokenPosition: number,
    contextLength: number,
    options: WebGpuInternalRunOptions,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): { output: WebGpuBufferLike; compute: ActiveComputePass } {
    let compute = computePass;
    let pass = compute.pass;
    const profileDetails = compute.profiling;
    const restartLayerSection = (section: string): void => {
      if (!profileDetails) {
        return;
      }
      compute = this.restartComputePass(encoder, compute, `layer.${layer.layer}.${section}`);
      pass = compute.pass;
    };
    const hiddenSize = this.manifest.embeddingLength;
    const tokenCount = 1;
    const queryDim = this.manifest.headCount * layer.headSize;
    const valueDim = this.manifest.headCount * layer.valueSize;
    const kvDim = layer.headCountKv * layer.headSize;
    const kvValueDim = layer.headCountKv * layer.valueSize;

    restartLayerSection("attn.q_proj");
    const qProjection = scratchF32(this.arena, queryDim, cleanup, `blk.${layer.layer}.q`);
    const attnQ8_0 = layer.q.type === "Q4_0"
      ? scratchQ8_0(this.arena, hiddenSize, tokenCount, hiddenSize / 32, cleanup, `blk.${layer.layer}.attn_norm.q8_0`)
      : undefined;
    const attnQ8K = attnQ8_0
      ? undefined
      : scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.attn_norm.q8k`);
    if (attnQ8_0) {
      this.dispatchRmsNormToQ8_0(pass, resources, input, layer.attnNorm.buffer, attnQ8_0, hiddenSize);
      dispatchQ8_0MatMul(pass, resources, layer.q, attnQ8_0, qProjection, tokenCount);
    } else if (attnQ8K) {
      this.dispatchRmsNormToQ8K(pass, resources, input, layer.attnNorm.buffer, attnQ8K, hiddenSize, cleanup, `blk.${layer.layer}.attn_norm`);
      dispatchKMatMul(pass, resources, layer.q, attnQ8K, qProjection, tokenCount);
    }
    restartLayerSection("attn.q_rope");
    const query = scratchF32(this.arena, queryDim, cleanup, `blk.${layer.layer}.q_rope`);
    if (this.usesStandardOptimizations()) {
      dispatchFullQuery(
        this.arena.device,
        pass,
        resources,
        qProjection,
        layer.qNorm.buffer,
        this.ropeFreqFactors.buffer,
        query,
        {
          headCount: this.manifest.headCount,
          headSize: layer.headSize,
          ropeDims: ropeDimensionCount(this.manifest, layer.kind),
          epsilon: this.epsilon,
          freqBase: ropeFreqBase(this.manifest, layer.kind),
          position: mropeTextPosition(positions, tokenPosition),
          hasFreqFactors: this.hasRopeFreqFactors && layer.kind === "full-attention",
        },
      );
    } else {
      const qNormed = scratchF32(this.arena, queryDim, cleanup, `blk.${layer.layer}.q_norm`);
      dispatchHeadRmsNorm(
        this.arena.device,
        pass,
        resources,
        qProjection,
        layer.qNorm.buffer,
        qNormed,
        {
          headCount: this.manifest.headCount,
          headSize: layer.headSize,
          epsilon: this.epsilon,
        },
      );
      dispatchRope(
        this.arena.device,
        pass,
        resources,
        qNormed,
        this.ropeFreqFactors.buffer,
        query,
        {
          headCount: this.manifest.headCount,
          headSize: layer.headSize,
          ropeDims: ropeDimensionCount(this.manifest, layer.kind),
          freqBase: ropeFreqBase(this.manifest, layer.kind),
          position: mropeTextPosition(positions, tokenPosition),
          hasFreqFactors: this.hasRopeFreqFactors && layer.kind === "full-attention",
        },
      );
    }
    if (layer.hasKv && options.skipKvUpdate !== true) {
      const layerState = gpuState.fullAttention.get(layer.layer);
      if (!layerState || !layer.k || !layer.kNorm || (layer.valueProjectionMode !== "shared-with-key" && !layer.v)) {
        throw new Error(`Missing WebGPU KV state or weights for layer ${layer.layer}`);
      }
      const kProjection = scratchF32(this.arena, kvDim, cleanup, `blk.${layer.layer}.k`);
      const vProjection = layer.valueProjectionMode === "shared-with-key"
        ? kProjection
        : scratchF32(this.arena, kvValueDim, cleanup, `blk.${layer.layer}.v`);
      restartLayerSection("attn.kv_proj");
      const dispatchedDualKv = layer.valueProjectionMode !== "shared-with-key" && this.usesStandardOptimizations() && (
        attnQ8_0
          ? dispatchDualQ4_0MatMul(pass, resources, layer.k, layer.v!, attnQ8_0, kProjection, vProjection, tokenCount)
          : dispatchDualQ4KMatMul(pass, resources, layer.k, layer.v!, attnQ8K!, kProjection, vProjection, tokenCount)
      );
      if (!dispatchedDualKv) {
        if (attnQ8_0) {
          dispatchQ8_0MatMul(pass, resources, layer.k, attnQ8_0, kProjection, tokenCount);
        } else {
          dispatchKMatMul(pass, resources, layer.k, attnQ8K!, kProjection, tokenCount);
        }
        if (layer.valueProjectionMode !== "shared-with-key") {
          if (attnQ8_0) {
            dispatchQ8_0MatMul(pass, resources, layer.v!, attnQ8_0, vProjection, tokenCount);
          } else {
            dispatchKMatMul(pass, resources, layer.v!, attnQ8K!, vProjection, tokenCount);
          }
        }
      }
      restartLayerSection("attn.kv_update");
      if (this.usesStandardOptimizations()) {
        dispatchFullKvUpdate(
          this.arena.device,
          pass,
          resources,
          kProjection,
          vProjection,
          layer.kNorm.buffer,
          this.ropeFreqFactors.buffer,
          layerState.key,
          layerState.value,
          {
            headCount: layer.headCountKv,
            headSize: layer.headSize,
            valueSize: layer.valueSize,
            ropeDims: ropeDimensionCount(this.manifest, layer.kind),
            epsilon: this.epsilon,
            freqBase: ropeFreqBase(this.manifest, layer.kind),
            position: mropeTextPosition(positions, tokenPosition),
            tokenPosition,
            contextLength: layerState.capacity,
            hasFreqFactors: this.hasRopeFreqFactors && layer.kind === "full-attention",
          },
        );
      } else {
        const kNormed = scratchF32(this.arena, kvDim, cleanup, `blk.${layer.layer}.k_norm`);
        dispatchHeadRmsNorm(
          this.arena.device,
          pass,
          resources,
          kProjection,
          layer.kNorm.buffer,
          kNormed,
          {
            headCount: layer.headCountKv,
            headSize: layer.headSize,
            epsilon: this.epsilon,
          },
        );
        dispatchKeyCacheRope(
          this.arena.device,
          pass,
          resources,
          kNormed,
          this.ropeFreqFactors.buffer,
          layerState.key,
          {
            headCount: layer.headCountKv,
            headSize: layer.headSize,
            ropeDims: ropeDimensionCount(this.manifest, layer.kind),
            freqBase: ropeFreqBase(this.manifest, layer.kind),
            position: mropeTextPosition(positions, tokenPosition),
            tokenPosition: tokenPosition % layerState.capacity,
            hasFreqFactors: this.hasRopeFreqFactors && layer.kind === "full-attention",
          },
        );
        const vNormed = scratchF32(this.arena, kvValueDim, cleanup, `blk.${layer.layer}.v_norm`);
        dispatchHeadRmsNormNoWeight(
          this.arena.device,
          pass,
          resources,
          vProjection,
          vNormed,
          {
            headCount: layer.headCountKv,
            headSize: layer.valueSize,
            epsilon: this.epsilon,
          },
        );
        dispatchValueCacheWrite(
          this.arena.device,
          pass,
          resources,
          vNormed,
          layerState.value,
          {
            headCount: layer.headCountKv,
            valueSize: layer.valueSize,
            tokenPosition: tokenPosition % layerState.capacity,
            contextLength: layerState.capacity,
          },
        );
      }
    }

    restartLayerSection("attn.score");
    const state = gpuState.fullAttention.get(layer.hasKv ? layer.layer : layer.kvSourceLayer);
    if (!state) {
      throw new Error(`Missing WebGPU KV state for layer ${layer.layer}`);
    }
    const slidingWindow = options.attentionCausal === false
      ? undefined
      : layer.kind === "sliding-attention" ? this.manifest.slidingWindow : undefined;
    const keyValueRange = options.attentionCausal === false
      ? { start: 0, count: Math.min(state.capacity, contextLength, options.keyValueTokenCount ?? tokenPosition + 1) }
      : attentionLogicalRange(tokenPosition, state.capacity, contextLength, slidingWindow);
    const probabilityTokenCapacity = bucketAttentionProbabilityTokenCount(keyValueRange.count);
    const probabilities = scratchF32(
      this.arena,
      this.manifest.headCount * probabilityTokenCapacity,
      cleanup,
      `blk.${layer.layer}.attention_probabilities`,
    );
    this.activeRunAttentionTempBytes = Math.max(
      this.activeRunAttentionTempBytes,
      this.manifest.headCount * probabilityTokenCapacity * Float32Array.BYTES_PER_ELEMENT,
    );
    const attention = scratchF32(this.arena, valueDim, cleanup, `blk.${layer.layer}.attention`);
    const attentionOptions = {
      headSize: layer.headSize,
      valueSize: layer.valueSize,
      queryHeadCount: this.manifest.headCount,
      keyValueHeadCount: state.headCountKv,
      keyValueTokenCount: keyValueRange.count,
      contextLength: state.capacity,
      keyValueStart: keyValueRange.start,
      scale: 1,
      tokenPosition,
      slidingWindow,
    };
    dispatchFullAttentionScore(this.arena.device, pass, resources, query, state.key, probabilities, attentionOptions);
    restartLayerSection("attn.apply");
    dispatchFullAttentionApply(this.arena.device, pass, resources, state.value, probabilities, attention, {
      ...attentionOptions,
      keyValueStart: keyValueRange.start,
    });

    restartLayerSection("attn.out");
    const attentionOut = this.dispatchQuantizedMatMul(pass, resources, layer.attnOut, attention, tokenCount, cleanup, `blk.${layer.layer}.attention_out`);
    const attentionResidual = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.attention_residual`);
    this.dispatchRmsNormThenResidualAdd(pass, resources, attentionOut, layer.postAttentionNorm.buffer, input, attentionResidual, hiddenSize, cleanup, `blk.${layer.layer}.attention_out_norm`);

    restartLayerSection("ffn.norm_quant");
    const ffn = this.dispatchFfn(pass, layer, attentionResidual, cleanup, resources, profileDetails
      ? (section) => {
          restartLayerSection(`ffn.${section}`);
          return pass;
        }
      : undefined);
    restartLayerSection("per_layer");
    const scaled = this.dispatchPerLayerInput(pass, layer, ffn, options, cleanup, resources);
    return { output: scaled, compute };
  }

  private dispatchFfn(
    pass: WebGpuComputePassLike,
    layer: GpuLayer,
    residual: WebGpuBufferLike,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
    restartSection?: (section: string) => WebGpuComputePassLike,
  ): WebGpuBufferLike {
    let activePass = pass;
    const hiddenSize = this.manifest.embeddingLength;
    const gate = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_gate`);
    const up = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_up`);
    const geglu = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_geglu`);
    if (layer.ffnGate.type === "Q4_0" && layer.ffnUp.type === "Q4_0") {
      const ffnQ8 = scratchQ8_0(this.arena, hiddenSize, 1, hiddenSize / 32, cleanup, `blk.${layer.layer}.ffn_norm.q8_0`);
      this.dispatchRmsNormToQ8_0(activePass, resources, residual, layer.ffnNorm.buffer, ffnQ8, hiddenSize);
      activePass = restartSection?.("gate_up") ?? activePass;
      if (!dispatchDualQ4_0MatMul(activePass, resources, layer.ffnGate, layer.ffnUp, ffnQ8, gate, up, 1)) {
        throw new Error(`WebGPU Q4_0 FFN dual matmul failed for layer ${layer.layer}`);
      }
    } else {
      const ffnQ8 = scratchQ8K(this.arena, hiddenSize, 1, cleanup, `blk.${layer.layer}.ffn_norm.q8k`);
      this.dispatchRmsNormToQ8K(activePass, resources, residual, layer.ffnNorm.buffer, ffnQ8, hiddenSize, cleanup, `blk.${layer.layer}.ffn_norm`);
      activePass = restartSection?.("gate") ?? activePass;
      dispatchKMatMul(activePass, resources, layer.ffnGate, ffnQ8, gate, 1);
      activePass = restartSection?.("up") ?? activePass;
      dispatchKMatMul(activePass, resources, layer.ffnUp, ffnQ8, up, 1);
    }
    activePass = restartSection?.("geglu") ?? activePass;
    dispatchGeglu(this.arena.device, activePass, resources, gate, up, geglu, this.manifest.feedForwardLength);
    activePass = restartSection?.("down") ?? activePass;
    const ffnOut = this.dispatchQuantizedMatMul(activePass, resources, layer.ffnDown, geglu, 1, cleanup, `blk.${layer.layer}.ffn_out`);
    activePass = restartSection?.("post") ?? activePass;
    const output = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.ffn_residual`);
    this.dispatchRmsNormThenResidualAdd(activePass, resources, ffnOut, layer.postFfwNorm.buffer, residual, output, hiddenSize, cleanup, `blk.${layer.layer}.ffn_out_norm`);
    return output;
  }

  private dispatchPerLayerInput(
    pass: WebGpuComputePassLike,
    layer: GpuLayer,
    input: WebGpuBufferLike,
    options: WebGpuInternalRunOptions,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    if (this.manifest.perLayerEmbeddingLength <= 0) {
      const scaled = scratchF32(this.arena, this.manifest.embeddingLength, cleanup, `blk.${layer.layer}.scaled`);
      dispatchScale(this.arena.device, pass, resources, input, layer.layerOutputScale.buffer, scaled, this.manifest.embeddingLength);
      return scaled;
    }
    if ((!options.perLayerInputs && !options.perLayerInputsBuffer) || !layer.perLayerInputGate || !layer.perLayerProjection || !layer.postNorm) {
      throw new Error("WebGPU per-layer input requires prepared per-layer inputs and weights.");
    }
    const perLayerLength = this.manifest.perLayerEmbeddingLength;
    const perLayerOffset = (layer.layer * options.sourceTokenCount + options.sourceTokenIndex) * perLayerLength;
    let perLayerBuffer = options.perLayerInputsBuffer;
    if (!perLayerBuffer) {
      const sourceOffset = (layer.layer * options.sourceTokenCount + options.sourceTokenIndex) * perLayerLength;
      const perLayerSlice = options.perLayerInputs!.slice(sourceOffset, sourceOffset + perLayerLength);
      if (perLayerSlice.length !== perLayerLength) {
        throw new Error(`WebGPU per-layer input shape mismatch for layer ${layer.layer}: ${perLayerSlice.length}`);
      }
      perLayerBuffer = this.arena.createBuffer(
        `blk.${layer.layer}.per_layer_input`,
        perLayerSlice.byteLength,
        GPU_STORAGE | GPU_COPY_DST,
      );
      this.arena.device.queue.writeBuffer(perLayerBuffer, 0, perLayerSlice);
      cleanup.push(perLayerBuffer);
    }

    const gate = scratchF32(this.arena, perLayerLength, cleanup, `blk.${layer.layer}.inp_gate`);
    const mixed = scratchF32(this.arena, perLayerLength, cleanup, `blk.${layer.layer}.inp_mixed`);
    dispatchF32MatMul(this.arena.device, pass, resources, layer.perLayerInputGate.buffer, input, gate, this.manifest.embeddingLength, perLayerLength, 1);
    if (options.perLayerInputsBuffer) {
      dispatchGegluSlice(this.arena.device, pass, resources, gate, perLayerBuffer, mixed, perLayerLength, perLayerOffset);
    } else {
      dispatchGeglu(this.arena.device, pass, resources, gate, perLayerBuffer, mixed, perLayerLength);
    }
    const projected = scratchF32(this.arena, this.manifest.embeddingLength, cleanup, `blk.${layer.layer}.inp_projected`);
    dispatchF32MatMul(this.arena.device, pass, resources, layer.perLayerProjection.buffer, mixed, projected, perLayerLength, this.manifest.embeddingLength, 1);
    const output = scratchF32(this.arena, this.manifest.embeddingLength, cleanup, `blk.${layer.layer}.scaled`);
    this.dispatchRmsNormThenResidualAddScale(
      pass,
      resources,
      projected,
      layer.postNorm.buffer,
      input,
      layer.layerOutputScale.buffer,
      output,
      this.manifest.embeddingLength,
      cleanup,
      `blk.${layer.layer}.inp_projected_norm`,
    );
    return output;
  }

  private dispatchQuantizedMatMul(
    pass: WebGpuComputePassLike,
    resources: Array<{ destroy: () => void }>,
    handle: QuantizedHandle,
    input: WebGpuBufferLike,
    columnCount: number,
    cleanup: GpuResource[],
    label: string,
  ): WebGpuBufferLike {
    const output = scratchF32(this.arena, handle.rowCount * columnCount, cleanup, label);
    if (handle.type === "Q4_0" || handle.type === "Q8_0") {
      const q8 = scratchQ8_0(this.arena, handle.inputSize, columnCount, handle.blockCount, cleanup, `${label}.q8_0`);
      dispatchQ8_0Quantize(this.arena.device, pass, resources, input, q8, handle.inputSize, columnCount, handle.blockCount);
      dispatchQ8_0MatMul(pass, resources, handle, q8, output, columnCount);
      return output;
    }
    const q8 = scratchQ8K(this.arena, handle.inputSize, columnCount, cleanup, `${label}.q8k`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, input, q8, handle.inputSize, columnCount);
    dispatchKMatMul(pass, resources, handle, q8, output, columnCount);
    return output;
  }

  private dispatchPreparedOutputMatMul(
    pass: WebGpuComputePassLike,
    resources: Array<{ destroy: () => void }>,
    stripe: OutputStripe,
    q8_0: ReturnType<typeof scratchQ8_0> | undefined,
    q8k: ReturnType<typeof scratchQ8K> | undefined,
    logits: WebGpuBufferLike,
  ): void {
    if (stripe.type === "Q4_0" || stripe.type === "Q8_0") {
      if (!q8_0) {
        throw new Error(`WebGPU ${stripe.type} output matmul requires Q8_0 activation.`);
      }
      dispatchQ8_0MatMul(pass, resources, stripe, q8_0, logits, 1);
      return;
    }
    if (!q8k) {
      throw new Error(`WebGPU ${stripe.type} output matmul requires Q8_K activation.`);
    }
    dispatchKMatMul(pass, resources, stripe, q8k, logits, 1);
  }

  private dispatchOutputTopK(
    pass: WebGpuComputePassLike,
    hidden: WebGpuBufferLike,
    topKCount: number,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const outputNorm = this.requireOutputNorm();
    const outputStripes = this.requireOutputStripes();
    const hiddenSize = this.manifest.embeddingLength;
    const usesQ8_0 = outputStripesUseQ8_0(outputStripes);
    const q8_0 = usesQ8_0 ? scratchQ8_0(this.arena, hiddenSize, 1, hiddenSize / 32, cleanup, "output_norm.q8_0") : undefined;
    const q8k = usesQ8_0 ? undefined : scratchQ8K(this.arena, hiddenSize, 1, cleanup, "output_norm.q8k");
    if (q8_0) {
      this.dispatchRmsNormToQ8_0(pass, resources, hidden, outputNorm.buffer, q8_0, hiddenSize);
    } else if (q8k) {
      this.dispatchRmsNormToQ8K(pass, resources, hidden, outputNorm.buffer, q8k, hiddenSize, cleanup, "output_norm");
    }

    const candidates = this.arena.createScratchBuffer(
      "output.topk.candidates",
      outputStripes.length * topKCount * 2 * Float32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE | GPU_COPY_SRC,
    );
    cleanup.push(candidates);
    for (let index = 0; index < outputStripes.length; index += 1) {
      const stripe = outputStripes[index];
      if (!stripe) {
        continue;
      }
      const logits = scratchF32(this.arena, stripe.rowCount, cleanup, `output.logits.${index}`);
      this.dispatchPreparedOutputMatMul(pass, resources, stripe, q8_0, q8k, logits);
      dispatchTopK(this.arena.device, pass, resources, logits, candidates, {
        rowCount: stripe.rowCount,
        rowOffset: stripe.rowOffset,
        topK: topKCount,
        candidateOffset: index * topKCount * 2,
      });
    }
    return candidates;
  }

  private dispatchOutputSelectedToken(
    pass: WebGpuComputePassLike,
    hidden: WebGpuBufferLike,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const outputNorm = this.requireOutputNorm();
    const outputStripes = this.requireOutputStripes();
    const hiddenSize = this.manifest.embeddingLength;
    const usesQ8_0 = outputStripesUseQ8_0(outputStripes);
    const q8_0 = usesQ8_0 ? scratchQ8_0(this.arena, hiddenSize, 1, hiddenSize / 32, cleanup, "output_norm.selected.q8_0") : undefined;
    const q8k = usesQ8_0 ? undefined : scratchQ8K(this.arena, hiddenSize, 1, cleanup, "output_norm.selected.q8k");
    if (q8_0) {
      this.dispatchRmsNormToQ8_0(pass, resources, hidden, outputNorm.buffer, q8_0, hiddenSize);
    } else if (q8k) {
      this.dispatchRmsNormToQ8K(pass, resources, hidden, outputNorm.buffer, q8k, hiddenSize, cleanup, "output_norm.selected");
    }
    const candidateCount = outputTop1CandidateCount(outputStripes);

    const candidates = this.arena.createScratchBuffer(
      "output.selected.candidates",
      candidateCount * 2 * Float32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE,
    );
    const selectedToken = this.arena.createScratchBuffer(
      "output.selected.token",
      Uint32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE | GPU_COPY_SRC,
    );
    cleanup.push(candidates, selectedToken);
    let candidateOffset = 0;
    for (let index = 0; index < outputStripes.length; index += 1) {
      const stripe = outputStripes[index];
      if (!stripe) {
        continue;
      }
      const logits = scratchF32(this.arena, stripe.rowCount, cleanup, `output.selected.logits.${index}`);
      this.dispatchPreparedOutputMatMul(pass, resources, stripe, q8_0, q8k, logits);
      candidateOffset += 2 * dispatchTop1Chunks(this.arena.device, pass, resources, logits, candidates, {
        rowCount: stripe.rowCount,
        rowOffset: stripe.rowOffset,
        candidateOffset,
      });
    }
    dispatchSelectTop1Candidate(
      this.arena.device,
      pass,
      resources,
      candidates,
      selectedToken,
      candidateCount,
    );
    return selectedToken;
  }

  private dispatchOutputSelectedTokenProfiled(
    encoder: WebGpuCommandEncoderLike,
    computePass: ActiveComputePass,
    hidden: WebGpuBufferLike,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): { selectedToken: WebGpuBufferLike; compute: ActiveComputePass } {
    if (!computePass.profiling) {
      return {
        selectedToken: this.dispatchOutputSelectedToken(computePass.pass, hidden, cleanup, resources),
        compute: computePass,
      };
    }
    let compute = computePass;
    let pass = compute.pass;
    const outputNorm = this.requireOutputNorm();
    const outputStripes = this.requireOutputStripes();
    const hiddenSize = this.manifest.embeddingLength;
    const usesQ8_0 = outputStripesUseQ8_0(outputStripes);
    const q8_0 = usesQ8_0 ? scratchQ8_0(this.arena, hiddenSize, 1, hiddenSize / 32, cleanup, "output_norm.selected.q8_0") : undefined;
    const q8k = usesQ8_0 ? undefined : scratchQ8K(this.arena, hiddenSize, 1, cleanup, "output_norm.selected.q8k");
    if (q8_0) {
      this.dispatchRmsNormToQ8_0(pass, resources, hidden, outputNorm.buffer, q8_0, hiddenSize);
    } else if (q8k) {
      this.dispatchRmsNormToQ8K(pass, resources, hidden, outputNorm.buffer, q8k, hiddenSize, cleanup, "output_norm.selected");
    }
    const candidateCount = outputTop1CandidateCount(outputStripes);

    const candidates = this.arena.createScratchBuffer(
      "output.selected.candidates",
      candidateCount * 2 * Float32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE,
    );
    const selectedToken = this.arena.createScratchBuffer(
      "output.selected.token",
      Uint32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE | GPU_COPY_SRC,
    );
    cleanup.push(candidates, selectedToken);

    let candidateOffset = 0;
    for (let index = 0; index < outputStripes.length; index += 1) {
      const stripe = outputStripes[index];
      if (!stripe) {
        continue;
      }
      compute = this.restartComputePass(encoder, compute, `output.selected.stripe.${index}.matmul`);
      pass = compute.pass;
      const logits = scratchF32(this.arena, stripe.rowCount, cleanup, `output.selected.logits.${index}`);
      this.dispatchPreparedOutputMatMul(pass, resources, stripe, q8_0, q8k, logits);
      compute = this.restartComputePass(encoder, compute, `output.selected.stripe.${index}.top1_chunks`);
      pass = compute.pass;
      candidateOffset += 2 * dispatchTop1Chunks(this.arena.device, pass, resources, logits, candidates, {
        rowCount: stripe.rowCount,
        rowOffset: stripe.rowOffset,
        candidateOffset,
      });
    }

    compute = this.restartComputePass(encoder, compute, "output.selected.reduce");
    dispatchSelectTop1Candidate(
      this.arena.device,
      compute.pass,
      resources,
      candidates,
      selectedToken,
      candidateCount,
    );
    return { selectedToken, compute };
  }

  private ensureGpuState(state: WebGpuStateLike): GpuState {
    if (state.contextLength <= 0) {
      throw new Error(`Invalid WebGPU state context length: ${state.contextLength}`);
    }
    const key = state as object;
    const existing = this.states.get(key);
    if (existing) {
      for (const layer of this.layers) {
        if (!layer.hasKv) {
          continue;
        }
        const expectedCapacity = state.fullAttention?.get(layer.layer)?.capacity;
        const layerState = existing.fullAttention.get(layer.layer);
        if (expectedCapacity !== undefined && layerState !== undefined && layerState.capacity < expectedCapacity) {
          this.resizeGpuKvCacheLayer(state, layer, layerState, expectedCapacity);
        }
      }
      return existing;
    }
    if (state.nextPosition !== 0) {
      throw new Error("WebGPU segment state is missing for a non-empty chat state; replay from position 0 is required.");
    }
    const fullAttention = new Map<number, FullAttentionGpuLayerState>();
    for (const layer of this.layers) {
      if (!layer.hasKv) {
        continue;
      }
      const capacity = state.fullAttention?.get(layer.layer)?.capacity ?? kvCacheCapacity(
        layer.kind === "sliding-attention" ? "sliding" : "full",
        state.contextLength,
        this.manifest.slidingWindow,
      );
      fullAttention.set(layer.layer, {
        key: this.createGpuKvCacheBuffer(layer, capacity, "key", layer.headSize),
        value: this.createGpuKvCacheBuffer(layer, capacity, "value", layer.valueSize),
        capacity,
        headCountKv: layer.headCountKv,
      });
    }
    const created = { fullAttention };
    this.states.set(key, created);
    addInferenceStateDisposeCallback(state as InferenceState, () => {
      destroyGpuState(created);
      this.states.delete(key);
    });
    return created;
  }

  private createGpuKvCacheBuffer(
    layer: GpuLayer,
    capacity: number,
    kind: "key" | "value",
    itemSize: number,
  ): WebGpuBufferLike {
    return this.arena.createBuffer(
      `blk.${layer.layer}.gpu.${kind}_cache`,
      webGpuKvCacheBufferByteLength(capacity, layer.headCountKv, itemSize),
      GPU_STORAGE | GPU_COPY_SRC,
    );
  }

  private resizeGpuKvCacheLayer(
    state: WebGpuStateLike,
    layer: GpuLayer,
    layerState: FullAttentionGpuLayerState,
    nextCapacity: number,
  ): void {
    const previousCapacity = layerState.capacity;
    const nextKey = this.createGpuKvCacheBuffer(layer, nextCapacity, "key", layer.headSize);
    const nextValue = this.createGpuKvCacheBuffer(layer, nextCapacity, "value", layer.valueSize);
    const cleanup: GpuResource[] = [layerState.key, layerState.value];
    const resources: KvResizeResource[] = [];
    try {
      const available = Math.min(state.contextLength, state.nextPosition, previousCapacity);
      const logicalStart = Math.max(0, state.nextPosition - available);
      if (available > 0) {
        const encoder = this.arena.device.createCommandEncoder();
        const pass = this.countedPass(encoder.beginComputePass());
        const keyResource = createKvResizeResource(this.arena.device, layerState.key, nextKey, {
          logicalStart,
          tokenCount: available,
          oldCapacity: previousCapacity,
          newCapacity: nextCapacity,
          elementStride: layer.headCountKv * layer.headSize,
          layout: "token-major",
        });
        resources.push(keyResource);
        pass.setPipeline(keyResource.pipeline);
        pass.setBindGroup(0, keyResource.bindGroup);
        pass.dispatchWorkgroups(Math.ceil((available * layer.headCountKv * layer.headSize) / 256));

        const valueResource = createKvResizeResource(this.arena.device, layerState.value, nextValue, {
          logicalStart,
          tokenCount: available,
          oldCapacity: previousCapacity,
          newCapacity: nextCapacity,
          elementStride: layer.headCountKv * layer.valueSize,
          layout: "dim-head-token",
        });
        resources.push(valueResource);
        pass.setPipeline(valueResource.pipeline);
        pass.setBindGroup(0, valueResource.bindGroup);
        pass.dispatchWorkgroups(Math.ceil((available * layer.headCountKv * layer.valueSize) / 256));
        pass.end();
        this.submitCommandBuffer(encoder.finish());
      }

      layerState.key = nextKey;
      layerState.value = nextValue;
      layerState.capacity = nextCapacity;
      cleanup.push(...resources.map((resource) => resource.paramsBuffer));
      this.deferResourceCleanup(cleanup);
    } catch (error) {
      nextKey.destroy?.();
      nextValue.destroy?.();
      for (const resource of resources) {
        resource.paramsBuffer.destroy?.();
      }
      throw error;
    }
  }

  private prefillChunkTokenCount(
    tokenPositions: Int32Array,
    chunkStart: number,
    tokenCount: number,
    requestedChunkSize: number,
    state: WebGpuStateLike,
    causal: boolean,
  ): number {
    const requested = Math.min(requestedChunkSize, tokenCount - chunkStart);
    if (!causal) {
      return requested;
    }
    const ringCapacity = this.minSlidingRingCapacity(state);
    if (ringCapacity === undefined) {
      return requested;
    }
    if (ringCapacity <= 1) {
      return 1;
    }
    let minPosition = Infinity;
    let maxPosition = -Infinity;
    let count = 0;
    for (let offset = 0; offset < requested; offset += 1) {
      const position = tokenPositions[chunkStart + offset] ?? 0;
      const nextMin = Math.min(minPosition, position);
      const nextMax = Math.max(maxPosition, position);
      if (offset > 0 && nextMax - nextMin >= ringCapacity) {
        break;
      }
      minPosition = nextMin;
      maxPosition = nextMax;
      count += 1;
    }
    return Math.max(1, count);
  }

  private minSlidingRingCapacity(state: WebGpuStateLike): number | undefined {
    let capacity: number | undefined;
    for (const layer of this.layers) {
      if (!layer.hasKv || layer.kind !== "sliding-attention") {
        continue;
      }
      const layerCapacity = state.fullAttention?.get(layer.layer)?.capacity ??
        kvCacheCapacity("sliding", state.contextLength, this.manifest.slidingWindow);
      if (layerCapacity >= state.contextLength) {
        continue;
      }
      capacity = capacity === undefined ? layerCapacity : Math.min(capacity, layerCapacity);
    }
    return capacity;
  }

  private assertBatchedHidden(inputHidden: Float32Array, positions: Int32Array): number {
    const tokenCount = inputHidden.length / this.manifest.embeddingLength;
    if (!Number.isInteger(tokenCount) || tokenCount <= 0) {
      throw new Error(`WebGPU segment batched input shape mismatch: ${inputHidden.length}`);
    }
    if (positions.length !== tokenCount && positions.length !== tokenCount * 4) {
      throw new Error(`WebGPU segment batched position shape mismatch: ${positions.length}`);
    }
    return tokenCount;
  }

  private assertTokenIdBatch(tokenIds: readonly number[], positions: Int32Array): number {
    const tokenCount = tokenIds.length;
    if (tokenCount <= 0) {
      throw new Error("WebGPU token-id batch must contain at least one token.");
    }
    if (positions.length !== tokenCount && positions.length !== tokenCount * 4) {
      throw new Error(`WebGPU token-id position shape mismatch: ${positions.length}`);
    }
    return tokenCount;
  }

  private requireOutputNorm(): F32Handle {
    if (!this.outputNorm) {
      throw new Error("WebGPU output norm was not loaded for this segment runner.");
    }
    return this.outputNorm;
  }

  private requireOutputStripes(): OutputStripe[] {
    if (!this.outputStripes) {
      throw new Error("WebGPU output stripes were not loaded for this segment runner.");
    }
    return this.outputStripes;
  }
}

function alignBufferCopyBytes(byteLength: number): number {
  if (byteLength <= 0) {
    return 0;
  }
  const remainder = byteLength % 4;
  return remainder === 0 ? byteLength : byteLength + 4 - remainder;
}

export function webGpuKvCacheBufferByteLength(capacity: number, headCountKv: number, itemSize: number): number {
  return alignBufferCopyBytes(capacity * headCountKv * itemSize * F16_BYTE_LENGTH);
}

function mtpTargetKvViewTokenCount(
  layer: GpuLayer,
  capacity: number,
  slidingWindow: number,
  tokenCount: number,
): number {
  if (tokenCount <= 0 || capacity <= 0) {
    return 0;
  }
  if (layer.kind === "sliding-attention") {
    return Math.min(tokenCount, slidingWindow, capacity);
  }
  return Math.min(tokenCount, capacity);
}

function compactMtpTargetKeyView(
  source: Float32Array,
  options: {
    logicalStart: number;
    tokenCount: number;
    capacity: number;
    headCountKv: number;
    headSize: number;
  },
): Float32Array {
  const tokenStride = options.headCountKv * options.headSize;
  const output = new Float32Array(options.tokenCount * tokenStride);
  for (let token = 0; token < options.tokenCount; token += 1) {
    const sourceSlot = (options.logicalStart + token) % options.capacity;
    output.set(
      source.subarray(sourceSlot * tokenStride, (sourceSlot + 1) * tokenStride),
      token * tokenStride,
    );
  }
  return output;
}

function compactMtpTargetValueView(
  source: Float32Array,
  options: {
    logicalStart: number;
    tokenCount: number;
    capacity: number;
    headCountKv: number;
    valueSize: number;
  },
): Float32Array {
  const output = new Float32Array(options.valueSize * options.headCountKv * options.tokenCount);
  for (let dim = 0; dim < options.valueSize; dim += 1) {
    for (let head = 0; head < options.headCountKv; head += 1) {
      const sourceRow = (dim * options.headCountKv + head) * options.capacity;
      const targetRow = (dim * options.headCountKv + head) * options.tokenCount;
      for (let token = 0; token < options.tokenCount; token += 1) {
        const sourceSlot = (options.logicalStart + token) % options.capacity;
        output[targetRow + token] = source[sourceRow + sourceSlot] ?? 0;
      }
    }
  }
  return output;
}

async function loadRopeFreqFactors(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
): Promise<{ handle: F32Handle; present: boolean }> {
  try {
    const tensor = tensorReader.getTensor("rope_freqs.weight");
    if (tensor.type === "F32") {
      return {
        handle: await loadF32Handle(arena, tensorReader, "rope_freqs.weight"),
        present: true,
      };
    }
  } catch {
    // No optional full-attention RoPE factors.
  }
  const value = new Float32Array([1]);
  const buffer = arena.createBuffer("rope_freqs.identity", value.byteLength, GPU_STORAGE | GPU_COPY_DST);
  arena.device.queue.writeBuffer(buffer, 0, value);
  return {
    handle: {
      length: 1,
      byteLength: value.byteLength,
      device: arena.device,
      buffer,
      destroy: () => buffer.destroy?.(),
    },
    present: false,
  };
}

function tokenPositionFromSingleMrope(positions: Int32Array): number {
  if (positions.length === 1 || positions.length === 4) {
    return positions[0] ?? 0;
  }
  throw new Error(`WebGPU token path expects one position, got ${positions.length}`);
}

function tokenPositionsFromBatch(positions: Int32Array, tokenIndex: number, tokenCount: number): Int32Array {
  if (positions.length === tokenCount) {
    return new Int32Array([positions[tokenIndex] ?? 0]);
  }
  if (positions.length === tokenCount * 4) {
    return new Int32Array([
      positions[tokenIndex] ?? 0,
      positions[tokenIndex + tokenCount] ?? 0,
      positions[tokenIndex + tokenCount * 2] ?? 0,
      positions[tokenIndex + tokenCount * 3] ?? 0,
    ]);
  }
  throw new Error(`WebGPU token batch expects ${tokenCount} or ${tokenCount * 4} positions, got ${positions.length}`);
}

function tokenPositionsFromBatchedMrope(positions: Int32Array, tokenCount: number): Int32Array {
  if (positions.length === tokenCount) {
    return positions;
  }
  if (positions.length === tokenCount * 4) {
    return positions.slice(0, tokenCount);
  }
  throw new Error(`WebGPU token batch expects ${tokenCount} or ${tokenCount * 4} positions, got ${positions.length}`);
}

function maxInt32(values: Int32Array): number {
  let max = -Infinity;
  for (const value of values) {
    max = Math.max(max, value);
  }
  return max;
}

function destroyGpuState(state: GpuState): void {
  for (const cache of state.fullAttention.values()) {
    cache.key.destroy?.();
    cache.value.destroy?.();
  }
  state.fullAttention.clear();
}

function mropeTextPosition(positions: Int32Array, fallback: number): number {
  return positions[0] ?? fallback;
}

function ropeDimensionCount(manifest: ModelManifest, kind: GpuLayer["kind"]): number {
  return kind === "sliding-attention"
    ? manifest.rope.slidingDimensionCount
    : manifest.rope.fullDimensionCount;
}

function ropeFreqBase(manifest: ModelManifest, kind: GpuLayer["kind"]): number {
  return kind === "sliding-attention"
    ? manifest.rope.slidingFreqBase
    : manifest.rope.fullFreqBase;
}

function isF32Handle(handle: QuantizedHandle | F32Handle): handle is F32Handle {
  return "buffer" in handle;
}

function isSupportedEmbeddingGatherType(type: string): boolean {
  return type === "F32" || type === "Q4_0" || type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "Q8_0";
}

function isF32CompatibleType(type: string): boolean {
  return type === "F32" || type === "F16" || type === "BF16";
}

function isSupportedProjectionType(type: string): boolean {
  return isF32CompatibleType(type) || type === "Q4_0" || type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "Q8_0";
}

function mergeTopCandidates(
  values: Float32Array,
  topKCount: number,
  finalLogitSoftcap: number | undefined,
): WebGpuTopToken[] {
  const best: WebGpuTopToken[] = [];
  for (let index = 0; index < values.length; index += 2) {
    const id = Math.trunc(values[index] ?? 0);
    const rawValue = values[index + 1] ?? -Infinity;
    if (!Number.isFinite(rawValue)) {
      continue;
    }
    const value = finalLogitSoftcap !== undefined && finalLogitSoftcap > 0
      ? Math.fround(Math.tanh(rawValue / finalLogitSoftcap) * finalLogitSoftcap)
      : rawValue;
    best.push({ id, value });
    best.sort((left, right) => right.value - left.value);
    if (best.length > topKCount) {
      best.pop();
    }
  }
  return best;
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

async function mapWithConcurrency<TInput, TOutput>(
  values: readonly TInput[],
  concurrency: number,
  mapper: (value: TInput, index: number) => Promise<TOutput>,
): Promise<TOutput[]> {
  const outputs = new Array<TOutput>(values.length);
  let nextIndex = 0;
  const workerCount = Math.min(values.length, Math.max(1, Math.floor(concurrency)));
  await Promise.all(Array.from({ length: workerCount }, async () => {
    for (;;) {
      const index = nextIndex;
      nextIndex += 1;
      if (index >= values.length) {
        return;
      }
      outputs[index] = await mapper(values[index]!, index);
    }
  }));
  return outputs;
}

function formatGpuSectionMs(sectionMs: ReadonlyMap<string, number>): string {
  return [...sectionMs.entries()]
    .map(([label, ms]) => `${label}=${Math.round(ms * 1000) / 1000}`)
    .join(";");
}

function outputTop1CandidateCount(outputStripes: readonly OutputStripe[]): number {
  return outputStripes.reduce((sum, stripe) => sum + Math.ceil(stripe.rowCount / TOP1_CHUNK_SIZE), 0);
}

function outputStripesUseQ8_0(outputStripes: readonly OutputStripe[]): boolean {
  return outputStripes.length > 0 && outputStripes.every((stripe) => stripe.type === "Q4_0" || stripe.type === "Q8_0");
}

function createKvResizeResource(
  device: WebGpuDeviceLike,
  source: WebGpuBufferLike,
  destination: WebGpuBufferLike,
  options: {
    logicalStart: number;
    tokenCount: number;
    oldCapacity: number;
    newCapacity: number;
    elementStride: number;
    layout: "token-major" | "dim-head-token";
  },
): KvResizeResource {
  const params = new Uint32Array([
    options.logicalStart,
    options.tokenCount,
    options.oldCapacity,
    options.newCapacity,
    options.elementStride,
    options.layout === "dim-head-token" ? 1 : 0,
    0,
    0,
  ]);
  const paramsBuffer = device.createBuffer({
    size: uniformBufferSize(params.byteLength),
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "storage"),
      { binding: 2, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: KV_CACHE_RESIZE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, source),
        bindBuffer(1, destination),
        bindBuffer(2, paramsBuffer),
      ],
    }),
    paramsBuffer,
  };
}

function uniformBufferSize(byteLength: number): number {
  return Math.max(32, Math.ceil(byteLength / 16) * 16);
}

function attentionLogicalRange(
  tokenPosition: number,
  cacheCapacity: number,
  contextLength: number,
  slidingWindow: number | undefined,
): { start: number; count: number } {
  const available = Math.min(contextLength, tokenPosition + 1);
  if (slidingWindow === undefined) {
    return { start: 0, count: Math.min(cacheCapacity, available) };
  }
  const count = Math.min(cacheCapacity, slidingWindow, available);
  return { start: Math.max(0, tokenPosition + 1 - count), count };
}

function bucketAttentionProbabilityTokenCount(tokenCount: number): number {
  return Math.max(1, Math.ceil(tokenCount / 256) * 256);
}

function normalizePrefillChunkSize(value: number | undefined): number {
  if (value === undefined) {
    return 64;
  }
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Invalid WebGPU prefill chunk size: ${value}`);
  }
  return value;
}

function normalizeOptimizationLevel(value: WebGpuOptimizationLevel | undefined): WebGpuOptimizationLevel {
  return value === "baseline" || value === "standard" ? value : "standard";
}

const KV_CACHE_RESIZE_WGSL = `
enable f16;

struct Params {
  logicalStart: u32,
  tokenCount: u32,
  oldCapacity: u32,
  newCapacity: u32,
  elementStride: u32,
  layout: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> sourceValues: array<f16>;
@group(0) @binding(1) var<storage, read_write> destinationValues: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let linear = globalId.x;
  let total = params.tokenCount * params.elementStride;
  if (linear >= total) {
    return;
  }

  let token = linear / params.elementStride;
  let element = linear % params.elementStride;
  let logicalPosition = params.logicalStart + token;
  let oldSlot = logicalPosition % params.oldCapacity;
  let newSlot = logicalPosition % params.newCapacity;
  var sourceIndex = oldSlot * params.elementStride + element;
  var destinationIndex = newSlot * params.elementStride + element;
  if (params.layout == 1u) {
    sourceIndex = element * params.oldCapacity + oldSlot;
    destinationIndex = element * params.newCapacity + newSlot;
  }
  destinationValues[destinationIndex] = sourceValues[sourceIndex];
}
`;
