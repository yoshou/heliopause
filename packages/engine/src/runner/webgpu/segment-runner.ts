import type { GgmlTypeName } from "../../gguf";
import { type GgufTensorReader, type TensorByteRange } from "../../tensor-reader";
import type { ModelManifest } from "../../model";
import { addInferenceStateDisposeCallback, type InferenceState } from "../../runtime";
import { dequantizeRow } from "../../quant";
import { GPU_COPY_DST, GPU_COPY_SRC, GPU_MAP_READ, GPU_QUERY_RESOLVE, GPU_STORAGE, WEBGPU_MEMORY_LIMIT_BYTES } from "./gpu-constants";
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
  dispatchBatchedRmsNormQ8KQuantize,
  dispatchBatchedRmsNormResidualAdd,
  dispatchBatchedRmsNormResidualAddScale,
  dispatchDualQ4KMatMul,
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
import type { WebGpuBufferLike, WebGpuCommandEncoderLike, WebGpuComputePassLike, WebGpuQuerySetLike, WebGpuTopToken } from "./gpu-types";
import type { SegmentRunner } from "../segment-runner";

export type WebGpuSegmentRunnerOptions = {
  tensorReader: GgufTensorReader;
  manifest: ModelManifest;
  epsilon: number;
  contextLength: number;
  memoryLimitBytes?: number;
  prefillChunkSize?: number;
  segmentStartLayer: number;
  segmentEndLayerExclusive?: number;
  loadOutput?: boolean;
};

export type WebGpuStateLike = {
  contextLength: number;
  nextPosition: number;
};

export type WebGpuTokenResult = {
  selectedTokenId?: number;
  topTokens?: WebGpuTopToken[];
};

export type WebGpuHiddenResult = {
  hidden: Float32Array;
  selectedTokenId?: number;
  topTokens?: WebGpuTopToken[];
};

export type WebGpuRunOptions = {
  computeSelectedToken?: boolean;
  computeTopK?: boolean;
  topK?: number;
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
  webgpuLastRunBufferCreateLabels: string;
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
  profileGpuPass: boolean;
  profileSections: boolean;
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
    gpuTimingStatus: "not-requested",
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
    this.lazyLoadMs = lazyLoadMs;
  }

  static async create(options: WebGpuSegmentRunnerOptions): Promise<WebGpuSegmentRunner> {
    const startMs = nowMs();
    const device = await webGpuDevice();
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
      webgpuLastRunBufferCreateLabels: this.lastRunStats.resourceStats?.bufferCreateLabels ?? "",
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
    const { device, cache } = installWebGpuRuntimeResourceCache(this.arena.device);
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
    this.activeRunTimestampStatus = webGpuGpuTimingEnabled() ? "not-requested" : "disabled";
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
    if (webGpuFusionEnabled()) {
      dispatchRmsNormQ8KQuantize(this.arena.device, pass, resources, input, weight, q8, length, this.epsilon);
      return;
    }
    const normalized = scratchF32(this.arena, length, cleanup, label);
    dispatchRmsNorm(this.arena.device, pass, resources, input, weight, normalized, length, this.epsilon);
    dispatchQ8KQuantize(this.arena.device, pass, resources, normalized, q8, length, 1);
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
    if (webGpuFusionEnabled()) {
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
    if (webGpuFusionEnabled()) {
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
    profileGpuPass: boolean,
    label = "gpu",
    profileSections = false,
  ): ActiveComputePass {
    const timestampPass = profileGpuPass ? this.allocateTimestampPass(label) : undefined;
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
      profileGpuPass,
      profileSections,
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
    if (!computePass.profileSections) {
      return computePass;
    }
    this.endComputePass(encoder, computePass);
    return this.beginComputePass(encoder, computePass.profileGpuPass, label, computePass.profileSections);
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
    const maxPasses = TIMESTAMP_MAX_PASSES;
    const byteLength = maxPasses * TIMESTAMP_RESOLVE_STRIDE_BYTES;
    try {
      this.timestampProfiler = {
        querySet: device.createQuerySet({ type: "timestamp", count: maxPasses * 2 }),
        resolveBuffer: device.createBuffer({
          size: byteLength,
          usage: GPU_QUERY_RESOLVE | GPU_COPY_SRC,
        }),
        readbackBuffer: device.createBuffer({
          size: byteLength,
          usage: GPU_MAP_READ | GPU_COPY_DST,
        }),
        maxPasses,
      };
    } catch {
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
    } catch {
      this.activeRunTimestampStatus = "timestamp-readback-failed";
      return;
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

  async prepareTokenIds(tokenIds: readonly number[]): Promise<WebGpuPreparedInput> {
    this.ensureRuntimeResources();
    const runtimeRun = this.beginRuntimeRun();
    try {
      const prepared = await this.prepareGpuInput(tokenIds, false);
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
      const profileGpuPasses = webGpuGpuTimingEnabled() &&
        tokenCount === 1 &&
        (options.computeTopK === true || options.computeSelectedToken === true);
      const prepared = await this.prepareGpuInput(tokenIds, profileGpuPasses);
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
      ? this.segmentEndLayerExclusive === this.manifest.blockCount
        ? 1
        : tokenCount
      : 0;
    const outputByteLength = outputTokenCount * hiddenByteLength;
    let hiddenReadback: WebGpuBufferLike | undefined;
    let topReadback: WebGpuBufferLike | undefined;
    let selectedTokenReadback: WebGpuBufferLike | undefined;
    const profileGpuPass = webGpuGpuTimingEnabled() &&
      (
        options.computeTopK === true ||
        options.computeSelectedToken === true ||
        webGpuGpuDetailedTimingEnabled()
      );

    const perLayerInputsBuffer = this.prepareBatchedPerLayerInputBuffer(options, persistentCleanup);

    const encodeStartMs = nowMs();
    if (readHidden) {
      hiddenReadback = this.arena.device.createBuffer({
        size: outputByteLength,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
    }

    let candidateCount = 0;
    let candidateByteLength = 0;

    try {
      // Match llama.cpp: non-causal attention is evaluated as one physical batch.
      const prefillChunkSize = options.attentionCausal === false ? tokenCount : this.prefillChunkSize;
      for (let chunkStart = 0; chunkStart < tokenCount; chunkStart += prefillChunkSize) {
        const cleanup: GpuResource[] = [];
        const resources: Array<{ destroy: () => void }> = [];
        const encoder = this.arena.device.createCommandEncoder();
        const chunkTokenCount = Math.min(prefillChunkSize, tokenCount - chunkStart);
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

          let compute = this.beginComputePass(encoder, profileGpuPass, `prefill.chunk.${chunkStart}`);
          for (const layer of this.layers) {
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
          const lastHiddenOffset = (chunkTokenCount - 1) * hiddenByteLength;
          const selectedHidden = isLastChunk &&
              this.segmentEndLayerExclusive === this.manifest.blockCount &&
              chunkTokenCount > 1 &&
              (options.computeTopK === true || options.computeSelectedToken === true)
            ? this.createLastTokenView(compute.pass, resources, currentBatch, cleanup, chunkTokenCount)
            : currentBatch;

          if (isLastChunk && options.computeTopK === true) {
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
            selectedTokenBuffer = this.dispatchOutputSelectedToken(compute.pass, selectedHidden, cleanup, resources);
            selectedTokenReadback = this.arena.device.createBuffer({
              size: Uint32Array.BYTES_PER_ELEMENT,
              usage: GPU_MAP_READ | GPU_COPY_DST,
            });
          }

          this.endComputePass(encoder, compute);
          if (hiddenReadback) {
            if (this.segmentEndLayerExclusive === this.manifest.blockCount) {
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
      }

      this.activeRunEncodeMs += nowMs() - encodeStartMs;

      if (!hiddenReadback && !topReadback && !selectedTokenReadback) {
        return {
          hidden: new Float32Array(),
          selectedTokenId: undefined,
          topTokens: undefined,
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
    const kvDim = this.manifest.headCountKv * layer.headSize;
    const kvValueDim = this.manifest.headCountKv * layer.valueSize;

    const attnQ8 = scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.prefill.attn_norm.q8k`);
    dispatchBatchedRmsNormQ8KQuantize(this.arena.device, pass, resources, input, layer.attnNorm.buffer, attnQ8, {
      length: hiddenSize,
      tokenCount,
      epsilon: this.epsilon,
    });

    const qProjection = scratchF32(this.arena, queryDim * tokenCount, cleanup, `blk.${layer.layer}.prefill.q`);
    dispatchKMatMul(pass, resources, layer.q, attnQ8, qProjection, tokenCount);
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
      if (!layerState || !layer.k || !layer.v || !layer.kNorm) {
        throw new Error(`Missing WebGPU KV state or weights for layer ${layer.layer}`);
      }
      const kProjection = scratchF32(this.arena, kvDim * tokenCount, cleanup, `blk.${layer.layer}.prefill.k`);
      const vProjection = scratchF32(this.arena, kvValueDim * tokenCount, cleanup, `blk.${layer.layer}.prefill.v`);
      const dispatchedDualKv = webGpuFusionEnabled() && dispatchDualQ4KMatMul(
        pass,
        resources,
        layer.k,
        layer.v,
        attnQ8,
        kProjection,
        vProjection,
        tokenCount,
      );
      if (!dispatchedDualKv) {
        dispatchKMatMul(pass, resources, layer.k, attnQ8, kProjection, tokenCount);
        dispatchKMatMul(pass, resources, layer.v, attnQ8, vProjection, tokenCount);
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
          headCount: this.manifest.headCountKv,
          headSize: layer.headSize,
          valueSize: layer.valueSize,
          ropeDims: ropeDimensionCount(this.manifest, layer.kind),
          epsilon: this.epsilon,
          freqBase: ropeFreqBase(this.manifest, layer.kind),
          tokenCount,
          contextLength,
          hasFreqFactors: this.hasRopeFreqFactors && layer.kind === "full-attention",
        },
      );
    }

    const state = gpuState.fullAttention.get(layer.hasKv ? layer.layer : layer.kvSourceLayer);
    if (!state) {
      throw new Error(`Missing WebGPU KV state for layer ${layer.layer}`);
    }
    const keyValueTokenCount = Math.min(contextLength, maxInt32(tokenPositions) + 1);
    const attention = scratchF32(this.arena, tokenCount * valueDim, cleanup, `blk.${layer.layer}.prefill.attention`);
    const slidingWindow = options.attentionCausal === false
      ? undefined
      : layer.kind === "sliding-attention" ? this.manifest.slidingWindow : undefined;
    const attentionOptions = {
      headSize: layer.headSize,
      valueSize: layer.valueSize,
      queryHeadCount: this.manifest.headCount,
      keyValueHeadCount: this.manifest.headCountKv,
      keyValueTokenCount,
      contextLength,
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
    const ffnQ8 = scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.prefill.ffn_norm.q8k`);
    dispatchBatchedRmsNormQ8KQuantize(this.arena.device, pass, resources, residual, layer.ffnNorm.buffer, ffnQ8, {
      length: hiddenSize,
      tokenCount,
      epsilon: this.epsilon,
    });
    const gate = scratchF32(this.arena, this.manifest.feedForwardLength * tokenCount, cleanup, `blk.${layer.layer}.prefill.ffn_gate`);
    const up = scratchF32(this.arena, this.manifest.feedForwardLength * tokenCount, cleanup, `blk.${layer.layer}.prefill.ffn_up`);
    const geglu = scratchF32(this.arena, this.manifest.feedForwardLength * tokenCount, cleanup, `blk.${layer.layer}.prefill.ffn_geglu`);
    dispatchKMatMul(pass, resources, layer.ffnGate, ffnQ8, gate, tokenCount);
    dispatchKMatMul(pass, resources, layer.ffnUp, ffnQ8, up, tokenCount);
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
    const output = scratchF32(this.arena, this.manifest.embeddingLength, cleanup, "prefill.last_hidden");
    dispatchTokenSlice(this.arena.device, pass, resources, input, output, {
      rowSize: this.manifest.embeddingLength,
      rowIndex: tokenCount - 1,
    });
    return output;
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

  private async prepareGpuInput(tokenIds: readonly number[], profileGpuPass: boolean): Promise<PreparedGpuInput> {
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
      const compute = this.beginComputePass(encoder, profileGpuPass, "input.prepare");
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
    if (!layerState || !layer.k || !layer.v || !layer.kNorm) {
      throw new Error(`Missing WebGPU KV state or weights for layer ${layer.layer}`);
    }
    const hiddenSize = this.manifest.embeddingLength;
    const tokenCount = 1;
    const kvDim = this.manifest.headCountKv * layer.headSize;
    const kvValueDim = this.manifest.headCountKv * layer.valueSize;
    const attnQ8 = scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.prefill_kv.attn_norm.q8k`);
    this.dispatchRmsNormToQ8K(pass, resources, input, layer.attnNorm.buffer, attnQ8, hiddenSize, cleanup, `blk.${layer.layer}.prefill_kv.attn_norm`);
    const kProjection = scratchF32(this.arena, kvDim, cleanup, `blk.${layer.layer}.prefill_kv.k`);
    const vProjection = scratchF32(this.arena, kvValueDim, cleanup, `blk.${layer.layer}.prefill_kv.v`);
    const dispatchedDualKv = webGpuFusionEnabled() && dispatchDualQ4KMatMul(
      pass,
      resources,
      layer.k,
      layer.v,
      attnQ8,
      kProjection,
      vProjection,
      tokenCount,
    );
    if (!dispatchedDualKv) {
      dispatchKMatMul(pass, resources, layer.k, attnQ8, kProjection, tokenCount);
      dispatchKMatMul(pass, resources, layer.v, attnQ8, vProjection, tokenCount);
    }
    if (webGpuFusionEnabled()) {
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
          headCount: this.manifest.headCountKv,
          headSize: layer.headSize,
          valueSize: layer.valueSize,
          ropeDims: ropeDimensionCount(this.manifest, layer.kind),
          epsilon: this.epsilon,
          freqBase: ropeFreqBase(this.manifest, layer.kind),
          position: mropeTextPosition(positions, tokenPosition),
          tokenPosition,
          contextLength,
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
        headCount: this.manifest.headCountKv,
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
        headCount: this.manifest.headCountKv,
        headSize: layer.headSize,
        ropeDims: ropeDimensionCount(this.manifest, layer.kind),
        freqBase: ropeFreqBase(this.manifest, layer.kind),
        position: mropeTextPosition(positions, tokenPosition),
        tokenPosition,
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
        headCount: this.manifest.headCountKv,
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
        headCount: this.manifest.headCountKv,
        valueSize: layer.valueSize,
        tokenPosition,
        contextLength,
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
    const profileGpuPass = webGpuGpuTimingEnabled() &&
      options.sourceTokenCount === 1 &&
      (options.computeTopK === true || options.computeSelectedToken === true);
    const profileSections = webGpuGpuDetailedTimingEnabled();
    let compute = this.beginComputePass(encoder, profileGpuPass, "decode", profileSections);
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
        compute = this.restartComputePass(
          encoder,
          compute,
          profileSections && shouldProfileLayerDetails(layer.layer) ? `layer.${layer.layer}.attn.norm_quant` : `layer.${layer.layer}`,
        );
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
    const profileDetails = compute.profileSections && shouldProfileLayerDetails(layer.layer);
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
    const kvDim = this.manifest.headCountKv * layer.headSize;
    const kvValueDim = this.manifest.headCountKv * layer.valueSize;

    const attnQ8 = scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.attn_norm.q8k`);
    this.dispatchRmsNormToQ8K(pass, resources, input, layer.attnNorm.buffer, attnQ8, hiddenSize, cleanup, `blk.${layer.layer}.attn_norm`);

    restartLayerSection("attn.q_proj");
    const qProjection = scratchF32(this.arena, queryDim, cleanup, `blk.${layer.layer}.q`);
    dispatchKMatMul(pass, resources, layer.q, attnQ8, qProjection, tokenCount);
    restartLayerSection("attn.q_rope");
    const query = scratchF32(this.arena, queryDim, cleanup, `blk.${layer.layer}.q_rope`);
    if (webGpuFusionEnabled()) {
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
      if (!layerState || !layer.k || !layer.v || !layer.kNorm) {
        throw new Error(`Missing WebGPU KV state or weights for layer ${layer.layer}`);
      }
      const kProjection = scratchF32(this.arena, kvDim, cleanup, `blk.${layer.layer}.k`);
      const vProjection = scratchF32(this.arena, kvValueDim, cleanup, `blk.${layer.layer}.v`);
      restartLayerSection("attn.kv_proj");
      const dispatchedDualKv = webGpuFusionEnabled() && dispatchDualQ4KMatMul(
        pass,
        resources,
        layer.k,
        layer.v,
        attnQ8,
        kProjection,
        vProjection,
        tokenCount,
      );
      if (!dispatchedDualKv) {
        dispatchKMatMul(pass, resources, layer.k, attnQ8, kProjection, tokenCount);
        dispatchKMatMul(pass, resources, layer.v, attnQ8, vProjection, tokenCount);
      }
      restartLayerSection("attn.kv_update");
      if (webGpuFusionEnabled()) {
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
            headCount: this.manifest.headCountKv,
            headSize: layer.headSize,
            valueSize: layer.valueSize,
            ropeDims: ropeDimensionCount(this.manifest, layer.kind),
            epsilon: this.epsilon,
            freqBase: ropeFreqBase(this.manifest, layer.kind),
            position: mropeTextPosition(positions, tokenPosition),
            tokenPosition,
            contextLength,
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
            headCount: this.manifest.headCountKv,
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
            headCount: this.manifest.headCountKv,
            headSize: layer.headSize,
            ropeDims: ropeDimensionCount(this.manifest, layer.kind),
            freqBase: ropeFreqBase(this.manifest, layer.kind),
            position: mropeTextPosition(positions, tokenPosition),
            tokenPosition,
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
            headCount: this.manifest.headCountKv,
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
            headCount: this.manifest.headCountKv,
            valueSize: layer.valueSize,
            tokenPosition,
            contextLength,
          },
        );
      }
    }

    restartLayerSection("attn.score");
    const state = gpuState.fullAttention.get(layer.hasKv ? layer.layer : layer.kvSourceLayer);
    if (!state) {
      throw new Error(`Missing WebGPU KV state for layer ${layer.layer}`);
    }
    const keyValueTokenCount = options.attentionCausal === false
      ? Math.min(contextLength, options.keyValueTokenCount ?? tokenPosition + 1)
      : Math.min(contextLength, tokenPosition + 1);
    const probabilityTokenCapacity = bucketAttentionProbabilityTokenCount(keyValueTokenCount);
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
      keyValueHeadCount: this.manifest.headCountKv,
      keyValueTokenCount,
      contextLength,
      scale: 1,
      tokenPosition,
      slidingWindow: options.attentionCausal === false
        ? undefined
        : layer.kind === "sliding-attention" ? this.manifest.slidingWindow : undefined,
    };
    const keyValueStart = attentionKeyValueStart(tokenPosition, attentionOptions.slidingWindow);
    dispatchFullAttentionScore(this.arena.device, pass, resources, query, state.key, probabilities, attentionOptions);
    restartLayerSection("attn.apply");
    dispatchFullAttentionApply(this.arena.device, pass, resources, state.value, probabilities, attention, {
      ...attentionOptions,
      keyValueStart,
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
    const ffnQ8 = scratchQ8K(this.arena, hiddenSize, 1, cleanup, `blk.${layer.layer}.ffn_norm.q8k`);
    this.dispatchRmsNormToQ8K(activePass, resources, residual, layer.ffnNorm.buffer, ffnQ8, hiddenSize, cleanup, `blk.${layer.layer}.ffn_norm`);
    const gate = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_gate`);
    const up = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_up`);
    const geglu = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_geglu`);
    activePass = restartSection?.("gate") ?? activePass;
    dispatchKMatMul(activePass, resources, layer.ffnGate, ffnQ8, gate, 1);
    activePass = restartSection?.("up") ?? activePass;
    dispatchKMatMul(activePass, resources, layer.ffnUp, ffnQ8, up, 1);
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
    if (handle.type === "Q8_0") {
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
    const q8 = scratchQ8K(this.arena, hiddenSize, 1, cleanup, "output_norm.q8k");
    this.dispatchRmsNormToQ8K(pass, resources, hidden, outputNorm.buffer, q8, hiddenSize, cleanup, "output_norm");

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
      dispatchKMatMul(pass, resources, stripe, q8, logits, 1);
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
    const q8 = scratchQ8K(this.arena, hiddenSize, 1, cleanup, "output_norm.selected.q8k");
    this.dispatchRmsNormToQ8K(pass, resources, hidden, outputNorm.buffer, q8, hiddenSize, cleanup, "output_norm.selected");
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
      dispatchKMatMul(pass, resources, stripe, q8, logits, 1);
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
    if (!computePass.profileSections) {
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
    const q8 = scratchQ8K(this.arena, hiddenSize, 1, cleanup, "output_norm.selected.q8k");
    this.dispatchRmsNormToQ8K(pass, resources, hidden, outputNorm.buffer, q8, hiddenSize, cleanup, "output_norm.selected");
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
      dispatchKMatMul(pass, resources, stripe, q8, logits, 1);
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
      fullAttention.set(layer.layer, {
        key: this.arena.createBuffer(
          `blk.${layer.layer}.gpu.key_cache`,
          state.contextLength * this.manifest.headCountKv * layer.headSize * F16_BYTE_LENGTH,
          GPU_STORAGE,
        ),
        value: this.arena.createBuffer(
          `blk.${layer.layer}.gpu.value_cache`,
          state.contextLength * this.manifest.headCountKv * layer.valueSize * F16_BYTE_LENGTH,
          GPU_STORAGE,
        ),
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
  return type === "F32" || type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "Q8_0";
}

function isF32CompatibleType(type: string): boolean {
  return type === "F32" || type === "F16" || type === "BF16";
}

function isSupportedProjectionType(type: string): boolean {
  return isF32CompatibleType(type) || type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "Q8_0";
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

function shouldProfileLayerDetails(layer: number): boolean {
  return layer < 3 || layer % 6 === 5;
}

function outputTop1CandidateCount(outputStripes: readonly OutputStripe[]): number {
  return outputStripes.reduce((sum, stripe) => sum + Math.ceil(stripe.rowCount / TOP1_CHUNK_SIZE), 0);
}

function attentionKeyValueStart(tokenPosition: number, slidingWindow: number | undefined): number {
  return slidingWindow === undefined ? 0 : Math.max(0, tokenPosition + 1 - slidingWindow);
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

function webGpuGpuTimingEnabled(): boolean {
  return (globalThis as { __heliopauseDisableWebGpuGpuTiming?: unknown }).__heliopauseDisableWebGpuGpuTiming !== true;
}

function webGpuGpuDetailedTimingEnabled(): boolean {
  return (globalThis as { __heliopauseEnableWebGpuDetailedTimings?: unknown }).__heliopauseEnableWebGpuDetailedTimings === true;
}

function webGpuFusionEnabled(): boolean {
  return (globalThis as { __heliopauseDisableWebGpuFusion?: unknown }).__heliopauseDisableWebGpuFusion !== true;
}
