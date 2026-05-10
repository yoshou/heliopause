import type { GgufTensorReader } from "../../tensor-reader";
import type { Gemma4ModelManifest } from "../../model";
import { GPU_COPY_DST, GPU_COPY_SRC, GPU_MAP_READ, GPU_STORAGE, GEMMA4_WEBGPU_MEMORY_LIMIT_BYTES } from "./gpu-constants";
import { webGpuDevice } from "./gpu-device";
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
  dispatchElementwiseMul,
  dispatchF16Cast,
  dispatchF32MatMul,
  dispatchFullAttentionApply,
  dispatchFullAttentionScore,
  dispatchFullKvUpdate,
  dispatchFullQuery,
  dispatchGeglu,
  dispatchGelu,
  dispatchKMatMul,
  dispatchQ8_0MatMul,
  dispatchQ8_0Quantize,
  dispatchQ8KQuantize,
  dispatchResidualAdd,
  dispatchRmsNorm,
  dispatchScale,
  dispatchTokenSlice,
  dispatchTopK,
} from "./dispatch";
import {
  loadF32Handle,
  loadGpuLayer,
  loadOutputStripes,
  type Gemma4GpuLayer,
  type OutputStripe,
} from "./segment-layer-loader";
import type { WebGpuBufferLike, WebGpuComputePassLike, WebGpuTopToken } from "./gpu-types";

export type Gemma4WebGpuSegmentRunnerOptions = {
  tensorReader: GgufTensorReader;
  manifest: Gemma4ModelManifest;
  epsilon: number;
  contextLength: number;
  memoryLimitBytes?: number;
  segmentStartLayer: number;
  segmentEndLayerExclusive?: number;
  loadOutput?: boolean;
};

export type Gemma4WebGpuStateLike = {
  contextLength: number;
  nextPosition: number;
};

export type Gemma4WebGpuTokenResult = {
  topTokens?: WebGpuTopToken[];
};

export type Gemma4WebGpuHiddenResult = {
  hidden: Float32Array;
  topTokens?: WebGpuTopToken[];
};

export type Gemma4WebGpuRunOptions = {
  computeTopK?: boolean;
  topK?: number;
  perLayerInputs?: Float32Array;
};

type FullAttentionGpuLayerState = {
  key: WebGpuBufferLike;
  value: WebGpuBufferLike;
};

type GpuState = {
  fullAttention: Map<number, FullAttentionGpuLayerState>;
};

export class Gemma4WebGpuSegmentRunner {
  readonly segmentStartLayer: number;
  readonly segmentEndLayerExclusive: number;

  private readonly states = new WeakMap<object, GpuState>();
  private readonly arena: GpuMemoryArena;
  private readonly manifest: Gemma4ModelManifest;
  private readonly epsilon: number;
  private readonly layers: Gemma4GpuLayer[];
  private readonly outputNorm?: F32Handle;
  private readonly outputStripes?: OutputStripe[];
  private readonly ropeFreqFactors: F32Handle;
  private readonly hasRopeFreqFactors: boolean;

  private constructor(
    arena: GpuMemoryArena,
    manifest: Gemma4ModelManifest,
    epsilon: number,
    layers: Gemma4GpuLayer[],
    ropeFreqFactors: F32Handle,
    hasRopeFreqFactors: boolean,
    outputNorm: F32Handle | undefined,
    outputStripes: OutputStripe[] | undefined,
    segmentStartLayer: number,
    segmentEndLayerExclusive: number,
  ) {
    this.arena = arena;
    this.manifest = manifest;
    this.epsilon = epsilon;
    this.layers = layers;
    this.ropeFreqFactors = ropeFreqFactors;
    this.hasRopeFreqFactors = hasRopeFreqFactors;
    this.outputNorm = outputNorm;
    this.outputStripes = outputStripes;
    this.segmentStartLayer = segmentStartLayer;
    this.segmentEndLayerExclusive = segmentEndLayerExclusive;
  }

  static async create(options: Gemma4WebGpuSegmentRunnerOptions): Promise<Gemma4WebGpuSegmentRunner> {
    const device = await webGpuDevice();
    if (!device) {
      throw new Error("WebGPU is not available for Gemma4 segment execution.");
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

    const arena = new GpuMemoryArena(device, options.memoryLimitBytes ?? GEMMA4_WEBGPU_MEMORY_LIMIT_BYTES);
    const layers: Gemma4GpuLayer[] = [];
    for (let layer = segmentStartLayer; layer < segmentEndLayerExclusive; layer += 1) {
      layers.push(await loadGpuLayer(arena, options.tensorReader, options.manifest, layer));
    }
    const { handle: ropeFreqFactors, present: hasRopeFreqFactors } = await loadRopeFreqFactors(arena, options.tensorReader);
    const loadOutput = options.loadOutput ?? true;
    const outputNorm = loadOutput
      ? await loadF32Handle(arena, options.tensorReader, "output_norm.weight")
      : undefined;
    const outputStripes = loadOutput
      ? await loadOutputStripes(arena, options.tensorReader, options.manifest)
      : undefined;

    return new Gemma4WebGpuSegmentRunner(
      arena,
      options.manifest,
      options.epsilon,
      layers,
      ropeFreqFactors,
      hasRopeFreqFactors,
      outputNorm,
      outputStripes,
      segmentStartLayer,
      segmentEndLayerExclusive,
    );
  }

  get residentBytes(): number {
    return this.arena.residentBytes;
  }

  async runToken(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: Gemma4WebGpuStateLike,
    options: Gemma4WebGpuRunOptions = {},
  ): Promise<Gemma4WebGpuTokenResult> {
    if (inputHidden.length !== this.manifest.embeddingLength) {
      throw new Error(`WebGPU segment input shape mismatch: ${inputHidden.length}`);
    }
    const boundary = this.arena.createBuffer(
      "segment boundary hidden",
      inputHidden.byteLength,
      GPU_STORAGE | GPU_COPY_DST,
    );
    this.arena.device.queue.writeBuffer(boundary, 0, inputHidden);
    try {
      return await this.runTokenFromBoundary(boundary, 0, positions, state, {
        ...options,
        sourceTokenCount: 1,
        sourceTokenIndex: 0,
      });
    } finally {
      boundary.destroy?.();
    }
  }

  async runTokenHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: Gemma4WebGpuStateLike,
    options: Gemma4WebGpuRunOptions = {},
  ): Promise<Gemma4WebGpuHiddenResult> {
    if (inputHidden.length !== this.manifest.embeddingLength) {
      throw new Error(`WebGPU segment input shape mismatch: ${inputHidden.length}`);
    }
    const boundary = this.arena.createBuffer(
      "segment boundary hidden",
      inputHidden.byteLength,
      GPU_STORAGE | GPU_COPY_DST,
    );
    this.arena.device.queue.writeBuffer(boundary, 0, inputHidden);
    try {
      return await this.runTokenFromBoundaryHidden(boundary, 0, positions, state, {
        ...options,
        sourceTokenCount: 1,
        sourceTokenIndex: 0,
      });
    } finally {
      boundary.destroy?.();
    }
  }

  async runTokens(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: Gemma4WebGpuStateLike,
    options: Gemma4WebGpuRunOptions = {},
  ): Promise<Gemma4WebGpuTokenResult> {
    const tokenCount = this.assertBatchedHidden(inputHidden, positions);
    const boundary = this.arena.createBuffer(
      "segment boundary hidden batch",
      inputHidden.byteLength,
      GPU_STORAGE | GPU_COPY_DST,
    );
    this.arena.device.queue.writeBuffer(boundary, 0, inputHidden);
    let topTokens: WebGpuTopToken[] | undefined;
    try {
      for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
        const tokenPositions = tokenPositionsFromBatch(positions, tokenIndex, tokenCount);
        const computeTopK = options.computeTopK === true && tokenIndex === tokenCount - 1;
        const result = await this.runTokenFromBoundary(boundary, tokenIndex, tokenPositions, state, {
          ...options,
          computeTopK,
          sourceTokenCount: tokenCount,
          sourceTokenIndex: tokenIndex,
        });
        if (computeTopK) {
          topTokens = result.topTokens;
        }
      }
    } finally {
      boundary.destroy?.();
    }
    return { topTokens };
  }

  async runTokensHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: Gemma4WebGpuStateLike,
    options: Gemma4WebGpuRunOptions = {},
  ): Promise<Gemma4WebGpuHiddenResult> {
    const tokenCount = this.assertBatchedHidden(inputHidden, positions);
    const boundary = this.arena.createBuffer(
      "segment boundary hidden batch",
      inputHidden.byteLength,
      GPU_STORAGE | GPU_COPY_DST,
    );
    this.arena.device.queue.writeBuffer(boundary, 0, inputHidden);
    const outputTokenCount = this.segmentEndLayerExclusive === this.manifest.blockCount && tokenCount > 1
      ? 1
      : tokenCount;
    const hidden = new Float32Array(outputTokenCount * this.manifest.embeddingLength);
    let topTokens: WebGpuTopToken[] | undefined;
    try {
      for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
        const tokenPositions = tokenPositionsFromBatch(positions, tokenIndex, tokenCount);
        const computeTopK = options.computeTopK === true && tokenIndex === tokenCount - 1;
        const result = await this.runTokenFromBoundaryHidden(boundary, tokenIndex, tokenPositions, state, {
          ...options,
          computeTopK,
          sourceTokenCount: tokenCount,
          sourceTokenIndex: tokenIndex,
        });
        if (outputTokenCount === tokenCount) {
          hidden.set(result.hidden, tokenIndex * this.manifest.embeddingLength);
        } else if (tokenIndex === tokenCount - 1) {
          hidden.set(result.hidden);
        }
        if (computeTopK) {
          topTokens = result.topTokens;
        }
      }
    } finally {
      boundary.destroy?.();
    }
    return { hidden, topTokens };
  }

  private async runTokenFromBoundary(
    boundary: WebGpuBufferLike,
    tokenIndex: number,
    positions: Int32Array,
    state: Gemma4WebGpuStateLike,
    options: Gemma4WebGpuRunOptions & { sourceTokenCount: number; sourceTokenIndex: number },
  ): Promise<Gemma4WebGpuTokenResult> {
    const result = await this.runTokenFromBoundaryInternal(boundary, tokenIndex, positions, state, options, false);
    return { topTokens: result.topTokens };
  }

  private async runTokenFromBoundaryHidden(
    boundary: WebGpuBufferLike,
    tokenIndex: number,
    positions: Int32Array,
    state: Gemma4WebGpuStateLike,
    options: Gemma4WebGpuRunOptions & { sourceTokenCount: number; sourceTokenIndex: number },
  ): Promise<Gemma4WebGpuHiddenResult> {
    return this.runTokenFromBoundaryInternal(boundary, tokenIndex, positions, state, options, true);
  }

  private async runTokenFromBoundaryInternal(
    boundary: WebGpuBufferLike,
    tokenIndex: number,
    positions: Int32Array,
    state: Gemma4WebGpuStateLike,
    options: Gemma4WebGpuRunOptions & { sourceTokenCount: number; sourceTokenIndex: number },
    readHidden: boolean,
  ): Promise<Gemma4WebGpuHiddenResult> {
    const tokenPosition = tokenPositionFromSingleMrope(positions);
    if (tokenPosition < 0 || tokenPosition >= state.contextLength) {
      throw new Error(`Position ${tokenPosition} is outside context length ${state.contextLength}`);
    }
    const gpuState = this.ensureGpuState(state);
    const cleanup: GpuResource[] = [];
    const resources: Array<{ destroy: () => void }> = [];
    const encoder = this.arena.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    let hiddenReadback: WebGpuBufferLike | undefined;
    let topReadback: WebGpuBufferLike | undefined;

    try {
      let current = scratchF32(this.arena, this.manifest.embeddingLength, cleanup, "segment.hidden.token");
      dispatchTokenSlice(this.arena.device, pass, resources, boundary, current, {
        rowSize: this.manifest.embeddingLength,
        rowIndex: tokenIndex,
      });

      for (const layer of this.layers) {
        current = this.dispatchLayer(pass, layer, gpuState, current, positions, tokenPosition, state.contextLength, options, cleanup, resources);
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
      if (options.computeTopK === true) {
        candidateCount = Math.max(1, options.topK ?? 1);
        const outputStripes = this.requireOutputStripes();
        candidateByteLength = outputStripes.length * candidateCount * 2 * Float32Array.BYTES_PER_ELEMENT;
        topBuffer = this.dispatchOutputTopK(pass, current, candidateCount, cleanup, resources);
        topReadback = this.arena.device.createBuffer({
          size: candidateByteLength,
          usage: GPU_MAP_READ | GPU_COPY_DST,
        });
      }

      pass.end();
      if (hiddenReadback) {
        encoder.copyBufferToBuffer(current, 0, hiddenReadback, 0, hiddenByteLength);
      }
      if (topReadback && topBuffer) {
        encoder.copyBufferToBuffer(topBuffer, 0, topReadback, 0, candidateByteLength);
      }
      this.arena.device.queue.submit([encoder.finish()]);
      await this.arena.device.queue.onSubmittedWorkDone?.();

      const hidden = new Float32Array(this.manifest.embeddingLength);
      if (hiddenReadback) {
        await hiddenReadback.mapAsync(GPU_MAP_READ);
        hidden.set(new Float32Array(hiddenReadback.getMappedRange()).slice());
        hiddenReadback.unmap();
        hiddenReadback.destroy?.();
        hiddenReadback = undefined;
      }

      let topTokens: WebGpuTopToken[] | undefined;
      if (topReadback) {
        await topReadback.mapAsync(GPU_MAP_READ);
        const values = new Float32Array(topReadback.getMappedRange()).slice();
        topReadback.unmap();
        topReadback.destroy?.();
        topReadback = undefined;
        topTokens = mergeTopCandidates(values, candidateCount, this.manifest.finalLogitSoftcap);
      }

      return { hidden, topTokens };
    } finally {
      hiddenReadback?.destroy?.();
      topReadback?.destroy?.();
      for (const resource of resources) {
        resource.destroy();
      }
      for (const item of cleanup.reverse()) {
        item.destroy?.();
      }
    }
  }

  private dispatchLayer(
    pass: WebGpuComputePassLike,
    layer: Gemma4GpuLayer,
    gpuState: GpuState,
    input: WebGpuBufferLike,
    positions: Int32Array,
    tokenPosition: number,
    contextLength: number,
    options: Gemma4WebGpuRunOptions & { sourceTokenCount: number; sourceTokenIndex: number },
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const hiddenSize = this.manifest.embeddingLength;
    const tokenCount = 1;
    const queryDim = this.manifest.headCount * layer.headSize;
    const valueDim = this.manifest.headCount * layer.valueSize;
    const kvDim = this.manifest.headCountKv * layer.headSize;
    const kvValueDim = this.manifest.headCountKv * layer.valueSize;

    const attnNorm = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.attn_norm`);
    dispatchRmsNorm(this.arena.device, pass, resources, input, layer.attnNorm.buffer, attnNorm, hiddenSize, this.epsilon);
    const attnQ8 = scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.attn_norm.q8k`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, attnNorm, attnQ8, hiddenSize, tokenCount);

    const qProjection = scratchF32(this.arena, queryDim, cleanup, `blk.${layer.layer}.q`);
    dispatchKMatMul(pass, resources, layer.q, attnQ8, qProjection, tokenCount);
    const query = scratchF32(this.arena, queryDim, cleanup, `blk.${layer.layer}.q_rope`);
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

    if (layer.hasKv) {
      const layerState = gpuState.fullAttention.get(layer.layer);
      if (!layerState || !layer.k || !layer.v || !layer.kNorm) {
        throw new Error(`Missing WebGPU KV state or weights for layer ${layer.layer}`);
      }
      const kProjection = scratchF32(this.arena, kvDim, cleanup, `blk.${layer.layer}.k`);
      const vProjection = scratchF32(this.arena, kvValueDim, cleanup, `blk.${layer.layer}.v`);
      dispatchKMatMul(pass, resources, layer.k, attnQ8, kProjection, tokenCount);
      dispatchKMatMul(pass, resources, layer.v, attnQ8, vProjection, tokenCount);
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
    }

    const state = gpuState.fullAttention.get(layer.hasKv ? layer.layer : layer.kvSourceLayer);
    if (!state) {
      throw new Error(`Missing WebGPU KV state for layer ${layer.layer}`);
    }
    const keyValueTokenCount = Math.min(contextLength, tokenPosition + 1);
    const probabilities = scratchF32(
      this.arena,
      this.manifest.headCount * keyValueTokenCount,
      cleanup,
      `blk.${layer.layer}.attention_probabilities`,
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
      slidingWindow: layer.kind === "sliding-attention" ? this.manifest.slidingWindow : undefined,
    };
    dispatchFullAttentionScore(this.arena.device, pass, resources, query, state.key, probabilities, attentionOptions);
    dispatchFullAttentionApply(this.arena.device, pass, resources, state.value, probabilities, attention, attentionOptions);

    const attentionForOutput = scratchF32(this.arena, valueDim, cleanup, `blk.${layer.layer}.attention.f16`);
    dispatchF16Cast(this.arena.device, pass, resources, attention, attentionForOutput, valueDim);
    const attentionOut = this.dispatchQuantizedMatMul(pass, resources, layer.attnOut, attentionForOutput, tokenCount, cleanup, `blk.${layer.layer}.attention_out`);
    const attentionPostNorm = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.attention_post_norm`);
    dispatchRmsNorm(this.arena.device, pass, resources, attentionOut, layer.postAttentionNorm.buffer, attentionPostNorm, hiddenSize, this.epsilon);
    const attentionResidual = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.attention_residual`);
    dispatchResidualAdd(this.arena.device, pass, resources, input, attentionPostNorm, attentionResidual, hiddenSize);

    const ffn = this.dispatchFfn(pass, layer, attentionResidual, cleanup, resources);
    const enriched = this.dispatchPerLayerInput(pass, layer, ffn, options, cleanup, resources);
    const scaled = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.scaled`);
    dispatchScale(this.arena.device, pass, resources, enriched, layer.layerOutputScale.buffer, scaled, hiddenSize);
    return scaled;
  }

  private dispatchFfn(
    pass: WebGpuComputePassLike,
    layer: Gemma4GpuLayer,
    residual: WebGpuBufferLike,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const hiddenSize = this.manifest.embeddingLength;
    const ffnNorm = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.ffn_norm`);
    dispatchRmsNorm(this.arena.device, pass, resources, residual, layer.ffnNorm.buffer, ffnNorm, hiddenSize, this.epsilon);
    const ffnQ8 = scratchQ8K(this.arena, hiddenSize, 1, cleanup, `blk.${layer.layer}.ffn_norm.q8k`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, ffnNorm, ffnQ8, hiddenSize, 1);
    const gate = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_gate`);
    const up = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_up`);
    const geglu = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_geglu`);
    dispatchKMatMul(pass, resources, layer.ffnGate, ffnQ8, gate, 1);
    dispatchKMatMul(pass, resources, layer.ffnUp, ffnQ8, up, 1);
    dispatchGeglu(this.arena.device, pass, resources, gate, up, geglu, this.manifest.feedForwardLength);
    const ffnOut = this.dispatchQuantizedMatMul(pass, resources, layer.ffnDown, geglu, 1, cleanup, `blk.${layer.layer}.ffn_out`);
    const postNorm = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.ffn_post_norm`);
    dispatchRmsNorm(this.arena.device, pass, resources, ffnOut, layer.postFfwNorm.buffer, postNorm, hiddenSize, this.epsilon);
    const output = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.ffn_residual`);
    dispatchResidualAdd(this.arena.device, pass, resources, residual, postNorm, output, hiddenSize);
    return output;
  }

  private dispatchPerLayerInput(
    pass: WebGpuComputePassLike,
    layer: Gemma4GpuLayer,
    input: WebGpuBufferLike,
    options: Gemma4WebGpuRunOptions & { sourceTokenCount: number; sourceTokenIndex: number },
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    if (this.manifest.perLayerEmbeddingLength <= 0) {
      return input;
    }
    if (!options.perLayerInputs || !layer.perLayerInputGate || !layer.perLayerProjection || !layer.postNorm) {
      throw new Error("WebGPU Gemma4 per-layer input requires prepared per-layer inputs and weights.");
    }
    const perLayerLength = this.manifest.perLayerEmbeddingLength;
    const sourceOffset = (layer.layer * options.sourceTokenCount + options.sourceTokenIndex) * perLayerLength;
    const perLayerSlice = options.perLayerInputs.slice(sourceOffset, sourceOffset + perLayerLength);
    if (perLayerSlice.length !== perLayerLength) {
      throw new Error(`WebGPU per-layer input shape mismatch for layer ${layer.layer}: ${perLayerSlice.length}`);
    }
    const perLayerBuffer = this.arena.createBuffer(
      `blk.${layer.layer}.per_layer_input`,
      perLayerSlice.byteLength,
      GPU_STORAGE | GPU_COPY_DST,
    );
    this.arena.device.queue.writeBuffer(perLayerBuffer, 0, perLayerSlice);
    cleanup.push(perLayerBuffer);

    const gate = scratchF32(this.arena, perLayerLength, cleanup, `blk.${layer.layer}.inp_gate`);
    const activated = scratchF32(this.arena, perLayerLength, cleanup, `blk.${layer.layer}.inp_gate_gelu`);
    const mixed = scratchF32(this.arena, perLayerLength, cleanup, `blk.${layer.layer}.inp_mixed`);
    dispatchF32MatMul(this.arena.device, pass, resources, layer.perLayerInputGate.buffer, input, gate, this.manifest.embeddingLength, perLayerLength, 1);
    dispatchGelu(this.arena.device, pass, resources, gate, activated, perLayerLength);
    dispatchElementwiseMul(this.arena.device, pass, resources, activated, perLayerBuffer, mixed, perLayerLength);
    const projected = scratchF32(this.arena, this.manifest.embeddingLength, cleanup, `blk.${layer.layer}.inp_projected`);
    dispatchF32MatMul(this.arena.device, pass, resources, layer.perLayerProjection.buffer, mixed, projected, perLayerLength, this.manifest.embeddingLength, 1);
    const norm = scratchF32(this.arena, this.manifest.embeddingLength, cleanup, `blk.${layer.layer}.inp_post_norm`);
    dispatchRmsNorm(this.arena.device, pass, resources, projected, layer.postNorm.buffer, norm, this.manifest.embeddingLength, this.epsilon);
    const output = scratchF32(this.arena, this.manifest.embeddingLength, cleanup, `blk.${layer.layer}.inp_residual`);
    dispatchResidualAdd(this.arena.device, pass, resources, input, norm, output, this.manifest.embeddingLength);
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
    const norm = scratchF32(this.arena, hiddenSize, cleanup, "output_norm");
    dispatchRmsNorm(this.arena.device, pass, resources, hidden, outputNorm.buffer, norm, hiddenSize, this.epsilon);
    const q8 = scratchQ8K(this.arena, hiddenSize, 1, cleanup, "output_norm.q8k");
    dispatchQ8KQuantize(this.arena.device, pass, resources, norm, q8, hiddenSize, 1);

    const candidates = this.arena.createBuffer(
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

  private ensureGpuState(state: Gemma4WebGpuStateLike): GpuState {
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
          state.contextLength * this.manifest.headCountKv * layer.headSize * Float32Array.BYTES_PER_ELEMENT,
          GPU_STORAGE,
        ),
        value: this.arena.createBuffer(
          `blk.${layer.layer}.gpu.value_cache`,
          state.contextLength * this.manifest.headCountKv * layer.valueSize * Float32Array.BYTES_PER_ELEMENT,
          GPU_STORAGE,
        ),
      });
    }
    const created = { fullAttention };
    this.states.set(key, created);
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

function mropeTextPosition(positions: Int32Array, fallback: number): number {
  return positions[0] ?? fallback;
}

function ropeDimensionCount(manifest: Gemma4ModelManifest, kind: Gemma4GpuLayer["kind"]): number {
  return kind === "sliding-attention"
    ? manifest.rope.slidingDimensionCount
    : manifest.rope.fullDimensionCount;
}

function ropeFreqBase(manifest: Gemma4ModelManifest, kind: Gemma4GpuLayer["kind"]): number {
  return kind === "sliding-attention"
    ? manifest.rope.slidingFreqBase
    : manifest.rope.fullFreqBase;
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
