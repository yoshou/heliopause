import type { GgufTensorReader } from "../tensor-reader";
import type { Qwen35ModelManifest } from "../qwen35";
import { GPU_COPY_DST, GPU_COPY_SRC, GPU_MAP_READ, GPU_STORAGE, QWEN35_WEBGPU_MEMORY_LIMIT_BYTES } from "./gpu-constants";
import { webGpuDevice } from "./gpu-device";
import { planQwen35WebGpuHybrid } from "./planning";
import { GpuMemoryArena, scratchF32, scratchQ8_0, scratchQ8K, type F32Handle, type GpuResource } from "./arena";
import { dispatchDeltaGate, dispatchF32MatMul, dispatchFullAttentionApply, dispatchFullAttentionScore, dispatchFullKvUpdate, dispatchFullQuery, dispatchGatedDeltaNet, dispatchKMatMul, dispatchQkvConv, dispatchQ8_0MatMul, dispatchQ8_0Quantize, dispatchQ8KQuantize, dispatchResidualAdd, dispatchRmsNorm, dispatchSsmNormGate, dispatchSwiGlu, dispatchTokenSlice, dispatchTopK } from "./dispatch";
import { loadF32Handle, loadGpuLayer, loadOutputStripes, type FullAttentionGpuLayer, type GpuLayer, type OutputStripe, type RecurrentGpuLayer } from "./suffix-layer-loader";
import type { WebGpuBufferLike, WebGpuComputePassLike, WebGpuTopToken } from "./gpu-types";

export type Qwen35WebGpuSuffixRunnerOptions = {
  tensorReader: GgufTensorReader;
  manifest: Qwen35ModelManifest;
  epsilon: number;
  contextLength: number;
  memoryLimitBytes?: number;
  firstGpuLayer?: number;
};

export type Qwen35WebGpuStateLike = {
  contextLength: number;
  nextPosition: number;
};

export type Qwen35WebGpuTokenResult = {
  topTokens?: WebGpuTopToken[];
};

type RecurrentGpuLayerState = {
  conv: WebGpuBufferLike;
  recurrent: WebGpuBufferLike;
};

type FullAttentionGpuLayerState = {
  key: WebGpuBufferLike;
  value: WebGpuBufferLike;
};

type GpuState = {
  recurrent: Map<number, RecurrentGpuLayerState>;
  fullAttention: Map<number, FullAttentionGpuLayerState>;
};

export class Qwen35WebGpuSuffixRunner {
  readonly firstGpuLayer: number;

  private readonly states = new WeakMap<object, GpuState>();
  private readonly arena: GpuMemoryArena;
  private readonly manifest: Qwen35ModelManifest;
  private readonly epsilon: number;
  private readonly layers: GpuLayer[];
  private readonly outputNorm: F32Handle;
  private readonly outputStripes: OutputStripe[];

  private constructor(
    arena: GpuMemoryArena,
    manifest: Qwen35ModelManifest,
    epsilon: number,
    layers: GpuLayer[],
    outputNorm: F32Handle,
    outputStripes: OutputStripe[],
    firstGpuLayer: number,
  ) {
    this.arena = arena;
    this.manifest = manifest;
    this.epsilon = epsilon;
    this.layers = layers;
    this.outputNorm = outputNorm;
    this.outputStripes = outputStripes;
    this.firstGpuLayer = firstGpuLayer;
  }

  static async create(options: Qwen35WebGpuSuffixRunnerOptions): Promise<Qwen35WebGpuSuffixRunner> {
    const device = await webGpuDevice();
    if (!device) {
      throw new Error("WebGPU is not available for Qwen3.5 suffix execution.");
    }
    const memoryLimitBytes = options.memoryLimitBytes ?? QWEN35_WEBGPU_MEMORY_LIMIT_BYTES;
    const plan = planQwen35WebGpuHybrid(
      options.tensorReader.metadata,
      options.manifest,
      {
        mode: "enabled",
        browserGate: "passed",
        contextLength: options.contextLength,
        memoryLimitBytes,
      },
    );
    const firstGpuLayer = options.firstGpuLayer ?? plan.firstGpuLayer;
    if (firstGpuLayer === undefined || firstGpuLayer >= options.manifest.blockCount) {
      throw new Error("WebGPU suffix planning selected no layers.");
    }
    if (plan.estimatedResidentBytes > memoryLimitBytes) {
      throw new Error(
        `WebGPU suffix plan exceeds memory cap: ${plan.estimatedResidentBytes} > ${memoryLimitBytes}`,
      );
    }

    const arena = new GpuMemoryArena(device, memoryLimitBytes);
    const layers: GpuLayer[] = [];
    for (let layer = firstGpuLayer; layer < options.manifest.blockCount; layer += 1) {
      layers.push(await loadGpuLayer(arena, options.tensorReader, options.manifest, layer));
    }
    const outputNorm = await loadF32Handle(arena, options.tensorReader, "output_norm.weight");
    const outputStripes = await loadOutputStripes(arena, options.tensorReader, options.manifest);
    return new Qwen35WebGpuSuffixRunner(
      arena,
      options.manifest,
      options.epsilon,
      layers,
      outputNorm,
      outputStripes,
      firstGpuLayer,
    );
  }

  get residentBytes(): number {
    return this.arena.residentBytes;
  }

  async runToken(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: Qwen35WebGpuStateLike,
    options: { computeTopK?: boolean; topK?: number } = {},
  ): Promise<Qwen35WebGpuTokenResult> {
    if (inputHidden.length !== this.manifest.embeddingLength) {
      throw new Error(`WebGPU suffix input shape mismatch: ${inputHidden.length}`);
    }
    const tokenPosition = tokenPositionFromSingleMrope(positions);
    const mropePosition = singleMropePosition(positions);
    const gpuState = this.ensureGpuState(state);
    const cleanup: GpuResource[] = [];
    const resources: Array<{ destroy: () => void }> = [];
    const encoder = this.arena.device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    let current = this.arena.createBuffer(
      "suffix boundary hidden",
      inputHidden.byteLength,
      GPU_STORAGE | GPU_COPY_DST,
    );
    cleanup.push(current);
    this.arena.device.queue.writeBuffer(current, 0, inputHidden);

    for (const layer of this.layers) {
      current = layer.kind === "recurrent"
        ? this.dispatchRecurrentLayer(pass, layer, gpuState, current, cleanup, resources)
        : this.dispatchFullAttentionLayer(
          pass,
          layer,
          gpuState,
          current,
          tokenPosition,
          mropePosition,
          state.contextLength,
          cleanup,
          resources,
        );
    }

    let topBuffer: WebGpuBufferLike | undefined;
    let topReadback: WebGpuBufferLike | undefined;
    const candidateCount = Math.max(1, options.topK ?? 1);
    const candidateByteLength = this.outputStripes.length * candidateCount * 2 * Float32Array.BYTES_PER_ELEMENT;
    if (options.computeTopK) {
      topBuffer = this.dispatchOutputTopK(pass, current, candidateCount, cleanup, resources);
      topReadback = this.arena.device.createBuffer({
        size: candidateByteLength,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
    }

    pass.end();
    if (topBuffer && topReadback) {
      encoder.copyBufferToBuffer(topBuffer, 0, topReadback, 0, candidateByteLength);
    }
    this.arena.device.queue.submit([encoder.finish()]);
    await this.arena.device.queue.onSubmittedWorkDone?.();

    let topTokens: WebGpuTopToken[] | undefined;
    if (topReadback) {
      await topReadback.mapAsync(GPU_MAP_READ);
      const values = new Float32Array(topReadback.getMappedRange()).slice();
      topReadback.unmap();
      topReadback.destroy?.();
      topTokens = mergeTopCandidates(values, candidateCount);
    }

    for (const resource of resources) {
      resource.destroy();
    }
    for (const item of cleanup.reverse()) {
      item.destroy?.();
    }
    return { topTokens };
  }

  async runTokens(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: Qwen35WebGpuStateLike,
    options: { computeTopK?: boolean; topK?: number } = {},
  ): Promise<Qwen35WebGpuTokenResult> {
    const tokenCount = inputHidden.length / this.manifest.embeddingLength;
    if (!Number.isInteger(tokenCount) || tokenCount <= 0) {
      throw new Error(`WebGPU suffix batched input shape mismatch: ${inputHidden.length}`);
    }
    if (positions.length !== tokenCount && positions.length !== tokenCount * 4) {
      throw new Error(`WebGPU suffix batched position shape mismatch: ${positions.length}`);
    }

    const boundary = this.arena.createBuffer(
      "suffix boundary hidden batch",
      inputHidden.byteLength,
      GPU_STORAGE | GPU_COPY_DST,
    );
    this.arena.device.queue.writeBuffer(boundary, 0, inputHidden);
    let topTokens: WebGpuTopToken[] | undefined;
    try {
      for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
        const tokenPositions = tokenPositionsFromBatch(positions, tokenIndex, tokenCount);
        const computeTopK = options.computeTopK === true && tokenIndex === tokenCount - 1;
        const result = await this.runTokenFromBoundary(
          boundary,
          tokenIndex,
          tokenPositions,
          state,
          {
            computeTopK,
            topK: options.topK,
          },
        );
        if (computeTopK) {
          topTokens = result.topTokens;
        }
      }
    } finally {
      boundary.destroy?.();
    }
    return { topTokens };
  }

  private async runTokenFromBoundary(
    boundary: WebGpuBufferLike,
    tokenIndex: number,
    positions: Int32Array,
    state: Qwen35WebGpuStateLike,
    options: { computeTopK?: boolean; topK?: number } = {},
  ): Promise<Qwen35WebGpuTokenResult> {
    const tokenPosition = tokenPositionFromSingleMrope(positions);
    const mropePosition = singleMropePosition(positions);
    const gpuState = this.ensureGpuState(state);
    const cleanup: GpuResource[] = [];
    const resources: Array<{ destroy: () => void }> = [];
    const encoder = this.arena.device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    let current = this.arena.createBuffer(
      "suffix boundary hidden token",
      this.manifest.embeddingLength * Float32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE,
    );
    cleanup.push(current);
    dispatchTokenSlice(
      this.arena.device,
      pass,
      resources,
      boundary,
      current,
      {
        rowSize: this.manifest.embeddingLength,
        rowIndex: tokenIndex,
      },
    );

    for (const layer of this.layers) {
      current = layer.kind === "recurrent"
        ? this.dispatchRecurrentLayer(pass, layer, gpuState, current, cleanup, resources)
        : this.dispatchFullAttentionLayer(
          pass,
          layer,
          gpuState,
          current,
          tokenPosition,
          mropePosition,
          state.contextLength,
          cleanup,
          resources,
        );
    }

    let topBuffer: WebGpuBufferLike | undefined;
    let topReadback: WebGpuBufferLike | undefined;
    const candidateCount = Math.max(1, options.topK ?? 1);
    const candidateByteLength = this.outputStripes.length * candidateCount * 2 * Float32Array.BYTES_PER_ELEMENT;
    if (options.computeTopK) {
      topBuffer = this.dispatchOutputTopK(pass, current, candidateCount, cleanup, resources);
      topReadback = this.arena.device.createBuffer({
        size: candidateByteLength,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
    }

    pass.end();
    if (topBuffer && topReadback) {
      encoder.copyBufferToBuffer(topBuffer, 0, topReadback, 0, candidateByteLength);
    }
    this.arena.device.queue.submit([encoder.finish()]);
    await this.arena.device.queue.onSubmittedWorkDone?.();

    let topTokens: WebGpuTopToken[] | undefined;
    if (topReadback) {
      await topReadback.mapAsync(GPU_MAP_READ);
      const values = new Float32Array(topReadback.getMappedRange()).slice();
      topReadback.unmap();
      topReadback.destroy?.();
      topTokens = mergeTopCandidates(values, candidateCount);
    }

    for (const resource of resources) {
      resource.destroy();
    }
    for (const item of cleanup.reverse()) {
      item.destroy?.();
    }
    return { topTokens };
  }

  private ensureGpuState(state: Qwen35WebGpuStateLike): GpuState {
    if (state.contextLength !== this.manifest.contextLength && state.contextLength <= 0) {
      throw new Error(`Invalid WebGPU state context length: ${state.contextLength}`);
    }
    const key = state as object;
    const existing = this.states.get(key);
    if (existing) {
      return existing;
    }
    if (state.nextPosition !== 0) {
      throw new Error(
        "WebGPU suffix state is missing for a non-empty chat state; replay from position 0 is required.",
      );
    }
    const recurrent = new Map<number, RecurrentGpuLayerState>();
    const fullAttention = new Map<number, FullAttentionGpuLayerState>();
    const convDim = this.manifest.ssm.stateSize * this.manifest.ssm.groupCount * 2 +
      this.manifest.ssm.stateSize * this.manifest.ssm.timeStepRank;
    const recurrentStateSize =
      this.manifest.ssm.stateSize * this.manifest.ssm.stateSize * this.manifest.ssm.timeStepRank;
    const fullCacheSize = state.contextLength * this.manifest.headCountKv * this.manifest.keyLength;
    for (const layer of this.layers) {
      if (layer.kind === "recurrent") {
        recurrent.set(layer.layer, {
          conv: this.arena.createBuffer(
            `blk.${layer.layer}.gpu.conv_state`,
            (this.manifest.ssm.convKernel - 1) * convDim * Float32Array.BYTES_PER_ELEMENT,
            GPU_STORAGE,
          ),
          recurrent: this.arena.createBuffer(
            `blk.${layer.layer}.gpu.recurrent_state`,
            recurrentStateSize * Float32Array.BYTES_PER_ELEMENT,
            GPU_STORAGE,
          ),
        });
      } else {
        fullAttention.set(layer.layer, {
          key: this.arena.createBuffer(
            `blk.${layer.layer}.gpu.key_cache`,
            fullCacheSize * Float32Array.BYTES_PER_ELEMENT,
            GPU_STORAGE,
          ),
          value: this.arena.createBuffer(
            `blk.${layer.layer}.gpu.value_cache`,
            fullCacheSize * Float32Array.BYTES_PER_ELEMENT,
            GPU_STORAGE,
          ),
        });
      }
    }
    const created = { recurrent, fullAttention };
    this.states.set(key, created);
    return created;
  }

  private dispatchRecurrentLayer(
    pass: WebGpuComputePassLike,
    layer: RecurrentGpuLayer,
    gpuState: GpuState,
    input: WebGpuBufferLike,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const state = gpuState.recurrent.get(layer.layer);
    if (!state) {
      throw new Error(`Missing WebGPU recurrent state for layer ${layer.layer}`);
    }
    const hiddenSize = this.manifest.embeddingLength;
    const convDim = this.manifest.ssm.stateSize * this.manifest.ssm.groupCount * 2 +
      this.manifest.ssm.stateSize * this.manifest.ssm.timeStepRank;
    const valueDim = this.manifest.ssm.stateSize * this.manifest.ssm.timeStepRank;
    const keyDim = this.manifest.ssm.stateSize * this.manifest.ssm.groupCount;
    const tokenCount = 1;

    const attnNorm = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.attn_norm`);
    dispatchRmsNorm(this.arena.device, pass, resources, input, layer.attnNorm.buffer, attnNorm, hiddenSize, this.epsilon);
    const q8 = scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.attn_norm.q8k`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, attnNorm, q8, hiddenSize, tokenCount);

    const qkv = scratchF32(this.arena, convDim, cleanup, `blk.${layer.layer}.qkv`);
    const z = scratchF32(this.arena, valueDim, cleanup, `blk.${layer.layer}.z`);
    const alpha = scratchF32(this.arena, this.manifest.ssm.timeStepRank, cleanup, `blk.${layer.layer}.alpha`);
    const beta = scratchF32(this.arena, this.manifest.ssm.timeStepRank, cleanup, `blk.${layer.layer}.beta`);
    dispatchKMatMul(pass, resources, layer.qkv, q8, qkv, tokenCount);
    dispatchKMatMul(pass, resources, layer.z, q8, z, tokenCount);
    dispatchF32MatMul(this.arena.device, pass, resources, layer.alpha.buffer, attnNorm, alpha, hiddenSize, this.manifest.ssm.timeStepRank, tokenCount);
    dispatchF32MatMul(this.arena.device, pass, resources, layer.beta.buffer, attnNorm, beta, hiddenSize, this.manifest.ssm.timeStepRank, tokenCount);

    const gate = scratchF32(this.arena, this.manifest.ssm.timeStepRank, cleanup, `blk.${layer.layer}.gate`);
    const betaSigmoid = scratchF32(this.arena, this.manifest.ssm.timeStepRank, cleanup, `blk.${layer.layer}.beta_sigmoid`);
    const q = scratchF32(this.arena, keyDim, cleanup, `blk.${layer.layer}.q_conv`);
    const k = scratchF32(this.arena, keyDim, cleanup, `blk.${layer.layer}.k_conv`);
    const v = scratchF32(this.arena, valueDim, cleanup, `blk.${layer.layer}.v_conv`);
    const nextConv = this.arena.createBuffer(
      `blk.${layer.layer}.next_conv`,
      (this.manifest.ssm.convKernel - 1) * convDim * Float32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE,
    );
    const delta = scratchF32(this.arena, valueDim, cleanup, `blk.${layer.layer}.delta`);
    const nextRecurrent = this.arena.createBuffer(
      `blk.${layer.layer}.next_recurrent`,
      this.manifest.ssm.timeStepRank * this.manifest.ssm.stateSize * this.manifest.ssm.stateSize * Float32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE,
    );
    const gated = scratchF32(this.arena, valueDim, cleanup, `blk.${layer.layer}.ssm_gated`);
    const attention = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.attention`);

    dispatchQkvConv(
      this.arena.device,
      pass,
      resources,
      qkv,
      state.conv,
      layer.convKernel.buffer,
      q,
      k,
      v,
      nextConv,
      {
        tokenCount,
        convDim,
        kernelSize: this.manifest.ssm.convKernel,
        stateSize: this.manifest.ssm.stateSize,
        groupCount: this.manifest.ssm.groupCount,
        valueDim,
      },
    );

    dispatchDeltaGate(
      this.arena.device,
      pass,
      resources,
      alpha,
      beta,
      layer.dtBias.buffer,
      layer.ssmA.buffer,
      gate,
      betaSigmoid,
      this.manifest.ssm.timeStepRank,
      tokenCount,
    );

    dispatchGatedDeltaNet(
      this.arena.device,
      pass,
      resources,
      q,
      k,
      v,
      gate,
      betaSigmoid,
      state.recurrent,
      delta,
      nextRecurrent,
      {
        stateSize: this.manifest.ssm.stateSize,
        keyHeadCount: this.manifest.ssm.groupCount,
        valueHeadCount: this.manifest.ssm.timeStepRank,
        tokenCount,
      },
    );

    dispatchSsmNormGate(
      this.arena.device,
      pass,
      resources,
      delta,
      z,
      layer.ssmNorm.buffer,
      gated,
      this.manifest.ssm.stateSize,
      this.manifest.ssm.timeStepRank,
      this.epsilon,
    );

    const outQ8 = scratchQ8_0(this.arena, valueDim, tokenCount, layer.out.blockCount, cleanup, `blk.${layer.layer}.ssm_gated.q8_0`);
    dispatchQ8_0Quantize(this.arena.device, pass, resources, gated, outQ8, valueDim, tokenCount, layer.out.blockCount);
    dispatchQ8_0MatMul(pass, resources, layer.out, outQ8, attention, tokenCount);

    cleanup.push(state.conv, state.recurrent);
    state.conv = nextConv;
    state.recurrent = nextRecurrent;
    return this.dispatchFfn(pass, layer, input, attention, cleanup, resources);
  }

  private dispatchFullAttentionLayer(
    pass: WebGpuComputePassLike,
    layer: FullAttentionGpuLayer,
    gpuState: GpuState,
    input: WebGpuBufferLike,
    tokenPosition: number,
    mropePosition: Int32Array,
    contextLength: number,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const state = gpuState.fullAttention.get(layer.layer);
    if (!state) {
      throw new Error(`Missing WebGPU full-attention state for layer ${layer.layer}`);
    }
    const hiddenSize = this.manifest.embeddingLength;
    const fullQueryDim = this.manifest.headCount * this.manifest.keyLength * 2;
    const fullKeyValueDim = this.manifest.headCountKv * this.manifest.keyLength;
    const keyValueTokenCount = Math.min(contextLength, tokenPosition + 1);
    const tokenCount = 1;

    const attnNorm = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.attn_norm`);
    dispatchRmsNorm(this.arena.device, pass, resources, input, layer.attnNorm.buffer, attnNorm, hiddenSize, this.epsilon);
    const q8 = scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.attn_norm.q8k`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, attnNorm, q8, hiddenSize, tokenCount);

    const qFull = scratchF32(this.arena, fullQueryDim, cleanup, `blk.${layer.layer}.q_full`);
    const kProjection = scratchF32(this.arena, fullKeyValueDim, cleanup, `blk.${layer.layer}.k`);
    const vProjection = scratchF32(this.arena, fullKeyValueDim, cleanup, `blk.${layer.layer}.v`);
    dispatchKMatMul(pass, resources, layer.q, q8, qFull, tokenCount);
    dispatchKMatMul(pass, resources, layer.k, q8, kProjection, tokenCount);
    dispatchKMatMul(pass, resources, layer.v, q8, vProjection, tokenCount);

    const query = scratchF32(this.arena, this.manifest.headCount * this.manifest.keyLength, cleanup, `blk.${layer.layer}.q_rope`);
    const gate = scratchF32(this.arena, this.manifest.headCount * this.manifest.keyLength, cleanup, `blk.${layer.layer}.gate`);
    dispatchFullQuery(
      this.arena.device,
      pass,
      resources,
      qFull,
      layer.qNorm.buffer,
      query,
      gate,
      {
        headCount: this.manifest.headCount,
        headSize: this.manifest.keyLength,
        ropeDims: this.manifest.rope.dimensionCount,
        epsilon: this.epsilon,
        freqBase: this.manifest.rope.freqBase,
        position: mropePosition[0] ?? tokenPosition,
      },
    );
    dispatchFullKvUpdate(
      this.arena.device,
      pass,
      resources,
      kProjection,
      vProjection,
      layer.kNorm.buffer,
      state.key,
      state.value,
      {
        headCount: this.manifest.headCountKv,
        headSize: this.manifest.keyLength,
        ropeDims: this.manifest.rope.dimensionCount,
        epsilon: this.epsilon,
        freqBase: this.manifest.rope.freqBase,
        position: mropePosition[0] ?? tokenPosition,
        tokenPosition,
        contextLength,
      },
    );

    const gated = scratchF32(this.arena, this.manifest.headCount * this.manifest.keyLength, cleanup, `blk.${layer.layer}.attention_gated`);
    const probabilities = scratchF32(
      this.arena,
      this.manifest.headCount * keyValueTokenCount,
      cleanup,
      `blk.${layer.layer}.attention_probabilities`,
    );
    const attentionOptions = {
      headSize: this.manifest.keyLength,
      queryHeadCount: this.manifest.headCount,
      keyValueHeadCount: this.manifest.headCountKv,
      keyValueTokenCount,
      contextLength,
      scale: 1 / Math.sqrt(this.manifest.keyLength),
    };
    dispatchFullAttentionScore(
      this.arena.device,
      pass,
      resources,
      query,
      state.key,
      probabilities,
      attentionOptions,
    );
    dispatchFullAttentionApply(
      this.arena.device,
      pass,
      resources,
      state.value,
      gate,
      probabilities,
      gated,
      attentionOptions,
    );

    const gatedQ8 = scratchQ8K(this.arena, this.manifest.headCount * this.manifest.keyLength, tokenCount, cleanup, `blk.${layer.layer}.attention_gated.q8k`);
    const attention = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.attention_out`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, gated, gatedQ8, this.manifest.headCount * this.manifest.keyLength, tokenCount);
    dispatchKMatMul(pass, resources, layer.out, gatedQ8, attention, tokenCount);
    return this.dispatchFfn(pass, layer, input, attention, cleanup, resources);
  }

  private dispatchFfn(
    pass: WebGpuComputePassLike,
    layer: GpuLayer,
    residualInput: WebGpuBufferLike,
    attention: WebGpuBufferLike,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const hiddenSize = this.manifest.embeddingLength;
    const residual = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.residual`);
    dispatchResidualAdd(this.arena.device, pass, resources, residualInput, attention, residual, hiddenSize);
    const postNorm = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.post_norm`);
    dispatchRmsNorm(this.arena.device, pass, resources, residual, layer.postNorm.buffer, postNorm, hiddenSize, this.epsilon);
    const postQ8 = scratchQ8K(this.arena, hiddenSize, 1, cleanup, `blk.${layer.layer}.post_norm.q8k`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, postNorm, postQ8, hiddenSize, 1);

    const gate = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_gate`);
    const up = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_up`);
    const swiglu = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_swiglu`);
    dispatchKMatMul(pass, resources, layer.ffnGate, postQ8, gate, 1);
    dispatchKMatMul(pass, resources, layer.ffnUp, postQ8, up, 1);

    dispatchSwiGlu(this.arena.device, pass, resources, gate, up, swiglu, this.manifest.feedForwardLength);

    const swigluQ8 = scratchQ8K(this.arena, this.manifest.feedForwardLength, 1, cleanup, `blk.${layer.layer}.ffn_swiglu.q8k`);
    const ffnOut = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.ffn_out`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, swiglu, swigluQ8, this.manifest.feedForwardLength, 1);
    dispatchKMatMul(pass, resources, layer.ffnDown, swigluQ8, ffnOut, 1);

    const output = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.output`);
    dispatchResidualAdd(this.arena.device, pass, resources, residual, ffnOut, output, hiddenSize);
    return output;
  }

  private dispatchOutputTopK(
    pass: WebGpuComputePassLike,
    hidden: WebGpuBufferLike,
    topKCount: number,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const hiddenSize = this.manifest.embeddingLength;
    const norm = scratchF32(this.arena, hiddenSize, cleanup, "output_norm");
    dispatchRmsNorm(this.arena.device, pass, resources, hidden, this.outputNorm.buffer, norm, hiddenSize, this.epsilon);
    const q8 = scratchQ8K(this.arena, hiddenSize, 1, cleanup, "output_norm.q8k");
    dispatchQ8KQuantize(this.arena.device, pass, resources, norm, q8, hiddenSize, 1);

    const candidates = this.arena.createBuffer(
      "output.topk.candidates",
      this.outputStripes.length * topKCount * 2 * Float32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE | GPU_COPY_SRC,
    );
    cleanup.push(candidates);
    for (let index = 0; index < this.outputStripes.length; index += 1) {
      const stripe = this.outputStripes[index];
      if (!stripe) {
        continue;
      }
      const logits = scratchF32(this.arena, stripe.rowCount, cleanup, `output.logits.${index}`);
      dispatchKMatMul(pass, resources, stripe, q8, logits, 1);
      dispatchTopK(
        this.arena.device,
        pass,
        resources,
        logits,
        candidates,
        {
          rowCount: stripe.rowCount,
          rowOffset: stripe.rowOffset,
          topK: topKCount,
          candidateOffset: index * topKCount * 2,
        },
      );
    }
    return candidates;
  }
}

function tokenPositionFromSingleMrope(positions: Int32Array): number {
  if (positions.length === 1 || positions.length === 4) {
    return positions[0] ?? 0;
  }
  throw new Error(`WebGPU token path expects one position, got ${positions.length}`);
}

function singleMropePosition(positions: Int32Array): Int32Array {
  if (positions.length === 4) {
    return positions;
  }
  if (positions.length === 1) {
    const position = positions[0] ?? 0;
    return new Int32Array([position, position, position, position]);
  }
  throw new Error(`WebGPU token path expects one M-RoPE position, got ${positions.length}`);
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

function mergeTopCandidates(values: Float32Array, topKCount: number): WebGpuTopToken[] {
  const best: WebGpuTopToken[] = [];
  for (let index = 0; index < values.length; index += 2) {
    const id = Math.trunc(values[index] ?? 0);
    const value = values[index + 1] ?? -Infinity;
    if (!Number.isFinite(value)) {
      continue;
    }
    best.push({ id, value });
    best.sort((left, right) => right.value - left.value);
    if (best.length > topKCount) {
      best.pop();
    }
  }
  return best;
}
