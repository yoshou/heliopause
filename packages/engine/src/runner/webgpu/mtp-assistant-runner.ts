import type {
  GgmlTypeName,
} from "../../gguf";
import {
  dequantizeRow,
} from "../../quant";
import {
  tensorByteLength,
} from "../../tensor-reader";
import type {
  MtpAssistantRunResult,
  MtpAssistantSession,
} from "../../mtp-assistant";
import type {
  MtpAssistantRunInput,
  MtpAssistantRunner,
  MtpAssistantRunners,
  MtpTargetKvLayerView,
} from "../mtp-assistant-runner";
import {
  GpuMemoryArena,
  scratchF32,
  type F32Handle,
  type GpuResource,
  type QuantizedHandle,
} from "./arena";
import {
  GPU_COPY_DST,
  GPU_MAP_READ,
  GPU_STORAGE,
  WEBGPU_MEMORY_LIMIT_BYTES,
} from "./gpu-constants";
import {
  webGpuDevice,
} from "./gpu-device";
import type {
  WebGpuConfiguredProvider,
} from "./execution-provider";
import type {
  WebGpuBufferLike,
  WebGpuComputePassLike,
  WebGpuDeviceLike,
} from "./gpu-types";
import {
  dispatchF32GatherRowsScale,
  dispatchF32MatMul,
  dispatchGeglu,
  dispatchHeadRmsNorm,
  dispatchQ8_0GatherRowsScale,
  dispatchQuantizedGatherRowsScale,
  dispatchRmsNorm,
  dispatchRmsNormResidualAdd,
  dispatchScale,
  dispatchTopK,
} from "./dispatch";
import {
  createMtpAttentionResources,
  createMtpConcat2Resources,
  createMtpEmbeddingDotResources,
  createMtpRopeResources,
} from "./mtp-assistant-kernels";
import {
  loadF32Handle,
  loadQuantizedHandle,
} from "./segment-layer-loader";

type WebGpuMtpAssistantStats = {
  runs: number;
  readbackBytes: number;
  residentBytes: number;
  lastRunMs: number;
};

type RunResources = {
  resources: Array<{ destroy: () => void }>;
  cleanup: GpuResource[];
};

const runners = new WeakMap<MtpAssistantSession, Promise<WebGpuMtpAssistantRunner>>();
const statsBySession = new WeakMap<MtpAssistantSession, WebGpuMtpAssistantStats>();

export function createWebGpuMtpAssistantRunners(): MtpAssistantRunners {
  return { runner: webGpuMtpAssistantRunner };
}

const webGpuMtpAssistantRunner: MtpAssistantRunner = {
  provider: "webgpu",
  async run(session, input, options) {
    throwIfAborted(options.signal);
    const stats = webGpuMtpAssistantStats(session);
    const runner = await webGpuMtpRunner(session);
    const started = nowMs();
    try {
      const result = await runner.run(input, options);
      stats.runs += 1;
      stats.readbackBytes = runner.readbackBytes;
      stats.residentBytes = runner.residentBytes;
      return result;
    } finally {
      stats.lastRunMs = nowMs() - started;
    }
  },
};

async function webGpuMtpRunner(session: MtpAssistantSession): Promise<WebGpuMtpAssistantRunner> {
  let runner = runners.get(session);
  if (!runner) {
    runner = createWebGpuMtpRunner(session);
    runners.set(session, runner);
  }
  return runner;
}

async function createWebGpuMtpRunner(session: MtpAssistantSession): Promise<WebGpuMtpAssistantRunner> {
  const device = await webGpuDevice();
  if (!device) {
    throw new Error("WebGPU is not available for MTP assistant execution.");
  }
  const options = session.provider<WebGpuConfiguredProvider>("webgpu")?.options;
  const runner = new WebGpuMtpAssistantRunner(
    session,
    device,
    options?.memoryLimitBytes ?? WEBGPU_MEMORY_LIMIT_BYTES,
  );
  session.addDisposeCallback(() => runner.dispose());
  return runner;
}

function webGpuMtpAssistantStats(session: MtpAssistantSession): WebGpuMtpAssistantStats {
  let stats = statsBySession.get(session);
  if (!stats) {
    stats = { runs: 0, readbackBytes: 0, residentBytes: 0, lastRunMs: 0 };
    statsBySession.set(session, stats);
    const captured = stats;
    session.setExecutionProviderStatsProvider(() => ({
      webgpuMtpAssistantRuns: captured.runs,
      webgpuMtpAssistantReadbackBytes: captured.readbackBytes,
      webgpuMtpAssistantResidentBytes: captured.residentBytes,
      webgpuMtpAssistantLastRunMs: captured.lastRunMs,
    }), "webgpu-mtp-assistant");
  }
  return stats;
}

class WebGpuMtpAssistantRunner {
  private readonly session: MtpAssistantSession;
  private readonly arena: GpuMemoryArena;
  private readonly f32Handles = new Map<string, F32Handle>();
  private readonly quantizedHandles = new Map<string, QuantizedHandle>();
  private dummyF32: WebGpuBufferLike | undefined;
  readbackBytes = 0;

  get residentBytes(): number {
    return this.arena.residentBytes;
  }

  constructor(session: MtpAssistantSession, device: WebGpuDeviceLike, memoryLimitBytes: number) {
    this.session = session;
    this.arena = new GpuMemoryArena(device, memoryLimitBytes);
  }

  dispose(): void {
    for (const handle of this.f32Handles.values()) {
      handle.destroy();
    }
    for (const handle of this.quantizedHandles.values()) {
      handle.destroy();
    }
    this.f32Handles.clear();
    this.quantizedHandles.clear();
    this.dummyF32?.destroy?.();
    this.dummyF32 = undefined;
    this.arena.destroyScratchBuffers();
  }

  async run(input: MtpAssistantRunInput, options: { signal?: AbortSignal }): Promise<MtpAssistantRunResult> {
    validateInput(this.session, input);
    const run: RunResources = { resources: [], cleanup: [] };
    let readbackBackbone: WebGpuBufferLike | undefined;
    let readbackCentroids: WebGpuBufferLike | undefined;
    try {
      if (input.topK < 1 || input.topK > 64) {
        throw new Error(`WebGPU MTP assistant topK supports 1..64, got ${input.topK}.`);
      }
      const manifest = this.session.manifest;
      const encoder = this.arena.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      const targetEmbedding = this.bufferFromF32("mtp.target_input_embedding", input.targetInputEmbedding, run.cleanup);
      const targetCurrentHidden = this.bufferFromF32("mtp.target_current_hidden", input.targetCurrentHidden, run.cleanup);
      const projectionInput = scratchF32(this.arena, manifest.backboneEmbeddingLength * 2, run.cleanup, "mtp.projection_input");
      this.dispatchConcat2(pass, run, targetEmbedding, targetCurrentHidden, projectionInput, manifest.backboneEmbeddingLength, manifest.backboneEmbeddingLength);

      let hidden = await this.dispatchMatMulWeight(pass, run, "mtp.pre_projection.weight", projectionInput, 1);
      for (let layer = 0; layer < manifest.blockCount; layer += 1) {
        throwIfAborted(options.signal);
        hidden = await this.dispatchLayer(pass, run, input, layer, hidden);
      }

      const normalizedHidden = scratchF32(this.arena, manifest.embeddingLength, run.cleanup, "mtp.output_norm");
      dispatchRmsNorm(this.arena.device, pass, run.resources, hidden, (await this.f32Handle("output_norm.weight")).buffer, normalizedHidden, manifest.embeddingLength, this.session.epsilon);
      const postProjection = await this.dispatchMatMulWeight(pass, run, "mtp.post_projection.weight", normalizedHidden, 1);
      const centroidLogits = await this.dispatchMatMulWeight(pass, run, "mtp.centroids.weight", normalizedHidden, 1);

      const centroidTopK = Math.min(manifest.centroidTopK, manifest.nCentroids);
      if (centroidTopK < 1 || centroidTopK > 64) {
        throw new Error(`WebGPU MTP assistant centroid topK supports 1..64, got ${centroidTopK}.`);
      }
      const centroidCandidates = scratchF32(this.arena, centroidTopK * 2, run.cleanup, "mtp.centroid_topk");
      dispatchTopK(this.arena.device, pass, run.resources, centroidLogits, centroidCandidates, {
        rowCount: manifest.nCentroids,
        rowOffset: 0,
        topK: centroidTopK,
        candidateOffset: 0,
      });

      readbackBackbone = this.readbackBuffer("mtp.backbone_hidden.readback", manifest.backboneEmbeddingLength);
      readbackCentroids = this.readbackBuffer("mtp.centroid_topk.readback", centroidTopK * 2);
      pass.end();
      encoder.copyBufferToBuffer(postProjection, 0, readbackBackbone, 0, manifest.backboneEmbeddingLength * Float32Array.BYTES_PER_ELEMENT);
      encoder.copyBufferToBuffer(centroidCandidates, 0, readbackCentroids, 0, centroidTopK * 2 * Float32Array.BYTES_PER_ELEMENT);
      this.arena.device.queue.submit([encoder.finish()]);

      const [backboneHidden, centroidPairs] = await Promise.all([
        this.readMappedF32(readbackBackbone, manifest.backboneEmbeddingLength),
        this.readMappedF32(readbackCentroids, centroidTopK * 2),
      ]);
      readbackBackbone = undefined;
      readbackCentroids = undefined;
      const selectedIds = await this.selectedTokenIds(centroidPairs);
      const topTokens = await this.topTokensFromSelectedIds(normalizedHidden, selectedIds, input.topK);

      return { backboneHidden, topTokens };
    } finally {
      readbackBackbone?.destroy?.();
      readbackCentroids?.destroy?.();
      for (const resource of run.resources) {
        resource.destroy();
      }
      for (const resource of run.cleanup.reverse()) {
        resource.destroy?.();
      }
      this.arena.destroyScratchBuffers();
    }
  }

  private async dispatchLayer(
    pass: WebGpuComputePassLike,
    run: RunResources,
    input: MtpAssistantRunInput,
    layer: number,
    hidden: WebGpuBufferLike,
  ): Promise<WebGpuBufferLike> {
    const manifest = this.session.manifest;
    const kind = manifest.layerKinds[layer] ?? "sliding-attention";
    const headSize = manifest.layerKeyLengths[layer] ?? manifest.keyLength;
    const valueSize = manifest.layerValueLengths[layer] ?? manifest.valueLength;
    const targetKv = input.targetKv.layers[layer];
    if (!targetKv) {
      throw new Error(`Missing target KV view for WebGPU assistant layer ${layer}`);
    }
    validateTargetKvLayer(targetKv, manifest.headCountKv, headSize, valueSize, input.position, layer);

    const attnNorm = scratchF32(this.arena, manifest.embeddingLength, run.cleanup, `mtp.blk.${layer}.attn_norm`);
    dispatchRmsNorm(this.arena.device, pass, run.resources, hidden, (await this.f32Handle(`blk.${layer}.attn_norm.weight`)).buffer, attnNorm, manifest.embeddingLength, this.session.epsilon);
    const qProjection = await this.dispatchMatMulWeight(pass, run, `blk.${layer}.attn_q.weight`, attnNorm, 1);
    const qNorm = scratchF32(this.arena, manifest.headCount * headSize, run.cleanup, `mtp.blk.${layer}.q_norm`);
    dispatchHeadRmsNorm(this.arena.device, pass, run.resources, qProjection, (await this.f32Handle(`blk.${layer}.attn_q_norm.weight`)).buffer, qNorm, {
      headCount: manifest.headCount,
      headSize,
      epsilon: this.session.epsilon,
    });
    const qRope = scratchF32(this.arena, manifest.headCount * headSize, run.cleanup, `mtp.blk.${layer}.q_rope`);
    if (kind === "full-attention" && this.session.hasTensor("rope_freqs.weight")) {
      await this.f32Handle("rope_freqs.weight");
    }
    this.dispatchRope(pass, run, qNorm, qRope, {
      headCount: manifest.headCount,
      headSize,
      ropeDims: kind === "sliding-attention" ? manifest.rope.slidingDimensionCount : manifest.rope.fullDimensionCount,
      activePairCount: kind === "full-attention" ? Math.floor(manifest.rope.fullDimensionCount * 0.25 / 2) : 0,
      freqBase: kind === "sliding-attention" ? manifest.rope.slidingFreqBase : manifest.rope.fullFreqBase,
      position: input.position,
      hasFreqFactors: kind === "full-attention" && this.session.hasTensor("rope_freqs.weight"),
    });

    const keyValueTokenCount = Math.min(targetKv.tokenCount, targetKv.contextLength, input.position + 1);
    const key = this.bufferFromF32(`mtp.blk.${layer}.target_key`, compactKey(targetKv, keyValueTokenCount), run.cleanup);
    const value = this.bufferFromF32(`mtp.blk.${layer}.target_value`, compactValue(targetKv, keyValueTokenCount), run.cleanup);
    const attention = scratchF32(this.arena, manifest.headCount * valueSize, run.cleanup, `mtp.blk.${layer}.attention`);
    const attentionResource = createMtpAttentionResources(this.arena.device, qRope, key, value, attention, {
      headSize,
      valueSize,
      queryHeadCount: manifest.headCount,
      keyValueHeadCount: manifest.headCountKv,
      keyValueTokenCount,
      contextLength: keyValueTokenCount,
      position: input.position,
      slidingWindow: kind === "sliding-attention" ? manifest.slidingWindow : undefined,
    });
    run.resources.push(attentionResource);
    pass.setPipeline(attentionResource.pipeline);
    pass.setBindGroup(0, attentionResource.bindGroup);
    pass.dispatchWorkgroups(manifest.headCount, valueSize);

    const attentionOutput = await this.dispatchMatMulWeight(pass, run, `blk.${layer}.attn_output.weight`, attention, 1);
    const attentionResidual = scratchF32(this.arena, manifest.embeddingLength, run.cleanup, `mtp.blk.${layer}.attention_residual`);
    dispatchRmsNormResidualAdd(this.arena.device, pass, run.resources, attentionOutput, (await this.f32Handle(`blk.${layer}.post_attention_norm.weight`)).buffer, hidden, attentionResidual, manifest.embeddingLength, this.session.epsilon);
    const ffn = await this.dispatchFfn(pass, run, layer, attentionResidual);
    const output = scratchF32(this.arena, manifest.embeddingLength, run.cleanup, `mtp.blk.${layer}.output`);
    dispatchScale(this.arena.device, pass, run.resources, ffn, (await this.f32Handle(`blk.${layer}.layer_output_scale.weight`)).buffer, output, manifest.embeddingLength);
    return output;
  }

  private async dispatchFfn(
    pass: WebGpuComputePassLike,
    run: RunResources,
    layer: number,
    residual: WebGpuBufferLike,
  ): Promise<WebGpuBufferLike> {
    const manifest = this.session.manifest;
    const ffnNorm = scratchF32(this.arena, manifest.embeddingLength, run.cleanup, `mtp.blk.${layer}.ffn_norm`);
    dispatchRmsNorm(this.arena.device, pass, run.resources, residual, (await this.f32Handle(`blk.${layer}.ffn_norm.weight`)).buffer, ffnNorm, manifest.embeddingLength, this.session.epsilon);
    const gate = await this.dispatchMatMulWeight(pass, run, `blk.${layer}.ffn_gate.weight`, ffnNorm, 1);
    const up = await this.dispatchMatMulWeight(pass, run, `blk.${layer}.ffn_up.weight`, ffnNorm, 1);
    const gated = scratchF32(this.arena, manifest.feedForwardLength, run.cleanup, `mtp.blk.${layer}.ffn_geglu`);
    dispatchGeglu(this.arena.device, pass, run.resources, gate, up, gated, manifest.feedForwardLength);
    const ffnOut = await this.dispatchMatMulWeight(pass, run, `blk.${layer}.ffn_down.weight`, gated, 1);
    const output = scratchF32(this.arena, manifest.embeddingLength, run.cleanup, `mtp.blk.${layer}.ffn_residual`);
    dispatchRmsNormResidualAdd(this.arena.device, pass, run.resources, ffnOut, (await this.f32Handle(`blk.${layer}.post_ffw_norm.weight`)).buffer, residual, output, manifest.embeddingLength, this.session.epsilon);
    return output;
  }

  private async dispatchMatMulWeight(
    pass: WebGpuComputePassLike,
    run: RunResources,
    weightName: string,
    input: WebGpuBufferLike,
    columnCount: number,
  ): Promise<WebGpuBufferLike> {
    const tensor = this.session.getTensor(weightName);
    const inputSize = tensor.dimensions[0] ?? 0;
    const rowCount = tensor.dimensions[1] ?? 0;
    if (inputSize <= 0 || rowCount <= 0) {
      throw new Error(`${weightName} must be a matrix tensor.`);
    }
    const output = scratchF32(this.arena, rowCount * columnCount, run.cleanup, `${weightName}.output`);
    dispatchF32MatMul(this.arena.device, pass, run.resources, (await this.matmulF32Handle(weightName)).buffer, input, output, inputSize, rowCount, columnCount);
    return output;
  }

  private dispatchConcat2(
    pass: WebGpuComputePassLike,
    run: RunResources,
    left: WebGpuBufferLike,
    right: WebGpuBufferLike,
    output: WebGpuBufferLike,
    leftLength: number,
    rightLength: number,
  ): void {
    const resource = createMtpConcat2Resources(this.arena.device, left, right, output, leftLength, rightLength);
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil((leftLength + rightLength) / 256));
  }

  private dispatchRope(
    pass: WebGpuComputePassLike,
    run: RunResources,
    input: WebGpuBufferLike,
    output: WebGpuBufferLike,
    options: {
      headCount: number;
      headSize: number;
      ropeDims: number;
      activePairCount: number;
      freqBase: number;
      position: number;
      hasFreqFactors: boolean;
    },
  ): void {
    const resource = createMtpRopeResources(
      this.arena.device,
      input,
      options.hasFreqFactors ? this.requireRopeFreqFactors().buffer : this.dummyF32Buffer(),
      output,
      options,
    );
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil((options.headCount * options.headSize) / 256));
  }

  private async selectedTokenIds(centroidPairs: Float32Array): Promise<number[]> {
    const manifest = this.session.manifest;
    const tokenOrdering = await this.session.readI32Tensor("mtp.token_ordering.weight");
    const tokensPerCentroid = Math.floor(tokenOrdering.length / manifest.nCentroids);
    const selectedIds: number[] = [];
    const seen = new Set<number>();
    for (let index = 0; index < centroidPairs.length; index += 2) {
      const centroidId = Math.trunc(centroidPairs[index] ?? -1);
      if (centroidId < 0 || centroidId >= manifest.nCentroids) {
        continue;
      }
      const end = Math.min((centroidId + 1) * tokensPerCentroid, tokenOrdering.length);
      for (let tokenIndex = centroidId * tokensPerCentroid; tokenIndex < end; tokenIndex += 1) {
        const tokenId = tokenOrdering[tokenIndex] ?? -1;
        if (tokenId >= 0 && tokenId < tokenOrdering.length && !seen.has(tokenId)) {
          seen.add(tokenId);
          selectedIds.push(tokenId);
        }
      }
    }
    return selectedIds;
  }

  private async topTokensFromSelectedIds(
    normalizedHidden: WebGpuBufferLike,
    selectedIds: readonly number[],
    topK: number,
  ): Promise<Array<{ id: number; value: number }>> {
    if (selectedIds.length === 0 || topK <= 0) {
      return [];
    }
    const run: RunResources = { resources: [], cleanup: [] };
    let readback: WebGpuBufferLike | undefined;
    try {
      const manifest = this.session.manifest;
      const encoder = this.arena.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      const tokenIds = Uint32Array.from(selectedIds);
      const tokenIdBuffer = this.arena.createBuffer("mtp.selected_token_ids", tokenIds.byteLength, GPU_STORAGE | GPU_COPY_DST);
      this.arena.device.queue.writeBuffer(tokenIdBuffer, 0, tokenIds);
      run.cleanup.push(tokenIdBuffer);
      const rows = scratchF32(this.arena, selectedIds.length * manifest.embeddingLength, run.cleanup, "mtp.selected_embedding_rows");
      await this.dispatchEmbeddingGather(pass, run, tokenIdBuffer, rows, selectedIds.length);
      const logits = scratchF32(this.arena, selectedIds.length, run.cleanup, "mtp.selected_logits");
      const dotResource = createMtpEmbeddingDotResources(this.arena.device, rows, normalizedHidden, logits, {
        candidateCount: selectedIds.length,
        embeddingLength: manifest.embeddingLength,
      });
      run.resources.push(dotResource);
      pass.setPipeline(dotResource.pipeline);
      pass.setBindGroup(0, dotResource.bindGroup);
      pass.dispatchWorkgroups(selectedIds.length);
      const candidateTopK = Math.min(topK, selectedIds.length);
      const top = scratchF32(this.arena, candidateTopK * 2, run.cleanup, "mtp.selected_topk");
      dispatchTopK(this.arena.device, pass, run.resources, logits, top, {
        rowCount: selectedIds.length,
        rowOffset: 0,
        topK: candidateTopK,
        candidateOffset: 0,
      });
      readback = this.readbackBuffer("mtp.selected_topk.readback", candidateTopK * 2);
      pass.end();
      encoder.copyBufferToBuffer(top, 0, readback, 0, candidateTopK * 2 * Float32Array.BYTES_PER_ELEMENT);
      this.arena.device.queue.submit([encoder.finish()]);
      const pairs = await this.readMappedF32(readback, candidateTopK * 2);
      readback = undefined;
      return pairsToMappedTopTokens(pairs, selectedIds, candidateTopK);
    } finally {
      readback?.destroy?.();
      for (const resource of run.resources) {
        resource.destroy();
      }
      for (const resource of run.cleanup.reverse()) {
        resource.destroy?.();
      }
    }
  }

  private async dispatchEmbeddingGather(
    pass: WebGpuComputePassLike,
    run: RunResources,
    tokenIdBuffer: WebGpuBufferLike,
    output: WebGpuBufferLike,
    tokenCount: number,
  ): Promise<void> {
    const tensor = this.session.getTensor("token_embd.weight");
    const rowSize = tensor.dimensions[0] ?? 0;
    if (rowSize !== this.session.manifest.embeddingLength) {
      throw new Error(`token_embd.weight row size ${rowSize} does not match assistant hidden size ${this.session.manifest.embeddingLength}.`);
    }
    if (tensor.type === "F32") {
      dispatchF32GatherRowsScale(this.arena.device, pass, run.resources, (await this.f32Handle("token_embd.weight")).buffer, tokenIdBuffer, output, {
        rowSize,
        tokenCount,
        scale: 1,
      });
      return;
    }
    const handle = await this.quantizedHandle("token_embd.weight");
    if (handle.type === "Q8_0") {
      dispatchQ8_0GatherRowsScale(this.arena.device, pass, run.resources, handle.weightBuffer, tokenIdBuffer, output, {
        rowSize,
        tokenCount,
        blockCount: handle.blockCount,
        rowByteLength: handle.rowByteLength,
        scale: 1,
      });
      return;
    }
    dispatchQuantizedGatherRowsScale(pass, run.resources, handle, tokenIdBuffer, output, {
      rowSize,
      tokenCount,
      scale: 1,
    });
  }

  private bufferFromF32(label: string, values: Float32Array, cleanup: GpuResource[]): WebGpuBufferLike {
    const buffer = this.arena.createBuffer(label, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
    this.arena.device.queue.writeBuffer(buffer, 0, values);
    cleanup.push(buffer);
    return buffer;
  }

  private readbackBuffer(label: string, length: number): WebGpuBufferLike {
    return this.arena.device.createBuffer({
      label,
      size: Math.max(4, length * Float32Array.BYTES_PER_ELEMENT),
      usage: GPU_MAP_READ | GPU_COPY_DST,
    });
  }

  private async readMappedF32(buffer: WebGpuBufferLike, length: number): Promise<Float32Array> {
    await buffer.mapAsync(GPU_MAP_READ);
    try {
      const output = new Float32Array(buffer.getMappedRange()).slice(0, length);
      this.readbackBytes += length * Float32Array.BYTES_PER_ELEMENT;
      return output;
    } finally {
      buffer.unmap();
      buffer.destroy?.();
    }
  }

  private async f32Handle(name: string): Promise<F32Handle> {
    let handle = this.f32Handles.get(name);
    if (!handle) {
      handle = await loadF32Handle(this.arena, this.session.tensorReader, name);
      this.f32Handles.set(name, handle);
    }
    return handle;
  }

  private async matmulF32Handle(name: string): Promise<F32Handle> {
    const tensor = this.session.getTensor(name);
    if (tensor.type === "F32") {
      return this.f32Handle(name);
    }
    assertSupportedQuantizedType(tensor.type, name);
    let handle = this.f32Handles.get(`${name}:accuracy-f32`);
    if (!handle) {
      const inputSize = tensor.dimensions[0] ?? 0;
      const rowCount = tensor.dimensions[1] ?? 0;
      const bytes = await this.session.readWeightBytes(name);
      const rowByteLength = tensorByteLength({ ...tensor, dimensions: [inputSize] });
      const values = new Float32Array(inputSize * rowCount);
      for (let row = 0; row < rowCount; row += 1) {
        values.set(dequantizeRow(
          tensor.type,
          bytes.subarray(row * rowByteLength, (row + 1) * rowByteLength),
          inputSize,
        ), row * inputSize);
      }
      const buffer = this.arena.createBuffer(`${name}.accuracy_f32`, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
      this.arena.device.queue.writeBuffer(buffer, 0, values);
      handle = {
        length: values.length,
        byteLength: values.byteLength,
        device: this.arena.device,
        buffer,
        destroy: () => buffer.destroy?.(),
      };
      this.f32Handles.set(`${name}:accuracy-f32`, handle);
    }
    return handle;
  }

  private async quantizedHandle(name: string): Promise<QuantizedHandle> {
    const tensor = this.session.getTensor(name);
    assertSupportedQuantizedType(tensor.type, name);
    let handle = this.quantizedHandles.get(name);
    if (!handle) {
      handle = await loadQuantizedHandle(this.arena, this.session.tensorReader, name);
      this.quantizedHandles.set(name, handle);
    }
    return handle;
  }

  private requireRopeFreqFactors(): F32Handle {
    const handle = this.f32Handles.get("rope_freqs.weight");
    if (!handle) {
      throw new Error("WebGPU MTP assistant rope frequency factors were not loaded.");
    }
    return handle;
  }

  private dummyF32Buffer(): WebGpuBufferLike {
    if (!this.dummyF32) {
      const value = new Float32Array([1]);
      this.dummyF32 = this.arena.createBuffer("mtp.dummy_f32", value.byteLength, GPU_STORAGE | GPU_COPY_DST);
      this.arena.device.queue.writeBuffer(this.dummyF32, 0, value);
    }
    return this.dummyF32;
  }
}

function assertSupportedQuantizedType(type: GgmlTypeName, name: string): void {
  if (type !== "Q4_K" && type !== "Q5_K" && type !== "Q6_K" && type !== "Q8_0") {
    throw new Error(`${name} has unsupported WebGPU MTP assistant tensor type ${type}.`);
  }
}

function validateInput(session: MtpAssistantSession, input: MtpAssistantRunInput): void {
  if (input.targetInputEmbedding.length !== session.manifest.backboneEmbeddingLength) {
    throw new Error(`targetInputEmbedding shape mismatch: ${input.targetInputEmbedding.length}`);
  }
  if (input.targetPreviousHidden.length !== session.manifest.backboneEmbeddingLength) {
    throw new Error(`targetPreviousHidden shape mismatch: ${input.targetPreviousHidden.length}`);
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
    throw new Error(`Target KV head count mismatch for WebGPU assistant layer ${layerIndex}: ${layer.headCountKv}`);
  }
  if (layer.keyLength !== expectedKeySize || layer.valueLength !== expectedValueSize) {
    throw new Error(`Target KV head size mismatch for WebGPU assistant layer ${layerIndex}: ${layer.keyLength}/${layer.valueLength}`);
  }
  if (layer.tokenCount <= position) {
    throw new Error(`Target KV for WebGPU assistant layer ${layerIndex} does not cover position ${position}`);
  }
}

function compactKey(layer: MtpTargetKvLayerView, keyValueTokenCount: number): Float32Array {
  const length = keyValueTokenCount * layer.headCountKv * layer.keyLength;
  return layer.key.length === length ? layer.key : layer.key.slice(0, length);
}

function compactValue(layer: MtpTargetKvLayerView, keyValueTokenCount: number): Float32Array {
  const output = new Float32Array(layer.valueLength * layer.headCountKv * keyValueTokenCount);
  if (layer.value.length === output.length && layer.contextLength === keyValueTokenCount) {
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

function pairsToMappedTopTokens(
  pairs: Float32Array,
  selectedIds: readonly number[],
  topK: number,
): Array<{ id: number; value: number }> {
  const output: Array<{ id: number; value: number }> = [];
  for (let index = 0; index < pairs.length; index += 2) {
    const candidateIndex = Math.trunc(pairs[index] ?? -1);
    const tokenId = selectedIds[candidateIndex];
    const value = pairs[index + 1] ?? -Infinity;
    if (tokenId !== undefined && Number.isFinite(value)) {
      output.push({ id: tokenId, value });
    }
  }
  return output.sort((left, right) => right.value - left.value || left.id - right.id).slice(0, topK);
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("MTP assistant WebGPU execution was aborted.", "AbortError");
  }
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
