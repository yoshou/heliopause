import type { GgmlTypeName } from "../../gguf";
import type {
  AudioEncodeResult,
  AudioFeatures,
  AudioSession,
} from "../../audio";
import type {
  AudioEncoderRunner,
} from "../audio-runner";
import {
  dequantizeRow,
} from "../../quant";
import { GpuMemoryArena, scratchF32, scratchQ8_0, scratchQ8K, type F32Handle, type GpuResource, type QuantizedHandle } from "./arena";
import { GPU_COPY_DST, GPU_COPY_SRC, GPU_MAP_READ, GPU_STORAGE, WEBGPU_MEMORY_LIMIT_BYTES } from "./gpu-constants";
import { webGpuDevice } from "./gpu-device";
import {
  dispatchF32MatMul,
  dispatchKMatMul,
  dispatchQ8_0MatMul,
  dispatchQ8_0Quantize,
  dispatchQ8KQuantize,
  dispatchResidualAdd,
} from "./dispatch";
import type { WebGpuBufferLike, WebGpuComputePassLike } from "./gpu-types";
import { createQuantizedHandleFromBytes, webGpuMatMulType } from "./quantized-handles";
import {
  createAudioAddBiasRowsResources,
  createAudioAttentionResources,
  createAudioClampResources,
  createAudioConv2dSubsampleResources,
  createAudioDepthwiseConv1dResources,
  createAudioFlattenChannelsLastResources,
  createAudioGluResources,
  createAudioResidualAddScaleResources,
  createAudioRmsNormResources,
  createAudioSiluResources,
} from "./audio-kernel-resources";

type AudioStats = {
  attempts: number;
  runs: number;
};

type AudioRunBuffers = {
  resources: Array<{ destroy: () => void }>;
  cleanup: GpuResource[];
};

type ConvResult = {
  values: WebGpuBufferLike;
  time: number;
  frequency: number;
  mask: Uint32Array;
};

const runners = new WeakMap<AudioSession, Promise<WebGpuAudioRunner | undefined>>();
const statsBySession = new WeakMap<AudioSession, AudioStats>();

export const webGpuAudioEncoderRunner: AudioEncoderRunner = {
  provider: "webgpu",
  run: (session, features, options) => runWebGpuAudioEncoder(session, features, options),
};

export async function runWebGpuAudioEncoder(
  session: AudioSession,
  features: AudioFeatures,
  options: { signal?: AbortSignal } = {},
): Promise<AudioEncodeResult> {
  if (!session.executionProvider("webgpu")) {
    throw new Error("WebGPU audio encoder provider is not enabled.");
  }
  const stats = audioStats(session);
  stats.attempts += 1;
  const runner = await audioRunner(session);
  if (!runner) {
    throw new Error("WebGPU audio encoder is unavailable.");
  }
  const result = await runner.run(features, options);
  stats.runs += 1;
  return result;
}

function audioStats(session: AudioSession): AudioStats {
  let stats = statsBySession.get(session);
  if (!stats) {
    stats = {
      attempts: 0,
      runs: 0,
    };
    statsBySession.set(session, stats);
    const captured = stats;
    session.setExecutionProviderStatsProvider(() => ({
      webgpuAudioAttempts: captured.attempts,
      webgpuAudioRuns: captured.runs,
    }), "webgpu-audio");
  }
  return stats;
}

async function audioRunner(session: AudioSession): Promise<WebGpuAudioRunner | undefined> {
  let runner = runners.get(session);
  if (!runner) {
    runner = createAudioRunner(session);
    runners.set(session, runner);
  }
  return runner;
}

async function createAudioRunner(session: AudioSession): Promise<WebGpuAudioRunner | undefined> {
  const device = await webGpuDevice();
  if (!device) {
    throw new Error("WebGPU is not available for audio encoder execution.");
  }
  const options = session.executionProvider("webgpu")?.options;
  const arena = new GpuMemoryArena(
    device,
    numberOption(options, "memoryLimitBytes") ?? WEBGPU_MEMORY_LIMIT_BYTES,
  );
  const runner = new WebGpuAudioRunner(session, arena);
  session.addDisposeCallback(() => runner.dispose());
  return runner;
}

class WebGpuAudioRunner {
  private readonly session: AudioSession;
  private readonly arena: GpuMemoryArena;
  private readonly f32Handles = new Map<string, F32Handle>();
  private readonly quantizedHandles = new Map<string, QuantizedHandle>();
  private readonly onesHandles = new Map<number, F32Handle>();

  constructor(session: AudioSession, arena: GpuMemoryArena) {
    this.session = session;
    this.arena = arena;
  }

  dispose(): void {
    for (const handle of this.f32Handles.values()) {
      handle.destroy();
    }
    for (const handle of this.quantizedHandles.values()) {
      handle.destroy();
    }
    for (const handle of this.onesHandles.values()) {
      handle.destroy();
    }
    this.f32Handles.clear();
    this.quantizedHandles.clear();
    this.onesHandles.clear();
  }

  async run(features: AudioFeatures, options: { signal?: AbortSignal } = {}): Promise<AudioEncodeResult> {
    throwIfAborted(options.signal);
    const manifest = this.session.manifest;
    if (features.featureSize !== manifest.featureSize) {
      throw new Error(`Audio feature size mismatch: ${features.featureSize}`);
    }

    const run: AudioRunBuffers = { resources: [], cleanup: [] };
    let readback: WebGpuBufferLike | undefined;
    try {
      const encoder = this.arena.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      const projected = await this.dispatchSubsampleConvProjection(pass, run, features);
      let hidden = projected.values;
      const mask = projected.mask;
      const tokenCount = projected.time;
      const positionEmbeddings = this.bufferFromF32("audio.relative_position", audioRelativePositionEmbeddings(manifest), run.cleanup);
      for (let layer = 0; layer < manifest.blockCount; layer += 1) {
        throwIfAborted(options.signal);
        hidden = await this.dispatchLayer(pass, run, hidden, mask, positionEmbeddings, layer, tokenCount);
      }
      hidden = await this.dispatchMatMulAudioWeight(pass, run, "a.pre_encode.out.weight", hidden, tokenCount);
      hidden = await this.dispatchAddOptionalBiasRows(pass, run, hidden, "a.pre_encode.out.bias", tokenCount);
      hidden = this.dispatchRowRmsNorm(
        pass,
        run,
        hidden,
        "mm.a.soft_emb_norm",
        tokenCount,
        manifest.outputProjectionDim,
        this.session.hasTensor("mm.a.soft_emb_norm.weight")
          ? await this.f32Handle("mm.a.soft_emb_norm.weight")
          : await this.onesHandle(manifest.outputProjectionDim),
        this.session.epsilon,
      );
      hidden = await this.dispatchMatMulAudioWeight(pass, run, "mm.a.input_projection.weight", hidden, tokenCount);

      const outputLength = tokenCount * manifest.projectionDim;
      pass.end();
      readback = this.arena.device.createBuffer({
        size: outputLength * Float32Array.BYTES_PER_ELEMENT,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
      encoder.copyBufferToBuffer(hidden, 0, readback, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
      this.arena.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPU_MAP_READ);
      const result = new Float32Array(readback.getMappedRange()).slice();
      readback.unmap();
      readback.destroy?.();
      readback = undefined;
      return {
        hidden: result,
        tokenCount,
        durationMs: features.durationMs,
      };
    } finally {
      readback?.destroy?.();
      for (const resource of run.resources) {
        resource.destroy();
      }
      for (const item of run.cleanup.reverse()) {
        item.destroy?.();
      }
    }
  }

  private async dispatchSubsampleConvProjection(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    features: AudioFeatures,
  ): Promise<ConvResult> {
    const featureBuffer = this.bufferFromF32("audio.features", features.values, run.cleanup);
    const firstMask = uint32Mask(features.attentionMask);
    const first = await this.dispatchConv2dSubsampleLayer(
      pass,
      run,
      featureBuffer,
      firstMask,
      features.frameCount,
      features.featureSize,
      1,
      128,
      "a.conv1d.0",
    );
    const second = await this.dispatchConv2dSubsampleLayer(
      pass,
      run,
      first.values,
      first.mask,
      first.time,
      first.frequency,
      128,
      32,
      "a.conv1d.1",
    );
    const flattened = this.dispatchFlattenChannelsLast(pass, run, second.values, second.time, second.frequency, 32);
    const hidden = await this.dispatchMatMulAudioWeight(pass, run, "a.input_projection.weight", flattened, second.time);
    return {
      values: await this.dispatchAddOptionalBiasRows(pass, run, hidden, "a.input_projection.bias", second.time),
      mask: second.mask,
      time: second.time,
      frequency: second.frequency,
    };
  }

  private async dispatchConv2dSubsampleLayer(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    mask: Uint32Array,
    time: number,
    frequency: number,
    inChannels: number,
    outChannels: number,
    prefix: string,
  ): Promise<ConvResult> {
    const outTime = Math.floor((time + 1) / 2);
    const outFrequency = Math.floor((frequency + 1) / 2);
    const maskBuffer = this.bufferFromU32(`${prefix}.mask`, mask, run.cleanup);
    const bias = this.session.hasTensor(`${prefix}.bias`)
      ? await this.f32Handle(`${prefix}.bias`)
      : await this.onesHandle(1);
    const output = this.scratchF32(`${prefix}.output`, outTime * outChannels * outFrequency, run.cleanup);
    const resource = createAudioConv2dSubsampleResources(
      this.arena.device,
      input,
      maskBuffer,
      (await this.f32Handle(`${prefix}.weight`)).buffer,
      bias.buffer,
      (await this.f32Handle(`${prefix}.norm.weight`)).buffer,
      output,
      {
        time,
        frequency,
        inChannels,
        outChannels,
        outTime,
        outFrequency,
        hasBias: this.session.hasTensor(`${prefix}.bias`),
        epsilon: this.session.epsilon,
      },
    );
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(outTime, outFrequency);
    return {
      values: output,
      mask: downsampleMaskByTwo(mask, outTime),
      time: outTime,
      frequency: outFrequency,
    };
  }

  private async dispatchLayer(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    mask: Uint32Array,
    positionEmbeddings: WebGpuBufferLike,
    layer: number,
    tokenCount: number,
  ): Promise<WebGpuBufferLike> {
    const manifest = this.session.manifest;
    let hidden = await this.dispatchFeedForward(pass, run, input, layer, "", manifest.residualWeight, tokenCount);
    const attentionResidual = hidden;
    hidden = this.dispatchClamp(pass, run, hidden, -1e10, 1e10, tokenCount * manifest.embeddingLength, `a.blk.${layer}.attn_pre_clamp`);
    hidden = this.dispatchRowRmsNorm(pass, run, hidden, `a.blk.${layer}.attn_pre_norm`, tokenCount, manifest.embeddingLength, await this.f32Handle(`a.blk.${layer}.attn_pre_norm.weight`), this.session.epsilon);
    hidden = await this.dispatchAttention(pass, run, hidden, mask, positionEmbeddings, layer, tokenCount);
    hidden = this.dispatchClamp(pass, run, hidden, -1e10, 1e10, tokenCount * manifest.embeddingLength, `a.blk.${layer}.attn_post_clamp`);
    hidden = this.dispatchRowRmsNorm(pass, run, hidden, `a.blk.${layer}.attn_post_norm`, tokenCount, manifest.embeddingLength, await this.f32Handle(`a.blk.${layer}.attn_post_norm.weight`), this.session.epsilon);
    hidden = this.dispatchResidualAdd(pass, run, hidden, attentionResidual, tokenCount * manifest.embeddingLength, `a.blk.${layer}.attn_residual`);
    hidden = await this.dispatchLightConv(pass, run, hidden, layer, tokenCount);
    hidden = await this.dispatchFeedForward(pass, run, hidden, layer, "_1", manifest.residualWeight, tokenCount);
    hidden = this.dispatchClamp(pass, run, hidden, -1e10, 1e10, tokenCount * manifest.embeddingLength, `a.blk.${layer}.ln2_clamp`);
    return this.dispatchRowRmsNorm(pass, run, hidden, `a.blk.${layer}.ln2`, tokenCount, manifest.embeddingLength, await this.f32Handle(`a.blk.${layer}.ln2.weight`), this.session.epsilon);
  }

  private async dispatchFeedForward(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    layer: number,
    suffix: "" | "_1",
    residualWeight: number,
    tokenCount: number,
  ): Promise<WebGpuBufferLike> {
    const manifest = this.session.manifest;
    const residual = input;
    let hidden = this.dispatchClamp(pass, run, input, -1e10, 1e10, tokenCount * manifest.embeddingLength, `a.blk.${layer}.ffn${suffix}.input_clamp`);
    hidden = this.dispatchRowRmsNorm(pass, run, hidden, `a.blk.${layer}.ffn_norm${suffix}`, tokenCount, manifest.embeddingLength, await this.f32Handle(`a.blk.${layer}.ffn_norm${suffix}.weight`), this.session.epsilon);
    hidden = await this.dispatchMatMulAudioWeight(pass, run, `a.blk.${layer}.ffn_up${suffix}.weight`, hidden, tokenCount);
    hidden = this.dispatchSilu(pass, run, hidden, tokenCount * manifest.feedForwardLength, `a.blk.${layer}.ffn_silu${suffix}`);
    hidden = await this.dispatchMatMulAudioWeight(pass, run, `a.blk.${layer}.ffn_down${suffix}.weight`, hidden, tokenCount);
    hidden = this.dispatchClamp(pass, run, hidden, -1e10, 1e10, tokenCount * manifest.embeddingLength, `a.blk.${layer}.ffn${suffix}.output_clamp`);
    hidden = this.dispatchRowRmsNorm(pass, run, hidden, `a.blk.${layer}.ffn_post_norm${suffix}`, tokenCount, manifest.embeddingLength, await this.f32Handle(`a.blk.${layer}.ffn_post_norm${suffix}.weight`), this.session.epsilon);
    return this.dispatchResidualAddScale(pass, run, residual, hidden, residualWeight, tokenCount * manifest.embeddingLength, `a.blk.${layer}.ffn_residual${suffix}`);
  }

  private async dispatchAttention(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    mask: Uint32Array,
    positionEmbeddings: WebGpuBufferLike,
    layer: number,
    tokenCount: number,
  ): Promise<WebGpuBufferLike> {
    const manifest = this.session.manifest;
    const q = await this.dispatchMatMulAudioWeight(pass, run, `a.blk.${layer}.attn_q.weight`, input, tokenCount);
    const k = await this.dispatchMatMulAudioWeight(pass, run, `a.blk.${layer}.attn_k.weight`, input, tokenCount);
    const v = await this.dispatchMatMulAudioWeight(pass, run, `a.blk.${layer}.attn_v.weight`, input, tokenCount);
    const relativeK = await this.dispatchMatMulAudioWeight(pass, run, `a.blk.${layer}.attn_k_rel.weight`, positionEmbeddings, 13);
    const maskBuffer = this.bufferFromU32(`a.blk.${layer}.attention_mask`, mask, run.cleanup);
    const perDimScale = await this.f32Handle(`a.blk.${layer}.per_dim_scale.weight`);
    const hasPerDimKScale = this.session.hasTensor(`a.blk.${layer}.per_dim_k_scale.weight`);
    const perDimKScale = hasPerDimKScale
      ? await this.f32Handle(`a.blk.${layer}.per_dim_k_scale.weight`)
      : perDimScale;
    const attention = this.scratchF32(`a.blk.${layer}.attention`, tokenCount * manifest.embeddingLength, run.cleanup);
    const resource = createAudioAttentionResources(
      this.arena.device,
      q,
      k,
      v,
      relativeK,
      maskBuffer,
      perDimScale.buffer,
      perDimKScale.buffer,
      attention,
      {
        tokenCount,
        headCount: manifest.headCount,
        headSize: manifest.headSize,
        embeddingLength: manifest.embeddingLength,
        attentionChunkSize: manifest.attentionChunkSize,
        maxPast: 12,
        invalidLogit: manifest.attentionInvalidLogitsValue,
        logitCap: manifest.attentionLogitCap,
        qScale: Math.pow(manifest.headSize, -0.5) / Math.log(2),
        kScale: Math.log(1 + Math.E) / Math.log(2),
        hasPerDimKScale,
      },
    );
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(tokenCount, manifest.headCount, manifest.headSize);
    return this.dispatchAddOptionalBiasRows(
      pass,
      run,
      await this.dispatchMatMulAudioWeight(pass, run, `a.blk.${layer}.attn_out.weight`, attention, tokenCount),
      `a.blk.${layer}.attn_out.bias`,
      tokenCount,
    );
  }

  private async dispatchLightConv(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    layer: number,
    tokenCount: number,
  ): Promise<WebGpuBufferLike> {
    const manifest = this.session.manifest;
    const residual = input;
    let hidden = this.dispatchRowRmsNorm(pass, run, input, `a.blk.${layer}.conv_norm`, tokenCount, manifest.embeddingLength, await this.f32Handle(`a.blk.${layer}.conv_norm.weight`), this.session.epsilon);
    hidden = await this.dispatchMatMulAudioWeight(pass, run, `a.blk.${layer}.conv_pw1.weight`, hidden, tokenCount);
    hidden = this.dispatchGlu(pass, run, hidden, tokenCount, manifest.embeddingLength, `a.blk.${layer}.conv_glu`);
    hidden = this.dispatchDepthwiseConv1d(pass, run, hidden, await this.f32Handle(`a.blk.${layer}.conv_dw.weight`), tokenCount, manifest.convKernelSize, manifest.embeddingLength, `a.blk.${layer}.conv_dw`);
    hidden = await this.dispatchAddOptionalBiasRows(pass, run, hidden, `a.blk.${layer}.conv_dw.bias`, tokenCount);
    hidden = this.dispatchClamp(pass, run, hidden, -1e10, 1e10, tokenCount * manifest.embeddingLength, `a.blk.${layer}.conv_clamp`);
    hidden = this.dispatchRowRmsNorm(pass, run, hidden, `a.blk.${layer}.norm_conv`, tokenCount, manifest.embeddingLength, await this.f32Handle(`a.blk.${layer}.norm_conv.weight`), this.session.epsilon);
    hidden = this.dispatchSilu(pass, run, hidden, tokenCount * manifest.embeddingLength, `a.blk.${layer}.conv_silu`);
    hidden = await this.dispatchMatMulAudioWeight(pass, run, `a.blk.${layer}.conv_pw2.weight`, hidden, tokenCount);
    return this.dispatchResidualAdd(pass, run, hidden, residual, tokenCount * manifest.embeddingLength, `a.blk.${layer}.conv_residual`);
  }

  private dispatchFlattenChannelsLast(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    timeCount: number,
    frequencyCount: number,
    channelCount: number,
  ): WebGpuBufferLike {
    const output = this.scratchF32("audio.flatten", timeCount * frequencyCount * channelCount, run.cleanup);
    const resource = createAudioFlattenChannelsLastResources(this.arena.device, input, output, {
      timeCount,
      frequencyCount,
      channelCount,
    });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil((timeCount * frequencyCount * channelCount) / 256));
    return output;
  }

  private dispatchRowRmsNorm(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    label: string,
    rowCount: number,
    rowSize: number,
    weight: F32Handle,
    epsilon: number,
  ): WebGpuBufferLike {
    const output = this.scratchF32(`${label}.rms_norm`, rowCount * rowSize, run.cleanup);
    const resource = createAudioRmsNormResources(this.arena.device, input, weight.buffer, output, {
      rowCount,
      rowSize,
      epsilon,
    });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(rowCount);
    return output;
  }

  private dispatchClamp(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    min: number,
    max: number,
    length: number,
    label: string,
  ): WebGpuBufferLike {
    const output = this.scratchF32(label, length, run.cleanup);
    const resource = createAudioClampResources(this.arena.device, input, output, { length, min, max });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(length / 256));
    return output;
  }

  private dispatchSilu(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    length: number,
    label: string,
  ): WebGpuBufferLike {
    const output = this.scratchF32(label, length, run.cleanup);
    const resource = createAudioSiluResources(this.arena.device, input, output, { length });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(length / 256));
    return output;
  }

  private dispatchGlu(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    tokenCount: number,
    outputSize: number,
    label: string,
  ): WebGpuBufferLike {
    const output = this.scratchF32(label, tokenCount * outputSize, run.cleanup);
    const resource = createAudioGluResources(this.arena.device, input, output, { tokenCount, outputSize });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil((tokenCount * outputSize) / 256));
    return output;
  }

  private dispatchDepthwiseConv1d(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    weight: F32Handle,
    tokenCount: number,
    kernelSize: number,
    channels: number,
    label: string,
  ): WebGpuBufferLike {
    const output = this.scratchF32(label, tokenCount * channels, run.cleanup);
    const resource = createAudioDepthwiseConv1dResources(this.arena.device, input, weight.buffer, output, {
      tokenCount,
      kernelSize,
      channels,
    });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil((tokenCount * channels) / 256));
    return output;
  }

  private async dispatchAddOptionalBiasRows(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    biasName: string,
    rowCount: number,
  ): Promise<WebGpuBufferLike> {
    if (!this.session.hasTensor(biasName)) {
      return input;
    }
    const bias = await this.f32Handle(biasName);
    const length = rowCount * bias.length;
    const output = this.scratchF32(`${biasName}.add`, length, run.cleanup);
    const resource = createAudioAddBiasRowsResources(this.arena.device, input, bias.buffer, output, {
      length,
      rowSize: bias.length,
    });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(length / 256));
    return output;
  }

  private dispatchResidualAdd(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    left: WebGpuBufferLike,
    right: WebGpuBufferLike,
    length: number,
    label: string,
  ): WebGpuBufferLike {
    const output = this.scratchF32(label, length, run.cleanup);
    dispatchResidualAdd(this.arena.device, pass, run.resources, left, right, output, length);
    return output;
  }

  private dispatchResidualAddScale(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    residual: WebGpuBufferLike,
    hidden: WebGpuBufferLike,
    scale: number,
    length: number,
    label: string,
  ): WebGpuBufferLike {
    const output = this.scratchF32(label, length, run.cleanup);
    const resource = createAudioResidualAddScaleResources(this.arena.device, residual, hidden, output, { length, scale });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(length / 256));
    return output;
  }

  private async dispatchMatMulAudioWeight(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    weightName: string,
    input: WebGpuBufferLike,
    columnCount: number,
  ): Promise<WebGpuBufferLike> {
    const tensor = this.session.getTensor(weightName);
    const inputSize = tensor.dimensions[0] ?? 0;
    const rowCount = tensor.dimensions[1] ?? 0;
    if (inputSize <= 0 || rowCount <= 0) {
      return unsupported(`${weightName} has invalid WebGPU audio matmul shape`);
    }
    let current = await this.dispatchClampTensor(pass, run, input, weightName.replace(/\.weight$/, ".input_min"), weightName.replace(/\.weight$/, ".input_max"), inputSize * columnCount);
    const output = this.scratchF32(`${weightName}.output`, rowCount * columnCount, run.cleanup);
    if (isDenseType(tensor.type)) {
      const handle = await this.f32Handle(weightName);
      dispatchF32MatMul(this.arena.device, pass, run.resources, handle.buffer, current, output, inputSize, rowCount, columnCount);
    } else {
      const handle = await this.quantizedHandle(weightName);
      if (handle.type === "Q8_0") {
        const q8 = scratchQ8_0(this.arena, inputSize, columnCount, handle.blockCount, run.cleanup, `${weightName}.q8_0`);
        dispatchQ8_0Quantize(this.arena.device, pass, run.resources, current, q8, inputSize, columnCount, handle.blockCount);
        dispatchQ8_0MatMul(pass, run.resources, handle, q8, output, columnCount);
      } else {
        const q8 = scratchQ8K(this.arena, inputSize, columnCount, run.cleanup, `${weightName}.q8k`);
        dispatchQ8KQuantize(this.arena.device, pass, run.resources, current, q8, inputSize, columnCount);
        dispatchKMatMul(pass, run.resources, handle, q8, output, columnCount);
      }
    }
    current = await this.dispatchClampTensor(pass, run, output, weightName.replace(/\.weight$/, ".output_min"), weightName.replace(/\.weight$/, ".output_max"), rowCount * columnCount);
    return current;
  }

  private async dispatchClampTensor(
    pass: WebGpuComputePassLike,
    run: AudioRunBuffers,
    input: WebGpuBufferLike,
    minName: string,
    maxName: string,
    length: number,
  ): Promise<WebGpuBufferLike> {
    if (!this.session.hasTensor(minName) || !this.session.hasTensor(maxName)) {
      return input;
    }
    const min = (await this.session.readF32Tensor(minName))[0] ?? -Infinity;
    const max = (await this.session.readF32Tensor(maxName))[0] ?? Infinity;
    return this.dispatchClamp(pass, run, input, min, max, length, `${minName}.clamp`);
  }

  private async f32Handle(name: string): Promise<F32Handle> {
    const cached = this.f32Handles.get(name);
    if (cached) {
      return cached;
    }
    const tensor = this.session.getTensor(name);
    if (!isDenseType(tensor.type)) {
      return unsupported(`${name} must be dense for WebGPU audio, got ${tensor.type}`);
    }
    const elementCount = tensor.dimensions.reduce((product, dimension) => product * dimension, 1);
    const bytes = await this.session.readWeightBytes(name);
    const values = tensor.type === "F32"
      ? new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Float32Array.BYTES_PER_ELEMENT).slice()
      : dequantizeRow(tensor.type, bytes, elementCount);
    const buffer = this.arena.createBuffer(name, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
    this.arena.device.queue.writeBuffer(buffer, 0, values);
    const handle: F32Handle = {
      length: values.length,
      byteLength: values.byteLength,
      device: this.arena.device,
      buffer,
      destroy: () => buffer.destroy?.(),
    };
    this.f32Handles.set(name, handle);
    return handle;
  }

  private async quantizedHandle(name: string): Promise<QuantizedHandle> {
    const cached = this.quantizedHandles.get(name);
    if (cached) {
      return cached;
    }
    const tensor = this.session.getTensor(name);
    let handle: QuantizedHandle;
    try {
      handle = createQuantizedHandleFromBytes(
        this.arena,
        name,
        webGpuMatMulType(tensor.type, name),
        tensor.dimensions[0] ?? 0,
        tensor.dimensions[1] ?? 0,
        await this.session.readWeightBytes(name),
      );
    } catch (error) {
      return unsupported(error instanceof Error ? error.message : String(error));
    }
    this.quantizedHandles.set(name, handle);
    return handle;
  }

  private async onesHandle(length: number): Promise<F32Handle> {
    const cached = this.onesHandles.get(length);
    if (cached) {
      return cached;
    }
    const values = new Float32Array(length);
    values.fill(1);
    const buffer = this.arena.createBuffer(`audio.ones.${length}`, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
    this.arena.device.queue.writeBuffer(buffer, 0, values);
    const handle: F32Handle = {
      length,
      byteLength: values.byteLength,
      device: this.arena.device,
      buffer,
      destroy: () => buffer.destroy?.(),
    };
    this.onesHandles.set(length, handle);
    return handle;
  }

  private bufferFromF32(label: string, values: Float32Array, cleanup: GpuResource[]): WebGpuBufferLike {
    const buffer = this.arena.createBuffer(label, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
    this.arena.device.queue.writeBuffer(buffer, 0, values);
    cleanup.push(buffer);
    return buffer;
  }

  private bufferFromU32(label: string, values: Uint32Array, cleanup: GpuResource[]): WebGpuBufferLike {
    const buffer = this.arena.createBuffer(label, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
    this.arena.device.queue.writeBuffer(buffer, 0, values);
    cleanup.push(buffer);
    return buffer;
  }

  private scratchF32(label: string, length: number, cleanup: GpuResource[]): WebGpuBufferLike {
    return scratchF32(this.arena, length, cleanup, label);
  }
}

function audioRelativePositionEmbeddings(manifest: AudioSession["manifest"]): Float32Array {
  const maxPast = 12;
  const count = maxPast + 1;
  const output = new Float32Array(count * manifest.embeddingLength);
  const numTimescales = manifest.embeddingLength / 2;
  const increment = Math.log(10000) / Math.max(numTimescales - 1, 1);
  for (let position = 0; position < count; position += 1) {
    const positionId = maxPast - position;
    for (let index = 0; index < numTimescales; index += 1) {
      const scaled = positionId * Math.exp(-index * increment);
      output[position * manifest.embeddingLength + index] = Math.sin(scaled);
      output[position * manifest.embeddingLength + numTimescales + index] = Math.cos(scaled);
    }
  }
  return output;
}

function uint32Mask(mask: Uint8Array): Uint32Array {
  const output = new Uint32Array(mask.length);
  for (let index = 0; index < mask.length; index += 1) {
    output[index] = mask[index] ?? 0;
  }
  return output;
}

function downsampleMaskByTwo(mask: Uint32Array, outputLength: number): Uint32Array {
  const output = new Uint32Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    output[index] = mask[index * 2] ?? 0;
  }
  return output;
}

function numberOption(options: Readonly<Record<string, unknown>> | undefined, name: string): number | undefined {
  const value = options?.[name];
  return typeof value === "number" ? value : undefined;
}

function isDenseType(type: GgmlTypeName): boolean {
  return type === "F32" || type === "F16" || type === "BF16";
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Generation was aborted.", "AbortError");
  }
}

function unsupported(message: string): never {
  throw new WebGpuAudioFallbackError(message);
}

function isFallbackError(error: unknown): boolean {
  return error instanceof WebGpuAudioFallbackError ||
    (error instanceof Error && error.message.includes("WebGPU memory cap exceeded"));
}

class WebGpuAudioFallbackError extends Error {}
