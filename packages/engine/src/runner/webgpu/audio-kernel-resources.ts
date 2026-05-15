import { bindBuffer, storageEntry } from "./gpu-bindings";
import { GPU_COPY_DST, GPU_SHADER_STAGE_COMPUTE, GPU_UNIFORM } from "./gpu-constants";
import type { WebGpuBufferLike, WebGpuDeviceLike } from "./gpu-types";
import {
  AUDIO_ADD_BIAS_ROWS_WGSL,
  AUDIO_ATTENTION_WGSL,
  AUDIO_CONV2D_SUBSAMPLE_WGSL,
  AUDIO_DEPTHWISE_CONV1D_WGSL,
  AUDIO_FLATTEN_CHANNELS_LAST_WGSL,
  AUDIO_FFT_BIT_REVERSE_WGSL,
  AUDIO_FFT_STAGE_WGSL,
  AUDIO_GLU_WGSL,
  AUDIO_LOG_MEL_WGSL,
  AUDIO_RESIDUAL_ADD_SCALE_WGSL,
  AUDIO_SILU_WGSL,
  AUDIO_WINDOW_FRAMES_WGSL,
  VISION_CLAMP_WGSL,
  VISION_RMS_NORM_WGSL,
} from "./shaders";

type Resource = { pipeline: unknown; bindGroup: unknown; destroy: () => void };

export function createAudioWindowFramesResources(
  device: WebGpuDeviceLike,
  pcm: WebGpuBufferLike,
  window: WebGpuBufferLike,
  real: WebGpuBufferLike,
  imag: WebGpuBufferLike,
  options: {
    sampleCount: number;
    frameCount: number;
    frameLength: number;
    hopLength: number;
    fftLength: number;
  },
): Resource {
  const params = new Uint32Array([
    options.sampleCount,
    options.frameCount,
    options.frameLength,
    options.hopLength,
    options.fftLength,
    Math.floor(options.frameLength / 2),
    0,
    0,
  ]);
  return createResource(device, AUDIO_WINDOW_FRAMES_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
    storageEntry(4, "storage"),
  ], [
    bindBuffer(0, pcm),
    bindBuffer(1, window),
    uniformBuffer(device, 2, params),
    bindBuffer(3, real),
    bindBuffer(4, imag),
  ]);
}

export function createAudioFftBitReverseResources(
  device: WebGpuDeviceLike,
  inputReal: WebGpuBufferLike,
  inputImag: WebGpuBufferLike,
  outputReal: WebGpuBufferLike,
  outputImag: WebGpuBufferLike,
  options: { frameCount: number; fftLength: number },
): Resource {
  const params = new Uint32Array([
    options.frameCount,
    options.fftLength,
    Math.log2(options.fftLength),
    0,
  ]);
  return createResource(device, AUDIO_FFT_BIT_REVERSE_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
    storageEntry(4, "storage"),
  ], [
    bindBuffer(0, inputReal),
    bindBuffer(1, inputImag),
    uniformBuffer(device, 2, params),
    bindBuffer(3, outputReal),
    bindBuffer(4, outputImag),
  ]);
}

export function createAudioFftStageResources(
  device: WebGpuDeviceLike,
  inputReal: WebGpuBufferLike,
  inputImag: WebGpuBufferLike,
  outputReal: WebGpuBufferLike,
  outputImag: WebGpuBufferLike,
  options: { frameCount: number; fftLength: number; stageSize: number },
): Resource {
  const params = new Uint32Array([
    options.frameCount,
    options.fftLength,
    options.stageSize,
    options.stageSize / 2,
  ]);
  return createResource(device, AUDIO_FFT_STAGE_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
    storageEntry(4, "storage"),
  ], [
    bindBuffer(0, inputReal),
    bindBuffer(1, inputImag),
    uniformBuffer(device, 2, params),
    bindBuffer(3, outputReal),
    bindBuffer(4, outputImag),
  ]);
}

export function createAudioLogMelResources(
  device: WebGpuDeviceLike,
  real: WebGpuBufferLike,
  imag: WebGpuBufferLike,
  filters: WebGpuBufferLike,
  output: WebGpuBufferLike,
  mask: WebGpuBufferLike,
  options: {
    frameCount: number;
    fftLength: number;
    featureSize: number;
    melFloor: number;
  },
): Resource {
  const paramsF32 = new Float32Array([0, 0, 0, 0, options.melFloor, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[0] = options.frameCount;
  params[1] = options.fftLength;
  params[2] = options.featureSize;
  params[3] = options.fftLength / 2 + 1;
  return createResource(device, AUDIO_LOG_MEL_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    storageEntry(2, "read-only-storage"),
    uniformEntry(3),
    storageEntry(4, "storage"),
    storageEntry(5, "storage"),
  ], [
    bindBuffer(0, real),
    bindBuffer(1, imag),
    bindBuffer(2, filters),
    uniformBuffer(device, 3, params),
    bindBuffer(4, output),
    bindBuffer(5, mask),
  ]);
}

export function createAudioConv2dSubsampleResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  mask: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  bias: WebGpuBufferLike,
  norm: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    time: number;
    frequency: number;
    inChannels: number;
    outChannels: number;
    outTime: number;
    outFrequency: number;
    hasBias: boolean;
    epsilon: number;
  },
): Resource {
  const paramsF32 = new Float32Array([options.epsilon, 0, 0, 0, 0, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[1] = options.time;
  params[2] = options.frequency;
  params[3] = options.inChannels;
  params[4] = options.outChannels;
  params[5] = options.outTime;
  params[6] = options.outFrequency;
  params[7] = options.hasBias ? 1 : 0;
  return createResource(device, AUDIO_CONV2D_SUBSAMPLE_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    storageEntry(2, "read-only-storage"),
    storageEntry(3, "read-only-storage"),
    storageEntry(4, "read-only-storage"),
    uniformEntry(5),
    storageEntry(6, "storage"),
  ], [
    bindBuffer(0, input),
    bindBuffer(1, mask),
    bindBuffer(2, weight),
    bindBuffer(3, bias),
    bindBuffer(4, norm),
    uniformBuffer(device, 5, params),
    bindBuffer(6, output),
  ]);
}

export function createAudioFlattenChannelsLastResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { timeCount: number; frequencyCount: number; channelCount: number },
): Resource {
  const params = new Uint32Array([options.timeCount, options.frequencyCount, options.channelCount, 0]);
  return createResource(device, AUDIO_FLATTEN_CHANNELS_LAST_WGSL, [
    storageEntry(0, "read-only-storage"),
    uniformEntry(1),
    storageEntry(2, "storage"),
  ], [
    bindBuffer(0, input),
    uniformBuffer(device, 1, params),
    bindBuffer(2, output),
  ]);
}

export function createAudioRmsNormResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { rowCount: number; rowSize: number; epsilon: number },
): Resource {
  const paramsF32 = new Float32Array([options.epsilon, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[1] = options.rowSize;
  params[2] = options.rowCount;
  params[3] = 1;
  return createResource(device, VISION_RMS_NORM_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
  ], [
    bindBuffer(0, input),
    bindBuffer(1, weight),
    uniformBuffer(device, 2, params),
    bindBuffer(3, output),
  ]);
}

export function createAudioClampResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { length: number; min: number; max: number },
): Resource {
  const paramsF32 = new Float32Array([options.min, options.max, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[2] = options.length;
  return createResource(device, VISION_CLAMP_WGSL, [
    storageEntry(0, "read-only-storage"),
    uniformEntry(1),
    storageEntry(2, "storage"),
  ], [
    bindBuffer(0, input),
    uniformBuffer(device, 1, params),
    bindBuffer(2, output),
  ]);
}

export function createAudioSiluResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { length: number },
): Resource {
  const params = new Uint32Array([options.length, 0, 0, 0]);
  return createResource(device, AUDIO_SILU_WGSL, [
    storageEntry(0, "read-only-storage"),
    uniformEntry(1),
    storageEntry(2, "storage"),
  ], [
    bindBuffer(0, input),
    uniformBuffer(device, 1, params),
    bindBuffer(2, output),
  ]);
}

export function createAudioGluResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { tokenCount: number; outputSize: number },
): Resource {
  const params = new Uint32Array([options.tokenCount, options.outputSize, 0, 0]);
  return createResource(device, AUDIO_GLU_WGSL, [
    storageEntry(0, "read-only-storage"),
    uniformEntry(1),
    storageEntry(2, "storage"),
  ], [
    bindBuffer(0, input),
    uniformBuffer(device, 1, params),
    bindBuffer(2, output),
  ]);
}

export function createAudioDepthwiseConv1dResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { tokenCount: number; kernelSize: number; channels: number },
): Resource {
  const params = new Uint32Array([options.tokenCount, options.kernelSize, options.channels, 0]);
  return createResource(device, AUDIO_DEPTHWISE_CONV1D_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
  ], [
    bindBuffer(0, input),
    bindBuffer(1, weight),
    uniformBuffer(device, 2, params),
    bindBuffer(3, output),
  ]);
}

export function createAudioAddBiasRowsResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  bias: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { length: number; rowSize: number },
): Resource {
  const params = new Uint32Array([options.length, options.rowSize, 0, 0]);
  return createResource(device, AUDIO_ADD_BIAS_ROWS_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
  ], [
    bindBuffer(0, input),
    bindBuffer(1, bias),
    uniformBuffer(device, 2, params),
    bindBuffer(3, output),
  ]);
}

export function createAudioResidualAddScaleResources(
  device: WebGpuDeviceLike,
  residual: WebGpuBufferLike,
  hidden: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { length: number; scale: number },
): Resource {
  const paramsF32 = new Float32Array([options.scale, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[1] = options.length;
  return createResource(device, AUDIO_RESIDUAL_ADD_SCALE_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
  ], [
    bindBuffer(0, residual),
    bindBuffer(1, hidden),
    uniformBuffer(device, 2, params),
    bindBuffer(3, output),
  ]);
}

export function createAudioAttentionResources(
  device: WebGpuDeviceLike,
  q: WebGpuBufferLike,
  k: WebGpuBufferLike,
  v: WebGpuBufferLike,
  relativeK: WebGpuBufferLike,
  mask: WebGpuBufferLike,
  perDimScale: WebGpuBufferLike,
  perDimKScale: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    tokenCount: number;
    headCount: number;
    headSize: number;
    embeddingLength: number;
    attentionChunkSize: number;
    maxPast: number;
    invalidLogit: number;
    logitCap: number;
    qScale: number;
    kScale: number;
    hasPerDimKScale: boolean;
  },
): Resource {
  const paramsF32 = new Float32Array([
    options.invalidLogit,
    options.logitCap,
    options.qScale,
    options.kScale,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  ]);
  const params = new Uint32Array(paramsF32.buffer);
  params[4] = options.tokenCount;
  params[5] = options.headCount;
  params[6] = options.headSize;
  params[7] = options.embeddingLength;
  params[8] = options.attentionChunkSize;
  params[9] = options.maxPast;
  params[10] = options.hasPerDimKScale ? 1 : 0;
  return createResource(device, AUDIO_ATTENTION_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    storageEntry(2, "read-only-storage"),
    storageEntry(3, "read-only-storage"),
    storageEntry(4, "read-only-storage"),
    storageEntry(5, "read-only-storage"),
    storageEntry(6, "read-only-storage"),
    uniformEntry(7),
    storageEntry(8, "storage"),
  ], [
    bindBuffer(0, q),
    bindBuffer(1, k),
    bindBuffer(2, v),
    bindBuffer(3, relativeK),
    bindBuffer(4, mask),
    bindBuffer(5, perDimScale),
    bindBuffer(6, perDimKScale),
    uniformBuffer(device, 7, params),
    bindBuffer(8, output),
  ]);
}

function createResource(
  device: WebGpuDeviceLike,
  code: string,
  layoutEntries: unknown[],
  entries: unknown[],
): Resource {
  const bindGroupLayout = device.createBindGroupLayout({ entries: layoutEntries });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code }),
      entryPoint: "main",
    },
  });
  const params = entries
    .map((entry) => typeof entry === "object" && entry !== null && "resource" in entry ? entry.resource : undefined)
    .filter(isDisposableUniform);
  return {
    pipeline,
    bindGroup: device.createBindGroup({ layout: bindGroupLayout, entries }),
    destroy: () => {
      for (const param of params) {
        param.__buffer.destroy?.();
      }
    },
  };
}

function uniformEntry(binding: number): unknown {
  return {
    binding,
    visibility: GPU_SHADER_STAGE_COMPUTE,
    buffer: { type: "uniform" },
  };
}

function uniformBuffer(device: WebGpuDeviceLike, binding: number, values: Uint32Array): { binding: number; resource: { buffer: WebGpuBufferLike } & DisposableUniform } {
  const buffer = device.createBuffer({ size: values.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(buffer, 0, values);
  return {
    binding,
    resource: Object.assign({ buffer }, { __buffer: buffer }),
  };
}

type DisposableUniform = { __buffer: WebGpuBufferLike };

function isDisposableUniform(value: unknown): value is DisposableUniform {
  return typeof value === "object" && value !== null && "__buffer" in value;
}
