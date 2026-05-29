import { bindBuffer, storageEntry } from "./gpu-bindings";
import { unwrapGpuBuffer } from "./gpu-buffer";
import { GPU_COPY_DST, GPU_SHADER_STAGE_COMPUTE, GPU_UNIFORM } from "./gpu-constants";
import type { WebGpuBufferLike, WebGpuDeviceLike } from "./gpu-types";
import {
  VISION_ADD_POSITION_WGSL,
  VISION_ATTENTION_APPLY_WGSL,
  VISION_ATTENTION_SCORE_WGSL,
  VISION_AVERAGE_POOL_WGSL,
  VISION_CLAMP_WGSL,
  VISION_GELU_MUL_WGSL,
  VISION_PATCH_EMBED_WGSL,
  VISION_PREPROCESS_RGBA_WGSL,
  VISION_RMS_NORM_WGSL,
  VISION_ROPE2D_WGSL,
  VISION_STD_NORMALIZE_WGSL,
} from "./shaders";

type Resource = { pipeline: unknown; bindGroup: unknown; destroy: () => void };

export function createVisionPreprocessRgbaResources(
  device: WebGpuDeviceLike,
  rgba: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    sourceWidth: number;
    sourceHeight: number;
    targetWidth: number;
    targetHeight: number;
    mean: readonly [number, number, number];
    std: readonly [number, number, number];
  },
): Resource {
  const paramsF32 = new Float32Array([
    0,
    0,
    0,
    0,
    options.mean[0],
    options.mean[1],
    options.mean[2],
    0,
    options.std[0],
    options.std[1],
    options.std[2],
    0,
  ]);
  const params = new Uint32Array(paramsF32.buffer);
  params[0] = options.sourceWidth;
  params[1] = options.sourceHeight;
  params[2] = options.targetWidth;
  params[3] = options.targetHeight;
  return createResource(device, VISION_PREPROCESS_RGBA_WGSL, [
    storageEntry(0, "read-only-storage"),
    uniformEntry(1),
    storageEntry(2, "storage"),
  ], [
    bindBuffer(0, rgba),
    uniformBuffer(device, 1, params),
    bindBuffer(2, output),
  ]);
}

export function createVisionPatchEmbedResources(
  device: WebGpuDeviceLike,
  pixels: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { imageWidth: number; patchSize: number; patchGridX: number; patchGridY: number; embeddingLength: number },
): Resource {
  const params = new Uint32Array([options.imageWidth, options.patchSize, options.patchGridX, options.patchGridY, options.embeddingLength, 0, 0, 0]);
  return createResource(device, VISION_PATCH_EMBED_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
  ], [
    bindBuffer(0, pixels),
    bindBuffer(1, weight),
    uniformBuffer(device, 2, params),
    bindBuffer(3, output),
  ]);
}

export function createVisionAddPositionResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  position: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { patchGridX: number; tokenCount: number; embeddingLength: number; tableSize: number },
): Resource {
  const params = new Uint32Array([options.patchGridX, options.tokenCount, options.embeddingLength, options.tableSize]);
  return createResource(device, VISION_ADD_POSITION_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
  ], [
    bindBuffer(0, input),
    bindBuffer(1, position),
    uniformBuffer(device, 2, params),
    bindBuffer(3, output),
  ]);
}

export function createVisionRmsNormResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike | undefined,
  output: WebGpuBufferLike,
  options: { rowCount: number; rowSize: number; epsilon: number },
): Resource {
  const paramsF32 = new Float32Array([options.epsilon, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[1] = options.rowSize;
  params[2] = options.rowCount;
  params[3] = weight ? 1 : 0;
  return createResource(device, VISION_RMS_NORM_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
  ], [
    bindBuffer(0, input),
    bindBuffer(1, weight ?? input),
    uniformBuffer(device, 2, params),
    bindBuffer(3, output),
  ]);
}

export function createVisionRope2dResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { patchGridX: number; tokenCount: number; headCount: number; headSize: number; freqBase: number },
): Resource {
  const paramsF32 = new Float32Array([options.freqBase, 0, 0, 0, 0, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[1] = options.patchGridX;
  params[2] = options.tokenCount;
  params[3] = options.headCount;
  params[4] = options.headSize;
  return createResource(device, VISION_ROPE2D_WGSL, [
    storageEntry(0, "read-only-storage"),
    uniformEntry(1),
    storageEntry(2, "storage"),
  ], [
    bindBuffer(0, input),
    uniformBuffer(device, 1, params),
    bindBuffer(2, output),
  ]);
}

export function createVisionAttentionScoreResources(
  device: WebGpuDeviceLike,
  q: WebGpuBufferLike,
  k: WebGpuBufferLike,
  probabilities: WebGpuBufferLike,
  options: { tokenCount: number; headCount: number; headSize: number; scale: number },
): Resource {
  const paramsF32 = new Float32Array([options.scale, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[1] = options.tokenCount;
  params[2] = options.headCount;
  params[3] = options.headSize;
  return createResource(device, VISION_ATTENTION_SCORE_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
  ], [
    bindBuffer(0, q),
    bindBuffer(1, k),
    uniformBuffer(device, 2, params),
    bindBuffer(3, probabilities),
  ]);
}

export function createVisionAttentionApplyResources(
  device: WebGpuDeviceLike,
  v: WebGpuBufferLike,
  probabilities: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { tokenCount: number; headCount: number; headSize: number },
): Resource {
  const params = new Uint32Array([options.tokenCount, options.headCount, options.headSize, 0]);
  return createResource(device, VISION_ATTENTION_APPLY_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
  ], [
    bindBuffer(0, v),
    bindBuffer(1, probabilities),
    uniformBuffer(device, 2, params),
    bindBuffer(3, output),
  ]);
}

export function createVisionAveragePoolResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { patchGridX: number; patchGridY: number; embeddingLength: number; kernelSize: number; outputScale: number },
): Resource {
  const paramsF32 = new Float32Array([options.outputScale, 0, 0, 0, 0, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[1] = options.patchGridX;
  params[2] = options.patchGridY;
  params[3] = options.embeddingLength;
  params[4] = options.kernelSize;
  return createResource(device, VISION_AVERAGE_POOL_WGSL, [
    storageEntry(0, "read-only-storage"),
    uniformEntry(1),
    storageEntry(2, "storage"),
  ], [
    bindBuffer(0, input),
    uniformBuffer(device, 1, params),
    bindBuffer(2, output),
  ]);
}

export function createVisionStdNormalizeResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  bias: WebGpuBufferLike,
  scale: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { length: number; rowSize: number },
): Resource {
  const params = new Uint32Array([options.length, options.rowSize, 0, 0]);
  return createResource(device, VISION_STD_NORMALIZE_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    storageEntry(2, "read-only-storage"),
    uniformEntry(3),
    storageEntry(4, "storage"),
  ], [
    bindBuffer(0, input),
    bindBuffer(1, bias),
    bindBuffer(2, scale),
    uniformBuffer(device, 3, params),
    bindBuffer(4, output),
  ]);
}

export function createVisionClampResources(
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

export function createVisionGeluMulResources(
  device: WebGpuDeviceLike,
  gate: WebGpuBufferLike,
  up: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { length: number },
): Resource {
  const params = new Uint32Array([options.length, 0, 0, 0]);
  return createResource(device, VISION_GELU_MUL_WGSL, [
    storageEntry(0, "read-only-storage"),
    storageEntry(1, "read-only-storage"),
    uniformEntry(2),
    storageEntry(3, "storage"),
  ], [
    bindBuffer(0, gate),
    bindBuffer(1, up),
    uniformBuffer(device, 2, params),
    bindBuffer(3, output),
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
  const disposables = entries.filter(isDisposableBindEntry).map((entry) => entry.dispose);
  return {
    pipeline,
    bindGroup: device.createBindGroup({ layout: bindGroupLayout, entries }),
    destroy: () => {
      for (const buffer of disposables) {
        buffer.destroy?.();
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

function uniformBuffer(device: WebGpuDeviceLike, binding: number, values: Uint32Array): DisposableBindEntry {
  const buffer = device.createBuffer({ size: values.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(buffer, 0, values);
  return {
    binding,
    resource: { buffer: unwrapGpuBuffer(buffer) },
    dispose: buffer,
  };
}

type DisposableBindEntry = {
  binding: number;
  resource: { buffer: WebGpuBufferLike };
  dispose: WebGpuBufferLike;
};

function isDisposableBindEntry(value: unknown): value is DisposableBindEntry {
  return typeof value === "object" && value !== null && "dispose" in value;
}
