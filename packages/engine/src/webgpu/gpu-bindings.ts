import { GPU_SHADER_STAGE_COMPUTE, GPU_STORAGE } from "./gpu-constants";
import type { WebGpuBufferLike, WebGpuDeviceLike } from "./gpu-types";

export function storageBuffer(
  device: WebGpuDeviceLike,
  size: number,
  extraUsage: number,
): WebGpuBufferLike {
  return device.createBuffer({
    size,
    usage: GPU_STORAGE | extraUsage,
  });
}

export function storageEntry(binding: number, type: "read-only-storage" | "storage"): unknown {
  return {
    binding,
    visibility: GPU_SHADER_STAGE_COMPUTE,
    buffer: { type },
  };
}

export function bindBuffer(binding: number, buffer: WebGpuBufferLike): unknown {
  return {
    binding,
    resource: { buffer },
  };
}
