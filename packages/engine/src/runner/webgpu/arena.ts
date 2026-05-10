import { GPU_STORAGE } from "./gpu-constants";
import type { WebGpuBufferLike, WebGpuDeviceLike, WebGpuF32TensorHandleInternal, WebGpuQuantizedWeightHandleInternal } from "./gpu-types";

export type GpuResource = {
  destroy?: () => void;
};

export type QuantizedHandle = WebGpuQuantizedWeightHandleInternal;
export type F32Handle = WebGpuF32TensorHandleInternal;

export class GpuMemoryArena {
  readonly device: WebGpuDeviceLike;
  readonly limitBytes: number;
  private allocatedBytes = 0;
  private scratchPools = new Map<string, ScratchBuffer[]>();

  constructor(
    device: WebGpuDeviceLike,
    limitBytes: number,
  ) {
    this.device = device;
    this.limitBytes = limitBytes;
  }

  get residentBytes(): number {
    return this.allocatedBytes;
  }

  createBuffer(label: string, size: number, usage: number, mappedAtCreation = false): WebGpuBufferLike {
    const byteLength = align4(size);
    if (this.allocatedBytes + byteLength > this.limitBytes) {
      throw new Error(
        `WebGPU memory cap exceeded while allocating ${label}: ` +
          `${this.allocatedBytes + byteLength} > ${this.limitBytes}`,
      );
    }
    const buffer = this.device.createBuffer({
      label,
      size: byteLength,
      usage,
      mappedAtCreation,
    });
    this.allocatedBytes += byteLength;
    const destroy = buffer.destroy?.bind(buffer);
    let destroyed = false;
    buffer.destroy = () => {
      if (destroyed) {
        return;
      }
      destroyed = true;
      this.allocatedBytes -= byteLength;
      destroy?.();
    };
    return buffer;
  }

  createScratchBuffer(label: string, size: number, usage: number): WebGpuBufferLike {
    const byteLength = align4(size);
    const key = `${usage}:${byteLength}`;
    const pool = this.scratchPools.get(key);
    const pooled = pool?.pop();
    if (pooled) {
      pooled.__heliopauseInScratchPool = false;
      return pooled;
    }
    if (this.allocatedBytes + byteLength > this.limitBytes) {
      throw new Error(
        `WebGPU memory cap exceeded while allocating scratch buffer: ` +
          `${this.allocatedBytes + byteLength} > ${this.limitBytes}`,
      );
    }
    const buffer = this.device.createBuffer({
      label,
      size: byteLength,
      usage,
    }) as ScratchBuffer;
    this.allocatedBytes += byteLength;
    const destroy = buffer.destroy?.bind(buffer);
    buffer.__heliopauseScratchKey = key;
    buffer.__heliopauseScratchBytes = byteLength;
    buffer.__heliopauseDestroyScratch = () => {
      if (buffer.__heliopauseScratchDestroyed) {
        return;
      }
      buffer.__heliopauseScratchDestroyed = true;
      this.allocatedBytes -= byteLength;
      destroy?.();
    };
    buffer.destroy = () => {
      if (buffer.__heliopauseScratchDestroyed || buffer.__heliopauseInScratchPool) {
        return;
      }
      buffer.__heliopauseInScratchPool = true;
      const targetPool = this.scratchPools.get(key);
      if (targetPool) {
        targetPool.push(buffer);
      } else {
        this.scratchPools.set(key, [buffer]);
      }
    };
    return buffer;
  }
}

type ScratchBuffer = WebGpuBufferLike & {
  __heliopauseScratchKey?: string;
  __heliopauseScratchBytes?: number;
  __heliopauseInScratchPool?: boolean;
  __heliopauseScratchDestroyed?: boolean;
  __heliopauseDestroyScratch?: () => void;
};

export type Q8KBuffers = {
  scale: WebGpuBufferLike;
  qs: WebGpuBufferLike;
  bsums: WebGpuBufferLike;
};

export type Q8_0Buffers = {
  scale: WebGpuBufferLike;
  qs: WebGpuBufferLike;
};

export function scratchF32(
  arena: GpuMemoryArena,
  length: number,
  cleanup: GpuResource[],
  label: string,
): WebGpuBufferLike {
  const buffer = arena.createScratchBuffer(label, length * Float32Array.BYTES_PER_ELEMENT, GPU_STORAGE);
  cleanup.push(buffer);
  return buffer;
}

export function scratchQ8K(
  arena: GpuMemoryArena,
  inputSize: number,
  columnCount: number,
  cleanup: GpuResource[],
  label: string,
): Q8KBuffers {
  const blockCount = inputSize / 256;
  const buffers = {
    scale: arena.createScratchBuffer(`${label}.scale`, columnCount * blockCount * Float32Array.BYTES_PER_ELEMENT, GPU_STORAGE),
    qs: arena.createScratchBuffer(`${label}.qs`, columnCount * inputSize * Int32Array.BYTES_PER_ELEMENT, GPU_STORAGE),
    bsums: arena.createScratchBuffer(`${label}.bsums`, columnCount * blockCount * 16 * Int32Array.BYTES_PER_ELEMENT, GPU_STORAGE),
  };
  cleanup.push(buffers.scale, buffers.qs, buffers.bsums);
  return buffers;
}

export function scratchQ8_0(
  arena: GpuMemoryArena,
  inputSize: number,
  columnCount: number,
  blockCount: number,
  cleanup: GpuResource[],
  label: string,
): Q8_0Buffers {
  const buffers = {
    scale: arena.createScratchBuffer(`${label}.scale`, columnCount * blockCount * Float32Array.BYTES_PER_ELEMENT, GPU_STORAGE),
    qs: arena.createScratchBuffer(`${label}.qs`, columnCount * inputSize * Int32Array.BYTES_PER_ELEMENT, GPU_STORAGE),
  };
  cleanup.push(buffers.scale, buffers.qs);
  return buffers;
}

export function align4(value: number): number {
  return Math.max(4, Math.ceil(value / 4) * 4);
}
