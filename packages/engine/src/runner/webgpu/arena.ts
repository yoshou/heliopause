import { GPU_COPY_SRC, GPU_STORAGE } from "./gpu-constants";
import { GpuBuffer, unwrapGpuBuffer } from "./gpu-buffer";
import type { WebGpuBufferLike, WebGpuDeviceLike, WebGpuF32TensorHandleInternal, WebGpuQuantizedWeightHandleInternal } from "./gpu-types";

export type GpuResource = {
  destroy?: () => void;
};

export type QuantizedHandle = WebGpuQuantizedWeightHandleInternal;
export type F32Handle = WebGpuF32TensorHandleInternal;

/** Buffer whose `destroy()` releases its bytes back to the owning arena exactly once. */
class ArenaBuffer extends GpuBuffer {
  #destroyed = false;
  #release: () => void;

  constructor(raw: WebGpuBufferLike, byteLength: number, release: () => void) {
    super(raw, byteLength);
    this.#release = release;
  }

  override destroy(): void {
    if (this.#destroyed) {
      return;
    }
    this.#destroyed = true;
    this.#release();
    this.raw.destroy?.();
  }
}

/**
 * Pooled scratch buffer. `destroy()` returns the buffer to the arena's scratch
 * pool for reuse; the real GPU buffer is released only via `destroyForReal()`
 * when the pool is drained.
 */
class ScratchBuffer extends GpuBuffer {
  inPool = false;
  readonly poolKey: string;
  #destroyed = false;
  #returnToPool: (buffer: ScratchBuffer) => void;
  #release: () => void;

  constructor(
    raw: WebGpuBufferLike,
    byteLength: number,
    poolKey: string,
    returnToPool: (buffer: ScratchBuffer) => void,
    release: () => void,
  ) {
    super(raw, byteLength);
    this.poolKey = poolKey;
    this.#returnToPool = returnToPool;
    this.#release = release;
  }

  override destroy(): void {
    if (this.#destroyed || this.inPool) {
      return;
    }
    this.inPool = true;
    this.#returnToPool(this);
  }

  destroyForReal(): void {
    if (this.#destroyed) {
      return;
    }
    this.#destroyed = true;
    this.#release();
    this.raw.destroy?.();
  }
}

export class GpuMemoryArena {
  device: WebGpuDeviceLike;
  readonly limitBytes: number;
  private allocatedBytes = 0;
  private peakAllocatedBytes = 0;
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

  get peakResidentBytes(): number {
    return this.peakAllocatedBytes;
  }

  resetPeakResidentBytes(): void {
    this.peakAllocatedBytes = this.allocatedBytes;
  }

  createBuffer(label: string, size: number, usage: number, mappedAtCreation = false): WebGpuBufferLike {
    const byteLength = align4(size);
    if (this.allocatedBytes + byteLength > this.limitBytes) {
      throw new Error(
        `WebGPU memory cap exceeded while allocating ${label}: ` +
          `${this.allocatedBytes + byteLength} > ${this.limitBytes}`,
      );
    }
    const raw = unwrapGpuBuffer(this.device.createBuffer({
      label,
      size: byteLength,
      usage,
      mappedAtCreation,
    }));
    this.allocatedBytes += byteLength;
    this.peakAllocatedBytes = Math.max(this.peakAllocatedBytes, this.allocatedBytes);
    return new ArenaBuffer(raw, byteLength, () => {
      this.allocatedBytes -= byteLength;
    });
  }

  createScratchBuffer(label: string, size: number, usage: number): WebGpuBufferLike {
    const byteLength = align4(size);
    const key = `${usage}:${byteLength}`;
    const pooled = this.scratchPools.get(key)?.pop();
    if (pooled) {
      pooled.inPool = false;
      return pooled;
    }
    if (this.allocatedBytes + byteLength > this.limitBytes) {
      throw new Error(
        `WebGPU memory cap exceeded while allocating scratch buffer: ` +
          `${this.allocatedBytes + byteLength} > ${this.limitBytes}`,
      );
    }
    const raw = unwrapGpuBuffer(this.device.createBuffer({
      label,
      size: byteLength,
      usage,
    }));
    this.allocatedBytes += byteLength;
    this.peakAllocatedBytes = Math.max(this.peakAllocatedBytes, this.allocatedBytes);
    return new ScratchBuffer(
      raw,
      byteLength,
      key,
      (buffer) => this.returnScratch(buffer),
      () => {
        this.allocatedBytes -= byteLength;
      },
    );
  }

  destroyScratchBuffers(): void {
    for (const pool of this.scratchPools.values()) {
      for (const buffer of pool) {
        buffer.destroyForReal();
      }
    }
    this.scratchPools.clear();
  }

  private returnScratch(buffer: ScratchBuffer): void {
    const pool = this.scratchPools.get(buffer.poolKey);
    if (pool) {
      pool.push(buffer);
    } else {
      this.scratchPools.set(buffer.poolKey, [buffer]);
    }
  }
}

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
  const buffer = arena.createScratchBuffer(label, length * Float32Array.BYTES_PER_ELEMENT, GPU_STORAGE | GPU_COPY_SRC);
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
