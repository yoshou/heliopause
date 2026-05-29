import assert from "node:assert/strict";
import test from "node:test";

import { GpuMemoryArena } from "../src/runner/webgpu/arena.ts";
import { GpuBuffer } from "../src/runner/webgpu/gpu-buffer.ts";
import { GPU_COPY_SRC, GPU_STORAGE } from "../src/runner/webgpu/gpu-constants.ts";
import type { WebGpuBufferLike, WebGpuDeviceLike } from "../src/runner/webgpu/gpu-types.ts";

const SCRATCH_USAGE = GPU_STORAGE | GPU_COPY_SRC;

test("arena scratch buffers are reused from the pool after destroy", () => {
  const { device, counts } = fakeDevice();
  const arena = new GpuMemoryArena(device, 1 << 20);

  const first = arena.createScratchBuffer("scratch", 64, SCRATCH_USAGE);
  assert.equal(counts.created, 1);
  assert.equal(arena.residentBytes, 64);

  // destroy() returns the buffer to the pool rather than freeing it.
  first.destroy?.();
  assert.equal(arena.residentBytes, 64);

  const second = arena.createScratchBuffer("scratch", 64, SCRATCH_USAGE);
  assert.equal(second, first, "same wrapper instance is handed back from the pool");
  assert.equal(counts.created, 1, "no new GPU buffer allocated on reuse");
  assert.equal(arena.residentBytes, 64);
});

test("arena scratch buffers with different keys are not interchangeable", () => {
  const { device, counts } = fakeDevice();
  const arena = new GpuMemoryArena(device, 1 << 20);

  const a = arena.createScratchBuffer("a", 64, SCRATCH_USAGE);
  a.destroy?.();
  // Different byte length -> different pool key -> fresh allocation.
  const b = arena.createScratchBuffer("b", 128, SCRATCH_USAGE);

  assert.notEqual(b, a);
  assert.equal(counts.created, 2);
});

test("destroyScratchBuffers frees pooled buffers and their resident bytes", () => {
  const { device, counts } = fakeDevice();
  const arena = new GpuMemoryArena(device, 1 << 20);

  const buffer = arena.createScratchBuffer("scratch", 64, SCRATCH_USAGE);
  buffer.destroy?.();
  assert.equal(arena.residentBytes, 64);

  arena.destroyScratchBuffers();
  assert.equal(arena.residentBytes, 0);
  assert.equal(counts.destroyed, 1);
});

test("arena buffer destroy releases bytes exactly once", () => {
  const { device, counts } = fakeDevice();
  const arena = new GpuMemoryArena(device, 1 << 20);

  const buffer = arena.createBuffer("weight", 100, GPU_STORAGE);
  assert.equal(arena.residentBytes, 100);
  assert.equal(arena.peakResidentBytes, 100);

  buffer.destroy?.();
  assert.equal(arena.residentBytes, 0);
  assert.equal(counts.destroyed, 1);

  // Idempotent: a second destroy neither double-decrements nor double-frees.
  buffer.destroy?.();
  assert.equal(arena.residentBytes, 0);
  assert.equal(counts.destroyed, 1);
});

test("peak resident bytes retains the high-water mark after a free", () => {
  const { device } = fakeDevice();
  const arena = new GpuMemoryArena(device, 1 << 20);

  const a = arena.createBuffer("a", 64, GPU_STORAGE);
  const b = arena.createBuffer("b", 64, GPU_STORAGE);
  assert.equal(arena.residentBytes, 128);
  assert.equal(arena.peakResidentBytes, 128);

  a.destroy?.();
  assert.equal(arena.residentBytes, 64);
  assert.equal(arena.peakResidentBytes, 128);

  b.destroy?.();
});

test("arena enforces the memory cap", () => {
  const { device } = fakeDevice();
  const arena = new GpuMemoryArena(device, 64);

  arena.createBuffer("fits", 64, GPU_STORAGE);
  assert.throws(() => arena.createBuffer("overflow", 4, GPU_STORAGE), /memory cap exceeded/);
});

test("arena buffers delegate map operations to the raw buffer", async () => {
  const { device } = fakeDevice();
  const arena = new GpuMemoryArena(device, 1 << 20);

  const buffer = arena.createBuffer("readback", 16, GPU_STORAGE);
  assert.ok(buffer instanceof GpuBuffer);
  const raw = (buffer as GpuBuffer).raw as FakeRawBuffer;

  await buffer.mapAsync(1);
  assert.equal(raw.mapAsyncCalls, 1);
  buffer.getMappedRange();
  assert.equal(raw.getMappedRangeCalls, 1);
  buffer.unmap();
  assert.equal(raw.unmapCalls, 1);
});

type FakeRawBuffer = WebGpuBufferLike & {
  mapAsyncCalls: number;
  getMappedRangeCalls: number;
  unmapCalls: number;
};

function fakeDevice(): {
  device: WebGpuDeviceLike;
  counts: { created: number; destroyed: number };
} {
  const counts = { created: 0, destroyed: 0 };
  const stub = () => ({});
  return {
    counts,
    device: {
      createBuffer() {
        counts.created += 1;
        const raw: FakeRawBuffer = {
          mapAsyncCalls: 0,
          getMappedRangeCalls: 0,
          unmapCalls: 0,
          mapAsync: async () => {
            raw.mapAsyncCalls += 1;
          },
          getMappedRange: () => {
            raw.getMappedRangeCalls += 1;
            return new ArrayBuffer(0);
          },
          unmap: () => {
            raw.unmapCalls += 1;
          },
          destroy: () => {
            counts.destroyed += 1;
          },
        };
        return raw;
      },
      createShaderModule: stub,
      createBindGroupLayout: stub,
      createPipelineLayout: stub,
      createComputePipeline: stub,
      createBindGroup: stub,
      createCommandEncoder: () => ({
        beginComputePass: () => ({
          setPipeline() {},
          setBindGroup() {},
          dispatchWorkgroups() {},
          end() {},
        }),
        copyBufferToBuffer() {},
        finish: () => ({}),
      }),
      queue: {
        writeBuffer() {},
        submit() {},
        onSubmittedWorkDone: async () => {},
      },
    },
  };
}
