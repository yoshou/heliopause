import assert from "node:assert/strict";
import test from "node:test";

import {
  createInferenceState,
  ensureSlidingKvCacheReserve,
} from "../src/runtime.ts";
import type { ModelManifest } from "../src/model.ts";
import { GpuMemoryArena } from "../src/runner/webgpu/arena.ts";
import { GPU_STORAGE } from "../src/runner/webgpu/gpu-constants.ts";
import type { WebGpuBufferLike, WebGpuDeviceLike } from "../src/runner/webgpu/gpu-types.ts";
import { WebGpuSegmentRunner } from "../src/runner/webgpu/segment-runner.ts";

test("WebGPU segment runner resizes existing sliding KV buffers when MTP reserve grows", async () => {
  const { device, records } = fakeDevice();
  const arena = new GpuMemoryArena(device, 1 << 20);
  const manifest = slidingManifest();
  const state = createInferenceState(manifest);
  const runner = webGpuRunnerForResizeTest(arena, manifest);

  const initialGpuState = runner.ensureGpuState(state);
  assert.equal(initialGpuState.fullAttention.get(0)?.capacity, 2);
  assert.deepEqual(kvCacheBufferSizes(records.buffers), [8, 12]);

  state.nextPosition = 2;
  ensureSlidingKvCacheReserve(state, manifest, 2);
  const resizedGpuState = runner.ensureGpuState(state);

  assert.equal(resizedGpuState.fullAttention.get(0)?.capacity, 4);
  assert.deepEqual(kvCacheBufferSizes(records.buffers), [8, 12, 16, 24]);
  assert.equal(records.submits, 1, "resize should submit a GPU copy pass");
  assert.deepEqual(records.dispatches, [1, 1], "key and value caches should each be copied by compute");
  assert.deepEqual(records.uniformWrites, [
    [0, 2, 2, 4, 2, 0, 0, 0],
    [0, 2, 2, 4, 3, 1, 0, 0],
  ]);

  await Promise.resolve();
  assert.deepEqual(
    records.destroyed.filter((label) => label?.includes(".gpu.") || label === undefined),
    [undefined, undefined, "blk.0.gpu.value_cache", "blk.0.gpu.key_cache"],
  );
});

function webGpuRunnerForResizeTest(
  arena: GpuMemoryArena,
  manifest: ModelManifest,
): WebGpuSegmentRunner & { ensureGpuState(state: unknown): { fullAttention: Map<number, { capacity: number }> } } {
  const buffer = arena.createBuffer("dummy", 4, GPU_STORAGE);
  const f32Handle = { buffer, length: 1, byteLength: 4, destroy() {} };
  const layer = {
    kind: "sliding-attention",
    layer: 0,
    hasKv: true,
    kvSourceLayer: 0,
    headSize: 2,
    valueSize: 3,
    headCountKv: 1,
    valueProjectionMode: "separate",
    attnNorm: f32Handle,
    q: f32Handle,
    k: f32Handle,
    v: f32Handle,
    qNorm: f32Handle,
    kNorm: f32Handle,
    attnOut: f32Handle,
    postAttentionNorm: f32Handle,
    ffnNorm: f32Handle,
    ffnGate: f32Handle,
    ffnUp: f32Handle,
    ffnDown: f32Handle,
    postFfwNorm: f32Handle,
    layerOutputScale: f32Handle,
  };
  return new (WebGpuSegmentRunner as never as new (...args: unknown[]) => WebGpuSegmentRunner)(
    arena,
    manifest,
    {},
    1e-6,
    [layer],
    f32Handle,
    false,
    undefined,
    undefined,
    0,
    1,
    64,
    0,
  ) as WebGpuSegmentRunner & { ensureGpuState(state: unknown): { fullAttention: Map<number, { capacity: number }> } };
}

function slidingManifest(): ModelManifest {
  return {
    architecture: "gemma4",
    tensorCount: 0,
    blockCount: 1,
    embeddingLength: 4,
    feedForwardLength: 4,
    headCount: 1,
    headCountKv: 1,
    keyLength: 2,
    valueLength: 3,
    slidingKeyLength: 2,
    slidingValueLength: 3,
    layerHeadCountKv: [],
    layerKeyLengths: [],
    layerValueLengths: [],
    contextLength: 8,
    slidingWindow: 2,
    layerKinds: ["sliding-attention"],
    slidingAttentionLayerCount: 1,
    fullAttentionLayerCount: 0,
    slidingAttentionLayers: [0],
    fullAttentionLayers: [],
    layerHasKv: [true],
    kvSourceLayers: [0],
    layerValueProjectionModes: ["separate"],
    perLayerEmbeddingLength: 0,
    rope: {
      slidingDimensionCount: 2,
      fullDimensionCount: 2,
      slidingFreqBase: 10_000,
      fullFreqBase: 1_000_000,
    },
    finalLogitSoftcap: 0,
    expectedTensors: [],
    tensorTypes: {},
  };
}

function kvCacheBufferSizes(buffers: readonly { label?: string; size: number }[]): number[] {
  return buffers
    .filter((buffer) => buffer.label?.endsWith(".gpu.key_cache") || buffer.label?.endsWith(".gpu.value_cache"))
    .map((buffer) => buffer.size);
}

function fakeDevice(): {
  device: WebGpuDeviceLike;
  records: {
    buffers: Array<{ label?: string; size: number }>;
    destroyed: Array<string | undefined>;
    uniformWrites: number[][];
    dispatches: number[];
    submits: number;
  };
} {
  const records = {
    buffers: [] as Array<{ label?: string; size: number }>,
    destroyed: [] as Array<string | undefined>,
    uniformWrites: [] as number[][],
    dispatches: [] as number[],
    submits: 0,
  };
  const stub = () => ({});
  return {
    records,
    device: {
      createBuffer(descriptor) {
        records.buffers.push({ label: descriptor.label, size: descriptor.size });
        return fakeBuffer(descriptor.label, records.destroyed);
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
          dispatchWorkgroups(x) {
            records.dispatches.push(x);
          },
          end() {},
        }),
        copyBufferToBuffer() {},
        finish: () => ({}),
      }),
      queue: {
        writeBuffer(_buffer, _bufferOffset, data) {
          if (data instanceof Uint32Array) {
            records.uniformWrites.push(Array.from(data));
          }
        },
        submit() {
          records.submits += 1;
        },
        onSubmittedWorkDone: async () => {},
      },
    },
  };
}

function fakeBuffer(label: string | undefined, destroyed: Array<string | undefined>): WebGpuBufferLike {
  return {
    getMappedRange: () => new ArrayBuffer(0),
    unmap() {},
    mapAsync: async () => {},
    destroy: () => {
      destroyed.push(label);
    },
  };
}
