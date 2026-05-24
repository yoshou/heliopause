import assert from "node:assert/strict";
import test from "node:test";

import {
  diffWebGpuRuntimeResourceStats,
  installWebGpuRuntimeResourceCache,
} from "../src/runner/webgpu/runtime-resources.ts";
import { GPU_COPY_DST, GPU_MAP_READ } from "../src/runner/webgpu/gpu-constants.ts";
import type { WebGpuDeviceLike } from "../src/runner/webgpu/gpu-types.ts";

test("WebGPU runtime resource cache creates pipelines and layouts only once per descriptor", () => {
  const { device, counts } = fakeDevice();
  const cache = installWebGpuRuntimeResourceCache(device);

  const moduleA = device.createShaderModule({ code: "kernel-a" });
  const moduleB = device.createShaderModule({ code: "kernel-a" });
  assert.equal(moduleA, moduleB);
  assert.equal(counts.shaderModules, 1);

  const layoutA = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: 4, buffer: { type: "storage" } }],
  });
  const layoutB = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: 4, buffer: { type: "storage" } }],
  });
  assert.equal(layoutA, layoutB);
  assert.equal(counts.bindGroupLayouts, 1);

  const pipelineLayoutA = device.createPipelineLayout({ bindGroupLayouts: [layoutA] });
  const pipelineLayoutB = device.createPipelineLayout({ bindGroupLayouts: [layoutB] });
  assert.equal(pipelineLayoutA, pipelineLayoutB);
  assert.equal(counts.pipelineLayouts, 1);

  const pipelineA = device.createComputePipeline({
    layout: pipelineLayoutA,
    compute: { module: moduleA, entryPoint: "main" },
  });
  const pipelineB = device.createComputePipeline({
    layout: pipelineLayoutB,
    compute: { module: moduleB, entryPoint: "main" },
  });
  assert.equal(pipelineA, pipelineB);
  assert.equal(counts.computePipelines, 1);

  const stats = cache.stats();
  assert.equal(stats.shaderModuleMisses, 1);
  assert.equal(stats.shaderModuleHits, 1);
  assert.equal(stats.bindGroupLayoutMisses, 1);
  assert.equal(stats.bindGroupLayoutHits, 1);
  assert.equal(stats.pipelineLayoutMisses, 1);
  assert.equal(stats.pipelineLayoutHits, 1);
  assert.equal(stats.computePipelineMisses, 1);
  assert.equal(stats.computePipelineHits, 1);
});

test("WebGPU runtime resource cache does not reuse bind groups", () => {
  const { device, counts } = fakeDevice();
  const cache = installWebGpuRuntimeResourceCache(device);
  const layout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: 4, buffer: { type: "storage" } }],
  });
  const buffer = device.createBuffer({ size: 4, usage: 1 });

  const bindGroupA = device.createBindGroup({
    layout,
    entries: [{ binding: 0, resource: { buffer } }],
  });
  const bindGroupB = device.createBindGroup({
    layout,
    entries: [{ binding: 0, resource: { buffer } }],
  });

  assert.notEqual(bindGroupA, bindGroupB);
  assert.equal(counts.bindGroups, 2);
  const stats = cache.stats();
  assert.equal(stats.bindGroupMisses, 2);
  assert.equal(stats.bindGroupHits, 0);
  assert.equal(stats.bindGroupCreates, 2);
});

test("WebGPU runtime resource cache reuses unmapped readback buffers", async () => {
  const { device, counts } = fakeDevice();
  const cache = installWebGpuRuntimeResourceCache(device);

  const bufferA = device.createBuffer({ size: 4, usage: GPU_MAP_READ | GPU_COPY_DST });
  await bufferA.mapAsync(GPU_MAP_READ);
  bufferA.unmap();
  bufferA.destroy?.();
  const beforeSecondCreate = cache.stats();

  const bufferB = device.createBuffer({ size: 4, usage: GPU_MAP_READ | GPU_COPY_DST });

  assert.equal(bufferA, bufferB);
  assert.equal(counts.buffers, 1);
  assert.equal(cache.stats().bufferCreates, beforeSecondCreate.bufferCreates);
});

test("WebGPU runtime resource stats diff returns per-run deltas", () => {
  const { device } = fakeDevice();
  const cache = installWebGpuRuntimeResourceCache(device);
  const before = cache.stats();

  const module = device.createShaderModule({ code: "kernel-b" });
  const layout = device.createBindGroupLayout({ entries: [] });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: "main" } });

  const diff = diffWebGpuRuntimeResourceStats(cache.stats(), before);
  assert.equal(diff.shaderModuleMisses, 1);
  assert.equal(diff.bindGroupLayoutMisses, 1);
  assert.equal(diff.pipelineLayoutMisses, 1);
  assert.equal(diff.computePipelineMisses, 1);
});

function fakeDevice(): {
  device: WebGpuDeviceLike;
  counts: {
    buffers: number;
    shaderModules: number;
    bindGroupLayouts: number;
    pipelineLayouts: number;
    computePipelines: number;
    bindGroups: number;
  };
} {
  const counts = {
    buffers: 0,
    shaderModules: 0,
    bindGroupLayouts: 0,
    pipelineLayouts: 0,
    computePipelines: 0,
    bindGroups: 0,
  };
  return {
    counts,
    device: {
      createBuffer() {
        counts.buffers += 1;
        return {
          getMappedRange: () => new ArrayBuffer(0),
          unmap() {},
          mapAsync: async () => {},
          destroy() {},
        };
      },
      createShaderModule() {
        counts.shaderModules += 1;
        return { kind: "shader", id: counts.shaderModules };
      },
      createBindGroupLayout() {
        counts.bindGroupLayouts += 1;
        return { kind: "bindGroupLayout", id: counts.bindGroupLayouts };
      },
      createPipelineLayout() {
        counts.pipelineLayouts += 1;
        return { kind: "pipelineLayout", id: counts.pipelineLayouts };
      },
      createComputePipeline() {
        counts.computePipelines += 1;
        return { kind: "computePipeline", id: counts.computePipelines };
      },
      createBindGroup() {
        counts.bindGroups += 1;
        return { kind: "bindGroup", id: counts.bindGroups };
      },
      createCommandEncoder() {
        return {
          beginComputePass() {
            return {
              setPipeline() {},
              setBindGroup() {},
              dispatchWorkgroups() {},
              end() {},
            };
          },
          copyBufferToBuffer() {},
          finish: () => ({}),
        };
      },
      queue: {
        writeBuffer() {},
        submit() {},
        onSubmittedWorkDone: async () => {},
      },
    },
  };
}
