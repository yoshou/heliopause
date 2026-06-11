import assert from "node:assert/strict";
import test from "node:test";

import type { GgufMetadata } from "../src/gguf.ts";
import type { ModelManifest } from "../src/model.ts";
import { createModelResourceRequirements } from "../src/runner/model-resources.ts";
import { createWebGpuProvider } from "../src/runner/webgpu/index.ts";
import { webGpuExecutionProviderOptions } from "../src/runner/webgpu/execution-provider.ts";
import { webGpuResourceRequirements } from "../src/runner/webgpu/planning.ts";

test("model resource requirements can estimate KV cache with native f16 element size", () => {
  const f32 = createModelResourceRequirements({
    provider: "webgpu",
    gguf: emptyGguf(),
    manifest: manifestWithKv(),
    contextLength: 3,
  });
  const f16 = createModelResourceRequirements({
    provider: "webgpu",
    gguf: emptyGguf(),
    manifest: manifestWithKv(),
    contextLength: 3,
    cacheElementByteLength: 2,
  });

  assert.equal(f32.layers[0]?.cacheBytes, 240);
  assert.equal(f16.layers[0]?.cacheBytes, 120);
  assert.equal(f16.layers[0]?.cacheBytes, (f32.layers[0]?.cacheBytes ?? 0) / 2);

  const webgpu = webGpuResourceRequirements(emptyGguf(), manifestWithKv(), { contextLength: 3 });
  assert.equal(webgpu.layers[0]?.cacheBytes, f16.layers[0]?.cacheBytes);
});

test("model resource requirements estimate sliding KV cache with ring capacity", () => {
  const manifest = {
    ...manifestWithKv(),
    contextLength: 16,
    slidingWindow: 4,
    layerKinds: ["sliding-attention"],
    slidingAttentionLayerCount: 1,
    fullAttentionLayerCount: 0,
    slidingAttentionLayers: [0],
    fullAttentionLayers: [],
  } satisfies ModelManifest;
  const requirements = createModelResourceRequirements({
    provider: "webgpu",
    gguf: emptyGguf(),
    manifest,
    contextLength: 16,
    cacheElementByteLength: 2,
  });

  assert.equal(requirements.layers[0]?.cacheBytes, 160);
});

test("model resource requirements include sliding KV reserve tokens", () => {
  const manifest = {
    ...manifestWithKv(),
    contextLength: 16,
    slidingWindow: 4,
    layerKinds: ["sliding-attention"],
    slidingAttentionLayerCount: 1,
    fullAttentionLayerCount: 0,
    slidingAttentionLayers: [0],
    fullAttentionLayers: [],
  } satisfies ModelManifest;
  const requirements = createModelResourceRequirements({
    provider: "webgpu",
    gguf: emptyGguf(),
    manifest,
    contextLength: 16,
    cacheElementByteLength: 2,
    slidingWindowReserveTokens: 3,
  });
  const webgpu = webGpuResourceRequirements(emptyGguf(), manifest, {
    contextLength: 16,
    slidingWindowReserveTokens: 3,
  });

  assert.equal(requirements.layers[0]?.cacheBytes, 280);
  assert.equal(webgpu.layers[0]?.cacheBytes, requirements.layers[0]?.cacheBytes);
});

test("WebGPU provider forwards sliding KV reserve tokens to resource planning", () => {
  const manifest = {
    ...manifestWithKv(),
    contextLength: 16,
    slidingWindow: 4,
    layerKinds: ["sliding-attention"],
    slidingAttentionLayerCount: 1,
    fullAttentionLayerCount: 0,
    slidingAttentionLayers: [0],
    fullAttentionLayers: [],
  } satisfies ModelManifest;
  const provider = createWebGpuProvider();
  const requirements = provider.modelResourceRequirements({
    tensorReader: { metadata: emptyGguf() },
    manifest,
    provider: (name: string) => name === "webgpu" ? provider : undefined,
  } as never, {
    contextLength: 16,
    slidingWindowReserveTokens: 3,
  });

  assert.equal(requirements.layers[0]?.cacheBytes, 280);
});

test("WebGPU provider options normalize profiling, allocation tracking, and optimization level", () => {
  const defaults = webGpuExecutionProviderOptions(fakeSession(createWebGpuProvider()));
  assert.equal(defaults?.optimizationLevel, "standard");
  assert.equal(defaults?.gpuProfiling, false);
  assert.equal(defaults?.trackBufferAllocations, false);

  const configured = webGpuExecutionProviderOptions(fakeSession(createWebGpuProvider({
    memoryLimitBytes: 123,
    segmentStartLayer: 2,
    prefillChunkSize: 8,
    optimizationLevel: "baseline",
    gpuProfiling: true,
    trackBufferAllocations: true,
  })));
  assert.equal(configured?.memoryLimitBytes, 123);
  assert.equal(configured?.segmentStartLayer, 2);
  assert.equal(configured?.prefillChunkSize, 8);
  assert.equal(configured?.optimizationLevel, "baseline");
  assert.equal(configured?.gpuProfiling, true);
  assert.equal(configured?.trackBufferAllocations, true);
});

function emptyGguf(): GgufMetadata {
  return {
    version: 3,
    tensorCount: 0,
    metadataCount: 0,
    metadata: {},
    tensors: [],
    dataStart: 0n,
  };
}

function fakeSession(provider: ReturnType<typeof createWebGpuProvider>) {
  return {
    provider: (name: string) => name === "webgpu" ? provider : undefined,
  } as never;
}

function manifestWithKv(): ModelManifest {
  return {
    architecture: "gemma4",
    tensorCount: 0,
    blockCount: 1,
    embeddingLength: 8,
    feedForwardLength: 16,
    headCount: 2,
    headCountKv: 2,
    layerHeadCountKv: [2],
    keyLength: 4,
    valueLength: 6,
    slidingKeyLength: 4,
    slidingValueLength: 6,
    layerKeyLengths: [4],
    layerValueLengths: [6],
    layerHasKv: [true],
    layerValueProjectionModes: ["separate"],
    kvSourceLayers: [0],
    contextLength: 16_384,
    slidingWindow: 0,
    layerKinds: ["full-attention"],
    slidingAttentionLayerCount: 0,
    fullAttentionLayerCount: 1,
    slidingAttentionLayers: [],
    fullAttentionLayers: [0],
    perLayerEmbeddingLength: 8,
    rope: {
      slidingDimensionCount: 4,
      fullDimensionCount: 4,
      dimensionCount: 4,
      dimensionSections: [1, 1, 0, 0],
      slidingFreqBase: 10_000,
      fullFreqBase: 10_000,
      freqBase: 10_000,
    },
    tensorTypes: {},
    expectedTensors: [],
  };
}
