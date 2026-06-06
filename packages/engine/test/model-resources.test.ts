import assert from "node:assert/strict";
import test from "node:test";

import type { GgufMetadata } from "../src/gguf.ts";
import type { ModelManifest } from "../src/model.ts";
import { createModelResourceRequirements } from "../src/runner/model-resources.ts";
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
