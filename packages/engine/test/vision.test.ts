import assert from "node:assert/strict";
import test from "node:test";

import {
  calculateGemma4VisionResize,
  createGemma4VisionSession,
  runGemma4VisionEncoder,
} from "../src/vision.ts";
import {
  GgufTensorReader,
} from "../src/tensor-reader.ts";
import {
  resetPrefillWasmForTesting,
} from "../src/runner/cpu/wasm-kernels.ts";

test("Gemma4V dynamic resize follows llama.cpp token limits", () => {
  const manifest = {
    patchSize: 16,
    spatialMergeSize: 3,
    imageMinTokens: 252,
    imageMaxTokens: 280,
  };

  assert.deepEqual(calculateGemma4VisionResize(manifest, 512, 512), {
    width: 768,
    height: 768,
    outputTokenCount: 256,
  });

  const wide = calculateGemma4VisionResize(manifest, 389, 244);
  assert.equal(wide.width % 48, 0);
  assert.equal(wide.height % 48, 0);
  assert.ok(wide.outputTokenCount >= 252);
  assert.ok(wide.outputTokenCount <= 280);

  const tall = calculateGemma4VisionResize(manifest, 215, 330);
  assert.equal(tall.width % 48, 0);
  assert.equal(tall.height % 48, 0);
  assert.ok(tall.outputTokenCount >= 252);
  assert.ok(tall.outputTokenCount <= 280);
});

test("Gemma4V session exposes provider config, caches weights, and disposes resources", async () => {
  const reader = visionTensorReader([
    f32Tensor("v.patch_embd.weight", [1, 1, 3, 2], new Float32Array(6)),
  ]);
  const session = createGemma4VisionSession(reader, {
    maxWeightCacheBytes: 128,
    executionProviders: [{
      name: "cpu",
      options: { wasmKernels: false, residentWeightCache: true },
    }],
  });

  assert.equal(session.executionProvider("cpu")?.options?.residentWeightCache, true);
  await session.readWeightBytes("v.patch_embd.weight");
  await session.readWeightBytes("v.patch_embd.weight");

  assert.equal(session.cacheStats().weightTensorCount, 1);
  assert.equal(session.cacheStats().weightCacheHits, 1);
  assert.equal(session.cacheStats().weightCacheMisses, 1);

  session.dispose();

  assert.equal(session.cacheStats().weightTensorCount, 0);
  assert.deepEqual(session.cacheStats().executionProviderStats, {});
});

test("Gemma4V falls back through WebGPU and unavailable WASM paths", async () => {
  const reader = visionTensorReader([
    f32Tensor("v.patch_embd.weight", [1, 1, 3, 2], new Float32Array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3])),
    f32Tensor("v.position_embd.weight", [2, 1, 2], new Float32Array([0.01, 0.02, 0.03, 0.04])),
    f32Tensor("mm.input_projection.weight", [2, 2], new Float32Array([1, 0, 0, 1])),
  ], {
    "clip.vision.projector.scale_factor": 1,
  });
  const session = createGemma4VisionSession(reader, {
    executionProviders: [
      { name: "webgpu", options: { memoryLimitBytes: 1 } },
      { name: "cpu", options: { wasmKernels: true } },
    ],
  });

  resetPrefillWasmForTesting("");
  try {
    const result = await runGemma4VisionEncoder(session, {
      values: new Float32Array([0.25, 0.5, 0.75]),
      width: 1,
      height: 1,
    });

    assert.equal(result.tokenCount, 1);
    assert.equal(result.hidden.length, 2);
    assert.equal(result.width, 1);
    assert.equal(result.height, 1);
    assert.equal(session.cacheStats().executionProviderStats.webgpuVisionAttempts, 1);
    assert.equal(session.cacheStats().executionProviderStats.webgpuVisionFallbacks, 1);
    assert.equal(session.cacheStats().executionProviderStats.webgpuVisionLastFallbackReason, "webgpu-unavailable");
  } finally {
    resetPrefillWasmForTesting();
  }
});

function visionTensorReader(
  tensors: Array<{
    name: string;
    dimensions: number[];
    type: "F32";
    bytes: Uint8Array;
  }>,
  metadataOverrides: Record<string, unknown> = {},
) {
  let offset = 0n;
  const bytes: Uint8Array[] = [];
  const infos = tensors.map((tensor, typeId) => {
    const dataOffset = offset;
    offset += BigInt(tensor.bytes.byteLength);
    bytes.push(tensor.bytes);
    return {
      name: tensor.name,
      dimensions: tensor.dimensions,
      type: tensor.type,
      typeId,
      offset: 0n,
      dataOffset,
    };
  });
  const data = concatBytes(bytes);
  return new GgufTensorReader({
    version: 3,
    tensorCount: infos.length,
    metadataCount: 13,
    dataStart: 0n,
    metadata: {
      "general.architecture": "clip",
      "clip.has_vision_encoder": true,
      "clip.vision.projector_type": "gemma4",
      "clip.vision.image_size": 1,
      "clip.vision.patch_size": 1,
      "clip.vision.embedding_length": 2,
      "clip.vision.feed_forward_length": 4,
      "clip.vision.block_count": 0,
      "clip.vision.attention.head_count": 1,
      "clip.vision.attention.layer_norm_epsilon": 1e-6,
      "clip.vision.projection_dim": 2,
      "clip.vision.image_mean": {
        type: "float32",
        length: 3,
        sample: [0, 0, 0],
        truncated: false,
      },
      "clip.vision.image_std": {
        type: "float32",
        length: 3,
        sample: [1, 1, 1],
        truncated: false,
      },
      ...metadataOverrides,
    },
    tensors: infos,
  }, {
    async read(readOffset, length) {
      return data.subarray(Number(readOffset), Number(readOffset) + length);
    },
  });
}

function f32Tensor(name: string, dimensions: number[], values: Float32Array) {
  return {
    name,
    dimensions,
    type: "F32" as const,
    bytes: new Uint8Array(values.buffer, values.byteOffset, values.byteLength).slice(),
  };
}

function concatBytes(parts: Uint8Array[]): Uint8Array {
  const length = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const output = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.byteLength;
  }
  return output;
}
