import assert from "node:assert/strict";
import test from "node:test";

import {
  calculateVisionResize,
  createVisionSession,
  preprocessVisionRgbaCpu,
  runVisionEncoder,
} from "../src/vision.ts";
import type {
  VisionManifest,
} from "../src/model.ts";
import {
  GgufTensorReader,
} from "../src/tensor-reader.ts";
import {
  createReferenceProvider,
  createWasmProvider,
  createWebGpuProvider,
} from "../src/index.ts";
import {
  resetPrefillWasmForTesting,
  visionPreprocessRgbaWasm,
} from "../src/runner/wasm/wasm-kernels.ts";
import {
  runVisionPreprocessProviders,
} from "../src/runner/wasm/vision-preprocess-runner.ts";
import {
  runWebGpuVisionPreprocessor,
} from "../src/runner/webgpu/vision-preprocess-runner.ts";

test("Vision dynamic resize follows llama.cpp token limits", () => {
  const manifest = {
    patchSize: 16,
    spatialMergeSize: 3,
    imageMinTokens: 252,
    imageMaxTokens: 280,
  };

  assert.deepEqual(calculateVisionResize(manifest, 512, 512), {
    width: 768,
    height: 768,
    outputTokenCount: 256,
  });

  const wide = calculateVisionResize(manifest, 389, 244);
  assert.equal(wide.width % 48, 0);
  assert.equal(wide.height % 48, 0);
  assert.ok(wide.outputTokenCount >= 252);
  assert.ok(wide.outputTokenCount <= 280);

  const tall = calculateVisionResize(manifest, 215, 330);
  assert.equal(tall.width % 48, 0);
  assert.equal(tall.height % 48, 0);
  assert.ok(tall.outputTokenCount >= 252);
  assert.ok(tall.outputTokenCount <= 280);
});

test("Vision session exposes provider config, caches weights, and disposes resources", async () => {
  const reader = visionTensorReader([
    f32Tensor("v.patch_embd.weight", [1, 1, 3, 2], new Float32Array(6)),
  ]);
  const session = createVisionSession(reader, {
    maxWeightCacheBytes: 128,
    providers: [createWasmProvider({ residentWeightCache: true })],
  });

  assert.equal(session.provider<ReturnType<typeof createWasmProvider>>("wasm")?.options.residentWeightCache, true);
  await session.readWeightBytes("v.patch_embd.weight");
  await session.readWeightBytes("v.patch_embd.weight");

  assert.equal(session.cacheStats().weightTensorCount, 1);
  assert.equal(session.cacheStats().weightCacheHits, 1);
  assert.equal(session.cacheStats().weightCacheMisses, 1);

  session.dispose();

  assert.equal(session.cacheStats().weightTensorCount, 0);
  assert.deepEqual(session.cacheStats().executionProviderStats, {});
});

test("Vision encoder errors instead of falling back when WebGPU is unavailable", async () => {
  const reader = visionTensorReader([
    f32Tensor("v.patch_embd.weight", [1, 1, 3, 2], new Float32Array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3])),
    f32Tensor("v.position_embd.weight", [2, 1, 2], new Float32Array([0.01, 0.02, 0.03, 0.04])),
    f32Tensor("mm.input_projection.weight", [2, 2], new Float32Array([1, 0, 0, 1])),
  ], {
    "clip.vision.projector.scale_factor": 1,
  });
  const session = createVisionSession(reader, {
    providers: [createWebGpuProvider({ memoryLimitBytes: 1 }), createReferenceProvider()],
  });

  resetPrefillWasmForTesting("");
  try {
    await assert.rejects(() => runVisionEncoder(session, {
      values: new Float32Array([0.25, 0.5, 0.75]),
      width: 1,
      height: 1,
    }), /WebGPU is not available for vision encoder execution/);

    assert.equal(session.cacheStats().executionProviderStats.webgpuVisionAttempts, 1);
  } finally {
    resetPrefillWasmForTesting();
  }
});

test("Vision reference encoder projects hidden to model embedding size", async () => {
  const reader = visionTensorReader([
    f32Tensor("v.patch_embd.weight", [1, 1, 3, 2], new Float32Array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3])),
    f32Tensor("v.position_embd.weight", [2, 1, 2], new Float32Array([0.01, 0.02, 0.03, 0.04])),
    f32Tensor("mm.input_projection.weight", [2, 2], new Float32Array([1, 0, 0, 1])),
  ], {
    "clip.vision.projector.scale_factor": 1,
  });
  const session = createVisionSession(reader, { providers: [createReferenceProvider()] });

  const encoded = await runVisionEncoder(session, {
    values: new Float32Array([0.25, 0.5, 0.75]),
    width: 1,
    height: 1,
  });

  assert.equal(encoded.tokenCount, 1);
  assert.equal(encoded.hidden.length, 2);
});

test("Vision WASM preprocessing matches CPU resize and normalization", async () => {
  const manifest: VisionManifest = {
    architecture: "clip",
    projectorType: "gemma4",
    tensorCount: 0,
    imageSize: 2,
    patchSize: 1,
    embeddingLength: 2,
    feedForwardLength: 4,
    blockCount: 0,
    headCount: 1,
    layerNormEpsilon: 1e-6,
    projectionDim: 2,
    spatialMergeSize: 1,
    imageMinTokens: 4,
    imageMaxTokens: 4,
    imageMean: [0.5, 0.25, 0.75],
    imageStd: [0.5, 0.25, 0.75],
    tensorTypes: {},
  };
  const rgba = new Uint8ClampedArray([
    0, 64, 128, 255,
    255, 192, 32, 128,
    128, 16, 240, 255,
    32, 224, 96, 0,
  ]);

  const cpu = preprocessVisionRgbaCpu(rgba, 2, 2, manifest);
  const wasm = await visionPreprocessRgbaWasm(
    rgba,
    2,
    2,
    cpu.width,
    cpu.height,
    manifest.imageMean,
    manifest.imageStd,
  );

  assert.ok(wasm);
  assert.equal(wasm.length, cpu.values.length);
  assert.ok(maxAbsDiff(cpu.values, wasm) <= 1e-6);
});

test("Vision preprocessor errors instead of falling back when WASM is unavailable", async () => {
  const reader = visionTensorReader([
    f32Tensor("v.patch_embd.weight", [1, 1, 3, 2], new Float32Array(6)),
  ], {
    "clip.vision.projector.scale_factor": 1,
  });
  const session = createVisionSession(reader, {
    preprocessProviderOrder: [
      "wasm",
      "reference",
    ],
    providers: [createWasmProvider(), createReferenceProvider()],
  });

  resetPrefillWasmForTesting("");
  try {
    await assert.rejects(() => runVisionPreprocessProviders(
      session,
      {
        rgba: new Uint8ClampedArray([32, 64, 96, 255]),
        sourceWidth: 1,
        sourceHeight: 1,
        resize: { width: 1, height: 1, outputTokenCount: 1 },
      },
      preprocessVisionRgbaCpu,
    ), /WASM vision preprocessing is unavailable/);

    const stats = session.cacheStats().executionProviderStats;
    assert.equal(stats.wasmVisionPreprocessAttempts, 1);
    assert.equal(stats.referenceVisionPreprocessRuns, undefined);
  } finally {
    resetPrefillWasmForTesting();
  }
});

test("Vision WebGPU preprocessor errors when unavailable", async () => {
  const reader = visionTensorReader([
    f32Tensor("v.patch_embd.weight", [1, 1, 3, 2], new Float32Array(6)),
  ], {
    "clip.vision.projector.scale_factor": 1,
  });
  const session = createVisionSession(reader, {
    providers: [createWebGpuProvider(), createReferenceProvider()],
  });

  await assert.rejects(() => runWebGpuVisionPreprocessor(session, {
    rgba: new Uint8ClampedArray([32, 64, 96, 255]),
    sourceWidth: 1,
    sourceHeight: 1,
    resize: { width: 1, height: 1, outputTokenCount: 1 },
  }), /WebGPU is not available for vision preprocessing/);

  const stats = session.cacheStats().executionProviderStats;
  assert.equal(stats.webgpuVisionPreprocessAttempts, 1);
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

function maxAbsDiff(left: Float32Array, right: Float32Array): number {
  assert.equal(left.length, right.length);
  let max = 0;
  for (let index = 0; index < left.length; index += 1) {
    max = Math.max(max, Math.abs((left[index] ?? 0) - (right[index] ?? 0)));
  }
  return max;
}
