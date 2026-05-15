import assert from "node:assert/strict";
import test from "node:test";

import {
  buildAudioManifest,
  isAudioGguf,
} from "../src/model.ts";
import {
  createAudioSession,
  preprocessAudioPcm,
  runAudioEncoder,
} from "../src/audio.ts";
import {
  GgufTensorReader,
} from "../src/tensor-reader.ts";

test("audio manifest reads projector metadata", () => {
  const reader = audioTensorReader([]);
  const manifest = buildAudioManifest(reader.metadata);

  assert.equal(isAudioGguf(reader.metadata), true);
  assert.equal(manifest.projectorType, "gemma4");
  assert.equal(manifest.blockCount, 0);
  assert.equal(manifest.embeddingLength, 1024);
  assert.equal(manifest.featureSize, 128);
  assert.equal(manifest.audioSeqLength, 750);
});

test("audio preprocessing creates 128-bin log-mel frames", () => {
  const samples = new Float32Array(16_000);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = Math.sin(index * 2 * Math.PI * 440 / 16_000) * 0.1;
  }

  const features = preprocessAudioPcm({
    pcm: samples,
    sampleRate: 16_000,
    durationMs: 1000,
  });

  assert.equal(features.featureSize, 128);
  assert.equal(features.frameCount, 99);
  assert.equal(features.values.length, 99 * 128);
  assert.equal(features.attentionMask.length, 99);
  assert.ok(features.values.some((value) => Number.isFinite(value)));
});

test("audio encoder projects hidden to model embedding size", async () => {
  const reader = audioTensorReader([
    f32Tensor("a.conv1d.0.weight", [3, 3, 1, 128], new Float32Array(3 * 3 * 128)),
    f32Tensor("a.conv1d.0.norm.weight", [128], ones(128)),
    f32Tensor("a.conv1d.1.weight", [3, 3, 128, 32], new Float32Array(3 * 3 * 128 * 32)),
    f32Tensor("a.conv1d.1.norm.weight", [32], ones(32)),
    f32Tensor("a.input_projection.weight", [1024, 1024], new Float32Array(1024 * 1024)),
    f32Tensor("a.pre_encode.out.weight", [1024, 1536], new Float32Array(1024 * 1536)),
    f32Tensor("a.pre_encode.out.bias", [1536], new Float32Array(1536)),
    f32Tensor("mm.a.input_projection.weight", [1536, 2560], new Float32Array(1536 * 2560)),
  ]);
  const session = createAudioSession(reader);
  const features = preprocessAudioPcm({
    pcm: new Float32Array(16_000),
    sampleRate: 16_000,
    durationMs: 1000,
  });

  const encoded = await runAudioEncoder(session, features);

  assert.equal(encoded.tokenCount, 25);
  assert.equal(encoded.hidden.length, 25 * 2560);
});

test("audio encoder falls back when WebGPU is unavailable", async () => {
  const reader = audioTensorReader([
    f32Tensor("a.conv1d.0.weight", [3, 3, 1, 128], new Float32Array(3 * 3 * 128)),
    f32Tensor("a.conv1d.0.norm.weight", [128], ones(128)),
    f32Tensor("a.conv1d.1.weight", [3, 3, 128, 32], new Float32Array(3 * 3 * 128 * 32)),
    f32Tensor("a.conv1d.1.norm.weight", [32], ones(32)),
    f32Tensor("a.input_projection.weight", [1024, 1024], new Float32Array(1024 * 1024)),
    f32Tensor("a.pre_encode.out.weight", [1024, 1536], new Float32Array(1024 * 1536)),
    f32Tensor("mm.a.input_projection.weight", [1536, 2560], new Float32Array(1536 * 2560)),
  ]);
  const session = createAudioSession(reader, {
    executionProviders: [
      { name: "webgpu" },
      { name: "cpu", options: { wasmKernels: false } },
    ],
  });
  const features = preprocessAudioPcm({
    pcm: new Float32Array(16_000),
    sampleRate: 16_000,
    durationMs: 1000,
  });

  const encoded = await runAudioEncoder(session, features);

  assert.equal(encoded.tokenCount, 25);
  assert.equal(session.cacheStats().executionProviderStats.webgpuAudioAttempts, 1);
  assert.equal(session.cacheStats().executionProviderStats.webgpuAudioFallbacks, 1);
  assert.equal(session.cacheStats().executionProviderStats.webgpuAudioLastFallbackReason, "webgpu-unavailable");
});

function audioTensorReader(
  tensors: Array<{
    name: string;
    dimensions: number[];
    type: "F32";
    bytes: Uint8Array;
  }>,
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
    metadataCount: 9,
    dataStart: 0n,
    metadata: {
      "general.architecture": "clip",
      "clip.has_audio_encoder": true,
      "clip.audio.projector_type": "gemma4a",
      "clip.audio.embedding_length": 1024,
      "clip.audio.feed_forward_length": 4096,
      "clip.audio.block_count": 0,
      "clip.audio.attention.head_count": 8,
      "clip.audio.attention.layer_norm_epsilon": 1e-5,
      "clip.audio.num_mel_bins": 128,
      "clip.audio.projection_dim": 2560,
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

function ones(length: number): Float32Array {
  const values = new Float32Array(length);
  values.fill(1);
  return values;
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
