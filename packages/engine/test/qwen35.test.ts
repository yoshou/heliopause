import assert from "node:assert/strict";
import test from "node:test";

import {
  auditQwen35TensorCoverage,
  buildQwen35Manifest,
  createQwen35ModelSession,
  createQwen35InferenceState,
  decodeQwen35,
  estimateQwen35WeightCacheBytes,
  GgufTensorReader,
  prefillQwen35,
} from "../src/index.ts";

test("tensor coverage audit fails closed on unknown and unused tensors", async () => {
  const gguf = {
    version: 3,
    tensorCount: 2,
    metadataCount: 14,
    dataStart: 0n,
    metadata: {
      "general.architecture": "qwen35",
      "qwen35.block_count": 1,
      "qwen35.embedding_length": 4,
      "qwen35.feed_forward_length": 8,
      "qwen35.attention.head_count": 1,
      "qwen35.attention.head_count_kv": 1,
      "qwen35.attention.key_length": 2,
      "qwen35.attention.value_length": 2,
      "qwen35.context_length": 16,
      "qwen35.full_attention_interval": 4,
      "qwen35.rope.dimension_count": 2,
      "qwen35.rope.dimension_sections": {
        type: "int32",
        length: 4,
        sample: [1, 1, 0, 0],
        truncated: false,
      },
      "qwen35.rope.freq_base": 10000,
      "qwen35.ssm.conv_kernel": 4,
      "qwen35.ssm.group_count": 1,
      "qwen35.ssm.inner_size": 2,
      "qwen35.ssm.state_size": 2,
      "qwen35.ssm.time_step_rank": 1,
    },
    tensors: [
      {
        name: "output_norm.weight",
        dimensions: [4],
        type: "F32",
        typeId: 0,
        offset: 0n,
        dataOffset: 0n,
      },
      {
        name: "unexpected.weight",
        dimensions: [4],
        type: "F32",
        typeId: 0,
        offset: 0n,
        dataOffset: 0n,
      },
    ],
  };

  const manifest = buildQwen35Manifest(gguf);
  const audit = auditQwen35TensorCoverage(gguf, manifest, ["output_norm.weight"]);

  assert.equal(audit.ok, false);
  assert.ok(audit.unknown.includes("unexpected.weight"));
  assert.ok(audit.loadedButUnused.includes("unexpected.weight"));
  assert.ok(audit.missing.length > 0);
});

test("Qwen35 inference state allocates recurrent and full-attention caches from manifest", () => {
  const manifest = buildQwen35Manifest({
    ...minimalGguf(),
    tensorCount: 0,
    tensors: [],
  });

  const state = createQwen35InferenceState(manifest);

  assert.deepEqual(Array.from(state.recurrent.keys()), []);
  assert.deepEqual(Array.from(state.fullAttention.keys()), [0]);
  assert.equal(state.nextPosition, 0);
  assert.equal(state.fullAttention.get(0)?.key.length, 2);
  assert.equal(state.fullAttention.get(0)?.value.length, 2);
});

test("Qwen35 model session can cap inference cache context", () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
  ], {
    "qwen35.block_count": 1,
    "qwen35.context_length": 16,
    "qwen35.full_attention_interval": 1,
  });
  const session = createQwen35ModelSession(reader, { maxContextLength: 4 });
  const state = session.createInferenceState();

  assert.equal(session.manifest.contextLength, 16);
  assert.equal(state.contextLength, 4);
  assert.equal(state.fullAttention.get(0)?.key.length, 8);
  assert.equal(state.fullAttention.get(0)?.value.length, 8);
});

test("Qwen35 inference state defaults capped context to manifest context", () => {
  const manifest = buildQwen35Manifest({
    ...minimalGguf(),
    tensorCount: 0,
    tensors: [],
  });
  const state = createQwen35InferenceState(manifest, { contextLength: 1 });

  assert.equal(state.contextLength, 1);
  assert.equal(state.fullAttention.get(0)?.key.length, 2);
});

test("Qwen35 prefill advances nextPosition from default and explicit positions", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
  ]);
  const session = createQwen35ModelSession(reader);

  const defaultResult = await prefillQwen35(session, [1, 2, 3]);
  assert.equal(defaultResult.state.nextPosition, 3);

  const explicitResult = await prefillQwen35(session, [1, 2], {
    positions: new Int32Array([4, 7]),
  });
  assert.equal(explicitResult.state.nextPosition, 8);

  const mropeResult = await prefillQwen35(session, [1, 2], {
    positions: new Int32Array([5, 6, 50, 60, 70, 80, 90, 100]),
  });
  assert.equal(mropeResult.state.nextPosition, 7);
});

test("Qwen35 decode uses state position, explicit position, and returns logits", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("output.weight", [4, 3], new Float32Array([
      0.2, 0.1, -0.1, 0.3,
      -0.4, 0.2, 0.5, 0.1,
      0.3, -0.2, 0.4, -0.3,
    ])),
  ]);
  const session = createQwen35ModelSession(reader);
  const state = session.createInferenceState();
  state.nextPosition = 4;

  const first = await decodeQwen35(session, 2, { state, logitsTopK: 2 });
  assert.equal(first.state.nextPosition, 5);
  assert.equal(first.logits.length, 3);
  assert.equal(first.topTokens.length, 2);

  const second = await decodeQwen35(session, 3, { state, position: 9 });
  assert.equal(second.state.nextPosition, 10);
});

test("Qwen35 model session caches F32 tensors and embedding rows", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 2, 3, 4])),
  ]);
  const session = createQwen35ModelSession(reader);

  await session.readF32Tensor("output_norm.weight");
  await session.readF32Tensor("output_norm.weight");
  await session.readEmbeddingRows([2]);
  await session.readEmbeddingRows([2]);

  assert.equal(reader.readCounts.get("output_norm.weight"), 1);
  assert.equal(reader.rangeReadCounts.get("token_embd.weight"), 1);
  assert.deepEqual(session.cacheStats(), {
    f32TensorCount: 1,
    weightTensorCount: 0,
    weightCacheBytes: 0,
    maxWeightCacheBytes: 256 * 1024 * 1024,
    weightCacheHits: 0,
    weightCacheMisses: 0,
    weightCacheEvictions: 0,
    wasmWeightCacheEnabled: false,
    wasmWeightCacheCount: 0,
    wasmWeightCacheBytes: 0,
    wasmWeightCacheHits: 0,
    wasmWeightCacheMisses: 0,
    embeddingRowCount: 1,
  });
});

test("Qwen35 model session evicts large weight bytes without evicting small F32 tensors", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 2, 3, 4])),
    bytesTensor("a.weight", [32, 1], "Q8_0", new Uint8Array(34).fill(1)),
    bytesTensor("b.weight", [32, 1], "Q8_0", new Uint8Array(34).fill(2)),
  ]);
  const session = createQwen35ModelSession(reader, { maxWeightCacheBytes: 40 });

  await session.readF32Tensor("output_norm.weight");
  await session.readWeightBytes("a.weight");
  await session.readWeightBytes("b.weight");
  await session.readWeightBytes("a.weight");
  await session.readF32Tensor("output_norm.weight");

  assert.equal(reader.readCounts.get("output_norm.weight"), 1);
  assert.equal(reader.readCounts.get("a.weight"), 2);
  assert.equal(reader.readCounts.get("b.weight"), 1);
  assert.equal(session.cacheStats().f32TensorCount, 1);
  assert.equal(session.cacheStats().weightTensorCount, 1);
  assert.equal(session.cacheStats().weightCacheHits, 0);
  assert.equal(session.cacheStats().weightCacheMisses, 3);
  assert.equal(session.cacheStats().weightCacheEvictions, 2);
});

test("Qwen35 weight cache estimate counts quantized matmul weights only", () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    bytesTensor("blk.0.attn_q.weight", [32, 2], "Q8_0", new Uint8Array(68).fill(1)),
    bytesTensor("blk.0.ffn_gate.weight", [256, 1], "Q4_K", new Uint8Array(144).fill(2)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
  ]);

  assert.equal(estimateQwen35WeightCacheBytes(reader), 212);
});

test("Qwen35 full-attention decode rejects positions outside context", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("output.weight", [4, 3], sequence(12)),
    ...fullAttentionLayerTensors(),
  ], {
    "qwen35.block_count": 1,
    "qwen35.context_length": 1,
    "qwen35.full_attention_interval": 1,
  });
  const session = createQwen35ModelSession(reader);

  await assert.rejects(
    decodeQwen35(session, 1, { position: 1 }),
    /outside context length/,
  );
});

function minimalGguf() {
  return {
    version: 3,
    tensorCount: 0,
    metadataCount: 14,
    dataStart: 0n,
    metadata: {
      "general.architecture": "qwen35",
      "qwen35.block_count": 1,
      "qwen35.embedding_length": 4,
      "qwen35.feed_forward_length": 8,
      "qwen35.attention.head_count": 1,
      "qwen35.attention.head_count_kv": 1,
      "qwen35.attention.key_length": 2,
      "qwen35.attention.value_length": 2,
      "qwen35.context_length": 1,
      "qwen35.full_attention_interval": 1,
      "qwen35.rope.dimension_count": 2,
      "qwen35.rope.dimension_sections": {
        type: "int32",
        length: 4,
        sample: [1, 1, 0, 0],
        truncated: false,
      },
      "qwen35.rope.freq_base": 10000,
      "qwen35.ssm.conv_kernel": 4,
      "qwen35.ssm.group_count": 1,
      "qwen35.ssm.inner_size": 2,
      "qwen35.ssm.state_size": 2,
      "qwen35.ssm.time_step_rank": 1,
    },
    tensors: [],
  };
}

function fullAttentionLayerTensors() {
  return [
    f32Tensor("blk.0.attn_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("blk.0.attn_q.weight", [4, 4], sequence(16)),
    f32Tensor("blk.0.attn_k.weight", [4, 2], sequence(8)),
    f32Tensor("blk.0.attn_v.weight", [4, 2], sequence(8)),
    f32Tensor("blk.0.attn_q_norm.weight", [2], new Float32Array([1, 1])),
    f32Tensor("blk.0.attn_k_norm.weight", [2], new Float32Array([1, 1])),
    f32Tensor("blk.0.attn_output.weight", [2, 4], sequence(8)),
    f32Tensor("blk.0.post_attention_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("blk.0.ffn_gate.weight", [4, 8], sequence(32)),
    f32Tensor("blk.0.ffn_up.weight", [4, 8], sequence(32)),
    f32Tensor("blk.0.ffn_down.weight", [8, 4], sequence(32)),
  ];
}

function tensorReaderFromTensors(
  tensors: Array<{
    name: string;
    dimensions: number[];
    type: "F32" | "Q8_0" | "Q4_K";
    bytes: Uint8Array;
  }>,
  metadataOverrides: Record<string, unknown> = {},
) {
  const readCounts = new Map<string, number>();
  const rangeReadCounts = new Map<string, number>();
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
  const gguf = {
    ...minimalGguf(),
    tensorCount: infos.length,
    tensors: infos,
    metadata: {
      ...minimalGguf().metadata,
      "qwen35.block_count": 0,
      "qwen35.context_length": 16,
      "qwen35.full_attention_interval": 1,
      "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
      ...metadataOverrides,
    },
  };
  const reader = new GgufTensorReader(gguf, {
    async read(readOffset, length) {
      const tensor = infos.find((item) =>
        readOffset >= item.dataOffset &&
        readOffset + BigInt(length) <= item.dataOffset + BigInt(tensors[infos.indexOf(item)]?.bytes.byteLength ?? 0),
      );
      if (tensor) {
        if (readOffset === tensor.dataOffset && length === (tensors[infos.indexOf(tensor)]?.bytes.byteLength ?? 0)) {
          readCounts.set(tensor.name, (readCounts.get(tensor.name) ?? 0) + 1);
        } else {
          rangeReadCounts.set(tensor.name, (rangeReadCounts.get(tensor.name) ?? 0) + 1);
        }
      }
      return data.subarray(Number(readOffset), Number(readOffset) + length);
    },
  });
  return Object.assign(reader, { readCounts, rangeReadCounts });
}

function f32Tensor(name: string, dimensions: number[], values: Float32Array) {
  return {
    name,
    dimensions,
    type: "F32" as const,
    bytes: new Uint8Array(values.buffer, values.byteOffset, values.byteLength).slice(),
  };
}

function bytesTensor(name: string, dimensions: number[], type: "Q8_0" | "Q4_K", bytes: Uint8Array) {
  return { name, dimensions, type, bytes };
}

function sequence(length: number): Float32Array {
  const values = new Float32Array(length);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = (index % 7 - 3) * 0.1;
  }
  return values;
}

function concatBytes(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const output = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}
