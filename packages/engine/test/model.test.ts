import assert from "node:assert/strict";
import test from "node:test";

import {
  auditTensorCoverage,
  buildModelManifest,
  cloneInferenceState,
  createReferenceProvider,
  createModelSession,
  createInferenceState,
  decode,
  estimateWeightCacheBytes,
  GgufTensorReader,
  prefill,
} from "../src/index.ts";

test("tensor coverage audit fails closed on unknown and unused tensors", async () => {
  const gguf = {
    version: 3,
    tensorCount: 2,
    metadataCount: 14,
    dataStart: 0n,
    metadata: {
      "general.architecture": "gemma4",
      "gemma4.block_count": 1,
      "gemma4.embedding_length": 4,
      "gemma4.feed_forward_length": 8,
      "gemma4.attention.head_count": 1,
      "gemma4.attention.head_count_kv": 1,
      "gemma4.attention.key_length": 2,
      "gemma4.attention.value_length": 2,
      "gemma4.context_length": 16,
      "gemma4.full_attention_interval": 4,
      "gemma4.rope.dimension_count": 2,
      "gemma4.rope.dimension_sections": {
        type: "int32",
        length: 4,
        sample: [1, 1, 0, 0],
        truncated: false,
      },
      "gemma4.rope.freq_base": 10000,
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

  const manifest = buildModelManifest(gguf);
  const audit = auditTensorCoverage(gguf, manifest, ["output_norm.weight"]);

  assert.equal(audit.ok, false);
  assert.ok(audit.unknown.includes("unexpected.weight"));
  assert.ok(audit.loadedButUnused.includes("unexpected.weight"));
  assert.ok(audit.missing.length > 0);
});

test("inference state allocates attention caches from manifest", () => {
  const manifest = buildModelManifest({
    ...minimalGguf(),
    tensorCount: 0,
    tensors: [],
  });

  const state = createInferenceState(manifest);

  assert.deepEqual(Array.from(state.fullAttention.keys()), [0]);
  assert.equal(state.nextPosition, 0);
  assert.equal(state.fullAttention.get(0)?.key.length, 2);
  assert.equal(state.fullAttention.get(0)?.value.length, 2);
  assert.equal(state.fullAttention.get(0)?.keyLength, 2);
  assert.equal(state.fullAttention.get(0)?.valueLength, 2);
});

test("model session can cap inference cache context", () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
  ], {
    "gemma4.block_count": 1,
    "gemma4.context_length": 16,
    "gemma4.full_attention_interval": 1,
  });
  const session = createModelSession(reader, { maxContextLength: 4 });
  const state = session.createInferenceState();

  assert.equal(session.manifest.contextLength, 16);
  assert.equal(state.contextLength, 4);
  assert.equal(state.fullAttention.get(0)?.key.length, 8);
  assert.equal(state.fullAttention.get(0)?.value.length, 8);
});

test("inference state defaults capped context to manifest context", () => {
  const manifest = buildModelManifest({
    ...minimalGguf(),
    tensorCount: 0,
    tensors: [],
  });
  const state = createInferenceState(manifest, { contextLength: 1 });

  assert.equal(state.contextLength, 1);
  assert.equal(state.fullAttention.get(0)?.key.length, 2);
});

test("inference state clone deep-copies cache arrays", () => {
  const manifest = buildModelManifest({
    ...minimalGguf(),
    metadata: {
      ...minimalGguf().metadata,
      "gemma4.block_count": 2,
      "gemma4.full_attention_interval": 2,
    },
    tensorCount: 0,
    tensors: [],
  });
  const state = createInferenceState(manifest);
  state.nextPosition = 3;
  const fullAttention = state.fullAttention.get(1);
  assert.ok(fullAttention);
  fullAttention.key[0] = 1;
  fullAttention.value[0] = 2;

  const clone = cloneInferenceState(state);
  const cloneFullAttention = clone.fullAttention.get(1);
  assert.ok(cloneFullAttention);
  clone.nextPosition = 7;
  cloneFullAttention.key[0] = 10;
  cloneFullAttention.value[0] = 20;

  assert.equal(state.nextPosition, 3);
  assert.equal(fullAttention.key[0], 1);
  assert.equal(fullAttention.value[0], 2);
});

test("prefill advances nextPosition from default and explicit positions", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
  ]);
  const session = createModelSession(reader, { runnerProviders: [createReferenceProvider()] });

  const defaultResult = await prefill(session, [1, 2, 3]);
  assert.equal(defaultResult.state.nextPosition, 3);

  const explicitResult = await prefill(session, [1, 2], {
    positions: new Int32Array([4, 7]),
  });
  assert.equal(explicitResult.state.nextPosition, 8);

  const mropeResult = await prefill(session, [1, 2], {
    positions: new Int32Array([5, 6, 50, 60, 70, 80, 90, 100]),
  });
  assert.equal(mropeResult.state.nextPosition, 7);
});

test("decode uses state position, explicit position, and returns fixed logits", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("output.weight", [4, 3], new Float32Array([
      0.2, 0.1, -0.1, 0.3,
      -0.4, 0.2, 0.5, 0.1,
      0.3, -0.2, 0.4, -0.3,
    ])),
  ]);
  const session = createModelSession(reader, { runnerProviders: [createReferenceProvider()] });
  const state = session.createInferenceState();
  state.nextPosition = 4;

  const first = await decode(session, 2, { state, logitsTopK: 2 });
  assert.equal(first.state.nextPosition, 5);
  assertFloatArrayClose(first.logits, new Float32Array([
    -0.16329793632030487,
    0.5715428590774536,
    -0.5715428590774536,
  ]), 2e-5);
  assert.deepEqual(first.topTokens.map((token) => token.id), [1, 0]);
  assertFloatArrayClose(
    Float32Array.from(first.topTokens.map((token) => token.value)),
    new Float32Array([0.5715428590774536, -0.16329793632030487]),
    2e-5,
  );

  const second = await decode(session, 3, { state, position: 9, logitsTopK: 2 });
  assert.equal(second.state.nextPosition, 10);
  assertFloatArrayClose(second.logits, new Float32Array([
    0.1568925976753235,
    -0.7452399134635925,
    -0.23533886671066284,
  ]), 2e-5);
  assert.deepEqual(second.topTokens.map((token) => token.id), [0, 2]);
  assertFloatArrayClose(
    Float32Array.from(second.topTokens.map((token) => token.value)),
    new Float32Array([0.1568925976753235, -0.23533886671066284]),
    2e-5,
  );
});

test("model session caches F32 tensors and embedding rows", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 2, 3, 4])),
  ]);
  const session = createModelSession(reader, { runnerProviders: [createReferenceProvider()] });

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
    embeddingRowCount: 1,
    executionProviderStats: {},
  });
});

test("model session evicts large weight bytes without evicting small F32 tensors", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 2, 3, 4])),
    bytesTensor("a.weight", [32, 1], "Q8_0", new Uint8Array(34).fill(1)),
    bytesTensor("b.weight", [32, 1], "Q8_0", new Uint8Array(34).fill(2)),
  ]);
  const session = createModelSession(reader, { maxWeightCacheBytes: 40 });

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

test("weight cache estimate counts quantized matmul weights only", () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    bytesTensor("blk.0.attn_q.weight", [32, 2], "Q8_0", new Uint8Array(68).fill(1)),
    bytesTensor("blk.0.ffn_gate.weight", [256, 1], "Q4_K", new Uint8Array(144).fill(2)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
  ]);

  assert.equal(estimateWeightCacheBytes(reader), 212);
});

test("full-attention decode rejects positions outside context", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("output.weight", [4, 3], sequence(12)),
    ...fullAttentionLayerTensors(),
  ], {
    "gemma4.block_count": 1,
    "gemma4.context_length": 1,
    "gemma4.attention.sliding_window_pattern": {
      type: "bool",
      length: 1,
      sample: [false],
      truncated: false,
    },
  });
  const session = createModelSession(reader, { runnerProviders: [createReferenceProvider()] });

  await assert.rejects(
    decode(session, 1, { position: 1 }),
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
      "general.architecture": "gemma4",
      "gemma4.block_count": 1,
      "gemma4.embedding_length": 4,
      "gemma4.feed_forward_length": 8,
      "gemma4.attention.head_count": 1,
      "gemma4.attention.head_count_kv": 1,
      "gemma4.attention.key_length": 2,
      "gemma4.attention.value_length": 2,
      "gemma4.context_length": 1,
      "gemma4.full_attention_interval": 1,
      "gemma4.rope.dimension_count": 2,
      "gemma4.rope.dimension_sections": {
        type: "int32",
        length: 4,
        sample: [1, 1, 0, 0],
        truncated: false,
      },
      "gemma4.rope.freq_base": 10000,
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
      "gemma4.block_count": 0,
      "gemma4.context_length": 16,
      "gemma4.full_attention_interval": 1,
      "gemma4.attention.layer_norm_rms_epsilon": 1e-6,
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

function assertFloatArrayClose(actual: Float32Array, expected: Float32Array, tolerance: number): void {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(
      Math.abs((actual[index] ?? 0) - (expected[index] ?? 0)) <= tolerance,
      `index ${index}: expected ${expected[index]}, got ${actual[index]}`,
    );
  }
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
