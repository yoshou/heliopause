import assert from "node:assert/strict";
import test from "node:test";

import { auditTensorCoverage } from "./audit-helpers.ts";
import {
  buildModelManifest,
  cloneInferenceState,
  createReferenceProvider,
  createModelSession,
  createInferenceState,
  decode,
  disposeInferenceState,
  estimateWeightCacheBytes,
  GgufTensorReader,
  prefillState,
  ReferenceSegmentRunner,
  createWasmProvider,
  WasmSegmentRunner,
  type ModelRunnerProvider,
} from "../src/index.ts";
import { webGpuKvCacheBufferByteLength } from "../src/runner/webgpu/segment-runner.ts";
import { addInferenceStateDisposeCallback, ensureSlidingKvCacheReserve, kvCacheCapacity, slidingWindowReserveTokensForState } from "../src/runtime.ts";

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

test("model manifest accepts layer-wise KV heads and shared key value projection", () => {
  const manifest = buildModelManifest({
    ...minimalGguf(),
    metadata: {
      ...minimalGguf().metadata,
      "gemma4.block_count": 6,
      "gemma4.attention.head_count": 8,
      "gemma4.attention.head_count_kv": {
        type: "int32",
        length: 6,
        sample: [8, 8, 8, 8, 8, 1],
        truncated: false,
      },
      "gemma4.attention.sliding_window_pattern": {
        type: "bool",
        length: 6,
        sample: [true, true, true, true, true, false],
        truncated: false,
      },
    },
    tensors: [{
      name: "blk.5.attn_k.weight",
      dimensions: [4, 2],
      type: "F32",
      typeId: 0,
      offset: 0n,
      dataOffset: 0n,
    }],
  });

  assert.deepEqual(manifest.layerHeadCountKv, [8, 8, 8, 8, 8, 1]);
  assert.equal(manifest.layerValueProjectionModes[5], "shared-with-key");
  assert.ok(!manifest.expectedTensors.some((tensor) => tensor.name === "blk.5.attn_v.weight"));
});

test("model manifest rejects layer-wise KV head metadata with wrong length", () => {
  assert.throws(
    () => buildModelManifest({
      ...minimalGguf(),
      metadata: {
        ...minimalGguf().metadata,
        "gemma4.block_count": 2,
        "gemma4.attention.head_count_kv": {
          type: "int32",
          length: 1,
          sample: [1],
          truncated: false,
        },
      },
    }),
    /head_count_kv length 1 does not match block_count 2/,
  );
});

test("inference state allocates layer-wise KV head cache sizes", () => {
  const manifest = buildModelManifest({
    ...minimalGguf(),
    metadata: {
      ...minimalGguf().metadata,
      "gemma4.block_count": 6,
      "gemma4.attention.head_count": 8,
      "gemma4.attention.head_count_kv": {
        type: "int32",
        length: 6,
        sample: [8, 8, 8, 8, 8, 1],
        truncated: false,
      },
    },
    tensors: [
      { name: "blk.0.attn_k.weight", dimensions: [4, 16], type: "F32", typeId: 0, offset: 0n, dataOffset: 0n },
      { name: "blk.5.attn_k.weight", dimensions: [4, 2], type: "F32", typeId: 0, offset: 0n, dataOffset: 0n },
    ],
  });
  const state = createInferenceState(manifest);

  assert.equal(state.fullAttention.get(0)?.headCountKv, 8);
  assert.equal(state.fullAttention.get(0)?.key.length, 16);
  assert.equal(state.fullAttention.get(5)?.headCountKv, 1);
  assert.equal(state.fullAttention.get(5)?.key.length, 2);
});

test("inference state uses ring capacity for sliding attention caches", () => {
  const manifest = buildModelManifest({
    ...minimalGguf(),
    metadata: {
      ...minimalGguf().metadata,
      "gemma4.block_count": 2,
      "gemma4.context_length": 8,
      "gemma4.attention.sliding_window": 3,
      "gemma4.full_attention_interval": 2,
      "gemma4.attention.sliding_window_pattern": {
        type: "bool",
        length: 2,
        sample: [true, false],
        truncated: false,
      },
    },
  });

  const state = createInferenceState(manifest);
  const sliding = state.fullAttention.get(0);
  const full = state.fullAttention.get(1);

  assert.equal(sliding?.kind, "sliding");
  assert.equal(sliding?.contextLength, 8);
  assert.equal(sliding?.capacity, 3);
  assert.equal(sliding?.key.length, 6);
  assert.equal(sliding?.value.length, 6);
  assert.equal(full?.kind, "full");
  assert.equal(full?.contextLength, 8);
  assert.equal(full?.capacity, 8);
  assert.equal(full?.key.length, 16);
  assert.equal(full?.value.length, 16);
});

test("MTP reserve expands sliding ring capacity and preserves cached slots", () => {
  const manifest = buildModelManifest({
    ...minimalGguf(),
    metadata: {
      ...minimalGguf().metadata,
      "gemma4.block_count": 1,
      "gemma4.context_length": 6,
      "gemma4.attention.sliding_window": 2,
      "gemma4.attention.sliding_window_pattern": {
        type: "bool",
        length: 1,
        sample: [true],
        truncated: false,
      },
    },
  });
  const state = createInferenceState(manifest);
  const cache = state.fullAttention.get(0);
  assert(cache);
  cache.key.set([10, 11], 0);
  cache.key.set([20, 21], 2);
  cache.value[0] = 100;
  cache.value[1] = 200;
  cache.value[2] = 101;
  cache.value[3] = 201;
  state.nextPosition = 2;

  ensureSlidingKvCacheReserve(state, manifest, 2);

  assert.equal(slidingWindowReserveTokensForState(state, manifest), 2);
  assert.equal(cache.capacity, 4);
  assert.equal(cache.key.length, 8);
  assert.equal(cache.value.length, 8);
  assert.deepEqual(Array.from(cache.key.slice(0, 4)), [10, 11, 20, 21]);
  assert.deepEqual(Array.from(cache.value), [100, 200, 0, 0, 101, 201, 0, 0]);
});

test("WebGPU KV cache buffers align odd MTP ring capacity for readback copies", () => {
  const capacity = kvCacheCapacity("sliding", 16, 4, 3);
  const logicalBytes = capacity * 1 * 1 * 2;
  const bufferBytes = webGpuKvCacheBufferByteLength(capacity, 1, 1);

  assert.equal(capacity, 7);
  assert.equal(logicalBytes, 14);
  assert.equal(bufferBytes, 16);
  assert.equal(bufferBytes % 4, 0);
  assert.ok(bufferBytes >= logicalBytes);
});

test("reference sliding ring chunking matches token-by-token decode", async () => {
  const reader = tensorReaderFromTensors(sharedKeyReferenceLayerTensors(), {
    "gemma4.block_count": 1,
    "gemma4.context_length": 4,
    "gemma4.attention.sliding_window": 2,
    "gemma4.attention.sliding_window_pattern": {
      type: "bool",
      length: 1,
      sample: [true],
      truncated: false,
    },
  });
  const session = createModelSession(reader, { providers: [createReferenceProvider()] });
  const runner = new ReferenceSegmentRunner({ session });
  const input = new Float32Array([
    0.1, -0.2, 0.3, -0.4,
    0.2, -0.1, 0.4, -0.3,
    0.3, -0.4, 0.1, -0.2,
  ]);

  const batched = await runner.runTokensHidden(
    input,
    new Int32Array([0, 1, 2]),
    session.createInferenceState(),
  );
  const tokenState = session.createInferenceState();
  let tokenOutput = new Float32Array(0);
  for (let token = 0; token < 3; token += 1) {
    const result = await runner.runTokenHidden(
      input.subarray(token * 4, (token + 1) * 4),
      new Int32Array([token]),
      tokenState,
    );
    tokenOutput = result.hidden;
  }

  assertFloatArrayClose(batched.hidden, tokenOutput, 1e-5);
});

test("reference sliding ring chunking preserves layer-major per-layer inputs", async () => {
  const reader = tensorReaderFromTensors([
    ...sharedKeyReferenceLayerTensorsFor(0),
    ...sharedKeyReferenceLayerTensorsFor(1),
  ], {
    "gemma4.block_count": 2,
    "gemma4.context_length": 4,
    "gemma4.attention.sliding_window": 2,
    "gemma4.embedding_length_per_layer_input": 2,
    "gemma4.attention.sliding_window_pattern": {
      type: "bool",
      length: 2,
      sample: [true, true],
      truncated: false,
    },
  });
  const session = createModelSession(reader, { providers: [createReferenceProvider()] });
  const runner = new ReferenceSegmentRunner({ session });
  const input = slidingRingInput();
  const perLayerInputs = new Float32Array([
    1, 0.5, 2, 0.5, 3, 0.5,
    -1, 0.25, -2, 0.25, -3, 0.25,
  ]);

  const batched = await runner.runTokensHidden(input, new Int32Array([0, 1, 2]), session.createInferenceState(), {
    perLayerInputs,
  });
  const tokenState = session.createInferenceState();
  let tokenOutput = new Float32Array(0);
  for (let token = 0; token < 3; token += 1) {
    const result = await runner.runTokenHidden(
      input.subarray(token * 4, (token + 1) * 4),
      new Int32Array([token]),
      tokenState,
      { perLayerInputs: slicePerTokenPerLayerInputs(perLayerInputs, 2, 3, 2, token) },
    );
    tokenOutput = result.hidden;
  }

  assertFloatArrayClose(batched.hidden, tokenOutput, 1e-5);
});

test("reference sliding attention reads ring cache without compacting", async () => {
  const reader = tensorReaderFromTensors(sharedKeyReferenceLayerTensors(), slidingRingMetadata());
  const session = createModelSession(reader, { providers: [createReferenceProvider()] });
  const runner = new ReferenceSegmentRunner({ session });
  const sections: string[] = [];

  await runner.runTokensHidden(
    slidingRingInput(),
    new Int32Array([0, 1, 2]),
    session.createInferenceState(),
    {
      trace: {
        phase: "prefill",
        onTiming: (event) => sections.push(event.section),
      },
    },
  );

  assert.ok(!sections.includes("key compact"), `Reference sliding attention compacted key cache: ${sections.join(", ")}`);
  assert.ok(!sections.includes("value compact"), `Reference sliding attention compacted value cache: ${sections.join(", ")}`);
});

test("WASM sliding attention reads ring cache without compacting", async () => {
  const reader = tensorReaderFromTensors(sharedKeyReferenceLayerTensors(), slidingRingMetadata());
  const session = createModelSession(reader, { providers: [createWasmProvider()] });
  const runner = new WasmSegmentRunner({ session });
  const sections: string[] = [];

  await runner.runTokensHidden(
    slidingRingInput(),
    new Int32Array([0, 1, 2]),
    session.createInferenceState(),
    {
      trace: {
        phase: "prefill",
        onTiming: (event) => sections.push(event.section),
      },
    },
  );

  assert.ok(!sections.includes("key compact"), `WASM sliding attention compacted key cache: ${sections.join(", ")}`);
  assert.ok(!sections.includes("value compact"), `WASM sliding attention compacted value cache: ${sections.join(", ")}`);
});

test("reference shared-with-key runs without attn_v weight", async () => {
  const reader = tensorReaderFromTensors(sharedKeyReferenceLayerTensors(), {
    "gemma4.block_count": 1,
    "gemma4.attention.sliding_window_pattern": {
      type: "bool",
      length: 1,
      sample: [false],
      truncated: false,
    },
  });
  const session = createModelSession(reader, { providers: [createReferenceProvider()] });
  const runner = new ReferenceSegmentRunner({ session });

  const result = await runner.runTokensHidden(
    new Float32Array([1, 2, 3, 4]),
    new Int32Array([0]),
    session.createInferenceState(),
  );

  assert.equal(result.hidden.length, 4);
});

test("model session can cap inference cache context", () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
  ], {
    "gemma4.block_count": 1,
    "gemma4.context_length": 16,
    "gemma4.full_attention_interval": 1,
  });
  const session = createModelSession(reader, {
    maxContextLength: 4,
    providers: [createReferenceProvider()],
  });
  const state = session.createInferenceState();

  assert.equal(session.manifest.contextLength, 16);
  assert.equal(state.contextLength, 4);
  assert.equal(state.fullAttention.get(0)?.key.length, 8);
  assert.equal(state.fullAttention.get(0)?.value.length, 8);
});

test("model session validates provider list at construction", () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
  ]);

  assert.throws(
    () => createModelSession(reader, { providers: [] }),
    /At least one runner provider is required/,
  );
  assert.throws(
    () => createModelSession(reader, { providers: [createReferenceProvider(), createReferenceProvider()] }),
    /Duplicate runner provider: reference/,
  );
  assert.throws(
    () => createModelSession(reader, {
      providers: [{ name: "reference" } as unknown as ModelRunnerProvider],
    }),
    /Runner provider reference is missing createModelRunner/,
  );
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

test("inference state dispose runs cleanup once and clears cache arrays", () => {
  const manifest = buildModelManifest({
    ...minimalGguf(),
    tensorCount: 0,
    tensors: [],
  });
  const state = createInferenceState(manifest);
  const calls: string[] = [];

  addInferenceStateDisposeCallback(state, () => calls.push("first"));
  addInferenceStateDisposeCallback(state, () => calls.push("second"));

  disposeInferenceState(state);
  disposeInferenceState(state);

  assert.deepEqual(calls, ["second", "first"]);
  assert.equal(state.fullAttention.size, 0);
  assert.throws(() => cloneInferenceState(state), /disposed/);
});

test("prefill advances nextPosition from default and explicit positions", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
  ]);
  const session = createModelSession(reader, { providers: [createReferenceProvider()] });

  const defaultState = session.createInferenceState();
  await prefillState(session, defaultState, [1, 2, 3]);
  assert.equal(defaultState.nextPosition, 3);

  const explicitState = session.createInferenceState();
  await prefillState(session, explicitState, [1, 2], {
    positions: new Int32Array([4, 7]),
  });
  assert.equal(explicitState.nextPosition, 8);

  const mropeState = session.createInferenceState();
  await prefillState(session, mropeState, [1, 2], {
    positions: new Int32Array([5, 6, 50, 60, 70, 80, 90, 100]),
  });
  assert.equal(mropeState.nextPosition, 7);
});

test("decode uses state position, explicit position, and returns top token candidates", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("output.weight", [4, 3], new Float32Array([
      0.2, 0.1, -0.1, 0.3,
      -0.4, 0.2, 0.5, 0.1,
      0.3, -0.2, 0.4, -0.3,
    ])),
  ]);
  const session = createModelSession(reader, { providers: [createReferenceProvider()] });
  const state = session.createInferenceState();
  state.nextPosition = 4;

  const first = await decode(session, state, 2, { logitsTopK: 2 });
  assert.equal(state.nextPosition, 5);
  assert.equal(first.nextTokenId, 1);
  assert.deepEqual(first.topTokens?.map((token) => token.id), [1, 0]);

  const second = await decode(session, state, 3, { position: 9, logitsTopK: 2 });
  assert.equal(state.nextPosition, 10);
  assert.equal(second.nextTokenId, 0);
  assert.deepEqual(second.topTokens?.map((token) => token.id), [0, 2]);
});

test("model session caches F32 tensors and embedding rows", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 2, 3, 4])),
  ]);
  const session = createModelSession(reader, { providers: [createReferenceProvider()] });

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
  const session = createModelSession(reader, {
    maxWeightCacheBytes: 40,
    providers: [createReferenceProvider()],
  });

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
  const session = createModelSession(reader, { providers: [createReferenceProvider()] });
  const state = session.createInferenceState();

  await assert.rejects(
    decode(session, state, 1, { position: 1 }),
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

function sharedKeyReferenceLayerTensors() {
  return [
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    ...sharedKeyReferenceLayerTensorsFor(0),
  ];
}

function sharedKeyReferenceLayerTensorsFor(layer: number) {
  return [
    f32Tensor(`blk.${layer}.attn_norm.weight`, [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor(`blk.${layer}.attn_q.weight`, [4, 2], sequence(8)),
    f32Tensor(`blk.${layer}.attn_k.weight`, [4, 2], sequence(8)),
    f32Tensor(`blk.${layer}.attn_q_norm.weight`, [2], new Float32Array([1, 1])),
    f32Tensor(`blk.${layer}.attn_k_norm.weight`, [2], new Float32Array([1, 1])),
    f32Tensor(`blk.${layer}.attn_output.weight`, [2, 4], sequence(8)),
    f32Tensor(`blk.${layer}.post_attention_norm.weight`, [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor(`blk.${layer}.ffn_norm.weight`, [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor(`blk.${layer}.ffn_gate.weight`, [4, 8], sequence(32)),
    f32Tensor(`blk.${layer}.ffn_up.weight`, [4, 8], sequence(32)),
    f32Tensor(`blk.${layer}.ffn_down.weight`, [8, 4], sequence(32)),
    f32Tensor(`blk.${layer}.post_ffw_norm.weight`, [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor(`blk.${layer}.layer_output_scale.weight`, [1], new Float32Array([1])),
    f32Tensor(`blk.${layer}.inp_gate.weight`, [4, 2], sequence(8)),
    f32Tensor(`blk.${layer}.proj.weight`, [2, 4], sequence(8)),
    f32Tensor(`blk.${layer}.post_norm.weight`, [4], new Float32Array([1, 1, 1, 1])),
  ];
}

function slidingRingMetadata() {
  return {
    "gemma4.block_count": 1,
    "gemma4.context_length": 4,
    "gemma4.attention.sliding_window": 2,
    "gemma4.attention.sliding_window_pattern": {
      type: "bool",
      length: 1,
      sample: [true],
      truncated: false,
    },
  };
}

function slidingRingInput(): Float32Array {
  return new Float32Array([
    0.1, -0.2, 0.3, -0.4,
    0.2, -0.1, 0.4, -0.3,
    0.3, -0.4, 0.1, -0.2,
  ]);
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

function slicePerTokenPerLayerInputs(
  values: Float32Array,
  layerCount: number,
  sourceTokenCount: number,
  perLayerLength: number,
  token: number,
): Float32Array {
  const output = new Float32Array(layerCount * perLayerLength);
  for (let layer = 0; layer < layerCount; layer += 1) {
    const sourceBase = layer * sourceTokenCount * perLayerLength + token * perLayerLength;
    output.set(values.subarray(sourceBase, sourceBase + perLayerLength), layer * perLayerLength);
  }
  return output;
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
