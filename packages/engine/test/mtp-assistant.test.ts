import assert from "node:assert/strict";
import test from "node:test";

import { auditMtpAssistantTensorCoverage } from "./audit-helpers.ts";
import {
  buildMtpAssistantManifest,
  createMtpAssistantSession,
  createReferenceProvider,
  createWasmProvider,
  createWebGpuProvider,
  GgufTensorReader,
  runMtpAssistant,
  type GgufMetadata,
  type MtpAssistantRunnerProvider,
} from "../src/index.ts";

test("MTP assistant tensor coverage reports missing and unknown tensors", () => {
  const reader = assistantTensorReader();
  const manifest = buildMtpAssistantManifest(reader.metadata);
  const missingGguf: GgufMetadata = {
    ...reader.metadata,
    tensors: reader.metadata.tensors.filter((tensor) => tensor.name !== "mtp.post_projection.weight"),
  };
  const unknownGguf: GgufMetadata = {
    ...reader.metadata,
    tensors: [...reader.metadata.tensors, tensorInfo("unexpected.weight", [2], "F32", 0n)],
  };
  assert.ok(auditMtpAssistantTensorCoverage(missingGguf, manifest).missing.includes("mtp.post_projection.weight"));
  assert.ok(auditMtpAssistantTensorCoverage(unknownGguf, manifest).unknown.includes("unexpected.weight"));
});

test("MTP assistant dispatch calls the selected provider runner", async () => {
  const reader = assistantTensorReader();
  let called = 0;
  const provider = {
    name: "reference",
    createMtpAssistantRunners() {
      return {
        runner: {
          provider: "reference",
          async run() {
            called += 1;
            return {
              backboneHidden: new Float32Array([3, 4]),
              topTokens: [{ id: 2, value: 9 }],
            };
          },
        },
      };
    },
  } as MtpAssistantRunnerProvider;
  const session = createMtpAssistantSession(reader, { providers: [provider] });

  const result = await runMtpAssistant(session, assistantInput());

  assert.equal(called, 1);
  assert.deepEqual(Array.from(result.backboneHidden), [3, 4]);
  assert.deepEqual(result.topTokens, [{ id: 2, value: 9 }]);
});

test("MTP assistant session validates provider capability and synthetic deterministic smoke", async () => {
  const reader = assistantTensorReader();
  assert.throws(
    () => createMtpAssistantSession(reader, { providers: [] }),
    /At least one runner provider is required/,
  );
  assert.throws(
    () => createMtpAssistantSession(reader, { providers: [{ name: "reference" } as unknown as MtpAssistantRunnerProvider] }),
    /Runner provider reference is missing createMtpAssistantRunners/,
  );
  const session = createMtpAssistantSession(reader, { providers: [createReferenceProvider()] });
  const tokenOrdering = await session.readI32Tensor("mtp.token_ordering.weight");
  const first = await runMtpAssistant(session, assistantInput());
  const second = await runMtpAssistant(session, assistantInput());
  assert.deepEqual(Array.from(tokenOrdering), [0, 1, 2, 3]);
  assert.equal(first.backboneHidden.length, 2);
  assert.ok(first.topTokens.length > 0);
  assert.deepEqual(Array.from(second.backboneHidden), Array.from(first.backboneHidden));
  assert.deepEqual(second.topTokens, first.topTokens);
});

test("WASM MTP assistant provider runs synthetic deterministic smoke", async () => {
  const session = createMtpAssistantSession(assistantTensorReader(), { providers: [createWasmProvider()] });
  const first = await runMtpAssistant(session, assistantInput());
  const second = await runMtpAssistant(session, assistantInput());

  assert.equal(first.backboneHidden.length, 2);
  assert.ok(first.topTokens.length > 0);
  assert.deepEqual(Array.from(second.backboneHidden), Array.from(first.backboneHidden));
  assert.deepEqual(second.topTokens, first.topTokens);
  assert.equal(session.cacheStats().executionProviderStats.wasmMtpAssistantRuns, 2);
});

test("WebGPU MTP assistant provider exposes provider capability", () => {
  const provider = createWebGpuProvider();
  assert.equal(typeof provider.createMtpAssistantRunners, "function");
  assert.equal(provider.createMtpAssistantRunners().runner.provider, "webgpu");
});

function assistantInput() {
  return {
    targetInputEmbedding: new Float32Array([0.1, -0.2]),
    targetCurrentHidden: new Float32Array([0.75, 0.125]),
    targetKv: {
      layers: [{
        key: new Float32Array([0, 0]),
        value: new Float32Array([0, 0]),
        keyLength: 2,
        valueLength: 2,
        headCountKv: 1,
        contextLength: 1,
        tokenCount: 1,
        logicalStart: 0,
      }],
    },
    position: 0,
    topK: 3,
  };
}

function assistantTensorReader() {
  return tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [2, 4], new Float32Array([0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
    f32Tensor("output_norm.weight", [2], new Float32Array([1, 1])),
    f32Tensor("mtp.pre_projection.weight", [4, 2], new Float32Array(8)),
    f32Tensor("mtp.post_projection.weight", [2, 2], new Float32Array([1, 0, 0, 1])),
    f32Tensor("mtp.centroids.weight", [2, 2], new Float32Array([1, 0, 0, 1])),
    i32Tensor("mtp.token_ordering.weight", [4], new Int32Array([0, 1, 2, 3])),
    f32Tensor("blk.0.attn_q.weight", [2, 2], new Float32Array(4)),
    f32Tensor("blk.0.attn_q_norm.weight", [2], new Float32Array([1, 1])),
    f32Tensor("blk.0.attn_output.weight", [2, 2], new Float32Array(4)),
    f32Tensor("blk.0.attn_norm.weight", [2], new Float32Array([1, 1])),
    f32Tensor("blk.0.post_attention_norm.weight", [2], new Float32Array([1, 1])),
    f32Tensor("blk.0.ffn_norm.weight", [2], new Float32Array([1, 1])),
    f32Tensor("blk.0.ffn_gate.weight", [2, 2], new Float32Array(4)),
    f32Tensor("blk.0.ffn_up.weight", [2, 2], new Float32Array(4)),
    f32Tensor("blk.0.ffn_down.weight", [2, 2], new Float32Array(4)),
    f32Tensor("blk.0.post_ffw_norm.weight", [2], new Float32Array([1, 1])),
    f32Tensor("blk.0.layer_output_scale.weight", [1], new Float32Array([1])),
  ]);
}

function tensorReaderFromTensors(tensors: Array<{ name: string; dimensions: number[]; type: "F32" | "I32"; bytes: Uint8Array }>) {
  let offset = 0n;
  const chunks: Uint8Array[] = [];
  const infos = tensors.map((tensor) => {
    const dataOffset = offset;
    offset += BigInt(tensor.bytes.byteLength);
    chunks.push(tensor.bytes);
    return tensorInfo(tensor.name, tensor.dimensions, tensor.type, dataOffset);
  });
  const data = concatBytes(chunks);
  const gguf: GgufMetadata = {
    version: 3,
    tensorCount: infos.length,
    metadataCount: 20,
    dataStart: 0n,
    metadata: {
      "general.architecture": "gemma4_assistant",
      "gemma4_assistant.block_count": 1,
      "gemma4_assistant.embedding_length": 2,
      "gemma4_assistant.n_embd_backbone": 2,
      "gemma4_assistant.feed_forward_length": 2,
      "gemma4_assistant.attention.head_count": 1,
      "gemma4_assistant.attention.head_count_kv": 1,
      "gemma4_assistant.attention.key_length": 2,
      "gemma4_assistant.attention.value_length": 2,
      "gemma4_assistant.attention.key_length_swa": 2,
      "gemma4_assistant.attention.value_length_swa": 2,
      "gemma4_assistant.context_length": 4,
      "gemma4_assistant.attention.sliding_window": 4,
      "gemma4_assistant.attention.sliding_window_pattern": { type: "bool", length: 1, sample: [true], truncated: false },
      "gemma4_assistant.n_centroids": 2,
      "gemma4_assistant.centroid_top_k": 1,
      "gemma4_assistant.use_ordered_embeddings": true,
      "gemma4_assistant.attention.layer_norm_rms_epsilon": 1e-6,
      "gemma4_assistant.rope.dimension_count_swa": 2,
      "gemma4_assistant.rope.freq_base_swa": 10000,
    },
    tensors: infos,
  };
  return new GgufTensorReader(gguf, {
    async read(readOffset, length) {
      return data.subarray(Number(readOffset), Number(readOffset) + length);
    },
  });
}

function tensorInfo(name: string, dimensions: number[], type: "F32" | "I32", dataOffset: bigint) {
  return { name, dimensions, type, typeId: type === "I32" ? 26 : 0, offset: 0n, dataOffset };
}

function f32Tensor(name: string, dimensions: number[], values: Float32Array) {
  return { name, dimensions, type: "F32" as const, bytes: new Uint8Array(values.buffer, values.byteOffset, values.byteLength).slice() };
}

function i32Tensor(name: string, dimensions: number[], values: Int32Array) {
  return { name, dimensions, type: "I32" as const, bytes: new Uint8Array(values.buffer, values.byteOffset, values.byteLength).slice() };
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
