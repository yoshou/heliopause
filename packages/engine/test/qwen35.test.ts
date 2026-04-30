import assert from "node:assert/strict";
import test from "node:test";

import {
  auditQwen35TensorCoverage,
  buildQwen35Manifest,
  createQwen35InferenceState,
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
  assert.equal(state.fullAttention.get(0)?.key.length, 2);
  assert.equal(state.fullAttention.get(0)?.value.length, 2);
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
