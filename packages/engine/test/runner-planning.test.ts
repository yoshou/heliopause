import assert from "node:assert/strict";
import test from "node:test";

import {
  auditGemma4RunnerPlacementCopies,
  buildGemma4Manifest,
  planGemma4RunnerPlacement,
} from "../src/index.ts";

test("runner placement planning handles off, blocked, and planned WebGPU placement", () => {
  const gguf = minimalGguf();
  const manifest = buildGemma4Manifest(gguf);

  const off = planGemma4RunnerPlacement(gguf, manifest);
  assert.equal(off.status, "off");
  assert.equal(off.selectedLayerCount, 0);

  const blocked = planGemma4RunnerPlacement(gguf, manifest, {
    mode: "enabled",
    memoryLimitBytes: 1,
  });
  assert.equal(blocked.status, "blocked");
  assert.equal(blocked.selectedLayerCount, 0);

  const planned = planGemma4RunnerPlacement(gguf, manifest, {
    mode: "enabled",
    memoryLimitBytes: 2 * 1024 * 1024 * 1024,
  });
  assert.equal(planned.status, "planned");
  assert.equal(planned.selectedLayerCount, 2);
  assert.equal(planned.segmentStartLayer, 0);
});

test("runner placement copy audit reports unexpected copies", () => {
  const gguf = minimalGguf();
  const manifest = buildGemma4Manifest(gguf);
  const plan = planGemma4RunnerPlacement(gguf, manifest, {
    mode: "enabled",
    memoryLimitBytes: 2 * 1024 * 1024 * 1024,
  });

  const audit = auditGemma4RunnerPlacementCopies(plan, {
    decodeTensorReads: 0,
    segmentIntermediateReadbacks: 1,
    logitsReadbacks: 0,
    boundaryUploads: 0,
    tokenReadbacks: 0,
  });

  assert.equal(audit.ok, false);
  assert.match(audit.errors.join("\n"), /segment intermediate readbacks/);
});

function minimalGguf() {
  return {
    version: 3,
    tensorCount: 0,
    metadataCount: 14,
    dataStart: 0n,
    metadata: {
      "general.architecture": "gemma4",
      "gemma4.block_count": 2,
      "gemma4.embedding_length": 4,
      "gemma4.feed_forward_length": 8,
      "gemma4.attention.head_count": 1,
      "gemma4.attention.head_count_kv": 1,
      "gemma4.attention.key_length": 2,
      "gemma4.attention.value_length": 2,
      "gemma4.context_length": 16,
      "gemma4.full_attention_interval": 2,
      "gemma4.attention.layer_norm_rms_epsilon": 1e-6,
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
