import assert from "node:assert/strict";
import test from "node:test";

import {
  auditQwen35RunnerPlacementCopies,
  buildQwen35Manifest,
  planQwen35RunnerPlacement,
} from "../src/index.ts";

test("runner placement planning handles off, blocked, and planned WebGPU placement", () => {
  const gguf = minimalGguf();
  const manifest = buildQwen35Manifest(gguf);

  const off = planQwen35RunnerPlacement(gguf, manifest);
  assert.equal(off.status, "off");
  assert.equal(off.selectedLayerCount, 0);

  const blocked = planQwen35RunnerPlacement(gguf, manifest, {
    mode: "enabled",
    memoryLimitBytes: 1,
  });
  assert.equal(blocked.status, "blocked");
  assert.equal(blocked.selectedLayerCount, 0);

  const planned = planQwen35RunnerPlacement(gguf, manifest, {
    mode: "enabled",
    memoryLimitBytes: 2 * 1024 * 1024 * 1024,
  });
  assert.equal(planned.status, "planned");
  assert.equal(planned.selectedLayerCount, 2);
  assert.equal(planned.segmentStartLayer, 0);
});

test("runner placement copy audit reports unexpected copies", () => {
  const gguf = minimalGguf();
  const manifest = buildQwen35Manifest(gguf);
  const plan = planQwen35RunnerPlacement(gguf, manifest, {
    mode: "enabled",
    memoryLimitBytes: 2 * 1024 * 1024 * 1024,
  });

  const audit = auditQwen35RunnerPlacementCopies(plan, {
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
      "general.architecture": "qwen35",
      "qwen35.block_count": 2,
      "qwen35.embedding_length": 4,
      "qwen35.feed_forward_length": 8,
      "qwen35.attention.head_count": 1,
      "qwen35.attention.head_count_kv": 1,
      "qwen35.attention.key_length": 2,
      "qwen35.attention.value_length": 2,
      "qwen35.context_length": 16,
      "qwen35.full_attention_interval": 2,
      "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
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
