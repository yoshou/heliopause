import assert from "node:assert/strict";
import test from "node:test";

import {
  auditRunnerPlacementCopies,
  buildModelManifest,
  planRunnerPlacement,
} from "../src/index.ts";

test("runner placement planning handles off, blocked, and planned WebGPU placement", () => {
  const gguf = minimalGguf();
  const manifest = buildModelManifest(gguf);

  const off = planRunnerPlacement(gguf, manifest);
  assert.equal(off.status, "off");
  assert.equal(off.selectedLayerCount, 0);
  assert.deepEqual(off.segments.map((segment) => segment.provider), ["wasm"]);

  const blocked = planRunnerPlacement(gguf, manifest, {
    mode: "enabled",
    memoryLimitBytes: 1,
  });
  assert.equal(blocked.status, "blocked");
  assert.equal(blocked.selectedLayerCount, 0);
  assert.deepEqual(blocked.segments.map((segment) => segment.provider), ["wasm"]);

  const planned = planRunnerPlacement(gguf, manifest, {
    mode: "enabled",
    memoryLimitBytes: 2 * 1024 * 1024 * 1024,
  });
  assert.equal(planned.status, "planned");
  assert.equal(planned.selectedLayerCount, 2);
  assert.equal(planned.webGpuSegmentStartLayer, 0);
  assert.deepEqual(planned.segments.map((segment) => segment.provider), ["webgpu"]);
  assert.deepEqual(planned.segments.map((segment) => [segment.startLayer, segment.endLayerExclusive]), [[0, 2]]);
  assert.deepEqual(planned.nodes.map((node) => node.kind), ["embedding", "segment", "output"]);
  assert.deepEqual(planned.nodes.map((node) => "provider" in node ? node.provider : `${node.from}->${node.to}`), [
    "webgpu",
    "webgpu",
    "webgpu",
  ]);
  assert.equal(planned.wasmSegmentLayerCount, 0);
  assert.equal(planned.webGpuSegmentLayerCount, 2);
  assert.equal(planned.copyAuditExpectations.expectedTokenReadbacks, 0);
  assert.equal(planned.copyAuditExpectations.expectedBoundaryUploads, 0);
  assert.equal(planned.copyAuditExpectations.expectedSelectedTokenReadbacks, 1);
});

test("runner placement planning exposes wasm prefix plus WebGPU suffix", () => {
  const gguf = minimalGguf();
  const manifest = buildModelManifest(gguf);
  const baseline = planRunnerPlacement(gguf, manifest);
  const planned = planRunnerPlacement(gguf, manifest, {
    mode: "enabled",
    memoryLimitBytes: baseline.fixedBytes + baseline.scratchBytes + baseline.outputBytes + 300,
  });

  assert.equal(planned.status, "planned");
  assert.deepEqual(planned.segments.map((segment) => segment.provider), ["wasm", "webgpu"]);
  assert.deepEqual(planned.nodes.map((node) => node.kind), ["embedding", "segment", "transfer", "segment", "output"]);
  assert.deepEqual(planned.nodes.filter((node) => node.kind === "transfer"), [{
    kind: "transfer",
    from: "wasm",
    to: "webgpu",
    via: "cpu",
    value: "hidden",
  }]);
  assert.equal(planned.segments[0]?.endLayerExclusive, planned.segments[1]?.startLayer);
  assert.equal(planned.copyAuditExpectations.expectedBoundaryUploads, 1);
});

test("runner placement copy audit reports unexpected copies", () => {
  const gguf = minimalGguf();
  const manifest = buildModelManifest(gguf);
  const plan = planRunnerPlacement(gguf, manifest, {
    mode: "enabled",
    memoryLimitBytes: 2 * 1024 * 1024 * 1024,
  });

  const audit = auditRunnerPlacementCopies(plan, {
    decodeTensorReads: 0,
    segmentIntermediateReadbacks: 1,
    logitsReadbacks: 0,
    boundaryUploads: 0,
    tokenReadbacks: 0,
    selectedTokenReadbacks: 0,
  });

  assert.equal(audit.ok, false);
  assert.match(audit.errors.join("\n"), /segment intermediate readbacks/);
});

test("runner placement copy audit treats selected token readback as the only normal WebGPU readback", () => {
  const gguf = minimalGguf();
  const manifest = buildModelManifest(gguf);
  const plan = planRunnerPlacement(gguf, manifest, {
    mode: "enabled",
    memoryLimitBytes: 2 * 1024 * 1024 * 1024,
  });

  assert.equal(auditRunnerPlacementCopies(plan, {
    decodeTensorReads: 0,
    segmentIntermediateReadbacks: 0,
    logitsReadbacks: 0,
    boundaryUploads: 0,
    tokenReadbacks: 0,
    selectedTokenReadbacks: 1,
  }).ok, true);

  const topKReadback = auditRunnerPlacementCopies(plan, {
    decodeTensorReads: 0,
    segmentIntermediateReadbacks: 0,
    logitsReadbacks: 0,
    boundaryUploads: 0,
    tokenReadbacks: 1,
    selectedTokenReadbacks: 1,
  });
  assert.equal(topKReadback.ok, false);
  assert.match(topKReadback.errors.join("\n"), /token readbacks/);
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
