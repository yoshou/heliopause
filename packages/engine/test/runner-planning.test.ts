import assert from "node:assert/strict";
import test from "node:test";

import { auditRunnerPlacementCopies } from "./audit-helpers.ts";
import {
  planModelPlacement,
  type LayerResourceRequirement,
  type ProviderResourceRequirements,
  type RunnerNodePlacement,
} from "../src/index.ts";

test("runner placement planning handles off, unavailable, blocked, and planned states", () => {
  const cpu = providerRequirements("wasm", { constrained: false });
  const accelerator = providerRequirements("webgpu", {
    constrained: true,
    memoryLimitBytes: 32,
    residentBytes: [16, 16, 16],
  });

  const cpuOnly = planModelPlacement([cpu]);
  assert.equal(cpuOnly.status, "planned");
  assert.deepEqual(cpuOnly.segments.map((segment) => segment.provider), ["wasm"]);
  assert.deepEqual(cpuOnly.nodes.map((node) => "provider" in node ? node.provider : `${node.from}->${node.to}`), [
    "wasm",
    "wasm",
    "wasm",
  ]);

  const off = planModelPlacement([{ ...accelerator, mode: "off" }]);
  assert.equal(off.status, "off");
  assert.equal(off.nodes.length, 0);

  const unavailable = planModelPlacement([{
    ...accelerator,
    support: { available: false, reason: "device missing" },
  }]);
  assert.equal(unavailable.status, "unavailable");
  assert.match(unavailable.reason ?? "", /device missing/);

  const blocked = planModelPlacement([cpu, {
    ...accelerator,
    memoryLimitBytes: 1,
  }]);
  assert.equal(blocked.status, "blocked");
  assert.equal(blocked.nodes.length, 0);

  const planned = planModelPlacement([cpu, accelerator]);
  assert.equal(planned.status, "planned");
  assert.deepEqual(planned.segments.map((segment) => segment.provider), ["wasm", "webgpu"]);
  assert.deepEqual(planned.nodes.map((node) => node.kind), ["embedding", "segment", "transfer", "segment", "output"]);
  assert.equal(planned.copyExpectations.expectedBoundaryUploads, 1);
  assert.equal(planned.copyExpectations.expectedSelectedTokenReadbacks, 1);
});

test("runner placement uses a common resource schema for CPU and constrained providers", () => {
  const cpu = providerRequirements("reference", { constrained: false });
  const accelerator = providerRequirements("webgpu", { constrained: true });

  assert.deepEqual(Object.keys(cpu.layers[0] ?? {}).sort(), Object.keys(accelerator.layers[0] ?? {}).sort());
  assert.equal(cpu.layers[0]?.residentBytes, 0);
  assert.equal(accelerator.layers[0]?.residentBytes, 10);
});

test("runner placement only selects candidates connected to the output-side interval", () => {
  const cpu = providerRequirements("wasm", { constrained: false });
  const accelerator = providerRequirements("webgpu", {
    constrained: true,
    memoryLimitBytes: 10,
    costs: [1, 100, 10],
    residentBytes: [10, 10, 10],
  });
  const planned = planModelPlacement([cpu, accelerator]);

  assert.equal(planned.status, "planned");
  assert.deepEqual(planned.selectedLayers.map((layer) => layer.layer), [2]);
  assert.deepEqual(planned.segments.map((segment) => [segment.provider, segment.startLayer, segment.endLayerExclusive]), [
    ["wasm", 0, 2],
    ["webgpu", 2, 3],
  ]);
});

test("runner placement expands required source layers as one connected candidate", () => {
  const cpu = providerRequirements("wasm", { constrained: false });
  const accelerator = providerRequirements("webgpu", {
    constrained: true,
    memoryLimitBytes: 20,
    residentBytes: [10, 10, 10],
    requiredSourceLayers: {
      2: [1],
    },
  });
  const planned = planModelPlacement([cpu, accelerator]);

  assert.equal(planned.status, "planned");
  assert.deepEqual(planned.selectedLayers.map((layer) => layer.layer), [1, 2]);
  assert.deepEqual(planned.segments.map((segment) => [segment.provider, segment.startLayer, segment.endLayerExclusive]), [
    ["wasm", 0, 1],
    ["webgpu", 1, 3],
  ]);
});

test("runner placement chooses fallback provider from priority instead of a fixed provider", () => {
  const reference = providerRequirements("reference", { constrained: false });
  const wasm = providerRequirements("wasm", { constrained: false });
  const accelerator = providerRequirements("webgpu", {
    constrained: true,
    memoryLimitBytes: 20,
    residentBytes: [10, 10, 10],
  });
  const planned = planModelPlacement([reference, wasm, accelerator], {
    providerPriority: ["webgpu", "reference", "wasm"],
  });

  assert.equal(planned.status, "planned");
  assert.deepEqual(planned.segments.map((segment) => segment.provider), ["reference", "webgpu"]);
});

test("runner placement blocks partial constrained placement without fallback provider", () => {
  const accelerator = providerRequirements("webgpu", {
    constrained: true,
    memoryLimitBytes: 20,
    residentBytes: [10, 10, 10],
  });
  const planned = planModelPlacement([accelerator]);

  assert.equal(planned.status, "blocked");
  assert.match(planned.reason ?? "", /fallback provider/);
});

test("runner placement plan has no WebGPU-specific fields or metadata", () => {
  const cpu = providerRequirements("wasm", { constrained: false });
  const accelerator = providerRequirements("webgpu", { constrained: true });
  const planned = planModelPlacement([cpu, accelerator]);

  assert.equal("webGpuSegmentStartLayer" in planned, false);
  assert.equal("wasmSegmentLayerCount" in planned, false);
  assert.equal("webGpuSegmentLayerCount" in planned, false);
  assert.equal("webGpuWeightBytes" in planned, false);
  assert.equal("webGpuCacheBytes" in planned, false);
  assert.equal("webGpuSelectedLayers" in planned, false);
  assert.equal("metadata" in planned, false);
});

test("runner placement copy audit reports unexpected copies", () => {
  const plan = planModelPlacement([
    providerRequirements("wasm", { constrained: false }),
    providerRequirements("webgpu", { constrained: true, memoryLimitBytes: 10 }),
  ]);

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

test("runner placement copy audit treats selected token readback as the only normal constrained readback", () => {
  const plan = planModelPlacement([
    providerRequirements("wasm", { constrained: false }),
    providerRequirements("webgpu", { constrained: true, memoryLimitBytes: 10 }),
  ]);

  assert.equal(auditRunnerPlacementCopies(plan, {
    decodeTensorReads: 0,
    segmentIntermediateReadbacks: 0,
    logitsReadbacks: 0,
    boundaryUploads: 1,
    tokenReadbacks: 0,
    selectedTokenReadbacks: 1,
  }).ok, true);

  const topKReadback = auditRunnerPlacementCopies(plan, {
    decodeTensorReads: 0,
    segmentIntermediateReadbacks: 0,
    logitsReadbacks: 0,
    boundaryUploads: 1,
    tokenReadbacks: 1,
    selectedTokenReadbacks: 1,
  });
  assert.equal(topKReadback.ok, false);
  assert.match(topKReadback.errors.join("\n"), /token readbacks/);
});

function providerRequirements(
  provider: "reference" | "wasm" | "webgpu",
  options: {
    constrained: boolean;
    memoryLimitBytes?: number;
    costs?: readonly number[];
    residentBytes?: readonly number[];
    requiredSourceLayers?: Readonly<Record<number, readonly number[]>>;
  },
): ProviderResourceRequirements {
  const layers = [0, 1, 2].map((layer): LayerResourceRequirement => ({
    provider,
    layer,
    layerKind: "sliding-attention",
    weightBytes: 0,
    cacheBytes: 0,
    scratchBytes: 0,
    residentBytes: options.constrained ? options.residentBytes?.[layer] ?? 10 : 0,
    estimatedComputeCost: options.costs?.[layer] ?? layer + 1,
    requiredSourceLayers: options.requiredSourceLayers?.[layer] ?? [],
    supportedValueTransfers: ["hidden"],
  }));
  return {
    provider,
    mode: "enabled",
    support: { available: true },
    memoryLimitBytes: options.memoryLimitBytes ?? 1_000,
    fixedBytes: 0,
    outputBytes: 0,
    scratchBytes: 0,
    targetResourceConstrained: options.constrained,
    canRunFullModel: !options.constrained,
    offReason: `${provider} off`,
    blockedReason: `${provider} blocked`,
    plannedReason: `${provider} planned`,
    layers,
    copyExpectations: ({ selectedLayers, nodes }): {
      decodeTensorReads: 0;
      segmentIntermediateReadbacks: 0;
      logitsReadbacks: 0;
      expectedBoundaryUploads: number;
      expectedTokenReadbacks: number;
      expectedSelectedTokenReadbacks: number;
    } => ({
      decodeTensorReads: 0,
      segmentIntermediateReadbacks: 0,
      logitsReadbacks: 0,
      expectedBoundaryUploads: nodes.some((node: RunnerNodePlacement) => node.kind === "transfer") ? 1 : 0,
      expectedTokenReadbacks: 0,
      expectedSelectedTokenReadbacks: selectedLayers.length > 0 && options.constrained ? 1 : 0,
    }),
  };
}
