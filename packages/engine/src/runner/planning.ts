import type { GgufMetadata, GgufTensorInfo } from "../gguf";
import type { LayerKind, ModelManifest } from "../model";
import { tensorByteLength } from "../tensor-reader";

export type RunnerExecutionMode = "off" | "verify" | "enabled";

export type RunnerPlanStatus =
  | "off"
  | "unavailable"
  | "blocked"
  | "planned";

export type RunnerProviderSupport =
  | {
      available: true;
      [key: string]: unknown;
    }
  | {
      available: false;
      reason: string;
      error?: string;
    };

export type RunnerLayerPlacement = {
  layer: number;
  layerKind: LayerKind;
  weightBytes: number;
  cacheBytes: number;
  totalBytes: number;
};

export type RunnerSegmentProvider = "reference" | "wasm" | "webgpu";

export type RunnerSegmentPlacement = {
  provider: RunnerSegmentProvider;
  startLayer: number;
  endLayerExclusive: number;
  layerCount: number;
  weightBytes: number;
  cacheBytes: number;
};

export type RunnerNodePlacement =
  | {
      kind: "embedding";
      provider: RunnerSegmentProvider;
    }
  | {
      kind: "segment";
      provider: RunnerSegmentProvider;
      startLayer: number;
      endLayerExclusive: number;
      layerCount: number;
      weightBytes: number;
      cacheBytes: number;
    }
  | {
      kind: "transfer";
      from: RunnerSegmentProvider;
      to: RunnerSegmentProvider;
      via: "cpu";
      value: "hidden";
    }
  | {
      kind: "output";
      provider: RunnerSegmentProvider;
    };

export type RunnerPlacementPlan = {
  status: RunnerPlanStatus;
  mode: RunnerExecutionMode;
  memoryLimitBytes: number;
  enabled: false;
  reason?: string;
  outputBytes: number;
  fixedBytes: number;
  scratchBytes: number;
  selectedLayerCount: number;
  webGpuSegmentStartLayer?: number;
  wasmSegmentLayerCount: number;
  webGpuSegmentLayerCount: number;
  webGpuWeightBytes: number;
  webGpuCacheBytes: number;
  estimatedResidentBytes: number;
  remainingBytes: number;
  webGpuSelectedLayers: RunnerLayerPlacement[];
  segments: RunnerSegmentPlacement[];
  nodes: RunnerNodePlacement[];
  copyAuditExpectations: {
    decodeTensorReads: 0;
    segmentIntermediateReadbacks: 0;
    logitsReadbacks: 0;
    expectedBoundaryUploads: number;
    expectedTokenReadbacks: number;
    expectedSelectedTokenReadbacks: number;
  };
};

export type RunnerCopyAuditObservation = {
  decodeTensorReads: number;
  segmentIntermediateReadbacks: number;
  logitsReadbacks: number;
  boundaryUploads: number;
  tokenReadbacks: number;
  selectedTokenReadbacks?: number;
};

export type RunnerCopyAuditResult = {
  ok: boolean;
  errors: string[];
  expected: RunnerPlacementPlan["copyAuditExpectations"];
  observed: RunnerCopyAuditObservation;
};

export type RunnerPlanningOptions = {
  mode?: RunnerExecutionMode;
  memoryLimitBytes?: number;
  contextLength?: number;
  support?: RunnerProviderSupport;
};

export type RunnerPlanningProvider = {
  name: string;
  defaultMemoryLimitBytes: number;
  fixedBytes: number;
  scratchBytes: number;
  outputTensorNames: readonly string[];
  offReason: string;
  blockedByMemoryReason: string;
  unavailableReason: (support: RunnerProviderSupport) => string;
  plannedReason: string;
  requiredSegmentStart?: (params: {
    manifest: ModelManifest;
    selectedLayers: readonly RunnerLayerPlacement[];
  }) => number | undefined;
  layerPlacement: (params: {
    tensorsByName: ReadonlyMap<string, GgufTensorInfo>;
    manifest: ModelManifest;
    layer: number;
    contextLength: number;
  }) => RunnerLayerPlacement;
  copyAuditExpectations: (selectedLayerCount: number) => RunnerPlacementPlan["copyAuditExpectations"];
};

export function planProviderPlacement(
  provider: RunnerPlanningProvider,
  gguf: GgufMetadata,
  manifest: ModelManifest,
  options: RunnerPlanningOptions = {},
): RunnerPlacementPlan {
  const mode = options.mode ?? "off";
  const memoryLimitBytes = options.memoryLimitBytes ?? provider.defaultMemoryLimitBytes;
  const contextLength = Math.min(
    options.contextLength ?? manifest.contextLength,
    manifest.contextLength,
  );
  const support = options.support;
  const tensorsByName = new Map(gguf.tensors.map((tensor) => [tensor.name, tensor]));
  const outputBytes = provider.outputTensorNames.reduce(
    (sum, name) => sum + tensorBytes(tensorsByName, name),
    0,
  );
  const layerPlans = buildLayerPlans(provider, tensorsByName, manifest, contextLength);

  if (mode === "off") {
    return emptyPlan(provider, {
      status: "off",
      mode,
      memoryLimitBytes,
      reason: provider.offReason,
      outputBytes,
      blockCount: manifest.blockCount,
    });
  }

  if (support && !support.available) {
    return emptyPlan(provider, {
      status: "unavailable",
      mode,
      memoryLimitBytes,
      reason: provider.unavailableReason(support),
      outputBytes,
      blockCount: manifest.blockCount,
    });
  }

  let selectedBytes = outputBytes + provider.fixedBytes + provider.scratchBytes;
  const selectedLayers: RunnerLayerPlacement[] = [];

  if (selectedBytes > memoryLimitBytes) {
    return emptyPlan(provider, {
      status: "blocked",
      mode,
      memoryLimitBytes,
      reason: provider.blockedByMemoryReason,
      outputBytes,
      blockCount: manifest.blockCount,
    });
  }

  for (let layer = manifest.blockCount - 1; layer >= 0; layer -= 1) {
    const candidate = layerPlans.get(layer);
    if (!candidate) {
      continue;
    }
    if (selectedBytes + candidate.totalBytes > memoryLimitBytes) {
      break;
    }
    selectedLayers.unshift(candidate);
    selectedBytes += candidate.totalBytes;
  }

  const requiredStart = provider.requiredSegmentStart?.({ manifest, selectedLayers });
  if (requiredStart !== undefined && selectedLayers.length > 0) {
    const currentStart = selectedLayers[0]?.layer ?? manifest.blockCount;
    for (let layer = currentStart - 1; layer >= requiredStart; layer -= 1) {
      const candidate = layerPlans.get(layer);
      if (!candidate) {
        continue;
      }
      if (selectedBytes + candidate.totalBytes > memoryLimitBytes) {
        return emptyPlan(provider, {
          status: "blocked",
          mode,
          memoryLimitBytes,
          reason: provider.blockedByMemoryReason,
          outputBytes,
          blockCount: manifest.blockCount,
        });
      }
      selectedLayers.unshift(candidate);
      selectedBytes += candidate.totalBytes;
    }
  }

  const webGpuSegmentStartLayer = selectedLayers[0]?.layer;
  const webGpuWeightBytes = outputBytes +
    selectedLayers.reduce((sum, layer) => sum + layer.weightBytes, 0);
  const webGpuCacheBytes = selectedLayers.reduce((sum, layer) => sum + layer.cacheBytes, 0);
  const segments = buildHybridSegments({
    blockCount: manifest.blockCount,
    webGpuSegmentStartLayer,
    webGpuSegmentEndLayer: selectedLayers.length > 0 ? manifest.blockCount : undefined,
    webGpuWeightBytes,
    webGpuCacheBytes,
  });
  const nodes = buildHybridNodes(segments);

  return {
    status: "planned",
    mode,
    memoryLimitBytes,
    enabled: false,
    reason: provider.plannedReason,
    outputBytes,
    fixedBytes: provider.fixedBytes,
    scratchBytes: provider.scratchBytes,
    selectedLayerCount: selectedLayers.length,
    webGpuSegmentStartLayer,
    wasmSegmentLayerCount: webGpuSegmentStartLayer === undefined ? manifest.blockCount : webGpuSegmentStartLayer,
    webGpuSegmentLayerCount: selectedLayers.length,
    webGpuWeightBytes,
    webGpuCacheBytes,
    estimatedResidentBytes: selectedBytes,
    remainingBytes: Math.max(0, memoryLimitBytes - selectedBytes),
    webGpuSelectedLayers: selectedLayers,
    segments,
    nodes,
    copyAuditExpectations: {
      ...provider.copyAuditExpectations(selectedLayers.length),
      expectedBoundaryUploads: nodes.some((node) =>
        node.kind === "transfer" && node.to === "webgpu"
      ) ? 1 : 0,
    },
  };
}

export function auditRunnerPlacementCopies(
  plan: RunnerPlacementPlan,
  observed: RunnerCopyAuditObservation,
): RunnerCopyAuditResult {
  const expected = plan.copyAuditExpectations;
  const errors: string[] = [];

  if (observed.decodeTensorReads !== expected.decodeTensorReads) {
    errors.push(
      `decode tensor reads: expected ${expected.decodeTensorReads}, got ${observed.decodeTensorReads}`,
    );
  }
  if (observed.segmentIntermediateReadbacks !== expected.segmentIntermediateReadbacks) {
    errors.push(
      `segment intermediate readbacks: expected ${expected.segmentIntermediateReadbacks}, got ${observed.segmentIntermediateReadbacks}`,
    );
  }
  if (observed.logitsReadbacks !== expected.logitsReadbacks) {
    errors.push(`logits readbacks: expected ${expected.logitsReadbacks}, got ${observed.logitsReadbacks}`);
  }
  if (observed.boundaryUploads > expected.expectedBoundaryUploads) {
    errors.push(
      `boundary uploads: expected at most ${expected.expectedBoundaryUploads}, got ${observed.boundaryUploads}`,
    );
  }
  if (observed.tokenReadbacks > expected.expectedTokenReadbacks) {
    errors.push(
      `token readbacks: expected at most ${expected.expectedTokenReadbacks}, got ${observed.tokenReadbacks}`,
    );
  }
  const observedSelectedTokenReadbacks = observed.selectedTokenReadbacks ?? 0;
  if (observedSelectedTokenReadbacks > expected.expectedSelectedTokenReadbacks) {
    errors.push(
      `selected token readbacks: expected at most ${expected.expectedSelectedTokenReadbacks}, got ${observedSelectedTokenReadbacks}`,
    );
  }

  return {
    ok: errors.length === 0,
    errors,
    expected,
    observed,
  };
}

function buildLayerPlans(
  provider: RunnerPlanningProvider,
  tensorsByName: ReadonlyMap<string, GgufTensorInfo>,
  manifest: ModelManifest,
  contextLength: number,
): Map<number, RunnerLayerPlacement> {
  const plans = new Map<number, RunnerLayerPlacement>();
  for (let layer = 0; layer < manifest.blockCount; layer += 1) {
    plans.set(layer, provider.layerPlacement({
      tensorsByName,
      manifest,
      layer,
      contextLength,
    }));
  }
  return plans;
}

function tensorBytes(tensorsByName: ReadonlyMap<string, GgufTensorInfo>, name: string): number {
  const tensor = tensorsByName.get(name);
  return tensor ? tensorByteLength(tensor) : 0;
}

function buildHybridSegments(params: {
  blockCount: number;
  webGpuSegmentStartLayer?: number;
  webGpuSegmentEndLayer?: number;
  webGpuWeightBytes: number;
  webGpuCacheBytes: number;
}): RunnerSegmentPlacement[] {
  const webGpuStart = params.webGpuSegmentStartLayer;
  if (webGpuStart === undefined || params.webGpuSegmentEndLayer === undefined) {
    return [{
      provider: "wasm",
      startLayer: 0,
      endLayerExclusive: params.blockCount,
      layerCount: params.blockCount,
      weightBytes: 0,
      cacheBytes: 0,
    }];
  }

  const segments: RunnerSegmentPlacement[] = [];
  if (webGpuStart > 0) {
    segments.push({
      provider: "wasm",
      startLayer: 0,
      endLayerExclusive: webGpuStart,
      layerCount: webGpuStart,
      weightBytes: 0,
      cacheBytes: 0,
    });
  }
  segments.push({
    provider: "webgpu",
    startLayer: webGpuStart,
    endLayerExclusive: params.webGpuSegmentEndLayer,
    layerCount: params.webGpuSegmentEndLayer - webGpuStart,
    weightBytes: params.webGpuWeightBytes,
    cacheBytes: params.webGpuCacheBytes,
  });
  return segments;
}

function buildHybridNodes(segments: readonly RunnerSegmentPlacement[]): RunnerNodePlacement[] {
  const first = segments[0];
  if (!first) {
    return [];
  }

  const nodes: RunnerNodePlacement[] = [
    {
      kind: "embedding",
      provider: first.provider,
    },
  ];
  let previousProvider = first.provider;
  for (const segment of segments) {
    if (segment.provider !== previousProvider) {
      nodes.push({
        kind: "transfer",
        from: previousProvider,
        to: segment.provider,
        via: "cpu",
        value: "hidden",
      });
    }
    if (segment.layerCount > 0) {
      nodes.push({
        kind: "segment",
        provider: segment.provider,
        startLayer: segment.startLayer,
        endLayerExclusive: segment.endLayerExclusive,
        layerCount: segment.layerCount,
        weightBytes: segment.weightBytes,
        cacheBytes: segment.cacheBytes,
      });
    }
    previousProvider = segment.provider;
  }
  nodes.push({
    kind: "output",
    provider: previousProvider,
  });
  return nodes;
}

function emptyPlan(
  provider: RunnerPlanningProvider,
  params: {
    status: RunnerPlanStatus;
    mode: RunnerExecutionMode;
    memoryLimitBytes: number;
    reason: string;
    outputBytes: number;
    blockCount: number;
  },
): RunnerPlacementPlan {
  const estimatedResidentBytes = params.outputBytes + provider.fixedBytes + provider.scratchBytes;
  const segments: RunnerSegmentPlacement[] = [{
    provider: "wasm",
    startLayer: 0,
    endLayerExclusive: params.blockCount,
    layerCount: params.blockCount,
    weightBytes: 0,
    cacheBytes: 0,
  }];
  return {
    status: params.status,
    mode: params.mode,
    memoryLimitBytes: params.memoryLimitBytes,
    enabled: false,
    reason: params.reason,
    outputBytes: params.outputBytes,
    fixedBytes: provider.fixedBytes,
    scratchBytes: provider.scratchBytes,
    selectedLayerCount: 0,
    wasmSegmentLayerCount: params.blockCount,
    webGpuSegmentLayerCount: 0,
    webGpuWeightBytes: params.outputBytes,
    webGpuCacheBytes: 0,
    estimatedResidentBytes,
    remainingBytes: Math.max(0, params.memoryLimitBytes - estimatedResidentBytes),
    webGpuSelectedLayers: [],
    segments,
    nodes: buildHybridNodes(segments),
    copyAuditExpectations: provider.copyAuditExpectations(0),
  };
}
