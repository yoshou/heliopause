import type { GgufMetadata, GgufTensorInfo } from "../gguf";
import type { Gemma4LayerKind, Gemma4ModelManifest } from "../model";
import { tensorByteLength } from "../tensor-reader";

export type Gemma4RunnerExecutionMode = "off" | "verify" | "enabled";

export type Gemma4RunnerPlanStatus =
  | "off"
  | "unavailable"
  | "blocked"
  | "planned";

export type Gemma4RunnerProviderSupport =
  | {
      available: true;
      [key: string]: unknown;
    }
  | {
      available: false;
      reason: string;
      error?: string;
    };

export type Gemma4RunnerLayerPlacement = {
  layer: number;
  layerKind: Gemma4LayerKind;
  weightBytes: number;
  cacheBytes: number;
  totalBytes: number;
};

export type Gemma4RunnerPlacementPlan = {
  status: Gemma4RunnerPlanStatus;
  mode: Gemma4RunnerExecutionMode;
  memoryLimitBytes: number;
  enabled: false;
  reason?: string;
  outputBytes: number;
  fixedBytes: number;
  scratchBytes: number;
  selectedLayerCount: number;
  segmentStartLayer?: number;
  cpuSegmentLayerCount: number;
  gpuSegmentLayerCount: number;
  gpuWeightBytes: number;
  gpuCacheBytes: number;
  estimatedResidentBytes: number;
  remainingBytes: number;
  selectedLayers: Gemma4RunnerLayerPlacement[];
  copyAuditExpectations: {
    decodeTensorReads: 0;
    segmentIntermediateReadbacks: 0;
    logitsReadbacks: 0;
    expectedBoundaryUploads: number;
    expectedTokenReadbacks: number;
    expectedSelectedTokenReadbacks: number;
  };
};

export type Gemma4RunnerCopyAuditObservation = {
  decodeTensorReads: number;
  segmentIntermediateReadbacks: number;
  logitsReadbacks: number;
  boundaryUploads: number;
  tokenReadbacks: number;
  selectedTokenReadbacks?: number;
};

export type Gemma4RunnerCopyAuditResult = {
  ok: boolean;
  errors: string[];
  expected: Gemma4RunnerPlacementPlan["copyAuditExpectations"];
  observed: Gemma4RunnerCopyAuditObservation;
};

export type Gemma4RunnerPlanningOptions = {
  mode?: Gemma4RunnerExecutionMode;
  memoryLimitBytes?: number;
  contextLength?: number;
  support?: Gemma4RunnerProviderSupport;
};

export type Gemma4RunnerPlanningProvider = {
  name: string;
  defaultMemoryLimitBytes: number;
  fixedBytes: number;
  scratchBytes: number;
  outputTensorNames: readonly string[];
  offReason: string;
  blockedByMemoryReason: string;
  unavailableReason: (support: Gemma4RunnerProviderSupport) => string;
  plannedReason: string;
  requiredSegmentStart?: (params: {
    manifest: Gemma4ModelManifest;
    selectedLayers: readonly Gemma4RunnerLayerPlacement[];
  }) => number | undefined;
  layerPlacement: (params: {
    tensorsByName: ReadonlyMap<string, GgufTensorInfo>;
    manifest: Gemma4ModelManifest;
    layer: number;
    contextLength: number;
  }) => Gemma4RunnerLayerPlacement;
  copyAuditExpectations: (selectedLayerCount: number) => Gemma4RunnerPlacementPlan["copyAuditExpectations"];
};

export function planGemma4ProviderPlacement(
  provider: Gemma4RunnerPlanningProvider,
  gguf: GgufMetadata,
  manifest: Gemma4ModelManifest,
  options: Gemma4RunnerPlanningOptions = {},
): Gemma4RunnerPlacementPlan {
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
  const selectedLayers: Gemma4RunnerLayerPlacement[] = [];

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

  const segmentStartLayer = selectedLayers[0]?.layer;
  const gpuWeightBytes = outputBytes +
    selectedLayers.reduce((sum, layer) => sum + layer.weightBytes, 0);
  const gpuCacheBytes = selectedLayers.reduce((sum, layer) => sum + layer.cacheBytes, 0);

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
    segmentStartLayer,
    cpuSegmentLayerCount: segmentStartLayer === undefined ? manifest.blockCount : segmentStartLayer,
    gpuSegmentLayerCount: selectedLayers.length,
    gpuWeightBytes,
    gpuCacheBytes,
    estimatedResidentBytes: selectedBytes,
    remainingBytes: Math.max(0, memoryLimitBytes - selectedBytes),
    selectedLayers,
    copyAuditExpectations: provider.copyAuditExpectations(selectedLayers.length),
  };
}

export function auditGemma4RunnerPlacementCopies(
  plan: Gemma4RunnerPlacementPlan,
  observed: Gemma4RunnerCopyAuditObservation,
): Gemma4RunnerCopyAuditResult {
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
  provider: Gemma4RunnerPlanningProvider,
  tensorsByName: ReadonlyMap<string, GgufTensorInfo>,
  manifest: Gemma4ModelManifest,
  contextLength: number,
): Map<number, Gemma4RunnerLayerPlacement> {
  const plans = new Map<number, Gemma4RunnerLayerPlacement>();
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

function emptyPlan(
  provider: Gemma4RunnerPlanningProvider,
  params: {
    status: Gemma4RunnerPlanStatus;
    mode: Gemma4RunnerExecutionMode;
    memoryLimitBytes: number;
    reason: string;
    outputBytes: number;
    blockCount: number;
  },
): Gemma4RunnerPlacementPlan {
  const estimatedResidentBytes = params.outputBytes + provider.fixedBytes + provider.scratchBytes;
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
    cpuSegmentLayerCount: params.blockCount,
    gpuSegmentLayerCount: 0,
    gpuWeightBytes: params.outputBytes,
    gpuCacheBytes: 0,
    estimatedResidentBytes,
    remainingBytes: Math.max(0, params.memoryLimitBytes - estimatedResidentBytes),
    selectedLayers: [],
    copyAuditExpectations: provider.copyAuditExpectations(0),
  };
}
