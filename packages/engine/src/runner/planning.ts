import type { LayerKind } from "../model";
import type { SegmentRunnerProvider } from "./segment-runner";

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

export type RunnerCopyExpectations = {
  decodeTensorReads: 0;
  segmentIntermediateReadbacks: 0;
  logitsReadbacks: 0;
  expectedBoundaryUploads: number;
  expectedTokenReadbacks: number;
  expectedSelectedTokenReadbacks: number;
};

export type LayerResourceRequirement = {
  provider: SegmentRunnerProvider;
  layer: number;
  layerKind: LayerKind;
  weightBytes: number;
  cacheBytes: number;
  scratchBytes: number;
  residentBytes: number;
  estimatedComputeCost: number;
  requiredSourceLayers: readonly number[];
  supportedValueTransfers: readonly "hidden"[];
};

export type RunnerLayerPlacement = LayerResourceRequirement;

export type ProviderResourceRequirements = {
  provider: SegmentRunnerProvider;
  mode: RunnerExecutionMode;
  support: RunnerProviderSupport;
  memoryLimitBytes: number;
  fixedBytes: number;
  outputBytes: number;
  scratchBytes: number;
  targetResourceConstrained: boolean;
  canRunFullModel: boolean;
  offReason: string;
  blockedReason: string;
  plannedReason: string;
  layers: readonly LayerResourceRequirement[];
  copyExpectations?: (params: {
    selectedLayers: readonly LayerResourceRequirement[];
    nodes: readonly RunnerNodePlacement[];
  }) => RunnerCopyExpectations;
};

export type ResourceBudget = {
  mode?: RunnerExecutionMode;
  memoryLimitBytes?: number;
  providerPriority?: readonly SegmentRunnerProvider[];
};

export type RunnerResourceUsage = {
  provider: SegmentRunnerProvider;
  memoryLimitBytes: number;
  fixedBytes: number;
  outputBytes: number;
  scratchBytes: number;
  selectedLayerCount: number;
  selectedResidentBytes: number;
  totalResidentBytes: number;
  remainingBytes: number;
};

export type RunnerSegmentPlacement = {
  provider: SegmentRunnerProvider;
  startLayer: number;
  endLayerExclusive: number;
  layerCount: number;
  weightBytes: number;
  cacheBytes: number;
  residentBytes: number;
  estimatedComputeCost: number;
};

export type RunnerNodePlacement =
  | {
      kind: "embedding";
      provider: SegmentRunnerProvider;
    }
  | {
      kind: "segment";
      provider: SegmentRunnerProvider;
      startLayer: number;
      endLayerExclusive: number;
      layerCount: number;
      weightBytes: number;
      cacheBytes: number;
    }
  | {
      kind: "transfer";
      from: SegmentRunnerProvider;
      to: SegmentRunnerProvider;
      via: "cpu";
      value: "hidden";
    }
  | {
      kind: "output";
      provider: SegmentRunnerProvider;
    };

export type RunnerPlacementPlan = {
  status: RunnerPlanStatus;
  mode: RunnerExecutionMode;
  reason?: string;
  segments: RunnerSegmentPlacement[];
  nodes: RunnerNodePlacement[];
  selectedLayers: RunnerLayerPlacement[];
  resourceUsage: RunnerResourceUsage[];
  copyExpectations: RunnerCopyExpectations;
};

const EMPTY_COPY_EXPECTATIONS: RunnerCopyExpectations = {
  decodeTensorReads: 0,
  segmentIntermediateReadbacks: 0,
  logitsReadbacks: 0,
  expectedBoundaryUploads: 0,
  expectedTokenReadbacks: 0,
  expectedSelectedTokenReadbacks: 0,
};

export function planModelPlacement(
  requirements: readonly ProviderResourceRequirements[],
  budget: ResourceBudget = {},
): RunnerPlacementPlan {
  const ordered = orderRequirements(requirements, budget.providerPriority);
  const mode = budget.mode ?? "enabled";
  const unavailable = ordered.find((item) => item.support.available === false);
  const off = ordered.find((item) => effectiveMode(item, mode) === "off");
  const fallback = ordered.find((item) =>
    effectiveMode(item, mode) !== "off" &&
    item.support.available &&
    item.canRunFullModel &&
    !item.targetResourceConstrained
  );
  const accelerated = ordered.find((item) =>
    effectiveMode(item, mode) !== "off" &&
    item.support.available &&
    item.targetResourceConstrained
  );

  if (!accelerated) {
    if (fallback) {
      return fullProviderPlan(fallback, mode, budget);
    }
    if (off) {
      return emptyPlan("off", mode, off.offReason);
    }
    if (unavailable && !unavailable.support.available) {
      return emptyPlan("unavailable", mode, unavailable.support.reason);
    }
    return emptyPlan("blocked", mode, "No executable model provider is available.");
  }

  const selected = selectConnectedLayers(accelerated, budget);
  if (selected.length === 0) {
    return emptyPlan("blocked", mode, accelerated.blockedReason);
  }

  const blockCount = accelerated.layers.length;
  const startLayer = selected[0]?.layer ?? blockCount;
  if (startLayer > 0 && !fallback) {
    return emptyPlan("blocked", mode, "No fallback provider can execute unselected layers.");
  }

  const segments: RunnerSegmentPlacement[] = [];
  if (startLayer > 0 && fallback) {
    segments.push(segmentFromLayers(fallback.provider, fallback.layers.slice(0, startLayer)));
  }
  segments.push(segmentFromLayers(accelerated.provider, selected));
  const nodes = buildNodes(segments);
  const resourceUsage = [
    resourceUsageFor(accelerated, selected, budget),
  ];
  if (fallback && startLayer > 0) {
    resourceUsage.unshift(resourceUsageFor(fallback, fallback.layers.slice(0, startLayer), budget));
  }

  return {
    status: "planned",
    mode,
    reason: accelerated.plannedReason,
    segments,
    nodes,
    selectedLayers: selected,
    resourceUsage,
    copyExpectations: copyExpectationsFor(accelerated, selected, nodes),
  };
}

function orderRequirements(
  requirements: readonly ProviderResourceRequirements[],
  priority: readonly SegmentRunnerProvider[] | undefined,
): ProviderResourceRequirements[] {
  if (!priority) {
    return requirements.slice();
  }
  const byProvider = new Map(requirements.map((item) => [item.provider, item]));
  const ordered = priority.flatMap((provider) => {
    const item = byProvider.get(provider);
    return item ? [item] : [];
  });
  for (const item of requirements) {
    if (!priority.includes(item.provider)) {
      ordered.push(item);
    }
  }
  return ordered;
}

function effectiveMode(
  requirements: ProviderResourceRequirements,
  mode: RunnerExecutionMode,
): RunnerExecutionMode {
  return mode === "off" ? "off" : requirements.mode;
}

function fullProviderPlan(
  provider: ProviderResourceRequirements,
  mode: RunnerExecutionMode,
  budget: ResourceBudget,
): RunnerPlacementPlan {
  const segments = provider.layers.length > 0 ? [segmentFromLayers(provider.provider, provider.layers)] : [];
  const nodes = provider.layers.length > 0
    ? buildNodes(segments)
    : [
      { kind: "embedding", provider: provider.provider },
      { kind: "output", provider: provider.provider },
    ] satisfies RunnerNodePlacement[];
  return {
    status: "planned",
    mode,
    reason: provider.plannedReason,
    segments,
    nodes,
    selectedLayers: provider.layers.slice(),
    resourceUsage: [resourceUsageFor(provider, provider.layers, budget)],
    copyExpectations: copyExpectationsFor(provider, provider.layers, nodes),
  };
}

function selectConnectedLayers(
  provider: ProviderResourceRequirements,
  budget: ResourceBudget,
): LayerResourceRequirement[] {
  const layers = provider.layers.slice().sort((left, right) => left.layer - right.layer);
  const limit = budget.memoryLimitBytes ?? provider.memoryLimitBytes;
  let selectedStart = layers.length;
  let selectedResidentBytes = 0;
  const selected = new Map<number, LayerResourceRequirement>();
  const rejected = new Set<number>();

  for (;;) {
    const candidates = layers
      .filter((layer) => !selected.has(layer.layer) && !rejected.has(layer.layer))
      .map((layer) => candidateGroup(layers, layer.layer, selectedStart))
      .filter((candidate): candidate is LayerResourceRequirement[] => candidate !== undefined)
      .sort(compareCandidateGroups);
    const candidate = candidates[0];
    if (!candidate) {
      break;
    }

    const additionalResidentBytes = candidate.reduce((sum, layer) => sum + layer.residentBytes, 0);
    const totalResidentBytes = provider.fixedBytes + provider.outputBytes + provider.scratchBytes +
      selectedResidentBytes + additionalResidentBytes;
    if (totalResidentBytes > limit) {
      for (const layer of candidate) {
        rejected.add(layer.layer);
      }
      continue;
    }

    for (const layer of candidate) {
      selected.set(layer.layer, layer);
      selectedStart = Math.min(selectedStart, layer.layer);
    }
    selectedResidentBytes += additionalResidentBytes;
  }

  return [...selected.values()].sort((left, right) => left.layer - right.layer);
}

function candidateGroup(
  layers: readonly LayerResourceRequirement[],
  layer: number,
  selectedStart: number,
): LayerResourceRequirement[] | undefined {
  if (layer !== selectedStart - 1) {
    return undefined;
  }
  const layerByIndex = new Map(layers.map((item) => [item.layer, item]));
  const current = layerByIndex.get(layer);
  if (!current) {
    return undefined;
  }
  const requiredStart = Math.min(layer, ...current.requiredSourceLayers.filter((source) => source < selectedStart));
  const group: LayerResourceRequirement[] = [];
  for (let index = requiredStart; index < selectedStart; index += 1) {
    const item = layerByIndex.get(index);
    if (!item) {
      return undefined;
    }
    group.push(item);
  }
  return group;
}

function compareCandidateGroups(
  left: readonly LayerResourceRequirement[],
  right: readonly LayerResourceRequirement[],
): number {
  const leftCost = sum(left, (item) => item.estimatedComputeCost);
  const rightCost = sum(right, (item) => item.estimatedComputeCost);
  if (leftCost !== rightCost) {
    return leftCost - rightCost;
  }
  const leftResident = sum(left, (item) => item.residentBytes);
  const rightResident = sum(right, (item) => item.residentBytes);
  if (leftResident !== rightResident) {
    return leftResident - rightResident;
  }
  return Math.max(...right.map((item) => item.layer)) - Math.max(...left.map((item) => item.layer));
}

function segmentFromLayers(
  provider: SegmentRunnerProvider,
  layers: readonly LayerResourceRequirement[],
): RunnerSegmentPlacement {
  const first = layers[0];
  const last = layers[layers.length - 1];
  if (!first || !last) {
    throw new Error(`Cannot build empty segment for ${provider}.`);
  }
  return {
    provider,
    startLayer: first.layer,
    endLayerExclusive: last.layer + 1,
    layerCount: layers.length,
    weightBytes: sum(layers, (layer) => layer.weightBytes),
    cacheBytes: sum(layers, (layer) => layer.cacheBytes),
    residentBytes: sum(layers, (layer) => layer.residentBytes),
    estimatedComputeCost: sum(layers, (layer) => layer.estimatedComputeCost),
  };
}

function buildNodes(segments: readonly RunnerSegmentPlacement[]): RunnerNodePlacement[] {
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

function resourceUsageFor(
  provider: ProviderResourceRequirements,
  layers: readonly LayerResourceRequirement[],
  budget: ResourceBudget,
): RunnerResourceUsage {
  const memoryLimitBytes = budget.memoryLimitBytes ?? provider.memoryLimitBytes;
  const selectedResidentBytes = sum(layers, (layer) => layer.residentBytes);
  const totalResidentBytes = provider.fixedBytes + provider.outputBytes + provider.scratchBytes + selectedResidentBytes;
  return {
    provider: provider.provider,
    memoryLimitBytes,
    fixedBytes: provider.fixedBytes,
    outputBytes: provider.outputBytes,
    scratchBytes: provider.scratchBytes,
    selectedLayerCount: layers.length,
    selectedResidentBytes,
    totalResidentBytes,
    remainingBytes: Math.max(0, memoryLimitBytes - totalResidentBytes),
  };
}

function copyExpectationsFor(
  provider: ProviderResourceRequirements,
  selectedLayers: readonly LayerResourceRequirement[],
  nodes: readonly RunnerNodePlacement[],
): RunnerCopyExpectations {
  return provider.copyExpectations?.({ selectedLayers, nodes }) ?? EMPTY_COPY_EXPECTATIONS;
}

function emptyPlan(
  status: Exclude<RunnerPlanStatus, "planned">,
  mode: RunnerExecutionMode,
  reason: string,
): RunnerPlacementPlan {
  return {
    status,
    mode,
    reason,
    segments: [],
    nodes: [],
    selectedLayers: [],
    resourceUsage: [],
    copyExpectations: EMPTY_COPY_EXPECTATIONS,
  };
}

function sum<T>(items: readonly T[], value: (item: T) => number): number {
  return items.reduce((total, item) => total + value(item), 0);
}
