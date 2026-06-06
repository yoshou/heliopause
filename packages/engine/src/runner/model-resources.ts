import type { GgufMetadata, GgufTensorInfo } from "../gguf";
import type { LayerKind, ModelManifest } from "../model";
import { tensorByteLength } from "../tensor-reader";
import type {
  LayerResourceRequirement,
  ProviderResourceRequirements,
  RunnerCopyExpectations,
  RunnerExecutionMode,
  RunnerNodePlacement,
  RunnerProviderSupport,
} from "./planning";
import type { SegmentRunnerProvider } from "./segment-runner";

export type ModelResourceRequirementOptions = {
  provider: SegmentRunnerProvider;
  gguf: GgufMetadata;
  manifest: ModelManifest;
  contextLength: number;
  mode?: RunnerExecutionMode;
  support?: RunnerProviderSupport;
  memoryLimitBytes?: number;
  fixedBytes?: number;
  scratchBytes?: number;
  cacheElementByteLength?: number;
  outputTensorNames?: readonly string[];
  targetResourceConstrained?: boolean;
  canRunFullModel?: boolean;
  offReason?: string;
  blockedReason?: string;
  plannedReason?: string;
  residentBytes?: (params: {
    layer: number;
    layerKind: LayerKind;
    weightBytes: number;
    cacheBytes: number;
    scratchBytes: number;
  }) => number;
  estimatedComputeCost?: (params: {
    layer: number;
    layerKind: LayerKind;
    weightBytes: number;
    cacheBytes: number;
    residentBytes: number;
  }) => number;
  requiredSourceLayers?: (params: {
    layer: number;
    layerKind: LayerKind;
    manifest: ModelManifest;
  }) => readonly number[];
  copyExpectations?: (params: {
    selectedLayers: readonly LayerResourceRequirement[];
    nodes: readonly RunnerNodePlacement[];
  }) => RunnerCopyExpectations;
};

export function createModelResourceRequirements(
  options: ModelResourceRequirementOptions,
): ProviderResourceRequirements {
  const tensorsByName = new Map(options.gguf.tensors.map((tensor) => [tensor.name, tensor]));
  const outputBytes = (options.outputTensorNames ?? []).reduce(
    (total, name) => total + tensorBytes(tensorsByName, name),
    0,
  );
  const layers: LayerResourceRequirement[] = [];
  for (let layer = 0; layer < options.manifest.blockCount; layer += 1) {
    const layerKind = options.manifest.layerKinds[layer] ?? "sliding-attention";
    const weightBytes = layerWeightBytes(tensorsByName, options.manifest, layer);
    const cacheBytes = attentionCacheBytes(
      options.manifest,
      layer,
      options.contextLength,
      options.cacheElementByteLength,
    );
    const scratchBytes = options.scratchBytes ?? 0;
    const residentBytes = options.residentBytes?.({
      layer,
      layerKind,
      weightBytes,
      cacheBytes,
      scratchBytes,
    }) ?? 0;
    layers.push({
      provider: options.provider,
      layer,
      layerKind,
      weightBytes,
      cacheBytes,
      scratchBytes,
      residentBytes,
      estimatedComputeCost: options.estimatedComputeCost?.({
        layer,
        layerKind,
        weightBytes,
        cacheBytes,
        residentBytes,
      }) ?? 0,
      requiredSourceLayers: options.requiredSourceLayers?.({
        layer,
        layerKind,
        manifest: options.manifest,
      }) ?? [],
      supportedValueTransfers: ["hidden"],
    });
  }

  return {
    provider: options.provider,
    mode: options.mode ?? "enabled",
    support: options.support ?? { available: true },
    memoryLimitBytes: options.memoryLimitBytes ?? Number.POSITIVE_INFINITY,
    fixedBytes: options.fixedBytes ?? 0,
    outputBytes,
    scratchBytes: options.scratchBytes ?? 0,
    targetResourceConstrained: options.targetResourceConstrained ?? false,
    canRunFullModel: options.canRunFullModel ?? true,
    offReason: options.offReason ?? `${options.provider} execution is off.`,
    blockedReason: options.blockedReason ?? `${options.provider} resources do not fit the configured budget.`,
    plannedReason: options.plannedReason ?? `${options.provider} layer placement is planned.`,
    layers,
    copyExpectations: options.copyExpectations,
  };
}

function tensorBytes(tensorsByName: ReadonlyMap<string, GgufTensorInfo>, name: string): number {
  const tensor = tensorsByName.get(name);
  return tensor ? tensorByteLength(tensor) : 0;
}

function layerWeightBytes(
  tensorsByName: ReadonlyMap<string, GgufTensorInfo>,
  manifest: ModelManifest,
  layer: number,
): number {
  return manifest.expectedTensors.reduce((total, expected) => {
    if (expected.layer !== layer) {
      return total;
    }
    return total + tensorBytes(tensorsByName, expected.name);
  }, 0);
}

function attentionCacheBytes(
  manifest: ModelManifest,
  layer: number,
  contextLength: number,
  elementByteLength = Float32Array.BYTES_PER_ELEMENT,
): number {
  if (!manifest.layerHasKv[layer]) {
    return 0;
  }
  const keyLength = manifest.layerKeyLengths[layer] ?? manifest.keyLength;
  const valueLength = manifest.layerValueLengths[layer] ?? manifest.valueLength;
  const headCountKv = manifest.layerHeadCountKv[layer] ?? manifest.headCountKv;
  return contextLength * headCountKv * (keyLength + valueLength) * elementByteLength;
}
