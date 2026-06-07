import type { GgufMetadata } from "../../gguf";
import type { ModelManifest } from "../../model";
import {
  createModelResourceRequirements,
} from "../model-resources";
import type {
  ProviderResourceRequirements,
  ResourceBudget,
  RunnerCopyExpectations,
  RunnerExecutionMode,
  RunnerProviderSupport,
} from "../planning";
import {
  planModelPlacement,
} from "../planning";
import {
  DEFAULT_GPU_FIXED_BYTES,
  DEFAULT_GPU_SCRATCH_BYTES,
  WEBGPU_MEMORY_LIMIT_BYTES,
} from "./gpu-constants";

export type WebGpuResourceOptions = {
  mode?: RunnerExecutionMode;
  memoryLimitBytes?: number;
  contextLength?: number;
  slidingWindowReserveTokens?: number;
  support?: RunnerProviderSupport;
};

export function webGpuResourceRequirements(
  gguf: GgufMetadata,
  manifest: ModelManifest,
  options: WebGpuResourceOptions = {},
): ProviderResourceRequirements {
  const contextLength = Math.min(
    options.contextLength ?? manifest.contextLength,
    manifest.contextLength,
  );
  return createModelResourceRequirements({
    provider: "webgpu",
    gguf,
    manifest,
    contextLength,
    mode: options.mode ?? "enabled",
    support: options.support,
    memoryLimitBytes: options.memoryLimitBytes ?? WEBGPU_MEMORY_LIMIT_BYTES,
    fixedBytes: DEFAULT_GPU_FIXED_BYTES,
    scratchBytes: DEFAULT_GPU_SCRATCH_BYTES,
    cacheElementByteLength: 2,
    slidingWindowReserveTokens: options.slidingWindowReserveTokens,
    outputTensorNames: ["token_embd.weight", "output_norm.weight"],
    targetResourceConstrained: true,
    canRunFullModel: false,
    offReason: "WebGPU execution is off; this is a placement plan only.",
    blockedReason: "No WebGPU layer placement fits the configured memory budget.",
    plannedReason: "Cost-based layer placement is planned.",
    residentBytes: ({ weightBytes, cacheBytes }) => weightBytes + cacheBytes,
    estimatedComputeCost: ({ layerKind, weightBytes, cacheBytes }) => {
      const attentionFactor = layerKind === "full-attention" ? 2 : 1;
      return (weightBytes + cacheBytes) * attentionFactor;
    },
    requiredSourceLayers: ({ layer, manifest: modelManifest }) => {
      const source = modelManifest.kvSourceLayers[layer] ?? layer;
      return source < layer ? [source] : [];
    },
    copyExpectations: webGpuCopyExpectations,
  });
}

export function planWebGpuResourcePlacement(
  gguf: GgufMetadata,
  manifest: ModelManifest,
  options: WebGpuResourceOptions = {},
  budget: ResourceBudget = {},
) {
  const requirements = webGpuResourceRequirements(gguf, manifest, options);
  return planModelPlacement([requirements], {
    mode: options.mode,
    memoryLimitBytes: options.memoryLimitBytes,
    ...budget,
  });
}

function webGpuCopyExpectations({
  selectedLayers,
  nodes,
}: Parameters<NonNullable<ProviderResourceRequirements["copyExpectations"]>>[0]): RunnerCopyExpectations {
  return {
    decodeTensorReads: 0,
    segmentIntermediateReadbacks: 0,
    logitsReadbacks: 0,
    expectedBoundaryUploads: nodes.some((node) =>
      node.kind === "transfer" && node.to === "webgpu"
    ) ? 1 : 0,
    expectedTokenReadbacks: 0,
    expectedSelectedTokenReadbacks: selectedLayers.length > 0 ? 1 : 0,
  };
}
