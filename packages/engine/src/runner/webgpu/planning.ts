import type { GgufMetadata, GgufTensorInfo } from "../../gguf";
import type { Gemma4LayerKind, Gemma4ModelManifest } from "../../model";
import { tensorByteLength } from "../../tensor-reader";
import {
  planGemma4ProviderPlacement,
  type Gemma4RunnerPlacementPlan,
  type Gemma4RunnerPlanningOptions,
} from "../planning";
import {
  DEFAULT_GPU_FIXED_BYTES,
  DEFAULT_GPU_SCRATCH_BYTES,
  GEMMA4_WEBGPU_MEMORY_LIMIT_BYTES,
} from "./gpu-constants";

type WebGpuPlanningSupport =
  | { available: true }
  | { available: false; reason: string };

type WebGpuLayerPlacementParams = {
  tensorsByName: ReadonlyMap<string, GgufTensorInfo>;
  manifest: Gemma4ModelManifest;
  layer: number;
  contextLength: number;
};

type WebGpuCopyAuditExpectations = {
  decodeTensorReads: 0;
  segmentIntermediateReadbacks: 0;
  logitsReadbacks: 0;
  expectedBoundaryUploads: number;
  expectedTokenReadbacks: number;
};

export const gemma4WebGpuPlanningProvider = {
  name: "webgpu",
  defaultMemoryLimitBytes: GEMMA4_WEBGPU_MEMORY_LIMIT_BYTES,
  fixedBytes: DEFAULT_GPU_FIXED_BYTES,
  scratchBytes: DEFAULT_GPU_SCRATCH_BYTES,
  outputTensorNames: ["token_embd.weight", "output_norm.weight"],
  offReason: "WebGPU execution is off; this is a placement plan only.",
  blockedByMemoryReason: "WebGPU Gemma4 layer suffix plus required KV source layers exceed the configured WebGPU memory cap.",
  unavailableReason: (support: WebGpuPlanningSupport) => `WebGPU unavailable: ${support.available ? "unknown" : support.reason}`,
  plannedReason: "WebGPU Gemma4 layer suffix placement is planned.",
  layerPlacement: ({ tensorsByName, manifest, layer, contextLength }: WebGpuLayerPlacementParams) => {
    const layerKind: Gemma4LayerKind = manifest.layerKinds[layer] ?? "sliding-attention";
    const weightBytes = manifest.expectedTensors.reduce((sum, expected) => {
      if (expected.layer !== layer) {
        return sum;
      }
      return sum + tensorBytes(tensorsByName, expected.name);
    }, 0);
    const cacheBytes = manifest.layerHasKv[layer]
      ? attentionCacheBytes(manifest, layer, contextLength)
      : 0;
    return {
      layer,
      layerKind,
      weightBytes,
      cacheBytes,
      totalBytes: weightBytes + cacheBytes,
    };
  },
  copyAuditExpectations: (selectedLayerCount: number): WebGpuCopyAuditExpectations => ({
    decodeTensorReads: 0,
    segmentIntermediateReadbacks: 0,
    logitsReadbacks: 0,
    expectedBoundaryUploads: selectedLayerCount > 0 ? 1 : 0,
    expectedTokenReadbacks: 1,
  }),
  requiredSegmentStart: ({
    manifest,
    selectedLayers,
  }: {
    manifest: Gemma4ModelManifest;
    selectedLayers: readonly { layer: number }[];
  }): number | undefined => {
    if (selectedLayers.length === 0) {
      return undefined;
    }
    let requiredStart = selectedLayers[0]?.layer;
    for (const layer of selectedLayers) {
      const source = manifest.kvSourceLayers[layer.layer] ?? layer.layer;
      if (manifest.layerHasKv[layer.layer] !== true && requiredStart !== undefined) {
        requiredStart = Math.min(requiredStart, source);
      }
    }
    return requiredStart;
  },
};

export function planGemma4RunnerPlacement(
  gguf: GgufMetadata,
  manifest: Gemma4ModelManifest,
  options: Gemma4RunnerPlanningOptions = {},
): Gemma4RunnerPlacementPlan {
  return planGemma4ProviderPlacement(
    gemma4WebGpuPlanningProvider,
    gguf,
    manifest,
    options,
  );
}

function tensorBytes(tensorsByName: ReadonlyMap<string, GgufTensorInfo>, name: string): number {
  const tensor = tensorsByName.get(name);
  return tensor ? tensorByteLength(tensor) : 0;
}

function attentionCacheBytes(
  manifest: Gemma4ModelManifest,
  layer: number,
  contextLength: number,
): number {
  const keyLength = manifest.layerKeyLengths[layer] ?? manifest.keyLength;
  const valueLength = manifest.layerValueLengths[layer] ?? manifest.valueLength;
  return contextLength * manifest.headCountKv * (keyLength + valueLength) * Float32Array.BYTES_PER_ELEMENT;
}
