import type { GgufTensorInfo } from "../../gguf";
import type { Qwen35LayerKind, Qwen35ModelManifest } from "../../model";
import { tensorByteLength } from "../../tensor-reader";
import {
  DEFAULT_GPU_FIXED_BYTES,
  DEFAULT_GPU_SCRATCH_BYTES,
  QWEN35_WEBGPU_MEMORY_LIMIT_BYTES,
} from "./gpu-constants";

type WebGpuPlanningSupport =
  | { available: true }
  | { available: false; reason: string };

type WebGpuPlanningBrowserGate = "required" | "passed";

type WebGpuLayerPlacementParams = {
  tensorsByName: ReadonlyMap<string, GgufTensorInfo>;
  manifest: Qwen35ModelManifest;
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

export const qwen35WebGpuPlanningProvider = {
  name: "webgpu",
  defaultMemoryLimitBytes: QWEN35_WEBGPU_MEMORY_LIMIT_BYTES,
  fixedBytes: DEFAULT_GPU_FIXED_BYTES,
  scratchBytes: DEFAULT_GPU_SCRATCH_BYTES,
  outputTensorNames: ["output.weight", "output_norm.weight"],
  offReason: "WebGPU execution is off; this is a placement plan only.",
  blockedByMemoryReason: "`output.weight` plus fixed GPU buffers exceed the configured WebGPU memory cap.",
  unavailableReason: (support: WebGpuPlanningSupport) => `WebGPU unavailable: ${support.available ? "unknown" : support.reason}`,
  plannedReason: (browserGate: WebGpuPlanningBrowserGate) => browserGate === "passed"
    ? "WebGPU segment placement is planned, but execution still requires verified kernels."
    : "Browser user check is required before WebGPU execution can be enabled.",
  layerPlacement: ({ tensorsByName, manifest, layer, contextLength }: WebGpuLayerPlacementParams) => {
    const layerKind: Qwen35LayerKind = manifest.fullAttentionLayers.includes(layer)
      ? "full-attention"
      : "recurrent";
    const weightBytes = manifest.expectedTensors.reduce((sum, expected) => {
      if (expected.layer !== layer) {
        return sum;
      }
      return sum + tensorBytes(tensorsByName, expected.name);
    }, 0);
    const cacheBytes = layerKind === "full-attention"
      ? fullAttentionCacheBytes(manifest, contextLength)
      : recurrentCacheBytes(manifest);
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
};

function tensorBytes(tensorsByName: ReadonlyMap<string, GgufTensorInfo>, name: string): number {
  const tensor = tensorsByName.get(name);
  return tensor ? tensorByteLength(tensor) : 0;
}

function recurrentCacheBytes(manifest: Qwen35ModelManifest): number {
  const convDim =
    manifest.ssm.stateSize * manifest.ssm.groupCount * 2 +
    manifest.ssm.stateSize * manifest.ssm.timeStepRank;
  const recurrentStateSize =
    manifest.ssm.stateSize * manifest.ssm.stateSize * manifest.ssm.timeStepRank;
  return (manifest.ssm.convKernel - 1) * convDim * Float32Array.BYTES_PER_ELEMENT +
    recurrentStateSize * Float32Array.BYTES_PER_ELEMENT;
}

function fullAttentionCacheBytes(
  manifest: Qwen35ModelManifest,
  contextLength: number,
): number {
  const perCache = contextLength * manifest.headCountKv * manifest.keyLength;
  return perCache * 2 * Float32Array.BYTES_PER_ELEMENT;
}
