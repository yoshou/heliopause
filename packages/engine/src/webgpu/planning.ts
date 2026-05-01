import type { GgufMetadata, GgufTensorInfo } from "../gguf";
import type { Qwen35ModelManifest } from "../qwen35";
import { tensorByteLength } from "../tensor-reader";
import { DEFAULT_GPU_FIXED_BYTES, DEFAULT_GPU_SCRATCH_BYTES, QWEN35_WEBGPU_MEMORY_LIMIT_BYTES } from "./gpu-constants";
import type { Qwen35WebGpuBrowserGate, Qwen35WebGpuCopyAuditObservation, Qwen35WebGpuCopyAuditResult, Qwen35WebGpuHybridPlan, Qwen35WebGpuLayerPlan, Qwen35WebGpuMode, Qwen35WebGpuPlanningOptions, Qwen35WebGpuPlanStatus } from "./gpu-types";

export function planQwen35WebGpuHybrid(
  gguf: GgufMetadata,
  manifest: Qwen35ModelManifest,
  options: Qwen35WebGpuPlanningOptions = {},
): Qwen35WebGpuHybridPlan {
  const mode = options.mode ?? "off";
  const memoryLimitBytes = options.memoryLimitBytes ?? QWEN35_WEBGPU_MEMORY_LIMIT_BYTES;
  const browserGate = options.browserGate ?? "required";
  const contextLength = Math.min(
    options.contextLength ?? manifest.contextLength,
    manifest.contextLength,
  );
  const support = options.support;
  const tensorsByName = new Map(gguf.tensors.map((tensor) => [tensor.name, tensor]));
  const outputBytes = tensorBytes(tensorsByName, "output.weight") +
    tensorBytes(tensorsByName, "output_norm.weight");
  const layerPlans = buildLayerPlans(tensorsByName, manifest, contextLength);

  if (mode === "off") {
    return emptyPlan({
      status: "off",
      mode,
      memoryLimitBytes,
      browserGate,
      reason: "WebGPU execution is off; this is a placement plan only.",
      outputBytes,
      fixedBytes: DEFAULT_GPU_FIXED_BYTES,
      scratchBytes: DEFAULT_GPU_SCRATCH_BYTES,
      blockCount: manifest.blockCount,
    });
  }

  if (support && !support.available) {
    return emptyPlan({
      status: "unavailable",
      mode,
      memoryLimitBytes,
      browserGate,
      reason: `WebGPU unavailable: ${support.reason}`,
      outputBytes,
      fixedBytes: DEFAULT_GPU_FIXED_BYTES,
      scratchBytes: DEFAULT_GPU_SCRATCH_BYTES,
      blockCount: manifest.blockCount,
    });
  }

  const fixedBytes = DEFAULT_GPU_FIXED_BYTES;
  const scratchBytes = DEFAULT_GPU_SCRATCH_BYTES;
  let selectedBytes = outputBytes + fixedBytes + scratchBytes;
  const selectedLayers: Qwen35WebGpuLayerPlan[] = [];

  if (selectedBytes > memoryLimitBytes) {
    return emptyPlan({
      status: "blocked",
      mode,
      memoryLimitBytes,
      browserGate,
      reason: "`output.weight` plus fixed GPU buffers exceed the configured WebGPU memory cap.",
      outputBytes,
      fixedBytes,
      scratchBytes,
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

  const segmentStartLayer = selectedLayers[0]?.layer;
  const gpuWeightBytes = outputBytes +
    selectedLayers.reduce((sum, layer) => sum + layer.weightBytes, 0);
  const gpuCacheBytes = selectedLayers.reduce((sum, layer) => sum + layer.cacheBytes, 0);
  const status: Qwen35WebGpuPlanStatus = browserGate === "passed" ? "planned" : "blocked";
  const reason = browserGate === "passed"
    ? "WebGPU segment placement is planned, but execution still requires verified kernels."
    : "Browser user check is required before WebGPU execution can be enabled.";

  return {
    status,
    mode,
    memoryLimitBytes,
    browserGate,
    enabled: false,
    reason,
    outputBytes,
    fixedBytes,
    scratchBytes,
    selectedLayerCount: selectedLayers.length,
    segmentStartLayer,
    cpuSegmentLayerCount: segmentStartLayer === undefined ? manifest.blockCount : segmentStartLayer,
    gpuSegmentLayerCount: selectedLayers.length,
    gpuWeightBytes,
    gpuCacheBytes,
    estimatedResidentBytes: selectedBytes,
    remainingBytes: Math.max(0, memoryLimitBytes - selectedBytes),
    selectedLayers,
    copyAuditExpectations: {
      decodeTensorReads: 0,
      segmentIntermediateReadbacks: 0,
      logitsReadbacks: 0,
      expectedBoundaryUploads: selectedLayers.length > 0 ? 1 : 0,
      expectedTokenReadbacks: 1,
    },
  };
}

export function auditQwen35WebGpuCopies(
  plan: Qwen35WebGpuHybridPlan,
  observed: Qwen35WebGpuCopyAuditObservation,
): Qwen35WebGpuCopyAuditResult {
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

  return {
    ok: errors.length === 0,
    errors,
    expected,
    observed,
  };
}

function buildLayerPlans(
  tensorsByName: Map<string, GgufTensorInfo>,
  manifest: Qwen35ModelManifest,
  contextLength: number,
): Map<number, Qwen35WebGpuLayerPlan> {
  const plans = new Map<number, Qwen35WebGpuLayerPlan>();
  for (let layer = 0; layer < manifest.blockCount; layer += 1) {
    const layerKind = manifest.fullAttentionLayers.includes(layer)
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
    plans.set(layer, {
      layer,
      layerKind,
      weightBytes,
      cacheBytes,
      totalBytes: weightBytes + cacheBytes,
    });
  }
  return plans;
}

function tensorBytes(tensorsByName: Map<string, GgufTensorInfo>, name: string): number {
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

function emptyPlan(params: {
  status: Qwen35WebGpuPlanStatus;
  mode: Qwen35WebGpuMode;
  memoryLimitBytes: number;
  browserGate: Qwen35WebGpuBrowserGate;
  reason: string;
  outputBytes: number;
  fixedBytes: number;
  scratchBytes: number;
  blockCount: number;
}): Qwen35WebGpuHybridPlan {
  const estimatedResidentBytes = params.outputBytes + params.fixedBytes + params.scratchBytes;
  return {
    status: params.status,
    mode: params.mode,
    memoryLimitBytes: params.memoryLimitBytes,
    browserGate: params.browserGate,
    enabled: false,
    reason: params.reason,
    outputBytes: params.outputBytes,
    fixedBytes: params.fixedBytes,
    scratchBytes: params.scratchBytes,
    selectedLayerCount: 0,
    cpuSegmentLayerCount: params.blockCount,
    gpuSegmentLayerCount: 0,
    gpuWeightBytes: params.outputBytes,
    gpuCacheBytes: 0,
    estimatedResidentBytes,
    remainingBytes: Math.max(0, params.memoryLimitBytes - estimatedResidentBytes),
    selectedLayers: [],
    copyAuditExpectations: {
      decodeTensorReads: 0,
      segmentIntermediateReadbacks: 0,
      logitsReadbacks: 0,
      expectedBoundaryUploads: 0,
      expectedTokenReadbacks: 1,
    },
  };
}
