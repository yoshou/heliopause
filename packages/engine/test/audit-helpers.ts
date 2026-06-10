import {
  buildModelManifest,
  buildMtpAssistantManifest,
  type GgmlTypeName,
  type GgufMetadata,
  type GgufTensorInfo,
  type ModelManifest,
  type MtpAssistantManifest,
  type RunnerCopyExpectations,
  type RunnerPlacementPlan,
} from "../src/index.ts";

export type TensorCoverageAudit = {
  ok: boolean;
  errors: string[];
  warnings: string[];
  missing: string[];
  unknown: string[];
  shapeMismatches: string[];
  typeMismatches: string[];
  loadedButUnused: string[];
  wrongLayerUse: string[];
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
  expected: RunnerCopyExpectations;
  observed: RunnerCopyAuditObservation;
};

export function auditTensorCoverage(
  gguf: GgufMetadata,
  manifest: ModelManifest = buildModelManifest(gguf),
  usedTensorNames?: Iterable<string>,
): TensorCoverageAudit {
  const tensorsByName = tensorMap(gguf);
  const expectedByName = new Map(manifest.expectedTensors.map((tensor) => [tensor.name, tensor]));
  const usedSet = usedTensorNames ? new Set(usedTensorNames) : undefined;
  const missing: string[] = [];
  const unknown: string[] = [];
  const shapeMismatches: string[] = [];
  const typeMismatches: string[] = [];
  const loadedButUnused: string[] = [];
  const wrongLayerUse: string[] = [];
  const errors: string[] = [];
  const warnings: string[] = [];

  for (const expected of manifest.expectedTensors) {
    const actual = tensorsByName.get(expected.name);
    if (!actual) {
      missing.push(expected.name);
      continue;
    }

    if (!sameDimensions(actual.dimensions, expected.dimensions)) {
      shapeMismatches.push(
        `${expected.name}: expected [${expected.dimensions.join(", ")}], got [${actual.dimensions.join(", ")}]`,
      );
    }

    if (!expected.allowedTypes.includes(actual.type as GgmlTypeName)) {
      typeMismatches.push(
        `${expected.name}: expected ${expected.allowedTypes.join(" | ")}, got ${actual.type}`,
      );
    }
  }

  for (const actual of gguf.tensors) {
    if (!expectedByName.has(actual.name)) {
      if (isReferenceOptionalSharedKvTensor(actual.name, manifest)) {
        continue;
      }
      unknown.push(actual.name);
    }
  }

  if (usedSet) {
    for (const name of tensorsByName.keys()) {
      if (isReferenceOptionalSharedKvTensor(name, manifest)) {
        continue;
      }
      if (!usedSet.has(name)) {
        loadedButUnused.push(name);
      }
    }

    for (const name of usedSet) {
      const expected = expectedByName.get(name);
      if (!expected) {
        unknown.push(name);
        continue;
      }

      const layer = parseLayerFromTensorName(name);
      if (layer === undefined || expected.layerKind === undefined) {
        continue;
      }

      const actualLayerKind = manifest.layerKinds[layer] ?? "sliding-attention";

      if (actualLayerKind !== expected.layerKind) {
        wrongLayerUse.push(`${name}: expected ${expected.layerKind}, got ${actualLayerKind}`);
      }
    }
  }

  for (const item of missing) errors.push(`Missing tensor: ${item}`);
  for (const item of unknown) errors.push(`Unknown tensor: ${item}`);
  for (const item of shapeMismatches) errors.push(`Shape mismatch: ${item}`);
  for (const item of typeMismatches) errors.push(`Type mismatch: ${item}`);
  for (const item of loadedButUnused) errors.push(`Loaded but unused tensor: ${item}`);
  for (const item of wrongLayerUse) errors.push(`Wrong layer tensor use: ${item}`);

  return {
    ok: errors.length === 0,
    errors,
    warnings,
    missing,
    unknown,
    shapeMismatches,
    typeMismatches,
    loadedButUnused,
    wrongLayerUse,
  };
}

export function auditMtpAssistantTensorCoverage(
  gguf: GgufMetadata,
  manifest: MtpAssistantManifest = buildMtpAssistantManifest(gguf),
  usedTensorNames?: Iterable<string>,
): TensorCoverageAudit {
  const tensorsByName = tensorMap(gguf);
  const expectedByName = new Map(manifest.expectedTensors.map((tensor) => [tensor.name, tensor]));
  const usedSet = usedTensorNames ? new Set(usedTensorNames) : undefined;
  const missing: string[] = [];
  const unknown: string[] = [];
  const shapeMismatches: string[] = [];
  const typeMismatches: string[] = [];
  const loadedButUnused: string[] = [];
  const wrongLayerUse: string[] = [];
  const errors: string[] = [];
  const warnings: string[] = [];

  for (const expected of manifest.expectedTensors) {
    const actual = tensorsByName.get(expected.name);
    if (!actual) {
      missing.push(expected.name);
      continue;
    }
    if (!sameDimensions(actual.dimensions, expected.dimensions)) {
      shapeMismatches.push(
        `${expected.name}: expected [${expected.dimensions.join(", ")}], got [${actual.dimensions.join(", ")}]`,
      );
    }
    if (!expected.allowedTypes.includes(actual.type as GgmlTypeName)) {
      typeMismatches.push(
        `${expected.name}: expected ${expected.allowedTypes.join(" | ")}, got ${actual.type}`,
      );
    }
  }

  for (const actual of gguf.tensors) {
    if (!expectedByName.has(actual.name)) {
      unknown.push(actual.name);
    }
  }

  if (usedSet) {
    for (const name of tensorsByName.keys()) {
      if (!usedSet.has(name)) {
        loadedButUnused.push(name);
      }
    }
    for (const name of usedSet) {
      if (!expectedByName.has(name)) {
        unknown.push(name);
      }
    }
  }

  for (const item of missing) errors.push(`Missing tensor: ${item}`);
  for (const item of unknown) errors.push(`Unknown tensor: ${item}`);
  for (const item of shapeMismatches) errors.push(`Shape mismatch: ${item}`);
  for (const item of typeMismatches) errors.push(`Type mismatch: ${item}`);
  for (const item of loadedButUnused) errors.push(`Loaded but unused tensor: ${item}`);

  return {
    ok: errors.length === 0,
    errors,
    warnings,
    missing,
    unknown,
    shapeMismatches,
    typeMismatches,
    loadedButUnused,
    wrongLayerUse,
  };
}

export function auditRunnerPlacementCopies(
  plan: RunnerPlacementPlan,
  observed: RunnerCopyAuditObservation,
): RunnerCopyAuditResult {
  const expected = plan.copyExpectations;
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

function tensorMap(gguf: GgufMetadata): Map<string, GgufTensorInfo> {
  const tensorsByName = new Map<string, GgufTensorInfo>();
  for (const tensor of gguf.tensors) {
    if (tensorsByName.has(tensor.name)) {
      throw new Error(`Duplicate tensor in GGUF: ${tensor.name}`);
    }
    tensorsByName.set(tensor.name, tensor);
  }
  return tensorsByName;
}

function parseLayerFromTensorName(name: string): number | undefined {
  const match = /^blk\.(\d+)\./.exec(name);
  return match ? Number(match[1]) : undefined;
}

function isReferenceOptionalSharedKvTensor(name: string, manifest: ModelManifest): boolean {
  const match = /^blk\.(\d+)\.(attn_k|attn_v|attn_k_norm)\.weight$/.exec(name);
  if (!match) {
    return false;
  }
  const layer = Number(match[1]);
  return manifest.layerHasKv[layer] === false;
}

function sameDimensions(left: number[], right: number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
