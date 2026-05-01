import {
  getMetadataNumber,
  getMetadataNumberArray,
  getMetadataString,
} from "./gguf";
import type {
  GgmlTypeName,
  GgufMetadata,
  GgufTensorInfo,
} from "./gguf";

export type Qwen35LayerKind = "recurrent" | "full-attention";

export type Qwen35ModelManifest = {
  architecture: "qwen35";
  tensorCount: number;
  blockCount: number;
  embeddingLength: number;
  feedForwardLength: number;
  headCount: number;
  headCountKv: number;
  keyLength: number;
  valueLength: number;
  contextLength: number;
  fullAttentionInterval: number;
  recurrentLayerCount: number;
  fullAttentionLayerCount: number;
  recurrentLayers: number[];
  fullAttentionLayers: number[];
  rope: {
    dimensionCount: number;
    dimensionSections: number[];
    freqBase: number;
  };
  ssm: {
    convKernel: number;
    groupCount: number;
    innerSize: number;
    stateSize: number;
    timeStepRank: number;
  };
  tensorTypes: Record<string, number>;
  expectedTensors: ExpectedTensor[];
};

export type ExpectedTensor = {
  name: string;
  dimensions: number[];
  allowedTypes: GgmlTypeName[];
  layer?: number;
  layerKind?: Qwen35LayerKind;
};

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

const REQUIRED_ARCHITECTURE = "qwen35";

export function buildQwen35Manifest(gguf: GgufMetadata): Qwen35ModelManifest {
  const metadata = gguf.metadata;
  const architecture = requiredString(metadata, "general.architecture");

  if (architecture !== REQUIRED_ARCHITECTURE) {
    throw new Error(`Expected architecture ${REQUIRED_ARCHITECTURE}, got ${architecture}`);
  }

  const blockCount = requiredNumber(metadata, "qwen35.block_count");
  const embeddingLength = requiredNumber(metadata, "qwen35.embedding_length");
  const feedForwardLength = requiredNumber(metadata, "qwen35.feed_forward_length");
  const headCount = requiredNumber(metadata, "qwen35.attention.head_count");
  const headCountKv = requiredNumber(metadata, "qwen35.attention.head_count_kv");
  const keyLength = requiredNumber(metadata, "qwen35.attention.key_length");
  const valueLength = requiredNumber(metadata, "qwen35.attention.value_length");
  const contextLength = requiredNumber(metadata, "qwen35.context_length");
  const fullAttentionInterval = requiredNumber(metadata, "qwen35.full_attention_interval");
  const dimensionSections = requiredNumberArray(metadata, "qwen35.rope.dimension_sections");
  const ssm = {
    convKernel: requiredNumber(metadata, "qwen35.ssm.conv_kernel"),
    groupCount: requiredNumber(metadata, "qwen35.ssm.group_count"),
    innerSize: requiredNumber(metadata, "qwen35.ssm.inner_size"),
    stateSize: requiredNumber(metadata, "qwen35.ssm.state_size"),
    timeStepRank: requiredNumber(metadata, "qwen35.ssm.time_step_rank"),
  };

  const fullAttentionLayers = range(blockCount).filter(
    (layer) => (layer + 1) % fullAttentionInterval === 0,
  );
  const recurrentLayers = range(blockCount).filter(
    (layer) => !fullAttentionLayers.includes(layer),
  );

  const tensorTypes: Record<string, number> = {};
  const tensorsByName = new Map(gguf.tensors.map((tensor) => [tensor.name, tensor]));
  for (const tensor of gguf.tensors) {
    tensorTypes[tensor.type] = (tensorTypes[tensor.type] ?? 0) + 1;
  }

  const expectedTensors = buildExpectedQwen35Tensors({
    blockCount,
    embeddingLength,
    feedForwardLength,
    headCount,
    headCountKv,
    keyLength,
    valueLength,
    ssm,
    recurrentLayers,
    fullAttentionLayers,
    tensorsByName,
  });

  return {
    architecture: REQUIRED_ARCHITECTURE,
    tensorCount: gguf.tensorCount,
    blockCount,
    embeddingLength,
    feedForwardLength,
    headCount,
    headCountKv,
    keyLength,
    valueLength,
    contextLength,
    fullAttentionInterval,
    recurrentLayerCount: recurrentLayers.length,
    fullAttentionLayerCount: fullAttentionLayers.length,
    recurrentLayers,
    fullAttentionLayers,
    rope: {
      dimensionCount: requiredNumber(metadata, "qwen35.rope.dimension_count"),
      dimensionSections,
      freqBase: requiredNumber(metadata, "qwen35.rope.freq_base"),
    },
    ssm,
    tensorTypes,
    expectedTensors,
  };
}

export function auditQwen35TensorCoverage(
  gguf: GgufMetadata,
  manifest: Qwen35ModelManifest = buildQwen35Manifest(gguf),
  usedTensorNames?: Iterable<string>,
): TensorCoverageAudit {
  const tensorsByName = new Map<string, GgufTensorInfo>();
  for (const tensor of gguf.tensors) {
    if (tensorsByName.has(tensor.name)) {
      throw new Error(`Duplicate tensor in GGUF: ${tensor.name}`);
    }
    tensorsByName.set(tensor.name, tensor);
  }

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

    if (!expected.allowedTypes.includes(actual.type)) {
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
      const expected = expectedByName.get(name);
      if (!expected) {
        unknown.push(name);
        continue;
      }

      const layer = parseLayerFromTensorName(name);
      if (layer === undefined || expected.layerKind === undefined) {
        continue;
      }

      const actualLayerKind = manifest.fullAttentionLayers.includes(layer)
        ? "full-attention"
        : "recurrent";

      if (actualLayerKind !== expected.layerKind) {
        wrongLayerUse.push(`${name}: expected ${expected.layerKind}, got ${actualLayerKind}`);
      }
    }
  }

  for (const item of missing) {
    errors.push(`Missing tensor: ${item}`);
  }
  for (const item of unknown) {
    errors.push(`Unknown tensor: ${item}`);
  }
  for (const item of shapeMismatches) {
    errors.push(`Shape mismatch: ${item}`);
  }
  for (const item of typeMismatches) {
    errors.push(`Type mismatch: ${item}`);
  }
  for (const item of loadedButUnused) {
    errors.push(`Loaded but unused tensor: ${item}`);
  }
  for (const item of wrongLayerUse) {
    errors.push(`Wrong layer tensor use: ${item}`);
  }

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

function buildExpectedQwen35Tensors(params: {
  blockCount: number;
  embeddingLength: number;
  feedForwardLength: number;
  headCount: number;
  headCountKv: number;
  keyLength: number;
  valueLength: number;
  ssm: Qwen35ModelManifest["ssm"];
  recurrentLayers: number[];
  fullAttentionLayers: number[];
  tensorsByName: Map<string, GgufTensorInfo>;
}): ExpectedTensor[] {
  const {
    blockCount,
    embeddingLength,
    feedForwardLength,
    headCount,
    headCountKv,
    keyLength,
    ssm,
    recurrentLayers,
    fullAttentionLayers,
    tensorsByName,
  } = params;
  const vocabSize = tensorsByName.get("token_embd.weight")?.dimensions[1] ?? 248320;
  const valueDim = ssm.stateSize * ssm.timeStepRank;
  const keyDim = ssm.stateSize * ssm.groupCount;
  const convDim = keyDim * 2 + valueDim;
  const fullQueryDim = keyLength * headCount * 2;
  const fullKeyValueDim = keyLength * headCountKv;
  const expected: ExpectedTensor[] = [
    {
      name: "output.weight",
      dimensions: [embeddingLength, vocabSize],
      allowedTypes: observedType(tensorsByName, "output.weight", ["Q6_K"]),
    },
    {
      name: "output_norm.weight",
      dimensions: [embeddingLength],
      allowedTypes: observedType(tensorsByName, "output_norm.weight", ["F32"]),
    },
    {
      name: "token_embd.weight",
      dimensions: [embeddingLength, vocabSize],
      allowedTypes: observedType(tensorsByName, "token_embd.weight", ["Q4_K"]),
    },
  ];

  for (const layer of range(blockCount)) {
    expected.push(
      layerTensor(layer, "attn_norm.weight", [embeddingLength], ["F32"]),
      layerTensor(layer, "post_attention_norm.weight", [embeddingLength], ["F32"]),
      layerTensor(layer, "ffn_gate.weight", [embeddingLength, feedForwardLength], observedType(tensorsByName, `blk.${layer}.ffn_gate.weight`, ["Q4_K", "Q5_K", "IQ4_XS"])),
      layerTensor(layer, "ffn_up.weight", [embeddingLength, feedForwardLength], observedType(tensorsByName, `blk.${layer}.ffn_up.weight`, ["Q4_K", "Q5_K", "IQ4_XS"])),
      layerTensor(layer, "ffn_down.weight", [feedForwardLength, embeddingLength], observedType(tensorsByName, `blk.${layer}.ffn_down.weight`, ["Q4_K", "Q5_K", "Q6_K"])),
    );
  }

  for (const layer of recurrentLayers) {
    expected.push(
      layerTensor(layer, "attn_qkv.weight", [embeddingLength, convDim], observedType(tensorsByName, `blk.${layer}.attn_qkv.weight`, ["Q4_K", "Q6_K"]), "recurrent"),
      layerTensor(layer, "attn_gate.weight", [embeddingLength, valueDim], observedType(tensorsByName, `blk.${layer}.attn_gate.weight`, ["Q5_K"]), "recurrent"),
      layerTensor(layer, "ssm_a", [ssm.timeStepRank], ["F32"], "recurrent"),
      layerTensor(layer, "ssm_alpha.weight", [embeddingLength, ssm.timeStepRank], ["F32"], "recurrent"),
      layerTensor(layer, "ssm_beta.weight", [embeddingLength, ssm.timeStepRank], ["F32"], "recurrent"),
      layerTensor(layer, "ssm_conv1d.weight", [ssm.convKernel, convDim], ["F32"], "recurrent"),
      layerTensor(layer, "ssm_dt.bias", [ssm.timeStepRank], ["F32"], "recurrent"),
      layerTensor(layer, "ssm_norm.weight", [ssm.stateSize], ["F32"], "recurrent"),
      layerTensor(layer, "ssm_out.weight", [valueDim, embeddingLength], observedType(tensorsByName, `blk.${layer}.ssm_out.weight`, ["Q8_0", "IQ4_XS"]), "recurrent"),
    );
  }

  for (const layer of fullAttentionLayers) {
    expected.push(
      layerTensor(layer, "attn_q.weight", [embeddingLength, fullQueryDim], observedType(tensorsByName, `blk.${layer}.attn_q.weight`, ["Q4_K", "Q5_K", "IQ4_XS"]), "full-attention"),
      layerTensor(layer, "attn_k.weight", [embeddingLength, fullKeyValueDim], observedType(tensorsByName, `blk.${layer}.attn_k.weight`, ["Q4_K", "Q5_K", "IQ4_XS"]), "full-attention"),
      layerTensor(layer, "attn_v.weight", [embeddingLength, fullKeyValueDim], observedType(tensorsByName, `blk.${layer}.attn_v.weight`, ["Q5_K", "Q6_K"]), "full-attention"),
      layerTensor(layer, "attn_output.weight", [keyLength * headCount, embeddingLength], observedType(tensorsByName, `blk.${layer}.attn_output.weight`, ["Q4_K"]), "full-attention"),
      layerTensor(layer, "attn_q_norm.weight", [keyLength], ["F32"], "full-attention"),
      layerTensor(layer, "attn_k_norm.weight", [keyLength], ["F32"], "full-attention"),
    );
  }

  return expected;
}

function observedType(
  tensorsByName: Map<string, GgufTensorInfo>,
  name: string,
  fallback: GgmlTypeName[],
): GgmlTypeName[] {
  return tensorsByName.has(name) ? [tensorsByName.get(name)!.type] : fallback;
}

function layerTensor(
  layer: number,
  suffix: string,
  dimensions: number[],
  allowedTypes: GgmlTypeName[],
  layerKind?: Qwen35LayerKind,
): ExpectedTensor {
  return {
    name: `blk.${layer}.${suffix}`,
    dimensions,
    allowedTypes,
    layer,
    layerKind,
  };
}

function parseLayerFromTensorName(name: string): number | undefined {
  const match = /^blk\.(\d+)\./.exec(name);
  return match ? Number(match[1]) : undefined;
}

function requiredNumber(metadata: GgufMetadata["metadata"], key: string): number {
  const value = getMetadataNumber(metadata, key);
  if (value === undefined) {
    throw new Error(`Missing numeric GGUF metadata: ${key}`);
  }
  return value;
}

function requiredString(metadata: GgufMetadata["metadata"], key: string): string {
  const value = getMetadataString(metadata, key);
  if (value === undefined) {
    throw new Error(`Missing string GGUF metadata: ${key}`);
  }
  return value;
}

function requiredNumberArray(metadata: GgufMetadata["metadata"], key: string): number[] {
  const value = getMetadataNumberArray(metadata, key);
  if (value === undefined) {
    throw new Error(`Missing numeric array GGUF metadata: ${key}`);
  }
  return value;
}

function sameDimensions(left: number[], right: number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function range(count: number): number[] {
  return Array.from({ length: count }, (_, index) => index);
}
