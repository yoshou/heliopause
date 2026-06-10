import {
  getMetadataNumber,
  getMetadataNumberArray,
  getMetadataString,
} from "./gguf";
import type {
  GgmlTypeName,
  GgufMetadata,
  GgufMetadataValue,
  GgufTensorInfo,
} from "./gguf";

export type LayerKind = "sliding-attention" | "full-attention";
export type LayerValueProjectionMode = "separate" | "shared-with-key";

export type ModelManifest = {
  architecture: "gemma4";
  tensorCount: number;
  blockCount: number;
  embeddingLength: number;
  feedForwardLength: number;
  headCount: number;
  headCountKv: number;
  layerHeadCountKv: number[];
  keyLength: number;
  valueLength: number;
  slidingKeyLength: number;
  slidingValueLength: number;
  layerKeyLengths: number[];
  layerValueLengths: number[];
  layerHasKv: boolean[];
  layerValueProjectionModes: LayerValueProjectionMode[];
  kvSourceLayers: number[];
  contextLength: number;
  slidingWindow: number;
  layerKinds: LayerKind[];
  slidingAttentionLayerCount: number;
  fullAttentionLayerCount: number;
  slidingAttentionLayers: number[];
  fullAttentionLayers: number[];
  perLayerEmbeddingLength: number;
  rope: {
    slidingDimensionCount: number;
    fullDimensionCount: number;
    slidingFreqBase: number;
    fullFreqBase: number;
    dimensionCount: number;
    dimensionSections: number[];
    freqBase: number;
  };
  finalLogitSoftcap?: number;
  tensorTypes: Record<string, number>;
  expectedTensors: ExpectedTensor[];
};

export type ExpectedTensor = {
  name: string;
  dimensions: number[];
  allowedTypes: GgmlTypeName[];
  layer?: number;
  layerKind?: LayerKind;
};

export type VisionManifest = {
  architecture: "clip";
  projectorType: "gemma4";
  tensorCount: number;
  imageSize: number;
  patchSize: number;
  embeddingLength: number;
  feedForwardLength: number;
  blockCount: number;
  headCount: number;
  layerNormEpsilon: number;
  projectionDim: number;
  spatialMergeSize: number;
  imageMinTokens: number;
  imageMaxTokens: number;
  imageMean: [number, number, number];
  imageStd: [number, number, number];
  tensorTypes: Record<string, number>;
};

export type AudioManifest = {
  architecture: "clip";
  projectorType: "gemma4";
  tensorCount: number;
  embeddingLength: number;
  feedForwardLength: number;
  blockCount: number;
  headCount: number;
  headSize: number;
  layerNormEpsilon: number;
  numMelBins: number;
  projectionDim: number;
  outputProjectionDim: number;
  convKernelSize: number;
  residualWeight: number;
  attentionChunkSize: number;
  attentionContextLeft: number;
  attentionContextRight: number;
  attentionLogitCap: number;
  attentionInvalidLogitsValue: number;
  audioSeqLength: number;
  audioMsPerToken: number;
  sampleRate: number;
  featureSize: number;
  fftLength: number;
  frameLength: number;
  hopLength: number;
  melFloor: number;
  maxSeconds: number;
  tensorTypes: Record<string, number>;
};

export type MtpAssistantManifest = {
  architecture: "gemma4_assistant";
  tensorCount: number;
  blockCount: number;
  embeddingLength: number;
  backboneEmbeddingLength: number;
  feedForwardLength: number;
  headCount: number;
  headCountKv: number;
  keyLength: number;
  valueLength: number;
  slidingKeyLength: number;
  slidingValueLength: number;
  layerKeyLengths: number[];
  layerValueLengths: number[];
  contextLength: number;
  slidingWindow: number;
  layerKinds: LayerKind[];
  nCentroids: number;
  centroidTopK: number;
  useOrderedEmbeddings: boolean;
  layerNormEpsilon: number;
  rope: {
    slidingDimensionCount: number;
    fullDimensionCount: number;
    slidingFreqBase: number;
    fullFreqBase: number;
  };
  tensorTypes: Record<string, number>;
  expectedTensors: ExpectedTensor[];
};

const REQUIRED_ARCHITECTURE = "gemma4";
const VISION_DEFAULT_IMAGE_MIN_TOKENS = 252;
const VISION_DEFAULT_IMAGE_MAX_TOKENS = 280;

export function isVisionGguf(gguf: GgufMetadata): boolean {
  return gguf.metadata["general.architecture"] === "clip" &&
    gguf.metadata["clip.has_vision_encoder"] === true &&
    normalizeVisionProjector(getMetadataString(gguf.metadata, "clip.vision.projector_type")) === "gemma4";
}

export function isAudioGguf(gguf: GgufMetadata): boolean {
  return gguf.metadata["general.architecture"] === "clip" &&
    gguf.metadata["clip.has_audio_encoder"] === true &&
    normalizeAudioProjector(getMetadataString(gguf.metadata, "clip.audio.projector_type")) === "gemma4";
}

export function isMtpAssistantGguf(gguf: GgufMetadata): boolean {
  return gguf.metadata["general.architecture"] === "gemma4_assistant";
}

export function buildVisionManifest(gguf: GgufMetadata): VisionManifest {
  const metadata = gguf.metadata;
  const architecture = requiredString(metadata, "general.architecture");
  if (architecture !== "clip") {
    throw new Error(`Expected architecture clip, got ${architecture}`);
  }

  const rawProjector = requiredString(metadata, "clip.vision.projector_type");
  const projectorType = normalizeVisionProjector(rawProjector);
  if (projectorType !== "gemma4") {
    throw new Error(`Unsupported vision projector type: ${rawProjector}`);
  }

  const tensorTypes: Record<string, number> = {};
  for (const tensor of gguf.tensors) {
    tensorTypes[tensor.type] = (tensorTypes[tensor.type] ?? 0) + 1;
  }

  return {
    architecture: "clip",
    projectorType,
    tensorCount: gguf.tensorCount,
    imageSize: requiredNumber(metadata, "clip.vision.image_size"),
    patchSize: requiredNumber(metadata, "clip.vision.patch_size"),
    embeddingLength: requiredNumber(metadata, "clip.vision.embedding_length"),
    feedForwardLength: requiredNumber(metadata, "clip.vision.feed_forward_length"),
    blockCount: requiredNumber(metadata, "clip.vision.block_count"),
    headCount: requiredNumber(metadata, "clip.vision.attention.head_count"),
    layerNormEpsilon: requiredNumber(metadata, "clip.vision.attention.layer_norm_epsilon"),
    projectionDim: requiredNumber(metadata, "clip.vision.projection_dim"),
    spatialMergeSize: getMetadataNumber(metadata, "clip.vision.projector.scale_factor") ?? 3,
    imageMinTokens: VISION_DEFAULT_IMAGE_MIN_TOKENS,
    imageMaxTokens: VISION_DEFAULT_IMAGE_MAX_TOKENS,
    imageMean: requiredNumberTuple3(metadata, "clip.vision.image_mean"),
    imageStd: requiredNumberTuple3(metadata, "clip.vision.image_std"),
    tensorTypes,
  };
}

export function buildAudioManifest(gguf: GgufMetadata): AudioManifest {
  const metadata = gguf.metadata;
  const architecture = requiredString(metadata, "general.architecture");
  if (architecture !== "clip") {
    throw new Error(`Expected architecture clip, got ${architecture}`);
  }

  const rawProjector = requiredString(metadata, "clip.audio.projector_type");
  const projectorType = normalizeAudioProjector(rawProjector);
  if (projectorType !== "gemma4") {
    throw new Error(`Unsupported audio projector type: ${rawProjector}`);
  }

  const tensorTypes: Record<string, number> = {};
  for (const tensor of gguf.tensors) {
    tensorTypes[tensor.type] = (tensorTypes[tensor.type] ?? 0) + 1;
  }

  const embeddingLength = requiredNumber(metadata, "clip.audio.embedding_length");
  const headCount = requiredNumber(metadata, "clip.audio.attention.head_count");
  return {
    architecture: "clip",
    projectorType,
    tensorCount: gguf.tensorCount,
    embeddingLength,
    feedForwardLength: requiredNumber(metadata, "clip.audio.feed_forward_length"),
    blockCount: requiredNumber(metadata, "clip.audio.block_count"),
    headCount,
    headSize: embeddingLength / headCount,
    layerNormEpsilon: 1e-6,
    numMelBins: requiredNumber(metadata, "clip.audio.num_mel_bins"),
    projectionDim: requiredNumber(metadata, "clip.audio.projection_dim"),
    outputProjectionDim: gguf.tensors.find((tensor) => tensor.name === "a.pre_encode.out.weight")?.dimensions[1] ?? 1536,
    convKernelSize: 5,
    residualWeight: 0.5,
    attentionChunkSize: 12,
    attentionContextLeft: 13,
    attentionContextRight: 0,
    attentionLogitCap: 50,
    attentionInvalidLogitsValue: -1e9,
    audioSeqLength: 750,
    audioMsPerToken: 40,
    sampleRate: 16000,
    featureSize: 128,
    fftLength: 512,
    frameLength: 320,
    hopLength: 160,
    melFloor: 0.001,
    maxSeconds: 30,
    tensorTypes,
  };
}

export function buildMtpAssistantManifest(gguf: GgufMetadata): MtpAssistantManifest {
  const metadata = gguf.metadata;
  const architecture = requiredString(metadata, "general.architecture");
  if (architecture !== "gemma4_assistant") {
    throw new Error(`Expected architecture gemma4_assistant, got ${architecture}`);
  }

  const blockCount = requiredNumber(metadata, "gemma4_assistant.block_count");
  const embeddingLength = requiredNumber(metadata, "gemma4_assistant.embedding_length");
  const backboneEmbeddingLength = requiredNumber(metadata, "gemma4_assistant.n_embd_backbone");
  const feedForwardLength = requiredNumber(metadata, "gemma4_assistant.feed_forward_length");
  const headCount = requiredNumber(metadata, "gemma4_assistant.attention.head_count");
  const headCountKv = requiredNumber(metadata, "gemma4_assistant.attention.head_count_kv");
  const keyLength = requiredNumber(metadata, "gemma4_assistant.attention.key_length");
  const valueLength = requiredNumber(metadata, "gemma4_assistant.attention.value_length");
  const slidingKeyLength = firstNumber(metadata, ["gemma4_assistant.attention.key_length_swa"], keyLength);
  const slidingValueLength = firstNumber(metadata, ["gemma4_assistant.attention.value_length_swa"], valueLength);
  const contextLength = requiredNumber(metadata, "gemma4_assistant.context_length");
  const slidingWindow = firstNumber(metadata, ["gemma4_assistant.attention.sliding_window"], contextLength);
  const slidingPattern = boolArray(metadata, "gemma4_assistant.attention.sliding_window_pattern", blockCount) ??
    range(blockCount).map((layer) => (layer + 1) % 6 !== 0);
  const layerKinds: LayerKind[] = slidingPattern.map((isSliding) => isSliding ? "sliding-attention" : "full-attention");
  const layerKeyLengths = layerKinds.map((kind) => kind === "sliding-attention" ? slidingKeyLength : keyLength);
  const layerValueLengths = layerKinds.map((kind) => kind === "sliding-attention" ? slidingValueLength : valueLength);
  const tensorTypes: Record<string, number> = {};
  const tensorsByName = new Map(gguf.tensors.map((tensor) => [tensor.name, tensor]));
  for (const tensor of gguf.tensors) {
    tensorTypes[tensor.type] = (tensorTypes[tensor.type] ?? 0) + 1;
  }
  const nCentroids = requiredNumber(metadata, "gemma4_assistant.n_centroids");

  return {
    architecture: "gemma4_assistant",
    tensorCount: gguf.tensorCount,
    blockCount,
    embeddingLength,
    backboneEmbeddingLength,
    feedForwardLength,
    headCount,
    headCountKv,
    keyLength,
    valueLength,
    slidingKeyLength,
    slidingValueLength,
    layerKeyLengths,
    layerValueLengths,
    contextLength,
    slidingWindow,
    layerKinds,
    nCentroids,
    centroidTopK: requiredNumber(metadata, "gemma4_assistant.centroid_top_k"),
    useOrderedEmbeddings: metadata["gemma4_assistant.use_ordered_embeddings"] === true,
    layerNormEpsilon: requiredNumber(metadata, "gemma4_assistant.attention.layer_norm_rms_epsilon"),
    rope: {
      slidingDimensionCount: firstNumber(metadata, ["gemma4_assistant.rope.dimension_count_swa"], slidingKeyLength),
      fullDimensionCount: firstNumber(metadata, ["gemma4_assistant.rope.dimension_count"], keyLength),
      slidingFreqBase: firstNumber(metadata, ["gemma4_assistant.rope.freq_base_swa"], 10000),
      fullFreqBase: firstNumber(metadata, ["gemma4_assistant.rope.freq_base"], 1000000),
    },
    tensorTypes,
    expectedTensors: buildMtpAssistantExpectedTensors({
      blockCount,
      embeddingLength,
      backboneEmbeddingLength,
      feedForwardLength,
      headCount,
      layerKeyLengths,
      layerValueLengths,
      nCentroids,
      tensorsByName,
    }),
  };
}

function normalizeVisionProjector(value: string | undefined): "gemma4" | string | undefined {
  return value === "gemma4v" ? "gemma4" : value;
}

function normalizeAudioProjector(value: string | undefined): "gemma4" | string | undefined {
  return value === "gemma4a" ? "gemma4" : value;
}

export function buildModelManifest(gguf: GgufMetadata): ModelManifest {
  const metadata = gguf.metadata;
  const architecture = requiredString(metadata, "general.architecture");

  if (architecture !== REQUIRED_ARCHITECTURE) {
    throw new Error(`Expected architecture ${REQUIRED_ARCHITECTURE}, got ${architecture}`);
  }

  const blockCount = requiredNumber(metadata, "gemma4.block_count");
  const embeddingLength = requiredNumber(metadata, "gemma4.embedding_length");
  const feedForwardLength = requiredNumber(metadata, "gemma4.feed_forward_length");
  const headCount = requiredNumber(metadata, "gemma4.attention.head_count");
  const layerHeadCountKv = readLayerHeadCountKv(metadata, blockCount);
  const headCountKv = layerHeadCountKv[0] ?? 0;
  const keyLength = requiredNumber(metadata, "gemma4.attention.key_length");
  const valueLength = requiredNumber(metadata, "gemma4.attention.value_length");
  const slidingKeyLength = firstNumber(metadata, ["gemma4.attention.key_length_swa"], keyLength);
  const slidingValueLength = firstNumber(metadata, ["gemma4.attention.value_length_swa"], valueLength);
  const contextLength = requiredNumber(metadata, "gemma4.context_length");
  const slidingWindow = firstNumber(metadata, ["gemma4.attention.sliding_window"], contextLength);
  const perLayerEmbeddingLength = getMetadataNumber(metadata, "gemma4.embedding_length_per_layer_input") ?? 0;
  const slidingPattern = boolArray(metadata, "gemma4.attention.sliding_window_pattern", blockCount) ??
    range(blockCount).map((layer) => (layer + 1) % 6 !== 0);
  const layerKinds: LayerKind[] = slidingPattern.map((isSliding) => isSliding ? "sliding-attention" : "full-attention");
  const slidingAttentionLayers = range(blockCount).filter((layer) => layerKinds[layer] === "sliding-attention");
  const fullAttentionLayers = range(blockCount).filter((layer) => layerKinds[layer] === "full-attention");
  const sharedKvLayers = getMetadataNumber(metadata, "gemma4.attention.shared_kv_layers") ?? 0;
  const layerKvFromStart = sharedKvLayers > 0 ? blockCount - sharedKvLayers : blockCount;
  const hasExplicitLayerKvTensors = gguf.tensors.some((tensor) =>
    /^blk\.\d+\.attn_[kv]\.weight$/.test(tensor.name),
  );
  const layerHasKv = range(blockCount).map((layer) => layer < layerKvFromStart && (
    !hasExplicitLayerKvTensors ||
    gguf.tensors.some((tensor) => tensor.name === `blk.${layer}.attn_k.weight`)
  ));
  const layerValueProjectionModes: LayerValueProjectionMode[] = range(blockCount).map((layer) =>
    layerHasKv[layer] && !gguf.tensors.some((tensor) => tensor.name === `blk.${layer}.attn_v.weight`)
      ? "shared-with-key"
      : "separate"
  );
  const kvSourceLayers = range(blockCount).map((layer) => {
    if (layerHasKv[layer]) {
      return layer;
    }
    const offset = layerKinds[layer] === "sliding-attention" ? 2 : 1;
    return Math.max(0, layerKvFromStart - offset);
  });
  const layerKeyLengths = layerKinds.map((kind) => kind === "sliding-attention" ? slidingKeyLength : keyLength);
  const layerValueLengths = layerKinds.map((kind) => kind === "sliding-attention" ? slidingValueLength : valueLength);

  const tensorTypes: Record<string, number> = {};
  const tensorsByName = new Map(gguf.tensors.map((tensor) => [tensor.name, tensor]));
  for (const tensor of gguf.tensors) {
    tensorTypes[tensor.type] = (tensorTypes[tensor.type] ?? 0) + 1;
  }

  const expectedTensors = buildExpectedTensors({
    blockCount,
    embeddingLength,
    feedForwardLength,
    headCount,
    headCountKv,
    layerHeadCountKv,
    keyLength,
    valueLength,
    layerKeyLengths,
    layerValueLengths,
    layerHasKv,
    layerValueProjectionModes,
    perLayerEmbeddingLength,
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
    layerHeadCountKv,
    keyLength,
    valueLength,
    slidingKeyLength,
    slidingValueLength,
    layerKeyLengths,
    layerValueLengths,
    layerHasKv,
    layerValueProjectionModes,
    kvSourceLayers,
    contextLength,
    slidingWindow,
    layerKinds,
    slidingAttentionLayerCount: slidingAttentionLayers.length,
    fullAttentionLayerCount: fullAttentionLayers.length,
    slidingAttentionLayers,
    fullAttentionLayers,
    perLayerEmbeddingLength,
    rope: {
      slidingDimensionCount: firstNumber(metadata, ["gemma4.rope.dimension_count_swa"], slidingKeyLength),
      fullDimensionCount: firstNumber(metadata, ["gemma4.rope.dimension_count"], keyLength),
      slidingFreqBase: firstNumber(metadata, ["gemma4.rope.freq_base_swa"], 10000),
      fullFreqBase: firstNumber(metadata, ["gemma4.rope.freq_base"], 1000000),
      dimensionCount: firstNumber(metadata, ["gemma4.rope.dimension_count_swa"], slidingKeyLength),
      dimensionSections: requiredNumberArray(metadata, "gemma4.rope.dimension_sections") ?? [slidingKeyLength / 2, slidingKeyLength / 2, 0, 0],
      freqBase: firstNumber(metadata, ["gemma4.rope.freq_base_swa"], 10000),
    },
    finalLogitSoftcap: getMetadataNumber(metadata, "gemma4.final_logit_softcapping"),
    tensorTypes,
    expectedTensors,
  };
}

export function mapMtpAssistantLayerToTargetKvLayer(
  targetManifest: ModelManifest,
  assistantManifest: MtpAssistantManifest,
  assistantLayer: number,
): number {
  if (!Number.isInteger(assistantLayer) || assistantLayer < 0 || assistantLayer >= assistantManifest.blockCount) {
    throw new Error(`Assistant layer ${assistantLayer} is outside assistant block count ${assistantManifest.blockCount}.`);
  }
  const kind = assistantManifest.layerKinds[assistantLayer] ?? "sliding-attention";
  const targetBucket = Math.min(
    targetManifest.blockCount - 1,
    Math.floor(((assistantLayer + 1) * targetManifest.blockCount) / assistantManifest.blockCount) - 1,
  );
  for (let layer = targetBucket; layer >= 0; layer -= 1) {
    if (targetManifest.layerHasKv[layer] === true && targetManifest.layerKinds[layer] === kind) {
      return layer;
    }
  }
  throw new Error(
    `No ${kind} target KV layer exists at or before target depth bucket ${targetBucket} for assistant layer ${assistantLayer}.`,
  );
}

function buildMtpAssistantExpectedTensors(params: {
  blockCount: number;
  embeddingLength: number;
  backboneEmbeddingLength: number;
  feedForwardLength: number;
  headCount: number;
  layerKeyLengths: number[];
  layerValueLengths: number[];
  nCentroids: number;
  tensorsByName: Map<string, GgufTensorInfo>;
}): ExpectedTensor[] {
  const {
    blockCount,
    embeddingLength,
    backboneEmbeddingLength,
    feedForwardLength,
    headCount,
    layerKeyLengths,
    layerValueLengths,
    nCentroids,
    tensorsByName,
  } = params;
  const vocabSize = tensorsByName.get("token_embd.weight")?.dimensions[1] ?? 262144;
  const expected: ExpectedTensor[] = [
    ...(tensorsByName.has("rope_freqs.weight")
      ? [{
          name: "rope_freqs.weight",
          dimensions: tensorsByName.get("rope_freqs.weight")?.dimensions ?? [layerKeyLengths.at(-1) ?? 0],
          allowedTypes: observedType(tensorsByName, "rope_freqs.weight", ["F32"]),
        } satisfies ExpectedTensor]
      : []),
    {
      name: "token_embd.weight",
      dimensions: [embeddingLength, vocabSize],
      allowedTypes: observedType(tensorsByName, "token_embd.weight", ["Q6_K", "Q4_K"]),
    },
    { name: "output_norm.weight", dimensions: [embeddingLength], allowedTypes: ["F32"] },
    {
      name: "mtp.pre_projection.weight",
      dimensions: [backboneEmbeddingLength * 2, embeddingLength],
      allowedTypes: observedType(tensorsByName, "mtp.pre_projection.weight", ["Q4_K"]),
    },
    {
      name: "mtp.post_projection.weight",
      dimensions: [embeddingLength, backboneEmbeddingLength],
      allowedTypes: observedType(tensorsByName, "mtp.post_projection.weight", ["Q4_K"]),
    },
    {
      name: "mtp.centroids.weight",
      dimensions: [embeddingLength, nCentroids],
      allowedTypes: observedType(tensorsByName, "mtp.centroids.weight", ["Q4_K"]),
    },
    { name: "mtp.token_ordering.weight", dimensions: [vocabSize], allowedTypes: ["I32"] },
  ];

  for (const layer of range(blockCount)) {
    const queryDim = (layerKeyLengths[layer] ?? 0) * headCount;
    const valueDim = (layerValueLengths[layer] ?? 0) * headCount;
    expected.push(
      layerTensor(layer, "attn_q.weight", [embeddingLength, queryDim], observedType(tensorsByName, `blk.${layer}.attn_q.weight`, ["Q4_K"])),
      layerTensor(layer, "attn_q_norm.weight", [layerKeyLengths[layer] ?? 0], ["F32"]),
      layerTensor(layer, "attn_output.weight", [valueDim, embeddingLength], observedType(tensorsByName, `blk.${layer}.attn_output.weight`, ["Q4_K"])),
      layerTensor(layer, "attn_norm.weight", [embeddingLength], ["F32"]),
      layerTensor(layer, "post_attention_norm.weight", [embeddingLength], ["F32"]),
      layerTensor(layer, "ffn_norm.weight", [embeddingLength], ["F32"]),
      layerTensor(layer, "ffn_gate.weight", [embeddingLength, feedForwardLength], observedType(tensorsByName, `blk.${layer}.ffn_gate.weight`, ["Q4_K"])),
      layerTensor(layer, "ffn_up.weight", [embeddingLength, feedForwardLength], observedType(tensorsByName, `blk.${layer}.ffn_up.weight`, ["Q4_K"])),
      layerTensor(layer, "ffn_down.weight", [feedForwardLength, embeddingLength], observedType(tensorsByName, `blk.${layer}.ffn_down.weight`, ["Q4_K", "Q6_K"])),
      layerTensor(layer, "post_ffw_norm.weight", [embeddingLength], ["F32"]),
      layerTensor(layer, "layer_output_scale.weight", [1], ["F32"]),
    );
  }
  return expected;
}

function buildExpectedTensors(params: {
  blockCount: number;
  embeddingLength: number;
  perLayerEmbeddingLength: number;
  feedForwardLength: number;
  headCount: number;
  headCountKv: number;
  layerHeadCountKv: number[];
  keyLength: number;
  valueLength: number;
  layerKeyLengths: number[];
  layerValueLengths: number[];
  layerHasKv: boolean[];
  layerValueProjectionModes: LayerValueProjectionMode[];
  fullAttentionLayers: number[];
  tensorsByName: Map<string, GgufTensorInfo>;
}): ExpectedTensor[] {
  const {
    blockCount,
    embeddingLength,
    perLayerEmbeddingLength,
    feedForwardLength,
    headCount,
    headCountKv,
    layerHeadCountKv,
    layerKeyLengths,
    layerValueLengths,
    layerHasKv,
    layerValueProjectionModes,
    tensorsByName,
  } = params;
  const vocabSize = tensorsByName.get("token_embd.weight")?.dimensions[1] ?? 248320;
  const expected: ExpectedTensor[] = [
    ...(tensorsByName.has("rope_freqs.weight")
      ? [{
          name: "rope_freqs.weight",
          dimensions: tensorsByName.get("rope_freqs.weight")?.dimensions ?? [params.keyLength],
          allowedTypes: observedType(tensorsByName, "rope_freqs.weight", ["F32"]),
        } satisfies ExpectedTensor]
      : []),
    {
      name: "output_norm.weight",
      dimensions: [embeddingLength],
      allowedTypes: observedType(tensorsByName, "output_norm.weight", ["F32"]),
    },
    {
      name: "token_embd.weight",
      dimensions: [embeddingLength, vocabSize],
      allowedTypes: observedType(tensorsByName, "token_embd.weight", ["Q4_0", "Q4_K"]),
    },
  ];
  if (tensorsByName.has("output.weight")) {
    expected.unshift({
      name: "output.weight",
      dimensions: [embeddingLength, vocabSize],
      allowedTypes: observedType(tensorsByName, "output.weight", ["Q4_0", "Q6_K"]),
    });
  }
  if (perLayerEmbeddingLength > 0) {
    expected.push(
      {
        name: "per_layer_token_embd.weight",
        dimensions: [perLayerEmbeddingLength * blockCount, vocabSize],
        allowedTypes: observedType(tensorsByName, "per_layer_token_embd.weight", ["Q5_K"]),
      },
      {
        name: "per_layer_model_proj.weight",
        dimensions: [embeddingLength, perLayerEmbeddingLength * blockCount],
        allowedTypes: observedType(tensorsByName, "per_layer_model_proj.weight", ["BF16", "F16", "F32"]),
      },
      {
        name: "per_layer_proj_norm.weight",
        dimensions: [perLayerEmbeddingLength],
        allowedTypes: ["F32"],
      },
    );
  }

  for (const layer of range(blockCount)) {
    const layerKind = params.fullAttentionLayers.includes(layer) ? "full-attention" : "sliding-attention";
    const headSize = layerKeyLengths[layer] ?? params.keyLength;
    const valueSize = layerValueLengths[layer] ?? params.valueLength;
    const queryDim = headSize * headCount;
    const layerHeadCountKvValue = layerHeadCountKv[layer] ?? headCountKv;
    const keyValueDim = headSize * layerHeadCountKvValue;
    const valueDim = valueSize * layerHeadCountKvValue;
    expected.push(
      layerTensor(layer, "attn_norm.weight", [embeddingLength], ["F32"]),
      layerTensor(layer, "post_attention_norm.weight", [embeddingLength], ["F32"]),
      layerTensor(layer, "post_ffw_norm.weight", [embeddingLength], ["F32"]),
      layerTensor(layer, "layer_output_scale.weight", [1], ["F32"]),
      layerTensor(layer, "attn_q.weight", [embeddingLength, queryDim], observedType(tensorsByName, `blk.${layer}.attn_q.weight`, ["Q4_0", "Q4_K", "Q5_K", "IQ4_XS"]), layerKind),
      layerTensor(layer, "attn_output.weight", [queryDim, embeddingLength], observedType(tensorsByName, `blk.${layer}.attn_output.weight`, ["Q4_0", "Q4_K"]), layerKind),
      layerTensor(layer, "attn_q_norm.weight", [headSize], ["F32"], layerKind),
      ...(layerHasKv[layer] ? [
        layerTensor(layer, "attn_k.weight", [embeddingLength, keyValueDim], observedType(tensorsByName, `blk.${layer}.attn_k.weight`, ["Q4_0", "Q4_K", "Q5_K", "IQ4_XS"]), layerKind),
        ...(layerValueProjectionModes[layer] === "separate"
          ? [layerTensor(layer, "attn_v.weight", [embeddingLength, valueDim], observedType(tensorsByName, `blk.${layer}.attn_v.weight`, ["Q4_0", "Q5_K", "Q6_K"]), layerKind)]
          : []),
        layerTensor(layer, "attn_k_norm.weight", [headSize], ["F32"], layerKind),
      ] : []),
      layerTensor(layer, "ffn_norm.weight", [embeddingLength], ["F32"]),
      layerTensor(layer, "ffn_gate.weight", [embeddingLength, feedForwardLength], observedType(tensorsByName, `blk.${layer}.ffn_gate.weight`, ["Q4_0", "Q4_K", "Q5_K", "IQ4_XS"])),
      layerTensor(layer, "ffn_up.weight", [embeddingLength, feedForwardLength], observedType(tensorsByName, `blk.${layer}.ffn_up.weight`, ["Q4_0", "Q4_K", "Q5_K", "IQ4_XS"])),
      layerTensor(layer, "ffn_down.weight", [feedForwardLength, embeddingLength], observedType(tensorsByName, `blk.${layer}.ffn_down.weight`, ["Q4_0", "Q4_K", "Q5_K", "Q6_K"])),
    );
    if (perLayerEmbeddingLength > 0) {
      expected.push(
        layerTensor(layer, "inp_gate.weight", [embeddingLength, perLayerEmbeddingLength], ["F32"]),
        layerTensor(layer, "proj.weight", [perLayerEmbeddingLength, embeddingLength], ["F32"]),
        layerTensor(layer, "post_norm.weight", [embeddingLength], ["F32"]),
      );
    }
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
  layerKind?: LayerKind,
): ExpectedTensor {
  return {
    name: `blk.${layer}.${suffix}`,
    dimensions,
    allowedTypes,
    layer,
    layerKind,
  };
}

function requiredNumber(metadata: GgufMetadata["metadata"], key: string): number {
  const value = getMetadataNumber(metadata, key);
  if (value === undefined) {
    throw new Error(`Missing numeric GGUF metadata: ${key}`);
  }
  return value;
}

function readLayerHeadCountKv(metadata: GgufMetadata["metadata"], blockCount: number): number[] {
  const arrayValue = getMetadataNumberArray(metadata, "gemma4.attention.head_count_kv");
  if (arrayValue) {
    if (arrayValue.length !== blockCount) {
      throw new Error(
        `gemma4.attention.head_count_kv length ${arrayValue.length} does not match block_count ${blockCount}`,
      );
    }
    return arrayValue;
  }
  const scalarValue = requiredNumber(metadata, "gemma4.attention.head_count_kv");
  return range(blockCount).map(() => scalarValue);
}

function requiredString(metadata: GgufMetadata["metadata"], key: string): string {
  const value = getMetadataString(metadata, key);
  if (value === undefined) {
    throw new Error(`Missing string GGUF metadata: ${key}`);
  }
  return value;
}

function requiredNumberArray(metadata: GgufMetadata["metadata"], key: string): number[] | undefined {
  const value = getMetadataNumberArray(metadata, key);
  return value;
}

function requiredNumberTuple3(metadata: GgufMetadata["metadata"], key: string): [number, number, number] {
  const value = requiredNumberArray(metadata, key);
  if (!value || value.length !== 3) {
    throw new Error(`Missing numeric[3] GGUF metadata: ${key}`);
  }
  return [value[0] ?? 0, value[1] ?? 0, value[2] ?? 0];
}

function firstNumber(metadata: GgufMetadata["metadata"], keys: string[], fallback: number): number {
  for (const key of keys) {
    const value = getMetadataNumber(metadata, key);
    if (value !== undefined) {
      return value;
    }
  }
  return fallback;
}

function boolArray(metadata: GgufMetadata["metadata"], key: string, expectedLength: number): boolean[] | undefined {
  const value: GgufMetadataValue | undefined = metadata[key];
  if (!isArraySummary(value) || value.truncated || value.length !== expectedLength) {
    return undefined;
  }
  const bools = value.sample.filter((item): item is boolean => typeof item === "boolean");
  return bools.length === expectedLength ? bools : undefined;
}

function isArraySummary(value: GgufMetadataValue | undefined): value is Extract<GgufMetadataValue, { sample: GgufMetadataValue[] }> {
  return typeof value === "object" && value !== null && "sample" in value;
}

function range(count: number): number[] {
  return Array.from({ length: count }, (_, index) => index);
}
