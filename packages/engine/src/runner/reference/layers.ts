import {
  getMetadataNumberArray,
  type GgmlTypeName,
} from "../../gguf";
import {
  gqaAttention,
  rmsNorm,
} from "./kernels";
import {
  dequantizeRow,
  float16ToFloat32,
  float32ToFloat16,
  quantizeQ8_0,
  quantizeQ8_K,
  vecDotIQ4_XS_Q8_K,
  vecDotQ4_K_Q8_K,
  vecDotQ5_K_Q8_K,
  vecDotQ6_K_Q8_K,
  vecDotQ8_0_Q8_0,
} from "../../quant";
import {
  tensorByteLength,
} from "../../tensor-reader";
import {
  type LayerKind,
  type ModelManifest,
} from "../../model";
import {
  type ForwardTrace,
  type FullAttentionCache,
  type InferenceState,
  type ModelSession,
  type OutputResult,
  requiredFullAttentionCache,
  timedAsync,
  timedSync,
  topK,
} from "../../runtime";
import type {
  SegmentRunnerProvider,
} from "../segment-runner";

export type ReferenceTensorSession = {
  getTensor(name: string): ReturnType<ModelSession["getTensor"]>;
  hasProvider(name: SegmentRunnerProvider): boolean;
  readF32Tensor(name: string): Promise<Float32Array>;
  readWeightBytes(name: string): Promise<Uint8Array>;
  tensorReader: ModelSession["tensorReader"];
};

export type PreparedInput = {
  hidden: Float32Array;
  perLayerInputs?: Float32Array;
};

export async function prepareInput(
  session: ModelSession,
  tokenIds: readonly number[],
  trace?: ForwardTrace,
): Promise<PreparedInput> {
  const manifest = session.manifest;
  const tokenCount = tokenIds.length;
  const hidden = await timedAsync(trace, "embedding read", async () => {
    const rows = await session.readEmbeddingRows(tokenIds);
    const scale = Math.sqrt(manifest.embeddingLength);
    for (let index = 0; index < rows.length; index += 1) {
      rows[index] = Math.fround((rows[index] ?? 0) * scale);
    }
    return rows;
  });

  if (manifest.perLayerEmbeddingLength <= 0 || tokenCount === 0) {
    return { hidden };
  }

  const perLayerInputs = await timedAsync(trace, "per-layer input projection", async () => {
    const perLayerLength = manifest.perLayerEmbeddingLength;
    const totalPerLayerLength = perLayerLength * manifest.blockCount;
    const tokenRows = await readTensorRows(session, "per_layer_token_embd.weight", tokenIds);
    const tokenScale = Math.sqrt(perLayerLength);
    for (let index = 0; index < tokenRows.length; index += 1) {
      tokenRows[index] = Math.fround((tokenRows[index] ?? 0) * tokenScale);
    }

    const projected = await matMulWeight(session, "per_layer_model_proj.weight", hidden, trace);
    const projectionScale = 1 / Math.sqrt(manifest.embeddingLength);
    const normWeight = await readF32ModelTensor(session, "per_layer_proj_norm.weight");
    const output = new Float32Array(manifest.blockCount * tokenCount * perLayerLength);
    for (let token = 0; token < tokenCount; token += 1) {
      for (let layer = 0; layer < manifest.blockCount; layer += 1) {
        const sourceOffset = token * totalPerLayerLength + layer * perLayerLength;
        const projectedSlice = new Float32Array(perLayerLength);
        for (let index = 0; index < perLayerLength; index += 1) {
          projectedSlice[index] = Math.fround((projected[sourceOffset + index] ?? 0) * projectionScale);
        }
        const projectedNorm = rmsNorm(projectedSlice, normWeight, session.epsilon);
        const targetOffset = (layer * tokenCount + token) * perLayerLength;
        for (let index = 0; index < perLayerLength; index += 1) {
          output[targetOffset + index] = Math.fround(
            Math.fround((tokenRows[sourceOffset + index] ?? 0) + (projectedNorm[index] ?? 0)) *
              Math.SQRT1_2,
          );
        }
      }
    }
    return output;
  });

  return { hidden, perLayerInputs };
}

export async function preparePreparedHiddenInput(
  session: ModelSession,
  hidden: Float32Array,
  trace?: ForwardTrace,
): Promise<PreparedInput> {
  const manifest = session.manifest;
  const tokenCount = hidden.length / manifest.embeddingLength;
  if (!Number.isInteger(tokenCount)) {
    throw new Error(`Prepared hidden shape mismatch: ${hidden.length}`);
  }
  if (manifest.perLayerEmbeddingLength <= 0 || tokenCount === 0) {
    return { hidden };
  }

  const perLayerInputs = await timedAsync(trace, "per-layer prepared input projection", async () => {
    const perLayerLength = manifest.perLayerEmbeddingLength;
    const totalPerLayerLength = perLayerLength * manifest.blockCount;
    const paddingRow = await readTensorRows(session, "per_layer_token_embd.weight", [0]);
    const tokenScale = Math.sqrt(perLayerLength);
    for (let index = 0; index < paddingRow.length; index += 1) {
      paddingRow[index] = Math.fround((paddingRow[index] ?? 0) * tokenScale);
    }

    const projected = await matMulWeight(session, "per_layer_model_proj.weight", hidden, trace);
    const projectionScale = 1 / Math.sqrt(manifest.embeddingLength);
    const normWeight = await readF32ModelTensor(session, "per_layer_proj_norm.weight");
    const output = new Float32Array(manifest.blockCount * tokenCount * perLayerLength);
    for (let token = 0; token < tokenCount; token += 1) {
      for (let layer = 0; layer < manifest.blockCount; layer += 1) {
        const sourceOffset = token * totalPerLayerLength + layer * perLayerLength;
        const projectedSlice = new Float32Array(perLayerLength);
        for (let index = 0; index < perLayerLength; index += 1) {
          projectedSlice[index] = Math.fround((projected[sourceOffset + index] ?? 0) * projectionScale);
        }
        const projectedNorm = rmsNorm(projectedSlice, normWeight, session.epsilon);
        const targetOffset = (layer * tokenCount + token) * perLayerLength;
        const paddingOffset = layer * perLayerLength;
        for (let index = 0; index < perLayerLength; index += 1) {
          output[targetOffset + index] = Math.fround(
            Math.fround((paddingRow[paddingOffset + index] ?? 0) + (projectedNorm[index] ?? 0)) *
              Math.SQRT1_2,
          );
        }
      }
    }
    return output;
  });

  return { hidden, perLayerInputs };
}

export async function forwardAttentionLayer(
  session: ModelSession,
  manifest: ModelManifest,
  state: InferenceState,
  layer: number,
  input: Float32Array,
  positions: Int32Array,
  perLayerInputs?: Float32Array,
  epsilon = session.epsilon,
  trace?: ForwardTrace,
  attentionCausal = true,
): Promise<Float32Array> {
  const kind = manifest.layerKinds[layer] ?? "sliding-attention";
  const cacheLayer = manifest.layerHasKv[layer] ? layer : manifest.kvSourceLayers[layer] ?? layer;
  const cache = requiredFullAttentionCache(state, cacheLayer);
  const tokenCount = input.length / manifest.embeddingLength;
  const tokenPositions = tokenPositionsFromMrope(positions, tokenCount);
  for (const position of tokenPositions) {
    if (position < 0 || position >= state.contextLength) {
      throw new Error(`Position ${position} is outside context length ${state.contextLength}`);
    }
  }
  const headSize = manifest.layerKeyLengths[layer] ?? manifest.keyLength;
  const valueSize = manifest.layerValueLengths[layer] ?? manifest.valueLength;
  const headCountKv = manifest.layerHeadCountKv[layer] ?? manifest.headCountKv;
  const queryDim = manifest.headCount * headSize;
  const valueDim = manifest.headCount * valueSize;
  const freqFactors = await readRopeFreqFactors(session, kind);

  const attnNorm = await timedAsync(
    trace,
    "attention norm",
    async () => rmsNormRows(input, await readF32ModelTensor(session, `blk.${layer}.attn_norm.weight`), epsilon),
    { layer, layerKind: kind },
  );
  const q = await timedAsync(
    trace,
    "attention projection q",
    () => matMulWeight(session, `blk.${layer}.attn_q.weight`, attnNorm, trace),
    { layer, layerKind: kind, weightName: `blk.${layer}.attn_q.weight` },
  );
  if (q.length !== tokenCount * queryDim) {
    throw new Error(`Q projection shape mismatch for layer ${layer}: ${q.length}`);
  }
  const qNorm = await timedAsync(
    trace,
    "q norm",
    async () => normHeads(q, await readF32ModelTensor(session, `blk.${layer}.attn_q_norm.weight`), epsilon),
    { layer, layerKind: kind },
  );
  const qRope = timedSync(
    trace,
    "RoPE q",
    () => ropeNeox(qNorm, {
      headSize,
      headCount: manifest.headCount,
      tokenCount,
      positions: tokenPositions,
      nDims: ropeDimensionCount(manifest, kind),
      freqBase: ropeFreqBase(manifest, kind),
      freqFactors,
    }),
    { layer, layerKind: kind },
  );
  if (manifest.layerHasKv[layer]) {
    const kProjection = await timedAsync(
      trace,
      "attention projection k",
      () => matMulWeight(session, `blk.${layer}.attn_k.weight`, attnNorm, trace),
      { layer, layerKind: kind, weightName: `blk.${layer}.attn_k.weight` },
    );
    const vProjection = manifest.layerValueProjectionModes[layer] === "shared-with-key"
      ? kProjection
      : await timedAsync(
        trace,
        "attention projection v",
        () => matMulWeight(session, `blk.${layer}.attn_v.weight`, attnNorm, trace),
        { layer, layerKind: kind, weightName: `blk.${layer}.attn_v.weight` },
      );
    const kNorm = await timedAsync(
      trace,
      "k norm",
      async () => normHeads(kProjection, await readF32ModelTensor(session, `blk.${layer}.attn_k_norm.weight`), epsilon),
      { layer, layerKind: kind },
    );
    const vNorm = timedSync(
      trace,
      "v norm",
      () => rmsNormRowsNoWeight(vProjection, valueSize, epsilon),
      { layer, layerKind: kind },
    );
    const kRope = timedSync(
      trace,
      "RoPE k",
      () => ropeNeox(kNorm, {
        headSize,
        headCount: headCountKv,
        tokenCount,
        positions: tokenPositions,
        nDims: ropeDimensionCount(manifest, kind),
        freqBase: ropeFreqBase(manifest, kind),
        freqFactors,
      }),
      { layer, layerKind: kind },
    );
    timedSync(
      trace,
      "KV cache update",
      () => updateFullAttentionCache(cache, kRope, vNorm, tokenPositions),
      { layer, layerKind: kind },
    );
  }

  const keyValueRange = attentionRange(
    Math.max(...Array.from(tokenPositions)),
    cache,
    kind === "sliding-attention" && attentionCausal ? manifest.slidingWindow : undefined,
  );
  const mask = attentionCausal
    ? timedSync(
        trace,
        "attention mask",
        () => causalMask(tokenPositions, keyValueRange.start, keyValueRange.count),
        { layer, layerKind: kind },
      )
    : undefined;
  const directRing = cache.kind === "sliding";
  const attentionKey = directRing
    ? cache.key
    : timedSync(
      trace,
      "key compact",
      () => compactKeyCache(cache, keyValueRange.start, keyValueRange.count),
      { layer, layerKind: kind },
    );
  const attentionValue = directRing
    ? cache.value
    : timedSync(
      trace,
      "value compact",
      () => compactValueCache(cache, keyValueRange.start, keyValueRange.count),
      { layer, layerKind: kind },
    );
  const attention = await timedAsync(
    trace,
    "GQA attention",
    async () => {
      const attentionOptions = {
        headSize,
        queryHeadCount: manifest.headCount,
        keyValueHeadCount: cache.headCountKv,
        tokenCount,
        keyValueTokenCount: keyValueRange.count,
        keyValueStart: directRing ? keyValueRange.start : 0,
        keyValueCapacity: directRing ? cache.capacity : keyValueRange.count,
        scale: 1,
        causal: attentionCausal,
        mask,
        valueLayout: "dim-head-token" as const,
        quantizeQueryForScore: "f16" as const,
      };
      return gqaAttention(
        qRope,
        attentionKey,
        attentionValue,
        attentionOptions,
      );
    },
    { layer, layerKind: kind },
  );
  if (attention.length !== tokenCount * valueDim) {
    throw new Error(`Attention output shape mismatch for layer ${layer}: ${attention.length}`);
  }
  const attentionForOutput = timedSync(
    trace,
    "attention output f16 cast",
    () => castF16Rows(attention),
    { layer, layerKind: kind },
  );

  const attentionOutput = await timedAsync(
    trace,
    "attention output projection",
    () => matMulWeight(session, `blk.${layer}.attn_output.weight`, attentionForOutput, trace),
    { layer, layerKind: kind, weightName: `blk.${layer}.attn_output.weight` },
  );
  const attentionPostNorm = await timedAsync(
    trace,
    "attention post norm",
    async () => rmsNormRows(attentionOutput, await readF32ModelTensor(session, `blk.${layer}.post_attention_norm.weight`), epsilon),
    { layer, layerKind: kind },
  );
  const attentionResidual = residualAdd(input, attentionPostNorm);
  const ffn = await forwardFfn(session, manifest, layer, attentionResidual, epsilon, trace);
  const enriched = await applyPerLayerInput(session, manifest, layer, ffn, tokenCount, perLayerInputs, epsilon, trace);
  const scaled = await timedAsync(
    trace,
    "layer output scale",
    async () => {
      const scale = (await readF32ModelTensor(session, `blk.${layer}.layer_output_scale.weight`))[0] ?? 1;
      const output = new Float32Array(enriched.length);
      for (let index = 0; index < output.length; index += 1) {
        output[index] = Math.fround((enriched[index] ?? 0) * scale);
      }
      return output;
    },
    { layer, layerKind: kind },
  );
  return layer === manifest.blockCount - 1 && tokenCount > 1
    ? scaled.slice(scaled.length - manifest.embeddingLength)
    : scaled;
}

export const forwardFullAttentionLayer = forwardAttentionLayer;

export async function forwardOutput(
  session: ModelSession,
  hidden: Float32Array,
  options: {
    topK?: number;
    trace?: ForwardTrace;
  } = {},
): Promise<OutputResult> {
  const norm = await timedAsync(
    options.trace,
    "final norm",
    async () => rmsNormRows(
      hidden,
      await readF32ModelTensor(session, "output_norm.weight"),
      session.epsilon,
    ),
  );
  const outputWeight = session.tensorReader.metadata.tensors.some((tensor) => tensor.name === "output.weight")
    ? "output.weight"
    : "token_embd.weight";
  const logits = await timedAsync(
    options.trace,
    "output logits",
    () => matMulWeight(session, outputWeight, norm, options.trace),
    { weightName: outputWeight },
  );
  const softcap = session.manifest.finalLogitSoftcap;
  if (softcap !== undefined && softcap > 0) {
    for (let index = 0; index < logits.length; index += 1) {
      logits[index] = Math.fround(Math.tanh((logits[index] ?? 0) / softcap) * softcap);
    }
  }
  const suppressTokens = getMetadataNumberArray(session.tensorReader.metadata.metadata, "tokenizer.ggml.suppress_tokens") ?? [];
  for (const token of suppressTokens) {
    if (Number.isInteger(token) && token >= 0 && token < logits.length) {
      logits[token] = -Infinity;
    }
  }
  return {
    topTokens: topK(logits, options.topK ?? 10),
  };
}

async function forwardFfn(
  session: ModelSession,
  manifest: ModelManifest,
  layer: number,
  residual: Float32Array,
  epsilon: number,
  trace?: ForwardTrace,
): Promise<Float32Array> {
  const ffnNorm = await timedAsync(
    trace,
    "FFN norm",
    async () => rmsNormRows(residual, await readF32ModelTensor(session, `blk.${layer}.ffn_norm.weight`), epsilon),
    { layer },
  );
  const gateUpBatch = await timedAsync(
    trace,
    "FFN gate/up projection batch",
    () => matMulWeightBatch(session, [`blk.${layer}.ffn_gate.weight`, `blk.${layer}.ffn_up.weight`], ffnNorm, trace),
    { layer },
  );
  const gate = gateUpBatch?.[0] ?? await timedAsync(
    trace,
    "FFN gate projection",
    () => matMulWeight(session, `blk.${layer}.ffn_gate.weight`, ffnNorm, trace),
    { layer, weightName: `blk.${layer}.ffn_gate.weight` },
  );
  const up = gateUpBatch?.[1] ?? await timedAsync(
    trace,
    "FFN up projection",
    () => matMulWeight(session, `blk.${layer}.ffn_up.weight`, ffnNorm, trace),
    { layer, weightName: `blk.${layer}.ffn_up.weight` },
  );
  const gated = timedSync(trace, "FFN GeGLU", () => {
    const output = new Float32Array(gate.length);
    for (let index = 0; index < output.length; index += 1) {
      output[index] = Math.fround(gelu(gate[index] ?? 0) * (up[index] ?? 0));
    }
    return output;
  }, { layer });
  const ffnOut = await timedAsync(
    trace,
    "FFN down projection",
    () => matMulWeight(session, `blk.${layer}.ffn_down.weight`, gated, trace),
    { layer, weightName: `blk.${layer}.ffn_down.weight` },
  );
  const postNorm = await timedAsync(
    trace,
    "FFN post norm",
    async () => rmsNormRows(ffnOut, await readF32ModelTensor(session, `blk.${layer}.post_ffw_norm.weight`), epsilon),
    { layer },
  );
  if (postNorm.length !== residual.length || postNorm.length % manifest.embeddingLength !== 0) {
    throw new Error(`FFN output shape mismatch for layer ${layer}: ${postNorm.length}`);
  }
  return residualAdd(residual, postNorm);
}

async function applyPerLayerInput(
  session: ModelSession,
  manifest: ModelManifest,
  layer: number,
  input: Float32Array,
  tokenCount: number,
  perLayerInputs: Float32Array | undefined,
  epsilon: number,
  trace?: ForwardTrace,
): Promise<Float32Array> {
  if (manifest.perLayerEmbeddingLength <= 0 || !perLayerInputs) {
    return input;
  }
  const perLayerLength = manifest.perLayerEmbeddingLength;
  const gate = await timedAsync(
    trace,
    "per-layer input gate",
    async () => {
      const projected = await matMulWeight(session, `blk.${layer}.inp_gate.weight`, input, trace);
      for (let index = 0; index < projected.length; index += 1) {
        projected[index] = gelu(projected[index] ?? 0);
      }
      return projected;
    },
    { layer },
  );
  const mixed = timedSync(trace, "per-layer input mix", () => {
    const output = new Float32Array(gate.length);
    const sourceBase = layer * tokenCount * perLayerLength;
    for (let token = 0; token < tokenCount; token += 1) {
      for (let index = 0; index < perLayerLength; index += 1) {
        const offset = token * perLayerLength + index;
        output[offset] = Math.fround((gate[offset] ?? 0) * (perLayerInputs[sourceBase + offset] ?? 0));
      }
    }
    return output;
  }, { layer });
  const projected = await timedAsync(
    trace,
    "per-layer output projection",
    () => matMulWeight(session, `blk.${layer}.proj.weight`, mixed, trace),
    { layer, weightName: `blk.${layer}.proj.weight` },
  );
  const norm = await timedAsync(
    trace,
    "per-layer post norm",
    async () => rmsNormRows(projected, await readF32ModelTensor(session, `blk.${layer}.post_norm.weight`), epsilon),
    { layer },
  );
  return residualAdd(input, norm);
}

async function readF32ModelTensor(
  session: ModelSession,
  name: string,
): Promise<Float32Array> {
  return session.readF32Tensor(name);
}

async function readTensorRows(
  session: ModelSession,
  tensorName: string,
  rowIds: readonly number[],
): Promise<Float32Array> {
  const tensor = session.getTensor(tensorName);
  const rowElements = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const rowByteLength = tensorByteLength({ ...tensor, dimensions: [rowElements] });
  const rows = new Float32Array(rowElements * rowIds.length);
  for (let index = 0; index < rowIds.length; index += 1) {
    const rowId = rowIds[index] ?? 0;
    if (rowId < 0 || rowId >= rowCount) {
      throw new Error(`${tensorName} row ${rowId} is outside ${rowCount}`);
    }
    const rowBytes = await session.tensorReader.readTensorRange({
      tensor,
      offset: BigInt(rowByteLength * rowId),
      length: rowByteLength,
    });
    rows.set(dequantizeRow(tensor.type, rowBytes, rowElements), index * rowElements);
  }
  return rows;
}

export async function matMulWeight(
  session: ReferenceTensorSession,
  weightName: string,
  inputColumns: Float32Array,
  trace?: ForwardTrace,
): Promise<Float32Array> {
  requireReferenceProvider(session);
  const tensor = session.getTensor(weightName);
  if (tensor.type === "F32" || tensor.type === "F16" || tensor.type === "BF16") {
    return matMulDenseRows(session, weightName, inputColumns);
  }
  if (tensor.type === "Q4_K" || tensor.type === "Q5_K" || tensor.type === "Q6_K" || tensor.type === "IQ4_XS") {
    return matMulKQ8K(session, weightName, inputColumns, tensor.type, trace);
  }
  if (tensor.type === "Q8_0") {
    return matMulQ8_0Weight(session, weightName, inputColumns, trace);
  }
  throw new Error(`${weightName} has unsupported matmul type ${tensor.type}`);
}

function requireReferenceProvider(session: ReferenceTensorSession): void {
  if (!session.hasProvider("reference")) {
    throw new Error("Reference tensor execution requires an enabled reference provider.");
  }
}

async function matMulWeightBatch(
  _session: ModelSession,
  _weightNames: readonly string[],
  _inputColumns: Float32Array,
  _trace?: ForwardTrace,
): Promise<Float32Array[] | undefined> {
  return undefined;
}

async function matMulDenseRows(
  session: ReferenceTensorSession,
  weightName: string,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  const bytes = await session.readWeightBytes(weightName);
  const rowByteLength = tensorByteLength({ ...tensor, dimensions: [inputSize] });
  const output = new Float32Array(rowCount * columnCount);
  for (let row = 0; row < rowCount; row += 1) {
    const rowBytes = bytes.subarray(row * rowByteLength, (row + 1) * rowByteLength);
    const weight = dequantizeRow(tensor.type, rowBytes, inputSize);
    for (let column = 0; column < columnCount; column += 1) {
      const inputOffset = column * inputSize;
      let sum = 0;
      for (let index = 0; index < inputSize; index += 1) {
        sum = Math.fround(sum + Math.fround((weight[index] ?? 0) * (inputColumns[inputOffset + index] ?? 0)));
      }
      output[column * rowCount + row] = sum;
    }
  }
  return output;
}

async function matMulKQ8K(
  session: ReferenceTensorSession,
  weightName: string,
  inputColumns: Float32Array,
  type: Extract<GgmlTypeName, "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS">,
  trace?: ForwardTrace,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  const rowByteLength = tensorByteLength({ ...tensor, dimensions: [inputSize] });
  const weightBytes = await session.readWeightBytes(weightName);
  return timedAsync(
    trace,
    "reference matmul",
    () => matMulKQ8KReference(type, weightBytes, inputColumns, inputSize, rowCount, columnCount, rowByteLength),
    { weightName },
  );
}

function matMulKQ8KReference(
  type: Extract<GgmlTypeName, "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS">,
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
  rowByteLength: number,
): Float32Array {
  const output = new Float32Array(rowCount * columnCount);
  for (let column = 0; column < columnCount; column += 1) {
    const q8 = quantizeQ8_K(inputColumns.slice(column * inputSize, (column + 1) * inputSize));
    for (let row = 0; row < rowCount; row += 1) {
      const rowBytes = weightBytes.subarray(row * rowByteLength, (row + 1) * rowByteLength);
      if (type === "Q4_K") {
        output[column * rowCount + row] = vecDotQ4_K_Q8_K(rowBytes, q8);
      } else if (type === "Q5_K") {
        output[column * rowCount + row] = vecDotQ5_K_Q8_K(rowBytes, q8);
      } else if (type === "Q6_K") {
        output[column * rowCount + row] = vecDotQ6_K_Q8_K(rowBytes, q8);
      } else {
        output[column * rowCount + row] = vecDotIQ4_XS_Q8_K(rowBytes, q8);
      }
    }
  }

  return output;
}

async function matMulQ8_0Weight(
  session: ReferenceTensorSession,
  weightName: string,
  inputColumns: Float32Array,
  trace?: ForwardTrace,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  if (tensor.type !== "Q8_0") {
    throw new Error(`${weightName} must be Q8_0, got ${tensor.type}`);
  }
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  const rowByteLength = tensorByteLength({ ...tensor, dimensions: [inputSize] });
  const weightBytes = await session.readWeightBytes(weightName);
  return timedAsync(
    trace,
    "reference Q8_0 matmul",
    () => matMulQ8_0Reference(weightBytes, inputColumns, inputSize, rowCount, columnCount, rowByteLength),
    { weightName },
  );
}

function matMulQ8_0Reference(
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
  rowByteLength: number,
): Float32Array {
  const output = new Float32Array(rowCount * columnCount);
  for (let column = 0; column < columnCount; column += 1) {
    const q8 = quantizeQ8_0(inputColumns.slice(column * inputSize, (column + 1) * inputSize));
    for (let row = 0; row < rowCount; row += 1) {
      output[column * rowCount + row] = vecDotQ8_0_Q8_0(
        weightBytes.subarray(row * rowByteLength, (row + 1) * rowByteLength),
        q8,
      );
    }
  }
  return output;
}

function rmsNormRows(input: Float32Array, weight: Float32Array, epsilon: number): Float32Array {
  const output = new Float32Array(input.length);
  for (let row = 0; row < input.length / weight.length; row += 1) {
    const offset = row * weight.length;
    output.set(rmsNorm(input.slice(offset, offset + weight.length), weight, epsilon), offset);
  }
  return output;
}

function rmsNormRowsNoWeight(input: Float32Array, rowSize: number, epsilon: number): Float32Array {
  const output = new Float32Array(input.length);
  for (let row = 0; row < input.length / rowSize; row += 1) {
    const offset = row * rowSize;
    let sumSquares = 0;
    for (let index = 0; index < rowSize; index += 1) {
      const value = input[offset + index] ?? 0;
      sumSquares += value * value;
    }
    const scale = 1 / Math.sqrt(sumSquares / rowSize + epsilon);
    for (let index = 0; index < rowSize; index += 1) {
      output[offset + index] = Math.fround((input[offset + index] ?? 0) * scale);
    }
  }
  return output;
}

function residualAdd(left: Float32Array, right: Float32Array): Float32Array {
  if (left.length !== right.length) {
    throw new Error(`Residual shape mismatch: left=${left.length} right=${right.length}`);
  }
  const output = new Float32Array(left.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Math.fround((left[index] ?? 0) + (right[index] ?? 0));
  }
  return output;
}

function normHeads(input: Float32Array, weight: Float32Array, epsilon: number): Float32Array {
  const output = new Float32Array(input.length);
  for (let row = 0; row < input.length / weight.length; row += 1) {
    const offset = row * weight.length;
    output.set(rmsNorm(input.slice(offset, offset + weight.length), weight, epsilon), offset);
  }
  return output;
}

function updateFullAttentionCache(
  cache: FullAttentionCache,
  key: Float32Array,
  value: Float32Array,
  positions: Int32Array,
): void {
  const tokenCount = positions.length;
  for (let token = 0; token < tokenCount; token += 1) {
    const position = positions[token] ?? 0;
    if (position < 0 || position >= cache.contextLength) {
      throw new Error(`Position ${position} is outside context length ${cache.contextLength}`);
    }
    const slot = position % cache.capacity;
    for (let head = 0; head < cache.headCountKv; head += 1) {
      for (let dim = 0; dim < cache.keyLength; dim += 1) {
        cache.key[(slot * cache.headCountKv + head) * cache.keyLength + dim] =
          key[(token * cache.headCountKv + head) * cache.keyLength + dim] ?? 0;
      }
      for (let dim = 0; dim < cache.valueLength; dim += 1) {
        cache.value[(dim * cache.headCountKv + head) * cache.capacity + slot] =
          value[(token * cache.headCountKv + head) * cache.valueLength + dim] ?? 0;
      }
    }
  }
}

function causalMask(positions: Int32Array, keyValueStart: number, keyValueTokenCount: number): Float32Array {
  const output = new Float32Array(positions.length * keyValueTokenCount);
  for (let token = 0; token < positions.length; token += 1) {
    const position = positions[token] ?? 0;
    for (let keyToken = 0; keyToken < keyValueTokenCount; keyToken += 1) {
      const keyPosition = keyValueStart + keyToken;
      output[token * keyValueTokenCount + keyToken] =
        keyPosition <= position ? 0 : -Infinity;
    }
  }
  return output;
}

function tokenPositionsFromMrope(positions: Int32Array, tokenCount: number): Int32Array {
  if (positions.length === tokenCount) {
    return positions;
  }
  if (positions.length === tokenCount * 4) {
    return positions.slice(0, tokenCount);
  }
  throw new Error(`Expected ${tokenCount} or ${tokenCount * 4} positions, got ${positions.length}`);
}

function compactValueCache(
  cache: FullAttentionCache,
  keyValueStart: number,
  keyValueTokenCount: number,
): Float32Array {
  const output = new Float32Array(cache.valueLength * cache.headCountKv * keyValueTokenCount);
  for (let dim = 0; dim < cache.valueLength; dim += 1) {
    for (let head = 0; head < cache.headCountKv; head += 1) {
      for (let token = 0; token < keyValueTokenCount; token += 1) {
        const slot = (keyValueStart + token) % cache.capacity;
        output[(dim * cache.headCountKv + head) * keyValueTokenCount + token] =
          cache.value[(dim * cache.headCountKv + head) * cache.capacity + slot] ?? 0;
      }
    }
  }
  return output;
}

function compactKeyCache(
  cache: FullAttentionCache,
  keyValueStart: number,
  keyValueTokenCount: number,
): Float32Array {
  const output = new Float32Array(keyValueTokenCount * cache.headCountKv * cache.keyLength);
  for (let token = 0; token < keyValueTokenCount; token += 1) {
    const slot = (keyValueStart + token) % cache.capacity;
    for (let head = 0; head < cache.headCountKv; head += 1) {
      for (let dim = 0; dim < cache.keyLength; dim += 1) {
        output[(token * cache.headCountKv + head) * cache.keyLength + dim] =
          cache.key[(slot * cache.headCountKv + head) * cache.keyLength + dim] ?? 0;
      }
    }
  }
  return output;
}

function attentionRange(
  tokenPosition: number,
  cache: FullAttentionCache,
  slidingWindow?: number,
): { start: number; count: number } {
  const available = Math.min(cache.contextLength, tokenPosition + 1);
  const count = Math.min(cache.capacity, slidingWindow ?? cache.capacity, available);
  return { start: Math.max(0, tokenPosition + 1 - count), count };
}

function castF16Rows(input: Float32Array): Float32Array {
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = float16ToFloat32(float32ToFloat16(input[index] ?? 0));
  }
  return output;
}

function ropeNeox(
  input: Float32Array,
  options: {
    headSize: number;
    headCount: number;
    tokenCount: number;
    positions: Int32Array;
    nDims: number;
    freqBase: number;
    freqFactors?: Float32Array;
  },
): Float32Array {
  const { headSize, headCount, tokenCount, positions, nDims, freqBase, freqFactors } = options;
  if (input.length !== headSize * headCount * tokenCount) {
    throw new Error(`RoPE input shape mismatch: ${input.length}`);
  }
  if (nDims > headSize || nDims % 2 !== 0) {
    throw new Error(`Invalid RoPE dimension count: ${nDims}`);
  }
  if (freqFactors && freqFactors.length < nDims / 2) {
    throw new Error(`RoPE freq_factors length mismatch: ${freqFactors.length} < ${nDims / 2}`);
  }
  const output = new Float32Array(input);
  const thetaScale = Math.pow(freqBase, -2 / nDims);
  for (let token = 0; token < tokenCount; token += 1) {
    const position = positions[token] ?? 0;
    for (let head = 0; head < headCount; head += 1) {
      const rowOffset = (token * headCount + head) * headSize;
      let theta = position;
      for (let i0 = 0; i0 < nDims; i0 += 2) {
        const index = i0 / 2;
        const x0 = input[rowOffset + index] ?? 0;
        const x1 = input[rowOffset + nDims / 2 + index] ?? 0;
        const thetaWithFactor = theta / (freqFactors?.[index] ?? 1);
        const cosTheta = Math.cos(thetaWithFactor);
        const sinTheta = Math.sin(thetaWithFactor);
        output[rowOffset + index] = Math.fround(Math.fround(x0 * cosTheta) - Math.fround(x1 * sinTheta));
        output[rowOffset + nDims / 2 + index] = Math.fround(Math.fround(x0 * sinTheta) + Math.fround(x1 * cosTheta));
        theta = Math.fround(theta * thetaScale);
      }
    }
  }
  return output;
}

function ropeDimensionCount(manifest: ModelManifest, kind: LayerKind): number {
  return kind === "sliding-attention"
    ? manifest.rope.slidingDimensionCount
    : manifest.rope.fullDimensionCount;
}

function ropeFreqBase(manifest: ModelManifest, kind: LayerKind): number {
  return kind === "sliding-attention"
    ? manifest.rope.slidingFreqBase
    : manifest.rope.fullFreqBase;
}

async function readRopeFreqFactors(session: ModelSession, kind: LayerKind): Promise<Float32Array | undefined> {
  if (kind === "sliding-attention") {
    return undefined;
  }
  try {
    const tensor = session.getTensor("rope_freqs.weight");
    if (tensor.type !== "F32") {
      return undefined;
    }
  } catch {
    return undefined;
  }
  return session.readF32Tensor("rope_freqs.weight");
}

function gelu(value: number): number {
  if (value <= -10) {
    return 0;
  }
  if (value >= 10) {
    return value;
  }
  const x = float16ToFloat32(float32ToFloat16(value));
  const inner = Math.fround(
    Math.fround(Math.sqrt(2 / Math.PI) * x) *
      Math.fround(1 + Math.fround(0.044715 * Math.fround(x * x))),
  );
  const activated = Math.fround(Math.fround(0.5 * x) * Math.fround(1 + Math.tanh(inner)));
  return float16ToFloat32(float32ToFloat16(activated));
}
