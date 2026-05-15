import {
  type GgmlTypeName,
} from "../../gguf";
import type {
  AudioEncodeResult,
  AudioFeatures,
  AudioSession,
} from "../../audio";
import type {
  AudioManifest,
} from "../../model";
import {
  dequantizeRow,
} from "../../quant";
import {
  tensorByteLength,
} from "../../tensor-reader";
import {
  audioAddBiasRowsWasm,
  audioClampWasm,
  audioConv2dSubsampleWasm,
  audioDepthwiseConv1dWasm,
  audioFlattenChannelsLastWasm,
  audioGluWasm,
  audioResidualAddScaleWasm,
  audioResidualAddWasm,
  audioRmsNormWasm,
  audioSiluWasm,
  createWasmQuantizedWeightHandle,
  matMulQuantizedMultiWasm,
  matMulQuantizedWasm,
  matMulQuantizedWasmResident,
  matMulQuantizedWasmResidentMulti,
  releaseWasmQuantizedWeightHandle,
  type QuantizedMatMulInput,
  type WasmQuantizedWeightHandle,
} from "./wasm-kernels";

type AudioWasmWeightCache = {
  handles: Map<string, WasmQuantizedWeightHandle>;
  bytes: number;
  hits: number;
  misses: number;
};

const residentWeightCaches = new WeakMap<AudioSession, AudioWasmWeightCache>();

export function releaseCpuAudioEncoder(session: AudioSession): void {
  const cache = residentWeightCaches.get(session);
  if (!cache) {
    return;
  }
  for (const handle of cache.handles.values()) {
    releaseWasmQuantizedWeightHandle(handle);
  }
  residentWeightCaches.delete(session);
}

function audioResidentWeightCacheEnabled(session: AudioSession): boolean {
  const provider = session.executionProvider("cpu");
  return provider?.options?.residentWeightCache === true;
}

function audioProjectionBatchingEnabled(session: AudioSession): boolean {
  const provider = session.executionProvider("cpu");
  return provider?.options?.projectionBatching !== false;
}

function audioWasmKernelsEnabled(session: AudioSession): boolean {
  const provider = session.executionProvider("cpu");
  return provider?.options?.wasmKernels !== false;
}

function residentWeightCache(session: AudioSession): AudioWasmWeightCache {
  let cache = residentWeightCaches.get(session);
  if (!cache) {
    cache = {
      handles: new Map(),
      bytes: 0,
      hits: 0,
      misses: 0,
    };
    residentWeightCaches.set(session, cache);
    session.addDisposeCallback(() => releaseCpuAudioEncoder(session));
    const statsCache = cache;
    session.setExecutionProviderStatsProvider(() => ({
      cpuAudioResidentWeightCacheEnabled: audioResidentWeightCacheEnabled(session),
      cpuAudioResidentWeightCacheCount: statsCache.handles.size,
      cpuAudioResidentWeightCacheBytes: statsCache.bytes,
      cpuAudioResidentWeightCacheHits: statsCache.hits,
      cpuAudioResidentWeightCacheMisses: statsCache.misses,
    }), "cpu-audio");
  }
  return cache;
}

async function readWasmWeightHandle(
  session: AudioSession,
  weightName: string,
  type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0",
  inputSize: number,
  rowCount: number,
): Promise<WasmQuantizedWeightHandle | undefined> {
  const cache = residentWeightCache(session);
  const cached = cache.handles.get(weightName);
  if (cached) {
    cache.hits += 1;
    return cached;
  }
  cache.misses += 1;
  const bytes = await session.readWeightBytes(weightName);
  const handle = await createWasmQuantizedWeightHandle(type, bytes, inputSize, rowCount);
  if (!handle) {
    return undefined;
  }
  cache.handles.set(weightName, handle);
  cache.bytes += handle.byteLength + handle.scaleByteLength;
  return handle;
}

export async function runCpuAudioEncoder(
  session: AudioSession,
  features: AudioFeatures,
  options: { signal?: AbortSignal } = {},
): Promise<AudioEncodeResult> {
  throwIfAborted(options.signal);
  const manifest = session.manifest;
  if (features.featureSize !== manifest.featureSize) {
    throw new Error(`Audio feature size mismatch: ${features.featureSize}`);
  }

  let { hidden, mask } = await subsampleConvProjection(session, features);
  const positionEmbeddings = audioRelativePositionEmbeddings(manifest);
  for (let layer = 0; layer < manifest.blockCount; layer += 1) {
    throwIfAborted(options.signal);
    hidden = await forwardAudioLayer(session, hidden, mask, positionEmbeddings, layer);
  }

  hidden = await matMulAudioWeight(session, "a.pre_encode.out.weight", hidden);
  if (session.hasTensor("a.pre_encode.out.bias")) {
    hidden = await addBiasRows(session, hidden, await session.readF32Tensor("a.pre_encode.out.bias"));
  }
  hidden = await rmsNormRows(
    hidden,
    await optionalTensor(session, "mm.a.soft_emb_norm.weight", manifest.outputProjectionDim),
    session.epsilon,
    session,
  );
  hidden = await matMulAudioWeight(session, "mm.a.input_projection.weight", hidden);
  return {
    hidden,
    tokenCount: hidden.length / manifest.projectionDim,
    durationMs: features.durationMs,
  };
}

async function subsampleConvProjection(
  session: AudioSession,
  features: AudioFeatures,
): Promise<{ hidden: Float32Array; mask: Uint8Array }> {
  const first = await conv2dSubsampleLayer(session, features.values, features.attentionMask, features.frameCount, features.featureSize, 1, 128, "a.conv1d.0");
  const second = await conv2dSubsampleLayer(session, first.values, first.mask, first.time, first.frequency, 128, 32, "a.conv1d.1");
  const flattened = await flattenChannelsLast(session, second.values, second.time, second.frequency, 32);
  return {
    hidden: await addOptionalBiasRows(
      session,
      await matMulAudioWeight(session, "a.input_projection.weight", flattened),
      "a.input_projection.bias",
    ),
    mask: second.mask,
  };
}

async function conv2dSubsampleLayer(
  session: AudioSession,
  input: Float32Array,
  mask: Uint8Array,
  time: number,
  frequency: number,
  inChannels: number,
  outChannels: number,
  prefix: string,
): Promise<{ values: Float32Array; mask: Uint8Array; time: number; frequency: number }> {
  const weight = await session.readF32Tensor(`${prefix}.weight`);
  const bias = session.hasTensor(`${prefix}.bias`) ? await session.readF32Tensor(`${prefix}.bias`) : undefined;
  const norm = await session.readF32Tensor(`${prefix}.norm.weight`);
  const outTime = Math.floor((time + 1) / 2);
  const outFrequency = Math.floor((frequency + 1) / 2);
  const wasm = audioWasmKernelsEnabled(session)
    ? await audioConv2dSubsampleWasm(input, mask, weight, bias, norm, time, frequency, inChannels, outChannels, session.epsilon)
    : undefined;
  if (wasm) {
    return {
      values: wasm,
      mask: downsampleMaskByTwo(mask, outTime),
      time: outTime,
      frequency: outFrequency,
    };
  }
  const output = new Float32Array(outTime * outChannels * outFrequency);

  for (let tOut = 0; tOut < outTime; tOut += 1) {
    for (let fOut = 0; fOut < outFrequency; fOut += 1) {
      const channelValues = new Float32Array(outChannels);
      for (let outChannel = 0; outChannel < outChannels; outChannel += 1) {
        let sum = 0;
        for (let kt = 0; kt < 3; kt += 1) {
          const tIn = tOut * 2 + kt - 1;
          if (tIn < 0 || tIn >= time || (mask[tIn] ?? 0) === 0) {
            continue;
          }
          for (let kf = 0; kf < 3; kf += 1) {
            const fIn = fOut * 2 + kf - 1;
            if (fIn < 0 || fIn >= frequency) {
              continue;
            }
            for (let inChannel = 0; inChannel < inChannels; inChannel += 1) {
              const inputValue = input[(tIn * inChannels + inChannel) * frequency + fIn] ?? 0;
              const weightValue = weight[kf + kt * 3 + inChannel * 9 + outChannel * 9 * inChannels] ?? 0;
              sum = Math.fround(sum + Math.fround(inputValue * weightValue));
            }
          }
        }
        if (bias) {
          sum = Math.fround(sum + (bias[outChannel] ?? 0));
        }
        channelValues[outChannel] = sum;
      }
      layerNormInPlace(channelValues, norm, session.epsilon);
      for (let outChannel = 0; outChannel < outChannels; outChannel += 1) {
        output[(tOut * outChannels + outChannel) * outFrequency + fOut] = Math.max(0, channelValues[outChannel] ?? 0);
      }
    }
  }

  return {
    values: output,
    mask: downsampleMaskByTwo(mask, outTime),
    time: outTime,
    frequency: outFrequency,
  };
}

async function forwardAudioLayer(
  session: AudioSession,
  input: Float32Array,
  mask: Uint8Array,
  positionEmbeddings: Float32Array,
  layer: number,
): Promise<Float32Array> {
  const manifest = session.manifest;
  let hidden = await forwardAudioFeedForward(session, input, layer, "", manifest.residualWeight);
  const attentionResidual = hidden;
  hidden = await clampValues(session, hidden, -1e10, 1e10);
  hidden = await rmsNormRows(hidden, await session.readF32Tensor(`a.blk.${layer}.attn_pre_norm.weight`), session.epsilon, session);
  hidden = await forwardAudioAttention(session, hidden, mask, positionEmbeddings, layer);
  hidden = await clampValues(session, hidden, -1e10, 1e10);
  hidden = await rmsNormRows(hidden, await session.readF32Tensor(`a.blk.${layer}.attn_post_norm.weight`), session.epsilon, session);
  hidden = await residualAdd(session, hidden, attentionResidual);
  hidden = await forwardAudioLightConv(session, hidden, layer);
  hidden = await forwardAudioFeedForward(session, hidden, layer, "_1", manifest.residualWeight);
  hidden = await clampValues(session, hidden, -1e10, 1e10);
  return rmsNormRows(hidden, await session.readF32Tensor(`a.blk.${layer}.ln2.weight`), session.epsilon, session);
}

async function forwardAudioFeedForward(
  session: AudioSession,
  input: Float32Array,
  layer: number,
  suffix: "" | "_1",
  residualWeight: number,
): Promise<Float32Array> {
  const residual = input;
  let hidden = await clampValues(session, input, -1e10, 1e10);
  hidden = await rmsNormRows(hidden, await session.readF32Tensor(`a.blk.${layer}.ffn_norm${suffix}.weight`), session.epsilon, session);
  hidden = await matMulAudioWeight(session, `a.blk.${layer}.ffn_up${suffix}.weight`, hidden);
  hidden = await siluValues(session, hidden);
  hidden = await matMulAudioWeight(session, `a.blk.${layer}.ffn_down${suffix}.weight`, hidden);
  hidden = await clampValues(session, hidden, -1e10, 1e10);
  hidden = await rmsNormRows(hidden, await session.readF32Tensor(`a.blk.${layer}.ffn_post_norm${suffix}.weight`), session.epsilon, session);
  return residualAddScale(session, residual, hidden, residualWeight);
}

async function forwardAudioAttention(
  session: AudioSession,
  input: Float32Array,
  mask: Uint8Array,
  positionEmbeddings: Float32Array,
  layer: number,
): Promise<Float32Array> {
  const manifest = session.manifest;
  const tokenCount = input.length / manifest.embeddingLength;
  const qkv = await matMulAudioWeightBatch(session, [
    `a.blk.${layer}.attn_q.weight`,
    `a.blk.${layer}.attn_k.weight`,
    `a.blk.${layer}.attn_v.weight`,
  ], input);
  const [q, k, v] = qkv ?? await Promise.all([
    matMulAudioWeight(session, `a.blk.${layer}.attn_q.weight`, input),
    matMulAudioWeight(session, `a.blk.${layer}.attn_k.weight`, input),
    matMulAudioWeight(session, `a.blk.${layer}.attn_v.weight`, input),
  ]);
  const relativeK = await matMulAudioWeight(session, `a.blk.${layer}.attn_k_rel.weight`, positionEmbeddings);
  const perDimScale = await session.readF32Tensor(`a.blk.${layer}.per_dim_scale.weight`);
  const perDimKScale = session.hasTensor(`a.blk.${layer}.per_dim_k_scale.weight`)
    ? await session.readF32Tensor(`a.blk.${layer}.per_dim_k_scale.weight`)
    : undefined;
  const output = new Float32Array(q.length);
  const qScale = Math.pow(manifest.headSize, -0.5) / Math.log(2);
  const kScale = Math.log(1 + Math.E) / Math.log(2);
  const maxPast = 12;
  const contextSize = manifest.attentionChunkSize + maxPast;
  const rpeLength = maxPast + 1;

  for (let queryToken = 0; queryToken < tokenCount; queryToken += 1) {
    if ((mask[queryToken] ?? 0) === 0) {
      continue;
    }
    const block = Math.floor(queryToken / manifest.attentionChunkSize);
    const contextStart = block * manifest.attentionChunkSize - maxPast;
    for (let head = 0; head < manifest.headCount; head += 1) {
      const scores = new Float32Array(contextSize);
      let maxScore = Number.NEGATIVE_INFINITY;
      const queryInBlock = queryToken - block * manifest.attentionChunkSize;
      for (let context = 0; context < contextSize; context += 1) {
        const keyToken = contextStart + context;
        const isInvalid = keyToken < 0 ||
          keyToken >= tokenCount ||
          keyToken > queryToken ||
          queryToken - keyToken >= maxPast ||
          (mask[keyToken] ?? 0) === 0;
        if (isInvalid) {
          scores[context] = manifest.attentionInvalidLogitsValue;
          continue;
        }
        let score = 0;
        const relIndex = context - queryInBlock;
        for (let dim = 0; dim < manifest.headSize; dim += 1) {
          const qValue = (q[(queryToken * manifest.embeddingLength) + head * manifest.headSize + dim] ?? 0) *
            qScale * (perDimScale[dim] ?? 1);
          const kValue = (k[(keyToken * manifest.embeddingLength) + head * manifest.headSize + dim] ?? 0) *
            kScale * (perDimKScale?.[dim] ?? 1);
          const relValue = relIndex >= 0 && relIndex < rpeLength
            ? relativeK[relIndex * manifest.embeddingLength + head * manifest.headSize + dim] ?? 0
            : 0;
          score += qValue * (kValue + relValue);
        }
        score = Math.tanh(score / manifest.attentionLogitCap) * manifest.attentionLogitCap;
        scores[context] = score;
        maxScore = Math.max(maxScore, score);
      }
      let sumExp = 0;
      for (let context = 0; context < contextSize; context += 1) {
        const exp = Math.exp((scores[context] ?? manifest.attentionInvalidLogitsValue) - maxScore);
        scores[context] = exp;
        sumExp += exp;
      }
      if (sumExp <= 0) {
        continue;
      }
      for (let context = 0; context < contextSize; context += 1) {
        const keyToken = contextStart + context;
        if (keyToken < 0 ||
          keyToken >= tokenCount ||
          keyToken > queryToken ||
          queryToken - keyToken >= maxPast ||
          (mask[keyToken] ?? 0) === 0) {
          continue;
        }
        const probability = (scores[context] ?? 0) / sumExp;
        for (let dim = 0; dim < manifest.headSize; dim += 1) {
          output[queryToken * manifest.embeddingLength + head * manifest.headSize + dim] +=
            probability * (v[keyToken * manifest.embeddingLength + head * manifest.headSize + dim] ?? 0);
        }
      }
    }
  }
  return addOptionalBiasRows(
    session,
    await matMulAudioWeight(session, `a.blk.${layer}.attn_out.weight`, output),
    `a.blk.${layer}.attn_out.bias`,
  );
}

async function forwardAudioLightConv(
  session: AudioSession,
  input: Float32Array,
  layer: number,
): Promise<Float32Array> {
  const manifest = session.manifest;
  const residual = input;
  let hidden = await rmsNormRows(input, await session.readF32Tensor(`a.blk.${layer}.conv_norm.weight`), session.epsilon, session);
  hidden = await matMulAudioWeight(session, `a.blk.${layer}.conv_pw1.weight`, hidden);
  hidden = await glu(session, hidden, manifest.embeddingLength);
  hidden = await depthwiseConv1d(session, hidden, await session.readF32Tensor(`a.blk.${layer}.conv_dw.weight`), manifest.convKernelSize, manifest.embeddingLength);
  if (session.hasTensor(`a.blk.${layer}.conv_dw.bias`)) {
    hidden = await addBiasRows(session, hidden, await session.readF32Tensor(`a.blk.${layer}.conv_dw.bias`));
  }
  hidden = await clampValues(session, hidden, -1e10, 1e10);
  hidden = await rmsNormRows(hidden, await session.readF32Tensor(`a.blk.${layer}.norm_conv.weight`), session.epsilon, session);
  hidden = await siluValues(session, hidden);
  hidden = await matMulAudioWeight(session, `a.blk.${layer}.conv_pw2.weight`, hidden);
  return residualAdd(session, hidden, residual);
}

async function matMulAudioWeight(
  session: AudioSession,
  weightName: string,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const clampedInput = await clampLinearInput(session, weightName, inputColumns);
  const output = isWasmQuantizedType(tensor.type)
    ? await matMulQuantizedRows(session, weightName, tensor.type, clampedInput)
    : await matMulDenseRows(session, weightName, clampedInput);
  return clampLinearOutput(session, weightName, output);
}

async function matMulAudioWeightBatch(
  session: AudioSession,
  weightNames: readonly string[],
  inputColumns: Float32Array,
): Promise<Float32Array[] | undefined> {
  if (!audioProjectionBatchingEnabled(session) || weightNames.length < 2 || weightNames.length > 4) {
    return undefined;
  }
  const tensors = weightNames.map((name) => session.getTensor(name));
  const inputSize = tensors[0]?.dimensions[0] ?? 0;
  if (inputSize <= 0 || inputColumns.length % inputSize !== 0) {
    return undefined;
  }
  const columnCount = inputColumns.length / inputSize;
  if (tensors.some((tensor) => tensor.dimensions[0] !== inputSize || !isWasmQuantizedType(tensor.type))) {
    return undefined;
  }
  const clamp = await sharedInputClamp(session, weightNames);
  if (!clamp) {
    return undefined;
  }
  const clampedInput = await clampValues(session, inputColumns, clamp.min, clamp.max);

  let outputs: Float32Array[] | undefined;
  if (audioResidentWeightCacheEnabled(session)) {
    const handles = await Promise.all(weightNames.map((name, index) => {
      const tensor = tensors[index];
      return readWasmWeightHandle(session, name, tensor?.type as "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0", inputSize, tensor?.dimensions[1] ?? 0);
    }));
    if (handles.every((handle): handle is WasmQuantizedWeightHandle => Boolean(handle))) {
      outputs = await matMulQuantizedWasmResidentMulti(handles, clampedInput, inputSize, columnCount);
      if (!outputs) {
        outputs = await Promise.all(handles.map((handle) =>
          matMulQuantizedWasmResident(handle, clampedInput, inputSize, handle.rowCount, columnCount),
        )) as Float32Array[];
      }
    }
  }

  if (!outputs && audioWasmKernelsEnabled(session)) {
    const weights: QuantizedMatMulInput[] = [];
    for (let index = 0; index < weightNames.length; index += 1) {
      const name = weightNames[index];
      const tensor = tensors[index];
      if (!name || !tensor || !isWasmQuantizedType(tensor.type)) {
        return undefined;
      }
      weights.push({
        type: tensor.type,
        weightBytes: await session.readWeightBytes(name),
        rowCount: tensor.dimensions[1] ?? 0,
      });
    }
    outputs = await matMulQuantizedMultiWasm(weights, clampedInput, inputSize, columnCount);
  }
  if (!outputs) {
    return undefined;
  }
  return Promise.all(outputs.map((output, index) =>
    clampLinearOutput(session, weightNames[index] ?? "", output),
  ));
}

async function sharedInputClamp(
  session: AudioSession,
  weightNames: readonly string[],
): Promise<{ min: number; max: number } | undefined> {
  let shared: { min: number; max: number } | undefined;
  for (const weightName of weightNames) {
    const clamp = await readClamp(session, weightName.replace(/\.weight$/, ".input_min"), weightName.replace(/\.weight$/, ".input_max"));
    if (!clamp) {
      return undefined;
    }
    if (!shared) {
      shared = clamp;
    } else if (shared.min !== clamp.min || shared.max !== clamp.max) {
      return undefined;
    }
  }
  return shared;
}

async function matMulDenseRows(
  session: AudioSession,
  weightName: string,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  if (!Number.isInteger(columnCount) || inputSize <= 0) {
    throw new Error(`${weightName} input shape mismatch: ${inputColumns.length}`);
  }
  const weightBytes = await session.readWeightBytes(weightName);
  const rowByteLength = tensorByteLength({ ...tensor, dimensions: [inputSize] });
  const output = new Float32Array(rowCount * columnCount);
  for (let row = 0; row < rowCount; row += 1) {
    const weight = dequantizeRow(tensor.type, weightBytes.subarray(row * rowByteLength, (row + 1) * rowByteLength), inputSize);
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

async function matMulQuantizedRows(
  session: AudioSession,
  weightName: string,
  type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0",
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  if (!Number.isInteger(columnCount) || inputSize <= 0) {
    throw new Error(`${weightName} input shape mismatch: ${inputColumns.length}`);
  }
  const handle = audioResidentWeightCacheEnabled(session)
    ? await readWasmWeightHandle(session, weightName, type, inputSize, rowCount)
    : undefined;
  const resident = handle
    ? await matMulQuantizedWasmResident(handle, inputColumns, inputSize, rowCount, columnCount)
    : undefined;
  if (resident) {
    return resident;
  }
  const weightBytes = await session.readWeightBytes(weightName);
  const wasm = audioWasmKernelsEnabled(session)
    ? await matMulQuantizedWasm(type, weightBytes, inputColumns, inputSize, rowCount, columnCount)
    : undefined;
  if (wasm) {
    return wasm;
  }
  const rowByteLength = tensorByteLength({ ...tensor, dimensions: [inputSize] });
  const output = new Float32Array(rowCount * columnCount);
  for (let row = 0; row < rowCount; row += 1) {
    const weight = dequantizeRow(type, weightBytes.subarray(row * rowByteLength, (row + 1) * rowByteLength), inputSize);
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

async function clampLinearInput(
  session: AudioSession,
  weightName: string,
  input: Float32Array,
): Promise<Float32Array> {
  return clampTensor(session, input, weightName.replace(/\.weight$/, ".input_min"), weightName.replace(/\.weight$/, ".input_max"));
}

async function clampLinearOutput(
  session: AudioSession,
  weightName: string,
  input: Float32Array,
): Promise<Float32Array> {
  return clampTensor(session, input, weightName.replace(/\.weight$/, ".output_min"), weightName.replace(/\.weight$/, ".output_max"));
}

async function clampTensor(
  session: AudioSession,
  input: Float32Array,
  minTensorName: string,
  maxTensorName: string,
): Promise<Float32Array> {
  const clamp = await readClamp(session, minTensorName, maxTensorName);
  if (!clamp) {
    return input;
  }
  return clampValues(session, input, clamp.min, clamp.max);
}

async function readClamp(
  session: AudioSession,
  minTensorName: string,
  maxTensorName: string,
): Promise<{ min: number; max: number } | undefined> {
  if (!session.hasTensor(minTensorName) || !session.hasTensor(maxTensorName)) {
    return undefined;
  }
  return {
    min: (await session.readF32Tensor(minTensorName))[0] ?? -Infinity,
    max: (await session.readF32Tensor(maxTensorName))[0] ?? Infinity,
  };
}

function isWasmQuantizedType(type: GgmlTypeName): type is "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0" {
  return type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "IQ4_XS" || type === "Q8_0";
}

function audioRelativePositionEmbeddings(manifest: AudioManifest): Float32Array {
  const maxPast = 12;
  const count = maxPast + 1;
  const output = new Float32Array(count * manifest.embeddingLength);
  const numTimescales = manifest.embeddingLength / 2;
  const increment = Math.log(10000) / Math.max(numTimescales - 1, 1);
  for (let position = 0; position < count; position += 1) {
    const positionId = maxPast - position;
    for (let index = 0; index < numTimescales; index += 1) {
      const scaled = positionId * Math.exp(-index * increment);
      output[position * manifest.embeddingLength + index] = Math.sin(scaled);
      output[position * manifest.embeddingLength + numTimescales + index] = Math.cos(scaled);
    }
  }
  return output;
}

async function flattenChannelsLast(
  session: AudioSession,
  input: Float32Array,
  timeCount: number,
  frequencyCount: number,
  channelCount: number,
): Promise<Float32Array> {
  const wasm = audioWasmKernelsEnabled(session)
    ? await audioFlattenChannelsLastWasm(input, timeCount, frequencyCount, channelCount)
    : undefined;
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(timeCount * frequencyCount * channelCount);
  for (let time = 0; time < timeCount; time += 1) {
    for (let frequency = 0; frequency < frequencyCount; frequency += 1) {
      for (let channel = 0; channel < channelCount; channel += 1) {
        output[time * frequencyCount * channelCount + frequency * channelCount + channel] =
          input[(time * channelCount + channel) * frequencyCount + frequency] ?? 0;
      }
    }
  }
  return output;
}

async function depthwiseConv1d(
  session: AudioSession,
  input: Float32Array,
  weight: Float32Array,
  kernelSize: number,
  channels: number,
): Promise<Float32Array> {
  const wasm = audioWasmKernelsEnabled(session)
    ? await audioDepthwiseConv1dWasm(input, weight, kernelSize, channels)
    : undefined;
  if (wasm) {
    return wasm;
  }
  const tokenCount = input.length / channels;
  const output = new Float32Array(input.length);
  const leftPad = kernelSize - 1;
  for (let token = 0; token < tokenCount; token += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      let sum = 0;
      for (let kernel = 0; kernel < kernelSize; kernel += 1) {
        const source = token + kernel - leftPad;
        if (source < 0 || source >= tokenCount) {
          continue;
        }
        sum += (input[source * channels + channel] ?? 0) * (weight[kernel + channel * kernelSize] ?? 0);
      }
      output[token * channels + channel] = sum;
    }
  }
  return output;
}

async function glu(session: AudioSession, input: Float32Array, outputSize: number): Promise<Float32Array> {
  const wasm = audioWasmKernelsEnabled(session) ? await audioGluWasm(input, outputSize) : undefined;
  if (wasm) {
    return wasm;
  }
  const tokenCount = input.length / (outputSize * 2);
  const output = new Float32Array(tokenCount * outputSize);
  for (let token = 0; token < tokenCount; token += 1) {
    const offset = token * outputSize * 2;
    for (let index = 0; index < outputSize; index += 1) {
      output[token * outputSize + index] = (input[offset + index] ?? 0) *
        sigmoidScalar(input[offset + outputSize + index] ?? 0);
    }
  }
  return output;
}

async function addBiasRows(session: AudioSession, input: Float32Array, bias: Float32Array): Promise<Float32Array> {
  const wasm = audioWasmKernelsEnabled(session) ? await audioAddBiasRowsWasm(input, bias) : undefined;
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(input);
  for (let token = 0; token < input.length / bias.length; token += 1) {
    for (let index = 0; index < bias.length; index += 1) {
      output[token * bias.length + index] += bias[index] ?? 0;
    }
  }
  return output;
}

async function addOptionalBiasRows(session: AudioSession, input: Float32Array, biasName: string): Promise<Float32Array> {
  if (!session.hasTensor(biasName)) {
    return input;
  }
  return addBiasRows(session, input, await session.readF32Tensor(biasName));
}

async function optionalTensor(session: AudioSession, name: string, size: number): Promise<Float32Array> {
  if (session.hasTensor(name)) {
    return session.readF32Tensor(name);
  }
  const output = new Float32Array(size);
  output.fill(1);
  return output;
}

async function rmsNormRows(
  input: Float32Array,
  weight: Float32Array,
  epsilon: number,
  session?: AudioSession,
): Promise<Float32Array> {
  const wasm = session && audioWasmKernelsEnabled(session)
    ? await audioRmsNormWasm(input, weight, epsilon)
    : undefined;
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(input.length);
  for (let row = 0; row < input.length / weight.length; row += 1) {
    const offset = row * weight.length;
    let sumSquares = 0;
    for (let index = 0; index < weight.length; index += 1) {
      const value = input[offset + index] ?? 0;
      sumSquares += value * value;
    }
    const scale = 1 / Math.sqrt(sumSquares / weight.length + epsilon);
    for (let index = 0; index < weight.length; index += 1) {
      output[offset + index] = Math.fround((input[offset + index] ?? 0) * scale * (weight[index] ?? 0));
    }
  }
  return output;
}

function layerNormInPlace(values: Float32Array, weight: Float32Array, epsilon: number): void {
  let mean = 0;
  for (const value of values) {
    mean += value;
  }
  mean /= values.length;
  let variance = 0;
  for (const value of values) {
    const centered = value - mean;
    variance += centered * centered;
  }
  const scale = 1 / Math.sqrt(variance / values.length + epsilon);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = Math.fround(((values[index] ?? 0) - mean) * scale * (weight[index] ?? 1));
  }
}

async function residualAdd(session: AudioSession, left: Float32Array, right: Float32Array): Promise<Float32Array> {
  if (left.length !== right.length) {
    throw new Error(`Residual shape mismatch: ${left.length} != ${right.length}`);
  }
  const wasm = audioWasmKernelsEnabled(session) ? await audioResidualAddWasm(left, right) : undefined;
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(left.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Math.fround((left[index] ?? 0) + (right[index] ?? 0));
  }
  return output;
}

async function residualAddScale(session: AudioSession, residual: Float32Array, hidden: Float32Array, scale: number): Promise<Float32Array> {
  if (residual.length !== hidden.length) {
    throw new Error(`Residual shape mismatch: ${residual.length} != ${hidden.length}`);
  }
  const wasm = audioWasmKernelsEnabled(session) ? await audioResidualAddScaleWasm(residual, hidden, scale) : undefined;
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(residual.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Math.fround((residual[index] ?? 0) + Math.fround((hidden[index] ?? 0) * scale));
  }
  return output;
}

async function clampValues(session: AudioSession, input: Float32Array, min: number, max: number): Promise<Float32Array> {
  const wasm = audioWasmKernelsEnabled(session) ? await audioClampWasm(input, min, max) : undefined;
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = Math.min(max, Math.max(min, input[index] ?? 0));
  }
  return output;
}

async function siluValues(session: AudioSession, input: Float32Array): Promise<Float32Array> {
  const wasm = audioWasmKernelsEnabled(session) ? await audioSiluWasm(input) : undefined;
  if (wasm) {
    return wasm;
  }
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    const value = input[index] ?? 0;
    output[index] = value / (1 + Math.exp(-value));
  }
  return output;
}

function sigmoidScalar(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function downsampleMaskByTwo(mask: Uint8Array, outputLength: number): Uint8Array {
  const output = new Uint8Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    output[index] = mask[index * 2] ?? 0;
  }
  return output;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Generation was aborted.", "AbortError");
  }
}
