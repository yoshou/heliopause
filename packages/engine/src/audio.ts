import {
  buildGemma4AudioManifest,
  type Gemma4AudioManifest,
} from "./model";
import {
  dequantizeRow,
} from "./quant";
import type {
  CacheStats,
  ExecutionProviderConfig,
  ExecutionProviderStats,
} from "./runtime";
import {
  GgufTensorReader,
  tensorByteLength,
} from "./tensor-reader";

export type Gemma4AudioPcmInput = {
  pcm: Float32Array;
  sampleRate: 16000;
  durationMs: number;
};

export type Gemma4AudioFeatures = {
  values: Float32Array;
  frameCount: number;
  featureSize: number;
  attentionMask: Uint8Array;
  durationMs: number;
};

export type Gemma4AudioEncodeResult = {
  hidden: Float32Array;
  tokenCount: number;
  durationMs: number;
};

export type Gemma4AudioSessionOptions = {
  maxWeightCacheBytes?: number;
  executionProviders?: readonly ExecutionProviderConfig[];
};

export class Gemma4AudioSession {
  readonly tensorReader: GgufTensorReader;
  readonly manifest: Gemma4AudioManifest;
  readonly epsilon: number;
  readonly executionProviders: readonly ExecutionProviderConfig[];

  private readonly maxWeightCacheBytes: number;
  private readonly f32TensorCache = new Map<string, Float32Array>();
  private readonly weightBytesCache = new Map<string, Uint8Array>();
  private readonly executionProviderStatsProviders = new Map<string, () => ExecutionProviderStats>();
  private readonly disposeCallbacks = new Set<() => void>();
  private weightCacheBytes = 0;
  private weightCacheHits = 0;
  private weightCacheMisses = 0;
  private weightCacheEvictions = 0;

  constructor(tensorReader: GgufTensorReader, options: Gemma4AudioSessionOptions = {}) {
    this.tensorReader = tensorReader;
    this.manifest = buildGemma4AudioManifest(tensorReader.metadata);
    this.epsilon = this.manifest.layerNormEpsilon;
    this.maxWeightCacheBytes = options.maxWeightCacheBytes ?? 256 * 1024 * 1024;
    this.executionProviders = (options.executionProviders ?? [{ name: "cpu" }]).map((provider) => ({
      name: provider.name,
      options: provider.options ? { ...provider.options } : undefined,
    }));
  }

  getTensor(name: string) {
    return this.tensorReader.getTensor(name);
  }

  hasTensor(name: string): boolean {
    return this.tensorReader.metadata.tensors.some((tensor) => tensor.name === name);
  }

  async readF32Tensor(name: string): Promise<Float32Array> {
    const cached = this.f32TensorCache.get(name);
    if (cached) {
      return cached;
    }
    const tensor = this.getTensor(name);
    if (tensor.type !== "F32") {
      throw new Error(`${name} must be F32, got ${tensor.type}`);
    }
    const bytes = await this.tensorReader.readTensorBytes(name);
    const value = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
    this.f32TensorCache.set(name, value);
    return value;
  }

  async readWeightBytes(name: string): Promise<Uint8Array> {
    const cached = this.weightBytesCache.get(name);
    if (cached) {
      this.weightCacheHits += 1;
      this.weightBytesCache.delete(name);
      this.weightBytesCache.set(name, cached);
      return cached;
    }
    this.weightCacheMisses += 1;
    const bytes = await this.tensorReader.readTensorBytes(name);
    if (bytes.byteLength <= this.maxWeightCacheBytes) {
      this.weightBytesCache.set(name, bytes);
      this.weightCacheBytes += bytes.byteLength;
      this.evictWeightBytes();
    }
    return bytes;
  }

  cacheStats(): CacheStats {
    return {
      f32TensorCount: this.f32TensorCache.size,
      weightTensorCount: this.weightBytesCache.size,
      weightCacheBytes: this.weightCacheBytes,
      maxWeightCacheBytes: this.maxWeightCacheBytes,
      weightCacheHits: this.weightCacheHits,
      weightCacheMisses: this.weightCacheMisses,
      weightCacheEvictions: this.weightCacheEvictions,
      embeddingRowCount: 0,
      executionProviderStats: this.executionProviderStats(),
    };
  }

  executionProvider(name: string): ExecutionProviderConfig | undefined {
    return this.executionProviders.find((provider) => provider.name === name);
  }

  setExecutionProviderStatsProvider(
    provider: (() => ExecutionProviderStats) | undefined,
    name = "default",
  ): void {
    if (!provider) {
      this.executionProviderStatsProviders.delete(name);
      return;
    }
    this.executionProviderStatsProviders.set(name, provider);
  }

  addDisposeCallback(callback: () => void): void {
    this.disposeCallbacks.add(callback);
  }

  dispose(): void {
    for (const callback of this.disposeCallbacks) {
      callback();
    }
    this.disposeCallbacks.clear();
    this.executionProviderStatsProviders.clear();
    this.f32TensorCache.clear();
    this.weightBytesCache.clear();
    this.weightCacheBytes = 0;
  }

  private executionProviderStats(): ExecutionProviderStats {
    const stats: ExecutionProviderStats = {};
    for (const provider of this.executionProviderStatsProviders.values()) {
      Object.assign(stats, provider());
    }
    return stats;
  }

  private evictWeightBytes(): void {
    while (this.weightCacheBytes > this.maxWeightCacheBytes) {
      const oldest = this.weightBytesCache.entries().next().value as [string, Uint8Array] | undefined;
      if (!oldest) {
        this.weightCacheBytes = 0;
        return;
      }
      this.weightBytesCache.delete(oldest[0]);
      this.weightCacheBytes -= oldest[1].byteLength;
      this.weightCacheEvictions += 1;
    }
  }
}

export function createGemma4AudioSession(
  tensorReader: GgufTensorReader,
  options: Gemma4AudioSessionOptions = {},
): Gemma4AudioSession {
  return new Gemma4AudioSession(tensorReader, options);
}

export function preprocessGemma4AudioPcm(
  audio: Gemma4AudioPcmInput,
  manifest?: Pick<Gemma4AudioManifest, "sampleRate" | "maxSeconds" | "frameLength" | "hopLength" | "fftLength" | "featureSize" | "melFloor">,
): Gemma4AudioFeatures {
  const config = {
    sampleRate: 16000,
    maxSeconds: 30,
    frameLength: 320,
    hopLength: 160,
    fftLength: 512,
    featureSize: 128,
    melFloor: 0.001,
    ...manifest,
  };
  if (audio.sampleRate !== config.sampleRate) {
    throw new Error(`Gemma4 audio expects ${config.sampleRate} Hz PCM, got ${audio.sampleRate}`);
  }

  const maxSamples = config.sampleRate * config.maxSeconds;
  const sampleCount = Math.min(audio.pcm.length, maxSamples);
  const pcm = audio.pcm.subarray(0, sampleCount);
  const chunkSamples = config.sampleRate * config.maxSeconds;
  const window = hannWindow(config.frameLength, config.fftLength);
  const filters = melFilterBank(config.sampleRate, config.fftLength, config.featureSize);
  const fftReal = new Float32Array(config.fftLength);
  const fftImag = new Float32Array(config.fftLength);
  const magnitude = new Float32Array(config.fftLength / 2 + 1);
  const chunks: Float32Array[] = [];
  let frameCount = 0;

  for (let chunkOffset = 0; chunkOffset < sampleCount; chunkOffset += chunkSamples) {
    const chunkLength = Math.min(chunkSamples, sampleCount - chunkOffset);
    const padLeft = Math.floor(config.frameLength / 2);
    const nWithLeft = chunkLength + padLeft;
    const chunkFrameCount = Math.floor((nWithLeft - (config.frameLength + 1)) / config.hopLength) + 1;
    if (chunkFrameCount <= 0) {
      continue;
    }
    const paddedNeeded = (chunkFrameCount - 1) * config.hopLength + config.fftLength;
    const totalPad = Math.max(paddedNeeded - chunkLength, padLeft);
    const padded = new Float32Array(totalPad + chunkLength);
    padded.set(pcm.subarray(chunkOffset, chunkOffset + chunkLength), padLeft);
    const chunkValues = new Float32Array(chunkFrameCount * config.featureSize);

    for (let frame = 0; frame < chunkFrameCount; frame += 1) {
      fftReal.fill(0);
      fftImag.fill(0);
      const sampleOffset = frame * config.hopLength;
      for (let index = 0; index < config.fftLength; index += 1) {
        fftReal[index] = (padded[sampleOffset + index] ?? 0) * (window[index] ?? 0);
      }
      fftRadix2(fftReal, fftImag);
      for (let bin = 0; bin < magnitude.length; bin += 1) {
        const real = fftReal[bin] ?? 0;
        const imag = fftImag[bin] ?? 0;
        magnitude[bin] = Math.sqrt(real * real + imag * imag);
      }
      for (let mel = 0; mel < config.featureSize; mel += 1) {
        let energy = 0;
        const filterOffset = mel * magnitude.length;
        for (let bin = 0; bin < magnitude.length; bin += 1) {
          energy += (magnitude[bin] ?? 0) * (filters[filterOffset + bin] ?? 0);
        }
        chunkValues[frame * config.featureSize + mel] = Math.log(Math.max(energy, config.melFloor));
      }
    }

    chunks.push(chunkValues);
    frameCount += chunkFrameCount;
  }

  const values = new Float32Array(frameCount * config.featureSize);
  const attentionMask = new Uint8Array(frameCount);
  let outputOffset = 0;
  for (const chunk of chunks) {
    values.set(chunk, outputOffset);
    outputOffset += chunk.length;
  }
  attentionMask.fill(1);

  return {
    values,
    frameCount,
    featureSize: config.featureSize,
    attentionMask,
    durationMs: Math.min(audio.durationMs, config.maxSeconds * 1000),
  };
}

export async function runGemma4AudioEncoder(
  session: Gemma4AudioSession,
  features: Gemma4AudioFeatures,
  options: { signal?: AbortSignal } = {},
): Promise<Gemma4AudioEncodeResult> {
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
    hidden = addBiasRows(hidden, await session.readF32Tensor("a.pre_encode.out.bias"));
  }
  hidden = await rmsNormRows(
    hidden,
    await optionalTensor(session, "mm.a.soft_emb_norm.weight", manifest.outputProjectionDim),
    session.epsilon,
  );
  hidden = await matMulAudioWeight(session, "mm.a.input_projection.weight", hidden);
  return {
    hidden,
    tokenCount: hidden.length / manifest.projectionDim,
    durationMs: features.durationMs,
  };
}

async function subsampleConvProjection(
  session: Gemma4AudioSession,
  features: Gemma4AudioFeatures,
): Promise<{ hidden: Float32Array; mask: Uint8Array }> {
  const first = await conv2dSubsampleLayer(session, features.values, features.attentionMask, features.frameCount, features.featureSize, 1, 128, "a.conv1d.0");
  const second = await conv2dSubsampleLayer(session, first.values, first.mask, first.time, first.frequency, 128, 32, "a.conv1d.1");
  const flattened = new Float32Array(second.time * second.frequency * 32);
  for (let time = 0; time < second.time; time += 1) {
    for (let frequency = 0; frequency < second.frequency; frequency += 1) {
      for (let channel = 0; channel < 32; channel += 1) {
        flattened[time * second.frequency * 32 + frequency * 32 + channel] =
          second.values[(time * 32 + channel) * second.frequency + frequency] ?? 0;
      }
    }
  }
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
  session: Gemma4AudioSession,
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
  session: Gemma4AudioSession,
  input: Float32Array,
  mask: Uint8Array,
  positionEmbeddings: Float32Array,
  layer: number,
): Promise<Float32Array> {
  const manifest = session.manifest;
  let hidden = await forwardAudioFeedForward(session, input, layer, "", manifest.residualWeight);
  const attentionResidual = hidden;
  hidden = clampValues(hidden, -1e10, 1e10);
  hidden = await rmsNormRows(hidden, await session.readF32Tensor(`a.blk.${layer}.attn_pre_norm.weight`), session.epsilon);
  hidden = await forwardAudioAttention(session, hidden, mask, positionEmbeddings, layer);
  hidden = clampValues(hidden, -1e10, 1e10);
  hidden = await rmsNormRows(hidden, await session.readF32Tensor(`a.blk.${layer}.attn_post_norm.weight`), session.epsilon);
  hidden = residualAdd(hidden, attentionResidual);
  hidden = await forwardAudioLightConv(session, hidden, layer);
  hidden = await forwardAudioFeedForward(session, hidden, layer, "_1", manifest.residualWeight);
  hidden = clampValues(hidden, -1e10, 1e10);
  return rmsNormRows(hidden, await session.readF32Tensor(`a.blk.${layer}.ln2.weight`), session.epsilon);
}

async function forwardAudioFeedForward(
  session: Gemma4AudioSession,
  input: Float32Array,
  layer: number,
  suffix: "" | "_1",
  residualWeight: number,
): Promise<Float32Array> {
  const residual = input;
  let hidden = clampValues(input, -1e10, 1e10);
  hidden = await rmsNormRows(hidden, await session.readF32Tensor(`a.blk.${layer}.ffn_norm${suffix}.weight`), session.epsilon);
  hidden = await matMulAudioWeight(session, `a.blk.${layer}.ffn_up${suffix}.weight`, hidden);
  hidden = siluValues(hidden);
  hidden = await matMulAudioWeight(session, `a.blk.${layer}.ffn_down${suffix}.weight`, hidden);
  hidden = clampValues(hidden, -1e10, 1e10);
  hidden = await rmsNormRows(hidden, await session.readF32Tensor(`a.blk.${layer}.ffn_post_norm${suffix}.weight`), session.epsilon);
  return residualAddScale(residual, hidden, residualWeight);
}

async function forwardAudioAttention(
  session: Gemma4AudioSession,
  input: Float32Array,
  mask: Uint8Array,
  positionEmbeddings: Float32Array,
  layer: number,
): Promise<Float32Array> {
  const manifest = session.manifest;
  const tokenCount = input.length / manifest.embeddingLength;
  const q = await matMulAudioWeight(session, `a.blk.${layer}.attn_q.weight`, input);
  const k = await matMulAudioWeight(session, `a.blk.${layer}.attn_k.weight`, input);
  const v = await matMulAudioWeight(session, `a.blk.${layer}.attn_v.weight`, input);
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
  session: Gemma4AudioSession,
  input: Float32Array,
  layer: number,
): Promise<Float32Array> {
  const manifest = session.manifest;
  const residual = input;
  let hidden = await rmsNormRows(input, await session.readF32Tensor(`a.blk.${layer}.conv_norm.weight`), session.epsilon);
  hidden = await matMulAudioWeight(session, `a.blk.${layer}.conv_pw1.weight`, hidden);
  hidden = glu(hidden, manifest.embeddingLength);
  hidden = depthwiseConv1d(hidden, await session.readF32Tensor(`a.blk.${layer}.conv_dw.weight`), manifest.convKernelSize, manifest.embeddingLength);
  if (session.hasTensor(`a.blk.${layer}.conv_dw.bias`)) {
    hidden = addBiasRows(hidden, await session.readF32Tensor(`a.blk.${layer}.conv_dw.bias`));
  }
  hidden = clampValues(hidden, -1e10, 1e10);
  hidden = await rmsNormRows(hidden, await session.readF32Tensor(`a.blk.${layer}.norm_conv.weight`), session.epsilon);
  hidden = siluValues(hidden);
  hidden = await matMulAudioWeight(session, `a.blk.${layer}.conv_pw2.weight`, hidden);
  return residualAdd(hidden, residual);
}

async function matMulAudioWeight(
  session: Gemma4AudioSession,
  weightName: string,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const clampedInput = await clampLinearInput(session, weightName, inputColumns);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  if (!Number.isInteger(columnCount)) {
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
        sum = Math.fround(sum + Math.fround((weight[index] ?? 0) * (clampedInput[inputOffset + index] ?? 0)));
      }
      output[column * rowCount + row] = sum;
    }
  }
  return clampLinearOutput(session, weightName, output);
}

async function clampLinearInput(
  session: Gemma4AudioSession,
  weightName: string,
  input: Float32Array,
): Promise<Float32Array> {
  return clampTensor(session, input, weightName.replace(/\.weight$/, ".input_min"), weightName.replace(/\.weight$/, ".input_max"));
}

async function clampLinearOutput(
  session: Gemma4AudioSession,
  weightName: string,
  input: Float32Array,
): Promise<Float32Array> {
  return clampTensor(session, input, weightName.replace(/\.weight$/, ".output_min"), weightName.replace(/\.weight$/, ".output_max"));
}

async function clampTensor(
  session: Gemma4AudioSession,
  input: Float32Array,
  minTensorName: string,
  maxTensorName: string,
): Promise<Float32Array> {
  if (!session.hasTensor(minTensorName) || !session.hasTensor(maxTensorName)) {
    return input;
  }
  const min = (await session.readF32Tensor(minTensorName))[0] ?? -Infinity;
  const max = (await session.readF32Tensor(maxTensorName))[0] ?? Infinity;
  return clampValues(input, min, max);
}

function audioRelativePositionEmbeddings(manifest: Gemma4AudioManifest): Float32Array {
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

function depthwiseConv1d(input: Float32Array, weight: Float32Array, kernelSize: number, channels: number): Float32Array {
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

function glu(input: Float32Array, outputSize: number): Float32Array {
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

function addBiasRows(input: Float32Array, bias: Float32Array): Float32Array {
  const output = new Float32Array(input);
  for (let token = 0; token < input.length / bias.length; token += 1) {
    for (let index = 0; index < bias.length; index += 1) {
      output[token * bias.length + index] += bias[index] ?? 0;
    }
  }
  return output;
}

async function addOptionalBiasRows(session: Gemma4AudioSession, input: Float32Array, biasName: string): Promise<Float32Array> {
  if (!session.hasTensor(biasName)) {
    return input;
  }
  return addBiasRows(input, await session.readF32Tensor(biasName));
}

async function optionalTensor(session: Gemma4AudioSession, name: string, size: number): Promise<Float32Array> {
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
): Promise<Float32Array> {
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

function residualAdd(left: Float32Array, right: Float32Array): Float32Array {
  if (left.length !== right.length) {
    throw new Error(`Residual shape mismatch: ${left.length} != ${right.length}`);
  }
  const output = new Float32Array(left.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Math.fround((left[index] ?? 0) + (right[index] ?? 0));
  }
  return output;
}

function residualAddScale(residual: Float32Array, hidden: Float32Array, scale: number): Float32Array {
  if (residual.length !== hidden.length) {
    throw new Error(`Residual shape mismatch: ${residual.length} != ${hidden.length}`);
  }
  const output = new Float32Array(residual.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Math.fround((residual[index] ?? 0) + Math.fround((hidden[index] ?? 0) * scale));
  }
  return output;
}

function clampValues(input: Float32Array, min: number, max: number): Float32Array {
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = Math.min(max, Math.max(min, input[index] ?? 0));
  }
  return output;
}

function siluValues(input: Float32Array): Float32Array {
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

function hannWindow(length: number, fftLength: number): Float32Array {
  const output = new Float32Array(fftLength);
  for (let index = 0; index < length; index += 1) {
    output[index] = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / length);
  }
  return output;
}

function melFilterBank(sampleRate: number, fftLength: number, filterCount: number): Float32Array {
  const binCount = fftLength / 2 + 1;
  const maxMel = hzToMel(sampleRate / 2);
  const points = Array.from({ length: filterCount + 2 }, (_, index) => melToHz((index / (filterCount + 1)) * maxMel));
  const output = new Float32Array(filterCount * binCount);
  const binHzStep = sampleRate / fftLength;
  for (let filter = 0; filter < filterCount; filter += 1) {
    const left = points[filter] ?? 0;
    const center = points[filter + 1] ?? 0;
    const right = points[filter + 2] ?? 0;
    const leftDenominator = Math.max(1e-30, center - left);
    const rightDenominator = Math.max(1e-30, right - center);
    for (let bin = 0; bin < binCount; bin += 1) {
      const hz = bin * binHzStep;
      let weight = 0;
      if (hz >= left && hz <= center) {
        weight = (hz - left) / leftDenominator;
      } else if (hz > center && hz <= right) {
        weight = (right - hz) / rightDenominator;
      }
      output[filter * binCount + bin] = weight;
    }
  }
  return output;
}

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

function fftRadix2(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  let j = 0;
  for (let i = 1; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      const realI = real[i] ?? 0;
      const imagI = imag[i] ?? 0;
      real[i] = real[j] ?? 0;
      imag[i] = imag[j] ?? 0;
      real[j] = realI;
      imag[j] = imagI;
    }
  }
  for (let length = 2; length <= n; length <<= 1) {
    const angle = -2 * Math.PI / length;
    const wLenReal = Math.cos(angle);
    const wLenImag = Math.sin(angle);
    for (let i = 0; i < n; i += length) {
      let wReal = 1;
      let wImag = 0;
      for (let k = 0; k < length / 2; k += 1) {
        const even = i + k;
        const odd = even + length / 2;
        const oddReal = Math.fround((real[odd] ?? 0) * wReal - (imag[odd] ?? 0) * wImag);
        const oddImag = Math.fround((real[odd] ?? 0) * wImag + (imag[odd] ?? 0) * wReal);
        real[odd] = Math.fround((real[even] ?? 0) - oddReal);
        imag[odd] = Math.fround((imag[even] ?? 0) - oddImag);
        real[even] = Math.fround((real[even] ?? 0) + oddReal);
        imag[even] = Math.fround((imag[even] ?? 0) + oddImag);
        const nextReal = Math.fround(wReal * wLenReal - wImag * wLenImag);
        wImag = Math.fround(wReal * wLenImag + wImag * wLenReal);
        wReal = nextReal;
      }
    }
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Generation was aborted.", "AbortError");
  }
}
