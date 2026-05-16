import {
  buildAudioManifest,
  type AudioManifest,
} from "./model";
import type {
  CacheStats,
  ExecutionProviderConfig,
  ExecutionProviderStats,
} from "./runtime";
import {
  resolveSessionExecutionProviders,
  resolvePreprocessProviders,
} from "./runtime";
import {
  dispatchAudioEncoder,
  dispatchAudioPreprocessor,
} from "./runner/audio";
import type {
  AudioRunners,
  AudioPreprocessOptions,
} from "./runner/audio-runner";
import type {
  RunnerProvider,
} from "./runner/provider";
import {
  GgufTensorReader,
} from "./tensor-reader";

export type {
  AudioPreprocessOptions,
} from "./runner/audio-runner";

export type AudioPcmInput = {
  pcm: Float32Array;
  sampleRate: 16000;
  durationMs: number;
};

export type AudioFeatures = {
  values: Float32Array;
  frameCount: number;
  featureSize: number;
  attentionMask: Uint8Array;
  durationMs: number;
};

export type AudioEncodeResult = {
  hidden: Float32Array;
  tokenCount: number;
  durationMs: number;
};

export type AudioSessionOptions = {
  maxWeightCacheBytes?: number;
  executionProviders?: readonly ExecutionProviderConfig[];
  preprocessProviders?: readonly ExecutionProviderConfig[];
  runnerProviders?: readonly RunnerProvider[];
  audioRunners?: readonly AudioRunners[];
};

export class AudioSession {
  readonly tensorReader: GgufTensorReader;
  readonly manifest: AudioManifest;
  readonly epsilon: number;
  readonly executionProviders: readonly ExecutionProviderConfig[];
  readonly preprocessProviders: readonly ExecutionProviderConfig[];
  readonly audioRunners: readonly AudioRunners[];

  private readonly maxWeightCacheBytes: number;
  private readonly f32TensorCache = new Map<string, Float32Array>();
  private readonly weightBytesCache = new Map<string, Uint8Array>();
  private readonly executionProviderStatsProviders = new Map<string, () => ExecutionProviderStats>();
  private readonly disposeCallbacks = new Set<() => void>();
  private weightCacheBytes = 0;
  private weightCacheHits = 0;
  private weightCacheMisses = 0;
  private weightCacheEvictions = 0;

  constructor(tensorReader: GgufTensorReader, options: AudioSessionOptions = {}) {
    this.tensorReader = tensorReader;
    this.manifest = buildAudioManifest(tensorReader.metadata);
    this.epsilon = this.manifest.layerNormEpsilon;
    this.maxWeightCacheBytes = options.maxWeightCacheBytes ?? 256 * 1024 * 1024;
    this.executionProviders = resolveSessionExecutionProviders(options);
    this.preprocessProviders = resolvePreprocessProviders(this.executionProviders, options.preprocessProviders);
    this.audioRunners = options.audioRunners ?? createAudioRunners(options.runnerProviders ?? []);
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

export function createAudioSession(
  tensorReader: GgufTensorReader,
  options: AudioSessionOptions = {},
): AudioSession {
  return new AudioSession(tensorReader, options);
}

export function preprocessAudioPcm(
  audio: AudioPcmInput,
  manifest?: Pick<AudioManifest, "sampleRate" | "maxSeconds" | "frameLength" | "hopLength" | "fftLength" | "featureSize" | "melFloor">,
): AudioFeatures {
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
    throw new Error(` audio expects ${config.sampleRate} Hz PCM, got ${audio.sampleRate}`);
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

export async function runAudioPreprocessor(
  session: AudioSession,
  audio: AudioPcmInput,
  options: AudioPreprocessOptions = {},
): Promise<AudioFeatures> {
  return dispatchAudioPreprocessor(session.audioRunners, session, audio, preprocessAudioPcm, options);
}

export async function runAudioEncoder(
  session: AudioSession,
  features: AudioFeatures,
  options: { signal?: AbortSignal } = {},
): Promise<AudioEncodeResult> {
  return dispatchAudioEncoder(session.audioRunners, session, features, options);
}

function createAudioRunners(providers: readonly RunnerProvider[]): readonly AudioRunners[] {
  return providers
    .map((provider) => provider.createAudioRunners?.())
    .filter((runner): runner is AudioRunners => runner !== undefined);
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
