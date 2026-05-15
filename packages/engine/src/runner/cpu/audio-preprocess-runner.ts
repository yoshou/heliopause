import type {
  AudioFeatures,
  AudioPcmInput,
  AudioSession,
} from "../../audio";
import type {
  AudioManifest,
} from "../../model";
import type {
  ExecutionProviderStats,
} from "../../runtime";
import {
  audioLogMelWasm,
} from "./wasm-kernels";

type AudioPreprocessStats = {
  webgpu: ProviderStats;
  wasm: ProviderStats;
  cpu: ProviderStats;
};

type ProviderStats = {
  attempts: number;
  runs: number;
  fallbacks: number;
  lastFallbackReason: string;
};

type AudioPreprocessSession = Pick<
  AudioSession,
  "manifest" | "preprocessProviders" | "setExecutionProviderStatsProvider"
>;

type AudioPreprocessConfig = Pick<
  AudioManifest,
  "sampleRate" | "maxSeconds" | "frameLength" | "hopLength" | "fftLength" | "featureSize" | "melFloor"
>;

export type AudioPreprocessOptions = {
  signal?: AbortSignal;
};

const statsBySession = new WeakMap<AudioPreprocessSession, AudioPreprocessStats>();
const windowCache = new Map<string, Float32Array>();
const filterCache = new Map<string, Float32Array>();

export async function runAudioPreprocessProviders(
  session: AudioPreprocessSession,
  audio: AudioPcmInput,
  cpuPreprocess: (audio: AudioPcmInput, manifest: AudioPreprocessConfig) => AudioFeatures,
  options: AudioPreprocessOptions = {},
): Promise<AudioFeatures> {
  const stats = audioPreprocessStats(session);
  for (const provider of session.preprocessProviders) {
    throwIfAborted(options.signal);
    if (provider.name === "webgpu") {
      recordFallback(stats.webgpu, "webgpu-preprocess-unimplemented");
      continue;
    }
    if (provider.name === "wasm") {
      const result = await tryAudioWasmProvider(session.manifest, audio, stats.wasm);
      if (result) {
        return result;
      }
      continue;
    }
    if (provider.name === "cpu") {
      stats.cpu.attempts += 1;
      const result = cpuPreprocess(audio, session.manifest);
      stats.cpu.runs += 1;
      stats.cpu.lastFallbackReason = "";
      return result;
    }
  }

  stats.cpu.attempts += 1;
  const result = cpuPreprocess(audio, session.manifest);
  stats.cpu.runs += 1;
  stats.cpu.lastFallbackReason = "";
  return result;
}

async function tryAudioWasmProvider(
  manifest: AudioPreprocessConfig,
  audio: AudioPcmInput,
  stats: ProviderStats,
): Promise<AudioFeatures | undefined> {
  stats.attempts += 1;
  try {
    const config = audioPreprocessConfig(manifest);
    if (audio.sampleRate !== config.sampleRate) {
      throw new Error(` audio expects ${config.sampleRate} Hz PCM, got ${audio.sampleRate}`);
    }
    const maxSamples = config.sampleRate * config.maxSeconds;
    const sampleCount = Math.min(audio.pcm.length, maxSamples);
    const pcm = audio.pcm.subarray(0, sampleCount);
    const frameCount = audioFrameCount(sampleCount, config.frameLength, config.hopLength);
    const wasm = await audioLogMelWasm(
      pcm,
      cachedHannWindow(config.frameLength, config.fftLength),
      cachedMelFilterBank(config.sampleRate, config.fftLength, config.featureSize),
      {
        frameLength: config.frameLength,
        hopLength: config.hopLength,
        fftLength: config.fftLength,
        featureSize: config.featureSize,
        melFloor: config.melFloor,
        frameCount,
      },
    );
    if (!wasm) {
      recordFallback(stats, "wasm-unavailable", false);
      return undefined;
    }
    stats.runs += 1;
    stats.lastFallbackReason = "";
    return {
      values: wasm.values,
      frameCount,
      featureSize: config.featureSize,
      attentionMask: wasm.attentionMask,
      durationMs: Math.min(audio.durationMs, config.maxSeconds * 1000),
    };
  } catch (error) {
    recordFallback(stats, error instanceof Error ? error.message : String(error), false);
    return undefined;
  }
}

function audioPreprocessConfig(manifest: Partial<AudioPreprocessConfig>): AudioPreprocessConfig {
  return {
    sampleRate: 16000,
    maxSeconds: 30,
    frameLength: 320,
    hopLength: 160,
    fftLength: 512,
    featureSize: 128,
    melFloor: 0.001,
    ...manifest,
  };
}

function audioFrameCount(sampleCount: number, frameLength: number, hopLength: number): number {
  const padLeft = Math.floor(frameLength / 2);
  const nWithLeft = sampleCount + padLeft;
  const frameCount = Math.floor((nWithLeft - (frameLength + 1)) / hopLength) + 1;
  return Math.max(0, frameCount);
}

function cachedHannWindow(length: number, fftLength: number): Float32Array {
  const key = `${length}:${fftLength}`;
  let window = windowCache.get(key);
  if (!window) {
    window = hannWindow(length, fftLength);
    windowCache.set(key, window);
  }
  return window;
}

function cachedMelFilterBank(sampleRate: number, fftLength: number, filterCount: number): Float32Array {
  const key = `${sampleRate}:${fftLength}:${filterCount}`;
  let filters = filterCache.get(key);
  if (!filters) {
    filters = melFilterBank(sampleRate, fftLength, filterCount);
    filterCache.set(key, filters);
  }
  return filters;
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

function audioPreprocessStats(session: AudioPreprocessSession): AudioPreprocessStats {
  let stats = statsBySession.get(session);
  if (!stats) {
    stats = {
      webgpu: createProviderStats(),
      wasm: createProviderStats(),
      cpu: createProviderStats(),
    };
    statsBySession.set(session, stats);
    const captured = stats;
    session.setExecutionProviderStatsProvider(() => audioPreprocessStatsSnapshot(captured), "audio-preprocess");
  }
  return stats;
}

function audioPreprocessStatsSnapshot(stats: AudioPreprocessStats): ExecutionProviderStats {
  return {
    webgpuAudioPreprocessAttempts: stats.webgpu.attempts,
    webgpuAudioPreprocessRuns: stats.webgpu.runs,
    webgpuAudioPreprocessFallbacks: stats.webgpu.fallbacks,
    webgpuAudioPreprocessLastFallbackReason: stats.webgpu.lastFallbackReason,
    wasmAudioPreprocessAttempts: stats.wasm.attempts,
    wasmAudioPreprocessRuns: stats.wasm.runs,
    wasmAudioPreprocessFallbacks: stats.wasm.fallbacks,
    wasmAudioPreprocessLastFallbackReason: stats.wasm.lastFallbackReason,
    cpuAudioPreprocessAttempts: stats.cpu.attempts,
    cpuAudioPreprocessRuns: stats.cpu.runs,
    cpuAudioPreprocessFallbacks: stats.cpu.fallbacks,
    cpuAudioPreprocessLastFallbackReason: stats.cpu.lastFallbackReason,
  };
}

function createProviderStats(): ProviderStats {
  return {
    attempts: 0,
    runs: 0,
    fallbacks: 0,
    lastFallbackReason: "",
  };
}

function recordFallback(stats: ProviderStats, reason: string, countAttempt = true): void {
  if (countAttempt) {
    stats.attempts += 1;
  }
  stats.fallbacks += 1;
  stats.lastFallbackReason = reason;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Audio preprocessing was aborted.", "AbortError");
  }
}
