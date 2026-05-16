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
import type {
  AudioPreprocessRunner,
} from "../audio-runner";
import { GpuMemoryArena, type F32Handle, type GpuResource } from "./arena";
import { GPU_COPY_DST, GPU_COPY_SRC, GPU_MAP_READ, GPU_STORAGE, WEBGPU_MEMORY_LIMIT_BYTES } from "./gpu-constants";
import { webGpuDevice } from "./gpu-device";
import type { WebGpuBufferLike } from "./gpu-types";
import {
  createAudioFftBitReverseResources,
  createAudioFftStageResources,
  createAudioLogMelResources,
  createAudioWindowFramesResources,
} from "./audio-kernel-resources";
import type {
  WebGpuConfiguredProvider,
} from "./execution-provider";

type AudioPreprocessStats = {
  attempts: number;
  runs: number;
};

type AudioPreprocessSession = Pick<
  AudioSession,
  "manifest" | "provider" | "setExecutionProviderStatsProvider" | "addDisposeCallback"
>;

type AudioPreprocessConfig = Pick<
  AudioManifest,
  "sampleRate" | "maxSeconds" | "frameLength" | "hopLength" | "fftLength" | "featureSize" | "melFloor"
>;

type AudioPreprocessRunBuffers = {
  resources: Array<{ destroy: () => void }>;
  cleanup: GpuResource[];
};

const runners = new WeakMap<AudioPreprocessSession, Promise<WebGpuAudioPreprocessRunner>>();
const statsBySession = new WeakMap<AudioPreprocessSession, AudioPreprocessStats>();

export const webGpuAudioPreprocessRunner: AudioPreprocessRunner = {
  provider: "webgpu",
  run: (session, audio, _referencePreprocess, options) =>
    runWebGpuAudioPreprocessor(session, audio, options),
};

export async function runWebGpuAudioPreprocessor(
  session: AudioPreprocessSession,
  audio: AudioPcmInput,
  options: { signal?: AbortSignal } = {},
): Promise<AudioFeatures> {
  const stats = audioPreprocessStats(session);
  stats.attempts += 1;
  const runner = await audioPreprocessRunner(session);
  const result = await runner.run(audio, options);
  stats.runs += 1;
  return result;
}

function audioPreprocessStats(session: AudioPreprocessSession): AudioPreprocessStats {
  let stats = statsBySession.get(session);
  if (!stats) {
    stats = {
      attempts: 0,
      runs: 0,
    };
    statsBySession.set(session, stats);
    const captured = stats;
    session.setExecutionProviderStatsProvider((): ExecutionProviderStats => ({
      webgpuAudioPreprocessAttempts: captured.attempts,
      webgpuAudioPreprocessRuns: captured.runs,
    }), "webgpu-audio-preprocess");
  }
  return stats;
}

async function audioPreprocessRunner(
  session: AudioPreprocessSession,
): Promise<WebGpuAudioPreprocessRunner> {
  let runner = runners.get(session);
  if (!runner) {
    runner = createAudioPreprocessRunner(session);
    runners.set(session, runner);
  }
  return runner;
}

async function createAudioPreprocessRunner(
  session: AudioPreprocessSession,
): Promise<WebGpuAudioPreprocessRunner> {
  const device = await webGpuDevice();
  if (!device) {
    throw new Error("WebGPU is not available for audio preprocessing.");
  }
  const options = session.provider<WebGpuConfiguredProvider>("webgpu")?.options;
  const arena = new GpuMemoryArena(
    device,
    options?.memoryLimitBytes ?? WEBGPU_MEMORY_LIMIT_BYTES,
  );
  const runner = new WebGpuAudioPreprocessRunner(session, arena);
  session.addDisposeCallback(() => runner.dispose());
  return runner;
}

class WebGpuAudioPreprocessRunner {
  private readonly session: AudioPreprocessSession;
  private readonly arena: GpuMemoryArena;
  private readonly windowHandles = new Map<string, F32Handle>();
  private readonly filterHandles = new Map<string, F32Handle>();

  constructor(session: AudioPreprocessSession, arena: GpuMemoryArena) {
    this.session = session;
    this.arena = arena;
  }

  dispose(): void {
    for (const handle of this.windowHandles.values()) {
      handle.destroy();
    }
    for (const handle of this.filterHandles.values()) {
      handle.destroy();
    }
    this.windowHandles.clear();
    this.filterHandles.clear();
  }

  async run(audio: AudioPcmInput, options: { signal?: AbortSignal } = {}): Promise<AudioFeatures> {
    throwIfAborted(options.signal);
    const config = audioPreprocessConfig(this.session.manifest);
    if (audio.sampleRate !== config.sampleRate) {
      throw new WebGpuAudioPreprocessFallbackError(` audio expects ${config.sampleRate} Hz PCM, got ${audio.sampleRate}`);
    }
    if (!Number.isInteger(config.fftLength) || config.fftLength <= 0 || !isPowerOfTwo(config.fftLength)) {
      throw new WebGpuAudioPreprocessFallbackError(`Audio FFT length must be a positive power of two, got ${config.fftLength}`);
    }
    if (config.frameLength <= 0 || config.hopLength <= 0 || config.featureSize <= 0 || config.frameLength > config.fftLength) {
      throw new WebGpuAudioPreprocessFallbackError("Audio preprocess configuration is invalid.");
    }

    const maxSamples = config.sampleRate * config.maxSeconds;
    const sampleCount = Math.min(audio.pcm.length, maxSamples);
    const pcm = audio.pcm.subarray(0, sampleCount);
    const frameCount = audioFrameCount(sampleCount, config.frameLength, config.hopLength);
    if (frameCount === 0) {
      return {
        values: new Float32Array(),
        frameCount,
        featureSize: config.featureSize,
        attentionMask: new Uint8Array(),
        durationMs: Math.min(audio.durationMs, config.maxSeconds * 1000),
      };
    }

    const run: AudioPreprocessRunBuffers = { resources: [], cleanup: [] };
    let outputReadback: WebGpuBufferLike | undefined;
    let maskReadback: WebGpuBufferLike | undefined;
    try {
      const valueLength = frameCount * config.featureSize;
      const fftLength = frameCount * config.fftLength;
      const pcmBuffer = this.bufferFromF32("audio.preprocess.pcm", pcm, run.cleanup);
      const window = this.windowHandle(config);
      const filters = this.filterHandle(config);
      const realA = this.scratchF32("audio.preprocess.real_a", fftLength, run.cleanup);
      const imagA = this.scratchF32("audio.preprocess.imag_a", fftLength, run.cleanup);
      const realB = this.scratchF32("audio.preprocess.real_b", fftLength, run.cleanup);
      const imagB = this.scratchF32("audio.preprocess.imag_b", fftLength, run.cleanup);
      const output = this.scratchF32("audio.preprocess.output", valueLength, run.cleanup);
      const mask = this.scratchU32("audio.preprocess.mask", frameCount, run.cleanup);

      const encoder = this.arena.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      let resource = createAudioWindowFramesResources(this.arena.device, pcmBuffer, window.buffer, realA, imagA, {
        sampleCount,
        frameCount,
        frameLength: config.frameLength,
        hopLength: config.hopLength,
        fftLength: config.fftLength,
      });
      run.resources.push(resource);
      pass.setPipeline(resource.pipeline);
      pass.setBindGroup(0, resource.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(fftLength / 256));

      resource = createAudioFftBitReverseResources(this.arena.device, realA, imagA, realB, imagB, {
        frameCount,
        fftLength: config.fftLength,
      });
      run.resources.push(resource);
      pass.setPipeline(resource.pipeline);
      pass.setBindGroup(0, resource.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(fftLength / 256));

      let inputReal = realB;
      let inputImag = imagB;
      let outputReal = realA;
      let outputImag = imagA;
      for (let stageSize = 2; stageSize <= config.fftLength; stageSize <<= 1) {
        resource = createAudioFftStageResources(this.arena.device, inputReal, inputImag, outputReal, outputImag, {
          frameCount,
          fftLength: config.fftLength,
          stageSize,
        });
        run.resources.push(resource);
        pass.setPipeline(resource.pipeline);
        pass.setBindGroup(0, resource.bindGroup);
        pass.dispatchWorkgroups(Math.ceil((frameCount * config.fftLength / 2) / 256));
        [inputReal, outputReal] = [outputReal, inputReal];
        [inputImag, outputImag] = [outputImag, inputImag];
      }

      resource = createAudioLogMelResources(this.arena.device, inputReal, inputImag, filters.buffer, output, mask, {
        frameCount,
        fftLength: config.fftLength,
        featureSize: config.featureSize,
        melFloor: config.melFloor,
      });
      run.resources.push(resource);
      pass.setPipeline(resource.pipeline);
      pass.setBindGroup(0, resource.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(config.featureSize / 16), Math.ceil(frameCount / 16));
      pass.end();

      outputReadback = this.arena.device.createBuffer({
        size: valueLength * Float32Array.BYTES_PER_ELEMENT,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
      maskReadback = this.arena.device.createBuffer({
        size: frameCount * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
      encoder.copyBufferToBuffer(output, 0, outputReadback, 0, valueLength * Float32Array.BYTES_PER_ELEMENT);
      encoder.copyBufferToBuffer(mask, 0, maskReadback, 0, frameCount * Uint32Array.BYTES_PER_ELEMENT);
      this.arena.device.queue.submit([encoder.finish()]);
      await Promise.all([
        outputReadback.mapAsync(GPU_MAP_READ),
        maskReadback.mapAsync(GPU_MAP_READ),
      ]);
      const values = new Float32Array(outputReadback.getMappedRange()).slice();
      const maskU32 = new Uint32Array(maskReadback.getMappedRange()).slice();
      const attentionMask = new Uint8Array(maskU32.length);
      for (let index = 0; index < maskU32.length; index += 1) {
        attentionMask[index] = maskU32[index] ? 1 : 0;
      }
      outputReadback.unmap();
      maskReadback.unmap();
      outputReadback.destroy?.();
      maskReadback.destroy?.();
      outputReadback = undefined;
      maskReadback = undefined;
      return {
        values,
        frameCount,
        featureSize: config.featureSize,
        attentionMask,
        durationMs: Math.min(audio.durationMs, config.maxSeconds * 1000),
      };
    } finally {
      outputReadback?.destroy?.();
      maskReadback?.destroy?.();
      for (const resource of run.resources) {
        resource.destroy();
      }
      for (const item of run.cleanup.reverse()) {
        item.destroy?.();
      }
    }
  }

  private windowHandle(config: AudioPreprocessConfig): F32Handle {
    const key = `${config.frameLength}:${config.fftLength}`;
    let handle = this.windowHandles.get(key);
    if (!handle) {
      handle = this.f32Handle(`audio.preprocess.window.${key}`, hannWindow(config.frameLength, config.fftLength));
      this.windowHandles.set(key, handle);
    }
    return handle;
  }

  private filterHandle(config: AudioPreprocessConfig): F32Handle {
    const key = `${config.sampleRate}:${config.fftLength}:${config.featureSize}`;
    let handle = this.filterHandles.get(key);
    if (!handle) {
      handle = this.f32Handle(`audio.preprocess.filters.${key}`, melFilterBank(config.sampleRate, config.fftLength, config.featureSize));
      this.filterHandles.set(key, handle);
    }
    return handle;
  }

  private f32Handle(label: string, values: Float32Array): F32Handle {
    const buffer = this.arena.createBuffer(label, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
    this.arena.device.queue.writeBuffer(buffer, 0, values);
    return {
      length: values.length,
      byteLength: values.byteLength,
      device: this.arena.device,
      buffer,
      destroy: () => buffer.destroy?.(),
    };
  }

  private bufferFromF32(label: string, values: Float32Array, cleanup: GpuResource[]): WebGpuBufferLike {
    const buffer = this.arena.createBuffer(label, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
    this.arena.device.queue.writeBuffer(buffer, 0, values);
    cleanup.push(buffer);
    return buffer;
  }

  private scratchF32(label: string, length: number, cleanup: GpuResource[]): WebGpuBufferLike {
    const buffer = this.arena.createBuffer(label, length * Float32Array.BYTES_PER_ELEMENT, GPU_STORAGE | GPU_COPY_SRC);
    cleanup.push(buffer);
    return buffer;
  }

  private scratchU32(label: string, length: number, cleanup: GpuResource[]): WebGpuBufferLike {
    const buffer = this.arena.createBuffer(label, length * Uint32Array.BYTES_PER_ELEMENT, GPU_STORAGE | GPU_COPY_SRC);
    cleanup.push(buffer);
    return buffer;
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

function numberOption(options: Readonly<Record<string, unknown>> | undefined, name: string): number | undefined {
  const value = options?.[name];
  return typeof value === "number" ? value : undefined;
}

function isPowerOfTwo(value: number): boolean {
  return (value & (value - 1)) === 0;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Audio preprocessing was aborted.", "AbortError");
  }
}

function isFallbackError(error: unknown): boolean {
  return error instanceof WebGpuAudioPreprocessFallbackError ||
    (error instanceof Error && error.message.includes("WebGPU memory cap exceeded"));
}

class WebGpuAudioPreprocessFallbackError extends Error {}
