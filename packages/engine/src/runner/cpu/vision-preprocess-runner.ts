import type {
  ExecutionProviderStats,
} from "../../runtime";
import type {
  VisionManifest,
} from "../../model";
import type {
  VisionPixelValues,
  VisionResize,
  VisionSession,
} from "../../vision";
import {
  visionPreprocessRgbaWasm,
} from "./wasm-kernels";

type VisionPreprocessStats = {
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

type VisionPreprocessSession = Pick<
  VisionSession,
  "manifest" | "preprocessProviders" | "setExecutionProviderStatsProvider"
>;

const statsBySession = new WeakMap<VisionPreprocessSession, VisionPreprocessStats>();

export type VisionRgbaPreprocessInput = {
  rgba: Uint8ClampedArray;
  sourceWidth: number;
  sourceHeight: number;
  resize: VisionResize;
};

export type VisionPreprocessOptions = {
  signal?: AbortSignal;
};

export async function runVisionPreprocessProviders(
  session: VisionPreprocessSession,
  input: VisionRgbaPreprocessInput,
  cpuPreprocess: (
    rgba: Uint8ClampedArray,
    sourceWidth: number,
    sourceHeight: number,
    manifest: VisionManifest,
  ) => VisionPixelValues,
  options: VisionPreprocessOptions = {},
): Promise<VisionPixelValues> {
  const stats = visionPreprocessStats(session);
  for (const provider of session.preprocessProviders) {
    throwIfAborted(options.signal);
    if (provider.name === "webgpu") {
      recordFallback(stats.webgpu, "webgpu-preprocess-unimplemented");
      continue;
    }
    if (provider.name === "wasm") {
      const result = await tryVisionWasmProvider(session, input, stats.wasm);
      if (result) {
        return result;
      }
      continue;
    }
    if (provider.name === "cpu") {
      stats.cpu.attempts += 1;
      const result = cpuPreprocess(input.rgba, input.sourceWidth, input.sourceHeight, session.manifest);
      stats.cpu.runs += 1;
      stats.cpu.lastFallbackReason = "";
      return result;
    }
  }

  stats.cpu.attempts += 1;
  const result = cpuPreprocess(input.rgba, input.sourceWidth, input.sourceHeight, session.manifest);
  stats.cpu.runs += 1;
  stats.cpu.lastFallbackReason = "";
  return result;
}

async function tryVisionWasmProvider(
  session: Pick<VisionSession, "manifest">,
  input: VisionRgbaPreprocessInput,
  stats: ProviderStats,
): Promise<VisionPixelValues | undefined> {
  stats.attempts += 1;
  try {
    const values = await visionPreprocessRgbaWasm(
      input.rgba,
      input.sourceWidth,
      input.sourceHeight,
      input.resize.width,
      input.resize.height,
      session.manifest.imageMean,
      session.manifest.imageStd,
    );
    if (!values) {
      recordFallback(stats, "wasm-unavailable", false);
      return undefined;
    }
    stats.runs += 1;
    stats.lastFallbackReason = "";
    return {
      values,
      width: input.resize.width,
      height: input.resize.height,
    };
  } catch (error) {
    recordFallback(stats, error instanceof Error ? error.message : String(error), false);
    return undefined;
  }
}

function visionPreprocessStats(session: VisionPreprocessSession): VisionPreprocessStats {
  let stats = statsBySession.get(session);
  if (!stats) {
    stats = {
      webgpu: createProviderStats(),
      wasm: createProviderStats(),
      cpu: createProviderStats(),
    };
    statsBySession.set(session, stats);
    const captured = stats;
    session.setExecutionProviderStatsProvider(() => visionPreprocessStatsSnapshot(captured), "vision-preprocess");
  }
  return stats;
}

function visionPreprocessStatsSnapshot(stats: VisionPreprocessStats): ExecutionProviderStats {
  return {
    webgpuVisionPreprocessAttempts: stats.webgpu.attempts,
    webgpuVisionPreprocessRuns: stats.webgpu.runs,
    webgpuVisionPreprocessFallbacks: stats.webgpu.fallbacks,
    webgpuVisionPreprocessLastFallbackReason: stats.webgpu.lastFallbackReason,
    wasmVisionPreprocessAttempts: stats.wasm.attempts,
    wasmVisionPreprocessRuns: stats.wasm.runs,
    wasmVisionPreprocessFallbacks: stats.wasm.fallbacks,
    wasmVisionPreprocessLastFallbackReason: stats.wasm.lastFallbackReason,
    cpuVisionPreprocessAttempts: stats.cpu.attempts,
    cpuVisionPreprocessRuns: stats.cpu.runs,
    cpuVisionPreprocessFallbacks: stats.cpu.fallbacks,
    cpuVisionPreprocessLastFallbackReason: stats.cpu.lastFallbackReason,
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
    throw new DOMException("Vision preprocessing was aborted.", "AbortError");
  }
}
