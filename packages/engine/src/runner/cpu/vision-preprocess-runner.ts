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
  wasm: ProviderStats;
  reference: ProviderStats;
};

type ProviderStats = {
  attempts: number;
  runs: number;
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
  const result = await tryVisionPreprocessProviders(
    session,
    input,
    cpuPreprocess,
    session.preprocessProviders,
    options,
  );
  if (result) {
    return result;
  }

  throw new Error("No vision preprocessor provider was selected.");
}

export async function tryVisionPreprocessProviders(
  session: VisionPreprocessSession,
  input: VisionRgbaPreprocessInput,
  cpuPreprocess: (
    rgba: Uint8ClampedArray,
    sourceWidth: number,
    sourceHeight: number,
    manifest: VisionManifest,
  ) => VisionPixelValues,
  providers: readonly { name: string }[],
  options: VisionPreprocessOptions = {},
): Promise<VisionPixelValues | undefined> {
  const stats = visionPreprocessStats(session);
  for (const provider of providers) {
    throwIfAborted(options.signal);
    if (provider.name === "wasm") {
      const result = await tryVisionWasmProvider(session, input, stats.wasm);
      if (result) {
        return result;
      }
      continue;
    }
    if (provider.name === "reference") {
      return runVisionReferencePreprocess(session, input, cpuPreprocess);
    }
    throw new Error(`Unsupported vision preprocess provider: ${provider.name}`);
  }

  return undefined;
}

function runVisionReferencePreprocess(
  session: VisionPreprocessSession,
  input: VisionRgbaPreprocessInput,
  cpuPreprocess: (
    rgba: Uint8ClampedArray,
    sourceWidth: number,
    sourceHeight: number,
    manifest: VisionManifest,
  ) => VisionPixelValues,
): VisionPixelValues {
  const stats = visionPreprocessStats(session);
  stats.reference.attempts += 1;
  const result = cpuPreprocess(input.rgba, input.sourceWidth, input.sourceHeight, session.manifest);
  stats.reference.runs += 1;
  stats.reference.lastFallbackReason = "";
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
      throw new Error("WASM vision preprocessing is unavailable.");
    }
    stats.runs += 1;
    stats.lastFallbackReason = "";
    return {
      values,
      width: input.resize.width,
      height: input.resize.height,
    };
  } catch (error) {
    throw error;
  }
}

function visionPreprocessStats(session: VisionPreprocessSession): VisionPreprocessStats {
  let stats = statsBySession.get(session);
  if (!stats) {
    stats = {
      wasm: createProviderStats(),
      reference: createProviderStats(),
    };
    statsBySession.set(session, stats);
    const captured = stats;
    session.setExecutionProviderStatsProvider(() => visionPreprocessStatsSnapshot(captured), "vision-preprocess");
  }
  return stats;
}

function visionPreprocessStatsSnapshot(stats: VisionPreprocessStats): ExecutionProviderStats {
  return {
    wasmVisionPreprocessAttempts: stats.wasm.attempts,
    wasmVisionPreprocessRuns: stats.wasm.runs,
    referenceVisionPreprocessAttempts: stats.reference.attempts,
    referenceVisionPreprocessRuns: stats.reference.runs,
  };
}

function createProviderStats(): ProviderStats {
  return {
    attempts: 0,
    runs: 0,
    lastFallbackReason: "",
  };
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Vision preprocessing was aborted.", "AbortError");
  }
}
