import type { InferenceState, ModelSession } from "../../runtime";
import { WEBGPU_MEMORY_LIMIT_BYTES } from "./gpu-constants";
import { WebGpuSegmentRunner } from "./segment-runner";

const segmentRunners = new WeakMap<ModelSession, Map<string, Promise<WebGpuSegmentRunner>>>();

export type WebGpuExecutionProviderOptions = {
  memoryLimitBytes: number;
  segmentStartLayer?: number;
  prefillChunkSize?: number;
};

export type WebGpuProviderOptions = Partial<WebGpuExecutionProviderOptions>;

export type WebGpuConfiguredProvider = {
  readonly name: "webgpu";
  readonly options: Readonly<WebGpuProviderOptions>;
};

export function webGpuExecutionProviderEnabled(session: ModelSession): boolean {
  return session.hasProvider("webgpu");
}

export function webGpuExecutionProviderOptions(
  session: ModelSession,
): WebGpuExecutionProviderOptions | undefined {
  const config = session.provider<WebGpuConfiguredProvider>("webgpu");
  if (!config) {
    return undefined;
  }
  return {
    memoryLimitBytes: config.options.memoryLimitBytes ?? WEBGPU_MEMORY_LIMIT_BYTES,
    segmentStartLayer: numberOption(config.options, "segmentStartLayer"),
    prefillChunkSize: numberOption(config.options, "prefillChunkSize"),
  };
}

export function webGpuSegmentRunner(
  session: ModelSession,
  state: InferenceState,
  options: { segmentStartLayer?: number; segmentEndLayerExclusive?: number } = {},
): Promise<WebGpuSegmentRunner> {
  const providerOptions = webGpuExecutionProviderOptions(session);
  if (!providerOptions) {
    throw new Error("WebGPU segment runner is not enabled for this  session.");
  }
  let runners = segmentRunners.get(session);
  if (!runners) {
    runners = new Map();
    segmentRunners.set(session, runners);
  }
  const segmentStartLayer = options.segmentStartLayer ?? providerOptions.segmentStartLayer;
  if (segmentStartLayer === undefined) {
    throw new Error("WebGPU segment planning selected no layers.");
  }
  const segmentEndLayerExclusive = options.segmentEndLayerExclusive ?? session.manifest.blockCount;
  const cacheKey = `${segmentStartLayer}:${segmentEndLayerExclusive}`;
  let runner = runners.get(cacheKey);
  if (!runner) {
    runner = WebGpuSegmentRunner.create({
      tensorReader: session.tensorReader,
      manifest: session.manifest,
      epsilon: session.epsilon,
      contextLength: state.contextLength,
      memoryLimitBytes: providerOptions.memoryLimitBytes,
      prefillChunkSize: providerOptions.prefillChunkSize,
      segmentStartLayer,
      segmentEndLayerExclusive,
    });
    runners.set(cacheKey, runner);
    void runner.then((resolved) => {
      session.setExecutionProviderStatsProvider(() => resolved.runtimeStats(), "webgpu");
    });
  }
  return runner;
}

function numberOption(options: Readonly<Record<string, unknown>> | undefined, name: string): number | undefined {
  const value = options?.[name];
  return typeof value === "number" ? value : undefined;
}
