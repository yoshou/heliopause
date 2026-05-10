import type { Gemma4InferenceState, Gemma4ModelSession } from "../../runtime";
import { GEMMA4_WEBGPU_MEMORY_LIMIT_BYTES } from "./gpu-constants";
import { Gemma4WebGpuSegmentRunner } from "./segment-runner";

const segmentRunners = new WeakMap<Gemma4ModelSession, Promise<Gemma4WebGpuSegmentRunner>>();

export type Gemma4WebGpuExecutionProviderOptions = {
  memoryLimitBytes: number;
  segmentStartLayer?: number;
};

export function webGpuExecutionProviderEnabled(session: Gemma4ModelSession): boolean {
  return session.executionProvider("webgpu") !== undefined;
}

export function webGpuExecutionProviderOptions(
  session: Gemma4ModelSession,
): Gemma4WebGpuExecutionProviderOptions | undefined {
  const config = session.executionProvider("webgpu");
  if (!config) {
    return undefined;
  }
  return {
    memoryLimitBytes: numberOption(config.options, "memoryLimitBytes") ?? GEMMA4_WEBGPU_MEMORY_LIMIT_BYTES,
    segmentStartLayer: numberOption(config.options, "segmentStartLayer"),
  };
}

export function gemma4WebGpuSegmentRunner(
  session: Gemma4ModelSession,
  state: Gemma4InferenceState,
  options: { segmentStartLayer?: number } = {},
): Promise<Gemma4WebGpuSegmentRunner> {
  const providerOptions = webGpuExecutionProviderOptions(session);
  if (!providerOptions) {
    throw new Error("WebGPU segment runner is not enabled for this Gemma4 session.");
  }
  let runner = segmentRunners.get(session);
  if (!runner) {
    const segmentStartLayer = options.segmentStartLayer ?? providerOptions.segmentStartLayer;
    if (segmentStartLayer === undefined) {
      throw new Error("WebGPU segment planning selected no layers.");
    }
    runner = Gemma4WebGpuSegmentRunner.create({
      tensorReader: session.tensorReader,
      manifest: session.manifest,
      epsilon: session.epsilon,
      contextLength: state.contextLength,
      memoryLimitBytes: providerOptions.memoryLimitBytes,
      segmentStartLayer,
    });
    segmentRunners.set(session, runner);
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
