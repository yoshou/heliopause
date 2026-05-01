import type { Qwen35InferenceState, Qwen35ModelSession } from "../../runtime";
import { QWEN35_WEBGPU_MEMORY_LIMIT_BYTES } from "./gpu-constants";
import { Qwen35WebGpuSegmentRunner } from "./segment-runner";

const segmentRunners = new WeakMap<Qwen35ModelSession, Promise<Qwen35WebGpuSegmentRunner>>();

export type Qwen35WebGpuExecutionProviderOptions = {
  memoryLimitBytes: number;
  segmentStartLayer?: number;
};

export function webGpuExecutionProviderEnabled(session: Qwen35ModelSession): boolean {
  return session.executionProvider("webgpu") !== undefined;
}

export function webGpuExecutionProviderOptions(
  session: Qwen35ModelSession,
): Qwen35WebGpuExecutionProviderOptions | undefined {
  const config = session.executionProvider("webgpu");
  if (!config) {
    return undefined;
  }
  return {
    memoryLimitBytes: numberOption(config.options, "memoryLimitBytes") ?? QWEN35_WEBGPU_MEMORY_LIMIT_BYTES,
    segmentStartLayer: numberOption(config.options, "segmentStartLayer"),
  };
}

export function qwen35WebGpuSegmentRunner(
  session: Qwen35ModelSession,
  state: Qwen35InferenceState,
  options: { segmentStartLayer?: number } = {},
): Promise<Qwen35WebGpuSegmentRunner> {
  const providerOptions = webGpuExecutionProviderOptions(session);
  if (!providerOptions) {
    throw new Error("WebGPU segment runner is not enabled for this Qwen35 session.");
  }
  let runner = segmentRunners.get(session);
  if (!runner) {
    const segmentStartLayer = options.segmentStartLayer ?? providerOptions.segmentStartLayer;
    if (segmentStartLayer === undefined) {
      throw new Error("WebGPU segment planning selected no layers.");
    }
    runner = Qwen35WebGpuSegmentRunner.create({
      tensorReader: session.tensorReader,
      manifest: session.manifest,
      epsilon: session.epsilon,
      contextLength: state.contextLength,
      memoryLimitBytes: providerOptions.memoryLimitBytes,
      segmentStartLayer,
    });
    segmentRunners.set(session, runner);
  }
  return runner;
}

function numberOption(options: Readonly<Record<string, unknown>> | undefined, name: string): number | undefined {
  const value = options?.[name];
  return typeof value === "number" ? value : undefined;
}
