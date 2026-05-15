import type {
  ForwardTrace,
  ModelSession,
  OutputResult,
} from "../../runtime";
import { forwardOutput } from "./layers";
import {
  CpuSegmentRunner,
  type CpuSegmentRunnerOptions,
} from "./segment-runner";

export function cpuSegmentRunner(
  options: CpuSegmentRunnerOptions,
): CpuSegmentRunner {
  return new CpuSegmentRunner(options);
}

export function cpuOutput(
  session: ModelSession,
  hidden: Float32Array,
  options: { topK: number; trace?: ForwardTrace },
): Promise<OutputResult> {
  return forwardOutput(session, hidden, options);
}
