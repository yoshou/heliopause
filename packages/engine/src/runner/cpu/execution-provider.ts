import type {
  ForwardTrace,
  Qwen35ModelSession,
  OutputResult,
} from "../../runtime";
import { forwardQwen35Output } from "./layers";
import {
  Qwen35CpuSegmentRunner,
  type Qwen35CpuSegmentRunnerOptions,
} from "./segment-runner";

export function qwen35CpuSegmentRunner(
  options: Qwen35CpuSegmentRunnerOptions,
): Qwen35CpuSegmentRunner {
  return new Qwen35CpuSegmentRunner(options);
}

export function qwen35CpuOutput(
  session: Qwen35ModelSession,
  hidden: Float32Array,
  options: { topK: number; trace?: ForwardTrace },
): Promise<OutputResult> {
  return forwardQwen35Output(session, hidden, options);
}
