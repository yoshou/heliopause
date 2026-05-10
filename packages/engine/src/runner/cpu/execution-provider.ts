import type {
  ForwardTrace,
  Gemma4ModelSession,
  OutputResult,
} from "../../runtime";
import { forwardGemma4Output } from "./layers";
import {
  Gemma4CpuSegmentRunner,
  type Gemma4CpuSegmentRunnerOptions,
} from "./segment-runner";

export function gemma4CpuSegmentRunner(
  options: Gemma4CpuSegmentRunnerOptions,
): Gemma4CpuSegmentRunner {
  return new Gemma4CpuSegmentRunner(options);
}

export function gemma4CpuOutput(
  session: Gemma4ModelSession,
  hidden: Float32Array,
  options: { topK: number; trace?: ForwardTrace },
): Promise<OutputResult> {
  return forwardGemma4Output(session, hidden, options);
}
