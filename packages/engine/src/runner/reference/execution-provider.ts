import type {
  ForwardTrace,
  ModelSession,
  OutputResult,
} from "../../runtime";
import { forwardOutput } from "./layers";
import {
  ReferenceSegmentRunner,
  type ReferenceSegmentRunnerOptions,
} from "./segment-runner";

export function referenceSegmentRunner(
  options: ReferenceSegmentRunnerOptions,
): ReferenceSegmentRunner {
  return new ReferenceSegmentRunner(options);
}

export function referenceOutput(
  session: ModelSession,
  hidden: Float32Array,
  options: { topK: number; trace?: ForwardTrace },
): Promise<OutputResult> {
  return forwardOutput(session, hidden, options);
}
