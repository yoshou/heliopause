import {
  ReferenceSegmentRunner,
  type ReferenceSegmentRunnerOptions,
} from "./segment-runner";

export function referenceSegmentRunner(
  options: ReferenceSegmentRunnerOptions,
): ReferenceSegmentRunner {
  return new ReferenceSegmentRunner(options);
}
