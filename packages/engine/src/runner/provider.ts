import type {
  AudioRunners,
} from "./audio-runner";
import type {
  ModelGraphRunner,
  ModelRunner,
} from "./model-runner";
import type {
  RunnerPlacementPlan,
} from "./planning";
import type {
  ModelSession,
} from "../runtime";
import type {
  SegmentRunnerProvider,
} from "./segment-runner";
import type {
  VisionRunners,
} from "./vision-runner";

export type RunnerProvider = {
  readonly name: SegmentRunnerProvider;
  createModelRunner?(): ModelRunner;
  createModelGraphRunner?(): ModelGraphRunner;
  planModelPlacement?(session: ModelSession, options: { contextLength: number }): RunnerPlacementPlan;
  createAudioRunners?(): AudioRunners;
  createVisionRunners?(): VisionRunners;
};
