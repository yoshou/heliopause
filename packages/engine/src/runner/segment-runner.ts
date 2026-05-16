import type {
  ForwardTrace,
  InferenceState,
} from "../runtime";

export type SegmentRunnerProvider = "reference" | "wasm" | "webgpu";

export type SegmentHiddenResult = {
  hidden: Float32Array;
};

export type SegmentRunOptions = {
  trace?: ForwardTrace;
  perLayerInputs?: Float32Array;
  attentionCausal?: boolean;
};

export type SegmentRunner = {
  readonly provider: SegmentRunnerProvider;
  readonly segmentStartLayer: number;
  readonly segmentEndLayerExclusive: number;
  runTokensHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: InferenceState,
    options?: SegmentRunOptions,
  ): Promise<SegmentHiddenResult>;
  runTokenHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: InferenceState,
    options?: SegmentRunOptions,
  ): Promise<SegmentHiddenResult>;
};
