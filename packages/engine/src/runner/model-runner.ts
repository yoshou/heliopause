import type {
  ModelManifest,
} from "../model";
import type {
  ForwardTrace,
  InferenceState,
  ModelSession,
  OutputResult,
} from "../runtime";
import type {
  ForwardRunnerNode,
} from "./graph";
import type {
  SegmentRunner,
  SegmentRunnerProvider,
} from "./segment-runner";

export type ModelPreparedInput = {
  hidden: Float32Array;
  perLayerInputs?: Float32Array;
};

export type ModelSegmentRunnerOptions = {
  session: ModelSession;
  state: InferenceState;
  manifest: ModelManifest;
  epsilon: number;
  segmentStartLayer: number;
  segmentEndLayerExclusive: number;
};

export type ModelDecodeTokenOptions = {
  state: InferenceState;
  position: number;
  logitsTopK: number;
  trace?: ForwardTrace;
};

export type ModelDecodeTokenResult = {
  hidden: Float32Array;
  state: InferenceState;
  logits?: Float32Array;
  selectedTokenId?: number;
  topTokens?: Array<{ id: number; value: number }>;
};

export type ModelGraphRunner = {
  embeddingNode(tokenIds: readonly number[]): ForwardRunnerNode;
  layerSegmentNode?(startLayer: number, endLayerExclusive: number, inputId: string): ForwardRunnerNode;
  outputNode?(inputId: string, topK?: number): ForwardRunnerNode;
  importHiddenNode?(inputId: string): ForwardRunnerNode;
  exportHiddenNode?(inputId: string): ForwardRunnerNode;
};

export type ModelRunner = {
  readonly provider: SegmentRunnerProvider;
  prepareInput(
    session: ModelSession,
    tokenIds: readonly number[],
    trace?: ForwardTrace,
  ): Promise<ModelPreparedInput>;
  preparePreparedHiddenInput(
    session: ModelSession,
    hidden: Float32Array,
    trace?: ForwardTrace,
  ): Promise<ModelPreparedInput>;
  segmentRunner(options: ModelSegmentRunnerOptions): SegmentRunner | Promise<SegmentRunner>;
  output(
    session: ModelSession,
    hidden: Float32Array,
    options: { topK: number; trace?: ForwardTrace },
  ): Promise<OutputResult>;
  decodeToken?(
    session: ModelSession,
    tokenId: number,
    options: ModelDecodeTokenOptions,
  ): Promise<ModelDecodeTokenResult>;
  readonly graph?: ModelGraphRunner;
};
