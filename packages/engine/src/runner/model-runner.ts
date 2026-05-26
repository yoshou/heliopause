import type {
  ModelManifest,
} from "../model";
import type {
  ForwardTrace,
  InferenceState,
  ModelSession,
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
  selectedTokenId?: number;
  topTokens?: Array<{ id: number; value: number }>;
};

export type ModelGraphNodeFactory = {
  createEmbeddingNode(tokenIds: readonly number[]): ForwardRunnerNode;
  createPreparedHiddenInputNode(hidden: Float32Array): ForwardRunnerNode;
  createLayerSegmentNode(startLayer: number, endLayerExclusive: number, inputId: string): ForwardRunnerNode;
  createOutputNode(inputId: string, topK?: number): ForwardRunnerNode;
  createImportHiddenNode(inputId: string): ForwardRunnerNode;
  createExportHiddenNode(inputId: string): ForwardRunnerNode;
};

export type ModelRunner = {
  readonly provider: SegmentRunnerProvider;
  readonly graphNodes: ModelGraphNodeFactory;
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
  decodeToken?(
    session: ModelSession,
    tokenId: number,
    options: ModelDecodeTokenOptions,
  ): Promise<ModelDecodeTokenResult>;
};
