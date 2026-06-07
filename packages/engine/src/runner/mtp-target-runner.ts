import type {
  MtpAssistantManifest,
} from "../model";
import type {
  InferenceState,
  ModelSession,
} from "../runtime";
import type {
  MtpTargetKvView,
} from "./mtp-assistant-runner";
import type {
  SegmentRunnerProvider,
} from "./segment-runner";

export type MtpTokenDistribution = {
  tokens: Array<{ id: number; logit: number }>;
  vocabularySize: number;
  masked?: boolean;
};

export type MtpTargetContext = {
  position: number;
  previousHidden: Float32Array;
  currentHidden: Float32Array;
  targetKv: MtpTargetKvView;
};

export type MtpTargetPrefillResult = {
  firstTokenId: number;
  firstTokenDistribution: MtpTokenDistribution;
  context: MtpTargetContext;
};

export type MtpTargetPrefillOptions = {
  positions?: Int32Array | number[];
  logitsTopK: number;
  assistantManifest: MtpAssistantManifest;
  maxSpeculativeTokens?: number;
  signal?: AbortSignal;
};

export type MtpTargetVerificationResult = {
  basePosition: number;
  verifiedLength: number;
  targetTokenIds: number[];
  targetDistributions: MtpTokenDistribution[];
  bonusTokenId: number;
  bonusDistribution: MtpTokenDistribution;
  contexts: MtpTargetContext[];
};

export type MtpTargetVerifyOptions = {
  logitsTopK: number;
  assistantManifest: MtpAssistantManifest;
  maxSpeculativeTokens?: number;
  signal?: AbortSignal;
};

export type MtpTargetRunner = {
  readonly provider: SegmentRunnerProvider;
  prefill(
    session: ModelSession,
    state: InferenceState,
    tokenIds: readonly number[],
    options: MtpTargetPrefillOptions,
  ): Promise<MtpTargetPrefillResult>;
  verify(
    session: ModelSession,
    state: InferenceState,
    draftTokenIds: readonly number[],
    options: MtpTargetVerifyOptions,
  ): Promise<MtpTargetVerificationResult>;
  finalize(
    state: InferenceState,
    verification: MtpTargetVerificationResult,
    committedLength: number,
  ): MtpTargetContext;
};

export type MtpTargetRunners = {
  runner: MtpTargetRunner;
};
