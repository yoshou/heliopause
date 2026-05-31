import type {
  MtpAssistantRunResult,
  MtpAssistantSession,
} from "../mtp-assistant";
import type {
  SegmentRunnerProvider,
} from "./segment-runner";

export type MtpTargetKvLayerView = {
  readonly key: Float32Array;
  readonly value: Float32Array;
  readonly keyLength: number;
  readonly valueLength: number;
  readonly headCountKv: number;
  readonly contextLength: number;
  readonly tokenCount: number;
};

export type MtpTargetKvView = {
  readonly layers: readonly MtpTargetKvLayerView[];
};

export type MtpAssistantRunInput = {
  tokenId: number;
  targetInputEmbedding: Float32Array;
  targetPreviousHidden: Float32Array;
  targetCurrentHidden: Float32Array;
  targetKv: MtpTargetKvView;
  position: number;
  topK: number;
};

export type MtpAssistantRunner = {
  readonly provider: SegmentRunnerProvider;
  run(
    session: MtpAssistantSession,
    input: MtpAssistantRunInput,
    options: { signal?: AbortSignal },
  ): Promise<MtpAssistantRunResult>;
};

export type MtpAssistantRunners = {
  runner: MtpAssistantRunner;
};
