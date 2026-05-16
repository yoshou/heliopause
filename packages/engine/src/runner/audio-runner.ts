import type {
  AudioEncodeResult,
  AudioFeatures,
  AudioPcmInput,
  AudioSession,
} from "../audio";
import type {
  AudioManifest,
} from "../model";
import type {
  SegmentRunnerProvider,
} from "./segment-runner";

export type AudioPreprocessOptions = {
  signal?: AbortSignal;
};

export type AudioPreprocessConfig = Pick<
  AudioManifest,
  "sampleRate" | "maxSeconds" | "frameLength" | "hopLength" | "fftLength" | "featureSize" | "melFloor"
>;

export type AudioPreprocessRunner = {
  readonly provider: SegmentRunnerProvider;
  run(
    session: AudioSession,
    audio: AudioPcmInput,
    audioPreprocess: (audio: AudioPcmInput, manifest: AudioPreprocessConfig) => AudioFeatures,
    options: AudioPreprocessOptions,
  ): Promise<AudioFeatures | undefined>;
};

export type AudioEncoderRunner = {
  readonly provider: SegmentRunnerProvider;
  run(
    session: AudioSession,
    features: AudioFeatures,
    options: { signal?: AbortSignal },
  ): Promise<AudioEncodeResult>;
};

export type AudioRunners = {
  preprocess: AudioPreprocessRunner;
  encoder: AudioEncoderRunner;
};
