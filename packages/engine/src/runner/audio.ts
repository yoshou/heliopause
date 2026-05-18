import type {
  AudioEncodeResult,
  AudioFeatures,
  AudioPcmInput,
  AudioSession,
} from "../audio";
import type {
  AudioEncoderRunner,
  AudioPreprocessConfig,
  AudioPreprocessOptions,
  AudioPreprocessRunner,
  AudioRunners,
} from "./audio-runner";

export type { AudioPreprocessOptions } from "./audio-runner";

export async function dispatchAudioPreprocessor(
  runners: readonly AudioRunners[],
  session: AudioSession,
  audio: AudioPcmInput,
  audioPreprocess: (audio: AudioPcmInput, manifest: AudioPreprocessConfig) => AudioFeatures,
  options: AudioPreprocessOptions = {},
): Promise<AudioFeatures> {
  const provider = session.preprocessProviders[0];
  if (!provider) {
    throw new Error("No audio preprocessor provider was selected.");
  }
  throwIfAborted(options.signal);
  const runner = audioPreprocessRunner(runners, provider.name);
  const result = await runner.run(session, audio, audioPreprocess, options);
  if (!result) {
    throw new Error(`Audio preprocess provider ${provider.name} did not return a result.`);
  }
  return result;
}

export async function dispatchAudioEncoder(
  runners: readonly AudioRunners[],
  session: AudioSession,
  features: AudioFeatures,
  options: { signal?: AbortSignal } = {},
): Promise<AudioEncodeResult> {
  const provider = session.providers[0];
  if (!provider) {
    throw new Error("No audio encoder provider was selected.");
  }
  const runner = audioEncoderRunner(runners, provider.name);
  return runner.run(session, features, options);
}

function audioPreprocessRunner(
  runners: readonly AudioRunners[],
  provider: string,
): AudioPreprocessRunner {
  const runner = runners.find((candidate) => candidate.preprocess.provider === provider)?.preprocess;
  if (!runner) {
    throw new Error(`Unsupported audio preprocess provider: ${provider}`);
  }
  return runner;
}

function audioEncoderRunner(
  runners: readonly AudioRunners[],
  provider: string,
): AudioEncoderRunner {
  const runner = runners.find((candidate) => candidate.encoder.provider === provider)?.encoder;
  if (!runner) {
    throw new Error(`Unsupported audio encoder provider: ${provider}`);
  }
  return runner;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Generation was aborted.", "AbortError");
  }
}
