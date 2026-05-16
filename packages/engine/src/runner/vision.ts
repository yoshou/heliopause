import type {
  VisionEncodeResult,
  VisionPixelValues,
  VisionSession,
} from "../vision";
import type {
  VisionEncoderRunner,
  VisionPreprocessOptions,
  VisionPreprocessRunner,
  VisionRunners,
  VisionRgbaPreprocessInput,
} from "./vision-runner";

export type { VisionPreprocessOptions, VisionRgbaPreprocessInput } from "./vision-runner";

export async function dispatchVisionPreprocessor(
  runners: readonly VisionRunners[],
  session: VisionSession,
  input: VisionRgbaPreprocessInput,
  visionPreprocess: (
    rgba: Uint8ClampedArray,
    sourceWidth: number,
    sourceHeight: number,
    manifest: VisionSession["manifest"],
  ) => VisionPixelValues,
  options: VisionPreprocessOptions = {},
): Promise<VisionPixelValues> {
  for (const provider of session.preprocessProviders) {
    throwIfAborted(options.signal);
    const runner = visionPreprocessRunner(runners, provider.name);
    const result = await runner.run(session, input, visionPreprocess, options);
    if (result) {
      return result;
    }
  }
  throw new Error("No vision preprocessor provider was selected.");
}

export async function dispatchVisionEncoder(
  runners: readonly VisionRunners[],
  session: VisionSession,
  pixels: VisionPixelValues,
): Promise<VisionEncodeResult> {
  for (const provider of session.executionProviders) {
    const runner = visionEncoderRunner(runners, provider.name);
    return runner.run(session, pixels);
  }
  throw new Error("No vision encoder provider was selected.");
}

function visionPreprocessRunner(
  runners: readonly VisionRunners[],
  provider: string,
): VisionPreprocessRunner {
  const runner = runners.find((candidate) => candidate.preprocess.provider === provider)?.preprocess;
  if (!runner) {
    throw new Error(`Unsupported vision preprocess provider: ${provider}`);
  }
  return runner;
}

function visionEncoderRunner(
  runners: readonly VisionRunners[],
  provider: string,
): VisionEncoderRunner {
  const runner = runners.find((candidate) => candidate.encoder.provider === provider)?.encoder;
  if (!runner) {
    throw new Error(`Unsupported vision encoder provider: ${provider}`);
  }
  return runner;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Vision preprocessing was aborted.", "AbortError");
  }
}
