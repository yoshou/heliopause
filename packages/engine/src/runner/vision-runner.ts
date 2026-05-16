import type {
  VisionEncodeResult,
  VisionPixelValues,
  VisionResize,
  VisionSession,
} from "../vision";
import type {
  SegmentRunnerProvider,
} from "./segment-runner";

export type VisionRgbaPreprocessInput = {
  rgba: Uint8ClampedArray;
  sourceWidth: number;
  sourceHeight: number;
  resize: VisionResize;
};

export type VisionPreprocessOptions = {
  signal?: AbortSignal;
};

export type VisionPreprocessRunner = {
  readonly provider: SegmentRunnerProvider;
  run(
    session: VisionSession,
    input: VisionRgbaPreprocessInput,
    visionPreprocess: (
      rgba: Uint8ClampedArray,
      sourceWidth: number,
      sourceHeight: number,
      manifest: VisionSession["manifest"],
    ) => VisionPixelValues,
    options: VisionPreprocessOptions,
  ): Promise<VisionPixelValues | undefined>;
};

export type VisionEncoderRunner = {
  readonly provider: SegmentRunnerProvider;
  run(
    session: VisionSession,
    pixels: VisionPixelValues,
  ): Promise<VisionEncodeResult>;
};

export type VisionRunners = {
  preprocess: VisionPreprocessRunner;
  encoder: VisionEncoderRunner;
};
