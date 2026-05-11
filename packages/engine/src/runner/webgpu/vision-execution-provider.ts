import type {
  Gemma4VisionEncodeResult,
  Gemma4VisionPixelValues,
  Gemma4VisionSession,
} from "../../vision";

export async function runGemma4WebGpuVisionEncoder(
  _session: Gemma4VisionSession,
  _pixels: Gemma4VisionPixelValues,
): Promise<Gemma4VisionEncodeResult | undefined> {
  return undefined;
}
