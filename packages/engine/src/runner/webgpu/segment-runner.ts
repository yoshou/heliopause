import type { GgufTensorReader } from "../../tensor-reader";
import type { Gemma4ModelManifest } from "../../model";
import type { WebGpuTopToken } from "./gpu-types";

export type Gemma4WebGpuSegmentRunnerOptions = {
  tensorReader: GgufTensorReader;
  manifest: Gemma4ModelManifest;
  epsilon: number;
  contextLength: number;
  memoryLimitBytes?: number;
  segmentStartLayer: number;
  segmentEndLayerExclusive?: number;
  loadOutput?: boolean;
};

export type Gemma4WebGpuStateLike = {
  contextLength: number;
  nextPosition: number;
};

export type Gemma4WebGpuTokenResult = {
  topTokens?: WebGpuTopToken[];
};

export type Gemma4WebGpuHiddenResult = {
  hidden: Float32Array;
  topTokens?: WebGpuTopToken[];
};

export class Gemma4WebGpuSegmentRunner {
  readonly segmentStartLayer: number;
  readonly segmentEndLayerExclusive: number;
  readonly residentBytes = 0;

  private constructor(segmentStartLayer: number, segmentEndLayerExclusive: number) {
    this.segmentStartLayer = segmentStartLayer;
    this.segmentEndLayerExclusive = segmentEndLayerExclusive;
  }

  static async create(options: Gemma4WebGpuSegmentRunnerOptions): Promise<Gemma4WebGpuSegmentRunner> {
    const segmentEndLayerExclusive = options.segmentEndLayerExclusive ?? options.manifest.blockCount;
    throw new Error(
      `Gemma4 WebGPU execution is disabled until the native Gemma4 attention, per-layer input, GEGLU, post-norm, and logits-softcap kernels pass logits parity; requested segment ${options.segmentStartLayer}..${segmentEndLayerExclusive}.`,
    );
  }

  async runToken(
    _inputHidden: Float32Array,
    _positions: Int32Array,
    _state: Gemma4WebGpuStateLike,
    _options: { computeTopK?: boolean; topK?: number } = {},
  ): Promise<Gemma4WebGpuTokenResult> {
    throw new Error("Gemma4 WebGPU execution is disabled until logits parity is verified.");
  }

  async runTokenHidden(
    _inputHidden: Float32Array,
    _positions: Int32Array,
    _state: Gemma4WebGpuStateLike,
    _options: { computeTopK?: boolean; topK?: number } = {},
  ): Promise<Gemma4WebGpuHiddenResult> {
    throw new Error("Gemma4 WebGPU execution is disabled until logits parity is verified.");
  }

  async runTokens(
    _inputHidden: Float32Array,
    _positions: Int32Array,
    _state: Gemma4WebGpuStateLike,
    _options: { computeTopK?: boolean; topK?: number } = {},
  ): Promise<Gemma4WebGpuTokenResult> {
    throw new Error("Gemma4 WebGPU execution is disabled until logits parity is verified.");
  }

  async runTokensHidden(
    _inputHidden: Float32Array,
    _positions: Int32Array,
    _state: Gemma4WebGpuStateLike,
    _options: { computeTopK?: boolean; topK?: number } = {},
  ): Promise<Gemma4WebGpuHiddenResult> {
    throw new Error("Gemma4 WebGPU execution is disabled until logits parity is verified.");
  }
}
