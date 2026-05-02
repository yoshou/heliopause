import type {
  ForwardTrace,
  Qwen35InferenceState,
  Qwen35ModelSession,
} from "../../runtime";
import type { Qwen35ModelManifest } from "../../model";
import {
  forwardQwen35FullAttentionLayer,
  forwardQwen35RecurrentLayer,
} from "./layers";
import {
  prefetchWasmShardedLayerWeights,
  prefetchWasmShardedOutputWeight,
  registerQwen35CpuExecutionProvider,
} from "./acceleration";

export type Qwen35CpuSegmentRunnerOptions = {
  session: Qwen35ModelSession;
  manifest?: Qwen35ModelManifest;
  epsilon?: number;
  segmentStartLayer?: number;
  segmentEndLayerExclusive?: number;
};

export type Qwen35CpuHiddenResult = {
  hidden: Float32Array;
};

export class Qwen35CpuSegmentRunner {
  readonly segmentStartLayer: number;
  readonly segmentEndLayerExclusive: number;

  private readonly session: Qwen35ModelSession;
  private readonly manifest: Qwen35ModelManifest;
  private readonly epsilon: number;

  constructor(options: Qwen35CpuSegmentRunnerOptions) {
    this.session = options.session;
    registerQwen35CpuExecutionProvider(this.session);
    this.manifest = options.manifest ?? options.session.manifest;
    this.epsilon = options.epsilon ?? options.session.epsilon;
    this.segmentStartLayer = options.segmentStartLayer ?? 0;
    this.segmentEndLayerExclusive = options.segmentEndLayerExclusive ?? this.manifest.blockCount;
    if (
      !Number.isInteger(this.segmentStartLayer) ||
      !Number.isInteger(this.segmentEndLayerExclusive) ||
      this.segmentStartLayer < 0 ||
      this.segmentEndLayerExclusive < this.segmentStartLayer ||
      this.segmentEndLayerExclusive > this.manifest.blockCount
    ) {
      throw new Error(`Invalid CPU layer segment: ${this.segmentStartLayer}..${this.segmentEndLayerExclusive}`);
    }
  }

  async runTokensHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: Qwen35InferenceState,
    options: { trace?: ForwardTrace } = {},
  ): Promise<Qwen35CpuHiddenResult> {
    let hidden = inputHidden;
    for (let layer = this.segmentStartLayer; layer < this.segmentEndLayerExclusive; layer += 1) {
      const lookaheadLayer = layer + 1;
      if (lookaheadLayer < this.segmentEndLayerExclusive) {
        prefetchWasmShardedLayerWeights(this.session, lookaheadLayer);
      } else if (this.segmentEndLayerExclusive === this.manifest.blockCount) {
        prefetchWasmShardedOutputWeight(this.session);
      }
      const isFullAttention = this.manifest.fullAttentionLayers.includes(layer);
      hidden = isFullAttention
        ? await forwardQwen35FullAttentionLayer(
          this.session,
          this.manifest,
          state,
          layer,
          hidden,
          positions,
          this.epsilon,
          options.trace,
        )
        : await forwardQwen35RecurrentLayer(
          this.session,
          this.manifest,
          state,
          layer,
          hidden,
          this.epsilon,
          options.trace,
        );
    }
    return { hidden };
  }

  async runTokenHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: Qwen35InferenceState,
    options: { trace?: ForwardTrace } = {},
  ): Promise<Qwen35CpuHiddenResult> {
    return this.runTokensHidden(inputHidden, positions, state, options);
  }
}
