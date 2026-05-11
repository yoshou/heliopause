import type {
  ForwardTrace,
  Gemma4InferenceState,
  Gemma4ModelSession,
} from "../../runtime";
import type { Gemma4ModelManifest } from "../../model";
import {
  forwardGemma4AttentionLayer,
} from "./layers";
import {
  prefetchWasmShardedLayerWeights,
  prefetchWasmShardedOutputWeight,
  registerGemma4CpuExecutionProvider,
} from "./acceleration";

export type Gemma4CpuSegmentRunnerOptions = {
  session: Gemma4ModelSession;
  manifest?: Gemma4ModelManifest;
  epsilon?: number;
  segmentStartLayer?: number;
  segmentEndLayerExclusive?: number;
};

export type Gemma4CpuHiddenResult = {
  hidden: Float32Array;
};

export class Gemma4CpuSegmentRunner {
  readonly segmentStartLayer: number;
  readonly segmentEndLayerExclusive: number;

  private readonly session: Gemma4ModelSession;
  private readonly manifest: Gemma4ModelManifest;
  private readonly epsilon: number;

  constructor(options: Gemma4CpuSegmentRunnerOptions) {
    this.session = options.session;
    registerGemma4CpuExecutionProvider(this.session);
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
    state: Gemma4InferenceState,
    options: { trace?: ForwardTrace; perLayerInputs?: Float32Array; attentionCausal?: boolean } = {},
  ): Promise<Gemma4CpuHiddenResult> {
    let hidden = inputHidden;
    for (let layer = this.segmentStartLayer; layer < this.segmentEndLayerExclusive; layer += 1) {
      const lookaheadLayer = layer + 1;
      if (lookaheadLayer < this.segmentEndLayerExclusive) {
        prefetchWasmShardedLayerWeights(this.session, lookaheadLayer);
      } else if (this.segmentEndLayerExclusive === this.manifest.blockCount) {
        prefetchWasmShardedOutputWeight(this.session);
      }
      hidden = await forwardGemma4AttentionLayer(
        this.session,
        this.manifest,
        state,
        layer,
        hidden,
        positions,
        options.perLayerInputs,
        this.epsilon,
        options.trace,
        options.attentionCausal ?? true,
      );
    }
    return { hidden };
  }

  async runTokenHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: Gemma4InferenceState,
    options: { trace?: ForwardTrace; perLayerInputs?: Float32Array; attentionCausal?: boolean } = {},
  ): Promise<Gemma4CpuHiddenResult> {
    return this.runTokensHidden(inputHidden, positions, state, options);
  }
}
