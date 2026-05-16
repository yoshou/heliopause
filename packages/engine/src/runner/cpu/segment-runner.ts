import type {
  ForwardTrace,
  InferenceState,
  ModelSession,
} from "../../runtime";
import type { ModelManifest } from "../../model";
import {
  forwardAttentionLayer,
} from "./layers";
import {
  prefetchWasmShardedLayerWeights,
  prefetchWasmShardedOutputWeight,
  registerCpuExecutionProvider,
} from "./acceleration";

export type CpuSegmentRunnerOptions = {
  session: ModelSession;
  manifest?: ModelManifest;
  epsilon?: number;
  segmentStartLayer?: number;
  segmentEndLayerExclusive?: number;
};

export type CpuHiddenResult = {
  hidden: Float32Array;
};

export class CpuSegmentRunner {
  readonly segmentStartLayer: number;
  readonly segmentEndLayerExclusive: number;

  private readonly session: ModelSession;
  private readonly manifest: ModelManifest;
  private readonly epsilon: number;

  constructor(options: CpuSegmentRunnerOptions) {
    this.session = options.session;
    registerCpuExecutionProvider(this.session);
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
    if (
      this.segmentEndLayerExclusive > this.segmentStartLayer &&
      !this.session.executionProvider("wasm") &&
      !this.session.executionProvider("reference")
    ) {
      throw new Error("CPU segment execution requires an enabled wasm or reference provider.");
    }
  }

  async runTokensHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: InferenceState,
    options: { trace?: ForwardTrace; perLayerInputs?: Float32Array; attentionCausal?: boolean } = {},
  ): Promise<CpuHiddenResult> {
    let hidden = inputHidden;
    for (let layer = this.segmentStartLayer; layer < this.segmentEndLayerExclusive; layer += 1) {
      const lookaheadLayer = layer + 1;
      if (lookaheadLayer < this.segmentEndLayerExclusive) {
        prefetchWasmShardedLayerWeights(this.session, lookaheadLayer);
      } else if (this.segmentEndLayerExclusive === this.manifest.blockCount) {
        prefetchWasmShardedOutputWeight(this.session);
      }
      hidden = await forwardAttentionLayer(
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
    state: InferenceState,
    options: { trace?: ForwardTrace; perLayerInputs?: Float32Array; attentionCausal?: boolean } = {},
  ): Promise<CpuHiddenResult> {
    return this.runTokensHidden(inputHidden, positions, state, options);
  }
}
