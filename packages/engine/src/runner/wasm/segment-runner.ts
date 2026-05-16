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
  registerWasmExecutionProvider,
} from "./acceleration";
import type {
  SegmentHiddenResult,
  SegmentRunner,
} from "../segment-runner";

export type WasmSegmentRunnerOptions = {
  session: ModelSession;
  manifest?: ModelManifest;
  epsilon?: number;
  segmentStartLayer?: number;
  segmentEndLayerExclusive?: number;
};

export type WasmHiddenResult = SegmentHiddenResult;

export class WasmSegmentRunner implements SegmentRunner {
  readonly provider = "wasm" as const;
  readonly segmentStartLayer: number;
  readonly segmentEndLayerExclusive: number;

  private readonly session: ModelSession;
  private readonly manifest: ModelManifest;
  private readonly epsilon: number;

  constructor(options: WasmSegmentRunnerOptions) {
    this.session = options.session;
    registerWasmExecutionProvider(this.session);
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
      throw new Error(`Invalid WASM layer segment: ${this.segmentStartLayer}..${this.segmentEndLayerExclusive}`);
    }
    if (
      this.segmentEndLayerExclusive > this.segmentStartLayer &&
      !this.session.hasProvider("wasm")
    ) {
      throw new Error("WASM segment execution requires an enabled wasm provider.");
    }
  }

  async runTokensHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: InferenceState,
    options: { trace?: ForwardTrace; perLayerInputs?: Float32Array; attentionCausal?: boolean } = {},
  ): Promise<WasmHiddenResult> {
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
  ): Promise<WasmHiddenResult> {
    return this.runTokensHidden(inputHidden, positions, state, options);
  }
}
