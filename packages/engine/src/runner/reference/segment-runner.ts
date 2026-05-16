import type {
  ForwardTrace,
  InferenceState,
  ModelSession,
} from "../../runtime";
import type { ModelManifest } from "../../model";
import {
  forwardAttentionLayer,
} from "./layers";
import type {
  SegmentHiddenResult,
  SegmentRunner,
} from "../segment-runner";

export type ReferenceSegmentRunnerOptions = {
  session: ModelSession;
  manifest?: ModelManifest;
  epsilon?: number;
  segmentStartLayer?: number;
  segmentEndLayerExclusive?: number;
};

export type ReferenceHiddenResult = SegmentHiddenResult;

export class ReferenceSegmentRunner implements SegmentRunner {
  readonly provider = "reference" as const;
  readonly segmentStartLayer: number;
  readonly segmentEndLayerExclusive: number;

  private readonly session: ModelSession;
  private readonly manifest: ModelManifest;
  private readonly epsilon: number;

  constructor(options: ReferenceSegmentRunnerOptions) {
    this.session = options.session;
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
      throw new Error(`Invalid reference layer segment: ${this.segmentStartLayer}..${this.segmentEndLayerExclusive}`);
    }
    if (
      this.segmentEndLayerExclusive > this.segmentStartLayer &&
      !options.session.executionProvider("reference")
    ) {
      throw new Error("Reference segment execution requires an enabled reference provider.");
    }
  }

  async runTokensHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: InferenceState,
    options: { trace?: ForwardTrace; perLayerInputs?: Float32Array; attentionCausal?: boolean } = {},
  ): Promise<ReferenceHiddenResult> {
    let hidden = inputHidden;
    for (let layer = this.segmentStartLayer; layer < this.segmentEndLayerExclusive; layer += 1) {
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
  ): Promise<ReferenceHiddenResult> {
    return this.runTokensHidden(inputHidden, positions, state, options);
  }
}
