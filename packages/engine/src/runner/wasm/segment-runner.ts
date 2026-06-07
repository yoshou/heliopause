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
    if (this.shouldChunkSlidingRing(inputHidden, state, options.attentionCausal ?? true)) {
      return this.runTokensHiddenChunked(inputHidden, positions, state, options);
    }
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

  private async runTokensHiddenChunked(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: InferenceState,
    options: { trace?: ForwardTrace; perLayerInputs?: Float32Array; attentionCausal?: boolean },
  ): Promise<WasmHiddenResult> {
    const tokenCount = inputHidden.length / this.manifest.embeddingLength;
    const returnsLastTokenOnly = this.segmentEndLayerExclusive === this.manifest.blockCount;
    const output = returnsLastTokenOnly ? undefined : new Float32Array(inputHidden.length);
    let lastHidden: Float32Array<ArrayBufferLike> = new Float32Array(0);
    for (let chunkStart = 0; chunkStart < tokenCount;) {
      const chunkTokenCount = slidingRingChunkTokenCount(
        this.manifest,
        positions,
        chunkStart,
        tokenCount,
        state,
      );
      let hidden: Float32Array<ArrayBufferLike> = new Float32Array(
        inputHidden.subarray(
          chunkStart * this.manifest.embeddingLength,
          (chunkStart + chunkTokenCount) * this.manifest.embeddingLength,
        ),
      );
      const chunkPositions = sliceTokenPositions(positions, chunkStart, chunkTokenCount, tokenCount);
      const chunkPerLayerInputs = slicePerLayerInputs(
        this.manifest,
        options.perLayerInputs,
        chunkStart,
        chunkTokenCount,
        tokenCount,
      );
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
          chunkPositions,
          chunkPerLayerInputs,
          this.epsilon,
          options.trace,
          true,
        );
      }
      lastHidden = hidden;
      output?.set(hidden, chunkStart * this.manifest.embeddingLength);
      chunkStart += chunkTokenCount;
    }
    return { hidden: returnsLastTokenOnly ? lastHidden : output! };
  }

  private shouldChunkSlidingRing(inputHidden: Float32Array, state: InferenceState, attentionCausal: boolean): boolean {
    if (!attentionCausal || this.manifest.slidingWindow <= 0 || this.manifest.slidingWindow >= state.contextLength) {
      return false;
    }
    const tokenCount = inputHidden.length / this.manifest.embeddingLength;
    if (!Number.isInteger(tokenCount) || tokenCount <= 1) {
      return false;
    }
    for (let layer = this.segmentStartLayer; layer < this.segmentEndLayerExclusive; layer += 1) {
      if (this.manifest.layerHasKv[layer] === true && this.manifest.layerKinds[layer] === "sliding-attention") {
        return true;
      }
    }
    return false;
  }
}

function slidingRingChunkTokenCount(
  manifest: ModelManifest,
  positions: Int32Array,
  chunkStart: number,
  tokenCount: number,
  state: InferenceState,
): number {
  const capacity = minSlidingRingCapacity(manifest, state);
  if (capacity <= 1) {
    return 1;
  }
  let minPosition = Infinity;
  let maxPosition = -Infinity;
  let count = 0;
  for (let token = chunkStart; token < tokenCount; token += 1) {
    const position = tokenPositionAt(positions, token, tokenCount);
    const nextMin = Math.min(minPosition, position);
    const nextMax = Math.max(maxPosition, position);
    if (token > chunkStart && nextMax - nextMin >= capacity) {
      break;
    }
    minPosition = nextMin;
    maxPosition = nextMax;
    count += 1;
  }
  return Math.max(1, count);
}

function minSlidingRingCapacity(manifest: ModelManifest, state: InferenceState): number {
  let capacity = state.contextLength;
  for (const [layer, cache] of state.fullAttention) {
    if (cache.kind === "sliding" && manifest.layerKinds[layer] === "sliding-attention") {
      capacity = Math.min(capacity, cache.capacity);
    }
  }
  return capacity;
}

function sliceTokenPositions(positions: Int32Array, tokenStart: number, tokenCount: number, sourceTokenCount: number): Int32Array {
  if (positions.length === sourceTokenCount) {
    return positions.slice(tokenStart, tokenStart + tokenCount);
  }
  if (positions.length === sourceTokenCount * 4) {
    const output = new Int32Array(tokenCount * 4);
    for (let token = 0; token < tokenCount; token += 1) {
      const sourceToken = tokenStart + token;
      output[token] = positions[sourceToken] ?? 0;
      output[token + tokenCount] = positions[sourceToken + sourceTokenCount] ?? 0;
      output[token + tokenCount * 2] = positions[sourceToken + sourceTokenCount * 2] ?? 0;
      output[token + tokenCount * 3] = positions[sourceToken + sourceTokenCount * 3] ?? 0;
    }
    return output;
  }
  throw new Error(`WASM token batch expects ${sourceTokenCount} or ${sourceTokenCount * 4} positions, got ${positions.length}`);
}

function tokenPositionAt(positions: Int32Array, token: number, tokenCount: number): number {
  if (positions.length === tokenCount || positions.length === tokenCount * 4) {
    return positions[token] ?? 0;
  }
  throw new Error(`WASM token batch expects ${tokenCount} or ${tokenCount * 4} positions, got ${positions.length}`);
}

function slicePerLayerInputs(
  manifest: ModelManifest,
  perLayerInputs: Float32Array | undefined,
  tokenStart: number,
  tokenCount: number,
  sourceTokenCount: number,
): Float32Array | undefined {
  if (!perLayerInputs || manifest.perLayerEmbeddingLength <= 0) {
    return undefined;
  }
  const output = new Float32Array(manifest.blockCount * tokenCount * manifest.perLayerEmbeddingLength);
  for (let layer = 0; layer < manifest.blockCount; layer += 1) {
    const sourceBase = layer * sourceTokenCount * manifest.perLayerEmbeddingLength;
    const targetBase = layer * tokenCount * manifest.perLayerEmbeddingLength;
    output.set(
      perLayerInputs.subarray(
        sourceBase + tokenStart * manifest.perLayerEmbeddingLength,
        sourceBase + (tokenStart + tokenCount) * manifest.perLayerEmbeddingLength,
      ),
      targetBase,
    );
  }
  return output;
}
