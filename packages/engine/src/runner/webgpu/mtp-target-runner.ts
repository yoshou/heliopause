import {
  mapMtpAssistantLayerToTargetKvLayer,
} from "../../model";
import {
  mtpDistributionFromTopTokens,
} from "../../mtp-target";
import type {
  InferenceState,
  ModelSession,
} from "../../runtime";
import type {
  MtpTargetContext,
  MtpTargetPrefillOptions,
  MtpTargetPrefillResult,
  MtpTargetRunner,
  MtpTargetRunners,
  MtpTargetVerificationResult,
  MtpTargetVerifyOptions,
} from "../mtp-target-runner";
import {
  webGpuSegmentRunner,
} from "./execution-provider";

const latestContexts = new WeakMap<InferenceState, MtpTargetContext>();

export function createWebGpuMtpTargetRunners(): MtpTargetRunners {
  return { runner: webGpuMtpTargetRunner };
}

const webGpuMtpTargetRunner: MtpTargetRunner = {
  provider: "webgpu",
  async prefill(session, state, tokenIds, options) {
    throwIfAborted(options.signal);
    if (tokenIds.length === 0) {
      throw new Error("Cannot prefill MTP target from an empty token sequence.");
    }
    const positions = normalizePositions(options.positions, tokenIds.length, state.nextPosition);
    const runner = await fullTargetRunner(session, state);
    const prepared = await runner.prepareTokenIds(tokenIds);
    try {
      const result = await runner.runPreparedInputHidden(prepared, positions, state, {
        computeSelectedToken: true,
        topK: options.logitsTopK,
        readAllHidden: true,
      });
      updateNextPosition(state, positions, tokenIds.length);
      const topTokens = result.topTokens ?? (result.selectedTokenId === undefined ? [] : [{ id: result.selectedTokenId, value: 0 }]);
      const firstTokenId = topTokens[0]?.id ?? result.selectedTokenId;
      if (firstTokenId === undefined) {
        throw new Error("WebGPU MTP target prefill did not produce a first token.");
      }
      const currentHidden = await normalizedHiddenSlice(session, result.hidden, tokenIds.length - 1);
      const previousHidden = tokenIds.length > 1
        ? await normalizedHiddenSlice(session, result.hidden, tokenIds.length - 2)
        : new Float32Array(session.manifest.embeddingLength);
      const context = await contextFromHidden(
        session,
        state,
        currentHidden,
        positions.at(-1) ?? state.nextPosition - 1,
        previousHidden,
        options,
        state.nextPosition,
      );
      latestContexts.set(state, context);
      return {
        firstTokenId,
        firstTokenDistribution: mtpDistributionFromTopTokens(topTokens, targetVocabularySize(session)),
        context,
      };
    } finally {
      prepared.destroy();
    }
  },
  async verify(session, state, draftTokenIds, options) {
    throwIfAborted(options.signal);
    if (draftTokenIds.length === 0) {
      throw new Error("Cannot verify an empty MTP draft.");
    }
    const baseContext = latestContexts.get(state);
    if (!baseContext) {
      throw new Error("MTP target verification requires a prior MTP prefill/finalize context.");
    }
    const runner = await fullTargetRunner(session, state);
    const basePosition = state.nextPosition;
    const positions = Int32Array.from({ length: draftTokenIds.length }, (_, index) => basePosition + index);
    const prepared = await runner.prepareTokenIds(draftTokenIds);
    try {
      const result = await runner.runPreparedInputHidden(prepared, positions, state, {
        computeSelectedToken: draftTokenIds.length === 1,
        computeSelectedTokens: draftTokenIds.length > 1,
        topK: options.logitsTopK,
        readAllHidden: true,
      });
      updateNextPosition(state, positions, draftTokenIds.length);
      const targetTokenIds = result.selectedTokenIds ??
        (result.selectedTokenId === undefined ? [] : [result.selectedTokenId]);
      if (targetTokenIds.length !== draftTokenIds.length) {
        throw new Error(`WebGPU MTP target verification produced ${targetTokenIds.length} tokens for ${draftTokenIds.length} inputs.`);
      }
      const targetDistributions = targetTokenIds.map((id) =>
        mtpDistributionFromTopTokens([{ id, value: 0 }], targetVocabularySize(session))
      );
      const contexts: MtpTargetContext[] = [];
      for (let index = 0; index < draftTokenIds.length; index += 1) {
        const currentHidden = await normalizedHiddenSlice(session, result.hidden, index);
        const previousHidden = index > 0
          ? await normalizedHiddenSlice(session, result.hidden, index - 1)
          : baseContext.currentHidden;
        contexts.push(await contextFromHidden(
          session,
          state,
          currentHidden,
          basePosition + index,
          previousHidden,
          options,
          basePosition + index + 1,
        ));
      }
      const bonusTokenId = targetTokenIds.at(-1);
      const bonusDistribution = targetDistributions.at(-1);
      if (bonusTokenId === undefined || !bonusDistribution) {
        throw new Error("WebGPU MTP target verification did not produce a bonus token.");
      }
      return {
        basePosition,
        verifiedLength: draftTokenIds.length,
        targetTokenIds,
        targetDistributions,
        bonusTokenId,
        bonusDistribution,
        contexts,
      };
    } finally {
      prepared.destroy();
    }
  },
  finalize(state, verification, committedLength) {
    if (!Number.isInteger(committedLength) || committedLength < 1 || committedLength > verification.verifiedLength) {
      throw new Error(`Invalid MTP committed length ${committedLength}.`);
    }
    state.nextPosition = verification.basePosition + committedLength;
    const context = verification.contexts[committedLength - 1];
    if (!context) {
      throw new Error(`Missing MTP context for committed length ${committedLength}.`);
    }
    latestContexts.set(state, context);
    return context;
  },
};

async function fullTargetRunner(session: ModelSession, state: InferenceState) {
  return webGpuSegmentRunner(session, state, {
    segmentStartLayer: 0,
    segmentEndLayerExclusive: session.manifest.blockCount,
  });
}

async function contextFromHidden(
  session: ModelSession,
  state: InferenceState,
  currentHidden: Float32Array,
  position: number,
  previousHidden: Float32Array,
  options: Pick<MtpTargetPrefillOptions, "assistantManifest">,
  tokenCount: number,
): Promise<MtpTargetContext> {
  const runner = await fullTargetRunner(session, state);
  const mappedLayers = Array.from(
    { length: options.assistantManifest.blockCount },
    (_, layer) => mapMtpAssistantLayerToTargetKvLayer(session.manifest, options.assistantManifest, layer),
  );
  return {
    position,
    previousHidden,
    currentHidden,
    targetKv: await runner.readMtpTargetKvView(state, mappedLayers, tokenCount),
  };
}

function normalizePositions(
  positions: MtpTargetPrefillOptions["positions"],
  tokenCount: number,
  basePosition: number,
): Int32Array {
  if (!positions) {
    return Int32Array.from({ length: tokenCount }, (_, index) => basePosition + index);
  }
  const output = positions instanceof Int32Array ? positions : Int32Array.from(positions);
  if (output.length !== tokenCount && output.length !== tokenCount * 4) {
    throw new Error(`Expected ${tokenCount} or ${tokenCount * 4} MTP target positions, got ${output.length}.`);
  }
  return output;
}

function updateNextPosition(state: InferenceState, positions: Int32Array, tokenCount: number): void {
  const tokenPositions = positions.length === tokenCount ? positions : positions.slice(0, tokenCount);
  for (const position of tokenPositions) {
    state.nextPosition = Math.max(state.nextPosition, position + 1);
  }
}

async function normalizedHiddenSlice(session: ModelSession, hidden: Float32Array, tokenIndex: number): Promise<Float32Array> {
  const size = session.manifest.embeddingLength;
  const offset = tokenIndex * size;
  const slice = hidden.slice(offset, offset + size);
  if (slice.length !== size) {
    throw new Error(`MTP target hidden shape mismatch: expected token ${tokenIndex} in ${hidden.length} values.`);
  }
  const weight = await session.readF32Tensor("output_norm.weight");
  let meanSquare = 0;
  for (let index = 0; index < size; index += 1) {
    const value = slice[index] ?? 0;
    meanSquare = Math.fround(meanSquare + Math.fround(value * value));
  }
  const scale = 1 / Math.sqrt(meanSquare / size + session.epsilon);
  const output = new Float32Array(size);
  for (let index = 0; index < size; index += 1) {
    output[index] = Math.fround(Math.fround((slice[index] ?? 0) * scale) * (weight[index] ?? 1));
  }
  return output;
}

function targetVocabularySize(session: ModelSession): number {
  return session.getTensor("token_embd.weight").dimensions[1] ?? 0;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("MTP target execution was aborted.", "AbortError");
  }
}
