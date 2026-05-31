import {
  sampleNextToken,
  type ResolvedGenerationSamplingOptions,
} from "./generation";
import {
  runMtpAssistant,
  type MtpAssistantSession,
} from "./mtp-assistant";
import type {
  MtpTargetContext,
  MtpTokenDistribution,
} from "./mtp-target";
import type {
  ModelSession,
} from "./runtime";
import {
  requireMtpTargetRunnerProvider,
} from "./runner/provider";

export type MtpGenerationOptions = {
  assistantSession: MtpAssistantSession;
  numSpeculativeTokens: number;
};

export type MtpProposal = {
  draftTokenIds: number[];
  draftDistributions: MtpTokenDistribution[];
  feedbackContexts: MtpTargetContext[];
};

export type GreedyMtpAcceptance = {
  acceptedDraftLength: number;
  nextTokenId: number;
  committedLength: number;
  acceptedAllDrafts: boolean;
};

export function validateMtpGenerationOptions(
  session: ModelSession,
  sampling: ResolvedGenerationSamplingOptions,
  mtp: MtpGenerationOptions | undefined,
): void {
  if (!mtp) {
    return;
  }
  if (!Number.isInteger(mtp.numSpeculativeTokens) || mtp.numSpeculativeTokens <= 0) {
    throw new Error("mtp.numSpeculativeTokens must be a positive integer.");
  }
  if (sampling.doSample !== false && sampling.temperature !== 0) {
    throw new Error(
      "Gemma 4 MTP generation currently supports greedy generation only. Pass doSample: false or temperature: 0.",
    );
  }
  requireMtpTargetRunnerProvider(session.providers[0]);
  validateMtpAssistantCompatibility(session, mtp.assistantSession);
}

export function validateMtpAssistantCompatibility(
  session: ModelSession,
  assistantSession: MtpAssistantSession,
): void {
  const target = session.manifest;
  const assistant = assistantSession.manifest;
  if (target.architecture !== "gemma4" || assistant.architecture !== "gemma4_assistant") {
    throw new Error("MTP requires a Gemma 4 target and a Gemma 4 assistant checkpoint.");
  }
  if (target.embeddingLength !== assistant.backboneEmbeddingLength) {
    throw new Error(
      `MTP assistant backbone hidden size ${assistant.backboneEmbeddingLength} does not match target hidden size ${target.embeddingLength}.`,
    );
  }
  if (target.headCountKv !== assistant.headCountKv) {
    throw new Error(
      `MTP assistant KV head count ${assistant.headCountKv} does not match target KV head count ${target.headCountKv}.`,
    );
  }
  const targetVocab = session.getTensor("token_embd.weight").dimensions[1] ?? 0;
  const assistantVocab = assistantSession.getTensor("token_embd.weight").dimensions[1] ?? 0;
  if (targetVocab !== assistantVocab) {
    throw new Error(`MTP assistant vocabulary size ${assistantVocab} does not match target vocabulary size ${targetVocab}.`);
  }
  if (
    assistant.embeddingLength <= 0 ||
    assistantSession.getTensor("token_embd.weight").dimensions[0] !== assistant.embeddingLength
  ) {
    throw new Error("MTP assistant draft lm head shape does not match assistant hidden size.");
  }
}

export async function proposeMtpDraft(
  targetSession: ModelSession,
  mtp: MtpGenerationOptions,
  seedTokenId: number,
  seedContext: MtpTargetContext,
  sampling: Pick<ResolvedGenerationSamplingOptions, "doSample" | "temperature" | "topP" | "topK">,
  rng: () => number,
  options: { signal?: AbortSignal } = {},
): Promise<MtpProposal> {
  const draftTokenIds: number[] = [];
  const draftDistributions: MtpTokenDistribution[] = [];
  const feedbackContexts: MtpTargetContext[] = [];
  const constantPosition = seedContext.position;
  const constantTargetKv = seedContext.targetKv;
  let tokenId = seedTokenId;
  let previousHidden = seedContext.previousHidden;
  let currentHidden = seedContext.currentHidden;

  for (let index = 0; index < mtp.numSpeculativeTokens; index += 1) {
    const result = await runMtpAssistant(mtp.assistantSession, {
      tokenId,
      targetInputEmbedding: await readTargetInputEmbedding(targetSession, tokenId),
      targetPreviousHidden: previousHidden,
      targetCurrentHidden: currentHidden,
      targetKv: constantTargetKv,
      position: constantPosition,
      topK: 1,
    }, options);
    const distribution: MtpTokenDistribution = {
      tokens: result.topTokens.map((token) => ({ id: token.id, logit: token.value })),
      vocabularySize: mtp.assistantSession.getTensor("token_embd.weight").dimensions[1] ?? result.topTokens.length,
      masked: true,
    };
    tokenId = sampleNextToken(result.topTokens, sampling, rng);
    draftTokenIds.push(tokenId);
    draftDistributions.push(distribution);
    feedbackContexts.push({
      position: constantPosition,
      previousHidden,
      currentHidden,
      targetKv: constantTargetKv,
    });
    previousHidden = currentHidden;
    currentHidden = result.backboneHidden;
  }

  return { draftTokenIds, draftDistributions, feedbackContexts };
}

async function readTargetInputEmbedding(session: ModelSession, tokenId: number): Promise<Float32Array> {
  const embedding = await session.readEmbeddingRows([tokenId]);
  const scale = Math.sqrt(session.manifest.embeddingLength);
  for (let index = 0; index < embedding.length; index += 1) {
    embedding[index] = Math.fround((embedding[index] ?? 0) * scale);
  }
  return embedding;
}

export function acceptGreedyMtpDraft(
  draftTokenIds: readonly number[],
  targetTokenIds: readonly number[],
): GreedyMtpAcceptance {
  for (let index = 0; index < draftTokenIds.length; index += 1) {
    const draftToken = draftTokenIds[index];
    const targetToken = targetTokenIds[index];
    if (draftToken === undefined || targetToken === undefined) {
      throw new Error(`Missing MTP greedy verification token at draft index ${index}.`);
    }
    if (draftToken !== targetToken) {
      return {
        acceptedDraftLength: index,
        nextTokenId: targetToken,
        committedLength: 1 + index,
        acceptedAllDrafts: false,
      };
    }
  }
  const bonusToken = targetTokenIds[draftTokenIds.length];
  if (bonusToken === undefined) {
    throw new Error("Missing MTP greedy bonus token.");
  }
  return {
    acceptedDraftLength: draftTokenIds.length,
    nextTokenId: bonusToken,
    committedLength: 1 + draftTokenIds.length,
    acceptedAllDrafts: true,
  };
}
