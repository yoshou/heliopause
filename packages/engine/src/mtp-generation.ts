import {
  sampleNextToken,
  tokenSamplingDistribution,
  type ResolvedGenerationSamplingOptions,
  type TopTokenCandidate,
} from "./generation";
import {
  runMtpAssistant,
  type MtpAssistantSession,
} from "./mtp-assistant";
import type {
  MtpTargetContext,
  MtpTargetVerificationResult,
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

export type MtpAcceptance = GreedyMtpAcceptance & {
  recovered: boolean;
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
      topK: mtpDraftTopK(sampling),
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

export function sampleMtpTokenDistribution(
  distribution: MtpTokenDistribution,
  sampling: Pick<ResolvedGenerationSamplingOptions, "doSample" | "temperature" | "topP" | "topK">,
  rng: () => number,
): number {
  return sampleNextToken(mtpCandidates(distribution), sampling, rng);
}

export function acceptMtpDraft(
  proposal: Pick<MtpProposal, "draftTokenIds" | "draftDistributions">,
  verification: MtpTargetVerificationResult,
  sampling: Pick<ResolvedGenerationSamplingOptions, "doSample" | "temperature" | "topP" | "topK">,
  rng: () => number,
): MtpAcceptance {
  if (sampling.doSample === false || sampling.temperature === 0) {
    return { ...acceptGreedyMtpDraft(proposal.draftTokenIds, verification.targetTokenIds), recovered: false };
  }

  for (let index = 0; index < proposal.draftTokenIds.length; index += 1) {
    const draftTokenId = proposal.draftTokenIds[index];
    const draftDistribution = proposal.draftDistributions[index];
    const targetDistribution = verification.targetDistributions[index];
    if (draftTokenId === undefined || !draftDistribution || !targetDistribution) {
      throw new Error(`Missing MTP sampling verification data at draft index ${index}.`);
    }
    const draftProbability = tokenProbability(draftDistribution, draftTokenId, sampling);
    const targetProbability = tokenProbability(targetDistribution, draftTokenId, sampling);
    if (
      draftProbability > 0 &&
      Math.min(1, targetProbability / draftProbability) >= rng()
    ) {
      continue;
    }
    return {
      acceptedDraftLength: index,
      nextTokenId: sampleRecoveredToken(targetDistribution, draftDistribution, sampling, rng),
      committedLength: 1 + index,
      acceptedAllDrafts: false,
      recovered: true,
    };
  }

  return {
    acceptedDraftLength: proposal.draftTokenIds.length,
    nextTokenId: sampleMtpTokenDistribution(verification.bonusDistribution, sampling, rng),
    committedLength: 1 + proposal.draftTokenIds.length,
    acceptedAllDrafts: true,
    recovered: false,
  };
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

function mtpDraftTopK(
  sampling: Pick<ResolvedGenerationSamplingOptions, "doSample" | "temperature" | "topK">,
): number {
  return sampling.doSample === false || sampling.temperature === 0 ? 1 : sampling.topK;
}

function tokenProbability(
  distribution: MtpTokenDistribution,
  tokenId: number,
  sampling: Pick<ResolvedGenerationSamplingOptions, "doSample" | "temperature" | "topP" | "topK">,
): number {
  return samplingDistribution(distribution, sampling).find((token) => token.id === tokenId)?.probability ?? 0;
}

function sampleRecoveredToken(
  targetDistribution: MtpTokenDistribution,
  draftDistribution: MtpTokenDistribution,
  sampling: Pick<ResolvedGenerationSamplingOptions, "doSample" | "temperature" | "topP" | "topK">,
  rng: () => number,
): number {
  const draftProbabilities = new Map(
    samplingDistribution(draftDistribution, sampling).map((token) => [token.id, token.probability]),
  );
  const recovered = samplingDistribution(targetDistribution, sampling)
    .map((token) => ({
      id: token.id,
      value: token.value,
      probability: Math.max(0, token.probability - (draftProbabilities.get(token.id) ?? 0)),
    }))
    .filter((token) => token.probability > 0);
  const total = recovered.reduce((sum, token) => sum + token.probability, 0);
  if (total <= 0 || !Number.isFinite(total)) {
    return sampleMtpTokenDistribution(targetDistribution, sampling, rng);
  }
  const draw = rng() * total;
  let cumulative = 0;
  for (const token of recovered) {
    cumulative += token.probability;
    if (draw < cumulative) {
      return token.id;
    }
  }
  return recovered.at(-1)?.id ?? sampleMtpTokenDistribution(targetDistribution, sampling, rng);
}

function samplingDistribution(
  distribution: MtpTokenDistribution,
  sampling: Pick<ResolvedGenerationSamplingOptions, "doSample" | "temperature" | "topP" | "topK">,
) {
  return tokenSamplingDistribution(mtpCandidates(distribution), sampling);
}

function mtpCandidates(distribution: MtpTokenDistribution): TopTokenCandidate[] {
  return distribution.tokens.map((token) => ({ id: token.id, value: token.logit }));
}
