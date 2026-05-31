import type {
  InferenceState,
  ModelSession,
} from "./runtime";
import {
  requireMtpTargetRunnerProvider,
} from "./runner/provider";
import type {
  MtpTargetContext,
  MtpTargetPrefillOptions,
  MtpTargetPrefillResult,
  MtpTargetRunner,
  MtpTargetVerifyOptions,
  MtpTargetVerificationResult,
  MtpTokenDistribution,
} from "./runner/mtp-target-runner";

export type {
  MtpTargetContext,
  MtpTargetPrefillOptions,
  MtpTargetPrefillResult,
  MtpTargetRunner,
  MtpTargetRunners,
  MtpTargetVerificationResult,
  MtpTargetVerifyOptions,
  MtpTokenDistribution,
} from "./runner/mtp-target-runner";

export async function prefillMtpTarget(
  session: ModelSession,
  state: InferenceState,
  tokenIds: readonly number[],
  options: MtpTargetPrefillOptions,
): Promise<MtpTargetPrefillResult> {
  return mtpTargetRunner(session).prefill(session, state, tokenIds, options);
}

export async function verifyMtpDraft(
  session: ModelSession,
  state: InferenceState,
  draftTokenIds: readonly number[],
  options: MtpTargetVerifyOptions,
): Promise<MtpTargetVerificationResult> {
  return mtpTargetRunner(session).verify(session, state, draftTokenIds, options);
}

export function finalizeMtpVerification(
  session: ModelSession,
  state: InferenceState,
  verification: MtpTargetVerificationResult,
  committedLength: number,
): MtpTargetContext {
  return mtpTargetRunner(session).finalize(state, verification, committedLength);
}

export function mtpDistributionFromTopTokens(
  tokens: readonly { id: number; value: number }[] | undefined,
  vocabularySize: number,
): MtpTokenDistribution {
  return {
    tokens: (tokens ?? []).map((token) => ({ id: token.id, logit: token.value })),
    vocabularySize,
  };
}

function mtpTargetRunner(session: ModelSession): MtpTargetRunner {
  return requireMtpTargetRunnerProvider(session.providers[0]).createMtpTargetRunners().runner;
}
