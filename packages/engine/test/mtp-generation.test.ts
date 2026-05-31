import assert from "node:assert/strict";
import test from "node:test";

import {
  acceptMtpDraft,
  sampleMtpTokenDistribution,
  type MtpTargetVerificationResult,
  type ResolvedGenerationSamplingOptions,
} from "../src/index.ts";

const sampling: Pick<ResolvedGenerationSamplingOptions, "doSample" | "temperature" | "topP" | "topK"> = {
  doSample: true,
  temperature: 1,
  topP: 1,
  topK: 2,
};

test("MTP sampling acceptance accepts drafts and samples the bonus token", () => {
  const acceptance = acceptMtpDraft(
    {
      draftTokenIds: [1],
      draftDistributions: [distribution([1, 0], [2, 0])],
    },
    verification([
      distribution([1, 2], [2, 0]),
      distribution([3, 0], [4, 0]),
    ]),
    sampling,
    rng(0.1, 0.2),
  );

  assert.equal(acceptance.acceptedDraftLength, 1);
  assert.equal(acceptance.nextTokenId, 3);
  assert.equal(acceptance.committedLength, 2);
  assert.equal(acceptance.acceptedAllDrafts, true);
  assert.equal(acceptance.recovered, false);
});

test("MTP sampling acceptance rejects and samples from the recovered distribution", () => {
  const acceptance = acceptMtpDraft(
    {
      draftTokenIds: [1],
      draftDistributions: [distribution([1, 4], [2, 0])],
    },
    verification([
      distribution([1, 0], [2, 4]),
      distribution([3, 0], [4, 0]),
    ]),
    sampling,
    rng(0.5, 0.1),
  );

  assert.equal(acceptance.acceptedDraftLength, 0);
  assert.equal(acceptance.nextTokenId, 2);
  assert.equal(acceptance.committedLength, 1);
  assert.equal(acceptance.acceptedAllDrafts, false);
  assert.equal(acceptance.recovered, true);
});

test("MTP distribution sampling uses generation sampling constraints", () => {
  const tokenId = sampleMtpTokenDistribution(
    distribution([10, 10], [11, 9], [12, 8]),
    { doSample: true, temperature: 1, topP: 1, topK: 1 },
    rng(0.99),
  );

  assert.equal(tokenId, 10);
});

function distribution(...tokens: Array<[number, number]>) {
  return {
    tokens: tokens.map(([id, logit]) => ({ id, logit })),
    vocabularySize: 16,
  };
}

function verification(targetDistributions: ReturnType<typeof distribution>[]): MtpTargetVerificationResult {
  return {
    basePosition: 0,
    verifiedLength: targetDistributions.length,
    targetTokenIds: targetDistributions.map((item) => item.tokens[0]?.id ?? 0),
    targetDistributions,
    bonusTokenId: targetDistributions.at(-1)?.tokens[0]?.id ?? 0,
    bonusDistribution: targetDistributions.at(-1) ?? distribution([0, 0]),
    contexts: targetDistributions.map(() => ({
      position: 0,
      previousHidden: new Float32Array(),
      currentHidden: new Float32Array(),
      targetKv: { layers: [] },
    })),
  };
}

function rng(...values: number[]): () => number {
  let index = 0;
  return () => values[index++] ?? 0;
}
