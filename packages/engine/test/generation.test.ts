import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_GENERATION_CONFIG,
  DEFAULT_SAMPLING_SEED,
  createDeterministicRng,
  resolveGenerationSamplingOptions,
  sampleNextToken,
} from "../src/index.ts";

test("generation sampling defaults to the official Gemma 4 config", () => {
  const resolved = resolveGenerationSamplingOptions();

  assert.deepEqual(DEFAULT_GENERATION_CONFIG, {
    doSample: true,
    temperature: 1.0,
    topP: 0.95,
    topK: 64,
    bosTokenId: 2,
    eosTokenIds: [1, 106, 50],
    padTokenId: 0,
  });
  assert.equal(resolved.doSample, true);
  assert.equal(resolved.temperature, 1.0);
  assert.equal(resolved.topP, 0.95);
  assert.equal(resolved.topK, 64);
  assert.equal(resolved.logitsTopK, 64);
  assert.equal(resolved.seed, DEFAULT_SAMPLING_SEED);
});

test("generation sampling validates inputs and maps greedy mode to top-1 logits", () => {
  assert.equal(resolveGenerationSamplingOptions({ doSample: false, topK: 64 }).logitsTopK, 1);
  assert.equal(resolveGenerationSamplingOptions({ temperature: 0, topK: 64 }).logitsTopK, 1);
  assert.equal(resolveGenerationSamplingOptions({ topK: 64.9 }).topK, 64);

  assert.throws(() => resolveGenerationSamplingOptions({ temperature: -1 }), /temperature/);
  assert.throws(() => resolveGenerationSamplingOptions({ topP: 0 }), /topP/);
  assert.throws(() => resolveGenerationSamplingOptions({ topP: 1.1 }), /topP/);
  assert.throws(() => resolveGenerationSamplingOptions({ topK: 0 }), /topK/);
});

test("generation sampling is deterministic and can select non-greedy candidates", () => {
  const rng = createDeterministicRng(DEFAULT_SAMPLING_SEED);
  assert.equal(rng(), 0.9744378970935941);

  const selected = sampleNextToken(
    [
      { id: 10, value: 3 },
      { id: 11, value: 2 },
      { id: 12, value: 1 },
      { id: 13, value: 0 },
    ],
    {
      doSample: true,
      temperature: 1,
      topP: 0.8,
      topK: 4,
    },
    createDeterministicRng(DEFAULT_SAMPLING_SEED),
  );

  assert.equal(selected, 11);
});

test("generation sampling keeps top-p boundary token and greedy returns top-1", () => {
  const candidates = [
    { id: 10, value: 3 },
    { id: 11, value: 2 },
    { id: 12, value: 1 },
  ];

  assert.equal(
    sampleNextToken(candidates, { doSample: false, temperature: 1, topP: 0.1, topK: 3 }, () => 0.99),
    10,
  );
  assert.equal(
    sampleNextToken(candidates, { doSample: true, temperature: 1, topP: 0.8, topK: 3 }, () => 0.99),
    11,
  );
});
