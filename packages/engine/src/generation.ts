export type GenerationSamplingOptions = {
  doSample?: boolean;
  temperature?: number;
  topP?: number;
  topK?: number;
  seed?: number;
};

export type TopTokenCandidate = {
  id: number;
  value: number;
};

export type TokenProbability = TopTokenCandidate & {
  probability: number;
};

export type ResolvedGenerationSamplingOptions = {
  doSample: boolean;
  temperature: number;
  topP: number;
  topK: number;
  seed: number;
  logitsTopK: number;
};

export const DEFAULT_GENERATION_CONFIG = {
  doSample: true,
  temperature: 1.0,
  topP: 0.95,
  topK: 64,
  bosTokenId: 2,
  eosTokenIds: [1, 106, 50],
  padTokenId: 0,
} as const;

export const DEFAULT_SAMPLING_SEED = 0x48454c49;

export function resolveGenerationSamplingOptions(
  options: GenerationSamplingOptions = {},
): ResolvedGenerationSamplingOptions {
  const doSample = options.doSample ?? DEFAULT_GENERATION_CONFIG.doSample;
  const temperature = options.temperature ?? DEFAULT_GENERATION_CONFIG.temperature;
  const topP = options.topP ?? DEFAULT_GENERATION_CONFIG.topP;
  const topK = options.topK ?? DEFAULT_GENERATION_CONFIG.topK;
  const seed = options.seed ?? DEFAULT_SAMPLING_SEED;

  if (!Number.isFinite(temperature) || temperature < 0) {
    throw new Error("temperature must be a finite number greater than or equal to 0.");
  }
  if (!Number.isFinite(topP) || topP <= 0 || topP > 1) {
    throw new Error("topP must be a finite number in the range (0, 1].");
  }
  if (!Number.isFinite(topK) || topK < 1) {
    throw new Error("topK must be a positive finite number.");
  }
  if (!Number.isFinite(seed)) {
    throw new Error("seed must be a finite number.");
  }

  const normalizedTopK = Math.max(1, Math.floor(topK));
  const greedy = doSample === false || temperature === 0;

  return {
    doSample,
    temperature,
    topP,
    topK: normalizedTopK,
    seed: Math.floor(seed) >>> 0,
    logitsTopK: greedy ? 1 : normalizedTopK,
  };
}

export function createDeterministicRng(seed: number): () => number {
  let state = Math.floor(seed) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function sampleNextToken(
  candidates: readonly TopTokenCandidate[],
  options: Pick<ResolvedGenerationSamplingOptions, "doSample" | "temperature" | "topP" | "topK">,
  rng: () => number,
): number {
  const distribution = tokenSamplingDistribution(candidates, options);
  if (options.doSample === false || options.temperature === 0) {
    return distribution[0]?.id ?? 0;
  }
  const draw = rng();
  let cumulative = 0;

  for (let index = 0; index < distribution.length; index += 1) {
    cumulative += distribution[index]?.probability ?? 0;
    if (draw < cumulative) {
      return distribution[index]?.id ?? distribution[0]?.id ?? 0;
    }
  }

  return distribution.at(-1)?.id ?? distribution[0]?.id ?? 0;
}

export function tokenSamplingDistribution(
  candidates: readonly TopTokenCandidate[],
  options: Pick<ResolvedGenerationSamplingOptions, "doSample" | "temperature" | "topP" | "topK">,
): TokenProbability[] {
  if (candidates.length === 0) {
    throw new Error("Cannot sample from an empty candidate list.");
  }

  const sorted = [...candidates].sort((left, right) => right.value - left.value || left.id - right.id);
  if (options.doSample === false || options.temperature === 0) {
    const token = sorted[0] ?? candidates[0];
    if (!token) {
      throw new Error("Cannot sample from an empty candidate list.");
    }
    return [{ ...token, probability: 1 }];
  }

  const topKCandidates = sorted.slice(0, Math.max(1, Math.floor(options.topK)));
  const nucleusCandidates = applyTopP(topKCandidates, options.topP);
  const probabilities = softmax(
    nucleusCandidates.map((candidate) => candidate.value / options.temperature),
  );
  return nucleusCandidates.map((candidate, index) => ({
    ...candidate,
    probability: probabilities[index] ?? 0,
  }));
}

function applyTopP(candidates: readonly TopTokenCandidate[], topP: number): readonly TopTokenCandidate[] {
  if (topP >= 1 || candidates.length <= 1) {
    return candidates;
  }

  const probabilities = softmax(candidates.map((candidate) => candidate.value));
  const kept: TopTokenCandidate[] = [];
  let cumulative = 0;
  for (let index = 0; index < candidates.length; index += 1) {
    const candidate = candidates[index];
    if (!candidate) {
      continue;
    }
    kept.push(candidate);
    cumulative += probabilities[index] ?? 0;
    if (cumulative >= topP) {
      break;
    }
  }
  return kept.length > 0 ? kept : candidates.slice(0, 1);
}

function softmax(values: readonly number[]): number[] {
  const max = values.reduce((current, value) => Math.max(current, value), Number.NEGATIVE_INFINITY);
  const expValues = values.map((value) => Math.exp(value - max));
  const sum = expValues.reduce((current, value) => current + value, 0);
  if (sum <= 0 || !Number.isFinite(sum)) {
    return values.map((_, index) => index === 0 ? 1 : 0);
  }
  return expValues.map((value) => value / sum);
}
