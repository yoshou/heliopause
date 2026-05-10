import {
  float16ToFloat32,
  float32ToFloat16,
} from "./quant";

export function rmsNorm(input: Float32Array, weight: Float32Array, epsilon: number): Float32Array {
  if (input.length !== weight.length) {
    throw new Error(`RMSNorm shape mismatch: input=${input.length} weight=${weight.length}`);
  }

  let sumSquares = 0;
  for (const value of input) {
    sumSquares += value * value;
  }

  const scale = 1 / Math.sqrt(sumSquares / input.length + epsilon);
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = (input[index] ?? 0) * scale * (weight[index] ?? 0);
  }
  return output;
}

export function maxTensorDiff(
  actual: Float32Array,
  expected: Float32Array,
): { maxAbs: number; maxRel: number } {
  if (actual.length !== expected.length) {
    throw new Error(`Tensor length mismatch: actual=${actual.length} expected=${expected.length}`);
  }

  let maxAbs = 0;
  let maxRel = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const abs = Math.abs((actual[index] ?? 0) - (expected[index] ?? 0));
    const rel = abs / Math.max(Math.abs(expected[index] ?? 0), 1e-12);
    maxAbs = Math.max(maxAbs, abs);
    maxRel = Math.max(maxRel, rel);
  }
  return { maxAbs, maxRel };
}

export function matMulRows(
  weightRows: Float32Array[],
  inputColumns: Float32Array,
  inputSize: number,
  columnCount: number,
): Float32Array {
  const output = new Float32Array(weightRows.length * columnCount);

  for (let column = 0; column < columnCount; column += 1) {
    const inputOffset = column * inputSize;
    const outputOffset = column * weightRows.length;

    for (let row = 0; row < weightRows.length; row += 1) {
      const weights = weightRows[row];
      if (!weights || weights.length !== inputSize) {
        throw new Error(`Weight row ${row} shape mismatch`);
      }

      let sum = 0;
      for (let index = 0; index < inputSize; index += 1) {
        sum = Math.fround(
          sum + Math.fround((weights[index] ?? 0) * (inputColumns[inputOffset + index] ?? 0)),
        );
      }
      output[outputOffset + row] = sum;
    }
  }

  return output;
}

export function sigmoid(input: Float32Array): Float32Array {
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = 1 / (1 + Math.exp(-(input[index] ?? 0)));
  }
  return output;
}

export function softplus(input: Float32Array): Float32Array {
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    const value = input[index] ?? 0;
    output[index] = value > 20 ? value : Math.log1p(Math.exp(value));
  }
  return output;
}

export function silu(input: Float32Array): Float32Array {
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    const value = input[index] ?? 0;
    output[index] = value / (1 + Math.exp(-value));
  }
  return output;
}

export function l2NormRows(input: Float32Array, rowSize: number, epsilon: number): Float32Array {
  if (input.length % rowSize !== 0) {
    throw new Error(`L2 norm row shape mismatch: input=${input.length} row=${rowSize}`);
  }

  const output = new Float32Array(input.length);
  for (let row = 0; row < input.length / rowSize; row += 1) {
    const offset = row * rowSize;
    let sum = 0;
    for (let index = 0; index < rowSize; index += 1) {
      const value = input[offset + index] ?? 0;
      sum += value * value;
    }
    const scale = 1 / Math.max(Math.sqrt(sum), epsilon);
    for (let index = 0; index < rowSize; index += 1) {
      output[offset + index] = (input[offset + index] ?? 0) * scale;
    }
  }

  return output;
}

export type RopeMultiOptions = {
  headSize: number;
  headCount: number;
  tokenCount: number;
  positions: Int32Array;
  nDims: number;
  sections: readonly number[];
  freqBase: number;
  freqScale?: number;
  nCtxOrig: number;
  extFactor?: number;
  attnFactor?: number;
  betaFast?: number;
  betaSlow?: number;
};

export function ropeMultiMropeNeox(
  input: Float32Array,
  {
    headSize,
    headCount,
    tokenCount,
    positions,
    nDims,
    sections,
    freqBase,
    freqScale = 1,
    nCtxOrig,
    extFactor = 0,
    attnFactor = 1,
    betaFast = 32,
    betaSlow = 1,
  }: RopeMultiOptions,
): Float32Array {
  if (input.length !== headSize * headCount * tokenCount) {
    throw new Error(`RoPE input shape mismatch: ${input.length}`);
  }
  if (positions.length !== tokenCount * 4) {
    throw new Error(`M-RoPE positions must have 4 ids per token: ${positions.length}`);
  }
  if (nDims > headSize || nDims % 2 !== 0) {
    throw new Error(`Invalid RoPE dimension count: ${nDims}`);
  }
  if (sections.length !== 4 || sections[0] + sections[1] + sections[2] + sections[3] <= 0) {
    throw new Error(`Invalid M-RoPE sections: ${sections.join(",")}`);
  }

  const output = new Float32Array(input);
  const thetaScale = Math.pow(freqBase, -2 / nDims);
  const corrDims = ropeYarnCorrDims(nDims, nCtxOrig, freqBase, betaFast, betaSlow);

  for (let token = 0; token < tokenCount; token += 1) {
    const cache = mropeCache({
      pT: positions[token] ?? 0,
      pH: positions[token + tokenCount] ?? 0,
      pW: positions[token + tokenCount * 2] ?? 0,
      pE: positions[token + tokenCount * 3] ?? 0,
      headSize,
      sections,
      freqScale,
      corrDims,
      extFactor,
      attnFactor,
      thetaScale,
    });

    for (let head = 0; head < headCount; head += 1) {
      const rowOffset = (token * headCount + head) * headSize;
      for (let i0 = 0; i0 < nDims; i0 += 2) {
        const ic = i0 / 2;
        const x0 = input[rowOffset + ic] ?? 0;
        const x1 = input[rowOffset + nDims / 2 + ic] ?? 0;
        const cosTheta = cache[i0] ?? 0;
        const sinTheta = cache[i0 + 1] ?? 0;

        output[rowOffset + ic] = Math.fround(
          Math.fround(x0 * cosTheta) - Math.fround(x1 * sinTheta),
        );
        output[rowOffset + nDims / 2 + ic] = Math.fround(
          Math.fround(x0 * sinTheta) + Math.fround(x1 * cosTheta),
        );
      }
    }
  }

  return output;
}

export type GqaAttentionOptions = {
  headSize: number;
  queryHeadCount: number;
  keyValueHeadCount: number;
  tokenCount: number;
  keyValueTokenCount?: number;
  scale: number;
  causal?: boolean;
  mask?: Float32Array;
  valueLayout?: "token-head-dim" | "dim-head-token";
  quantizeQueryForScore?: "f16";
};

export function gqaAttention(
  query: Float32Array,
  key: Float32Array,
  value: Float32Array,
  {
    headSize,
    queryHeadCount,
    keyValueHeadCount,
    tokenCount,
    keyValueTokenCount = tokenCount,
    scale,
    causal = true,
    mask,
    valueLayout = "token-head-dim",
    quantizeQueryForScore,
  }: GqaAttentionOptions,
): Float32Array {
  if (query.length !== tokenCount * queryHeadCount * headSize) {
    throw new Error(`GQA query shape mismatch: ${query.length}`);
  }
  if (key.length !== keyValueTokenCount * keyValueHeadCount * headSize) {
    throw new Error(`GQA key shape mismatch: ${key.length}`);
  }
  if (value.length !== keyValueTokenCount * keyValueHeadCount * headSize) {
    throw new Error(`GQA value shape mismatch: ${value.length}`);
  }
  if (mask && mask.length !== tokenCount * keyValueTokenCount) {
    throw new Error(`GQA mask shape mismatch: ${mask.length}`);
  }
  if (queryHeadCount % keyValueHeadCount !== 0) {
    throw new Error(`GQA head count mismatch: q=${queryHeadCount} kv=${keyValueHeadCount}`);
  }

  const output = new Float32Array(tokenCount * queryHeadCount * headSize);
  const groupSize = queryHeadCount / keyValueHeadCount;
  const scores = new Float32Array(keyValueTokenCount);

  for (let token = 0; token < tokenCount; token += 1) {
    const maxKeyToken = causal && !mask ? token : keyValueTokenCount - 1;

    for (let qHead = 0; qHead < queryHeadCount; qHead += 1) {
      const kvHead = Math.floor(qHead / groupSize);
      const queryOffset = (token * queryHeadCount + qHead) * headSize;
      const queryValues = quantizeQueryForScore === "f16"
        ? quantizedF16Query(query, queryOffset, headSize)
        : query.subarray(queryOffset, queryOffset + headSize);

      let maxScore = -Infinity;
      for (let keyToken = 0; keyToken < keyValueTokenCount; keyToken += 1) {
        if (keyToken > maxKeyToken) {
          scores[keyToken] = -Infinity;
          continue;
        }

        const keyOffset = (keyToken * keyValueHeadCount + kvHead) * headSize;
        let dot = 0;
        for (let index = 0; index < headSize; index += 1) {
          dot = Math.fround(
            dot + Math.fround((queryValues[index] ?? 0) * (key[keyOffset + index] ?? 0)),
          );
        }
        const maskValue = mask ? (mask[token * keyValueTokenCount + keyToken] ?? 0) : 0;
        const score = maskValue === -Infinity ? -Infinity : Math.fround(Math.fround(dot * scale) + maskValue);
        scores[keyToken] = score;
        maxScore = Math.max(maxScore, score);
      }

      let sum = 0;
      for (let keyToken = 0; keyToken <= maxKeyToken; keyToken += 1) {
        const score = scores[keyToken] ?? -Infinity;
        const probability = score === -Infinity ? 0 : Math.exp(score - maxScore);
        scores[keyToken] = probability;
        sum += probability;
      }

      const outputOffset = (token * queryHeadCount + qHead) * headSize;
      for (let index = 0; index < headSize; index += 1) {
        let weighted = 0;
        for (let keyToken = 0; keyToken <= maxKeyToken; keyToken += 1) {
          const valueOffset =
            valueLayout === "token-head-dim"
              ? (keyToken * keyValueHeadCount + kvHead) * headSize + index
              : (index * keyValueHeadCount + kvHead) * keyValueTokenCount + keyToken;
          weighted = Math.fround(
            weighted +
              Math.fround(((scores[keyToken] ?? 0) / sum) * (value[valueOffset] ?? 0)),
          );
        }
        output[outputOffset + index] = weighted;
      }
    }
  }

  return output;
}

function quantizedF16Query(query: Float32Array, offset: number, headSize: number): Float32Array {
  const output = new Float32Array(headSize);
  for (let index = 0; index < headSize; index += 1) {
    output[index] = float16ToFloat32(float32ToFloat16(query[offset + index] ?? 0));
  }
  return output;
}

function mropeCache({
  pT,
  pH,
  pW,
  pE,
  headSize,
  sections,
  freqScale,
  corrDims,
  extFactor,
  attnFactor,
  thetaScale,
}: {
  pT: number;
  pH: number;
  pW: number;
  pE: number;
  headSize: number;
  sections: readonly number[];
  freqScale: number;
  corrDims: [number, number];
  extFactor: number;
  attnFactor: number;
  thetaScale: number;
}): Float32Array {
  const cache = new Float32Array(headSize);
  let thetaT = pT;
  let thetaH = pH;
  let thetaW = pW;
  let thetaE = pE;
  const sectDims = (sections[0] ?? 0) + (sections[1] ?? 0) + (sections[2] ?? 0) + (sections[3] ?? 0);
  const secW = (sections[0] ?? 0) + (sections[1] ?? 0);
  const secE = secW + (sections[2] ?? 0);

  for (let i0 = 0; i0 < headSize; i0 += 2) {
    const sector = (i0 / 2) % sectDims;
    let theta = thetaT;
    if (sector >= (sections[0] ?? 0) && sector < secW) {
      theta = thetaH;
    } else if (sector >= secW && sector < secE) {
      theta = thetaW;
    } else if (sector >= secE) {
      theta = thetaE;
    }

    const [cosTheta, sinTheta] = ropeYarn(
      theta,
      freqScale,
      corrDims,
      i0,
      extFactor,
      attnFactor,
    );
    cache[i0] = cosTheta;
    cache[i0 + 1] = sinTheta;

    thetaT = Math.fround(thetaT * thetaScale);
    thetaH = Math.fround(thetaH * thetaScale);
    thetaW = Math.fround(thetaW * thetaScale);
    thetaE = Math.fround(thetaE * thetaScale);
  }

  return cache;
}

function ropeYarn(
  thetaExtrap: number,
  freqScale: number,
  corrDims: [number, number],
  i0: number,
  extFactor: number,
  mscale: number,
): [number, number] {
  const thetaInterp = Math.fround(freqScale * thetaExtrap);
  let theta = thetaInterp;
  if (extFactor !== 0) {
    const rampMix = Math.fround(ropeYarnRamp(corrDims[0], corrDims[1], i0) * extFactor);
    theta = Math.fround(
      Math.fround(thetaInterp * Math.fround(1 - rampMix)) +
        Math.fround(thetaExtrap * rampMix),
    );
    mscale = Math.fround(mscale * Math.fround(1 + Math.fround(0.1 * Math.log(1 / freqScale))));
  }

  return [
    Math.fround(Math.cos(theta) * mscale),
    Math.fround(Math.sin(theta) * mscale),
  ];
}

function ropeYarnRamp(low: number, high: number, i0: number): number {
  const y = (i0 / 2 - low) / Math.max(0.001, high - low);
  return 1 - Math.min(1, Math.max(0, y));
}

function ropeYarnCorrDims(
  nDims: number,
  nCtxOrig: number,
  freqBase: number,
  betaFast: number,
  betaSlow: number,
): [number, number] {
  return [
    Math.floor(ropeYarnCorrDim(nDims, nCtxOrig, betaFast, freqBase)),
    Math.ceil(ropeYarnCorrDim(nDims, nCtxOrig, betaSlow, freqBase)),
  ];
}

function ropeYarnCorrDim(nDims: number, nCtxOrig: number, nRot: number, base: number): number {
  return (nDims * Math.log(nCtxOrig / (nRot * 2 * Math.PI))) / (2 * Math.log(base));
}
