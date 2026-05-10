export const TOKEN_SLICE_WGSL = `
struct Params {
  rowSize: u32,
  rowIndex: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.rowSize) {
    return;
  }
  outputValues[index] = inputValues[params.rowIndex * params.rowSize + index];
}
`;

export const RMS_NORM_WGSL = `
struct Params {
  epsilon: f32,
  length: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
  var meanSquare = 0.0;
  for (var index = 0u; index < params.length; index = index + 1u) {
    let value = inputValues[index];
    meanSquare = meanSquare + value * value;
  }
  let scale = inverseSqrt(meanSquare / f32(params.length) + params.epsilon);
  for (var index = 0u; index < params.length; index = index + 1u) {
    outputValues[index] = inputValues[index] * scale * weightValues[index];
  }
}
`;

export const RESIDUAL_ADD_WGSL = `
struct Params {
  length: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> leftValues: array<f32>;
@group(0) @binding(1) var<storage, read> rightValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  outputValues[index] = leftValues[index] + rightValues[index];
}
`;

export const FULL_QUERY_NORM_ROPE_WGSL = `
struct Params {
  epsilon: f32,
  freqBase: f32,
  position: f32,
  hasFreqFactors: u32,
  headCount: u32,
  headSize: u32,
  ropeDims: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> qValues: array<f32>;
@group(0) @binding(1) var<storage, read> normWeights: array<f32>;
@group(0) @binding(2) var<storage, read> freqFactors: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> queryValues: array<f32>;

fn normed(head: u32, dim: u32, scale: f32) -> f32 {
  let base = head * params.headSize;
  return qValues[base + dim] * scale * normWeights[dim];
}

fn ropeFactor(index: u32) -> f32 {
  if (params.hasFreqFactors == 0u) {
    return 1.0;
  }
  return freqFactors[index];
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let head = id.x;
  if (head >= params.headCount) {
    return;
  }
  let qBase = head * params.headSize;
  let outBase = head * params.headSize;
  var meanSquare = 0.0;
  for (var dim = 0u; dim < params.headSize; dim = dim + 1u) {
    let value = qValues[qBase + dim];
    meanSquare = meanSquare + value * value;
  }
  let scale = inverseSqrt(meanSquare / f32(params.headSize) + params.epsilon);
  for (var dim = 0u; dim < params.headSize; dim = dim + 1u) {
    queryValues[outBase + dim] = normed(head, dim, scale);
  }
  let thetaScale = pow(params.freqBase, -2.0 / f32(params.ropeDims));
  var theta = params.position;
  for (var i0 = 0u; i0 < params.ropeDims; i0 = i0 + 2u) {
    let ic = i0 / 2u;
    let x0 = normed(head, ic, scale);
    let x1 = normed(head, params.ropeDims / 2u + ic, scale);
    let thetaWithFactor = theta / ropeFactor(ic);
    let cosTheta = cos(thetaWithFactor);
    let sinTheta = sin(thetaWithFactor);
    queryValues[outBase + ic] = x0 * cosTheta - x1 * sinTheta;
    queryValues[outBase + params.ropeDims / 2u + ic] = x0 * sinTheta + x1 * cosTheta;
    theta = theta * thetaScale;
  }
}
`;

export const FULL_KV_UPDATE_WGSL = `
struct Params {
  epsilon: f32,
  freqBase: f32,
  position: f32,
  hasFreqFactors: u32,
  headCount: u32,
  headSize: u32,
  valueSize: u32,
  ropeDims: u32,
  tokenPosition: u32,
  contextLength: u32,
};

@group(0) @binding(0) var<storage, read> kProjectionValues: array<f32>;
@group(0) @binding(1) var<storage, read> vProjectionValues: array<f32>;
@group(0) @binding(2) var<storage, read> normWeights: array<f32>;
@group(0) @binding(3) var<storage, read> freqFactors: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> keyCache: array<f32>;
@group(0) @binding(6) var<storage, read_write> valueCache: array<f32>;

fn normed(head: u32, dim: u32, scale: f32) -> f32 {
  let base = head * params.headSize;
  return kProjectionValues[base + dim] * scale * normWeights[dim];
}

fn ropeFactor(index: u32) -> f32 {
  if (params.hasFreqFactors == 0u) {
    return 1.0;
  }
  return freqFactors[index];
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let head = id.x;
  if (head >= params.headCount) {
    return;
  }
  let base = head * params.headSize;
  var meanSquare = 0.0;
  for (var dim = 0u; dim < params.headSize; dim = dim + 1u) {
    let value = kProjectionValues[base + dim];
    meanSquare = meanSquare + value * value;
  }
  let scale = inverseSqrt(meanSquare / f32(params.headSize) + params.epsilon);
  let keyBase = (params.tokenPosition * params.headCount + head) * params.headSize;
  for (var dim = 0u; dim < params.headSize; dim = dim + 1u) {
    keyCache[keyBase + dim] = normed(head, dim, scale);
  }
  var valueMeanSquare = 0.0;
  let valueBase = head * params.valueSize;
  for (var dim = 0u; dim < params.valueSize; dim = dim + 1u) {
    let value = vProjectionValues[valueBase + dim];
    valueMeanSquare = valueMeanSquare + value * value;
  }
  let valueScale = inverseSqrt(valueMeanSquare / f32(params.valueSize) + params.epsilon);
  for (var dim = 0u; dim < params.valueSize; dim = dim + 1u) {
    valueCache[(dim * params.headCount + head) * params.contextLength + params.tokenPosition] =
      vProjectionValues[valueBase + dim] * valueScale;
  }
  let thetaScale = pow(params.freqBase, -2.0 / f32(params.ropeDims));
  var theta = params.position;
  for (var i0 = 0u; i0 < params.ropeDims; i0 = i0 + 2u) {
    let ic = i0 / 2u;
    let x0 = normed(head, ic, scale);
    let x1 = normed(head, params.ropeDims / 2u + ic, scale);
    let thetaWithFactor = theta / ropeFactor(ic);
    let cosTheta = cos(thetaWithFactor);
    let sinTheta = sin(thetaWithFactor);
    keyCache[keyBase + ic] = x0 * cosTheta - x1 * sinTheta;
    keyCache[keyBase + params.ropeDims / 2u + ic] = x0 * sinTheta + x1 * cosTheta;
    theta = theta * thetaScale;
  }
}
`;

export const TOPK_WGSL = `
struct Params {
  rowCount: u32,
  rowOffset: u32,
  slot: u32,
  candidateOffset: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
  let slot = params.slot;
  var bestId = 0u;
  var bestValue = -3.4028234663852886e38;
  for (var row = 0u; row < params.rowCount; row = row + 1u) {
    let value = logits[row];
    var alreadyUsed = false;
    for (var prev = 0u; prev < slot; prev = prev + 1u) {
      let prevId = u32(outputValues[params.candidateOffset + prev * 2u]);
      if (prevId == row + params.rowOffset) {
        alreadyUsed = true;
      }
    }
    if (!alreadyUsed && value > bestValue) {
      bestValue = value;
      bestId = row;
    }
  }
  let outputBase = params.candidateOffset + slot * 2u;
  outputValues[outputBase] = f32(bestId + params.rowOffset);
  outputValues[outputBase + 1u] = bestValue;
}
`;

export const Q8_0_MATMUL_WGSL = `
struct Params {
  inputSize: u32,
  rowCount: u32,
  columnCount: u32,
  blockCount: u32,
  rowByteLength: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> weightWords: array<u32>;
@group(0) @binding(1) var<storage, read> inputScales: array<f32>;
@group(0) @binding(2) var<storage, read> inputQs: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;

fn byteAt(index: u32) -> u32 {
  let word = weightWords[index / 4u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 255u;
}

fn signedByteAt(index: u32) -> i32 {
  let value = i32(byteAt(index));
  if (value >= 128) {
    return value - 256;
  }
  return value;
}

fn f16At(index: u32) -> f32 {
  let bits = byteAt(index) | (byteAt(index + 1u) << 8u);
  let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
  let exponent = (bits >> 10u) & 31u;
  let fraction = bits & 1023u;
  if (exponent == 0u) {
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 31u) {
    return sign * 65504.0;
  }
  return sign * exp2(f32(exponent) - 15.0) * (1.0 + f32(fraction) / 1024.0);
}

@compute @workgroup_size(8, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let column = id.y;
  if (row >= params.rowCount || column >= params.columnCount) {
    return;
  }
  var sum = 0.0;
  for (var block = 0u; block < params.blockCount; block = block + 1u) {
    let weightBase = row * params.rowByteLength + block * 34u;
    let weightScale = f16At(weightBase);
    let inputScale = inputScales[column * params.blockCount + block];
    var isum = 0i;
    for (var index = 0u; index < 32u; index = index + 1u) {
      let w = signedByteAt(weightBase + 2u + index);
      let q = inputQs[column * params.inputSize + block * 32u + index];
      isum = isum + w * q;
    }
    sum = sum + f32(isum) * weightScale * inputScale;
  }
  outputValues[column * params.rowCount + row] = sum;
}
`;

export const SWIGLU_WGSL = `
struct Params {
  length: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> gateValues: array<f32>;
@group(0) @binding(1) var<storage, read> upValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  let gate = gateValues[index];
  outputValues[index] = (gate / (1.0 + exp(-gate))) * upValues[index];
}
`;

export const GEGLU_WGSL = `
struct Params {
  length: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> gateValues: array<f32>;
@group(0) @binding(1) var<storage, read> upValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

fn f16BitsToF32(bits: u32) -> f32 {
  let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
  let exponent = (bits >> 10u) & 31u;
  let fraction = bits & 1023u;
  if (exponent == 0u) {
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 31u) {
    return sign * 65504.0;
  }
  return sign * exp2(f32(exponent) - 15.0) * (1.0 + f32(fraction) / 1024.0);
}

fn f32ToF16Bits(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  let sign = (bits >> 16u) & 0x8000u;
  let absBits = bits & 0x7fffffffu;
  if (absBits == 0u) {
    return sign;
  }
  if (absBits >= 0x7f800000u) {
    return sign | 0x7c00u;
  }
  var exponent = i32((absBits >> 23u) & 255u) - 127 + 15;
  let mantissa = absBits & 0x7fffffu;
  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    let shifted = (mantissa | 0x800000u) >> u32(1 - exponent);
    return sign | ((shifted + 0x1000u) >> 13u);
  }
  var halfMantissa = (mantissa + 0x1000u) >> 13u;
  if (halfMantissa == 0x400u) {
    halfMantissa = 0u;
    exponent = exponent + 1;
  }
  if (exponent >= 31) {
    return sign | 0x7c00u;
  }
  return sign | (u32(exponent) << 10u) | halfMantissa;
}

fn castF16(value: f32) -> f32 {
  return f16BitsToF32(f32ToF16Bits(value));
}

fn gelu(value: f32) -> f32 {
  if (value <= -10.0) {
    return 0.0;
  }
  if (value >= 10.0) {
    return value;
  }
  let x = castF16(value);
  let inner = sqrt(2.0 / 3.141592653589793) * x * (1.0 + 0.044715 * x * x);
  return castF16(0.5 * x * (1.0 + tanh(inner)));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  outputValues[index] = gelu(gateValues[index]) * upValues[index];
}
`;

export const GELU_WGSL = `
struct Params {
  length: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

fn f16BitsToF32(bits: u32) -> f32 {
  let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
  let exponent = (bits >> 10u) & 31u;
  let fraction = bits & 1023u;
  if (exponent == 0u) {
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 31u) {
    return sign * 65504.0;
  }
  return sign * exp2(f32(exponent) - 15.0) * (1.0 + f32(fraction) / 1024.0);
}

fn f32ToF16Bits(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  let sign = (bits >> 16u) & 0x8000u;
  let absBits = bits & 0x7fffffffu;
  if (absBits == 0u) {
    return sign;
  }
  if (absBits >= 0x7f800000u) {
    return sign | 0x7c00u;
  }
  var exponent = i32((absBits >> 23u) & 255u) - 127 + 15;
  let mantissa = absBits & 0x7fffffu;
  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    let shifted = (mantissa | 0x800000u) >> u32(1 - exponent);
    return sign | ((shifted + 0x1000u) >> 13u);
  }
  var halfMantissa = (mantissa + 0x1000u) >> 13u;
  if (halfMantissa == 0x400u) {
    halfMantissa = 0u;
    exponent = exponent + 1;
  }
  if (exponent >= 31) {
    return sign | 0x7c00u;
  }
  return sign | (u32(exponent) << 10u) | halfMantissa;
}

fn castF16(value: f32) -> f32 {
  return f16BitsToF32(f32ToF16Bits(value));
}

fn gelu(value: f32) -> f32 {
  if (value <= -10.0) {
    return 0.0;
  }
  if (value >= 10.0) {
    return value;
  }
  let x = castF16(value);
  let inner = sqrt(2.0 / 3.141592653589793) * x * (1.0 + 0.044715 * x * x);
  return castF16(0.5 * x * (1.0 + tanh(inner)));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  outputValues[index] = gelu(inputValues[index]);
}
`;

export const ELEMENTWISE_MUL_WGSL = `
struct Params {
  length: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> leftValues: array<f32>;
@group(0) @binding(1) var<storage, read> rightValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  outputValues[index] = leftValues[index] * rightValues[index];
}
`;

export const SCALE_WGSL = `
struct Params {
  length: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> scaleValue: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  outputValues[index] = inputValues[index] * scaleValue[0];
}
`;

export const F16_CAST_WGSL = `
struct Params {
  length: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

fn f16BitsToF32(bits: u32) -> f32 {
  let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
  let exponent = (bits >> 10u) & 31u;
  let fraction = bits & 1023u;
  if (exponent == 0u) {
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 31u) {
    return sign * 65504.0;
  }
  return sign * exp2(f32(exponent) - 15.0) * (1.0 + f32(fraction) / 1024.0);
}

fn f32ToF16Bits(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  let sign = (bits >> 16u) & 0x8000u;
  let absBits = bits & 0x7fffffffu;
  if (absBits == 0u) {
    return sign;
  }
  if (absBits >= 0x7f800000u) {
    return sign | 0x7c00u;
  }
  var exponent = i32((absBits >> 23u) & 255u) - 127 + 15;
  let mantissa = absBits & 0x7fffffu;
  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    let shifted = (mantissa | 0x800000u) >> u32(1 - exponent);
    return sign | ((shifted + 0x1000u) >> 13u);
  }
  var halfMantissa = (mantissa + 0x1000u) >> 13u;
  if (halfMantissa == 0x400u) {
    halfMantissa = 0u;
    exponent = exponent + 1;
  }
  if (exponent >= 31) {
    return sign | 0x7c00u;
  }
  return sign | (u32(exponent) << 10u) | halfMantissa;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  outputValues[index] = f16BitsToF32(f32ToF16Bits(inputValues[index]));
}
`;

export const F32_MATMUL_WGSL = `
struct Params {
  inputSize: u32,
  rowCount: u32,
  columnCount: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> weightValues: array<f32>;
@group(0) @binding(1) var<storage, read> inputValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(8, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let column = id.y;
  if (row >= params.rowCount || column >= params.columnCount) {
    return;
  }
  var sum = 0.0;
  for (var index = 0u; index < params.inputSize; index = index + 1u) {
    sum = sum + weightValues[row * params.inputSize + index] * inputValues[column * params.inputSize + index];
  }
  outputValues[column * params.rowCount + row] = sum;
}
`;

export const DELTA_GATE_WGSL = `
struct Params {
  valueHeadCount: u32,
  tokenCount: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> alphaValues: array<f32>;
@group(0) @binding(1) var<storage, read> betaValues: array<f32>;
@group(0) @binding(2) var<storage, read> dtBias: array<f32>;
@group(0) @binding(3) var<storage, read> ssmA: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> gateValues: array<f32>;
@group(0) @binding(6) var<storage, read_write> betaSigmoidValues: array<f32>;

fn softplus(value: f32) -> f32 {
  if (value > 20.0) {
    return value;
  }
  if (value < -20.0) {
    return exp(value);
  }
  return log(1.0 + exp(value));
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let head = id.x;
  let token = id.y;
  if (head >= params.valueHeadCount || token >= params.tokenCount) {
    return;
  }
  let index = token * params.valueHeadCount + head;
  gateValues[index] = softplus(alphaValues[index] + dtBias[head]) * ssmA[head];
  let beta = betaValues[index];
  betaSigmoidValues[index] = 1.0 / (1.0 + exp(-beta));
}
`;

export const TOP1_WGSL = `
struct Params {
  rowCount: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
  var bestId = 0u;
  var bestValue = -3.4028234663852886e38;
  for (var row = 0u; row < params.rowCount; row = row + 1u) {
    let value = logits[row];
    if (value > bestValue) {
      bestValue = value;
      bestId = row;
    }
  }
  outputValues[0] = f32(bestId);
  outputValues[1] = bestValue;
}
`;

export const Q8_K_QUANTIZE_WGSL = `
struct Params {
  inputSize: u32,
  columnCount: u32,
  blockCount: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputScales: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputQs: array<i32>;
@group(0) @binding(3) var<storage, read_write> outputBsums: array<i32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x;
  let block = id.y;
  if (column >= params.columnCount || block >= params.blockCount) {
    return;
  }
  let base = column * params.inputSize + block * 256u;
  var maxValue = 0.0;
  var amax = 0.0;
  for (var index = 0u; index < 256u; index = index + 1u) {
    let value = inputValues[base + index];
    let absValue = abs(value);
    if (absValue > amax) {
      amax = absValue;
      maxValue = value;
    }
  }
  let blockIndex = column * params.blockCount + block;
  if (amax == 0.0) {
    outputScales[blockIndex] = 0.0;
    for (var index = 0u; index < 256u; index = index + 1u) {
      outputQs[base + index] = 0;
    }
    for (var group = 0u; group < 16u; group = group + 1u) {
      outputBsums[blockIndex * 16u + group] = 0;
    }
    return;
  }
  let inverseScale = -127.0 / maxValue;
  outputScales[blockIndex] = 1.0 / inverseScale;
  for (var group = 0u; group < 16u; group = group + 1u) {
    var sum = 0i;
    for (var lane = 0u; lane < 16u; lane = lane + 1u) {
      let index = group * 16u + lane;
      let q = min(127i, i32(round(inverseScale * inputValues[base + index])));
      outputQs[base + index] = q;
      sum = sum + q;
    }
    outputBsums[blockIndex * 16u + group] = sum;
  }
}
`;

export const Q8_0_QUANTIZE_WGSL = `
struct Params {
  inputSize: u32,
  columnCount: u32,
  blockCount: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputScales: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputQs: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;

fn f16BitsToF32(bits: u32) -> f32 {
  let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
  let exponent = (bits >> 10u) & 31u;
  let fraction = bits & 1023u;
  if (exponent == 0u) {
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 31u) {
    return sign * 65504.0;
  }
  return sign * exp2(f32(exponent) - 15.0) * (1.0 + f32(fraction) / 1024.0);
}

fn f32ToF16Bits(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  let sign = (bits >> 16u) & 0x8000u;
  let absBits = bits & 0x7fffffffu;
  if (absBits == 0u) {
    return sign;
  }
  if (absBits >= 0x7f800000u) {
    return sign | 0x7c00u;
  }
  var exponent = i32((absBits >> 23u) & 255u) - 127 + 15;
  let mantissa = absBits & 0x7fffffu;
  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    let shifted = (mantissa | 0x800000u) >> u32(1 - exponent);
    return sign | ((shifted + 0x1000u) >> 13u);
  }
  var halfMantissa = (mantissa + 0x1000u) >> 13u;
  if (halfMantissa == 0x400u) {
    halfMantissa = 0u;
    exponent = exponent + 1;
  }
  if (exponent >= 31) {
    return sign | 0x7c00u;
  }
  return sign | (u32(exponent) << 10u) | halfMantissa;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x;
  let block = id.y;
  if (column >= params.columnCount || block >= params.blockCount) {
    return;
  }
  let base = column * params.inputSize + block * 32u;
  var amax = 0.0;
  for (var index = 0u; index < 32u; index = index + 1u) {
    amax = max(amax, abs(inputValues[base + index]));
  }
  let blockIndex = column * params.blockCount + block;
  let scale = f16BitsToF32(f32ToF16Bits(amax / 127.0));
  outputScales[blockIndex] = scale;
  let inverseScale = select(0.0, 1.0 / scale, scale != 0.0);
  for (var index = 0u; index < 32u; index = index + 1u) {
    outputQs[base + index] = i32(round(inputValues[base + index] * inverseScale));
  }
}
`;

export const SSM_NORM_GATE_WGSL = `
struct Params {
  epsilon: f32,
  rowCount: u32,
  columnCount: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> deltaValues: array<f32>;
@group(0) @binding(1) var<storage, read> zValues: array<f32>;
@group(0) @binding(2) var<storage, read> normWeights: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x;
  if (column >= params.columnCount) {
    return;
  }
  let base = column * params.rowCount;
  var meanSquare = 0.0;
  for (var row = 0u; row < params.rowCount; row = row + 1u) {
    let value = deltaValues[base + row];
    meanSquare = meanSquare + value * value;
  }
  let scale = inverseSqrt(meanSquare / f32(params.rowCount) + params.epsilon);
  for (var row = 0u; row < params.rowCount; row = row + 1u) {
    let index = base + row;
    let z = zValues[index];
    outputValues[index] = deltaValues[index] * scale * normWeights[row] * (z / (1.0 + exp(-z)));
  }
}
`;

export const QKV_CONV_SPLIT_WGSL = `
struct Params {
  tokenCount: u32,
  convDim: u32,
  kernelSize: u32,
  stateSize: u32,
  groupCount: u32,
  valueDim: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> qkvValues: array<f32>;
@group(0) @binding(1) var<storage, read> convState: array<f32>;
@group(0) @binding(2) var<storage, read> convKernel: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> qValues: array<f32>;
@group(0) @binding(5) var<storage, read_write> kValues: array<f32>;
@group(0) @binding(6) var<storage, read_write> vValues: array<f32>;
@group(0) @binding(7) var<storage, read_write> newConvState: array<f32>;

fn convInputValue(channel: u32, inputIndex: u32) -> f32 {
  let history = params.kernelSize - 1u;
  if (inputIndex < history) {
    return convState[channel * history + inputIndex];
  }
  let token = inputIndex - history;
  return qkvValues[token * params.convDim + channel];
}

fn convSilu(channel: u32, token: u32) -> f32 {
  var sum = 0.0;
  for (var k = 0u; k < params.kernelSize; k = k + 1u) {
    sum = sum + convInputValue(channel, token + k) * convKernel[channel * params.kernelSize + k];
  }
  return sum / (1.0 + exp(-sum));
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let work = id.x;
  let mode = id.y;
  let keyDim = params.stateSize * params.groupCount;
  let history = params.kernelSize - 1u;
  if (mode == 0u) {
    let rowCount = params.tokenCount * params.groupCount;
    if (work >= rowCount) {
      return;
    }
    let token = work / params.groupCount;
    let group = work - token * params.groupCount;
    var qSum = 0.0;
    var kSum = 0.0;
    for (var index = 0u; index < params.stateSize; index = index + 1u) {
      let channel = group * params.stateSize + index;
      let q = convSilu(channel, token);
      let k = convSilu(keyDim + channel, token);
      qSum = qSum + q * q;
      kSum = kSum + k * k;
    }
    let qScale = 1.0 / max(sqrt(qSum), 0.000001);
    let kScale = 1.0 / max(sqrt(kSum), 0.000001);
    for (var index = 0u; index < params.stateSize; index = index + 1u) {
      let channel = group * params.stateSize + index;
      let outIndex = token * keyDim + group * params.stateSize + index;
      qValues[outIndex] = convSilu(channel, token) * qScale;
      kValues[outIndex] = convSilu(keyDim + channel, token) * kScale;
    }
    return;
  }
  if (mode == 1u) {
    let valueHeadCount = params.valueDim / params.stateSize;
    let rowCount = params.tokenCount * valueHeadCount;
    if (work >= rowCount) {
      return;
    }
    let token = work / valueHeadCount;
    let head = work - token * valueHeadCount;
    for (var index = 0u; index < params.stateSize; index = index + 1u) {
      let valueIndex = head * params.stateSize + index;
      let channel = keyDim * 2u + valueIndex;
      vValues[token * params.valueDim + valueIndex] = convSilu(channel, token);
    }
    return;
  }
  if (mode == 2u) {
    if (work >= params.convDim) {
      return;
    }
    for (var index = 0u; index < history; index = index + 1u) {
      let source = params.tokenCount + index;
      newConvState[work * history + index] = convInputValue(work, source);
    }
  }
}
`;


export const FULL_ATTENTION_SCORE_WGSL = `
struct Params {
  scale: f32,
  headSize: u32,
  queryHeadCount: u32,
  keyValueHeadCount: u32,
  keyValueTokenCount: u32,
  contextLength: u32,
  tokenPosition: u32,
  slidingWindow: u32,
};

@group(0) @binding(0) var<storage, read> queryValues: array<f32>;
@group(0) @binding(1) var<storage, read> keyValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> probabilityValues: array<f32>;

fn f16BitsToF32(bits: u32) -> f32 {
  let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
  let exponent = (bits >> 10u) & 31u;
  let fraction = bits & 1023u;
  if (exponent == 0u) {
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 31u) {
    return sign * 65504.0;
  }
  return sign * exp2(f32(exponent) - 15.0) * (1.0 + f32(fraction) / 1024.0);
}

fn f32ToF16Bits(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  let sign = (bits >> 16u) & 0x8000u;
  let absBits = bits & 0x7fffffffu;
  if (absBits == 0u) {
    return sign;
  }
  if (absBits >= 0x7f800000u) {
    return sign | 0x7c00u;
  }
  var exponent = i32((absBits >> 23u) & 255u) - 127 + 15;
  let mantissa = absBits & 0x7fffffu;
  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    let shifted = (mantissa | 0x800000u) >> u32(1 - exponent);
    return sign | ((shifted + 0x1000u) >> 13u);
  }
  var halfMantissa = (mantissa + 0x1000u) >> 13u;
  if (halfMantissa == 0x400u) {
    halfMantissa = 0u;
    exponent = exponent + 1;
  }
  if (exponent >= 31) {
    return sign | 0x7c00u;
  }
  return sign | (u32(exponent) << 10u) | halfMantissa;
}

fn queryValue(index: u32) -> f32 {
  return f16BitsToF32(f32ToF16Bits(queryValues[index]));
}

fn attentionScore(qHead: u32, kvHead: u32, keyToken: u32) -> f32 {
  let minPosition = select(
    0u,
    params.tokenPosition + 1u - params.slidingWindow,
    params.slidingWindow != 0u && params.tokenPosition + 1u > params.slidingWindow,
  );
  if (keyToken > params.tokenPosition || keyToken < minPosition) {
    return -3.4028234663852886e38;
  }
  let queryOffset = qHead * params.headSize;
  let keyOffset = (keyToken * params.keyValueHeadCount + kvHead) * params.headSize;
  var dot = 0.0;
  for (var dim = 0u; dim < params.headSize; dim = dim + 1u) {
    dot = dot + queryValue(queryOffset + dim) * keyValues[keyOffset + dim];
  }
  return dot * params.scale;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let qHead = id.x;
  if (qHead >= params.queryHeadCount) {
    return;
  }
  let groupSize = params.queryHeadCount / params.keyValueHeadCount;
  let kvHead = qHead / groupSize;
  var maxScore = -3.4028234663852886e38;
  for (var keyToken = 0u; keyToken < params.keyValueTokenCount; keyToken = keyToken + 1u) {
    maxScore = max(maxScore, attentionScore(qHead, kvHead, keyToken));
  }
  var sum = 0.0;
  for (var keyToken = 0u; keyToken < params.keyValueTokenCount; keyToken = keyToken + 1u) {
    let probability = exp(attentionScore(qHead, kvHead, keyToken) - maxScore);
    probabilityValues[qHead * params.keyValueTokenCount + keyToken] = probability;
    sum = sum + probability;
  }
  for (var keyToken = 0u; keyToken < params.keyValueTokenCount; keyToken = keyToken + 1u) {
    let index = qHead * params.keyValueTokenCount + keyToken;
    probabilityValues[index] = probabilityValues[index] / sum;
  }
}
`;

export const FULL_ATTENTION_APPLY_WGSL = `
struct Params {
  scale: f32,
  valueSize: u32,
  queryHeadCount: u32,
  keyValueHeadCount: u32,
  keyValueTokenCount: u32,
  contextLength: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> valueValues: array<f32>;
@group(0) @binding(1) var<storage, read> probabilityValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let qHead = id.x;
  let dim = id.y;
  if (qHead >= params.queryHeadCount || dim >= params.valueSize) {
    return;
  }
  let groupSize = params.queryHeadCount / params.keyValueHeadCount;
  let kvHead = qHead / groupSize;
  var weighted = 0.0;
  let probabilityOffset = qHead * params.keyValueTokenCount;
  for (var keyToken = 0u; keyToken < params.keyValueTokenCount; keyToken = keyToken + 1u) {
    let probability = probabilityValues[probabilityOffset + keyToken];
    let valueIndex = (dim * params.keyValueHeadCount + kvHead) * params.contextLength + keyToken;
    weighted = weighted + probability * valueValues[valueIndex];
  }
  let outputIndex = qHead * params.valueSize + dim;
  outputValues[outputIndex] = weighted;
}
`;

export const Q4_K_MATMUL_WGSL = `
struct Params {
  inputSize: u32,
  rowCount: u32,
  columnCount: u32,
  blockCount: u32,
  rowByteLength: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> weightWords: array<u32>;
@group(0) @binding(1) var<storage, read> inputScales: array<f32>;
@group(0) @binding(2) var<storage, read> inputQs: array<i32>;
@group(0) @binding(3) var<storage, read> inputBsums: array<i32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> outputValues: array<f32>;

fn byteAt(index: u32) -> u32 {
  let word = weightWords[index / 4u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 255u;
}

fn f16At(index: u32) -> f32 {
  let bits = byteAt(index) | (byteAt(index + 1u) << 8u);
  let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
  let exponent = (bits >> 10u) & 31u;
  let fraction = bits & 1023u;
  if (exponent == 0u) {
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 31u) {
    return sign * 65504.0;
  }
  return sign * exp2(f32(exponent) - 15.0) * (1.0 + f32(fraction) / 1024.0);
}

fn q4Scale(blockBase: u32, index: u32) -> u32 {
  let qBase = blockBase + 4u;
  if (index < 4u) {
    return byteAt(qBase + index) & 63u;
  }
  return (byteAt(qBase + index + 4u) & 15u) | ((byteAt(qBase + index - 4u) >> 6u) << 4u);
}

fn q4Min(blockBase: u32, index: u32) -> u32 {
  let qBase = blockBase + 4u;
  if (index < 4u) {
    return byteAt(qBase + index + 4u) & 63u;
  }
  return (byteAt(qBase + index + 4u) >> 4u) | ((byteAt(qBase + index) >> 6u) << 4u);
}

fn q4Value(blockBase: u32, element: u32) -> i32 {
  let group64 = element / 64u;
  let within = element - group64 * 64u;
  let packed = byteAt(blockBase + 16u + group64 * 32u + (within & 31u));
  if (within < 32u) {
    return i32(packed & 15u);
  }
  return i32(packed >> 4u);
}

@compute @workgroup_size(8, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let column = id.y;
  if (row >= params.rowCount || column >= params.columnCount) {
    return;
  }
  var sum = 0.0;
  for (var block = 0u; block < params.blockCount; block = block + 1u) {
    let blockBase = row * params.rowByteLength + block * 144u;
    let inputBase = column * params.inputSize + block * 256u;
    let scaleBase = (column * params.blockCount + block) * 16u;
    var sumi = 0i;
    for (var group = 0u; group < 16u; group = group + 1u) {
      sumi = sumi + inputBsums[scaleBase + group] * i32(q4Min(blockBase, group / 2u));
    }

    var dot = 0i;
    for (var group32 = 0u; group32 < 8u; group32 = group32 + 1u) {
      let scale = i32(q4Scale(blockBase, group32));
      for (var index = 0u; index < 32u; index = index + 1u) {
        let element = group32 * 32u + index;
        dot = dot + scale * q4Value(blockBase, element) * inputQs[inputBase + element];
      }
    }

    let inputScale = inputScales[column * params.blockCount + block];
    let d = f16At(blockBase) * inputScale;
    let dmin = f16At(blockBase + 2u) * inputScale;
    sum = sum + d * f32(dot) - dmin * f32(sumi);
  }
  outputValues[column * params.rowCount + row] = sum;
}
`;

export const Q5_K_MATMUL_WGSL = `
struct Params {
  inputSize: u32,
  rowCount: u32,
  columnCount: u32,
  blockCount: u32,
  rowByteLength: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> weightWords: array<u32>;
@group(0) @binding(1) var<storage, read> inputScales: array<f32>;
@group(0) @binding(2) var<storage, read> inputQs: array<i32>;
@group(0) @binding(3) var<storage, read> inputBsums: array<i32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> outputValues: array<f32>;

fn byteAt(index: u32) -> u32 {
  let word = weightWords[index / 4u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 255u;
}

fn f16At(index: u32) -> f32 {
  let bits = byteAt(index) | (byteAt(index + 1u) << 8u);
  let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
  let exponent = (bits >> 10u) & 31u;
  let fraction = bits & 1023u;
  if (exponent == 0u) {
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 31u) {
    return sign * 65504.0;
  }
  return sign * exp2(f32(exponent) - 15.0) * (1.0 + f32(fraction) / 1024.0);
}

fn kScale(blockBase: u32, index: u32) -> u32 {
  let qBase = blockBase + 4u;
  if (index < 4u) {
    return byteAt(qBase + index) & 63u;
  }
  return (byteAt(qBase + index + 4u) & 15u) | ((byteAt(qBase + index - 4u) >> 6u) << 4u);
}

fn kMin(blockBase: u32, index: u32) -> u32 {
  let qBase = blockBase + 4u;
  if (index < 4u) {
    return byteAt(qBase + index + 4u) & 63u;
  }
  return (byteAt(qBase + index + 4u) >> 4u) | ((byteAt(qBase + index) >> 6u) << 4u);
}

fn q5Value(blockBase: u32, element: u32) -> i32 {
  let group64 = element / 64u;
  let within = element - group64 * 64u;
  let lane = within & 31u;
  let packed = byteAt(blockBase + 48u + group64 * 32u + lane);
  let highMask = 1u << (group64 * 2u + select(0u, 1u, within >= 32u));
  let high = select(0i, 16i, (byteAt(blockBase + 16u + lane) & highMask) != 0u);
  if (within < 32u) {
    return i32(packed & 15u) + high;
  }
  return i32(packed >> 4u) + high;
}

@compute @workgroup_size(8, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let column = id.y;
  if (row >= params.rowCount || column >= params.columnCount) {
    return;
  }
  var sum = 0.0;
  for (var block = 0u; block < params.blockCount; block = block + 1u) {
    let blockBase = row * params.rowByteLength + block * 176u;
    let inputBase = column * params.inputSize + block * 256u;
    let scaleBase = (column * params.blockCount + block) * 16u;
    var sumi = 0i;
    for (var group = 0u; group < 16u; group = group + 1u) {
      sumi = sumi + inputBsums[scaleBase + group] * i32(kMin(blockBase, group / 2u));
    }

    var dot = 0i;
    for (var group32 = 0u; group32 < 8u; group32 = group32 + 1u) {
      let scale = i32(kScale(blockBase, group32));
      for (var index = 0u; index < 32u; index = index + 1u) {
        let element = group32 * 32u + index;
        dot = dot + scale * q5Value(blockBase, element) * inputQs[inputBase + element];
      }
    }

    let inputScale = inputScales[column * params.blockCount + block];
    let d = f16At(blockBase) * inputScale;
    let dmin = f16At(blockBase + 2u) * inputScale;
    sum = sum + d * f32(dot) - dmin * f32(sumi);
  }
  outputValues[column * params.rowCount + row] = sum;
}
`;

export const Q6_K_MATMUL_WGSL = `
struct Params {
  inputSize: u32,
  rowCount: u32,
  columnCount: u32,
  blockCount: u32,
  rowByteLength: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> weightWords: array<u32>;
@group(0) @binding(1) var<storage, read> inputScales: array<f32>;
@group(0) @binding(2) var<storage, read> inputQs: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;

fn byteAt(index: u32) -> u32 {
  let word = weightWords[index / 4u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 255u;
}

fn signedByteAt(index: u32) -> i32 {
  let value = i32(byteAt(index));
  if (value >= 128) {
    return value - 256;
  }
  return value;
}

fn f16At(index: u32) -> f32 {
  let bits = byteAt(index) | (byteAt(index + 1u) << 8u);
  let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
  let exponent = (bits >> 10u) & 31u;
  let fraction = bits & 1023u;
  if (exponent == 0u) {
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 31u) {
    return sign * 65504.0;
  }
  return sign * exp2(f32(exponent) - 15.0) * (1.0 + f32(fraction) / 1024.0);
}

fn q6Value(blockBase: u32, element: u32) -> i32 {
  let group128 = element / 128u;
  let within = element - group128 * 128u;
  let lane = within & 31u;
  let qlBase = blockBase + group128 * 64u;
  let qhByte = byteAt(blockBase + 128u + group128 * 32u + lane);
  if (within < 32u) {
    return i32((byteAt(qlBase + lane) & 15u) | (((qhByte >> 0u) & 3u) << 4u)) - 32;
  }
  if (within < 64u) {
    return i32((byteAt(qlBase + lane + 32u) & 15u) | (((qhByte >> 2u) & 3u) << 4u)) - 32;
  }
  if (within < 96u) {
    return i32((byteAt(qlBase + lane) >> 4u) | (((qhByte >> 4u) & 3u) << 4u)) - 32;
  }
  return i32((byteAt(qlBase + lane + 32u) >> 4u) | (((qhByte >> 6u) & 3u) << 4u)) - 32;
}

@compute @workgroup_size(8, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let column = id.y;
  if (row >= params.rowCount || column >= params.columnCount) {
    return;
  }
  var sum = 0.0;
  for (var block = 0u; block < params.blockCount; block = block + 1u) {
    let blockBase = row * params.rowByteLength + block * 210u;
    let inputBase = column * params.inputSize + block * 256u;
    var dot = 0i;
    for (var group = 0u; group < 16u; group = group + 1u) {
      let scale = signedByteAt(blockBase + 192u + group);
      for (var index = 0u; index < 16u; index = index + 1u) {
        let element = group * 16u + index;
        dot = dot + scale * q6Value(blockBase, element) * inputQs[inputBase + element];
      }
    }
    let inputScale = inputScales[column * params.blockCount + block];
    sum = sum + f16At(blockBase + 208u) * inputScale * f32(dot);
  }
  outputValues[column * params.rowCount + row] = sum;
}
`;
