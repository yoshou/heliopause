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

export const F32_GATHER_ROWS_SCALE_WGSL = `
struct Params {
  rowSize: u32,
  tokenCount: u32,
  scale: f32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> rows: array<f32>;
@group(0) @binding(1) var<storage, read> tokenIds: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x;
  let token = id.y;
  if (column >= params.rowSize || token >= params.tokenCount) {
    return;
  }
  let row = tokenIds[token];
  outputValues[token * params.rowSize + column] = rows[row * params.rowSize + column] * params.scale;
}
`;

export const Q8_0_GATHER_ROWS_SCALE_WGSL = `
struct Params {
  rowSize: u32,
  tokenCount: u32,
  blockCount: u32,
  rowByteLength: u32,
  scale: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> weightWords: array<u32>;
@group(0) @binding(1) var<storage, read> tokenIds: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

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

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x;
  let token = id.y;
  if (column >= params.rowSize || token >= params.tokenCount) {
    return;
  }
  let row = tokenIds[token];
  let block = column / 32u;
  let index = column & 31u;
  let base = row * params.rowByteLength + block * 34u;
  outputValues[token * params.rowSize + column] =
    f32(signedByteAt(base + 2u + index)) * f16At(base) * params.scale;
}
`;

export const Q4_K_GATHER_ROWS_SCALE_WGSL = `
struct Params {
  rowSize: u32,
  tokenCount: u32,
  blockCount: u32,
  rowByteLength: u32,
  scale: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> weightWords: array<u32>;
@group(0) @binding(1) var<storage, read> tokenIds: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

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

fn q4Value(blockBase: u32, element: u32) -> u32 {
  let group64 = element / 64u;
  let within = element - group64 * 64u;
  let packed = byteAt(blockBase + 16u + group64 * 32u + (within & 31u));
  if (within < 32u) {
    return packed & 15u;
  }
  return packed >> 4u;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x;
  let token = id.y;
  if (column >= params.rowSize || token >= params.tokenCount) {
    return;
  }
  let row = tokenIds[token];
  let block = column / 256u;
  let element = column - block * 256u;
  let scaleIndex = element / 32u;
  let blockBase = row * params.rowByteLength + block * 144u;
  let value = f16At(blockBase) * f32(kScale(blockBase, scaleIndex)) * f32(q4Value(blockBase, element)) -
    f16At(blockBase + 2u) * f32(kMin(blockBase, scaleIndex));
  outputValues[token * params.rowSize + column] = value * params.scale;
}
`;

export const Q5_K_GATHER_ROWS_SCALE_WGSL = `
struct Params {
  rowSize: u32,
  tokenCount: u32,
  blockCount: u32,
  rowByteLength: u32,
  scale: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> weightWords: array<u32>;
@group(0) @binding(1) var<storage, read> tokenIds: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

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

fn q5Value(blockBase: u32, element: u32) -> u32 {
  let group64 = element / 64u;
  let within = element - group64 * 64u;
  let lane = within & 31u;
  let packed = byteAt(blockBase + 48u + group64 * 32u + lane);
  let highMask = 1u << (group64 * 2u + select(0u, 1u, within >= 32u));
  let high = select(0u, 16u, (byteAt(blockBase + 16u + lane) & highMask) != 0u);
  if (within < 32u) {
    return (packed & 15u) + high;
  }
  return (packed >> 4u) + high;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x;
  let token = id.y;
  if (column >= params.rowSize || token >= params.tokenCount) {
    return;
  }
  let row = tokenIds[token];
  let block = column / 256u;
  let element = column - block * 256u;
  let scaleIndex = element / 32u;
  let blockBase = row * params.rowByteLength + block * 176u;
  let value = f16At(blockBase) * f32(kScale(blockBase, scaleIndex)) * f32(q5Value(blockBase, element)) -
    f16At(blockBase + 2u) * f32(kMin(blockBase, scaleIndex));
  outputValues[token * params.rowSize + column] = value * params.scale;
}
`;

export const Q6_K_GATHER_ROWS_SCALE_WGSL = `
struct Params {
  rowSize: u32,
  tokenCount: u32,
  blockCount: u32,
  rowByteLength: u32,
  scale: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> weightWords: array<u32>;
@group(0) @binding(1) var<storage, read> tokenIds: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

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
  let within128 = element - group128 * 128u;
  let lane = within128 & 31u;
  let qlBase = blockBase + group128 * 64u;
  let qhByte = byteAt(blockBase + 128u + group128 * 32u + lane);
  if (within128 < 32u) {
    return i32((byteAt(qlBase + lane) & 15u) | (((qhByte >> 0u) & 3u) << 4u)) - 32;
  }
  if (within128 < 64u) {
    return i32((byteAt(qlBase + lane + 32u) & 15u) | (((qhByte >> 2u) & 3u) << 4u)) - 32;
  }
  if (within128 < 96u) {
    return i32((byteAt(qlBase + lane) >> 4u) | (((qhByte >> 4u) & 3u) << 4u)) - 32;
  }
  return i32((byteAt(qlBase + lane + 32u) >> 4u) | (((qhByte >> 6u) & 3u) << 4u)) - 32;
}

fn q6ScaleIndex(element: u32) -> u32 {
  let group128 = element / 128u;
  let within128 = element - group128 * 128u;
  let pair = (within128 & 31u) / 16u;
  if (within128 < 32u) {
    return group128 * 8u + pair;
  }
  if (within128 < 64u) {
    return group128 * 8u + pair + 2u;
  }
  if (within128 < 96u) {
    return group128 * 8u + pair + 4u;
  }
  return group128 * 8u + pair + 6u;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x;
  let token = id.y;
  if (column >= params.rowSize || token >= params.tokenCount) {
    return;
  }
  let row = tokenIds[token];
  let block = column / 256u;
  let element = column - block * 256u;
  let blockBase = row * params.rowByteLength + block * 210u;
  let value = f16At(blockBase + 208u) *
    f32(signedByteAt(blockBase + 192u + q6ScaleIndex(element))) *
    f32(q6Value(blockBase, element));
  outputValues[token * params.rowSize + column] = value * params.scale;
}
`;

export const PREPARE_PER_LAYER_INPUTS_WGSL = `
struct Params {
  perLayerLength: u32,
  totalPerLayerLength: u32,
  tokenCount: u32,
  blockCount: u32,
  projectionScale: f32,
  epsilon: f32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> tokenRows: array<f32>;
@group(0) @binding(1) var<storage, read> projectedValues: array<f32>;
@group(0) @binding(2) var<storage, read> normWeight: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let token = id.y;
  let layer = id.z;
  if (index >= params.perLayerLength || token >= params.tokenCount || layer >= params.blockCount) {
    return;
  }

  let sliceBase = token * params.totalPerLayerLength + layer * params.perLayerLength;
  var meanSquare = 0.0;
  for (var dim = 0u; dim < params.perLayerLength; dim = dim + 1u) {
    let value = projectedValues[sliceBase + dim] * params.projectionScale;
    meanSquare = meanSquare + value * value;
  }
  let normScale = inverseSqrt(meanSquare / f32(params.perLayerLength) + params.epsilon);
  let projected = projectedValues[sliceBase + index] * params.projectionScale * normScale * normWeight[index];
  let tokenValue = tokenRows[sliceBase + index];
  outputValues[(layer * params.tokenCount + token) * params.perLayerLength + index] =
    (tokenValue + projected) * 0.7071067811865476;
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

var<workgroup> rmsReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  var localSum = 0.0;
  for (var index = lane; index < params.length; index = index + 256u) {
    let value = inputValues[index];
    localSum = localSum + value * value;
  }
  rmsReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      rmsReduceValues[lane] = rmsReduceValues[lane] + rmsReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let meanSquare = rmsReduceValues[0];
  let scale = inverseSqrt(meanSquare / f32(params.length) + params.epsilon);
  for (var index = lane; index < params.length; index = index + 256u) {
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

export const RMS_NORM_RESIDUAL_ADD_WGSL = `
struct Params {
  epsilon: f32,
  length: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<storage, read> residualValues: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;

var<workgroup> rmsResidualReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  var localSum = 0.0;
  for (var index = lane; index < params.length; index = index + 256u) {
    let value = inputValues[index];
    localSum = localSum + value * value;
  }
  rmsResidualReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      rmsResidualReduceValues[lane] = rmsResidualReduceValues[lane] + rmsResidualReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let meanSquare = rmsResidualReduceValues[0];
  let scale = inverseSqrt(meanSquare / f32(params.length) + params.epsilon);
  for (var index = lane; index < params.length; index = index + 256u) {
    outputValues[index] = residualValues[index] + inputValues[index] * scale * weightValues[index];
  }
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

var<workgroup> q80MatmulReduceValues: array<f32, 256>;

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

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  let rowLane = lane & 31u;
  let localRow = lane / 32u;
  let row = workgroupId.x * 8u + localRow;
  let column = workgroupId.y;
  if (column >= params.columnCount) {
    return;
  }
  var localSum = 0.0;
  if (row < params.rowCount) {
    for (var block = 0u; block < params.blockCount; block = block + 1u) {
      let weightBase = row * params.rowByteLength + block * 34u;
      let weightScale = f16At(weightBase);
      let inputScale = inputScales[column * params.blockCount + block];
      let w = signedByteAt(weightBase + 2u + rowLane);
      let q = inputQs[column * params.inputSize + block * 32u + rowLane];
      localSum = localSum + f32(w * q) * weightScale * inputScale;
    }
  }
  q80MatmulReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 16u; stride > 0u; stride = stride / 2u) {
    if (rowLane < stride) {
      q80MatmulReduceValues[lane] = q80MatmulReduceValues[lane] + q80MatmulReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  if (rowLane == 0u && row < params.rowCount) {
    outputValues[column * params.rowCount + row] = q80MatmulReduceValues[lane];
  }
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

export const SIGMOID_MUL_WGSL = `
struct Params {
  length: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> leftValues: array<f32>;
@group(0) @binding(1) var<storage, read> gateValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  let gate = gateValues[index];
  outputValues[index] = leftValues[index] * (1.0 / (1.0 + exp(-gate)));
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

export const RESIDUAL_ADD_SCALE_WGSL = `
struct Params {
  length: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> leftValues: array<f32>;
@group(0) @binding(1) var<storage, read> rightValues: array<f32>;
@group(0) @binding(2) var<storage, read> scaleValue: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  outputValues[index] = (leftValues[index] + rightValues[index]) * scaleValue[0];
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

var<workgroup> f32MatmulReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  let rowLane = lane & 31u;
  let localRow = lane / 32u;
  let row = workgroupId.x * 8u + localRow;
  let column = workgroupId.y;
  if (column >= params.columnCount) {
    return;
  }
  var localSum = 0.0;
  if (row < params.rowCount) {
    for (var index = rowLane; index < params.inputSize; index = index + 32u) {
      localSum = localSum + weightValues[row * params.inputSize + index] * inputValues[column * params.inputSize + index];
    }
  }
  f32MatmulReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 16u; stride > 0u; stride = stride / 2u) {
    if (rowLane < stride) {
      f32MatmulReduceValues[lane] = f32MatmulReduceValues[lane] + f32MatmulReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  if (rowLane == 0u && row < params.rowCount) {
    outputValues[column * params.rowCount + row] = f32MatmulReduceValues[lane];
  }
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

export const TOP1_CHUNK_WGSL = `
struct Params {
  rowCount: u32,
  rowOffset: u32,
  candidateOffset: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

var<workgroup> bestValues: array<f32, 256>;
var<workgroup> bestIds: array<u32, 256>;

fn betterValue(leftValue: f32, leftId: u32, rightValue: f32, rightId: u32) -> bool {
  return leftValue > rightValue || (leftValue == rightValue && leftId < rightId);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let lane = local.x;
  let row = group.x * 256u + lane;
  var value = -3.4028234663852886e38;
  var tokenId = params.rowOffset + row;
  if (row < params.rowCount) {
    value = logits[row];
  } else {
    tokenId = 4294967295u;
  }
  bestValues[lane] = value;
  bestIds[lane] = tokenId;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (lane < stride) {
      let otherLane = lane + stride;
      let otherValue = bestValues[otherLane];
      let otherId = bestIds[otherLane];
      if (betterValue(otherValue, otherId, bestValues[lane], bestIds[lane])) {
        bestValues[lane] = otherValue;
        bestIds[lane] = otherId;
      }
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  if (lane == 0u) {
    let outputBase = params.candidateOffset + group.x * 2u;
    outputValues[outputBase] = f32(bestIds[0]);
    outputValues[outputBase + 1u] = bestValues[0];
  }
}
`;

export const SELECT_TOP1_CANDIDATE_WGSL = `
struct Params {
  candidateCount: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> candidates: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> selectedToken: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
  var bestId = 0u;
  var bestValue = -3.4028234663852886e38;
  for (var index = 0u; index < params.candidateCount; index = index + 1u) {
    let base = index * 2u;
    let value = candidates[base + 1u];
    if (value > bestValue) {
      bestValue = value;
      bestId = u32(candidates[base]);
    }
  }
  selectedToken[0] = bestId;
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

var<workgroup> q8AbsValues: array<f32, 256>;
var<workgroup> q8Values: array<f32, 256>;
var<workgroup> q8Indices: array<u32, 256>;
var<workgroup> q8Quants: array<i32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>,
) {
  let column = workgroupId.x;
  let block = workgroupId.y;
  let lane = localId.x;
  if (column >= params.columnCount || block >= params.blockCount) {
    return;
  }
  let base = column * params.inputSize + block * 256u;
  let value = inputValues[base + lane];
  q8AbsValues[lane] = abs(value);
  q8Values[lane] = value;
  q8Indices[lane] = lane;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      let otherLane = lane + stride;
      let otherAbs = q8AbsValues[otherLane];
      let otherIndex = q8Indices[otherLane];
      if (otherAbs > q8AbsValues[lane] || (otherAbs == q8AbsValues[lane] && otherIndex < q8Indices[lane])) {
        q8AbsValues[lane] = otherAbs;
        q8Values[lane] = q8Values[otherLane];
        q8Indices[lane] = otherIndex;
      }
    }
    workgroupBarrier();
  }

  let amax = q8AbsValues[0];
  let maxValue = q8Values[0];
  let blockIndex = column * params.blockCount + block;
  var inverseScale = 0.0;
  if (amax != 0.0) {
    inverseScale = -127.0 / maxValue;
  }
  if (lane == 0u) {
    var scale = 0.0;
    if (amax != 0.0) {
      scale = 1.0 / inverseScale;
    }
    outputScales[blockIndex] = scale;
  }
  var q = 0i;
  if (amax != 0.0) {
    q = min(127i, i32(round(inverseScale * value)));
  }
  q8Quants[lane] = q;
  outputQs[base + lane] = q;
  workgroupBarrier();

  if (lane < 16u) {
    var sum = 0i;
    for (var index = 0u; index < 16u; index = index + 1u) {
      sum = sum + q8Quants[lane * 16u + index];
    }
    outputBsums[blockIndex * 16u + lane] = sum;
  }
}
`;

export const RMS_NORM_Q8_K_QUANTIZE_WGSL = `
struct Params {
  epsilon: f32,
  length: u32,
  blockCount: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputScales: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputQs: array<i32>;
@group(0) @binding(4) var<storage, read_write> outputBsums: array<i32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> rmsQ8ReduceValues: array<f32, 256>;
var<workgroup> rmsQ8AbsValues: array<f32, 256>;
var<workgroup> rmsQ8Values: array<f32, 256>;
var<workgroup> rmsQ8Indices: array<u32, 256>;
var<workgroup> rmsQ8Quants: array<i32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  var localSum = 0.0;
  for (var index = lane; index < params.length; index = index + 256u) {
    let value = inputValues[index];
    localSum = localSum + value * value;
  }
  rmsQ8ReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      rmsQ8ReduceValues[lane] = rmsQ8ReduceValues[lane] + rmsQ8ReduceValues[lane + stride];
    }
    workgroupBarrier();
  }

  let meanSquare = rmsQ8ReduceValues[0];
  let rmsScale = inverseSqrt(meanSquare / f32(params.length) + params.epsilon);
  for (var block = 0u; block < params.blockCount; block = block + 1u) {
    let index = block * 256u + lane;
    let value = inputValues[index] * rmsScale * weightValues[index];
    rmsQ8AbsValues[lane] = abs(value);
    rmsQ8Values[lane] = value;
    rmsQ8Indices[lane] = lane;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        let otherLane = lane + stride;
        let otherAbs = rmsQ8AbsValues[otherLane];
        let otherIndex = rmsQ8Indices[otherLane];
        if (otherAbs > rmsQ8AbsValues[lane] || (otherAbs == rmsQ8AbsValues[lane] && otherIndex < rmsQ8Indices[lane])) {
          rmsQ8AbsValues[lane] = otherAbs;
          rmsQ8Values[lane] = rmsQ8Values[otherLane];
          rmsQ8Indices[lane] = otherIndex;
        }
      }
      workgroupBarrier();
    }

    let amax = rmsQ8AbsValues[0];
    let maxValue = rmsQ8Values[0];
    var inverseScale = 0.0;
    if (amax != 0.0) {
      inverseScale = -127.0 / maxValue;
    }
    if (lane == 0u) {
      var scale = 0.0;
      if (amax != 0.0) {
        scale = 1.0 / inverseScale;
      }
      outputScales[block] = scale;
    }
    var q = 0i;
    if (amax != 0.0) {
      q = min(127i, i32(round(inverseScale * value)));
    }
    rmsQ8Quants[lane] = q;
    outputQs[index] = q;
    workgroupBarrier();

    if (lane < 16u) {
      var sum = 0i;
      for (var offset = 0u; offset < 16u; offset = offset + 1u) {
        sum = sum + rmsQ8Quants[lane * 16u + offset];
      }
      outputBsums[block * 16u + lane] = sum;
    }
    workgroupBarrier();
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

export const FULL_ATTENTION_SCORE_WGSL = `
struct Params {
  scale: f32,
  headSize: u32,
  queryHeadCount: u32,
  keyValueHeadCount: u32,
  keyValueTokenCount: u32,
  contextLength: u32,
  tokenPosition: u32,
  keyValueStart: u32,
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
  return queryValues[index];
}

fn attentionScore(qHead: u32, kvHead: u32, keyToken: u32) -> f32 {
  let queryOffset = qHead * params.headSize;
  let keyOffset = (keyToken * params.keyValueHeadCount + kvHead) * params.headSize;
  var dot = 0.0;
  for (var dim = 0u; dim < params.headSize; dim = dim + 1u) {
    dot = dot + queryValue(queryOffset + dim) * keyValues[keyOffset + dim];
  }
  return dot * params.scale;
}

var<workgroup> reduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>,
) {
  let qHead = workgroupId.x;
  let lane = localId.x;
  if (qHead >= params.queryHeadCount) {
    return;
  }
  let groupSize = params.queryHeadCount / params.keyValueHeadCount;
  let kvHead = qHead / groupSize;
  let probabilityOffset = qHead * params.keyValueTokenCount;
  var localMax = -3.4028234663852886e38;
  for (var keyToken = params.keyValueStart + lane; keyToken < params.keyValueTokenCount; keyToken = keyToken + 256u) {
    let score = attentionScore(qHead, kvHead, keyToken);
    probabilityValues[probabilityOffset + keyToken] = score;
    localMax = max(localMax, score);
  }
  reduceValues[lane] = localMax;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      reduceValues[lane] = max(reduceValues[lane], reduceValues[lane + stride]);
    }
    workgroupBarrier();
  }
  let maxScore = reduceValues[0];
  var localSum = 0.0;
  for (var keyToken = params.keyValueStart + lane; keyToken < params.keyValueTokenCount; keyToken = keyToken + 256u) {
    let probability = exp(probabilityValues[probabilityOffset + keyToken] - maxScore);
    probabilityValues[probabilityOffset + keyToken] = probability;
    localSum = localSum + probability;
  }
  reduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      reduceValues[lane] = reduceValues[lane] + reduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let sum = reduceValues[0];
  for (var keyToken = params.keyValueStart + lane; keyToken < params.keyValueTokenCount; keyToken = keyToken + 256u) {
    let index = probabilityOffset + keyToken;
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
  keyValueStart: u32,
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
  for (var keyToken = params.keyValueStart; keyToken < params.keyValueTokenCount; keyToken = keyToken + 1u) {
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

var<workgroup> q4MatmulReduceValues: array<f32, 256>;

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

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  let rowLane = lane & 31u;
  let localRow = lane / 32u;
  let row = workgroupId.x * 8u + localRow;
  let column = workgroupId.y;
  if (column >= params.columnCount) {
    return;
  }
  var localSum = 0.0;
  if (row < params.rowCount) {
    for (var block = 0u; block < params.blockCount; block = block + 1u) {
      let blockBase = row * params.rowByteLength + block * 144u;
      let inputBase = column * params.inputSize + block * 256u;
      let scaleBase = (column * params.blockCount + block) * 16u;
      var dot = 0i;
      for (var element = rowLane; element < 256u; element = element + 32u) {
        let group32 = element / 32u;
        dot = dot + i32(q4Scale(blockBase, group32)) * q4Value(blockBase, element) * inputQs[inputBase + element];
      }
      var sumi = 0i;
      if (rowLane < 16u) {
        sumi = inputBsums[scaleBase + rowLane] * i32(q4Min(blockBase, rowLane / 2u));
      }
      let inputScale = inputScales[column * params.blockCount + block];
      let d = f16At(blockBase) * inputScale;
      let dmin = f16At(blockBase + 2u) * inputScale;
      localSum = localSum + d * f32(dot) - dmin * f32(sumi);
    }
  }
  q4MatmulReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 16u; stride > 0u; stride = stride / 2u) {
    if (rowLane < stride) {
      q4MatmulReduceValues[lane] = q4MatmulReduceValues[lane] + q4MatmulReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  if (rowLane == 0u && row < params.rowCount) {
    outputValues[column * params.rowCount + row] = q4MatmulReduceValues[lane];
  }
}
`;

export const Q4_K_DUAL_MATMUL_WGSL = `
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

@group(0) @binding(0) var<storage, read> leftWeightWords: array<u32>;
@group(0) @binding(1) var<storage, read> rightWeightWords: array<u32>;
@group(0) @binding(2) var<storage, read> inputScales: array<f32>;
@group(0) @binding(3) var<storage, read> inputQs: array<i32>;
@group(0) @binding(4) var<storage, read> inputBsums: array<i32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> leftOutputValues: array<f32>;
@group(0) @binding(7) var<storage, read_write> rightOutputValues: array<f32>;

var<workgroup> q4DualMatmulReduceValues: array<f32, 256>;

fn leftByteAt(index: u32) -> u32 {
  let word = leftWeightWords[index / 4u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 255u;
}

fn rightByteAt(index: u32) -> u32 {
  let word = rightWeightWords[index / 4u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 255u;
}

fn f16FromBytes(low: u32, high: u32) -> f32 {
  let bits = low | (high << 8u);
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

fn leftF16At(index: u32) -> f32 {
  return f16FromBytes(leftByteAt(index), leftByteAt(index + 1u));
}

fn rightF16At(index: u32) -> f32 {
  return f16FromBytes(rightByteAt(index), rightByteAt(index + 1u));
}

fn leftQ4Scale(blockBase: u32, index: u32) -> u32 {
  let qBase = blockBase + 4u;
  if (index < 4u) {
    return leftByteAt(qBase + index) & 63u;
  }
  return (leftByteAt(qBase + index + 4u) & 15u) | ((leftByteAt(qBase + index - 4u) >> 6u) << 4u);
}

fn rightQ4Scale(blockBase: u32, index: u32) -> u32 {
  let qBase = blockBase + 4u;
  if (index < 4u) {
    return rightByteAt(qBase + index) & 63u;
  }
  return (rightByteAt(qBase + index + 4u) & 15u) | ((rightByteAt(qBase + index - 4u) >> 6u) << 4u);
}

fn leftQ4Min(blockBase: u32, index: u32) -> u32 {
  let qBase = blockBase + 4u;
  if (index < 4u) {
    return leftByteAt(qBase + index + 4u) & 63u;
  }
  return (leftByteAt(qBase + index + 4u) >> 4u) | ((leftByteAt(qBase + index) >> 6u) << 4u);
}

fn rightQ4Min(blockBase: u32, index: u32) -> u32 {
  let qBase = blockBase + 4u;
  if (index < 4u) {
    return rightByteAt(qBase + index + 4u) & 63u;
  }
  return (rightByteAt(qBase + index + 4u) >> 4u) | ((rightByteAt(qBase + index) >> 6u) << 4u);
}

fn leftQ4Value(blockBase: u32, element: u32) -> i32 {
  let group64 = element / 64u;
  let within = element - group64 * 64u;
  let packed = leftByteAt(blockBase + 16u + group64 * 32u + (within & 31u));
  if (within < 32u) {
    return i32(packed & 15u);
  }
  return i32(packed >> 4u);
}

fn rightQ4Value(blockBase: u32, element: u32) -> i32 {
  let group64 = element / 64u;
  let within = element - group64 * 64u;
  let packed = rightByteAt(blockBase + 16u + group64 * 32u + (within & 31u));
  if (within < 32u) {
    return i32(packed & 15u);
  }
  return i32(packed >> 4u);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  let rowLane = lane & 31u;
  let localRow = lane / 32u;
  let row = workgroupId.x * 8u + localRow;
  let column = workgroupId.y;
  let side = workgroupId.z;
  if (column >= params.columnCount || side >= 2u) {
    return;
  }
  var localSum = 0.0;
  if (row < params.rowCount && side == 0u) {
    for (var block = 0u; block < params.blockCount; block = block + 1u) {
      let blockBase = row * params.rowByteLength + block * 144u;
      let inputBase = column * params.inputSize + block * 256u;
      let scaleBase = (column * params.blockCount + block) * 16u;
      var dot = 0i;
      for (var element = rowLane; element < 256u; element = element + 32u) {
        let group32 = element / 32u;
        dot = dot + i32(leftQ4Scale(blockBase, group32)) * leftQ4Value(blockBase, element) * inputQs[inputBase + element];
      }
      var sumi = 0i;
      if (rowLane < 16u) {
        sumi = inputBsums[scaleBase + rowLane] * i32(leftQ4Min(blockBase, rowLane / 2u));
      }
      let inputScale = inputScales[column * params.blockCount + block];
      let d = leftF16At(blockBase) * inputScale;
      let dmin = leftF16At(blockBase + 2u) * inputScale;
      localSum = localSum + d * f32(dot) - dmin * f32(sumi);
    }
  }
  if (row < params.rowCount && side == 1u) {
    for (var block = 0u; block < params.blockCount; block = block + 1u) {
      let blockBase = row * params.rowByteLength + block * 144u;
      let inputBase = column * params.inputSize + block * 256u;
      let scaleBase = (column * params.blockCount + block) * 16u;
      var dot = 0i;
      for (var element = rowLane; element < 256u; element = element + 32u) {
        let group32 = element / 32u;
        dot = dot + i32(rightQ4Scale(blockBase, group32)) * rightQ4Value(blockBase, element) * inputQs[inputBase + element];
      }
      var sumi = 0i;
      if (rowLane < 16u) {
        sumi = inputBsums[scaleBase + rowLane] * i32(rightQ4Min(blockBase, rowLane / 2u));
      }
      let inputScale = inputScales[column * params.blockCount + block];
      let d = rightF16At(blockBase) * inputScale;
      let dmin = rightF16At(blockBase + 2u) * inputScale;
      localSum = localSum + d * f32(dot) - dmin * f32(sumi);
    }
  }
  q4DualMatmulReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 16u; stride > 0u; stride = stride / 2u) {
    if (rowLane < stride) {
      q4DualMatmulReduceValues[lane] = q4DualMatmulReduceValues[lane] + q4DualMatmulReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  if (rowLane == 0u && row < params.rowCount) {
    if (side == 0u) {
      leftOutputValues[column * params.rowCount + row] = q4DualMatmulReduceValues[lane];
    } else {
      rightOutputValues[column * params.rowCount + row] = q4DualMatmulReduceValues[lane];
    }
  }
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

var<workgroup> q5MatmulReduceValues: array<f32, 256>;

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

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  let rowLane = lane & 31u;
  let localRow = lane / 32u;
  let row = workgroupId.x * 8u + localRow;
  let column = workgroupId.y;
  if (column >= params.columnCount) {
    return;
  }
  var localSum = 0.0;
  if (row < params.rowCount) {
    for (var block = 0u; block < params.blockCount; block = block + 1u) {
      let blockBase = row * params.rowByteLength + block * 176u;
      let inputBase = column * params.inputSize + block * 256u;
      let scaleBase = (column * params.blockCount + block) * 16u;
      var dot = 0i;
      for (var element = rowLane; element < 256u; element = element + 32u) {
        let group32 = element / 32u;
        dot = dot + i32(kScale(blockBase, group32)) * q5Value(blockBase, element) * inputQs[inputBase + element];
      }
      var sumi = 0i;
      if (rowLane < 16u) {
        sumi = inputBsums[scaleBase + rowLane] * i32(kMin(blockBase, rowLane / 2u));
      }
      let inputScale = inputScales[column * params.blockCount + block];
      let d = f16At(blockBase) * inputScale;
      let dmin = f16At(blockBase + 2u) * inputScale;
      localSum = localSum + d * f32(dot) - dmin * f32(sumi);
    }
  }
  q5MatmulReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 16u; stride > 0u; stride = stride / 2u) {
    if (rowLane < stride) {
      q5MatmulReduceValues[lane] = q5MatmulReduceValues[lane] + q5MatmulReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  if (rowLane == 0u && row < params.rowCount) {
    outputValues[column * params.rowCount + row] = q5MatmulReduceValues[lane];
  }
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

var<workgroup> q6MatmulReduceValues: array<f32, 256>;

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

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  let rowLane = lane & 31u;
  let localRow = lane / 32u;
  let row = workgroupId.x * 8u + localRow;
  let column = workgroupId.y;
  if (column >= params.columnCount) {
    return;
  }
  var localSum = 0.0;
  if (row < params.rowCount) {
    for (var block = 0u; block < params.blockCount; block = block + 1u) {
      let blockBase = row * params.rowByteLength + block * 210u;
      let inputBase = column * params.inputSize + block * 256u;
      var dot = 0i;
      for (var element = rowLane; element < 256u; element = element + 32u) {
        let group = element / 16u;
        let scale = signedByteAt(blockBase + 192u + group);
        dot = dot + scale * q6Value(blockBase, element) * inputQs[inputBase + element];
      }
      let inputScale = inputScales[column * params.blockCount + block];
      localSum = localSum + f16At(blockBase + 208u) * inputScale * f32(dot);
    }
  }
  q6MatmulReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 16u; stride > 0u; stride = stride / 2u) {
    if (rowLane < stride) {
      q6MatmulReduceValues[lane] = q6MatmulReduceValues[lane] + q6MatmulReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  if (rowLane == 0u && row < params.rowCount) {
    outputValues[column * params.rowCount + row] = q6MatmulReduceValues[lane];
  }
}
`;
