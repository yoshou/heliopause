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

export const VISION_PREPROCESS_RGBA_WGSL = `
struct Params {
  sourceWidth: u32,
  sourceHeight: u32,
  targetWidth: u32,
  targetHeight: u32,
  mean0: f32,
  mean1: f32,
  mean2: f32,
  _pad0: f32,
  std0: f32,
  std1: f32,
  std2: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<storage, read> rgbaValues: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

fn byteAt(index: u32) -> u32 {
  return rgbaValues[index];
}

fn lerp(left: f32, right: f32, amount: f32) -> f32 {
  return left + (right - left) * amount;
}

fn mean(channel: u32) -> f32 {
  if (channel == 0u) {
    return params.mean0;
  }
  if (channel == 1u) {
    return params.mean1;
  }
  return params.mean2;
}

fn stdValue(channel: u32) -> f32 {
  if (channel == 0u) {
    return params.std0;
  }
  if (channel == 1u) {
    return params.std1;
  }
  return params.std2;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let length = params.targetWidth * params.targetHeight * 3u;
  if (index >= length) {
    return;
  }

  let channel = index % 3u;
  let pixel = index / 3u;
  let x = pixel % params.targetWidth;
  let y = pixel / params.targetWidth;
  let xRatio = select(0.0, f32(params.sourceWidth - 1u) / f32(params.targetWidth - 1u), params.targetWidth > 1u);
  let yRatio = select(0.0, f32(params.sourceHeight - 1u) / f32(params.targetHeight - 1u), params.targetHeight > 1u);
  let px = f32(x) * xRatio;
  let py = f32(y) * yRatio;
  let x0 = min(u32(px), params.sourceWidth - 1u);
  let y0 = min(u32(py), params.sourceHeight - 1u);
  let x1 = min(x0 + 1u, params.sourceWidth - 1u);
  let y1 = min(y0 + 1u, params.sourceHeight - 1u);
  let xf = px - f32(x0);
  let yf = py - f32(y0);
  let top = lerp(
    f32(byteAt((y0 * params.sourceWidth + x0) * 4u + channel)),
    f32(byteAt((y0 * params.sourceWidth + x1) * 4u + channel)),
    xf,
  );
  let bottom = lerp(
    f32(byteAt((y1 * params.sourceWidth + x0) * 4u + channel)),
    f32(byteAt((y1 * params.sourceWidth + x1) * 4u + channel)),
    xf,
  );
  let resized = floor(lerp(top, bottom, yf));
  outputValues[index] = ((resized / 255.0) - mean(channel)) / stdValue(channel);
}
`;

export const AUDIO_WINDOW_FRAMES_WGSL = `
struct Params {
  sampleCount: u32,
  frameCount: u32,
  frameLength: u32,
  hopLength: u32,
  fftLength: u32,
  padLeft: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> pcmValues: array<f32>;
@group(0) @binding(1) var<storage, read> windowValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> realValues: array<f32>;
@group(0) @binding(4) var<storage, read_write> imagValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let length = params.frameCount * params.fftLength;
  if (index >= length) {
    return;
  }
  let frame = index / params.fftLength;
  let bin = index % params.fftLength;
  let paddedIndex = i32(frame * params.hopLength + bin) - i32(params.padLeft);
  var sample = 0.0;
  if (paddedIndex >= 0 && u32(paddedIndex) < params.sampleCount) {
    sample = pcmValues[u32(paddedIndex)];
  }
  realValues[index] = sample * windowValues[bin];
  imagValues[index] = 0.0;
}
`;

export const AUDIO_FFT_BIT_REVERSE_WGSL = `
struct Params {
  frameCount: u32,
  fftLength: u32,
  log2Length: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> inputReal: array<f32>;
@group(0) @binding(1) var<storage, read> inputImag: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputReal: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputImag: array<f32>;

fn reverseBits(value: u32, width: u32) -> u32 {
  var input = value;
  var output = 0u;
  for (var bit = 0u; bit < width; bit = bit + 1u) {
    output = (output << 1u) | (input & 1u);
    input = input >> 1u;
  }
  return output;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let length = params.frameCount * params.fftLength;
  if (index >= length) {
    return;
  }
  let frame = index / params.fftLength;
  let bin = index % params.fftLength;
  let reversed = reverseBits(bin, params.log2Length);
  let outputIndex = frame * params.fftLength + reversed;
  outputReal[outputIndex] = inputReal[index];
  outputImag[outputIndex] = inputImag[index];
}
`;

export const AUDIO_FFT_STAGE_WGSL = `
struct Params {
  frameCount: u32,
  fftLength: u32,
  stageSize: u32,
  halfSize: u32,
};

@group(0) @binding(0) var<storage, read> inputReal: array<f32>;
@group(0) @binding(1) var<storage, read> inputImag: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputReal: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputImag: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let pairIndex = id.x;
  let pairsPerFrame = params.fftLength / 2u;
  let length = params.frameCount * pairsPerFrame;
  if (pairIndex >= length) {
    return;
  }
  let frame = pairIndex / pairsPerFrame;
  let pair = pairIndex % pairsPerFrame;
  let group = pair / params.halfSize;
  let k = pair % params.halfSize;
  let even = frame * params.fftLength + group * params.stageSize + k;
  let odd = even + params.halfSize;
  let angle = -6.283185307179586 * f32(k) / f32(params.stageSize);
  let wr = cos(angle);
  let wi = sin(angle);
  let oddReal = inputReal[odd] * wr - inputImag[odd] * wi;
  let oddImag = inputReal[odd] * wi + inputImag[odd] * wr;
  outputReal[even] = inputReal[even] + oddReal;
  outputImag[even] = inputImag[even] + oddImag;
  outputReal[odd] = inputReal[even] - oddReal;
  outputImag[odd] = inputImag[even] - oddImag;
}
`;

export const AUDIO_LOG_MEL_WGSL = `
struct Params {
  frameCount: u32,
  fftLength: u32,
  featureSize: u32,
  binCount: u32,
  melFloor: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<storage, read> realValues: array<f32>;
@group(0) @binding(1) var<storage, read> imagValues: array<f32>;
@group(0) @binding(2) var<storage, read> filterValues: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(5) var<storage, read_write> maskValues: array<u32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let mel = id.x;
  let frame = id.y;
  if (mel >= params.featureSize || frame >= params.frameCount) {
    return;
  }
  var energy = 0.0;
  let frameOffset = frame * params.fftLength;
  let filterOffset = mel * params.binCount;
  for (var bin = 0u; bin < params.binCount; bin = bin + 1u) {
    let real = realValues[frameOffset + bin];
    let imag = imagValues[frameOffset + bin];
    let magnitude = sqrt(real * real + imag * imag);
    energy = energy + magnitude * filterValues[filterOffset + bin];
  }
  outputValues[frame * params.featureSize + mel] = log(max(energy, params.melFloor));
  if (mel == 0u) {
    maskValues[frame] = 1u;
  }
}
`;

export const TOKEN_WRITE_WGSL = `
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
  outputValues[params.rowIndex * params.rowSize + index] = inputValues[index];
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

export const RMS_NORM_RESIDUAL_ADD_SCALE_WGSL = `
struct Params {
  epsilon: f32,
  length: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<storage, read> residualValues: array<f32>;
@group(0) @binding(3) var<storage, read> scaleValue: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> outputValues: array<f32>;

var<workgroup> rmsResidualScaleReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  var localSum = 0.0;
  for (var index = lane; index < params.length; index = index + 256u) {
    let value = inputValues[index];
    localSum = localSum + value * value;
  }
  rmsResidualScaleReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      rmsResidualScaleReduceValues[lane] = rmsResidualScaleReduceValues[lane] + rmsResidualScaleReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let meanSquare = rmsResidualScaleReduceValues[0];
  let normScale = inverseSqrt(meanSquare / f32(params.length) + params.epsilon);
  let outputScale = scaleValue[0];
  for (var index = lane; index < params.length; index = index + 256u) {
    outputValues[index] = (residualValues[index] + inputValues[index] * normScale * weightValues[index]) * outputScale;
  }
}
`;

export const BATCHED_RMS_NORM_RESIDUAL_ADD_WGSL = `
struct Params {
  epsilon: f32,
  length: u32,
  tokenCount: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<storage, read> residualValues: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;

var<workgroup> batchedRmsResidualReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let token = workgroupId.y;
  let lane = localId.x;
  if (token >= params.tokenCount) {
    return;
  }
  let base = token * params.length;
  var localSum = 0.0;
  for (var index = lane; index < params.length; index = index + 256u) {
    let value = inputValues[base + index];
    localSum = localSum + value * value;
  }
  batchedRmsResidualReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      batchedRmsResidualReduceValues[lane] = batchedRmsResidualReduceValues[lane] + batchedRmsResidualReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let scale = inverseSqrt(batchedRmsResidualReduceValues[0] / f32(params.length) + params.epsilon);
  for (var index = lane; index < params.length; index = index + 256u) {
    outputValues[base + index] = residualValues[base + index] + inputValues[base + index] * scale * weightValues[index];
  }
}
`;

export const BATCHED_RMS_NORM_RESIDUAL_ADD_SCALE_WGSL = `
struct Params {
  epsilon: f32,
  length: u32,
  tokenCount: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<storage, read> residualValues: array<f32>;
@group(0) @binding(3) var<storage, read> scaleValue: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> outputValues: array<f32>;

var<workgroup> batchedRmsResidualScaleReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let token = workgroupId.y;
  let lane = localId.x;
  if (token >= params.tokenCount) {
    return;
  }
  let base = token * params.length;
  var localSum = 0.0;
  for (var index = lane; index < params.length; index = index + 256u) {
    let value = inputValues[base + index];
    localSum = localSum + value * value;
  }
  batchedRmsResidualScaleReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      batchedRmsResidualScaleReduceValues[lane] = batchedRmsResidualScaleReduceValues[lane] + batchedRmsResidualScaleReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let normScale = inverseSqrt(batchedRmsResidualScaleReduceValues[0] / f32(params.length) + params.epsilon);
  let outputScale = scaleValue[0];
  for (var index = lane; index < params.length; index = index + 256u) {
    outputValues[base + index] = (residualValues[base + index] + inputValues[base + index] * normScale * weightValues[index]) * outputScale;
  }
}
`;

export const HEAD_RMS_NORM_WGSL = `
struct Params {
  epsilon: f32,
  headCount: u32,
  headSize: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> normWeights: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

var<workgroup> headRmsReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let head = workgroupId.x;
  let lane = localId.x;
  if (head >= params.headCount) {
    return;
  }
  let base = head * params.headSize;
  var meanSquare = 0.0;
  for (var dim = lane; dim < params.headSize; dim = dim + 256u) {
    let value = inputValues[base + dim];
    meanSquare = meanSquare + value * value;
  }
  headRmsReduceValues[lane] = meanSquare;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      headRmsReduceValues[lane] = headRmsReduceValues[lane] + headRmsReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let scale = inverseSqrt(headRmsReduceValues[0] / f32(params.headSize) + params.epsilon);
  for (var dim = lane; dim < params.headSize; dim = dim + 256u) {
    outputValues[base + dim] = inputValues[base + dim] * scale * normWeights[dim];
  }
}
`;

export const HEAD_RMS_NORM_NO_WEIGHT_WGSL = `
struct Params {
  epsilon: f32,
  headCount: u32,
  headSize: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

var<workgroup> headRmsNoWeightReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let head = workgroupId.x;
  let lane = localId.x;
  if (head >= params.headCount) {
    return;
  }
  let base = head * params.headSize;
  var meanSquare = 0.0;
  for (var dim = lane; dim < params.headSize; dim = dim + 256u) {
    let value = inputValues[base + dim];
    meanSquare = meanSquare + value * value;
  }
  headRmsNoWeightReduceValues[lane] = meanSquare;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      headRmsNoWeightReduceValues[lane] = headRmsNoWeightReduceValues[lane] + headRmsNoWeightReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let scale = inverseSqrt(headRmsNoWeightReduceValues[0] / f32(params.headSize) + params.epsilon);
  for (var dim = lane; dim < params.headSize; dim = dim + 256u) {
    outputValues[base + dim] = inputValues[base + dim] * scale;
  }
}
`;

export const ROPE_WGSL = `
struct Params {
  freqBase: f32,
  position: f32,
  hasFreqFactors: u32,
  headCount: u32,
  headSize: u32,
  ropeDims: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> freqFactors: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

fn ropeFactor(index: u32) -> f32 {
  if (params.hasFreqFactors == 0u) {
    return 1.0;
  }
  return freqFactors[index];
}

fn ropeTheta(pairIndex: u32) -> f32 {
  let thetaScale = pow(params.freqBase, -2.0 / f32(params.ropeDims));
  var theta = params.position;
  for (var index = 0u; index < pairIndex; index = index + 1u) {
    theta = theta * thetaScale;
  }
  return theta;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let head = workgroupId.x;
  let lane = localId.x;
  let base = head * params.headSize;
  if (head >= params.headCount) {
    return;
  }
  for (var dim = lane + params.ropeDims; dim < params.headSize; dim = dim + 256u) {
    outputValues[base + dim] = inputValues[base + dim];
  }
  let ropePairCount = params.ropeDims / 2u;
  for (var ic = lane; ic < ropePairCount; ic = ic + 256u) {
    let x0 = inputValues[base + ic];
    let x1 = inputValues[base + ropePairCount + ic];
    let theta = ropeTheta(ic);
    let thetaWithFactor = theta / ropeFactor(ic);
    let cosTheta = cos(thetaWithFactor);
    let sinTheta = sin(thetaWithFactor);
    outputValues[base + ic] = x0 * cosTheta - x1 * sinTheta;
    outputValues[base + ropePairCount + ic] = x0 * sinTheta + x1 * cosTheta;
  }
}
`;

export const KEY_CACHE_ROPE_WGSL = `
struct Params {
  freqBase: f32,
  position: f32,
  hasFreqFactors: u32,
  headCount: u32,
  headSize: u32,
  ropeDims: u32,
  tokenPosition: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> freqFactors: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> keyCache: array<f32>;

fn ropeFactor(index: u32) -> f32 {
  if (params.hasFreqFactors == 0u) {
    return 1.0;
  }
  return freqFactors[index];
}

fn ropeTheta(pairIndex: u32) -> f32 {
  let thetaScale = pow(params.freqBase, -2.0 / f32(params.ropeDims));
  var theta = params.position;
  for (var index = 0u; index < pairIndex; index = index + 1u) {
    theta = theta * thetaScale;
  }
  return theta;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let head = workgroupId.x;
  let lane = localId.x;
  let base = head * params.headSize;
  let keyBase = (params.tokenPosition * params.headCount + head) * params.headSize;
  if (head >= params.headCount) {
    return;
  }
  for (var dim = lane + params.ropeDims; dim < params.headSize; dim = dim + 256u) {
    keyCache[keyBase + dim] = inputValues[base + dim];
  }
  let ropePairCount = params.ropeDims / 2u;
  for (var ic = lane; ic < ropePairCount; ic = ic + 256u) {
    let x0 = inputValues[base + ic];
    let x1 = inputValues[base + ropePairCount + ic];
    let theta = ropeTheta(ic);
    let thetaWithFactor = theta / ropeFactor(ic);
    let cosTheta = cos(thetaWithFactor);
    let sinTheta = sin(thetaWithFactor);
    keyCache[keyBase + ic] = x0 * cosTheta - x1 * sinTheta;
    keyCache[keyBase + ropePairCount + ic] = x0 * sinTheta + x1 * cosTheta;
  }
}
`;

export const VALUE_CACHE_WRITE_WGSL = `
struct Params {
  headCount: u32,
  valueSize: u32,
  tokenPosition: u32,
  contextLength: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> valueCache: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let head = workgroupId.x;
  let lane = localId.x;
  let base = head * params.valueSize;
  if (head >= params.headCount) {
    return;
  }
  for (var dim = lane; dim < params.valueSize; dim = dim + 256u) {
    valueCache[(dim * params.headCount + head) * params.contextLength + params.tokenPosition] = inputValues[base + dim];
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

fn ropeTheta(pairIndex: u32) -> f32 {
  let thetaScale = pow(params.freqBase, -2.0 / f32(params.ropeDims));
  var theta = params.position;
  for (var index = 0u; index < pairIndex; index = index + 1u) {
    theta = theta * thetaScale;
  }
  return theta;
}

var<workgroup> queryReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let head = workgroupId.x;
  let lane = localId.x;
  let qBase = head * params.headSize;
  let outBase = head * params.headSize;
  var meanSquare = 0.0;
  for (var dim = lane; dim < params.headSize; dim = dim + 256u) {
    let value = qValues[qBase + dim];
    meanSquare = meanSquare + value * value;
  }
  queryReduceValues[lane] = meanSquare;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      queryReduceValues[lane] = queryReduceValues[lane] + queryReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let scale = inverseSqrt(queryReduceValues[0] / f32(params.headSize) + params.epsilon);
  for (var dim = lane; dim < params.headSize; dim = dim + 256u) {
    queryValues[outBase + dim] = normed(head, dim, scale);
  }
  let ropePairCount = params.ropeDims / 2u;
  for (var ic = lane; ic < ropePairCount; ic = ic + 256u) {
    let x0 = normed(head, ic, scale);
    let x1 = normed(head, ropePairCount + ic, scale);
    let theta = ropeTheta(ic);
    let thetaWithFactor = theta / ropeFactor(ic);
    let cosTheta = cos(thetaWithFactor);
    let sinTheta = sin(thetaWithFactor);
    queryValues[outBase + ic] = x0 * cosTheta - x1 * sinTheta;
    queryValues[outBase + ropePairCount + ic] = x0 * sinTheta + x1 * cosTheta;
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

fn ropeTheta(pairIndex: u32) -> f32 {
  let thetaScale = pow(params.freqBase, -2.0 / f32(params.ropeDims));
  var theta = params.position;
  for (var index = 0u; index < pairIndex; index = index + 1u) {
    theta = theta * thetaScale;
  }
  return theta;
}

var<workgroup> kvReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let head = workgroupId.x;
  let lane = localId.x;
  let base = head * params.headSize;
  var meanSquare = 0.0;
  for (var dim = lane; dim < params.headSize; dim = dim + 256u) {
    let value = kProjectionValues[base + dim];
    meanSquare = meanSquare + value * value;
  }
  kvReduceValues[lane] = meanSquare;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      kvReduceValues[lane] = kvReduceValues[lane] + kvReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let scale = inverseSqrt(kvReduceValues[0] / f32(params.headSize) + params.epsilon);
  let keyBase = (params.tokenPosition * params.headCount + head) * params.headSize;
  for (var dim = lane; dim < params.headSize; dim = dim + 256u) {
    keyCache[keyBase + dim] = normed(head, dim, scale);
  }
  var valueMeanSquare = 0.0;
  let valueBase = head * params.valueSize;
  for (var dim = lane; dim < params.valueSize; dim = dim + 256u) {
    let value = vProjectionValues[valueBase + dim];
    valueMeanSquare = valueMeanSquare + value * value;
  }
  kvReduceValues[lane] = valueMeanSquare;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      kvReduceValues[lane] = kvReduceValues[lane] + kvReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let valueScale = inverseSqrt(kvReduceValues[0] / f32(params.valueSize) + params.epsilon);
  for (var dim = lane; dim < params.valueSize; dim = dim + 256u) {
    valueCache[(dim * params.headCount + head) * params.contextLength + params.tokenPosition] =
      vProjectionValues[valueBase + dim] * valueScale;
  }
  let ropePairCount = params.ropeDims / 2u;
  for (var ic = lane; ic < ropePairCount; ic = ic + 256u) {
    let x0 = normed(head, ic, scale);
    let x1 = normed(head, ropePairCount + ic, scale);
    let theta = ropeTheta(ic);
    let thetaWithFactor = theta / ropeFactor(ic);
    let cosTheta = cos(thetaWithFactor);
    let sinTheta = sin(thetaWithFactor);
    keyCache[keyBase + ic] = x0 * cosTheta - x1 * sinTheta;
    keyCache[keyBase + ropePairCount + ic] = x0 * sinTheta + x1 * cosTheta;
  }
}
`;

export const BATCHED_FULL_QUERY_NORM_ROPE_WGSL = `
struct Params {
  epsilon: f32,
  freqBase: f32,
  hasFreqFactors: u32,
  headCount: u32,
  headSize: u32,
  ropeDims: u32,
  tokenCount: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> qValues: array<f32>;
@group(0) @binding(1) var<storage, read> normWeights: array<f32>;
@group(0) @binding(2) var<storage, read> freqFactors: array<f32>;
@group(0) @binding(3) var<storage, read> positions: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> queryValues: array<f32>;

fn batchedQueryInput(token: u32, head: u32, dim: u32) -> f32 {
  return qValues[token * params.headCount * params.headSize + head * params.headSize + dim];
}

fn batchedQueryNormed(token: u32, head: u32, dim: u32, scale: f32) -> f32 {
  return batchedQueryInput(token, head, dim) * scale * normWeights[dim];
}

fn batchedQueryRopeFactor(index: u32) -> f32 {
  if (params.hasFreqFactors == 0u) {
    return 1.0;
  }
  return freqFactors[index];
}

fn batchedQueryRopeTheta(position: u32, pairIndex: u32) -> f32 {
  let thetaScale = pow(params.freqBase, -2.0 / f32(params.ropeDims));
  var theta = f32(position);
  for (var index = 0u; index < pairIndex; index = index + 1u) {
    theta = theta * thetaScale;
  }
  return theta;
}

var<workgroup> batchedQueryReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let head = workgroupId.x;
  let token = workgroupId.y;
  let lane = localId.x;
  if (head >= params.headCount || token >= params.tokenCount) {
    return;
  }
  var meanSquare = 0.0;
  for (var dim = lane; dim < params.headSize; dim = dim + 256u) {
    let value = batchedQueryInput(token, head, dim);
    meanSquare = meanSquare + value * value;
  }
  batchedQueryReduceValues[lane] = meanSquare;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      batchedQueryReduceValues[lane] = batchedQueryReduceValues[lane] + batchedQueryReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let scale = inverseSqrt(batchedQueryReduceValues[0] / f32(params.headSize) + params.epsilon);
  let outBase = token * params.headCount * params.headSize + head * params.headSize;
  for (var dim = lane; dim < params.headSize; dim = dim + 256u) {
    queryValues[outBase + dim] = batchedQueryNormed(token, head, dim, scale);
  }
  let ropePairCount = params.ropeDims / 2u;
  let position = positions[token];
  for (var ic = lane; ic < ropePairCount; ic = ic + 256u) {
    let x0 = batchedQueryNormed(token, head, ic, scale);
    let x1 = batchedQueryNormed(token, head, ropePairCount + ic, scale);
    let theta = batchedQueryRopeTheta(position, ic);
    let thetaWithFactor = theta / batchedQueryRopeFactor(ic);
    let cosTheta = cos(thetaWithFactor);
    let sinTheta = sin(thetaWithFactor);
    queryValues[outBase + ic] = x0 * cosTheta - x1 * sinTheta;
    queryValues[outBase + ropePairCount + ic] = x0 * sinTheta + x1 * cosTheta;
  }
}
`;

export const BATCHED_FULL_KV_UPDATE_WGSL = `
struct Params {
  epsilon: f32,
  freqBase: f32,
  hasFreqFactors: u32,
  headCount: u32,
  headSize: u32,
  valueSize: u32,
  ropeDims: u32,
  tokenCount: u32,
  contextLength: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> kProjectionValues: array<f32>;
@group(0) @binding(1) var<storage, read> vProjectionValues: array<f32>;
@group(0) @binding(2) var<storage, read> normWeights: array<f32>;
@group(0) @binding(3) var<storage, read> freqFactors: array<f32>;
@group(0) @binding(4) var<storage, read> positions: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> keyCache: array<f32>;
@group(0) @binding(7) var<storage, read_write> valueCache: array<f32>;

fn batchedKvInput(token: u32, head: u32, dim: u32) -> f32 {
  return kProjectionValues[token * params.headCount * params.headSize + head * params.headSize + dim];
}

fn batchedKvNormed(token: u32, head: u32, dim: u32, scale: f32) -> f32 {
  return batchedKvInput(token, head, dim) * scale * normWeights[dim];
}

fn batchedKvRopeFactor(index: u32) -> f32 {
  if (params.hasFreqFactors == 0u) {
    return 1.0;
  }
  return freqFactors[index];
}

fn batchedKvRopeTheta(position: u32, pairIndex: u32) -> f32 {
  let thetaScale = pow(params.freqBase, -2.0 / f32(params.ropeDims));
  var theta = f32(position);
  for (var index = 0u; index < pairIndex; index = index + 1u) {
    theta = theta * thetaScale;
  }
  return theta;
}

var<workgroup> batchedKvReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let head = workgroupId.x;
  let token = workgroupId.y;
  let lane = localId.x;
  if (head >= params.headCount || token >= params.tokenCount) {
    return;
  }
  var meanSquare = 0.0;
  for (var dim = lane; dim < params.headSize; dim = dim + 256u) {
    let value = batchedKvInput(token, head, dim);
    meanSquare = meanSquare + value * value;
  }
  batchedKvReduceValues[lane] = meanSquare;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      batchedKvReduceValues[lane] = batchedKvReduceValues[lane] + batchedKvReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let scale = inverseSqrt(batchedKvReduceValues[0] / f32(params.headSize) + params.epsilon);
  let tokenPosition = positions[token];
  let keyBase = (tokenPosition * params.headCount + head) * params.headSize;
  for (var dim = lane; dim < params.headSize; dim = dim + 256u) {
    keyCache[keyBase + dim] = batchedKvNormed(token, head, dim, scale);
  }
  var valueMeanSquare = 0.0;
  let valueBase = token * params.headCount * params.valueSize + head * params.valueSize;
  for (var dim = lane; dim < params.valueSize; dim = dim + 256u) {
    let value = vProjectionValues[valueBase + dim];
    valueMeanSquare = valueMeanSquare + value * value;
  }
  batchedKvReduceValues[lane] = valueMeanSquare;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      batchedKvReduceValues[lane] = batchedKvReduceValues[lane] + batchedKvReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let valueScale = inverseSqrt(batchedKvReduceValues[0] / f32(params.valueSize) + params.epsilon);
  for (var dim = lane; dim < params.valueSize; dim = dim + 256u) {
    valueCache[(dim * params.headCount + head) * params.contextLength + tokenPosition] =
      vProjectionValues[valueBase + dim] * valueScale;
  }
  let ropePairCount = params.ropeDims / 2u;
  for (var ic = lane; ic < ropePairCount; ic = ic + 256u) {
    let x0 = batchedKvNormed(token, head, ic, scale);
    let x1 = batchedKvNormed(token, head, ropePairCount + ic, scale);
    let theta = batchedKvRopeTheta(tokenPosition, ic);
    let thetaWithFactor = theta / batchedKvRopeFactor(ic);
    let cosTheta = cos(thetaWithFactor);
    let sinTheta = sin(thetaWithFactor);
    keyCache[keyBase + ic] = x0 * cosTheta - x1 * sinTheta;
    keyCache[keyBase + ropePairCount + ic] = x0 * sinTheta + x1 * cosTheta;
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

export const GEGLU_SLICE_WGSL = `
struct Params {
  length: u32,
  rightOffset: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> gateValues: array<f32>;
@group(0) @binding(1) var<storage, read> rightValues: array<f32>;
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
  outputValues[index] = gelu(gateValues[index]) * rightValues[params.rightOffset + index];
}
`;

export const BATCHED_GEGLU_SLICE_WGSL = `
struct Params {
  length: u32,
  tokenCount: u32,
  rightOffset: u32,
  rightStride: u32,
};

@group(0) @binding(0) var<storage, read> gateValues: array<f32>;
@group(0) @binding(1) var<storage, read> rightValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

fn batchedGegluSliceF16BitsToF32(bits: u32) -> f32 {
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

fn batchedGegluSliceF32ToF16Bits(value: f32) -> u32 {
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

fn batchedGegluSliceCastF16(value: f32) -> f32 {
  return batchedGegluSliceF16BitsToF32(batchedGegluSliceF32ToF16Bits(value));
}

fn batchedGegluSliceGelu(value: f32) -> f32 {
  if (value <= -10.0) {
    return 0.0;
  }
  if (value >= 10.0) {
    return value;
  }
  let x = batchedGegluSliceCastF16(value);
  let inner = sqrt(2.0 / 3.141592653589793) * x * (1.0 + 0.044715 * x * x);
  return batchedGegluSliceCastF16(0.5 * x * (1.0 + tanh(inner)));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let token = id.y;
  if (index >= params.length || token >= params.tokenCount) {
    return;
  }
  let leftIndex = token * params.length + index;
  let rightIndex = params.rightOffset + token * params.rightStride + index;
  outputValues[leftIndex] = batchedGegluSliceGelu(gateValues[leftIndex]) * rightValues[rightIndex];
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

export const BATCHED_RMS_NORM_Q8_K_QUANTIZE_WGSL = `
struct Params {
  epsilon: f32,
  length: u32,
  blockCount: u32,
  tokenCount: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputScales: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputQs: array<i32>;
@group(0) @binding(4) var<storage, read_write> outputBsums: array<i32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> batchedRmsQ8ReduceValues: array<f32, 256>;
var<workgroup> batchedRmsQ8AbsValues: array<f32, 256>;
var<workgroup> batchedRmsQ8Values: array<f32, 256>;
var<workgroup> batchedRmsQ8Indices: array<u32, 256>;
var<workgroup> batchedRmsQ8Quants: array<i32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let token = workgroupId.y;
  let lane = localId.x;
  if (token >= params.tokenCount) {
    return;
  }
  let tokenBase = token * params.length;
  var localSum = 0.0;
  for (var index = lane; index < params.length; index = index + 256u) {
    let value = inputValues[tokenBase + index];
    localSum = localSum + value * value;
  }
  batchedRmsQ8ReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      batchedRmsQ8ReduceValues[lane] = batchedRmsQ8ReduceValues[lane] + batchedRmsQ8ReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let rmsScale = inverseSqrt(batchedRmsQ8ReduceValues[0] / f32(params.length) + params.epsilon);
  for (var block = 0u; block < params.blockCount; block = block + 1u) {
    let element = block * 256u + lane;
    let value = inputValues[tokenBase + element] * rmsScale * weightValues[element];
    batchedRmsQ8AbsValues[lane] = abs(value);
    batchedRmsQ8Values[lane] = value;
    batchedRmsQ8Indices[lane] = lane;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        let otherLane = lane + stride;
        let otherAbs = batchedRmsQ8AbsValues[otherLane];
        let otherIndex = batchedRmsQ8Indices[otherLane];
        if (otherAbs > batchedRmsQ8AbsValues[lane] || (otherAbs == batchedRmsQ8AbsValues[lane] && otherIndex < batchedRmsQ8Indices[lane])) {
          batchedRmsQ8AbsValues[lane] = otherAbs;
          batchedRmsQ8Values[lane] = batchedRmsQ8Values[otherLane];
          batchedRmsQ8Indices[lane] = otherIndex;
        }
      }
      workgroupBarrier();
    }
    let amax = batchedRmsQ8AbsValues[0];
    let maxValue = batchedRmsQ8Values[0];
    var inverseScale = 0.0;
    if (amax != 0.0) {
      inverseScale = -127.0 / maxValue;
    }
    let blockIndex = token * params.blockCount + block;
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
    batchedRmsQ8Quants[lane] = q;
    outputQs[tokenBase + element] = q;
    workgroupBarrier();
    if (lane < 16u) {
      var sum = 0i;
      for (var offset = 0u; offset < 16u; offset = offset + 1u) {
        sum = sum + batchedRmsQ8Quants[lane * 16u + offset];
      }
      outputBsums[blockIndex * 16u + lane] = sum;
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

var<workgroup> q8_0AbsValues: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let column = workgroupId.x;
  let block = workgroupId.y;
  let lane = localId.x;
  let base = column * params.inputSize + block * 32u;
  q8_0AbsValues[lane] = abs(inputValues[base + lane]);
  workgroupBarrier();
  for (var stride = 16u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      q8_0AbsValues[lane] = max(q8_0AbsValues[lane], q8_0AbsValues[lane + stride]);
    }
    workgroupBarrier();
  }
  let blockIndex = column * params.blockCount + block;
  let amax = q8_0AbsValues[0];
  let scale = f16BitsToF32(f32ToF16Bits(amax / 127.0));
  if (lane == 0u) {
    outputScales[blockIndex] = scale;
  }
  let inverseScale = select(0.0, 1.0 / scale, scale != 0.0);
  outputQs[base + lane] = i32(round(inputValues[base + lane] * inverseScale));
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

var<workgroup> attentionApplyValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let qHead = workgroupId.x;
  let dim = workgroupId.y;
  let lane = localId.x;
  let groupSize = params.queryHeadCount / params.keyValueHeadCount;
  let kvHead = qHead / groupSize;
  var weighted = 0.0;
  let probabilityOffset = qHead * params.keyValueTokenCount;
  for (var keyToken = params.keyValueStart + lane; keyToken < params.keyValueTokenCount; keyToken = keyToken + 256u) {
    let probability = probabilityValues[probabilityOffset + keyToken];
    let valueIndex = (dim * params.keyValueHeadCount + kvHead) * params.contextLength + keyToken;
    weighted = weighted + probability * valueValues[valueIndex];
  }
  attentionApplyValues[lane] = weighted;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      attentionApplyValues[lane] = attentionApplyValues[lane] + attentionApplyValues[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let outputIndex = qHead * params.valueSize + dim;
    outputValues[outputIndex] = attentionApplyValues[0];
  }
}
`;

export const BATCHED_FULL_ATTENTION_SCORE_WGSL = `
struct Params {
  scale: f32,
  headSize: u32,
  queryHeadCount: u32,
  keyValueHeadCount: u32,
  keyValueTokenCount: u32,
  contextLength: u32,
  probabilityTokenCapacity: u32,
  slidingWindow: u32,
  hasSlidingWindow: u32,
  tokenCount: u32,
  causal: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> queryValues: array<f32>;
@group(0) @binding(1) var<storage, read> keyValues: array<f32>;
@group(0) @binding(2) var<storage, read> positions: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> probabilityValues: array<f32>;

fn batchedScoreAllowed(queryPosition: u32, keyToken: u32) -> bool {
  if (params.causal != 0u && keyToken > queryPosition) {
    return false;
  }
  if (params.hasSlidingWindow != 0u && keyToken + params.slidingWindow <= queryPosition) {
    return false;
  }
  return keyToken < params.keyValueTokenCount;
}

fn batchedAttentionScore(token: u32, qHead: u32, kvHead: u32, keyToken: u32) -> f32 {
  let queryOffset = token * params.queryHeadCount * params.headSize + qHead * params.headSize;
  let keyOffset = (keyToken * params.keyValueHeadCount + kvHead) * params.headSize;
  var dot = 0.0;
  for (var dim = 0u; dim < params.headSize; dim = dim + 1u) {
    dot = dot + queryValues[queryOffset + dim] * keyValues[keyOffset + dim];
  }
  return dot * params.scale;
}

var<workgroup> batchedAttentionScoreReduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let qHead = workgroupId.x;
  let token = workgroupId.y;
  let lane = localId.x;
  if (qHead >= params.queryHeadCount || token >= params.tokenCount) {
    return;
  }
  let groupSize = params.queryHeadCount / params.keyValueHeadCount;
  let kvHead = qHead / groupSize;
  let queryPosition = positions[token];
  let probabilityOffset = (token * params.queryHeadCount + qHead) * params.probabilityTokenCapacity;
  var localMax = -3.4028234663852886e38;
  for (var keyToken = lane; keyToken < params.keyValueTokenCount; keyToken = keyToken + 256u) {
    var score = -3.4028234663852886e38;
    if (batchedScoreAllowed(queryPosition, keyToken)) {
      score = batchedAttentionScore(token, qHead, kvHead, keyToken);
    }
    probabilityValues[probabilityOffset + keyToken] = score;
    localMax = max(localMax, score);
  }
  batchedAttentionScoreReduceValues[lane] = localMax;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      batchedAttentionScoreReduceValues[lane] = max(batchedAttentionScoreReduceValues[lane], batchedAttentionScoreReduceValues[lane + stride]);
    }
    workgroupBarrier();
  }
  let maxScore = batchedAttentionScoreReduceValues[0];
  var localSum = 0.0;
  for (var keyToken = lane; keyToken < params.keyValueTokenCount; keyToken = keyToken + 256u) {
    let index = probabilityOffset + keyToken;
    var probability = 0.0;
    if (batchedScoreAllowed(queryPosition, keyToken)) {
      probability = exp(probabilityValues[index] - maxScore);
    }
    probabilityValues[index] = probability;
    localSum = localSum + probability;
  }
  batchedAttentionScoreReduceValues[lane] = localSum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      batchedAttentionScoreReduceValues[lane] = batchedAttentionScoreReduceValues[lane] + batchedAttentionScoreReduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let sum = batchedAttentionScoreReduceValues[0];
  for (var keyToken = lane; keyToken < params.keyValueTokenCount; keyToken = keyToken + 256u) {
    let index = probabilityOffset + keyToken;
    if (sum != 0.0) {
      probabilityValues[index] = probabilityValues[index] / sum;
    }
  }
}
`;

export const BATCHED_FULL_ATTENTION_APPLY_WGSL = `
struct Params {
  scale: f32,
  valueSize: u32,
  queryHeadCount: u32,
  keyValueHeadCount: u32,
  keyValueTokenCount: u32,
  contextLength: u32,
  probabilityTokenCapacity: u32,
  slidingWindow: u32,
  hasSlidingWindow: u32,
  tokenCount: u32,
  causal: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> valueValues: array<f32>;
@group(0) @binding(1) var<storage, read> probabilityValues: array<f32>;
@group(0) @binding(2) var<storage, read> positions: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;

fn batchedApplyAllowed(queryPosition: u32, keyToken: u32) -> bool {
  if (params.causal != 0u && keyToken > queryPosition) {
    return false;
  }
  if (params.hasSlidingWindow != 0u && keyToken + params.slidingWindow <= queryPosition) {
    return false;
  }
  return keyToken < params.keyValueTokenCount;
}

var<workgroup> batchedAttentionApplyValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let qHead = workgroupId.x;
  let dim = workgroupId.y;
  let token = workgroupId.z;
  let lane = localId.x;
  if (qHead >= params.queryHeadCount || dim >= params.valueSize || token >= params.tokenCount) {
    return;
  }
  let groupSize = params.queryHeadCount / params.keyValueHeadCount;
  let kvHead = qHead / groupSize;
  let queryPosition = positions[token];
  let probabilityOffset = (token * params.queryHeadCount + qHead) * params.probabilityTokenCapacity;
  var weighted = 0.0;
  for (var keyToken = lane; keyToken < params.keyValueTokenCount; keyToken = keyToken + 256u) {
    if (batchedApplyAllowed(queryPosition, keyToken)) {
      let probability = probabilityValues[probabilityOffset + keyToken];
      let valueIndex = (dim * params.keyValueHeadCount + kvHead) * params.contextLength + keyToken;
      weighted = weighted + probability * valueValues[valueIndex];
    }
  }
  batchedAttentionApplyValues[lane] = weighted;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      batchedAttentionApplyValues[lane] = batchedAttentionApplyValues[lane] + batchedAttentionApplyValues[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let outputIndex = token * params.queryHeadCount * params.valueSize + qHead * params.valueSize + dim;
    outputValues[outputIndex] = batchedAttentionApplyValues[0];
  }
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

export const VISION_PATCH_EMBED_WGSL = `
struct Params {
  imageWidth: u32,
  patchSize: u32,
  patchGridX: u32,
  patchGridY: u32,
  embeddingLength: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> pixels: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let patchCount = params.patchGridX * params.patchGridY;
  if (index >= patchCount * params.embeddingLength) {
    return;
  }
  let emb = index % params.embeddingLength;
  let patchIndex = index / params.embeddingLength;
  let patchX = patchIndex % params.patchGridX;
  let patchY = patchIndex / params.patchGridX;
  var sum = 0.0;
  for (var ky = 0u; ky < params.patchSize; ky = ky + 1u) {
    let y = patchY * params.patchSize + ky;
    for (var kx = 0u; kx < params.patchSize; kx = kx + 1u) {
      let x = patchX * params.patchSize + kx;
      let pixelOffset = (y * params.imageWidth + x) * 3u;
      for (var channel = 0u; channel < 3u; channel = channel + 1u) {
        let weightOffset = kx + params.patchSize * (ky + params.patchSize * (channel + 3u * emb));
        let scaledPixel = pixels[pixelOffset + channel] * 2.0 - 1.0;
        sum = sum + weights[weightOffset] * scaledPixel;
      }
    }
  }
  outputValues[index] = sum;
}
`;

export const VISION_ADD_POSITION_WGSL = `
struct Params {
  patchGridX: u32,
  tokenCount: u32,
  embeddingLength: u32,
  tableSize: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> positionValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.tokenCount * params.embeddingLength) {
    return;
  }
  let emb = index % params.embeddingLength;
  let token = index / params.embeddingLength;
  let x = token % params.patchGridX;
  let y = token / params.patchGridX;
  outputValues[index] = inputValues[index] +
    positionValues[x * params.embeddingLength + emb] +
    positionValues[(params.tableSize + y) * params.embeddingLength + emb];
}
`;

export const VISION_RMS_NORM_WGSL = `
struct Params {
  epsilon: f32,
  rowSize: u32,
  rowCount: u32,
  hasWeight: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

var<workgroup> reduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let row = workgroupId.x;
  let lane = localId.x;
  if (row >= params.rowCount) {
    return;
  }
  let base = row * params.rowSize;
  var sum = 0.0;
  for (var dim = lane; dim < params.rowSize; dim = dim + 256u) {
    let value = inputValues[base + dim];
    sum = sum + value * value;
  }
  reduceValues[lane] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      reduceValues[lane] = reduceValues[lane] + reduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  let normScale = inverseSqrt(reduceValues[0] / f32(params.rowSize) + params.epsilon);
  for (var dim = lane; dim < params.rowSize; dim = dim + 256u) {
    let weight = select(1.0, weightValues[dim], params.hasWeight == 1u);
    outputValues[base + dim] = inputValues[base + dim] * normScale * weight;
  }
}
`;

export const VISION_ROPE2D_WGSL = `
struct Params {
  freqBase: f32,
  patchGridX: u32,
  tokenCount: u32,
  headCount: u32,
  headSize: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

fn rotatedValue(token: u32, head: u32, dim: u32, sliceStart: u32, position: u32) -> f32 {
  let sliceLength = params.headSize / 2u;
  let half = sliceLength / 2u;
  let local = dim - sliceStart;
  let pair = local % half;
  let base = (token * params.headCount + head) * params.headSize + sliceStart;
  let x0 = inputValues[base + pair];
  let x1 = inputValues[base + half + pair];
  let thetaScale = pow(params.freqBase, -2.0 / f32(sliceLength));
  let theta = f32(position) * pow(thetaScale, f32(pair));
  let c = cos(theta);
  let s = sin(theta);
  if (local < half) {
    return x0 * c - x1 * s;
  }
  return x0 * s + x1 * c;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let total = params.tokenCount * params.headCount * params.headSize;
  if (index >= total) {
    return;
  }
  let dim = index % params.headSize;
  let headToken = index / params.headSize;
  let head = headToken % params.headCount;
  let token = headToken / params.headCount;
  let sliceLength = params.headSize / 2u;
  if (dim < sliceLength) {
    outputValues[index] = rotatedValue(token, head, dim, 0u, token % params.patchGridX);
    return;
  }
  outputValues[index] = rotatedValue(token, head, dim, sliceLength, token / params.patchGridX);
}
`;

export const VISION_ATTENTION_SCORE_WGSL = `
struct Params {
  scale: f32,
  tokenCount: u32,
  headCount: u32,
  headSize: u32,
};

@group(0) @binding(0) var<storage, read> queryValues: array<f32>;
@group(0) @binding(1) var<storage, read> keyValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> probabilityValues: array<f32>;

var<workgroup> reduceValues: array<f32, 256>;

fn dotScore(queryToken: u32, head: u32, keyToken: u32) -> f32 {
  let qBase = (queryToken * params.headCount + head) * params.headSize;
  let kBase = (keyToken * params.headCount + head) * params.headSize;
  var dot = 0.0;
  for (var dim = 0u; dim < params.headSize; dim = dim + 1u) {
    dot = dot + queryValues[qBase + dim] * keyValues[kBase + dim];
  }
  return dot * params.scale;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let queryToken = workgroupId.x;
  let head = workgroupId.y;
  let lane = localId.x;
  let probabilityBase = (queryToken * params.headCount + head) * params.tokenCount;
  var localMax = -3.4028234663852886e38;
  for (var keyToken = lane; keyToken < params.tokenCount; keyToken = keyToken + 256u) {
    let score = dotScore(queryToken, head, keyToken);
    probabilityValues[probabilityBase + keyToken] = score;
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
  for (var keyToken = lane; keyToken < params.tokenCount; keyToken = keyToken + 256u) {
    let probability = exp(probabilityValues[probabilityBase + keyToken] - maxScore);
    probabilityValues[probabilityBase + keyToken] = probability;
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
  let total = reduceValues[0];
  for (var keyToken = lane; keyToken < params.tokenCount; keyToken = keyToken + 256u) {
    let index = probabilityBase + keyToken;
    probabilityValues[index] = probabilityValues[index] / total;
  }
}
`;

export const VISION_ATTENTION_APPLY_WGSL = `
struct Params {
  tokenCount: u32,
  headCount: u32,
  headSize: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> valueValues: array<f32>;
@group(0) @binding(1) var<storage, read> probabilityValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

var<workgroup> reduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let queryToken = workgroupId.x;
  let head = workgroupId.y;
  let dim = workgroupId.z;
  let lane = localId.x;
  let probabilityBase = (queryToken * params.headCount + head) * params.tokenCount;
  var local = 0.0;
  for (var keyToken = lane; keyToken < params.tokenCount; keyToken = keyToken + 256u) {
    let valueIndex = (keyToken * params.headCount + head) * params.headSize + dim;
    local = local + probabilityValues[probabilityBase + keyToken] * valueValues[valueIndex];
  }
  reduceValues[lane] = local;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      reduceValues[lane] = reduceValues[lane] + reduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    outputValues[(queryToken * params.headCount + head) * params.headSize + dim] = reduceValues[0];
  }
}
`;

export const VISION_AVERAGE_POOL_WGSL = `
struct Params {
  outputScale: f32,
  patchGridX: u32,
  patchGridY: u32,
  embeddingLength: u32,
  kernelSize: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let outX = params.patchGridX / params.kernelSize;
  let outY = params.patchGridY / params.kernelSize;
  let index = id.x;
  if (index >= outX * outY * params.embeddingLength) {
    return;
  }
  let emb = index % params.embeddingLength;
  let outToken = index / params.embeddingLength;
  let ox = outToken % outX;
  let oy = outToken / outX;
  let scale = params.outputScale / f32(params.kernelSize * params.kernelSize);
  var sum = 0.0;
  for (var ky = 0u; ky < params.kernelSize; ky = ky + 1u) {
    for (var kx = 0u; kx < params.kernelSize; kx = kx + 1u) {
      let inToken = (oy * params.kernelSize + ky) * params.patchGridX + ox * params.kernelSize + kx;
      sum = sum + inputValues[inToken * params.embeddingLength + emb] * scale;
    }
  }
  outputValues[index] = sum;
}
`;

export const VISION_STD_NORMALIZE_WGSL = `
struct Params {
  length: u32,
  rowSize: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> biasValues: array<f32>;
@group(0) @binding(2) var<storage, read> scaleValues: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  let dim = index % params.rowSize;
  outputValues[index] = (inputValues[index] - biasValues[dim]) * scaleValues[dim];
}
`;

export const VISION_CLAMP_WGSL = `
struct Params {
  minValue: f32,
  maxValue: f32,
  length: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  outputValues[index] = min(params.maxValue, max(params.minValue, inputValues[index]));
}
`;

export const VISION_GELU_MUL_WGSL = `
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

fn gelu(value: f32) -> f32 {
  if (value <= -10.0) {
    return 0.0;
  }
  if (value >= 10.0) {
    return value;
  }
  let inner = sqrt(2.0 / 3.141592653589793) * value * (1.0 + 0.044715 * value * value);
  return 0.5 * value * (1.0 + tanh(inner));
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

export const AUDIO_CONV2D_SUBSAMPLE_WGSL = `
struct Params {
  epsilon: f32,
  time: u32,
  frequency: u32,
  inChannels: u32,
  outChannels: u32,
  outTime: u32,
  outFrequency: u32,
  hasBias: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> maskValues: array<u32>;
@group(0) @binding(2) var<storage, read> weightValues: array<f32>;
@group(0) @binding(3) var<storage, read> biasValues: array<f32>;
@group(0) @binding(4) var<storage, read> normValues: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> outputValues: array<f32>;

var<workgroup> channelValues: array<f32, 256>;
var<workgroup> reduceValues: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let tOut = workgroupId.x;
  let fOut = workgroupId.y;
  let channel = localId.x;
  if (tOut >= params.outTime || fOut >= params.outFrequency) {
    return;
  }

  var sum = 0.0;
  if (channel < params.outChannels) {
    for (var kt = 0u; kt < 3u; kt = kt + 1u) {
      let tSigned = i32(tOut * 2u + kt) - 1;
      if (tSigned >= 0 && tSigned < i32(params.time) && maskValues[u32(tSigned)] != 0u) {
        let tIn = u32(tSigned);
        for (var kf = 0u; kf < 3u; kf = kf + 1u) {
          let fSigned = i32(fOut * 2u + kf) - 1;
          if (fSigned >= 0 && fSigned < i32(params.frequency)) {
            let fIn = u32(fSigned);
            for (var inChannel = 0u; inChannel < params.inChannels; inChannel = inChannel + 1u) {
              let inputIndex = (tIn * params.inChannels + inChannel) * params.frequency + fIn;
              let weightIndex = kf + kt * 3u + inChannel * 9u + channel * 9u * params.inChannels;
              sum = sum + inputValues[inputIndex] * weightValues[weightIndex];
            }
          }
        }
      }
    }
    if (params.hasBias == 1u) {
      sum = sum + biasValues[channel];
    }
  }
  channelValues[channel] = sum;
  let meanTerm = select(0.0, sum, channel < params.outChannels);
  reduceValues[channel] = meanTerm;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (channel < stride) {
      reduceValues[channel] = reduceValues[channel] + reduceValues[channel + stride];
    }
    workgroupBarrier();
  }
  let mean = reduceValues[0] / f32(params.outChannels);
  let centered = select(0.0, channelValues[channel] - mean, channel < params.outChannels);
  reduceValues[channel] = centered * centered;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (channel < stride) {
      reduceValues[channel] = reduceValues[channel] + reduceValues[channel + stride];
    }
    workgroupBarrier();
  }
  if (channel < params.outChannels) {
    let scale = inverseSqrt(reduceValues[0] / f32(params.outChannels) + params.epsilon);
    let normalized = centered * scale * normValues[channel];
    outputValues[(tOut * params.outChannels + channel) * params.outFrequency + fOut] = max(0.0, normalized);
  }
}
`;

export const AUDIO_FLATTEN_CHANNELS_LAST_WGSL = `
struct Params {
  timeCount: u32,
  frequencyCount: u32,
  channelCount: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let total = params.timeCount * params.frequencyCount * params.channelCount;
  if (index >= total) {
    return;
  }
  let channel = index % params.channelCount;
  let frequency = (index / params.channelCount) % params.frequencyCount;
  let time = index / (params.frequencyCount * params.channelCount);
  outputValues[index] = inputValues[(time * params.channelCount + channel) * params.frequencyCount + frequency];
}
`;

export const AUDIO_SILU_WGSL = `
struct Params {
  length: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  let value = inputValues[index];
  outputValues[index] = value / (1.0 + exp(-value));
}
`;

export const AUDIO_GLU_WGSL = `
struct Params {
  tokenCount: u32,
  outputSize: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.tokenCount * params.outputSize) {
    return;
  }
  let dim = index % params.outputSize;
  let token = index / params.outputSize;
  let inputOffset = token * params.outputSize * 2u;
  let gate = inputValues[inputOffset + params.outputSize + dim];
  outputValues[index] = inputValues[inputOffset + dim] / (1.0 + exp(-gate));
}
`;

export const AUDIO_DEPTHWISE_CONV1D_WGSL = `
struct Params {
  tokenCount: u32,
  kernelSize: u32,
  channels: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.tokenCount * params.channels) {
    return;
  }
  let channel = index % params.channels;
  let token = index / params.channels;
  let leftPad = params.kernelSize - 1u;
  var sum = 0.0;
  for (var kernel = 0u; kernel < params.kernelSize; kernel = kernel + 1u) {
    let source = i32(token + kernel) - i32(leftPad);
    if (source >= 0 && source < i32(params.tokenCount)) {
      sum = sum + inputValues[u32(source) * params.channels + channel] * weightValues[kernel + channel * params.kernelSize];
    }
  }
  outputValues[index] = sum;
}
`;

export const AUDIO_ADD_BIAS_ROWS_WGSL = `
struct Params {
  length: u32,
  rowSize: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> biasValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  outputValues[index] = inputValues[index] + biasValues[index % params.rowSize];
}
`;

export const AUDIO_RESIDUAL_ADD_SCALE_WGSL = `
struct Params {
  scale: f32,
  length: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> residualValues: array<f32>;
@group(0) @binding(1) var<storage, read> hiddenValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.length) {
    return;
  }
  outputValues[index] = residualValues[index] + hiddenValues[index] * params.scale;
}
`;

export const AUDIO_ATTENTION_WGSL = `
struct Params {
  invalidLogit: f32,
  logitCap: f32,
  qScale: f32,
  kScale: f32,
  tokenCount: u32,
  headCount: u32,
  headSize: u32,
  embeddingLength: u32,
  attentionChunkSize: u32,
  maxPast: u32,
  hasPerDimKScale: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> qValues: array<f32>;
@group(0) @binding(1) var<storage, read> kValues: array<f32>;
@group(0) @binding(2) var<storage, read> vValues: array<f32>;
@group(0) @binding(3) var<storage, read> relativeKValues: array<f32>;
@group(0) @binding(4) var<storage, read> maskValues: array<u32>;
@group(0) @binding(5) var<storage, read> perDimScaleValues: array<f32>;
@group(0) @binding(6) var<storage, read> perDimKScaleValues: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;
@group(0) @binding(8) var<storage, read_write> outputValues: array<f32>;

var<workgroup> scoreValues: array<f32, 256>;

fn validKey(queryToken: u32, keyToken: u32) -> bool {
  return keyToken < params.tokenCount &&
    keyToken <= queryToken &&
    queryToken - keyToken < params.maxPast &&
    maskValues[keyToken] != 0u;
}

fn attentionScore(queryToken: u32, head: u32, keyToken: u32, relIndex: u32) -> f32 {
  let qBase = queryToken * params.embeddingLength + head * params.headSize;
  let kBase = keyToken * params.embeddingLength + head * params.headSize;
  let relBase = relIndex * params.embeddingLength + head * params.headSize;
  var score = 0.0;
  for (var dim = 0u; dim < params.headSize; dim = dim + 1u) {
    let qValue = qValues[qBase + dim] * params.qScale * perDimScaleValues[dim];
    let kScale = select(1.0, perDimKScaleValues[dim], params.hasPerDimKScale == 1u);
    let kValue = kValues[kBase + dim] * params.kScale * kScale;
    let relValue = relativeKValues[relBase + dim];
    score = score + qValue * (kValue + relValue);
  }
  return tanh(score / params.logitCap) * params.logitCap;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let queryToken = workgroupId.x;
  let head = workgroupId.y;
  let dim = workgroupId.z;
  let lane = localId.x;
  let outIndex = queryToken * params.embeddingLength + head * params.headSize + dim;
  let contextSize = params.attentionChunkSize + params.maxPast;
  if (queryToken >= params.tokenCount || head >= params.headCount || dim >= params.headSize || maskValues[queryToken] == 0u) {
    if (lane == 0u && queryToken < params.tokenCount && head < params.headCount && dim < params.headSize) {
      outputValues[outIndex] = 0.0;
    }
    return;
  }

  let block = queryToken / params.attentionChunkSize;
  let blockStart = block * params.attentionChunkSize;
  let queryInBlock = queryToken - blockStart;
  let contextStart = i32(blockStart) - i32(params.maxPast);
  var localMax = -3.4028234663852886e38;
  for (var context = lane; context < contextSize; context = context + 256u) {
    let keySigned = contextStart + i32(context);
    var score = params.invalidLogit;
    if (keySigned >= 0) {
      let keyToken = u32(keySigned);
      let relIndex = context - queryInBlock;
      if (validKey(queryToken, keyToken) && relIndex < params.maxPast + 1u) {
        score = attentionScore(queryToken, head, keyToken, relIndex);
      }
    }
    scoreValues[context] = score;
    localMax = max(localMax, score);
  }
  scoreValues[128u + lane] = localMax;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      scoreValues[128u + lane] = max(scoreValues[128u + lane], scoreValues[128u + lane + stride]);
    }
    workgroupBarrier();
  }
  let maxScore = scoreValues[128u];
  var localSum = 0.0;
  for (var context = lane; context < contextSize; context = context + 256u) {
    let expValue = exp(scoreValues[context] - maxScore);
    scoreValues[context] = expValue;
    localSum = localSum + expValue;
  }
  scoreValues[128u + lane] = localSum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      scoreValues[128u + lane] = scoreValues[128u + lane] + scoreValues[128u + lane + stride];
    }
    workgroupBarrier();
  }
  let total = scoreValues[128u];
  var local = 0.0;
  if (total > 0.0) {
    for (var context = lane; context < contextSize; context = context + 256u) {
      let keySigned = contextStart + i32(context);
      if (keySigned >= 0) {
        let keyToken = u32(keySigned);
        if (validKey(queryToken, keyToken)) {
          let probability = scoreValues[context] / total;
          local = local + probability * vValues[keyToken * params.embeddingLength + head * params.headSize + dim];
        }
      }
    }
  }
  scoreValues[128u + lane] = local;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      scoreValues[128u + lane] = scoreValues[128u + lane] + scoreValues[128u + lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    outputValues[outIndex] = scoreValues[128u];
  }
}
`;
