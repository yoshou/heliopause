import {
  GPU_COPY_DST,
  GPU_SHADER_STAGE_COMPUTE,
  GPU_UNIFORM,
} from "./gpu-constants";
import {
  bindBuffer,
  storageEntry,
} from "./gpu-bindings";
import type {
  WebGpuBufferLike,
  WebGpuDeviceLike,
} from "./gpu-types";

export type MtpKernelResource = { pipeline: unknown; bindGroup: unknown; destroy: () => void };

function uniformBufferSize(byteLength: number): number {
  return Math.max(32, Math.ceil(byteLength / 16) * 16);
}

export function createMtpConcat2Resources(
  device: WebGpuDeviceLike,
  left: WebGpuBufferLike,
  right: WebGpuBufferLike,
  output: WebGpuBufferLike,
  leftLength: number,
  rightLength: number,
): MtpKernelResource {
  const params = new Uint32Array([leftLength, rightLength, 0, 0]);
  const paramsBuffer = device.createBuffer({
    size: uniformBufferSize(params.byteLength),
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const layout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      { binding: 2, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(3, "storage"),
    ],
  });
  return {
    pipeline: device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: device.createShaderModule({ code: MTP_CONCAT2_WGSL }), entryPoint: "main" },
    }),
    bindGroup: device.createBindGroup({
      layout,
      entries: [bindBuffer(0, left), bindBuffer(1, right), bindBuffer(2, paramsBuffer), bindBuffer(3, output)],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createMtpRopeResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  freqFactors: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    activePairCount: number;
    freqBase: number;
    position: number;
    hasFreqFactors: boolean;
  },
): MtpKernelResource {
  const params = new ArrayBuffer(48);
  const f32 = new Float32Array(params);
  const u32 = new Uint32Array(params);
  f32[0] = options.freqBase;
  f32[1] = options.position;
  u32[2] = options.hasFreqFactors ? 1 : 0;
  u32[3] = options.headCount;
  u32[4] = options.headSize;
  u32[5] = options.ropeDims;
  u32[6] = options.activePairCount;
  const paramsBuffer = device.createBuffer({
    size: uniformBufferSize(params.byteLength),
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const layout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      { binding: 2, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(3, "storage"),
    ],
  });
  return {
    pipeline: device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: device.createShaderModule({ code: MTP_ROPE_WGSL }), entryPoint: "main" },
    }),
    bindGroup: device.createBindGroup({
      layout,
      entries: [bindBuffer(0, input), bindBuffer(1, freqFactors), bindBuffer(2, paramsBuffer), bindBuffer(3, output)],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createMtpAttentionResources(
  device: WebGpuDeviceLike,
  query: WebGpuBufferLike,
  key: WebGpuBufferLike,
  value: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    headSize: number;
    valueSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    position: number;
    slidingWindow?: number;
  },
): MtpKernelResource {
  const params = new Uint32Array([
    options.headSize,
    options.valueSize,
    options.queryHeadCount,
    options.keyValueHeadCount,
    options.keyValueTokenCount,
    options.contextLength,
    options.position,
    options.slidingWindow ?? 0,
    options.slidingWindow === undefined ? 0 : 1,
    0,
    0,
    0,
  ]);
  const paramsBuffer = device.createBuffer({
    size: uniformBufferSize(params.byteLength),
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const layout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      { binding: 3, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(4, "storage"),
    ],
  });
  return {
    pipeline: device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: device.createShaderModule({ code: MTP_ATTENTION_WGSL }), entryPoint: "main" },
    }),
    bindGroup: device.createBindGroup({
      layout,
      entries: [
        bindBuffer(0, query),
        bindBuffer(1, key),
        bindBuffer(2, value),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, output),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createMtpEmbeddingDotResources(
  device: WebGpuDeviceLike,
  rows: WebGpuBufferLike,
  hidden: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    candidateCount: number;
    embeddingLength: number;
  },
): MtpKernelResource {
  const params = new Uint32Array([options.candidateCount, options.embeddingLength, 0, 0]);
  const paramsBuffer = device.createBuffer({
    size: uniformBufferSize(params.byteLength),
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const layout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      { binding: 2, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(3, "storage"),
    ],
  });
  return {
    pipeline: device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: device.createShaderModule({ code: MTP_EMBEDDING_DOT_WGSL }), entryPoint: "main" },
    }),
    bindGroup: device.createBindGroup({
      layout,
      entries: [bindBuffer(0, rows), bindBuffer(1, hidden), bindBuffer(2, paramsBuffer), bindBuffer(3, output)],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

const MTP_CONCAT2_WGSL = `
struct Params {
  leftLength: u32,
  rightLength: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> leftValues: array<f32>;
@group(0) @binding(1) var<storage, read> rightValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let total = params.leftLength + params.rightLength;
  if (index >= total) {
    return;
  }
  if (index < params.leftLength) {
    outputValues[index] = leftValues[index];
  } else {
    outputValues[index] = rightValues[index - params.leftLength];
  }
}
`;

const MTP_ROPE_WGSL = `
struct Params {
  freqBase: f32,
  position: f32,
  hasFreqFactors: u32,
  headCount: u32,
  headSize: u32,
  ropeDims: u32,
  activePairCount: u32,
  _pad0: u32,
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let total = params.headCount * params.headSize;
  if (index >= total) {
    return;
  }
  let headOffset = (index / params.headSize) * params.headSize;
  let dim = index % params.headSize;
  let pairCount = select(params.ropeDims / 2u, params.activePairCount, params.activePairCount > 0u);
  let half = params.ropeDims / 2u;
  if (dim < pairCount) {
    let pair = dim;
    let x0 = inputValues[headOffset + pair];
    let x1 = inputValues[headOffset + half + pair];
    let theta = ropeTheta(pair) / ropeFactor(pair);
    outputValues[index] = x0 * cos(theta) - x1 * sin(theta);
  } else if (dim >= half && dim < half + pairCount) {
    let pair = dim - half;
    let x0 = inputValues[headOffset + pair];
    let x1 = inputValues[headOffset + half + pair];
    let theta = ropeTheta(pair) / ropeFactor(pair);
    outputValues[index] = x0 * sin(theta) + x1 * cos(theta);
  } else {
    outputValues[index] = inputValues[index];
  }
}
`;

const MTP_ATTENTION_WGSL = `
struct Params {
  headSize: u32,
  valueSize: u32,
  queryHeadCount: u32,
  keyValueHeadCount: u32,
  keyValueTokenCount: u32,
  contextLength: u32,
  position: u32,
  slidingWindow: u32,
  hasSlidingWindow: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> queryValues: array<f32>;
@group(0) @binding(1) var<storage, read> keyValues: array<f32>;
@group(0) @binding(2) var<storage, read> valueValues: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;

fn keyAllowed(token: u32) -> bool {
  if (token > params.position) {
    return false;
  }
  if (params.hasSlidingWindow == 0u) {
    return true;
  }
  let minPosition = select(0u, params.position + 1u - params.slidingWindow, params.position + 1u > params.slidingWindow);
  return token >= minPosition;
}

fn attentionScore(qHead: u32, kvHead: u32, token: u32) -> f32 {
  let qOffset = qHead * params.headSize;
  let kOffset = (token * params.keyValueHeadCount + kvHead) * params.headSize;
  var sum = 0.0;
  for (var dim = 0u; dim < params.headSize; dim = dim + 1u) {
    sum = sum + queryValues[qOffset + dim] * keyValues[kOffset + dim];
  }
  return sum;
}

var<workgroup> reduceValues: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let qHead = group.x;
  let dim = group.y;
  let lane = local.x;
  if (qHead >= params.queryHeadCount || dim >= params.valueSize) {
    return;
  }
  let groupSize = params.queryHeadCount / params.keyValueHeadCount;
  let kvHead = qHead / groupSize;
  var localMax = -3.4028234663852886e38;
  for (var token = lane; token < params.keyValueTokenCount; token = token + 256u) {
    if (keyAllowed(token)) {
      localMax = max(localMax, attentionScore(qHead, kvHead, token));
    }
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
  workgroupBarrier();
  var localSum = 0.0;
  var localWeighted = 0.0;
  for (var token = lane; token < params.keyValueTokenCount; token = token + 256u) {
    if (keyAllowed(token)) {
      let probability = exp(attentionScore(qHead, kvHead, token) - maxScore);
      localSum = localSum + probability;
      let valueIndex = (dim * params.keyValueHeadCount + kvHead) * params.contextLength + token;
      localWeighted = localWeighted + probability * valueValues[valueIndex];
    }
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
  workgroupBarrier();
  reduceValues[lane] = localWeighted;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      reduceValues[lane] = reduceValues[lane] + reduceValues[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    outputValues[qHead * params.valueSize + dim] = reduceValues[0] / sum;
  }
}
`;

const MTP_EMBEDDING_DOT_WGSL = `
struct Params {
  candidateCount: u32,
  embeddingLength: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> rowValues: array<f32>;
@group(0) @binding(1) var<storage, read> hiddenValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;

var<workgroup> dotValues: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let candidate = group.x;
  let lane = local.x;
  if (candidate >= params.candidateCount) {
    return;
  }
  var sum = 0.0;
  let rowOffset = candidate * params.embeddingLength;
  for (var index = lane; index < params.embeddingLength; index = index + 256u) {
    sum = sum + rowValues[rowOffset + index] * hiddenValues[index];
  }
  dotValues[lane] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      dotValues[lane] = dotValues[lane] + dotValues[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    outputValues[candidate] = dotValues[0];
  }
}
`;
