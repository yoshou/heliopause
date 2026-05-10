import { GPU_COPY_DST, GPU_SHADER_STAGE_COMPUTE, GPU_UNIFORM } from "./gpu-constants";
import { bindBuffer, storageEntry } from "./gpu-bindings";
import type { WebGpuBufferLike, WebGpuDeviceLike, WebGpuQuantizedMatMulType, WebGpuQuantizedWeightHandleInternal } from "./gpu-types";
import {
  TOKEN_SLICE_WGSL,
  F32_GATHER_ROWS_SCALE_WGSL,
  Q8_0_GATHER_ROWS_SCALE_WGSL,
  Q4_K_GATHER_ROWS_SCALE_WGSL,
  Q5_K_GATHER_ROWS_SCALE_WGSL,
  Q6_K_GATHER_ROWS_SCALE_WGSL,
  PREPARE_PER_LAYER_INPUTS_WGSL,
  RMS_NORM_WGSL,
  RESIDUAL_ADD_WGSL,
  RESIDUAL_ADD_SCALE_WGSL,
  RMS_NORM_RESIDUAL_ADD_WGSL,
  FULL_QUERY_NORM_ROPE_WGSL,
  FULL_KV_UPDATE_WGSL,
  TOPK_WGSL,
  Q8_0_MATMUL_WGSL,
  SWIGLU_WGSL,
  GEGLU_WGSL,
  GELU_WGSL,
  ELEMENTWISE_MUL_WGSL,
  SIGMOID_MUL_WGSL,
  SCALE_WGSL,
  F16_CAST_WGSL,
  F32_MATMUL_WGSL,
  DELTA_GATE_WGSL,
  TOP1_WGSL,
  TOP1_CHUNK_WGSL,
  SELECT_TOP1_CANDIDATE_WGSL,
  Q8_K_QUANTIZE_WGSL,
  RMS_NORM_Q8_K_QUANTIZE_WGSL,
  Q8_0_QUANTIZE_WGSL,
  SSM_NORM_GATE_WGSL,
  QKV_CONV_SPLIT_WGSL,
  FULL_ATTENTION_SCORE_WGSL,
  FULL_ATTENTION_APPLY_WGSL,
  Q4_K_DUAL_MATMUL_WGSL,
  Q4_K_MATMUL_WGSL,
  Q5_K_MATMUL_WGSL,
  Q6_K_MATMUL_WGSL,
} from "./shaders";

export function createKMatMulBindResources(
  handle: WebGpuQuantizedWeightHandleInternal,
  inputScaleBuffer: WebGpuBufferLike,
  inputQsBuffer: WebGpuBufferLike,
  inputBsumsBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  columnCount: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const usesBsums = handle.type !== "Q6_K";
  const params = new Uint32Array([
    handle.inputSize,
    handle.rowCount,
    columnCount,
    handle.blockCount,
    handle.rowByteLength,
    0,
    0,
    0,
  ]);
  const paramsBuffer = handle.device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  handle.device.queue.writeBuffer(paramsBuffer, 0, params);

  const shaderModule = handle.device.createShaderModule({
    code: handle.type === "Q6_K"
      ? Q6_K_MATMUL_WGSL
      : handle.type === "Q5_K"
        ? Q5_K_MATMUL_WGSL
        : Q4_K_MATMUL_WGSL,
  });
  const bindGroupLayout = handle.device.createBindGroupLayout({
    entries: usesBsums
      ? [
          storageEntry(0, "read-only-storage"),
          storageEntry(1, "read-only-storage"),
          storageEntry(2, "read-only-storage"),
          storageEntry(3, "read-only-storage"),
          {
            binding: 4,
            visibility: GPU_SHADER_STAGE_COMPUTE,
            buffer: { type: "uniform" },
          },
          storageEntry(5, "storage"),
        ]
      : [
          storageEntry(0, "read-only-storage"),
          storageEntry(1, "read-only-storage"),
          storageEntry(2, "read-only-storage"),
          {
            binding: 3,
            visibility: GPU_SHADER_STAGE_COMPUTE,
            buffer: { type: "uniform" },
          },
          storageEntry(4, "storage"),
        ],
  });
  const pipeline = handle.device.createComputePipeline({
    layout: handle.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });
  const entries = usesBsums
    ? [
        bindBuffer(0, handle.weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, inputBsumsBuffer),
        bindBuffer(4, paramsBuffer),
        bindBuffer(5, outputBuffer),
      ]
    : [
        bindBuffer(0, handle.weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, outputBuffer),
      ];
  return {
    pipeline,
    bindGroup: handle.device.createBindGroup({
      layout: bindGroupLayout,
      entries,
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createDualQ4KMatMulBindResources(
  leftHandle: WebGpuQuantizedWeightHandleInternal,
  rightHandle: WebGpuQuantizedWeightHandleInternal,
  inputScaleBuffer: WebGpuBufferLike,
  inputQsBuffer: WebGpuBufferLike,
  inputBsumsBuffer: WebGpuBufferLike,
  leftOutputBuffer: WebGpuBufferLike,
  rightOutputBuffer: WebGpuBufferLike,
  columnCount: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([
    leftHandle.inputSize,
    leftHandle.rowCount,
    columnCount,
    leftHandle.blockCount,
    leftHandle.rowByteLength,
    0,
    0,
    0,
  ]);
  const paramsBuffer = leftHandle.device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  leftHandle.device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = leftHandle.device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      storageEntry(3, "read-only-storage"),
      storageEntry(4, "read-only-storage"),
      {
        binding: 5,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(6, "storage"),
      storageEntry(7, "storage"),
    ],
  });
  const pipeline = leftHandle.device.createComputePipeline({
    layout: leftHandle.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: leftHandle.device.createShaderModule({ code: Q4_K_DUAL_MATMUL_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: leftHandle.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, leftHandle.weightBuffer),
        bindBuffer(1, rightHandle.weightBuffer),
        bindBuffer(2, inputScaleBuffer),
        bindBuffer(3, inputQsBuffer),
        bindBuffer(4, inputBsumsBuffer),
        bindBuffer(5, paramsBuffer),
        bindBuffer(6, leftOutputBuffer),
        bindBuffer(7, rightOutputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createSwiGluResources(
  device: WebGpuDeviceLike,
  gateOutputBuffer: WebGpuBufferLike,
  upOutputBuffer: WebGpuBufferLike,
  swigluBuffer: WebGpuBufferLike,
  length: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([length, 0, 0, 0]);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      {
        binding: 2,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(3, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: SWIGLU_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, gateOutputBuffer),
        bindBuffer(1, upOutputBuffer),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, swigluBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createGegluResources(
  device: WebGpuDeviceLike,
  gateOutputBuffer: WebGpuBufferLike,
  upOutputBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  length: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  return createBinaryElementwiseResources(device, gateOutputBuffer, upOutputBuffer, outputBuffer, length, GEGLU_WGSL);
}

export function createElementwiseMulResources(
  device: WebGpuDeviceLike,
  leftBuffer: WebGpuBufferLike,
  rightBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  length: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  return createBinaryElementwiseResources(device, leftBuffer, rightBuffer, outputBuffer, length, ELEMENTWISE_MUL_WGSL);
}

export function createSigmoidMulResources(
  device: WebGpuDeviceLike,
  leftBuffer: WebGpuBufferLike,
  gateBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  length: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  return createBinaryElementwiseResources(device, leftBuffer, gateBuffer, outputBuffer, length, SIGMOID_MUL_WGSL);
}

export function createGeluResources(
  device: WebGpuDeviceLike,
  inputBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  length: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  return createUnaryElementwiseResources(device, inputBuffer, outputBuffer, length, GELU_WGSL);
}

export function createF16CastResources(
  device: WebGpuDeviceLike,
  inputBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  length: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  return createUnaryElementwiseResources(device, inputBuffer, outputBuffer, length, F16_CAST_WGSL);
}

export function createScaleResources(
  device: WebGpuDeviceLike,
  inputBuffer: WebGpuBufferLike,
  scaleBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  length: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  return createBinaryElementwiseResources(device, inputBuffer, scaleBuffer, outputBuffer, length, SCALE_WGSL);
}

function createBinaryElementwiseResources(
  device: WebGpuDeviceLike,
  leftBuffer: WebGpuBufferLike,
  rightBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  length: number,
  shaderCode: string,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([length, 0, 0, 0]);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      {
        binding: 2,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(3, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: shaderCode }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, leftBuffer),
        bindBuffer(1, rightBuffer),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, outputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

function createUnaryElementwiseResources(
  device: WebGpuDeviceLike,
  inputBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  length: number,
  shaderCode: string,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([length, 0, 0, 0]);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      {
        binding: 1,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(2, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: shaderCode }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, inputBuffer),
        bindBuffer(1, paramsBuffer),
        bindBuffer(2, outputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createF32MatMulResources(
  device: WebGpuDeviceLike,
  weightBuffer: WebGpuBufferLike,
  inputBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([inputSize, rowCount, columnCount, 0]);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      {
        binding: 2,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(3, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: F32_MATMUL_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, weightBuffer),
        bindBuffer(1, inputBuffer),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, outputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createQ8KQuantizeResources(
  device: WebGpuDeviceLike,
  inputBuffer: WebGpuBufferLike,
  scaleBuffer: WebGpuBufferLike,
  qsBuffer: WebGpuBufferLike,
  bsumsBuffer: WebGpuBufferLike,
  inputSize: number,
  columnCount: number,
  blockCount: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([inputSize, columnCount, blockCount, 0]);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "storage"),
      storageEntry(2, "storage"),
      storageEntry(3, "storage"),
      {
        binding: 4,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: Q8_K_QUANTIZE_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, inputBuffer),
        bindBuffer(1, scaleBuffer),
        bindBuffer(2, qsBuffer),
        bindBuffer(3, bsumsBuffer),
        bindBuffer(4, paramsBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createRmsNormQ8KQuantizeResources(
  device: WebGpuDeviceLike,
  inputBuffer: WebGpuBufferLike,
  weightBuffer: WebGpuBufferLike,
  scaleBuffer: WebGpuBufferLike,
  qsBuffer: WebGpuBufferLike,
  bsumsBuffer: WebGpuBufferLike,
  length: number,
  epsilon: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([epsilon, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = length;
  paramsU32[2] = length / 256;
  const paramsBuffer = device.createBuffer({
    size: paramsU32.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "storage"),
      storageEntry(3, "storage"),
      storageEntry(4, "storage"),
      {
        binding: 5,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: RMS_NORM_Q8_K_QUANTIZE_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, inputBuffer),
        bindBuffer(1, weightBuffer),
        bindBuffer(2, scaleBuffer),
        bindBuffer(3, qsBuffer),
        bindBuffer(4, bsumsBuffer),
        bindBuffer(5, paramsBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createDeltaGateResources(
  device: WebGpuDeviceLike,
  alphaBuffer: WebGpuBufferLike,
  betaBuffer: WebGpuBufferLike,
  dtBiasBuffer: WebGpuBufferLike,
  ssmABuffer: WebGpuBufferLike,
  gateBuffer: WebGpuBufferLike,
  betaSigmoidBuffer: WebGpuBufferLike,
  valueHeadCount: number,
  tokenCount: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([valueHeadCount, tokenCount, 0, 0]);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      storageEntry(3, "read-only-storage"),
      {
        binding: 4,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(5, "storage"),
      storageEntry(6, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: DELTA_GATE_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, alphaBuffer),
        bindBuffer(1, betaBuffer),
        bindBuffer(2, dtBiasBuffer),
        bindBuffer(3, ssmABuffer),
        bindBuffer(4, paramsBuffer),
        bindBuffer(5, gateBuffer),
        bindBuffer(6, betaSigmoidBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createTop1Resources(
  device: WebGpuDeviceLike,
  logitsBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  rowCount: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([rowCount, 0, 0, 0]);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      {
        binding: 1,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(2, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: TOP1_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, logitsBuffer),
        bindBuffer(1, paramsBuffer),
        bindBuffer(2, outputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createSelectTop1CandidateResources(
  device: WebGpuDeviceLike,
  candidatesBuffer: WebGpuBufferLike,
  selectedTokenBuffer: WebGpuBufferLike,
  candidateCount: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([candidateCount, 0, 0, 0]);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      {
        binding: 1,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(2, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: SELECT_TOP1_CANDIDATE_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, candidatesBuffer),
        bindBuffer(1, paramsBuffer),
        bindBuffer(2, selectedTokenBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createQ8_0QuantizeResources(
  device: WebGpuDeviceLike,
  inputBuffer: WebGpuBufferLike,
  scaleBuffer: WebGpuBufferLike,
  qsBuffer: WebGpuBufferLike,
  inputSize: number,
  columnCount: number,
  blockCount: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([inputSize, columnCount, blockCount, 0]);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "storage"),
      storageEntry(2, "storage"),
      {
        binding: 3,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: Q8_0_QUANTIZE_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, inputBuffer),
        bindBuffer(1, scaleBuffer),
        bindBuffer(2, qsBuffer),
        bindBuffer(3, paramsBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createSsmNormGateResources(
  device: WebGpuDeviceLike,
  deltaBuffer: WebGpuBufferLike,
  zBuffer: WebGpuBufferLike,
  normWeightBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  rowCount: number,
  columnCount: number,
  epsilon: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([epsilon, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = rowCount;
  paramsU32[2] = columnCount;
  const paramsBuffer = device.createBuffer({
    size: paramsU32.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      {
        binding: 3,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(4, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: SSM_NORM_GATE_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, deltaBuffer),
        bindBuffer(1, zBuffer),
        bindBuffer(2, normWeightBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, outputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createQkvConvResources(
  device: WebGpuDeviceLike,
  qkvBuffer: WebGpuBufferLike,
  convStateBuffer: WebGpuBufferLike,
  convKernelBuffer: WebGpuBufferLike,
  qBuffer: WebGpuBufferLike,
  kBuffer: WebGpuBufferLike,
  vBuffer: WebGpuBufferLike,
  newStateBuffer: WebGpuBufferLike,
  options: {
    tokenCount: number;
    convDim: number;
    kernelSize: number;
    stateSize: number;
    groupCount: number;
    valueDim: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([
    options.tokenCount,
    options.convDim,
    options.kernelSize,
    options.stateSize,
    options.groupCount,
    options.valueDim,
    0,
    0,
  ]);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      {
        binding: 3,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(4, "storage"),
      storageEntry(5, "storage"),
      storageEntry(6, "storage"),
      storageEntry(7, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: QKV_CONV_SPLIT_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, qkvBuffer),
        bindBuffer(1, convStateBuffer),
        bindBuffer(2, convKernelBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, qBuffer),
        bindBuffer(5, kBuffer),
        bindBuffer(6, vBuffer),
        bindBuffer(7, newStateBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createFullAttentionScoreResources(
  device: WebGpuDeviceLike,
  queryBuffer: WebGpuBufferLike,
  keyBuffer: WebGpuBufferLike,
  probabilitiesBuffer: WebGpuBufferLike,
  options: {
    headSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    scale: number;
    tokenPosition: number;
    slidingWindow?: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const keyValueStart = attentionKeyValueStart(options.tokenPosition, options.slidingWindow);
  const paramsF32 = new Float32Array([options.scale, 0, 0, 0, 0, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = options.headSize;
  paramsU32[2] = options.queryHeadCount;
  paramsU32[3] = options.keyValueHeadCount;
  paramsU32[4] = options.keyValueTokenCount;
  paramsU32[5] = options.contextLength;
  paramsU32[6] = options.tokenPosition;
  paramsU32[7] = keyValueStart;
  const paramsBuffer = device.createBuffer({
    size: paramsU32.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      {
        binding: 2,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(3, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: FULL_ATTENTION_SCORE_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, queryBuffer),
        bindBuffer(1, keyBuffer),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, probabilitiesBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createFullAttentionApplyResources(
  device: WebGpuDeviceLike,
  valueBuffer: WebGpuBufferLike,
  probabilitiesBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  options: {
    valueSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    scale: number;
    keyValueStart?: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.scale, 0, 0, 0, 0, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = options.valueSize;
  paramsU32[2] = options.queryHeadCount;
  paramsU32[3] = options.keyValueHeadCount;
  paramsU32[4] = options.keyValueTokenCount;
  paramsU32[5] = options.contextLength;
  paramsU32[6] = options.keyValueStart ?? 0;
  const paramsBuffer = device.createBuffer({
    size: paramsU32.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      {
        binding: 2,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(3, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: FULL_ATTENTION_APPLY_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, valueBuffer),
        bindBuffer(1, probabilitiesBuffer),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, outputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

function attentionKeyValueStart(tokenPosition: number, slidingWindow: number | undefined): number {
  return slidingWindow === undefined ? 0 : Math.max(0, tokenPosition + 1 - slidingWindow);
}

export function createQ8_0MatMulBindResources(
  handle: WebGpuQuantizedWeightHandleInternal,
  inputScaleBuffer: WebGpuBufferLike,
  inputQsBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  columnCount: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([
    handle.inputSize,
    handle.rowCount,
    columnCount,
    handle.blockCount,
    handle.rowByteLength,
    0,
    0,
    0,
  ]);
  const paramsBuffer = handle.device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  handle.device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = handle.device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      {
        binding: 3,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(4, "storage"),
    ],
  });
  const pipeline = handle.device.createComputePipeline({
    layout: handle.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: handle.device.createShaderModule({ code: Q8_0_MATMUL_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: handle.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, handle.weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, outputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createRmsNormResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
  epsilon: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([epsilon, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = length;
  const paramsBuffer = device.createBuffer({ size: paramsU32.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      { binding: 2, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(3, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: RMS_NORM_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [bindBuffer(0, input), bindBuffer(1, weight), bindBuffer(2, paramsBuffer), bindBuffer(3, output)],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createResidualAddResources(
  device: WebGpuDeviceLike,
  left: WebGpuBufferLike,
  right: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([length, 0, 0, 0]);
  const paramsBuffer = device.createBuffer({ size: params.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      { binding: 2, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(3, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: RESIDUAL_ADD_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [bindBuffer(0, left), bindBuffer(1, right), bindBuffer(2, paramsBuffer), bindBuffer(3, output)],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createResidualAddScaleResources(
  device: WebGpuDeviceLike,
  left: WebGpuBufferLike,
  right: WebGpuBufferLike,
  scale: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([length, 0, 0, 0]);
  const paramsBuffer = device.createBuffer({ size: params.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      { binding: 3, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(4, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: RESIDUAL_ADD_SCALE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, left),
        bindBuffer(1, right),
        bindBuffer(2, scale),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, output),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createRmsNormResidualAddResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  residual: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
  epsilon: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([epsilon, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = length;
  const paramsBuffer = device.createBuffer({ size: paramsU32.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      { binding: 3, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(4, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: RMS_NORM_RESIDUAL_ADD_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, input),
        bindBuffer(1, weight),
        bindBuffer(2, residual),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, output),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createFullQueryResources(
  device: WebGpuDeviceLike,
  qProjection: WebGpuBufferLike,
  qNorm: WebGpuBufferLike,
  freqFactors: WebGpuBufferLike,
  query: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    epsilon: number;
    freqBase: number;
    position: number;
    hasFreqFactors: boolean;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.epsilon, options.freqBase, options.position, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[3] = options.hasFreqFactors ? 1 : 0;
  const params = new Uint32Array(8);
  params.set(paramsU32, 0);
  params[4] = options.headCount;
  params[5] = options.headSize;
  params[6] = options.ropeDims;
  const paramsBuffer = device.createBuffer({ size: params.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      { binding: 3, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(4, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: FULL_QUERY_NORM_ROPE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, qProjection),
        bindBuffer(1, qNorm),
        bindBuffer(2, freqFactors),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, query),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createFullKvUpdateResources(
  device: WebGpuDeviceLike,
  kProjection: WebGpuBufferLike,
  vProjection: WebGpuBufferLike,
  kNorm: WebGpuBufferLike,
  freqFactors: WebGpuBufferLike,
  keyCache: WebGpuBufferLike,
  valueCache: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    valueSize: number;
    ropeDims: number;
    epsilon: number;
    freqBase: number;
    position: number;
    tokenPosition: number;
    contextLength: number;
    hasFreqFactors: boolean;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.epsilon, options.freqBase, options.position, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[3] = options.hasFreqFactors ? 1 : 0;
  const params = new Uint32Array(12);
  params.set(paramsU32, 0);
  params[4] = options.headCount;
  params[5] = options.headSize;
  params[6] = options.valueSize;
  params[7] = options.ropeDims;
  params[8] = options.tokenPosition;
  params[9] = options.contextLength;
  const paramsBuffer = device.createBuffer({ size: params.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      storageEntry(3, "read-only-storage"),
      { binding: 4, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(5, "storage"),
      storageEntry(6, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: FULL_KV_UPDATE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, kProjection),
        bindBuffer(1, vProjection),
        bindBuffer(2, kNorm),
        bindBuffer(3, freqFactors),
        bindBuffer(4, paramsBuffer),
        bindBuffer(5, keyCache),
        bindBuffer(6, valueCache),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createTopKResources(
  device: WebGpuDeviceLike,
  logits: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowCount: number;
    rowOffset: number;
    topK: number;
    candidateOffset: number;
    slot: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([options.rowCount, options.rowOffset, options.slot, options.candidateOffset]);
  const paramsBuffer = device.createBuffer({ size: params.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      { binding: 1, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(2, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: TOPK_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [bindBuffer(0, logits), bindBuffer(1, paramsBuffer), bindBuffer(2, output)],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createTop1ChunkResources(
  device: WebGpuDeviceLike,
  logits: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowCount: number;
    rowOffset: number;
    candidateOffset: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([options.rowCount, options.rowOffset, options.candidateOffset, 0]);
  const paramsBuffer = device.createBuffer({ size: params.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      { binding: 1, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(2, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: TOP1_CHUNK_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [bindBuffer(0, logits), bindBuffer(1, paramsBuffer), bindBuffer(2, output)],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createTokenSliceResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowSize: number;
    rowIndex: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([options.rowSize, options.rowIndex, 0, 0]);
  const paramsBuffer = device.createBuffer({ size: params.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      { binding: 1, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(2, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: TOKEN_SLICE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [bindBuffer(0, input), bindBuffer(1, paramsBuffer), bindBuffer(2, output)],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createF32GatherRowsScaleResources(
  device: WebGpuDeviceLike,
  rows: WebGpuBufferLike,
  tokenIds: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowSize: number;
    tokenCount: number;
    scale: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([0, 0, options.scale, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[0] = options.rowSize;
  paramsU32[1] = options.tokenCount;
  const paramsBuffer = device.createBuffer({ size: paramsU32.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      { binding: 2, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(3, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: F32_GATHER_ROWS_SCALE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, rows),
        bindBuffer(1, tokenIds),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, output),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createQ8_0GatherRowsScaleResources(
  device: WebGpuDeviceLike,
  weight: WebGpuBufferLike,
  tokenIds: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowSize: number;
    tokenCount: number;
    blockCount: number;
    rowByteLength: number;
    scale: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([0, 0, 0, 0, options.scale, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[0] = options.rowSize;
  paramsU32[1] = options.tokenCount;
  paramsU32[2] = options.blockCount;
  paramsU32[3] = options.rowByteLength;
  const paramsBuffer = device.createBuffer({ size: paramsU32.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      { binding: 2, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(3, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: Q8_0_GATHER_ROWS_SCALE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, weight),
        bindBuffer(1, tokenIds),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, output),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createQuantizedGatherRowsScaleResources(
  device: WebGpuDeviceLike,
  type: WebGpuQuantizedMatMulType,
  weight: WebGpuBufferLike,
  tokenIds: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowSize: number;
    tokenCount: number;
    blockCount: number;
    rowByteLength: number;
    scale: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  if (type === "Q8_0") {
    return createQ8_0GatherRowsScaleResources(device, weight, tokenIds, output, options);
  }
  const paramsF32 = new Float32Array([0, 0, 0, 0, options.scale, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[0] = options.rowSize;
  paramsU32[1] = options.tokenCount;
  paramsU32[2] = options.blockCount;
  paramsU32[3] = options.rowByteLength;
  const paramsBuffer = device.createBuffer({ size: paramsU32.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      { binding: 2, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(3, "storage"),
    ],
  });
  const shaderCode = type === "Q6_K"
    ? Q6_K_GATHER_ROWS_SCALE_WGSL
    : type === "Q5_K"
      ? Q5_K_GATHER_ROWS_SCALE_WGSL
      : Q4_K_GATHER_ROWS_SCALE_WGSL;
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: shaderCode }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, weight),
        bindBuffer(1, tokenIds),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, output),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createPreparePerLayerInputsResources(
  device: WebGpuDeviceLike,
  tokenRows: WebGpuBufferLike,
  projected: WebGpuBufferLike,
  normWeight: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    perLayerLength: number;
    totalPerLayerLength: number;
    tokenCount: number;
    blockCount: number;
    projectionScale: number;
    epsilon: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([0, 0, 0, 0, options.projectionScale, options.epsilon, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[0] = options.perLayerLength;
  paramsU32[1] = options.totalPerLayerLength;
  paramsU32[2] = options.tokenCount;
  paramsU32[3] = options.blockCount;
  const paramsBuffer = device.createBuffer({ size: paramsU32.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      { binding: 3, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(4, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: PREPARE_PER_LAYER_INPUTS_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, tokenRows),
        bindBuffer(1, projected),
        bindBuffer(2, normWeight),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, output),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}
