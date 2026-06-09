import { GPU_COPY_DST, GPU_SHADER_STAGE_COMPUTE, GPU_UNIFORM } from "./gpu-constants";
import { bindBuffer, storageEntry } from "./gpu-bindings";
import type { WebGpuBufferLike, WebGpuDeviceLike, WebGpuQuantizedMatMulType, WebGpuQuantizedWeightHandleInternal } from "./gpu-types";
import {
  TOKEN_SLICE_WGSL,
  TOKEN_WRITE_WGSL,
  F32_GATHER_ROWS_SCALE_WGSL,
  Q8_0_GATHER_ROWS_SCALE_WGSL,
  Q4_0_GATHER_ROWS_SCALE_WGSL,
  Q4_K_GATHER_ROWS_SCALE_WGSL,
  Q5_K_GATHER_ROWS_SCALE_WGSL,
  Q6_K_GATHER_ROWS_SCALE_WGSL,
  PREPARE_PER_LAYER_INPUTS_WGSL,
  BATCHED_RMS_NORM_Q8_0_QUANTIZE_WGSL,
  BATCHED_RMS_NORM_Q8_K_QUANTIZE_WGSL,
  BATCHED_FULL_QUERY_NORM_ROPE_WGSL,
  BATCHED_FULL_KV_UPDATE_WGSL,
  BATCHED_FULL_ATTENTION_MATERIALIZED_APPLY_WGSL,
  BATCHED_FULL_ATTENTION_MATERIALIZED_SCORE_WGSL,
  BATCHED_FULL_ATTENTION_ROLLING_TILE_APPLY_WGSL,
  BATCHED_FULL_ATTENTION_ROLLING_TILE_FINAL_WGSL,
  BATCHED_FULL_ATTENTION_ROLLING_TILE_INIT_WGSL,
  BATCHED_FULL_ATTENTION_ROLLING_TILE_MERGE_WGSL,
  BATCHED_FULL_ATTENTION_ROLLING_TILE_PROBABILITY_WGSL,
  BATCHED_RMS_NORM_RESIDUAL_ADD_WGSL,
  BATCHED_RMS_NORM_RESIDUAL_ADD_SCALE_WGSL,
  BATCHED_GEGLU_SLICE_WGSL,
  RMS_NORM_WGSL,
  RMS_NORM_Q8_K_QUANTIZE_WGSL,
  RESIDUAL_ADD_WGSL,
  RESIDUAL_ADD_SCALE_WGSL,
  RMS_NORM_RESIDUAL_ADD_WGSL,
  RMS_NORM_RESIDUAL_ADD_SCALE_WGSL,
  HEAD_RMS_NORM_WGSL,
  HEAD_RMS_NORM_NO_WEIGHT_WGSL,
  ROPE_WGSL,
  KEY_CACHE_ROPE_WGSL,
  VALUE_CACHE_WRITE_WGSL,
  FULL_QUERY_NORM_ROPE_WGSL,
  FULL_KV_UPDATE_WGSL,
  TOPK_CHUNK_CANDIDATES_WGSL,
  TOPK_MERGE_CANDIDATES_WGSL,
  Q8_0_MATMUL_WGSL,
  Q4_0_DUAL_MATMUL_WGSL,
  Q4_0_MATMUL_WGSL,
  SWIGLU_WGSL,
  GEGLU_WGSL,
  GEGLU_SLICE_WGSL,
  GELU_WGSL,
  ELEMENTWISE_MUL_WGSL,
  SIGMOID_MUL_WGSL,
  SCALE_WGSL,
  F16_CAST_WGSL,
  F32_MATMUL_WGSL,
  TOP1_WGSL,
  TOP1_CHUNK_WGSL,
  SELECT_TOP1_CANDIDATE_WGSL,
  Q8_K_QUANTIZE_WGSL,
  Q8_0_QUANTIZE_WGSL,
  FULL_ATTENTION_SCORE_WGSL,
  FULL_ATTENTION_APPLY_WGSL,
  Q4_K_DUAL_MATMUL_WGSL,
  Q4_K_MATMUL_WGSL,
  Q5_K_MATMUL_WGSL,
  Q6_K_MATMUL_WGSL,
} from "./shaders";

function uniformBufferSize(byteLength: number): number {
  // Deno/wgpu validation reported 16-byte bindings as too small for some WGSL uniform structs expecting 32 bytes.
  return Math.max(32, Math.ceil(byteLength / 16) * 16);
}

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
    size: uniformBufferSize(params.byteLength),
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
    size: uniformBufferSize(params.byteLength),
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

export function createDualQ4_0MatMulBindResources(
  leftHandle: WebGpuQuantizedWeightHandleInternal,
  rightHandle: WebGpuQuantizedWeightHandleInternal,
  inputScaleBuffer: WebGpuBufferLike,
  inputQsBuffer: WebGpuBufferLike,
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
    size: uniformBufferSize(params.byteLength),
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  leftHandle.device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = leftHandle.device.createBindGroupLayout({
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
  const pipeline = leftHandle.device.createComputePipeline({
    layout: leftHandle.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: leftHandle.device.createShaderModule({ code: Q4_0_DUAL_MATMUL_WGSL }), entryPoint: "main" },
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
        bindBuffer(4, paramsBuffer),
        bindBuffer(5, leftOutputBuffer),
        bindBuffer(6, rightOutputBuffer),
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
    size: uniformBufferSize(params.byteLength),
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

export function createGegluSliceResources(
  device: WebGpuDeviceLike,
  gateOutputBuffer: WebGpuBufferLike,
  rightBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  length: number,
  rightOffset: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([length, rightOffset, 0, 0]);
  const paramsBuffer = device.createBuffer({
    size: uniformBufferSize(params.byteLength),
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
      module: device.createShaderModule({ code: GEGLU_SLICE_WGSL }),
      entryPoint: "main",
    },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, gateOutputBuffer),
        bindBuffer(1, rightBuffer),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, outputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
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
    size: uniformBufferSize(params.byteLength),
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
    size: uniformBufferSize(params.byteLength),
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
    size: uniformBufferSize(params.byteLength),
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
    size: uniformBufferSize(params.byteLength),
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
    size: uniformBufferSize(paramsU32.byteLength),
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

export function createBatchedRmsNormQ8KQuantizeResources(
  device: WebGpuDeviceLike,
  inputBuffer: WebGpuBufferLike,
  weightBuffer: WebGpuBufferLike,
  scaleBuffer: WebGpuBufferLike,
  qsBuffer: WebGpuBufferLike,
  bsumsBuffer: WebGpuBufferLike,
  options: {
    length: number;
    tokenCount: number;
    epsilon: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.epsilon, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = options.length;
  paramsU32[2] = options.length / 256;
  paramsU32[3] = options.tokenCount;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(paramsU32.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "storage"),
      storageEntry(3, "storage"),
      storageEntry(4, "storage"),
      { binding: 5, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: BATCHED_RMS_NORM_Q8_K_QUANTIZE_WGSL }), entryPoint: "main" },
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

export function createBatchedRmsNormQ8_0QuantizeResources(
  device: WebGpuDeviceLike,
  inputBuffer: WebGpuBufferLike,
  weightBuffer: WebGpuBufferLike,
  scaleBuffer: WebGpuBufferLike,
  qsBuffer: WebGpuBufferLike,
  options: {
    length: number;
    tokenCount: number;
    epsilon: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.epsilon, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = options.length;
  paramsU32[2] = options.length / 32;
  paramsU32[3] = options.tokenCount;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(paramsU32.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "storage"),
      storageEntry(3, "storage"),
      { binding: 4, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: BATCHED_RMS_NORM_Q8_0_QUANTIZE_WGSL }), entryPoint: "main" },
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
        bindBuffer(4, paramsBuffer),
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
    size: uniformBufferSize(params.byteLength),
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
    size: uniformBufferSize(params.byteLength),
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
    size: uniformBufferSize(params.byteLength),
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
    keyValueStart?: number;
    scale: number;
    tokenPosition: number;
    slidingWindow?: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const keyValueStart = options.keyValueStart ?? attentionKeyValueStart(options.tokenPosition, options.slidingWindow);
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
    size: uniformBufferSize(paramsU32.byteLength),
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
    size: uniformBufferSize(paramsU32.byteLength),
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

export function createBatchedFullQueryResources(
  device: WebGpuDeviceLike,
  qProjectionBuffer: WebGpuBufferLike,
  qNormBuffer: WebGpuBufferLike,
  freqFactorsBuffer: WebGpuBufferLike,
  positionsBuffer: WebGpuBufferLike,
  queryBuffer: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    epsilon: number;
    freqBase: number;
    tokenCount: number;
    hasFreqFactors: boolean;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.epsilon, options.freqBase, 0, 0, 0, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[2] = options.hasFreqFactors ? 1 : 0;
  params[3] = options.headCount;
  params[4] = options.headSize;
  params[5] = options.ropeDims;
  params[6] = options.tokenCount;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      storageEntry(3, "read-only-storage"),
      { binding: 4, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(5, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: BATCHED_FULL_QUERY_NORM_ROPE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, qProjectionBuffer),
        bindBuffer(1, qNormBuffer),
        bindBuffer(2, freqFactorsBuffer),
        bindBuffer(3, positionsBuffer),
        bindBuffer(4, paramsBuffer),
        bindBuffer(5, queryBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createBatchedFullKvUpdateResources(
  device: WebGpuDeviceLike,
  kProjectionBuffer: WebGpuBufferLike,
  vProjectionBuffer: WebGpuBufferLike,
  kNormBuffer: WebGpuBufferLike,
  freqFactorsBuffer: WebGpuBufferLike,
  positionsBuffer: WebGpuBufferLike,
  keyCacheBuffer: WebGpuBufferLike,
  valueCacheBuffer: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    valueSize: number;
    ropeDims: number;
    epsilon: number;
    freqBase: number;
    tokenCount: number;
    contextLength: number;
    hasFreqFactors: boolean;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.epsilon, options.freqBase, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[2] = options.hasFreqFactors ? 1 : 0;
  params[3] = options.headCount;
  params[4] = options.headSize;
  params[5] = options.valueSize;
  params[6] = options.ropeDims;
  params[7] = options.tokenCount;
  params[8] = options.contextLength;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      storageEntry(3, "read-only-storage"),
      storageEntry(4, "read-only-storage"),
      { binding: 5, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(6, "storage"),
      storageEntry(7, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: BATCHED_FULL_KV_UPDATE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, kProjectionBuffer),
        bindBuffer(1, vProjectionBuffer),
        bindBuffer(2, kNormBuffer),
        bindBuffer(3, freqFactorsBuffer),
        bindBuffer(4, positionsBuffer),
        bindBuffer(5, paramsBuffer),
        bindBuffer(6, keyCacheBuffer),
        bindBuffer(7, valueCacheBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createBatchedFullAttentionRollingTileInitResources(
  device: WebGpuDeviceLike,
  outputBuffer: WebGpuBufferLike,
  options: {
    valueSize: number;
    queryHeadCount: number;
    tokenCount: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([
    options.valueSize,
    options.queryHeadCount,
    options.tokenCount,
    0,
  ]);
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(1, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: BATCHED_FULL_ATTENTION_ROLLING_TILE_INIT_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, paramsBuffer),
        bindBuffer(1, outputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createBatchedFullAttentionRollingTileProbabilityResources(
  device: WebGpuDeviceLike,
  queryBuffer: WebGpuBufferLike,
  keyBuffer: WebGpuBufferLike,
  positionsBuffer: WebGpuBufferLike,
  probabilityTileBuffer: WebGpuBufferLike,
  tileMaxBuffer: WebGpuBufferLike,
  tileSumBuffer: WebGpuBufferLike,
  options: {
    headSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    keyValueStart?: number;
    slidingWindow?: number;
    tokenCount: number;
    scale: number;
    causal: boolean;
    tileSize: number;
    tileStart: number;
    tileLength: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.scale, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[1] = options.headSize;
  params[2] = options.queryHeadCount;
  params[3] = options.keyValueHeadCount;
  params[4] = options.keyValueTokenCount;
  params[5] = options.contextLength;
  params[6] = options.slidingWindow ?? 0;
  params[7] = options.slidingWindow === undefined ? 0 : 1;
  params[8] = options.tokenCount;
  params[9] = options.causal ? 1 : 0;
  params[10] = options.tileSize;
  params[11] = options.tileStart;
  params[12] = options.tileLength;
  params[13] = options.keyValueStart ?? 0;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      { binding: 3, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(4, "storage"),
      storageEntry(5, "storage"),
      storageEntry(6, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: BATCHED_FULL_ATTENTION_ROLLING_TILE_PROBABILITY_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, queryBuffer),
        bindBuffer(1, keyBuffer),
        bindBuffer(2, positionsBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, probabilityTileBuffer),
        bindBuffer(5, tileMaxBuffer),
        bindBuffer(6, tileSumBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createBatchedFullAttentionRollingTileMergeResources(
  device: WebGpuDeviceLike,
  rowMaxBuffer: WebGpuBufferLike,
  rowSumBuffer: WebGpuBufferLike,
  tileMaxBuffer: WebGpuBufferLike,
  tileSumBuffer: WebGpuBufferLike,
  oldScaleBuffer: WebGpuBufferLike,
  tileScaleBuffer: WebGpuBufferLike,
  options: {
    queryHeadCount: number;
    tokenCount: number;
    firstTile: boolean;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([
    options.queryHeadCount,
    options.tokenCount,
    options.firstTile ? 1 : 0,
    0,
  ]);
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(1, "storage"),
      storageEntry(2, "storage"),
      storageEntry(3, "read-only-storage"),
      storageEntry(4, "read-only-storage"),
      storageEntry(5, "storage"),
      storageEntry(6, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: BATCHED_FULL_ATTENTION_ROLLING_TILE_MERGE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, paramsBuffer),
        bindBuffer(1, rowMaxBuffer),
        bindBuffer(2, rowSumBuffer),
        bindBuffer(3, tileMaxBuffer),
        bindBuffer(4, tileSumBuffer),
        bindBuffer(5, oldScaleBuffer),
        bindBuffer(6, tileScaleBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createBatchedFullAttentionRollingTileApplyResources(
  device: WebGpuDeviceLike,
  valueBuffer: WebGpuBufferLike,
  probabilityTileBuffer: WebGpuBufferLike,
  oldScaleBuffer: WebGpuBufferLike,
  tileScaleBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  options: {
    valueSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    contextLength: number;
    keyValueStart?: number;
    tileSize: number;
    tileStart: number;
    tileLength: number;
    tokenCount: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([
    options.valueSize,
    options.queryHeadCount,
    options.keyValueHeadCount,
    options.contextLength,
    options.tileSize,
    options.tileStart,
    options.tileLength,
    options.tokenCount,
    options.keyValueStart ?? 0,
  ]);
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      storageEntry(3, "read-only-storage"),
      { binding: 4, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(5, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: BATCHED_FULL_ATTENTION_ROLLING_TILE_APPLY_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, valueBuffer),
        bindBuffer(1, probabilityTileBuffer),
        bindBuffer(2, oldScaleBuffer),
        bindBuffer(3, tileScaleBuffer),
        bindBuffer(4, paramsBuffer),
        bindBuffer(5, outputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createBatchedFullAttentionRollingTileFinalResources(
  device: WebGpuDeviceLike,
  rowSumBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  options: {
    valueSize: number;
    queryHeadCount: number;
    tokenCount: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([
    options.valueSize,
    options.queryHeadCount,
    options.tokenCount,
    0,
  ]);
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: BATCHED_FULL_ATTENTION_ROLLING_TILE_FINAL_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, paramsBuffer),
        bindBuffer(1, rowSumBuffer),
        bindBuffer(2, outputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createBatchedFullAttentionMaterializedScoreResources(
  device: WebGpuDeviceLike,
  queryBuffer: WebGpuBufferLike,
  keyBuffer: WebGpuBufferLike,
  positionsBuffer: WebGpuBufferLike,
  probabilitiesBuffer: WebGpuBufferLike,
  options: {
    headSize: number;
    valueSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    probabilityTokenCapacity: number;
    slidingWindow?: number;
    tokenCount: number;
    scale: number;
    causal: boolean;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.scale, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[1] = options.headSize;
  params[2] = options.queryHeadCount;
  params[3] = options.keyValueHeadCount;
  params[4] = options.keyValueTokenCount;
  params[5] = options.contextLength;
  params[6] = options.probabilityTokenCapacity;
  params[7] = options.slidingWindow ?? 0;
  params[8] = options.slidingWindow === undefined ? 0 : 1;
  params[9] = options.tokenCount;
  params[10] = options.causal ? 1 : 0;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    compute: { module: device.createShaderModule({ code: BATCHED_FULL_ATTENTION_MATERIALIZED_SCORE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, queryBuffer),
        bindBuffer(1, keyBuffer),
        bindBuffer(2, positionsBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, probabilitiesBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createBatchedFullAttentionMaterializedApplyResources(
  device: WebGpuDeviceLike,
  valueBuffer: WebGpuBufferLike,
  probabilitiesBuffer: WebGpuBufferLike,
  positionsBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  options: {
    valueSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    probabilityTokenCapacity: number;
    slidingWindow?: number;
    tokenCount: number;
    scale: number;
    causal: boolean;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.scale, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[1] = options.valueSize;
  params[2] = options.queryHeadCount;
  params[3] = options.keyValueHeadCount;
  params[4] = options.keyValueTokenCount;
  params[5] = options.contextLength;
  params[6] = options.probabilityTokenCapacity;
  params[7] = options.slidingWindow ?? 0;
  params[8] = options.slidingWindow === undefined ? 0 : 1;
  params[9] = options.tokenCount;
  params[10] = options.causal ? 1 : 0;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    compute: { module: device.createShaderModule({ code: BATCHED_FULL_ATTENTION_MATERIALIZED_APPLY_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, valueBuffer),
        bindBuffer(1, probabilitiesBuffer),
        bindBuffer(2, positionsBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, outputBuffer),
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
    size: uniformBufferSize(params.byteLength),
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
      module: handle.device.createShaderModule({ code: handle.type === "Q4_0" ? Q4_0_MATMUL_WGSL : Q8_0_MATMUL_WGSL }),
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

export function createBatchedGegluSliceResources(
  device: WebGpuDeviceLike,
  gateBuffer: WebGpuBufferLike,
  rightBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  options: {
    length: number;
    tokenCount: number;
    rightOffset: number;
    rightStride: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([options.length, options.tokenCount, options.rightOffset, options.rightStride]);
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    compute: { module: device.createShaderModule({ code: BATCHED_GEGLU_SLICE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, gateBuffer),
        bindBuffer(1, rightBuffer),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, outputBuffer),
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
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(paramsU32.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(paramsU32.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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

export function createRmsNormResidualAddScaleResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  residual: WebGpuBufferLike,
  scale: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
  epsilon: number,
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([epsilon, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = length;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(paramsU32.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      storageEntry(3, "read-only-storage"),
      { binding: 4, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(5, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: RMS_NORM_RESIDUAL_ADD_SCALE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, input),
        bindBuffer(1, weight),
        bindBuffer(2, residual),
        bindBuffer(3, scale),
        bindBuffer(4, paramsBuffer),
        bindBuffer(5, output),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createBatchedRmsNormResidualAddResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  residual: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    length: number;
    tokenCount: number;
    epsilon: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.epsilon, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[1] = options.length;
  params[2] = options.tokenCount;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    compute: { module: device.createShaderModule({ code: BATCHED_RMS_NORM_RESIDUAL_ADD_WGSL }), entryPoint: "main" },
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

export function createBatchedRmsNormResidualAddScaleResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  residual: WebGpuBufferLike,
  scale: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    length: number;
    tokenCount: number;
    epsilon: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.epsilon, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[1] = options.length;
  params[2] = options.tokenCount;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      storageEntry(3, "read-only-storage"),
      { binding: 4, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(5, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: BATCHED_RMS_NORM_RESIDUAL_ADD_SCALE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, input),
        bindBuffer(1, weight),
        bindBuffer(2, residual),
        bindBuffer(3, scale),
        bindBuffer(4, paramsBuffer),
        bindBuffer(5, output),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createHeadRmsNormResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    epsilon: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.epsilon, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = options.headCount;
  paramsU32[2] = options.headSize;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(paramsU32.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    compute: { module: device.createShaderModule({ code: HEAD_RMS_NORM_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, input),
        bindBuffer(1, weight),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, output),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createHeadRmsNormNoWeightResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    epsilon: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.epsilon, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = options.headCount;
  paramsU32[2] = options.headSize;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(paramsU32.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsU32);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      { binding: 1, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(2, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: device.createShaderModule({ code: HEAD_RMS_NORM_NO_WEIGHT_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, input),
        bindBuffer(1, paramsBuffer),
        bindBuffer(2, output),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createRopeResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  freqFactors: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    freqBase: number;
    position: number;
    hasFreqFactors: boolean;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.freqBase, options.position, 0, 0, 0, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[2] = options.hasFreqFactors ? 1 : 0;
  params[3] = options.headCount;
  params[4] = options.headSize;
  params[5] = options.ropeDims;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    compute: { module: device.createShaderModule({ code: ROPE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, input),
        bindBuffer(1, freqFactors),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, output),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createKeyCacheRopeResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  freqFactors: WebGpuBufferLike,
  keyCache: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    freqBase: number;
    position: number;
    tokenPosition: number;
    hasFreqFactors: boolean;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.freqBase, options.position, 0, 0, 0, 0, 0, 0]);
  const params = new Uint32Array(paramsF32.buffer);
  params[2] = options.hasFreqFactors ? 1 : 0;
  params[3] = options.headCount;
  params[4] = options.headSize;
  params[5] = options.ropeDims;
  params[6] = options.tokenPosition;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    compute: { module: device.createShaderModule({ code: KEY_CACHE_ROPE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, input),
        bindBuffer(1, freqFactors),
        bindBuffer(2, paramsBuffer),
        bindBuffer(3, keyCache),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createValueCacheWriteResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  valueCache: WebGpuBufferLike,
  options: {
    headCount: number;
    valueSize: number;
    tokenPosition: number;
    contextLength: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([options.headCount, options.valueSize, options.tokenPosition, options.contextLength]);
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    compute: { module: device.createShaderModule({ code: VALUE_CACHE_WRITE_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, input),
        bindBuffer(1, paramsBuffer),
        bindBuffer(2, valueCache),
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
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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

export function createTopKChunkCandidatesResources(
  device: WebGpuDeviceLike,
  logits: WebGpuBufferLike,
  candidates: WebGpuBufferLike,
  options: {
    rowCount: number;
    rowOffset: number;
    topK: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([options.rowCount, options.rowOffset, options.topK, 0]);
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    compute: { module: device.createShaderModule({ code: TOPK_CHUNK_CANDIDATES_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [bindBuffer(0, logits), bindBuffer(1, paramsBuffer), bindBuffer(2, candidates)],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

export function createTopKMergeCandidatesResources(
  device: WebGpuDeviceLike,
  candidates: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    candidateCount: number;
    topK: number;
    candidateOffset: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([options.candidateCount, options.topK, options.candidateOffset, 0]);
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    compute: { module: device.createShaderModule({ code: TOPK_MERGE_CANDIDATES_WGSL }), entryPoint: "main" },
  });
  return {
    pipeline,
    bindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [bindBuffer(0, candidates), bindBuffer(1, paramsBuffer), bindBuffer(2, output)],
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
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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

export function createTokenWriteResources(
  device: WebGpuDeviceLike,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowSize: number;
    rowIndex: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([options.rowSize, options.rowIndex, 0, 0]);
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(params.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    compute: { module: device.createShaderModule({ code: TOKEN_WRITE_WGSL }), entryPoint: "main" },
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
    outputTokenOffset?: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([0, 0, options.scale, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[0] = options.rowSize;
  paramsU32[1] = options.tokenCount;
  paramsU32[3] = options.outputTokenOffset ?? 0;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(paramsU32.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    outputTokenOffset?: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([0, 0, 0, 0, options.scale, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[0] = options.rowSize;
  paramsU32[1] = options.tokenCount;
  paramsU32[2] = options.blockCount;
  paramsU32[3] = options.rowByteLength;
  paramsU32[5] = options.outputTokenOffset ?? 0;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(paramsU32.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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

export function createQ4_0GatherRowsScaleResources(
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
    outputTokenOffset?: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([0, 0, 0, 0, options.scale, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[0] = options.rowSize;
  paramsU32[1] = options.tokenCount;
  paramsU32[2] = options.blockCount;
  paramsU32[3] = options.rowByteLength;
  paramsU32[5] = options.outputTokenOffset ?? 0;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(paramsU32.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
    compute: { module: device.createShaderModule({ code: Q4_0_GATHER_ROWS_SCALE_WGSL }), entryPoint: "main" },
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
    outputTokenOffset?: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  if (type === "Q8_0") {
    return createQ8_0GatherRowsScaleResources(device, weight, tokenIds, output, options);
  }
  if (type === "Q4_0") {
    return createQ4_0GatherRowsScaleResources(device, weight, tokenIds, output, options);
  }
  const paramsF32 = new Float32Array([0, 0, 0, 0, options.scale, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[0] = options.rowSize;
  paramsU32[1] = options.tokenCount;
  paramsU32[2] = options.blockCount;
  paramsU32[3] = options.rowByteLength;
  paramsU32[5] = options.outputTokenOffset ?? 0;
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(paramsU32.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
  const paramsBuffer = device.createBuffer({ size: uniformBufferSize(paramsU32.byteLength), usage: GPU_UNIFORM | GPU_COPY_DST });
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
