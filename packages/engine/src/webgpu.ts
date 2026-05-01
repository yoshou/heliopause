import type {
  GgmlTypeName,
  GgufMetadata,
  GgufTensorInfo,
} from "./gguf";
import type {
  Qwen35LayerKind,
  Qwen35ModelManifest,
} from "./qwen35";
import { tensorByteLength, type GgufTensorReader } from "./tensor-reader";

export type WebGpuSupport =
  | {
      available: true;
      adapterName?: string;
      maxBufferSize?: number;
      maxStorageBufferBindingSize?: number;
    }
  | {
      available: false;
      reason: "navigator-missing" | "api-missing" | "adapter-missing" | "request-failed";
      error?: string;
    };

export type WebGpuSmokeTest =
  | {
      ok: true;
      durationMs: number;
      output: number[];
    }
  | {
      ok: false;
      reason: "navigator-missing" | "api-missing" | "adapter-missing" | "device-request-failed" | "compute-failed" | "mismatch";
      durationMs: number;
      error?: string;
      output?: number[];
    };

type NavigatorWithWebGpu = Navigator & {
  gpu?: {
    requestAdapter: () => Promise<WebGpuAdapterLike | null>;
  };
};

type WebGpuAdapterLike = {
  requestDevice?: (descriptor?: {
    requiredLimits?: Record<string, number>;
  }) => Promise<WebGpuDeviceLike>;
  info?: {
    description?: string;
    vendor?: string;
    architecture?: string;
    device?: string;
  };
  limits?: {
    maxBufferSize?: number;
    maxStorageBufferBindingSize?: number;
  };
};

type WebGpuDeviceLike = {
  createBuffer: (descriptor: {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }) => WebGpuBufferLike;
  createShaderModule: (descriptor: { code: string }) => unknown;
  createBindGroupLayout: (descriptor: unknown) => unknown;
  createPipelineLayout: (descriptor: unknown) => unknown;
  createComputePipeline: (descriptor: unknown) => unknown;
  createBindGroup: (descriptor: unknown) => unknown;
  createCommandEncoder: () => WebGpuCommandEncoderLike;
  queue: {
    writeBuffer: (
      buffer: WebGpuBufferLike,
      bufferOffset: number,
      data: ArrayBuffer | ArrayBufferView,
      dataOffset?: number,
      size?: number,
    ) => void;
    submit: (commandBuffers: unknown[]) => void;
    onSubmittedWorkDone?: () => Promise<void>;
  };
};

type WebGpuBufferLike = {
  getMappedRange: () => ArrayBuffer;
  unmap: () => void;
  mapAsync: (mode: number) => Promise<void>;
  destroy?: () => void;
};

type WebGpuCommandEncoderLike = {
  beginComputePass: () => WebGpuComputePassLike;
  copyBufferToBuffer: (
    source: WebGpuBufferLike,
    sourceOffset: number,
    destination: WebGpuBufferLike,
    destinationOffset: number,
    size: number,
  ) => void;
  finish: () => unknown;
};

type WebGpuComputePassLike = {
  setPipeline: (pipeline: unknown) => void;
  setBindGroup: (index: number, bindGroup: unknown) => void;
  dispatchWorkgroups: (x: number, y?: number, z?: number) => void;
  end: () => void;
};

export const QWEN35_WEBGPU_MEMORY_LIMIT_BYTES = 12 * 1024 * 1024 * 1024;

export type Qwen35WebGpuMode = "off" | "verify" | "enabled";

export type Qwen35WebGpuPlanStatus =
  | "off"
  | "unavailable"
  | "blocked"
  | "planned";

export type Qwen35WebGpuBrowserGate = "required" | "passed";

export type WebGpuQuantizedMatMulType = "Q4_K" | "Q5_K" | "Q6_K" | "Q8_0";

export type WebGpuQuantizedWeightHandle = {
  readonly type: WebGpuQuantizedMatMulType;
  readonly inputSize: number;
  readonly rowCount: number;
  readonly byteLength: number;
  destroy: () => void;
};

export type WebGpuF32TensorHandle = {
  readonly length: number;
  readonly byteLength: number;
  destroy: () => void;
};

export type WebGpuQkvConvResult = {
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  newConvState: Float32Array;
};

export type WebGpuGatedDeltaNetResult = {
  output: Float32Array;
  newState: Float32Array;
};

export type WebGpuTopToken = {
  id: number;
  value: number;
};

type WebGpuQuantizedWeightHandleInternal = WebGpuQuantizedWeightHandle & {
  readonly device: WebGpuDeviceLike;
  readonly weightBuffer: WebGpuBufferLike;
  readonly blockCount: number;
  readonly rowByteLength: number;
};

type WebGpuF32TensorHandleInternal = WebGpuF32TensorHandle & {
  readonly device: WebGpuDeviceLike;
  readonly buffer: WebGpuBufferLike;
};

export type Qwen35WebGpuLayerPlan = {
  layer: number;
  layerKind: Qwen35LayerKind;
  weightBytes: number;
  cacheBytes: number;
  totalBytes: number;
};

export type Qwen35WebGpuHybridPlan = {
  status: Qwen35WebGpuPlanStatus;
  mode: Qwen35WebGpuMode;
  memoryLimitBytes: number;
  browserGate: Qwen35WebGpuBrowserGate;
  enabled: false;
  reason?: string;
  outputBytes: number;
  fixedBytes: number;
  scratchBytes: number;
  selectedLayerCount: number;
  firstGpuLayer?: number;
  cpuPrefixLayerCount: number;
  gpuSuffixLayerCount: number;
  gpuWeightBytes: number;
  gpuCacheBytes: number;
  estimatedResidentBytes: number;
  remainingBytes: number;
  selectedLayers: Qwen35WebGpuLayerPlan[];
  copyAuditExpectations: {
    decodeTensorReads: 0;
    suffixIntermediateReadbacks: 0;
    logitsReadbacks: 0;
    expectedBoundaryUploads: number;
    expectedTokenReadbacks: number;
  };
};

export type Qwen35WebGpuCopyAuditObservation = {
  decodeTensorReads: number;
  suffixIntermediateReadbacks: number;
  logitsReadbacks: number;
  boundaryUploads: number;
  tokenReadbacks: number;
};

export type Qwen35WebGpuCopyAuditResult = {
  ok: boolean;
  errors: string[];
  expected: Qwen35WebGpuHybridPlan["copyAuditExpectations"];
  observed: Qwen35WebGpuCopyAuditObservation;
};

export type Qwen35WebGpuPlanningOptions = {
  mode?: Qwen35WebGpuMode;
  memoryLimitBytes?: number;
  contextLength?: number;
  browserGate?: Qwen35WebGpuBrowserGate;
  support?: WebGpuSupport;
};

const DEFAULT_GPU_SCRATCH_BYTES = 512 * 1024 * 1024;
const DEFAULT_GPU_FIXED_BYTES = 256 * 1024 * 1024;
const GPU_MAP_READ = 1;
const GPU_COPY_SRC = 4;
const GPU_COPY_DST = 8;
const GPU_STORAGE = 128;
const GPU_UNIFORM = 64;
const GPU_SHADER_STAGE_COMPUTE = 4;
let devicePromise: Promise<WebGpuDeviceLike | undefined> | undefined;
let adapterLimitsPromise: Promise<Pick<WebGpuAdapterLike, "limits">["limits"] | undefined> | undefined;

export async function checkWebGpuSupport(): Promise<WebGpuSupport> {
  if (typeof navigator === "undefined") {
    return {
      available: false,
      reason: "navigator-missing",
    };
  }

  const gpu = (navigator as NavigatorWithWebGpu).gpu;

  if (!gpu) {
    return {
      available: false,
      reason: "api-missing",
    };
  }

  try {
    const adapter = await gpu.requestAdapter();

    if (!adapter) {
      return {
        available: false,
        reason: "adapter-missing",
      };
    }

    return {
      available: true,
      adapterName: describeAdapter(adapter),
      maxBufferSize: adapter.limits?.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize,
    };
  } catch (error) {
    return {
      available: false,
      reason: "request-failed",
      error: error instanceof Error ? error.message : undefined,
    };
  }
}

export async function runWebGpuSmokeTest(): Promise<WebGpuSmokeTest> {
  const start = nowMs();
  const gpu = typeof navigator === "undefined"
    ? undefined
    : (navigator as NavigatorWithWebGpu).gpu;
  if (typeof navigator === "undefined") {
    return smokeFailure("navigator-missing", start);
  }
  if (!gpu) {
    return smokeFailure("api-missing", start);
  }

  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter?.requestDevice) {
      return smokeFailure("adapter-missing", start);
    }

    const device = await adapter.requestDevice();
    const input = new Float32Array([1, 2, 3, 4]);
    const expected = [3, 5, 7, 9];
    const inputBuffer = device.createBuffer({
      size: input.byteLength,
      usage: GPU_STORAGE | GPU_COPY_DST,
    });
    const outputBuffer = device.createBuffer({
      size: input.byteLength,
      usage: GPU_STORAGE | GPU_COPY_SRC,
    });
    const readbackBuffer = device.createBuffer({
      size: input.byteLength,
      usage: GPU_MAP_READ | GPU_COPY_DST,
    });

    try {
      device.queue.writeBuffer(inputBuffer, 0, input);
      const shaderModule = device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read> inputValues: array<f32>;
          @group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;

          @compute @workgroup_size(4)
          fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let index = id.x;
            if (index < 4u) {
              outputValues[index] = inputValues[index] * 2.0 + 1.0;
            }
          }
        `,
      });
      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPU_SHADER_STAGE_COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 1,
            visibility: GPU_SHADER_STAGE_COMPUTE,
            buffer: { type: "storage" },
          },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      });
      const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: "main",
        },
      });
      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
        ],
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, input.byteLength);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone?.();
      await readbackBuffer.mapAsync(GPU_MAP_READ);
      const output = Array.from(new Float32Array(readbackBuffer.getMappedRange()).slice());
      readbackBuffer.unmap();
      if (!sameNumbers(output, expected)) {
        return {
          ok: false,
          reason: "mismatch",
          durationMs: nowMs() - start,
          output,
        };
      }
      return {
        ok: true,
        durationMs: nowMs() - start,
        output,
      };
    } finally {
      inputBuffer.destroy?.();
      outputBuffer.destroy?.();
      readbackBuffer.destroy?.();
    }
  } catch (error) {
    return {
      ok: false,
      reason: error instanceof DOMException && error.name === "OperationError"
        ? "device-request-failed"
        : "compute-failed",
      durationMs: nowMs() - start,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function matMulQ8_0WebGpu(
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  if (inputSize % 32 !== 0) {
    throw new Error(`WebGPU Q8_0 matmul input size must be divisible by 32, got ${inputSize}`);
  }
  if (inputColumns.length !== inputSize * columnCount) {
    throw new Error(`WebGPU Q8_0 matmul input shape mismatch: ${inputColumns.length}`);
  }
  const blockCount = inputSize / 32;
  const rowByteLength = blockCount * 34;
  if (weightBytes.byteLength !== rowByteLength * rowCount) {
    throw new Error(`WebGPU Q8_0 matmul weight shape mismatch: ${weightBytes.byteLength}`);
  }

  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }

  const q8 = quantizeQ8_0Columns(inputColumns, inputSize, columnCount);
  const packedWeight = packBytesToU32(weightBytes);
  await assertStorageBindingFits("Q8_0 weight", packedWeight.byteLength);
  const outputLength = rowCount * columnCount;
  const params = new Uint32Array([inputSize, rowCount, columnCount, blockCount, rowByteLength, 0, 0, 0]);

  const weightBuffer = storageBuffer(device, packedWeight.byteLength, GPU_COPY_DST);
  const inputScaleBuffer = storageBuffer(device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(device, q8.qs.byteLength, GPU_COPY_DST);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  const outputBuffer = storageBuffer(device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    device.queue.writeBuffer(weightBuffer, 0, packedWeight);
    device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const shaderModule = device.createShaderModule({ code: Q8_0_MATMUL_WGSL });
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
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });
    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, outputBuffer),
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    return output;
  } finally {
    weightBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    paramsBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

export async function matMulQ4_KWebGpu(
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  if (inputSize % 256 !== 0) {
    throw new Error(`WebGPU Q4_K matmul input size must be divisible by 256, got ${inputSize}`);
  }
  if (inputColumns.length !== inputSize * columnCount) {
    throw new Error(`WebGPU Q4_K matmul input shape mismatch: ${inputColumns.length}`);
  }
  const blockCount = inputSize / 256;
  const rowByteLength = blockCount * 144;
  if (weightBytes.byteLength !== rowByteLength * rowCount) {
    throw new Error(`WebGPU Q4_K matmul weight shape mismatch: ${weightBytes.byteLength}`);
  }

  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }

  const q8 = quantizeQ8_KColumns(inputColumns, inputSize, columnCount);
  const packedWeight = packBytesToU32(weightBytes);
  await assertStorageBindingFits("Q4_K weight", packedWeight.byteLength);
  const outputLength = rowCount * columnCount;
  const params = new Uint32Array([inputSize, rowCount, columnCount, blockCount, rowByteLength, 0, 0, 0]);

  const weightBuffer = storageBuffer(device, packedWeight.byteLength, GPU_COPY_DST);
  const inputScaleBuffer = storageBuffer(device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(device, q8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(device, q8.bsums.byteLength, GPU_COPY_DST);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  const outputBuffer = storageBuffer(device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    device.queue.writeBuffer(weightBuffer, 0, packedWeight);
    device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const shaderModule = device.createShaderModule({ code: Q4_K_MATMUL_WGSL });
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
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });
    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, inputBsumsBuffer),
        bindBuffer(4, paramsBuffer),
        bindBuffer(5, outputBuffer),
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    return output;
  } finally {
    weightBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
    paramsBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

export async function matMulQ5_KWebGpu(
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  if (inputSize % 256 !== 0) {
    throw new Error(`WebGPU Q5_K matmul input size must be divisible by 256, got ${inputSize}`);
  }
  if (inputColumns.length !== inputSize * columnCount) {
    throw new Error(`WebGPU Q5_K matmul input shape mismatch: ${inputColumns.length}`);
  }
  const blockCount = inputSize / 256;
  const rowByteLength = blockCount * 176;
  if (weightBytes.byteLength !== rowByteLength * rowCount) {
    throw new Error(`WebGPU Q5_K matmul weight shape mismatch: ${weightBytes.byteLength}`);
  }

  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }

  const q8 = quantizeQ8_KColumns(inputColumns, inputSize, columnCount);
  const packedWeight = packBytesToU32(weightBytes);
  await assertStorageBindingFits("Q5_K weight", packedWeight.byteLength);
  const outputLength = rowCount * columnCount;
  const params = new Uint32Array([inputSize, rowCount, columnCount, blockCount, rowByteLength, 0, 0, 0]);

  const weightBuffer = storageBuffer(device, packedWeight.byteLength, GPU_COPY_DST);
  const inputScaleBuffer = storageBuffer(device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(device, q8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(device, q8.bsums.byteLength, GPU_COPY_DST);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  const outputBuffer = storageBuffer(device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    device.queue.writeBuffer(weightBuffer, 0, packedWeight);
    device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const shaderModule = device.createShaderModule({ code: Q5_K_MATMUL_WGSL });
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
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });
    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, inputBsumsBuffer),
        bindBuffer(4, paramsBuffer),
        bindBuffer(5, outputBuffer),
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    return output;
  } finally {
    weightBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
    paramsBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

export async function matMulQ6_KWebGpu(
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  if (inputSize % 256 !== 0) {
    throw new Error(`WebGPU Q6_K matmul input size must be divisible by 256, got ${inputSize}`);
  }
  if (inputColumns.length !== inputSize * columnCount) {
    throw new Error(`WebGPU Q6_K matmul input shape mismatch: ${inputColumns.length}`);
  }
  const blockCount = inputSize / 256;
  const rowByteLength = blockCount * 210;
  if (weightBytes.byteLength !== rowByteLength * rowCount) {
    throw new Error(`WebGPU Q6_K matmul weight shape mismatch: ${weightBytes.byteLength}`);
  }

  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }

  const q8 = quantizeQ8_KColumns(inputColumns, inputSize, columnCount);
  const packedWeight = packBytesToU32(weightBytes);
  await assertStorageBindingFits("Q6_K weight", packedWeight.byteLength);
  const outputLength = rowCount * columnCount;
  const params = new Uint32Array([inputSize, rowCount, columnCount, blockCount, rowByteLength, 0, 0, 0]);

  const weightBuffer = storageBuffer(device, packedWeight.byteLength, GPU_COPY_DST);
  const inputScaleBuffer = storageBuffer(device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(device, q8.qs.byteLength, GPU_COPY_DST);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  const outputBuffer = storageBuffer(device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    device.queue.writeBuffer(weightBuffer, 0, packedWeight);
    device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const shaderModule = device.createShaderModule({ code: Q6_K_MATMUL_WGSL });
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
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });
    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, outputBuffer),
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    return output;
  } finally {
    weightBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    paramsBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

export async function createWebGpuQuantizedWeightHandle(
  type: WebGpuQuantizedMatMulType,
  weightBytes: Uint8Array,
  inputSize: number,
  rowCount: number,
): Promise<WebGpuQuantizedWeightHandle | undefined> {
  const layout = webGpuQuantizedWeightLayout(type, inputSize);
  if (weightBytes.byteLength !== layout.rowByteLength * rowCount) {
    throw new Error(`WebGPU ${type} weight shape mismatch: ${weightBytes.byteLength}`);
  }
  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }
  const packedWeight = packBytesToU32(weightBytes);
  await assertStorageBindingFits(`${type} weight`, packedWeight.byteLength);
  const weightBuffer = storageBuffer(device, packedWeight.byteLength, GPU_COPY_DST);
  device.queue.writeBuffer(weightBuffer, 0, packedWeight);
  const handle: WebGpuQuantizedWeightHandleInternal = {
    type,
    inputSize,
    rowCount,
    byteLength: packedWeight.byteLength,
    device,
    weightBuffer,
    blockCount: layout.blockCount,
    rowByteLength: layout.rowByteLength,
    destroy: () => weightBuffer.destroy?.(),
  };
  return handle;
}

export async function createWebGpuF32TensorHandle(
  values: Float32Array,
): Promise<WebGpuF32TensorHandle | undefined> {
  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }
  await assertStorageBindingFits("F32 tensor", values.byteLength);
  const buffer = storageBuffer(device, values.byteLength, GPU_COPY_DST);
  device.queue.writeBuffer(buffer, 0, values);
  const handle: WebGpuF32TensorHandleInternal = {
    length: values.length,
    byteLength: values.byteLength,
    device,
    buffer,
    destroy: () => buffer.destroy?.(),
  };
  return handle;
}

export async function matMulWebGpuQuantizedResident(
  handle: WebGpuQuantizedWeightHandle,
  inputColumns: Float32Array,
  columnCount: number,
): Promise<Float32Array> {
  const resident = handle as WebGpuQuantizedWeightHandleInternal;
  if (inputColumns.length !== resident.inputSize * columnCount) {
    throw new Error(`WebGPU ${resident.type} resident matmul input shape mismatch: ${inputColumns.length}`);
  }

  if (resident.type === "Q8_0") {
    return matMulQ8_0Resident(resident, inputColumns, columnCount);
  }
  if (resident.type === "Q6_K") {
    return matMulKResident(resident, inputColumns, columnCount, Q6_K_MATMUL_WGSL, false);
  }
  if (resident.type === "Q5_K") {
    return matMulKResident(resident, inputColumns, columnCount, Q5_K_MATMUL_WGSL, true);
  }
  return matMulKResident(resident, inputColumns, columnCount, Q4_K_MATMUL_WGSL, true);
}

export async function matMulTop1WebGpuQuantizedResident(
  handle: WebGpuQuantizedWeightHandle,
  inputColumns: Float32Array,
): Promise<WebGpuTopToken> {
  const resident = handle as WebGpuQuantizedWeightHandleInternal;
  if (inputColumns.length !== resident.inputSize) {
    throw new Error(`WebGPU ${resident.type} top-1 input shape mismatch: ${inputColumns.length}`);
  }
  if (resident.type === "Q8_0") {
    return top1Cpu(await matMulWebGpuQuantizedResident(handle, inputColumns, 1));
  }

  const q8 = quantizeQ8_KColumns(inputColumns, resident.inputSize, 1);
  const logitsBuffer = storageBuffer(resident.device, resident.rowCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const topBuffer = storageBuffer(resident.device, 2 * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = resident.device.createBuffer({
    size: 2 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const inputScaleBuffer = storageBuffer(resident.device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(resident.device, q8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(resident.device, q8.bsums.byteLength, GPU_COPY_DST);

  try {
    resident.device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    resident.device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    resident.device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);
    const matmulResources = createKMatMulBindResources(resident, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, logitsBuffer, 1);
    const topResources = createTop1Resources(resident.device, logitsBuffer, topBuffer, resident.rowCount);
    const encoder = resident.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(matmulResources.pipeline);
    pass.setBindGroup(0, matmulResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(resident.rowCount / 8), 1);
    pass.setPipeline(topResources.pipeline);
    pass.setBindGroup(0, topResources.bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(topBuffer, 0, readbackBuffer, 0, 2 * Float32Array.BYTES_PER_ELEMENT);
    resident.device.queue.submit([encoder.finish()]);
    await resident.device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const mapped = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    matmulResources.destroy();
    topResources.destroy();
    return { id: Math.trunc(mapped[0] ?? 0), value: mapped[1] ?? -Infinity };
  } finally {
    logitsBuffer.destroy?.();
    topBuffer.destroy?.();
    readbackBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
  }
}

export async function matMulSwiGluWebGpuResident(
  gateHandle: WebGpuQuantizedWeightHandle,
  upHandle: WebGpuQuantizedWeightHandle,
  inputColumns: Float32Array,
  columnCount: number,
): Promise<Float32Array> {
  const gate = gateHandle as WebGpuQuantizedWeightHandleInternal;
  const up = upHandle as WebGpuQuantizedWeightHandleInternal;
  if (gate.device !== up.device) {
    throw new Error("WebGPU SwiGLU gate/up handles must belong to the same device");
  }
  if (gate.inputSize !== up.inputSize || gate.rowCount !== up.rowCount) {
    throw new Error("WebGPU SwiGLU gate/up handle shape mismatch");
  }
  if (inputColumns.length !== gate.inputSize * columnCount) {
    throw new Error(`WebGPU SwiGLU input shape mismatch: ${inputColumns.length}`);
  }
  if (gate.type === "Q8_0" || up.type === "Q8_0") {
    throw new Error("WebGPU SwiGLU currently supports K-quant gate/up weights only");
  }

  const q8 = quantizeQ8_KColumns(inputColumns, gate.inputSize, columnCount);
  const outputLength = gate.rowCount * columnCount;
  const gateOutputBuffer = storageBuffer(gate.device, outputLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const upOutputBuffer = storageBuffer(gate.device, outputLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const swigluBuffer = storageBuffer(gate.device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = gate.device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const inputScaleBuffer = storageBuffer(gate.device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(gate.device, q8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(gate.device, q8.bsums.byteLength, GPU_COPY_DST);

  try {
    gate.device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    gate.device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    gate.device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);

    const gateResources = createKMatMulBindResources(
      gate,
      inputScaleBuffer,
      inputQsBuffer,
      inputBsumsBuffer,
      gateOutputBuffer,
      columnCount,
    );
    const upResources = createKMatMulBindResources(
      up,
      inputScaleBuffer,
      inputQsBuffer,
      inputBsumsBuffer,
      upOutputBuffer,
      columnCount,
    );
    const swigluResources = createSwiGluResources(gate.device, gateOutputBuffer, upOutputBuffer, swigluBuffer, outputLength);

    const encoder = gate.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(gateResources.pipeline);
    pass.setBindGroup(0, gateResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(gate.rowCount / 8), columnCount);
    pass.setPipeline(upResources.pipeline);
    pass.setBindGroup(0, upResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(up.rowCount / 8), columnCount);
    pass.setPipeline(swigluResources.pipeline);
    pass.setBindGroup(0, swigluResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outputLength / 256));
    pass.end();
    encoder.copyBufferToBuffer(swigluBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    gate.device.queue.submit([encoder.finish()]);
    await gate.device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    gateResources.destroy();
    upResources.destroy();
    swigluResources.destroy();
    return output;
  } finally {
    gateOutputBuffer.destroy?.();
    upOutputBuffer.destroy?.();
    swigluBuffer.destroy?.();
    readbackBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
  }
}

export async function matMulSwiGluDownWebGpuResident(
  gateHandle: WebGpuQuantizedWeightHandle,
  upHandle: WebGpuQuantizedWeightHandle,
  downHandle: WebGpuQuantizedWeightHandle,
  inputColumns: Float32Array,
  columnCount: number,
): Promise<Float32Array> {
  const gate = gateHandle as WebGpuQuantizedWeightHandleInternal;
  const up = upHandle as WebGpuQuantizedWeightHandleInternal;
  const down = downHandle as WebGpuQuantizedWeightHandleInternal;
  if (gate.device !== up.device || gate.device !== down.device) {
    throw new Error("WebGPU FFN handles must belong to the same device");
  }
  if (gate.inputSize !== up.inputSize || gate.rowCount !== up.rowCount || down.inputSize !== gate.rowCount) {
    throw new Error("WebGPU FFN handle shape mismatch");
  }
  if (inputColumns.length !== gate.inputSize * columnCount) {
    throw new Error(`WebGPU FFN input shape mismatch: ${inputColumns.length}`);
  }
  if (gate.type === "Q8_0" || up.type === "Q8_0" || down.type === "Q8_0") {
    throw new Error("WebGPU FFN fusion currently supports K-quant weights only");
  }

  const inputQ8 = quantizeQ8_KColumns(inputColumns, gate.inputSize, columnCount);
  const hiddenLength = gate.rowCount * columnCount;
  const outputLength = down.rowCount * columnCount;
  const gateOutputBuffer = storageBuffer(gate.device, hiddenLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const upOutputBuffer = storageBuffer(gate.device, hiddenLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const swigluBuffer = storageBuffer(gate.device, hiddenLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const outputBuffer = storageBuffer(gate.device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = gate.device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const inputScaleBuffer = storageBuffer(gate.device, inputQ8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(gate.device, inputQ8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(gate.device, inputQ8.bsums.byteLength, GPU_COPY_DST);
  const downScaleBuffer = storageBuffer(gate.device, columnCount * down.blockCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const downQsBuffer = storageBuffer(gate.device, hiddenLength * Int32Array.BYTES_PER_ELEMENT, 0);
  const downBsumsBuffer = storageBuffer(gate.device, columnCount * down.blockCount * 16 * Int32Array.BYTES_PER_ELEMENT, 0);

  try {
    gate.device.queue.writeBuffer(inputScaleBuffer, 0, inputQ8.scales);
    gate.device.queue.writeBuffer(inputQsBuffer, 0, inputQ8.qs);
    gate.device.queue.writeBuffer(inputBsumsBuffer, 0, inputQ8.bsums);

    const gateResources = createKMatMulBindResources(gate, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, gateOutputBuffer, columnCount);
    const upResources = createKMatMulBindResources(up, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, upOutputBuffer, columnCount);
    const swigluResources = createSwiGluResources(gate.device, gateOutputBuffer, upOutputBuffer, swigluBuffer, hiddenLength);
    const quantizeResources = createQ8KQuantizeResources(
      gate.device,
      swigluBuffer,
      downScaleBuffer,
      downQsBuffer,
      downBsumsBuffer,
      down.inputSize,
      columnCount,
      down.blockCount,
    );
    const downResources = createKMatMulBindResources(down, downScaleBuffer, downQsBuffer, downBsumsBuffer, outputBuffer, columnCount);

    const encoder = gate.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(gateResources.pipeline);
    pass.setBindGroup(0, gateResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(gate.rowCount / 8), columnCount);
    pass.setPipeline(upResources.pipeline);
    pass.setBindGroup(0, upResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(up.rowCount / 8), columnCount);
    pass.setPipeline(swigluResources.pipeline);
    pass.setBindGroup(0, swigluResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(hiddenLength / 256));
    pass.setPipeline(quantizeResources.pipeline);
    pass.setBindGroup(0, quantizeResources.bindGroup);
    pass.dispatchWorkgroups(columnCount, down.blockCount);
    pass.setPipeline(downResources.pipeline);
    pass.setBindGroup(0, downResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(down.rowCount / 8), columnCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    gate.device.queue.submit([encoder.finish()]);
    await gate.device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();

    gateResources.destroy();
    upResources.destroy();
    swigluResources.destroy();
    quantizeResources.destroy();
    downResources.destroy();
    return output;
  } finally {
    gateOutputBuffer.destroy?.();
    upOutputBuffer.destroy?.();
    swigluBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
    downScaleBuffer.destroy?.();
    downQsBuffer.destroy?.();
    downBsumsBuffer.destroy?.();
  }
}

export async function matMulSsmNormGateOutWebGpuResident(
  zHandle: WebGpuQuantizedWeightHandle,
  outHandle: WebGpuQuantizedWeightHandle,
  attnNormColumns: Float32Array,
  deltaOutput: Float32Array,
  normWeight: Float32Array,
  epsilon: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  const z = zHandle as WebGpuQuantizedWeightHandleInternal;
  const out = outHandle as WebGpuQuantizedWeightHandleInternal;
  if (z.device !== out.device) {
    throw new Error("WebGPU SSM z/out handles must belong to the same device");
  }
  if (out.type !== "Q8_0") {
    return undefined;
  }
  if (z.type === "Q8_0") {
    throw new Error("WebGPU SSM z projection currently supports K-quant weights only");
  }
  if (attnNormColumns.length !== z.inputSize * columnCount) {
    throw new Error(`WebGPU SSM z input shape mismatch: ${attnNormColumns.length}`);
  }
  if (deltaOutput.length !== z.rowCount * columnCount || normWeight.length !== z.rowCount || out.inputSize !== z.rowCount) {
    throw new Error("WebGPU SSM norm/gate/out shape mismatch");
  }

  const q8 = quantizeQ8_KColumns(attnNormColumns, z.inputSize, columnCount);
  const hiddenLength = z.rowCount * columnCount;
  const outputLength = out.rowCount * columnCount;
  const zOutputBuffer = storageBuffer(z.device, hiddenLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const deltaBuffer = storageBuffer(z.device, deltaOutput.byteLength, GPU_COPY_DST);
  const normWeightBuffer = storageBuffer(z.device, normWeight.byteLength, GPU_COPY_DST);
  const gatedBuffer = storageBuffer(z.device, hiddenLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const outputBuffer = storageBuffer(z.device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = z.device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const inputScaleBuffer = storageBuffer(z.device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(z.device, q8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(z.device, q8.bsums.byteLength, GPU_COPY_DST);
  const outScaleBuffer = storageBuffer(z.device, columnCount * out.blockCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const outQsBuffer = storageBuffer(z.device, hiddenLength * Int32Array.BYTES_PER_ELEMENT, 0);

  try {
    z.device.queue.writeBuffer(deltaBuffer, 0, deltaOutput);
    z.device.queue.writeBuffer(normWeightBuffer, 0, normWeight);
    z.device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    z.device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    z.device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);

    const zResources = createKMatMulBindResources(z, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, zOutputBuffer, columnCount);
    const normGateResources = createSsmNormGateResources(
      z.device,
      deltaBuffer,
      zOutputBuffer,
      normWeightBuffer,
      gatedBuffer,
      z.rowCount,
      columnCount,
      epsilon,
    );
    const quantizeResources = createQ8_0QuantizeResources(
      z.device,
      gatedBuffer,
      outScaleBuffer,
      outQsBuffer,
      out.inputSize,
      columnCount,
      out.blockCount,
    );
    const outResources = createQ8_0MatMulBindResources(out, outScaleBuffer, outQsBuffer, outputBuffer, columnCount);

    const encoder = z.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(zResources.pipeline);
    pass.setBindGroup(0, zResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(z.rowCount / 8), columnCount);
    pass.setPipeline(normGateResources.pipeline);
    pass.setBindGroup(0, normGateResources.bindGroup);
    pass.dispatchWorkgroups(columnCount);
    pass.setPipeline(quantizeResources.pipeline);
    pass.setBindGroup(0, quantizeResources.bindGroup);
    pass.dispatchWorkgroups(columnCount, out.blockCount);
    pass.setPipeline(outResources.pipeline);
    pass.setBindGroup(0, outResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(out.rowCount / 8), columnCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    z.device.queue.submit([encoder.finish()]);
    await z.device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();

    zResources.destroy();
    normGateResources.destroy();
    quantizeResources.destroy();
    outResources.destroy();
    return output;
  } finally {
    zOutputBuffer.destroy?.();
    deltaBuffer.destroy?.();
    normWeightBuffer.destroy?.();
    gatedBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
    outScaleBuffer.destroy?.();
    outQsBuffer.destroy?.();
  }
}

export async function matMulQkvConvWebGpuResident(
  qkvHandle: WebGpuQuantizedWeightHandle,
  inputColumns: Float32Array,
  convState: Float32Array,
  convKernel: Float32Array,
  options: {
    tokenCount: number;
    convDim: number;
    kernelSize: number;
    stateSize: number;
    groupCount: number;
    valueDim: number;
  },
): Promise<WebGpuQkvConvResult | undefined> {
  const qkv = qkvHandle as WebGpuQuantizedWeightHandleInternal;
  if (qkv.type === "Q8_0") {
    return undefined;
  }
  const { tokenCount, convDim, kernelSize, stateSize, groupCount, valueDim } = options;
  const history = kernelSize - 1;
  const keyDim = stateSize * groupCount;
  if (
    qkv.rowCount !== convDim ||
    inputColumns.length !== qkv.inputSize * tokenCount ||
    convState.length !== history * convDim ||
    convKernel.length !== kernelSize * convDim ||
    valueDim % stateSize !== 0
  ) {
    throw new Error("WebGPU qkv/conv shape mismatch");
  }

  const inputQ8 = quantizeQ8_KColumns(inputColumns, qkv.inputSize, tokenCount);
  const qkvLength = convDim * tokenCount;
  const qLength = keyDim * tokenCount;
  const kLength = keyDim * tokenCount;
  const vLength = valueDim * tokenCount;
  const stateLength = history * convDim;
  const qkvBuffer = storageBuffer(qkv.device, qkvLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const qBuffer = storageBuffer(qkv.device, qLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const kBuffer = storageBuffer(qkv.device, kLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const vBuffer = storageBuffer(qkv.device, vLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const newStateBuffer = storageBuffer(qkv.device, stateLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = qkv.device.createBuffer({
    size: (qLength + kLength + vLength + stateLength) * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const inputScaleBuffer = storageBuffer(qkv.device, inputQ8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(qkv.device, inputQ8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(qkv.device, inputQ8.bsums.byteLength, GPU_COPY_DST);
  const convStateBuffer = storageBuffer(qkv.device, convState.byteLength, GPU_COPY_DST);
  const convKernelBuffer = storageBuffer(qkv.device, convKernel.byteLength, GPU_COPY_DST);

  try {
    qkv.device.queue.writeBuffer(inputScaleBuffer, 0, inputQ8.scales);
    qkv.device.queue.writeBuffer(inputQsBuffer, 0, inputQ8.qs);
    qkv.device.queue.writeBuffer(inputBsumsBuffer, 0, inputQ8.bsums);
    qkv.device.queue.writeBuffer(convStateBuffer, 0, convState);
    qkv.device.queue.writeBuffer(convKernelBuffer, 0, convKernel);

    const qkvResources = createKMatMulBindResources(qkv, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, qkvBuffer, tokenCount);
    const convResources = createQkvConvResources(
      qkv.device,
      qkvBuffer,
      convStateBuffer,
      convKernelBuffer,
      qBuffer,
      kBuffer,
      vBuffer,
      newStateBuffer,
      options,
    );

    const encoder = qkv.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(qkvResources.pipeline);
    pass.setBindGroup(0, qkvResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(qkv.rowCount / 8), tokenCount);
    pass.setPipeline(convResources.pipeline);
    pass.setBindGroup(0, convResources.bindGroup);
    pass.dispatchWorkgroups(Math.max(tokenCount * groupCount, tokenCount * (valueDim / stateSize), convDim), 3);
    pass.end();
    let offset = 0;
    encoder.copyBufferToBuffer(qBuffer, 0, readbackBuffer, offset, qLength * Float32Array.BYTES_PER_ELEMENT);
    offset += qLength * Float32Array.BYTES_PER_ELEMENT;
    encoder.copyBufferToBuffer(kBuffer, 0, readbackBuffer, offset, kLength * Float32Array.BYTES_PER_ELEMENT);
    offset += kLength * Float32Array.BYTES_PER_ELEMENT;
    encoder.copyBufferToBuffer(vBuffer, 0, readbackBuffer, offset, vLength * Float32Array.BYTES_PER_ELEMENT);
    offset += vLength * Float32Array.BYTES_PER_ELEMENT;
    encoder.copyBufferToBuffer(newStateBuffer, 0, readbackBuffer, offset, stateLength * Float32Array.BYTES_PER_ELEMENT);
    qkv.device.queue.submit([encoder.finish()]);
    await qkv.device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const mapped = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    qkvResources.destroy();
    convResources.destroy();
    return {
      q: mapped.slice(0, qLength),
      k: mapped.slice(qLength, qLength + kLength),
      v: mapped.slice(qLength + kLength, qLength + kLength + vLength),
      newConvState: mapped.slice(qLength + kLength + vLength),
    };
  } finally {
    qkvBuffer.destroy?.();
    qBuffer.destroy?.();
    kBuffer.destroy?.();
    vBuffer.destroy?.();
    newStateBuffer.destroy?.();
    readbackBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
    convStateBuffer.destroy?.();
    convKernelBuffer.destroy?.();
  }
}

export async function gatedDeltaNetWebGpu(
  query: Float32Array,
  key: Float32Array,
  value: Float32Array,
  gate: Float32Array,
  beta: Float32Array,
  state: Float32Array,
  options: {
    stateSize: number;
    keyHeadCount: number;
    valueHeadCount: number;
    tokenCount: number;
  },
): Promise<WebGpuGatedDeltaNetResult | undefined> {
  const { stateSize, keyHeadCount, valueHeadCount, tokenCount } = options;
  if (
    query.length !== tokenCount * keyHeadCount * stateSize ||
    key.length !== tokenCount * keyHeadCount * stateSize ||
    value.length !== tokenCount * valueHeadCount * stateSize ||
    gate.length !== tokenCount * valueHeadCount ||
    beta.length !== tokenCount * valueHeadCount ||
    state.length !== valueHeadCount * stateSize * stateSize ||
    valueHeadCount % keyHeadCount !== 0
  ) {
    throw new Error("WebGPU Gated DeltaNet shape mismatch");
  }

  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }

  const outputLength = tokenCount * valueHeadCount * stateSize;
  const queryBuffer = storageBuffer(device, query.byteLength, GPU_COPY_DST);
  const keyBuffer = storageBuffer(device, key.byteLength, GPU_COPY_DST);
  const valueBuffer = storageBuffer(device, value.byteLength, GPU_COPY_DST);
  const gateBuffer = storageBuffer(device, gate.byteLength, GPU_COPY_DST);
  const betaBuffer = storageBuffer(device, beta.byteLength, GPU_COPY_DST);
  const stateBuffer = storageBuffer(device, state.byteLength, GPU_COPY_DST);
  const outputBuffer = storageBuffer(device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const newStateBuffer = storageBuffer(device, state.byteLength, GPU_COPY_SRC);
  const readbackBuffer = device.createBuffer({
    size: (outputLength + state.length) * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    device.queue.writeBuffer(queryBuffer, 0, query);
    device.queue.writeBuffer(keyBuffer, 0, key);
    device.queue.writeBuffer(valueBuffer, 0, value);
    device.queue.writeBuffer(gateBuffer, 0, gate);
    device.queue.writeBuffer(betaBuffer, 0, beta);
    device.queue.writeBuffer(stateBuffer, 0, state);

    const resources = createGatedDeltaNetResources(
      device,
      queryBuffer,
      keyBuffer,
      valueBuffer,
      gateBuffer,
      betaBuffer,
      stateBuffer,
      outputBuffer,
      newStateBuffer,
      options,
    );
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(resources.pipeline);
    pass.setBindGroup(0, resources.bindGroup);
    pass.dispatchWorkgroups(valueHeadCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    encoder.copyBufferToBuffer(
      newStateBuffer,
      0,
      readbackBuffer,
      outputLength * Float32Array.BYTES_PER_ELEMENT,
      state.byteLength,
    );
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const mapped = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    resources.destroy();
    return {
      output: mapped.slice(0, outputLength),
      newState: mapped.slice(outputLength),
    };
  } finally {
    queryBuffer.destroy?.();
    keyBuffer.destroy?.();
    valueBuffer.destroy?.();
    gateBuffer.destroy?.();
    betaBuffer.destroy?.();
    stateBuffer.destroy?.();
    outputBuffer.destroy?.();
    newStateBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

export async function recurrentAttentionDecodeWebGpuResident(
  handles: {
    qkv: WebGpuQuantizedWeightHandle;
    alpha: WebGpuF32TensorHandle;
    beta: WebGpuF32TensorHandle;
    z: WebGpuQuantizedWeightHandle;
    out: WebGpuQuantizedWeightHandle;
    convKernel: WebGpuF32TensorHandle;
    dtBias: WebGpuF32TensorHandle;
    ssmA: WebGpuF32TensorHandle;
    normWeight: WebGpuF32TensorHandle;
  },
  inputColumns: Float32Array,
  convState: Float32Array,
  recurrentState: Float32Array,
  options: {
    inputSize: number;
    outputSize: number;
    convDim: number;
    kernelSize: number;
    stateSize: number;
    groupCount: number;
    valueHeadCount: number;
    epsilon: number;
  },
): Promise<{ attention: Float32Array; newConvState: Float32Array; newRecurrentState: Float32Array } | undefined> {
  const qkv = handles.qkv as WebGpuQuantizedWeightHandleInternal;
  const alpha = handles.alpha as WebGpuF32TensorHandleInternal;
  const beta = handles.beta as WebGpuF32TensorHandleInternal;
  const z = handles.z as WebGpuQuantizedWeightHandleInternal;
  const out = handles.out as WebGpuQuantizedWeightHandleInternal;
  const convKernel = handles.convKernel as WebGpuF32TensorHandleInternal;
  const dtBias = handles.dtBias as WebGpuF32TensorHandleInternal;
  const ssmA = handles.ssmA as WebGpuF32TensorHandleInternal;
  const normWeight = handles.normWeight as WebGpuF32TensorHandleInternal;
  const device = qkv.device;
  const sameDevice = [
    alpha.device,
    beta.device,
    z.device,
    out.device,
    convKernel.device,
    dtBias.device,
    ssmA.device,
    normWeight.device,
  ].every((candidate) => candidate === device);
  if (!sameDevice) {
    throw new Error("WebGPU recurrent decode handles must belong to the same device");
  }
  if (qkv.type === "Q8_0" || z.type === "Q8_0" || out.type !== "Q8_0") {
    return undefined;
  }

  const tokenCount = 1;
  const {
    inputSize,
    outputSize,
    convDim,
    kernelSize,
    stateSize,
    groupCount,
    valueHeadCount,
    epsilon,
  } = options;
  const valueDim = stateSize * valueHeadCount;
  const keyDim = stateSize * groupCount;
  const history = kernelSize - 1;
  if (
    qkv.inputSize !== inputSize ||
    qkv.rowCount !== convDim ||
    z.inputSize !== inputSize ||
    z.rowCount !== valueDim ||
    out.inputSize !== valueDim ||
    out.rowCount !== outputSize ||
    alpha.length !== inputSize * valueHeadCount ||
    beta.length !== inputSize * valueHeadCount ||
    convKernel.length !== convDim * kernelSize ||
    dtBias.length !== valueHeadCount ||
    ssmA.length !== valueHeadCount ||
    normWeight.length !== stateSize ||
    inputColumns.length !== inputSize ||
    convState.length !== convDim * history ||
    recurrentState.length !== valueHeadCount * stateSize * stateSize
  ) {
    throw new Error("WebGPU recurrent decode shape mismatch");
  }

  const inputQ8 = quantizeQ8_KColumns(inputColumns, inputSize, tokenCount);
  const qkvBuffer = storageBuffer(device, convDim * Float32Array.BYTES_PER_ELEMENT, 0);
  const zOutputBuffer = storageBuffer(device, valueDim * Float32Array.BYTES_PER_ELEMENT, 0);
  const alphaBuffer = storageBuffer(device, valueHeadCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const betaBuffer = storageBuffer(device, valueHeadCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const gateBuffer = storageBuffer(device, valueHeadCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const betaSigmoidBuffer = storageBuffer(device, valueHeadCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const qBuffer = storageBuffer(device, keyDim * Float32Array.BYTES_PER_ELEMENT, 0);
  const kBuffer = storageBuffer(device, keyDim * Float32Array.BYTES_PER_ELEMENT, 0);
  const vBuffer = storageBuffer(device, valueDim * Float32Array.BYTES_PER_ELEMENT, 0);
  const convStateBuffer = storageBuffer(device, convState.byteLength, GPU_COPY_DST);
  const newConvStateBuffer = storageBuffer(device, convState.byteLength, GPU_COPY_SRC);
  const recurrentStateBuffer = storageBuffer(device, recurrentState.byteLength, GPU_COPY_DST);
  const deltaBuffer = storageBuffer(device, valueDim * Float32Array.BYTES_PER_ELEMENT, 0);
  const newRecurrentStateBuffer = storageBuffer(device, recurrentState.byteLength, GPU_COPY_SRC);
  const gatedBuffer = storageBuffer(device, valueDim * Float32Array.BYTES_PER_ELEMENT, 0);
  const attentionBuffer = storageBuffer(device, outputSize * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const inputBuffer = storageBuffer(device, inputColumns.byteLength, GPU_COPY_DST);
  const inputScaleBuffer = storageBuffer(device, inputQ8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(device, inputQ8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(device, inputQ8.bsums.byteLength, GPU_COPY_DST);
  const outScaleBuffer = storageBuffer(device, out.blockCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const outQsBuffer = storageBuffer(device, valueDim * Int32Array.BYTES_PER_ELEMENT, 0);
  const readbackBuffer = device.createBuffer({
    size: (outputSize + convState.length + recurrentState.length) * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    device.queue.writeBuffer(inputBuffer, 0, inputColumns);
    device.queue.writeBuffer(inputScaleBuffer, 0, inputQ8.scales);
    device.queue.writeBuffer(inputQsBuffer, 0, inputQ8.qs);
    device.queue.writeBuffer(inputBsumsBuffer, 0, inputQ8.bsums);
    device.queue.writeBuffer(convStateBuffer, 0, convState);
    device.queue.writeBuffer(recurrentStateBuffer, 0, recurrentState);

    const qkvResources = createKMatMulBindResources(qkv, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, qkvBuffer, tokenCount);
    const zResources = createKMatMulBindResources(z, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, zOutputBuffer, tokenCount);
    const alphaResources = createF32MatMulResources(device, alpha.buffer, inputBuffer, alphaBuffer, inputSize, valueHeadCount, tokenCount);
    const betaResources = createF32MatMulResources(device, beta.buffer, inputBuffer, betaBuffer, inputSize, valueHeadCount, tokenCount);
    const convResources = createQkvConvResources(
      device,
      qkvBuffer,
      convStateBuffer,
      convKernel.buffer,
      qBuffer,
      kBuffer,
      vBuffer,
      newConvStateBuffer,
      { tokenCount, convDim, kernelSize, stateSize, groupCount, valueDim },
    );
    const gateResources = createDeltaGateResources(
      device,
      alphaBuffer,
      betaBuffer,
      dtBias.buffer,
      ssmA.buffer,
      gateBuffer,
      betaSigmoidBuffer,
      valueHeadCount,
      tokenCount,
    );
    const deltaResources = createGatedDeltaNetResources(
      device,
      qBuffer,
      kBuffer,
      vBuffer,
      gateBuffer,
      betaSigmoidBuffer,
      recurrentStateBuffer,
      deltaBuffer,
      newRecurrentStateBuffer,
      { stateSize, keyHeadCount: groupCount, valueHeadCount, tokenCount },
    );
    const normGateResources = createSsmNormGateResources(device, deltaBuffer, zOutputBuffer, normWeight.buffer, gatedBuffer, stateSize, valueHeadCount, epsilon);
    const quantizeResources = createQ8_0QuantizeResources(device, gatedBuffer, outScaleBuffer, outQsBuffer, out.inputSize, tokenCount, out.blockCount);
    const outResources = createQ8_0MatMulBindResources(out, outScaleBuffer, outQsBuffer, attentionBuffer, tokenCount);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(qkvResources.pipeline);
    pass.setBindGroup(0, qkvResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(qkv.rowCount / 8), tokenCount);
    pass.setPipeline(zResources.pipeline);
    pass.setBindGroup(0, zResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(z.rowCount / 8), tokenCount);
    pass.setPipeline(alphaResources.pipeline);
    pass.setBindGroup(0, alphaResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(valueHeadCount / 8), tokenCount);
    pass.setPipeline(betaResources.pipeline);
    pass.setBindGroup(0, betaResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(valueHeadCount / 8), tokenCount);
    pass.setPipeline(convResources.pipeline);
    pass.setBindGroup(0, convResources.bindGroup);
    pass.dispatchWorkgroups(Math.max(groupCount, valueHeadCount, convDim), 3);
    pass.setPipeline(gateResources.pipeline);
    pass.setBindGroup(0, gateResources.bindGroup);
    pass.dispatchWorkgroups(valueHeadCount, tokenCount);
    pass.setPipeline(deltaResources.pipeline);
    pass.setBindGroup(0, deltaResources.bindGroup);
    pass.dispatchWorkgroups(valueHeadCount);
    pass.setPipeline(normGateResources.pipeline);
    pass.setBindGroup(0, normGateResources.bindGroup);
    pass.dispatchWorkgroups(valueHeadCount);
    pass.setPipeline(quantizeResources.pipeline);
    pass.setBindGroup(0, quantizeResources.bindGroup);
    pass.dispatchWorkgroups(tokenCount, out.blockCount);
    pass.setPipeline(outResources.pipeline);
    pass.setBindGroup(0, outResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(out.rowCount / 8), tokenCount);
    pass.end();

    let offset = 0;
    encoder.copyBufferToBuffer(attentionBuffer, 0, readbackBuffer, offset, outputSize * Float32Array.BYTES_PER_ELEMENT);
    offset += outputSize * Float32Array.BYTES_PER_ELEMENT;
    encoder.copyBufferToBuffer(newConvStateBuffer, 0, readbackBuffer, offset, convState.byteLength);
    offset += convState.byteLength;
    encoder.copyBufferToBuffer(newRecurrentStateBuffer, 0, readbackBuffer, offset, recurrentState.byteLength);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const mapped = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();

    qkvResources.destroy();
    zResources.destroy();
    alphaResources.destroy();
    betaResources.destroy();
    convResources.destroy();
    gateResources.destroy();
    deltaResources.destroy();
    normGateResources.destroy();
    quantizeResources.destroy();
    outResources.destroy();
    return {
      attention: mapped.slice(0, outputSize),
      newConvState: mapped.slice(outputSize, outputSize + convState.length),
      newRecurrentState: mapped.slice(outputSize + convState.length),
    };
  } finally {
    qkvBuffer.destroy?.();
    zOutputBuffer.destroy?.();
    alphaBuffer.destroy?.();
    betaBuffer.destroy?.();
    gateBuffer.destroy?.();
    betaSigmoidBuffer.destroy?.();
    qBuffer.destroy?.();
    kBuffer.destroy?.();
    vBuffer.destroy?.();
    convStateBuffer.destroy?.();
    newConvStateBuffer.destroy?.();
    recurrentStateBuffer.destroy?.();
    deltaBuffer.destroy?.();
    newRecurrentStateBuffer.destroy?.();
    gatedBuffer.destroy?.();
    attentionBuffer.destroy?.();
    inputBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
    outScaleBuffer.destroy?.();
    outQsBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

export async function fullAttentionDecodeOutWebGpuResident(
  outHandle: WebGpuQuantizedWeightHandle,
  query: Float32Array,
  keyCache: Float32Array,
  valueCache: Float32Array,
  gate: Float32Array,
  options: {
    headSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    scale: number;
  },
): Promise<Float32Array | undefined> {
  const out = outHandle as WebGpuQuantizedWeightHandleInternal;
  if (out.type === "Q8_0") {
    return undefined;
  }
  const { headSize, queryHeadCount, keyValueHeadCount, keyValueTokenCount, contextLength, scale } = options;
  const hiddenSize = headSize * queryHeadCount;
  if (
    out.inputSize !== hiddenSize ||
    query.length !== hiddenSize ||
    gate.length !== hiddenSize ||
    keyCache.length < keyValueTokenCount * keyValueHeadCount * headSize ||
    valueCache.length < headSize * keyValueHeadCount * contextLength ||
    queryHeadCount % keyValueHeadCount !== 0
  ) {
    throw new Error("WebGPU full attention decode shape mismatch");
  }

  const queryBuffer = storageBuffer(out.device, query.byteLength, GPU_COPY_DST);
  const keyBuffer = storageBuffer(out.device, keyCache.byteLength, GPU_COPY_DST);
  const valueBuffer = storageBuffer(out.device, valueCache.byteLength, GPU_COPY_DST);
  const gateBuffer = storageBuffer(out.device, gate.byteLength, GPU_COPY_DST);
  const gatedBuffer = storageBuffer(out.device, hiddenSize * Float32Array.BYTES_PER_ELEMENT, 0);
  const probabilitiesBuffer = storageBuffer(
    out.device,
    queryHeadCount * keyValueTokenCount * Float32Array.BYTES_PER_ELEMENT,
    0,
  );
  const outputBuffer = storageBuffer(out.device, out.rowCount * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = out.device.createBuffer({
    size: out.rowCount * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const inputScaleBuffer = storageBuffer(out.device, out.blockCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const inputQsBuffer = storageBuffer(out.device, hiddenSize * Int32Array.BYTES_PER_ELEMENT, 0);
  const inputBsumsBuffer = storageBuffer(out.device, out.blockCount * 16 * Int32Array.BYTES_PER_ELEMENT, 0);

  try {
    out.device.queue.writeBuffer(queryBuffer, 0, query);
    out.device.queue.writeBuffer(keyBuffer, 0, keyCache);
    out.device.queue.writeBuffer(valueBuffer, 0, valueCache);
    out.device.queue.writeBuffer(gateBuffer, 0, gate);

    const scoreResources = createFullAttentionScoreResources(
      out.device,
      queryBuffer,
      keyBuffer,
      probabilitiesBuffer,
      options,
    );
    const applyResources = createFullAttentionApplyResources(
      out.device,
      valueBuffer,
      gateBuffer,
      probabilitiesBuffer,
      gatedBuffer,
      options,
    );
    const quantizeResources = createQ8KQuantizeResources(
      out.device,
      gatedBuffer,
      inputScaleBuffer,
      inputQsBuffer,
      inputBsumsBuffer,
      out.inputSize,
      1,
      out.blockCount,
    );
    const outResources = createKMatMulBindResources(out, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, outputBuffer, 1);

    const encoder = out.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(scoreResources.pipeline);
    pass.setBindGroup(0, scoreResources.bindGroup);
    pass.dispatchWorkgroups(queryHeadCount);
    pass.setPipeline(applyResources.pipeline);
    pass.setBindGroup(0, applyResources.bindGroup);
    pass.dispatchWorkgroups(queryHeadCount, headSize);
    pass.setPipeline(quantizeResources.pipeline);
    pass.setBindGroup(0, quantizeResources.bindGroup);
    pass.dispatchWorkgroups(1, out.blockCount);
    pass.setPipeline(outResources.pipeline);
    pass.setBindGroup(0, outResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(out.rowCount / 8), 1);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, out.rowCount * Float32Array.BYTES_PER_ELEMENT);
    out.device.queue.submit([encoder.finish()]);
    await out.device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    scoreResources.destroy();
    applyResources.destroy();
    quantizeResources.destroy();
    outResources.destroy();
    return output;
  } finally {
    queryBuffer.destroy?.();
    keyBuffer.destroy?.();
    valueBuffer.destroy?.();
    gateBuffer.destroy?.();
    gatedBuffer.destroy?.();
    probabilitiesBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
  }
}

export function planQwen35WebGpuHybrid(
  gguf: GgufMetadata,
  manifest: Qwen35ModelManifest,
  options: Qwen35WebGpuPlanningOptions = {},
): Qwen35WebGpuHybridPlan {
  const mode = options.mode ?? "off";
  const memoryLimitBytes = options.memoryLimitBytes ?? QWEN35_WEBGPU_MEMORY_LIMIT_BYTES;
  const browserGate = options.browserGate ?? "required";
  const contextLength = Math.min(
    options.contextLength ?? manifest.contextLength,
    manifest.contextLength,
  );
  const support = options.support;
  const tensorsByName = new Map(gguf.tensors.map((tensor) => [tensor.name, tensor]));
  const outputBytes = tensorBytes(tensorsByName, "output.weight") +
    tensorBytes(tensorsByName, "output_norm.weight");
  const layerPlans = buildLayerPlans(tensorsByName, manifest, contextLength);

  if (mode === "off") {
    return emptyPlan({
      status: "off",
      mode,
      memoryLimitBytes,
      browserGate,
      reason: "WebGPU execution is off; this is a placement plan only.",
      outputBytes,
      fixedBytes: DEFAULT_GPU_FIXED_BYTES,
      scratchBytes: DEFAULT_GPU_SCRATCH_BYTES,
      blockCount: manifest.blockCount,
    });
  }

  if (support && !support.available) {
    return emptyPlan({
      status: "unavailable",
      mode,
      memoryLimitBytes,
      browserGate,
      reason: `WebGPU unavailable: ${support.reason}`,
      outputBytes,
      fixedBytes: DEFAULT_GPU_FIXED_BYTES,
      scratchBytes: DEFAULT_GPU_SCRATCH_BYTES,
      blockCount: manifest.blockCount,
    });
  }

  const fixedBytes = DEFAULT_GPU_FIXED_BYTES;
  const scratchBytes = DEFAULT_GPU_SCRATCH_BYTES;
  let selectedBytes = outputBytes + fixedBytes + scratchBytes;
  const selectedLayers: Qwen35WebGpuLayerPlan[] = [];

  if (selectedBytes > memoryLimitBytes) {
    return emptyPlan({
      status: "blocked",
      mode,
      memoryLimitBytes,
      browserGate,
      reason: "`output.weight` plus fixed GPU buffers exceed the configured WebGPU memory cap.",
      outputBytes,
      fixedBytes,
      scratchBytes,
      blockCount: manifest.blockCount,
    });
  }

  for (let layer = manifest.blockCount - 1; layer >= 0; layer -= 1) {
    const candidate = layerPlans.get(layer);
    if (!candidate) {
      continue;
    }
    if (selectedBytes + candidate.totalBytes > memoryLimitBytes) {
      break;
    }
    selectedLayers.unshift(candidate);
    selectedBytes += candidate.totalBytes;
  }

  const firstGpuLayer = selectedLayers[0]?.layer;
  const gpuWeightBytes = outputBytes +
    selectedLayers.reduce((sum, layer) => sum + layer.weightBytes, 0);
  const gpuCacheBytes = selectedLayers.reduce((sum, layer) => sum + layer.cacheBytes, 0);
  const status: Qwen35WebGpuPlanStatus = browserGate === "passed" ? "planned" : "blocked";
  const reason = browserGate === "passed"
    ? "WebGPU suffix placement is planned, but execution still requires verified kernels."
    : "Browser user check is required before WebGPU execution can be enabled.";

  return {
    status,
    mode,
    memoryLimitBytes,
    browserGate,
    enabled: false,
    reason,
    outputBytes,
    fixedBytes,
    scratchBytes,
    selectedLayerCount: selectedLayers.length,
    firstGpuLayer,
    cpuPrefixLayerCount: firstGpuLayer === undefined ? manifest.blockCount : firstGpuLayer,
    gpuSuffixLayerCount: selectedLayers.length,
    gpuWeightBytes,
    gpuCacheBytes,
    estimatedResidentBytes: selectedBytes,
    remainingBytes: Math.max(0, memoryLimitBytes - selectedBytes),
    selectedLayers,
    copyAuditExpectations: {
      decodeTensorReads: 0,
      suffixIntermediateReadbacks: 0,
      logitsReadbacks: 0,
      expectedBoundaryUploads: selectedLayers.length > 0 ? 1 : 0,
      expectedTokenReadbacks: 1,
    },
  };
}

export function auditQwen35WebGpuCopies(
  plan: Qwen35WebGpuHybridPlan,
  observed: Qwen35WebGpuCopyAuditObservation,
): Qwen35WebGpuCopyAuditResult {
  const expected = plan.copyAuditExpectations;
  const errors: string[] = [];

  if (observed.decodeTensorReads !== expected.decodeTensorReads) {
    errors.push(
      `decode tensor reads: expected ${expected.decodeTensorReads}, got ${observed.decodeTensorReads}`,
    );
  }
  if (observed.suffixIntermediateReadbacks !== expected.suffixIntermediateReadbacks) {
    errors.push(
      `suffix intermediate readbacks: expected ${expected.suffixIntermediateReadbacks}, got ${observed.suffixIntermediateReadbacks}`,
    );
  }
  if (observed.logitsReadbacks !== expected.logitsReadbacks) {
    errors.push(`logits readbacks: expected ${expected.logitsReadbacks}, got ${observed.logitsReadbacks}`);
  }
  if (observed.boundaryUploads > expected.expectedBoundaryUploads) {
    errors.push(
      `boundary uploads: expected at most ${expected.expectedBoundaryUploads}, got ${observed.boundaryUploads}`,
    );
  }
  if (observed.tokenReadbacks > expected.expectedTokenReadbacks) {
    errors.push(
      `token readbacks: expected at most ${expected.expectedTokenReadbacks}, got ${observed.tokenReadbacks}`,
    );
  }

  return {
    ok: errors.length === 0,
    errors,
    expected,
    observed,
  };
}

function buildLayerPlans(
  tensorsByName: Map<string, GgufTensorInfo>,
  manifest: Qwen35ModelManifest,
  contextLength: number,
): Map<number, Qwen35WebGpuLayerPlan> {
  const plans = new Map<number, Qwen35WebGpuLayerPlan>();
  for (let layer = 0; layer < manifest.blockCount; layer += 1) {
    const layerKind = manifest.fullAttentionLayers.includes(layer)
      ? "full-attention"
      : "recurrent";
    const weightBytes = manifest.expectedTensors.reduce((sum, expected) => {
      if (expected.layer !== layer) {
        return sum;
      }
      return sum + tensorBytes(tensorsByName, expected.name);
    }, 0);
    const cacheBytes = layerKind === "full-attention"
      ? fullAttentionCacheBytes(manifest, contextLength)
      : recurrentCacheBytes(manifest);
    plans.set(layer, {
      layer,
      layerKind,
      weightBytes,
      cacheBytes,
      totalBytes: weightBytes + cacheBytes,
    });
  }
  return plans;
}

function tensorBytes(tensorsByName: Map<string, GgufTensorInfo>, name: string): number {
  const tensor = tensorsByName.get(name);
  return tensor ? tensorByteLength(tensor) : 0;
}

function recurrentCacheBytes(manifest: Qwen35ModelManifest): number {
  const convDim =
    manifest.ssm.stateSize * manifest.ssm.groupCount * 2 +
    manifest.ssm.stateSize * manifest.ssm.timeStepRank;
  const recurrentStateSize =
    manifest.ssm.stateSize * manifest.ssm.stateSize * manifest.ssm.timeStepRank;
  return (manifest.ssm.convKernel - 1) * convDim * Float32Array.BYTES_PER_ELEMENT +
    recurrentStateSize * Float32Array.BYTES_PER_ELEMENT;
}

function fullAttentionCacheBytes(
  manifest: Qwen35ModelManifest,
  contextLength: number,
): number {
  const perCache = contextLength * manifest.headCountKv * manifest.keyLength;
  return perCache * 2 * Float32Array.BYTES_PER_ELEMENT;
}

function emptyPlan(params: {
  status: Qwen35WebGpuPlanStatus;
  mode: Qwen35WebGpuMode;
  memoryLimitBytes: number;
  browserGate: Qwen35WebGpuBrowserGate;
  reason: string;
  outputBytes: number;
  fixedBytes: number;
  scratchBytes: number;
  blockCount: number;
}): Qwen35WebGpuHybridPlan {
  const estimatedResidentBytes = params.outputBytes + params.fixedBytes + params.scratchBytes;
  return {
    status: params.status,
    mode: params.mode,
    memoryLimitBytes: params.memoryLimitBytes,
    browserGate: params.browserGate,
    enabled: false,
    reason: params.reason,
    outputBytes: params.outputBytes,
    fixedBytes: params.fixedBytes,
    scratchBytes: params.scratchBytes,
    selectedLayerCount: 0,
    cpuPrefixLayerCount: params.blockCount,
    gpuSuffixLayerCount: 0,
    gpuWeightBytes: params.outputBytes,
    gpuCacheBytes: 0,
    estimatedResidentBytes,
    remainingBytes: Math.max(0, params.memoryLimitBytes - estimatedResidentBytes),
    selectedLayers: [],
    copyAuditExpectations: {
      decodeTensorReads: 0,
      suffixIntermediateReadbacks: 0,
      logitsReadbacks: 0,
      expectedBoundaryUploads: 0,
      expectedTokenReadbacks: 1,
    },
  };
}

function describeAdapter(adapter: WebGpuAdapterLike): string | undefined {
  const info = adapter.info;
  if (!info) {
    return undefined;
  }
  return [
    info.vendor,
    info.architecture,
    info.device,
    info.description,
  ].filter(Boolean).join(" / ") || undefined;
}

async function webGpuDevice(): Promise<WebGpuDeviceLike | undefined> {
  if (!devicePromise) {
    devicePromise = requestWebGpuDevice();
  }
  return devicePromise;
}

async function webGpuAdapterLimits(): Promise<Pick<WebGpuAdapterLike, "limits">["limits"] | undefined> {
  adapterLimitsPromise ??= (async () => {
    if (typeof navigator === "undefined") {
      return undefined;
    }
    const gpu = (navigator as NavigatorWithWebGpu).gpu;
    const adapter = await gpu?.requestAdapter();
    return adapter?.limits;
  })();
  return adapterLimitsPromise;
}

async function assertStorageBindingFits(label: string, byteLength: number): Promise<void> {
  const limits = await webGpuAdapterLimits();
  const maxStorageBufferBindingSize = limits?.maxStorageBufferBindingSize;
  if (maxStorageBufferBindingSize !== undefined && byteLength > maxStorageBufferBindingSize) {
    throw new Error(
      `WebGPU ${label} buffer ${byteLength} bytes exceeds maxStorageBufferBindingSize ${maxStorageBufferBindingSize}; row sharding is required.`,
    );
  }
  const maxBufferSize = limits?.maxBufferSize;
  if (maxBufferSize !== undefined && byteLength > maxBufferSize) {
    throw new Error(`WebGPU ${label} buffer ${byteLength} bytes exceeds maxBufferSize ${maxBufferSize}.`);
  }
}

async function requestWebGpuDevice(): Promise<WebGpuDeviceLike | undefined> {
  if (typeof navigator === "undefined") {
    return undefined;
  }
  const gpu = (navigator as NavigatorWithWebGpu).gpu;
  if (!gpu) {
    return undefined;
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter?.requestDevice) {
    return undefined;
  }
  return adapter.requestDevice({
    requiredLimits: requestedDeviceLimits(adapter.limits),
  });
}

function requestedDeviceLimits(limits: WebGpuAdapterLike["limits"]): Record<string, number> {
  const requiredLimits: Record<string, number> = {};
  if (limits?.maxStorageBufferBindingSize !== undefined) {
    requiredLimits.maxStorageBufferBindingSize = limits.maxStorageBufferBindingSize;
  }
  if (limits?.maxBufferSize !== undefined) {
    requiredLimits.maxBufferSize = limits.maxBufferSize;
  }
  return requiredLimits;
}

function smokeFailure(reason: WebGpuSmokeTest extends infer T
  ? T extends { ok: false; reason: infer R } ? R : never
  : never, start: number): WebGpuSmokeTest {
  return {
    ok: false,
    reason,
    durationMs: nowMs() - start,
  };
}

function storageBuffer(
  device: WebGpuDeviceLike,
  size: number,
  extraUsage: number,
): WebGpuBufferLike {
  return device.createBuffer({
    size,
    usage: GPU_STORAGE | extraUsage,
  });
}

function storageEntry(binding: number, type: "read-only-storage" | "storage"): unknown {
  return {
    binding,
    visibility: GPU_SHADER_STAGE_COMPUTE,
    buffer: { type },
  };
}

function bindBuffer(binding: number, buffer: WebGpuBufferLike): unknown {
  return {
    binding,
    resource: { buffer },
  };
}

function webGpuQuantizedWeightLayout(
  type: WebGpuQuantizedMatMulType,
  inputSize: number,
): { blockCount: number; rowByteLength: number } {
  if (type === "Q8_0") {
    if (inputSize % 32 !== 0) {
      throw new Error(`WebGPU Q8_0 matmul input size must be divisible by 32, got ${inputSize}`);
    }
    const blockCount = inputSize / 32;
    return { blockCount, rowByteLength: blockCount * 34 };
  }
  if (inputSize % 256 !== 0) {
    throw new Error(`WebGPU ${type} matmul input size must be divisible by 256, got ${inputSize}`);
  }
  const blockCount = inputSize / 256;
  if (type === "Q4_K") {
    return { blockCount, rowByteLength: blockCount * 144 };
  }
  if (type === "Q5_K") {
    return { blockCount, rowByteLength: blockCount * 176 };
  }
  return { blockCount, rowByteLength: blockCount * 210 };
}

function createKMatMulBindResources(
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

function createSwiGluResources(
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

function createF32MatMulResources(
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

function createQ8KQuantizeResources(
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

function createDeltaGateResources(
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

function createTop1Resources(
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

function createQ8_0QuantizeResources(
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

function createSsmNormGateResources(
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

function createQkvConvResources(
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

function createGatedDeltaNetResources(
  device: WebGpuDeviceLike,
  queryBuffer: WebGpuBufferLike,
  keyBuffer: WebGpuBufferLike,
  valueBuffer: WebGpuBufferLike,
  gateBuffer: WebGpuBufferLike,
  betaBuffer: WebGpuBufferLike,
  stateBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  newStateBuffer: WebGpuBufferLike,
  options: {
    stateSize: number;
    keyHeadCount: number;
    valueHeadCount: number;
    tokenCount: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const params = new Uint32Array([
    options.stateSize,
    options.keyHeadCount,
    options.valueHeadCount,
    options.tokenCount,
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
      storageEntry(3, "read-only-storage"),
      storageEntry(4, "read-only-storage"),
      storageEntry(5, "read-only-storage"),
      {
        binding: 6,
        visibility: GPU_SHADER_STAGE_COMPUTE,
        buffer: { type: "uniform" },
      },
      storageEntry(7, "storage"),
      storageEntry(8, "storage"),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module: device.createShaderModule({ code: GATED_DELTA_NET_WGSL }),
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
        bindBuffer(2, valueBuffer),
        bindBuffer(3, gateBuffer),
        bindBuffer(4, betaBuffer),
        bindBuffer(5, stateBuffer),
        bindBuffer(6, paramsBuffer),
        bindBuffer(7, outputBuffer),
        bindBuffer(8, newStateBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

function createFullAttentionScoreResources(
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
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.scale, 0, 0, 0, 0, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = options.headSize;
  paramsU32[2] = options.queryHeadCount;
  paramsU32[3] = options.keyValueHeadCount;
  paramsU32[4] = options.keyValueTokenCount;
  paramsU32[5] = options.contextLength;
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

function createFullAttentionApplyResources(
  device: WebGpuDeviceLike,
  valueBuffer: WebGpuBufferLike,
  gateBuffer: WebGpuBufferLike,
  probabilitiesBuffer: WebGpuBufferLike,
  outputBuffer: WebGpuBufferLike,
  options: {
    headSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    scale: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.scale, 0, 0, 0, 0, 0, 0, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[1] = options.headSize;
  paramsU32[2] = options.queryHeadCount;
  paramsU32[3] = options.keyValueHeadCount;
  paramsU32[4] = options.keyValueTokenCount;
  paramsU32[5] = options.contextLength;
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
        bindBuffer(1, gateBuffer),
        bindBuffer(2, probabilitiesBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, outputBuffer),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

function createQ8_0MatMulBindResources(
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

async function matMulQ8_0Resident(
  handle: WebGpuQuantizedWeightHandleInternal,
  inputColumns: Float32Array,
  columnCount: number,
): Promise<Float32Array> {
  const q8 = quantizeQ8_0Columns(inputColumns, handle.inputSize, columnCount);
  const outputLength = handle.rowCount * columnCount;
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

  const inputScaleBuffer = storageBuffer(handle.device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(handle.device, q8.qs.byteLength, GPU_COPY_DST);
  const paramsBuffer = handle.device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  const outputBuffer = storageBuffer(handle.device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = handle.device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    handle.device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    handle.device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    handle.device.queue.writeBuffer(paramsBuffer, 0, params);

    const shaderModule = handle.device.createShaderModule({ code: Q8_0_MATMUL_WGSL });
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
        module: shaderModule,
        entryPoint: "main",
      },
    });
    const bindGroup = handle.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, handle.weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, outputBuffer),
      ],
    });

    return await dispatchMatMulReadback(handle.device, pipeline, bindGroup, outputBuffer, readbackBuffer, outputLength, handle.rowCount, columnCount);
  } finally {
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    paramsBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

async function matMulKResident(
  handle: WebGpuQuantizedWeightHandleInternal,
  inputColumns: Float32Array,
  columnCount: number,
  shaderCode: string,
  usesBsums: boolean,
): Promise<Float32Array> {
  const q8 = quantizeQ8_KColumns(inputColumns, handle.inputSize, columnCount);
  const outputLength = handle.rowCount * columnCount;
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

  const inputScaleBuffer = storageBuffer(handle.device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(handle.device, q8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = usesBsums ? storageBuffer(handle.device, q8.bsums.byteLength, GPU_COPY_DST) : undefined;
  const paramsBuffer = handle.device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  const outputBuffer = storageBuffer(handle.device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = handle.device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    handle.device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    handle.device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    if (inputBsumsBuffer) {
      handle.device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);
    }
    handle.device.queue.writeBuffer(paramsBuffer, 0, params);

    const shaderModule = handle.device.createShaderModule({ code: shaderCode });
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
          bindBuffer(3, inputBsumsBuffer as WebGpuBufferLike),
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
    const bindGroup = handle.device.createBindGroup({
      layout: bindGroupLayout,
      entries,
    });

    return await dispatchMatMulReadback(handle.device, pipeline, bindGroup, outputBuffer, readbackBuffer, outputLength, handle.rowCount, columnCount);
  } finally {
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer?.destroy?.();
    paramsBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

async function dispatchMatMulReadback(
  device: WebGpuDeviceLike,
  pipeline: unknown,
  bindGroup: unknown,
  outputBuffer: WebGpuBufferLike,
  readbackBuffer: WebGpuBufferLike,
  outputLength: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone?.();
  await readbackBuffer.mapAsync(GPU_MAP_READ);
  const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
  readbackBuffer.unmap();
  return output;
}

function top1Cpu(values: Float32Array): WebGpuTopToken {
  let id = 0;
  let value = values[0] ?? -Infinity;
  for (let index = 1; index < values.length; index += 1) {
    const candidate = values[index] ?? -Infinity;
    if (candidate > value) {
      id = index;
      value = candidate;
    }
  }
  return { id, value };
}

function quantizeQ8_0Columns(
  inputColumns: Float32Array,
  inputSize: number,
  columnCount: number,
): { scales: Float32Array; qs: Int32Array } {
  const blockCount = inputSize / 32;
  const scales = new Float32Array(columnCount * blockCount);
  const qs = new Int32Array(inputColumns.length);
  for (let column = 0; column < columnCount; column += 1) {
    for (let block = 0; block < blockCount; block += 1) {
      const base = column * inputSize + block * 32;
      let amax = 0;
      for (let index = 0; index < 32; index += 1) {
        amax = Math.max(amax, Math.abs(inputColumns[base + index] ?? 0));
      }
      const scale = float16ToFloat32(float32ToFloat16(amax / 127));
      const inverseScale = scale ? 1 / scale : 0;
      scales[column * blockCount + block] = scale;
      for (let index = 0; index < 32; index += 1) {
        qs[base + index] = Math.round((inputColumns[base + index] ?? 0) * inverseScale);
      }
    }
  }
  return { scales, qs };
}

function quantizeQ8_KColumns(
  inputColumns: Float32Array,
  inputSize: number,
  columnCount: number,
): { scales: Float32Array; qs: Int32Array; bsums: Int32Array } {
  const blockCount = inputSize / 256;
  const scales = new Float32Array(columnCount * blockCount);
  const qs = new Int32Array(inputColumns.length);
  const bsums = new Int32Array(columnCount * blockCount * 16);
  for (let column = 0; column < columnCount; column += 1) {
    for (let block = 0; block < blockCount; block += 1) {
      const base = column * inputSize + block * 256;
      let max = 0;
      let amax = 0;
      for (let index = 0; index < 256; index += 1) {
        const value = inputColumns[base + index] ?? 0;
        const abs = Math.abs(value);
        if (abs > amax) {
          amax = abs;
          max = value;
        }
      }
      if (amax === 0) {
        continue;
      }
      const inverseScale = -127 / max;
      scales[column * blockCount + block] = 1 / inverseScale;
      for (let index = 0; index < 256; index += 1) {
        qs[base + index] = Math.min(127, Math.round(inverseScale * (inputColumns[base + index] ?? 0)));
      }
      for (let group = 0; group < 16; group += 1) {
        let sum = 0;
        for (let index = 0; index < 16; index += 1) {
          sum += qs[base + group * 16 + index] ?? 0;
        }
        bsums[(column * blockCount + block) * 16 + group] = sum;
      }
    }
  }
  return { scales, qs, bsums };
}

function packBytesToU32(bytes: Uint8Array): Uint32Array {
  const packed = new Uint32Array(Math.ceil(bytes.byteLength / 4));
  for (let index = 0; index < bytes.byteLength; index += 1) {
    packed[index >> 2] |= (bytes[index] ?? 0) << ((index & 3) * 8);
  }
  return packed;
}

function float16ToFloat32(value: number): number {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) {
    return sign * Math.pow(2, -14) * (fraction / 1024);
  }
  if (exponent === 31) {
    return fraction === 0 ? sign * Infinity : NaN;
  }
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function float32ToFloat16(value: number): number {
  if (Number.isNaN(value)) {
    return 0x7e00;
  }
  if (value === Infinity) {
    return 0x7c00;
  }
  if (value === -Infinity) {
    return 0xfc00;
  }
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const abs = Math.abs(value);
  if (abs === 0) {
    return sign;
  }
  if (abs >= 65504) {
    return sign | 0x7bff;
  }
  if (abs < 2 ** -24) {
    return sign;
  }
  let exponent = Math.floor(Math.log2(abs));
  const mantissa = abs / 2 ** exponent - 1;
  if (exponent < -14) {
    const subnormal = Math.round(abs / 2 ** -24);
    return sign | subnormal;
  }
  let halfMantissa = Math.round(mantissa * 1024);
  if (halfMantissa === 1024) {
    exponent += 1;
    halfMantissa = 0;
  }
  return sign | ((exponent + 15) << 10) | halfMantissa;
}

function sameNumbers(left: number[], right: number[]): boolean {
  return left.length === right.length &&
    left.every((value, index) => Math.abs(value - (right[index] ?? Number.NaN)) < 1e-6);
}

function nowMs(): number {
  return globalThis.performance?.now() ?? Date.now();
}

export type Qwen35WebGpuSuffixRunnerOptions = {
  tensorReader: GgufTensorReader;
  manifest: Qwen35ModelManifest;
  epsilon: number;
  contextLength: number;
  memoryLimitBytes?: number;
  firstGpuLayer?: number;
};

export type Qwen35WebGpuStateLike = {
  contextLength: number;
  nextPosition: number;
};

export type Qwen35WebGpuTokenResult = {
  topTokens?: WebGpuTopToken[];
};

type GpuResource = {
  destroy?: () => void;
};

type QuantizedHandle = WebGpuQuantizedWeightHandleInternal;
type F32Handle = WebGpuF32TensorHandleInternal;

type OutputStripe = QuantizedHandle & {
  readonly rowOffset: number;
};

type RecurrentGpuLayer = {
  kind: "recurrent";
  layer: number;
  attnNorm: F32Handle;
  qkv: QuantizedHandle;
  alpha: F32Handle;
  beta: F32Handle;
  z: QuantizedHandle;
  convKernel: F32Handle;
  dtBias: F32Handle;
  ssmA: F32Handle;
  ssmNorm: F32Handle;
  out: QuantizedHandle;
  postNorm: F32Handle;
  ffnGate: QuantizedHandle;
  ffnUp: QuantizedHandle;
  ffnDown: QuantizedHandle;
};

type FullAttentionGpuLayer = {
  kind: "full-attention";
  layer: number;
  attnNorm: F32Handle;
  q: QuantizedHandle;
  k: QuantizedHandle;
  v: QuantizedHandle;
  out: QuantizedHandle;
  qNorm: F32Handle;
  kNorm: F32Handle;
  postNorm: F32Handle;
  ffnGate: QuantizedHandle;
  ffnUp: QuantizedHandle;
  ffnDown: QuantizedHandle;
};

type GpuLayer = RecurrentGpuLayer | FullAttentionGpuLayer;

type RecurrentGpuLayerState = {
  conv: WebGpuBufferLike;
  recurrent: WebGpuBufferLike;
};

type FullAttentionGpuLayerState = {
  key: WebGpuBufferLike;
  value: WebGpuBufferLike;
};

type GpuState = {
  recurrent: Map<number, RecurrentGpuLayerState>;
  fullAttention: Map<number, FullAttentionGpuLayerState>;
};

class GpuMemoryArena {
  readonly device: WebGpuDeviceLike;
  readonly limitBytes: number;
  private allocatedBytes = 0;

  constructor(
    device: WebGpuDeviceLike,
    limitBytes: number,
  ) {
    this.device = device;
    this.limitBytes = limitBytes;
  }

  get residentBytes(): number {
    return this.allocatedBytes;
  }

  createBuffer(label: string, size: number, usage: number, mappedAtCreation = false): WebGpuBufferLike {
    const byteLength = align4(size);
    if (this.allocatedBytes + byteLength > this.limitBytes) {
      throw new Error(
        `WebGPU memory cap exceeded while allocating ${label}: ` +
          `${this.allocatedBytes + byteLength} > ${this.limitBytes}`,
      );
    }
    const buffer = this.device.createBuffer({
      size: byteLength,
      usage,
      mappedAtCreation,
    });
    this.allocatedBytes += byteLength;
    const destroy = buffer.destroy?.bind(buffer);
    let destroyed = false;
    buffer.destroy = () => {
      if (destroyed) {
        return;
      }
      destroyed = true;
      this.allocatedBytes -= byteLength;
      destroy?.();
    };
    return buffer;
  }
}

export class Qwen35WebGpuSuffixRunner {
  readonly firstGpuLayer: number;

  private readonly states = new WeakMap<object, GpuState>();
  private readonly arena: GpuMemoryArena;
  private readonly manifest: Qwen35ModelManifest;
  private readonly epsilon: number;
  private readonly layers: GpuLayer[];
  private readonly outputNorm: F32Handle;
  private readonly outputStripes: OutputStripe[];

  private constructor(
    arena: GpuMemoryArena,
    manifest: Qwen35ModelManifest,
    epsilon: number,
    layers: GpuLayer[],
    outputNorm: F32Handle,
    outputStripes: OutputStripe[],
    firstGpuLayer: number,
  ) {
    this.arena = arena;
    this.manifest = manifest;
    this.epsilon = epsilon;
    this.layers = layers;
    this.outputNorm = outputNorm;
    this.outputStripes = outputStripes;
    this.firstGpuLayer = firstGpuLayer;
  }

  static async create(options: Qwen35WebGpuSuffixRunnerOptions): Promise<Qwen35WebGpuSuffixRunner> {
    const device = await webGpuDevice();
    if (!device) {
      throw new Error("WebGPU is not available for Qwen3.5 suffix execution.");
    }
    const memoryLimitBytes = options.memoryLimitBytes ?? QWEN35_WEBGPU_MEMORY_LIMIT_BYTES;
    const plan = planQwen35WebGpuHybrid(
      options.tensorReader.metadata,
      options.manifest,
      {
        mode: "enabled",
        browserGate: "passed",
        contextLength: options.contextLength,
        memoryLimitBytes,
      },
    );
    const firstGpuLayer = options.firstGpuLayer ?? plan.firstGpuLayer;
    if (firstGpuLayer === undefined || firstGpuLayer >= options.manifest.blockCount) {
      throw new Error("WebGPU suffix planning selected no layers.");
    }
    if (plan.estimatedResidentBytes > memoryLimitBytes) {
      throw new Error(
        `WebGPU suffix plan exceeds memory cap: ${plan.estimatedResidentBytes} > ${memoryLimitBytes}`,
      );
    }

    const arena = new GpuMemoryArena(device, memoryLimitBytes);
    const layers: GpuLayer[] = [];
    for (let layer = firstGpuLayer; layer < options.manifest.blockCount; layer += 1) {
      layers.push(await loadGpuLayer(arena, options.tensorReader, options.manifest, layer));
    }
    const outputNorm = await loadF32Handle(arena, options.tensorReader, "output_norm.weight");
    const outputStripes = await loadOutputStripes(arena, options.tensorReader, options.manifest);
    return new Qwen35WebGpuSuffixRunner(
      arena,
      options.manifest,
      options.epsilon,
      layers,
      outputNorm,
      outputStripes,
      firstGpuLayer,
    );
  }

  get residentBytes(): number {
    return this.arena.residentBytes;
  }

  async runToken(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: Qwen35WebGpuStateLike,
    options: { computeTopK?: boolean; topK?: number } = {},
  ): Promise<Qwen35WebGpuTokenResult> {
    if (inputHidden.length !== this.manifest.embeddingLength) {
      throw new Error(`WebGPU suffix input shape mismatch: ${inputHidden.length}`);
    }
    const tokenPosition = tokenPositionFromSingleMrope(positions);
    const mropePosition = singleMropePosition(positions);
    const gpuState = this.ensureGpuState(state);
    const cleanup: GpuResource[] = [];
    const resources: Array<{ destroy: () => void }> = [];
    const encoder = this.arena.device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    let current = this.arena.createBuffer(
      "suffix boundary hidden",
      inputHidden.byteLength,
      GPU_STORAGE | GPU_COPY_DST,
    );
    cleanup.push(current);
    this.arena.device.queue.writeBuffer(current, 0, inputHidden);

    for (const layer of this.layers) {
      current = layer.kind === "recurrent"
        ? this.dispatchRecurrentLayer(pass, layer, gpuState, current, cleanup, resources)
        : this.dispatchFullAttentionLayer(
          pass,
          layer,
          gpuState,
          current,
          tokenPosition,
          mropePosition,
          state.contextLength,
          cleanup,
          resources,
        );
    }

    let topBuffer: WebGpuBufferLike | undefined;
    let topReadback: WebGpuBufferLike | undefined;
    const candidateCount = Math.max(1, options.topK ?? 1);
    const candidateByteLength = this.outputStripes.length * candidateCount * 2 * Float32Array.BYTES_PER_ELEMENT;
    if (options.computeTopK) {
      topBuffer = this.dispatchOutputTopK(pass, current, candidateCount, cleanup, resources);
      topReadback = this.arena.device.createBuffer({
        size: candidateByteLength,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
    }

    pass.end();
    if (topBuffer && topReadback) {
      encoder.copyBufferToBuffer(topBuffer, 0, topReadback, 0, candidateByteLength);
    }
    this.arena.device.queue.submit([encoder.finish()]);
    await this.arena.device.queue.onSubmittedWorkDone?.();

    let topTokens: WebGpuTopToken[] | undefined;
    if (topReadback) {
      await topReadback.mapAsync(GPU_MAP_READ);
      const values = new Float32Array(topReadback.getMappedRange()).slice();
      topReadback.unmap();
      topReadback.destroy?.();
      topTokens = mergeTopCandidates(values, candidateCount);
    }

    for (const resource of resources) {
      resource.destroy();
    }
    for (const item of cleanup.reverse()) {
      item.destroy?.();
    }
    return { topTokens };
  }

  async runTokens(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: Qwen35WebGpuStateLike,
    options: { computeTopK?: boolean; topK?: number } = {},
  ): Promise<Qwen35WebGpuTokenResult> {
    const tokenCount = inputHidden.length / this.manifest.embeddingLength;
    if (!Number.isInteger(tokenCount) || tokenCount <= 0) {
      throw new Error(`WebGPU suffix batched input shape mismatch: ${inputHidden.length}`);
    }
    if (positions.length !== tokenCount && positions.length !== tokenCount * 4) {
      throw new Error(`WebGPU suffix batched position shape mismatch: ${positions.length}`);
    }

    const boundary = this.arena.createBuffer(
      "suffix boundary hidden batch",
      inputHidden.byteLength,
      GPU_STORAGE | GPU_COPY_DST,
    );
    this.arena.device.queue.writeBuffer(boundary, 0, inputHidden);
    let topTokens: WebGpuTopToken[] | undefined;
    try {
      for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
        const tokenPositions = tokenPositionsFromBatch(positions, tokenIndex, tokenCount);
        const computeTopK = options.computeTopK === true && tokenIndex === tokenCount - 1;
        const result = await this.runTokenFromBoundary(
          boundary,
          tokenIndex,
          tokenPositions,
          state,
          {
            computeTopK,
            topK: options.topK,
          },
        );
        if (computeTopK) {
          topTokens = result.topTokens;
        }
      }
    } finally {
      boundary.destroy?.();
    }
    return { topTokens };
  }

  private async runTokenFromBoundary(
    boundary: WebGpuBufferLike,
    tokenIndex: number,
    positions: Int32Array,
    state: Qwen35WebGpuStateLike,
    options: { computeTopK?: boolean; topK?: number } = {},
  ): Promise<Qwen35WebGpuTokenResult> {
    const tokenPosition = tokenPositionFromSingleMrope(positions);
    const mropePosition = singleMropePosition(positions);
    const gpuState = this.ensureGpuState(state);
    const cleanup: GpuResource[] = [];
    const resources: Array<{ destroy: () => void }> = [];
    const encoder = this.arena.device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    let current = this.arena.createBuffer(
      "suffix boundary hidden token",
      this.manifest.embeddingLength * Float32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE,
    );
    cleanup.push(current);
    dispatchTokenSlice(
      this.arena.device,
      pass,
      resources,
      boundary,
      current,
      {
        rowSize: this.manifest.embeddingLength,
        rowIndex: tokenIndex,
      },
    );

    for (const layer of this.layers) {
      current = layer.kind === "recurrent"
        ? this.dispatchRecurrentLayer(pass, layer, gpuState, current, cleanup, resources)
        : this.dispatchFullAttentionLayer(
          pass,
          layer,
          gpuState,
          current,
          tokenPosition,
          mropePosition,
          state.contextLength,
          cleanup,
          resources,
        );
    }

    let topBuffer: WebGpuBufferLike | undefined;
    let topReadback: WebGpuBufferLike | undefined;
    const candidateCount = Math.max(1, options.topK ?? 1);
    const candidateByteLength = this.outputStripes.length * candidateCount * 2 * Float32Array.BYTES_PER_ELEMENT;
    if (options.computeTopK) {
      topBuffer = this.dispatchOutputTopK(pass, current, candidateCount, cleanup, resources);
      topReadback = this.arena.device.createBuffer({
        size: candidateByteLength,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
    }

    pass.end();
    if (topBuffer && topReadback) {
      encoder.copyBufferToBuffer(topBuffer, 0, topReadback, 0, candidateByteLength);
    }
    this.arena.device.queue.submit([encoder.finish()]);
    await this.arena.device.queue.onSubmittedWorkDone?.();

    let topTokens: WebGpuTopToken[] | undefined;
    if (topReadback) {
      await topReadback.mapAsync(GPU_MAP_READ);
      const values = new Float32Array(topReadback.getMappedRange()).slice();
      topReadback.unmap();
      topReadback.destroy?.();
      topTokens = mergeTopCandidates(values, candidateCount);
    }

    for (const resource of resources) {
      resource.destroy();
    }
    for (const item of cleanup.reverse()) {
      item.destroy?.();
    }
    return { topTokens };
  }

  private ensureGpuState(state: Qwen35WebGpuStateLike): GpuState {
    if (state.contextLength !== this.manifest.contextLength && state.contextLength <= 0) {
      throw new Error(`Invalid WebGPU state context length: ${state.contextLength}`);
    }
    const key = state as object;
    const existing = this.states.get(key);
    if (existing) {
      return existing;
    }
    if (state.nextPosition !== 0) {
      throw new Error(
        "WebGPU suffix state is missing for a non-empty chat state; replay from position 0 is required.",
      );
    }
    const recurrent = new Map<number, RecurrentGpuLayerState>();
    const fullAttention = new Map<number, FullAttentionGpuLayerState>();
    const convDim = this.manifest.ssm.stateSize * this.manifest.ssm.groupCount * 2 +
      this.manifest.ssm.stateSize * this.manifest.ssm.timeStepRank;
    const recurrentStateSize =
      this.manifest.ssm.stateSize * this.manifest.ssm.stateSize * this.manifest.ssm.timeStepRank;
    const fullCacheSize = state.contextLength * this.manifest.headCountKv * this.manifest.keyLength;
    for (const layer of this.layers) {
      if (layer.kind === "recurrent") {
        recurrent.set(layer.layer, {
          conv: this.arena.createBuffer(
            `blk.${layer.layer}.gpu.conv_state`,
            (this.manifest.ssm.convKernel - 1) * convDim * Float32Array.BYTES_PER_ELEMENT,
            GPU_STORAGE,
          ),
          recurrent: this.arena.createBuffer(
            `blk.${layer.layer}.gpu.recurrent_state`,
            recurrentStateSize * Float32Array.BYTES_PER_ELEMENT,
            GPU_STORAGE,
          ),
        });
      } else {
        fullAttention.set(layer.layer, {
          key: this.arena.createBuffer(
            `blk.${layer.layer}.gpu.key_cache`,
            fullCacheSize * Float32Array.BYTES_PER_ELEMENT,
            GPU_STORAGE,
          ),
          value: this.arena.createBuffer(
            `blk.${layer.layer}.gpu.value_cache`,
            fullCacheSize * Float32Array.BYTES_PER_ELEMENT,
            GPU_STORAGE,
          ),
        });
      }
    }
    const created = { recurrent, fullAttention };
    this.states.set(key, created);
    return created;
  }

  private dispatchRecurrentLayer(
    pass: WebGpuComputePassLike,
    layer: RecurrentGpuLayer,
    gpuState: GpuState,
    input: WebGpuBufferLike,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const state = gpuState.recurrent.get(layer.layer);
    if (!state) {
      throw new Error(`Missing WebGPU recurrent state for layer ${layer.layer}`);
    }
    const hiddenSize = this.manifest.embeddingLength;
    const convDim = this.manifest.ssm.stateSize * this.manifest.ssm.groupCount * 2 +
      this.manifest.ssm.stateSize * this.manifest.ssm.timeStepRank;
    const valueDim = this.manifest.ssm.stateSize * this.manifest.ssm.timeStepRank;
    const keyDim = this.manifest.ssm.stateSize * this.manifest.ssm.groupCount;
    const tokenCount = 1;

    const attnNorm = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.attn_norm`);
    dispatchRmsNorm(this.arena.device, pass, resources, input, layer.attnNorm.buffer, attnNorm, hiddenSize, this.epsilon);
    const q8 = scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.attn_norm.q8k`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, attnNorm, q8, hiddenSize, tokenCount);

    const qkv = scratchF32(this.arena, convDim, cleanup, `blk.${layer.layer}.qkv`);
    const z = scratchF32(this.arena, valueDim, cleanup, `blk.${layer.layer}.z`);
    const alpha = scratchF32(this.arena, this.manifest.ssm.timeStepRank, cleanup, `blk.${layer.layer}.alpha`);
    const beta = scratchF32(this.arena, this.manifest.ssm.timeStepRank, cleanup, `blk.${layer.layer}.beta`);
    dispatchKMatMul(pass, resources, layer.qkv, q8, qkv, tokenCount);
    dispatchKMatMul(pass, resources, layer.z, q8, z, tokenCount);
    dispatchF32MatMul(this.arena.device, pass, resources, layer.alpha.buffer, attnNorm, alpha, hiddenSize, this.manifest.ssm.timeStepRank, tokenCount);
    dispatchF32MatMul(this.arena.device, pass, resources, layer.beta.buffer, attnNorm, beta, hiddenSize, this.manifest.ssm.timeStepRank, tokenCount);

    const gate = scratchF32(this.arena, this.manifest.ssm.timeStepRank, cleanup, `blk.${layer.layer}.gate`);
    const betaSigmoid = scratchF32(this.arena, this.manifest.ssm.timeStepRank, cleanup, `blk.${layer.layer}.beta_sigmoid`);
    const q = scratchF32(this.arena, keyDim, cleanup, `blk.${layer.layer}.q_conv`);
    const k = scratchF32(this.arena, keyDim, cleanup, `blk.${layer.layer}.k_conv`);
    const v = scratchF32(this.arena, valueDim, cleanup, `blk.${layer.layer}.v_conv`);
    const nextConv = this.arena.createBuffer(
      `blk.${layer.layer}.next_conv`,
      (this.manifest.ssm.convKernel - 1) * convDim * Float32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE,
    );
    const delta = scratchF32(this.arena, valueDim, cleanup, `blk.${layer.layer}.delta`);
    const nextRecurrent = this.arena.createBuffer(
      `blk.${layer.layer}.next_recurrent`,
      this.manifest.ssm.timeStepRank * this.manifest.ssm.stateSize * this.manifest.ssm.stateSize * Float32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE,
    );
    const gated = scratchF32(this.arena, valueDim, cleanup, `blk.${layer.layer}.ssm_gated`);
    const attention = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.attention`);

    const convResources = createQkvConvResources(
      this.arena.device,
      qkv,
      state.conv,
      layer.convKernel.buffer,
      q,
      k,
      v,
      nextConv,
      {
        tokenCount,
        convDim,
        kernelSize: this.manifest.ssm.convKernel,
        stateSize: this.manifest.ssm.stateSize,
        groupCount: this.manifest.ssm.groupCount,
        valueDim,
      },
    );
    resources.push(convResources);
    pass.setPipeline(convResources.pipeline);
    pass.setBindGroup(0, convResources.bindGroup);
    pass.dispatchWorkgroups(Math.max(this.manifest.ssm.groupCount, this.manifest.ssm.timeStepRank, convDim), 3);

    const gateResources = createDeltaGateResources(
      this.arena.device,
      alpha,
      beta,
      layer.dtBias.buffer,
      layer.ssmA.buffer,
      gate,
      betaSigmoid,
      this.manifest.ssm.timeStepRank,
      tokenCount,
    );
    resources.push(gateResources);
    pass.setPipeline(gateResources.pipeline);
    pass.setBindGroup(0, gateResources.bindGroup);
    pass.dispatchWorkgroups(this.manifest.ssm.timeStepRank, tokenCount);

    const deltaResources = createGatedDeltaNetResources(
      this.arena.device,
      q,
      k,
      v,
      gate,
      betaSigmoid,
      state.recurrent,
      delta,
      nextRecurrent,
      {
        stateSize: this.manifest.ssm.stateSize,
        keyHeadCount: this.manifest.ssm.groupCount,
        valueHeadCount: this.manifest.ssm.timeStepRank,
        tokenCount,
      },
    );
    resources.push(deltaResources);
    pass.setPipeline(deltaResources.pipeline);
    pass.setBindGroup(0, deltaResources.bindGroup);
    pass.dispatchWorkgroups(this.manifest.ssm.timeStepRank);

    const normGateResources = createSsmNormGateResources(
      this.arena.device,
      delta,
      z,
      layer.ssmNorm.buffer,
      gated,
      this.manifest.ssm.stateSize,
      this.manifest.ssm.timeStepRank,
      this.epsilon,
    );
    resources.push(normGateResources);
    pass.setPipeline(normGateResources.pipeline);
    pass.setBindGroup(0, normGateResources.bindGroup);
    pass.dispatchWorkgroups(this.manifest.ssm.timeStepRank);

    const outQ8 = scratchQ8_0(this.arena, valueDim, tokenCount, layer.out.blockCount, cleanup, `blk.${layer.layer}.ssm_gated.q8_0`);
    dispatchQ8_0Quantize(this.arena.device, pass, resources, gated, outQ8, valueDim, tokenCount, layer.out.blockCount);
    dispatchQ8_0MatMul(pass, resources, layer.out, outQ8, attention, tokenCount);

    cleanup.push(state.conv, state.recurrent);
    state.conv = nextConv;
    state.recurrent = nextRecurrent;
    return this.dispatchFfn(pass, layer, input, attention, cleanup, resources);
  }

  private dispatchFullAttentionLayer(
    pass: WebGpuComputePassLike,
    layer: FullAttentionGpuLayer,
    gpuState: GpuState,
    input: WebGpuBufferLike,
    tokenPosition: number,
    mropePosition: Int32Array,
    contextLength: number,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const state = gpuState.fullAttention.get(layer.layer);
    if (!state) {
      throw new Error(`Missing WebGPU full-attention state for layer ${layer.layer}`);
    }
    const hiddenSize = this.manifest.embeddingLength;
    const fullQueryDim = this.manifest.headCount * this.manifest.keyLength * 2;
    const fullKeyValueDim = this.manifest.headCountKv * this.manifest.keyLength;
    const keyValueTokenCount = Math.min(contextLength, tokenPosition + 1);
    const tokenCount = 1;

    const attnNorm = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.attn_norm`);
    dispatchRmsNorm(this.arena.device, pass, resources, input, layer.attnNorm.buffer, attnNorm, hiddenSize, this.epsilon);
    const q8 = scratchQ8K(this.arena, hiddenSize, tokenCount, cleanup, `blk.${layer.layer}.attn_norm.q8k`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, attnNorm, q8, hiddenSize, tokenCount);

    const qFull = scratchF32(this.arena, fullQueryDim, cleanup, `blk.${layer.layer}.q_full`);
    const kProjection = scratchF32(this.arena, fullKeyValueDim, cleanup, `blk.${layer.layer}.k`);
    const vProjection = scratchF32(this.arena, fullKeyValueDim, cleanup, `blk.${layer.layer}.v`);
    dispatchKMatMul(pass, resources, layer.q, q8, qFull, tokenCount);
    dispatchKMatMul(pass, resources, layer.k, q8, kProjection, tokenCount);
    dispatchKMatMul(pass, resources, layer.v, q8, vProjection, tokenCount);

    const query = scratchF32(this.arena, this.manifest.headCount * this.manifest.keyLength, cleanup, `blk.${layer.layer}.q_rope`);
    const gate = scratchF32(this.arena, this.manifest.headCount * this.manifest.keyLength, cleanup, `blk.${layer.layer}.gate`);
    dispatchFullQuery(
      this.arena.device,
      pass,
      resources,
      qFull,
      layer.qNorm.buffer,
      query,
      gate,
      {
        headCount: this.manifest.headCount,
        headSize: this.manifest.keyLength,
        ropeDims: this.manifest.rope.dimensionCount,
        epsilon: this.epsilon,
        freqBase: this.manifest.rope.freqBase,
        position: mropePosition[0] ?? tokenPosition,
      },
    );
    dispatchFullKvUpdate(
      this.arena.device,
      pass,
      resources,
      kProjection,
      vProjection,
      layer.kNorm.buffer,
      state.key,
      state.value,
      {
        headCount: this.manifest.headCountKv,
        headSize: this.manifest.keyLength,
        ropeDims: this.manifest.rope.dimensionCount,
        epsilon: this.epsilon,
        freqBase: this.manifest.rope.freqBase,
        position: mropePosition[0] ?? tokenPosition,
        tokenPosition,
        contextLength,
      },
    );

    const gated = scratchF32(this.arena, this.manifest.headCount * this.manifest.keyLength, cleanup, `blk.${layer.layer}.attention_gated`);
    const probabilities = scratchF32(
      this.arena,
      this.manifest.headCount * keyValueTokenCount,
      cleanup,
      `blk.${layer.layer}.attention_probabilities`,
    );
    const attentionOptions = {
      headSize: this.manifest.keyLength,
      queryHeadCount: this.manifest.headCount,
      keyValueHeadCount: this.manifest.headCountKv,
      keyValueTokenCount,
      contextLength,
      scale: 1 / Math.sqrt(this.manifest.keyLength),
    };
    const scoreResources = createFullAttentionScoreResources(
      this.arena.device,
      query,
      state.key,
      probabilities,
      attentionOptions,
    );
    resources.push(scoreResources);
    pass.setPipeline(scoreResources.pipeline);
    pass.setBindGroup(0, scoreResources.bindGroup);
    pass.dispatchWorkgroups(this.manifest.headCount);
    const applyResources = createFullAttentionApplyResources(
      this.arena.device,
      state.value,
      gate,
      probabilities,
      gated,
      attentionOptions,
    );
    resources.push(applyResources);
    pass.setPipeline(applyResources.pipeline);
    pass.setBindGroup(0, applyResources.bindGroup);
    pass.dispatchWorkgroups(this.manifest.headCount, this.manifest.keyLength);

    const gatedQ8 = scratchQ8K(this.arena, this.manifest.headCount * this.manifest.keyLength, tokenCount, cleanup, `blk.${layer.layer}.attention_gated.q8k`);
    const attention = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.attention_out`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, gated, gatedQ8, this.manifest.headCount * this.manifest.keyLength, tokenCount);
    dispatchKMatMul(pass, resources, layer.out, gatedQ8, attention, tokenCount);
    return this.dispatchFfn(pass, layer, input, attention, cleanup, resources);
  }

  private dispatchFfn(
    pass: WebGpuComputePassLike,
    layer: GpuLayer,
    residualInput: WebGpuBufferLike,
    attention: WebGpuBufferLike,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const hiddenSize = this.manifest.embeddingLength;
    const residual = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.residual`);
    dispatchResidualAdd(this.arena.device, pass, resources, residualInput, attention, residual, hiddenSize);
    const postNorm = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.post_norm`);
    dispatchRmsNorm(this.arena.device, pass, resources, residual, layer.postNorm.buffer, postNorm, hiddenSize, this.epsilon);
    const postQ8 = scratchQ8K(this.arena, hiddenSize, 1, cleanup, `blk.${layer.layer}.post_norm.q8k`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, postNorm, postQ8, hiddenSize, 1);

    const gate = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_gate`);
    const up = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_up`);
    const swiglu = scratchF32(this.arena, this.manifest.feedForwardLength, cleanup, `blk.${layer.layer}.ffn_swiglu`);
    dispatchKMatMul(pass, resources, layer.ffnGate, postQ8, gate, 1);
    dispatchKMatMul(pass, resources, layer.ffnUp, postQ8, up, 1);

    const swigluResources = createSwiGluResources(this.arena.device, gate, up, swiglu, this.manifest.feedForwardLength);
    resources.push(swigluResources);
    pass.setPipeline(swigluResources.pipeline);
    pass.setBindGroup(0, swigluResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.manifest.feedForwardLength / 256));

    const swigluQ8 = scratchQ8K(this.arena, this.manifest.feedForwardLength, 1, cleanup, `blk.${layer.layer}.ffn_swiglu.q8k`);
    const ffnOut = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.ffn_out`);
    dispatchQ8KQuantize(this.arena.device, pass, resources, swiglu, swigluQ8, this.manifest.feedForwardLength, 1);
    dispatchKMatMul(pass, resources, layer.ffnDown, swigluQ8, ffnOut, 1);

    const output = scratchF32(this.arena, hiddenSize, cleanup, `blk.${layer.layer}.output`);
    dispatchResidualAdd(this.arena.device, pass, resources, residual, ffnOut, output, hiddenSize);
    return output;
  }

  private dispatchOutputTopK(
    pass: WebGpuComputePassLike,
    hidden: WebGpuBufferLike,
    topKCount: number,
    cleanup: GpuResource[],
    resources: Array<{ destroy: () => void }>,
  ): WebGpuBufferLike {
    const hiddenSize = this.manifest.embeddingLength;
    const norm = scratchF32(this.arena, hiddenSize, cleanup, "output_norm");
    dispatchRmsNorm(this.arena.device, pass, resources, hidden, this.outputNorm.buffer, norm, hiddenSize, this.epsilon);
    const q8 = scratchQ8K(this.arena, hiddenSize, 1, cleanup, "output_norm.q8k");
    dispatchQ8KQuantize(this.arena.device, pass, resources, norm, q8, hiddenSize, 1);

    const candidates = this.arena.createBuffer(
      "output.topk.candidates",
      this.outputStripes.length * topKCount * 2 * Float32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE | GPU_COPY_SRC,
    );
    cleanup.push(candidates);
    for (let index = 0; index < this.outputStripes.length; index += 1) {
      const stripe = this.outputStripes[index];
      if (!stripe) {
        continue;
      }
      const logits = scratchF32(this.arena, stripe.rowCount, cleanup, `output.logits.${index}`);
      dispatchKMatMul(pass, resources, stripe, q8, logits, 1);
      dispatchTopK(
        this.arena.device,
        pass,
        resources,
        logits,
        candidates,
        {
          rowCount: stripe.rowCount,
          rowOffset: stripe.rowOffset,
          topK: topKCount,
          candidateOffset: index * topKCount * 2,
        },
      );
    }
    return candidates;
  }
}

async function loadGpuLayer(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
  manifest: Qwen35ModelManifest,
  layer: number,
): Promise<GpuLayer> {
  const ffn = {
    postNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.post_attention_norm.weight`),
    ffnGate: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.ffn_gate.weight`),
    ffnUp: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.ffn_up.weight`),
    ffnDown: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.ffn_down.weight`),
  };
  if (manifest.fullAttentionLayers.includes(layer)) {
    return {
      kind: "full-attention",
      layer,
      attnNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.attn_norm.weight`),
      q: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_q.weight`),
      k: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_k.weight`),
      v: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_v.weight`),
      out: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_output.weight`),
      qNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.attn_q_norm.weight`),
      kNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.attn_k_norm.weight`),
      ...ffn,
    };
  }
  return {
    kind: "recurrent",
    layer,
    attnNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.attn_norm.weight`),
    qkv: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_qkv.weight`),
    alpha: await loadF32Handle(arena, tensorReader, `blk.${layer}.ssm_alpha.weight`),
    beta: await loadF32Handle(arena, tensorReader, `blk.${layer}.ssm_beta.weight`),
    z: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_gate.weight`),
    convKernel: await loadF32Handle(arena, tensorReader, `blk.${layer}.ssm_conv1d.weight`),
    dtBias: await loadF32Handle(arena, tensorReader, `blk.${layer}.ssm_dt.bias`),
    ssmA: await loadF32Handle(arena, tensorReader, `blk.${layer}.ssm_a`),
    ssmNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.ssm_norm.weight`),
    out: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.ssm_out.weight`),
    ...ffn,
  };
}

async function loadF32Handle(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
  name: string,
): Promise<F32Handle> {
  const tensor = tensorReader.getTensor(name);
  if (tensor.type !== "F32") {
    throw new Error(`${name} must be F32 for WebGPU suffix, got ${tensor.type}`);
  }
  const bytes = await tensorReader.readTensorBytes(name);
  const buffer = arena.createBuffer(name, bytes.byteLength, GPU_STORAGE | GPU_COPY_DST);
  arena.device.queue.writeBuffer(buffer, 0, bytes);
  return {
    length: tensor.dimensions.reduce((product, dimension) => product * dimension, 1),
    byteLength: bytes.byteLength,
    device: arena.device,
    buffer,
    destroy: () => buffer.destroy?.(),
  };
}

async function loadQuantizedHandle(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
  name: string,
): Promise<QuantizedHandle> {
  const tensor = tensorReader.getTensor(name);
  return createQuantizedHandleFromBytes(
    arena,
    name,
    webGpuMatMulType(tensor.type, name),
    tensor.dimensions[0] ?? 0,
    tensor.dimensions[1] ?? 0,
    await tensorReader.readTensorBytes(name),
  );
}

async function loadOutputStripes(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
  manifest: Qwen35ModelManifest,
): Promise<OutputStripe[]> {
  const tensor = tensorReader.getTensor("output.weight");
  const type = webGpuMatMulType(tensor.type, "output.weight");
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  if (inputSize !== manifest.embeddingLength) {
    throw new Error(`output.weight input mismatch: ${inputSize}`);
  }
  const layout = webGpuQuantizedWeightLayout(type, inputSize);
  const limits = await webGpuAdapterLimits();
  const bindingLimit = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const targetBytes = Math.max(layout.rowByteLength, Math.min(bindingLimit, 128 * 1024 * 1024));
  const rowsPerStripe = Math.max(1, Math.floor(targetBytes / layout.rowByteLength));
  const stripes: OutputStripe[] = [];
  for (let rowOffset = 0; rowOffset < rowCount; rowOffset += rowsPerStripe) {
    const stripeRows = Math.min(rowsPerStripe, rowCount - rowOffset);
    const bytes = await tensorReader.readTensorRange({
      tensor,
      offset: BigInt(rowOffset * layout.rowByteLength),
      length: stripeRows * layout.rowByteLength,
    });
    stripes.push({
      ...createQuantizedHandleFromBytes(
        arena,
        `output.weight.${rowOffset}`,
        type,
        inputSize,
        stripeRows,
        bytes,
      ),
      rowOffset,
    });
  }
  return stripes;
}

function createQuantizedHandleFromBytes(
  arena: GpuMemoryArena,
  label: string,
  type: WebGpuQuantizedMatMulType,
  inputSize: number,
  rowCount: number,
  bytes: Uint8Array,
): QuantizedHandle {
  const layout = webGpuQuantizedWeightLayout(type, inputSize);
  const expected = layout.rowByteLength * rowCount;
  if (bytes.byteLength !== expected) {
    throw new Error(`WebGPU ${label} weight shape mismatch: ${bytes.byteLength} !== ${expected}`);
  }
  const packed = packBytesToU32(bytes);
  const weightBuffer = arena.createBuffer(label, packed.byteLength, GPU_STORAGE | GPU_COPY_DST);
  arena.device.queue.writeBuffer(weightBuffer, 0, packed);
  return {
    type,
    inputSize,
    rowCount,
    byteLength: bytes.byteLength,
    device: arena.device,
    weightBuffer,
    blockCount: layout.blockCount,
    rowByteLength: layout.rowByteLength,
    destroy: () => weightBuffer.destroy?.(),
  };
}

function webGpuMatMulType(type: GgmlTypeName, name: string): WebGpuQuantizedMatMulType {
  if (type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "Q8_0") {
    return type;
  }
  throw new Error(`${name} has unsupported WebGPU suffix type ${type}`);
}

type Q8KBuffers = {
  scale: WebGpuBufferLike;
  qs: WebGpuBufferLike;
  bsums: WebGpuBufferLike;
};

type Q8_0Buffers = {
  scale: WebGpuBufferLike;
  qs: WebGpuBufferLike;
};

function scratchF32(
  arena: GpuMemoryArena,
  length: number,
  cleanup: GpuResource[],
  label: string,
): WebGpuBufferLike {
  const buffer = arena.createBuffer(label, length * Float32Array.BYTES_PER_ELEMENT, GPU_STORAGE);
  cleanup.push(buffer);
  return buffer;
}

function scratchQ8K(
  arena: GpuMemoryArena,
  inputSize: number,
  columnCount: number,
  cleanup: GpuResource[],
  label: string,
): Q8KBuffers {
  const blockCount = inputSize / 256;
  const buffers = {
    scale: arena.createBuffer(`${label}.scale`, columnCount * blockCount * Float32Array.BYTES_PER_ELEMENT, GPU_STORAGE),
    qs: arena.createBuffer(`${label}.qs`, columnCount * inputSize * Int32Array.BYTES_PER_ELEMENT, GPU_STORAGE),
    bsums: arena.createBuffer(`${label}.bsums`, columnCount * blockCount * 16 * Int32Array.BYTES_PER_ELEMENT, GPU_STORAGE),
  };
  cleanup.push(buffers.scale, buffers.qs, buffers.bsums);
  return buffers;
}

function scratchQ8_0(
  arena: GpuMemoryArena,
  inputSize: number,
  columnCount: number,
  blockCount: number,
  cleanup: GpuResource[],
  label: string,
): Q8_0Buffers {
  const buffers = {
    scale: arena.createBuffer(`${label}.scale`, columnCount * blockCount * Float32Array.BYTES_PER_ELEMENT, GPU_STORAGE),
    qs: arena.createBuffer(`${label}.qs`, columnCount * inputSize * Int32Array.BYTES_PER_ELEMENT, GPU_STORAGE),
  };
  cleanup.push(buffers.scale, buffers.qs);
  return buffers;
}

function dispatchKMatMul(
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  handle: QuantizedHandle,
  q8: Q8KBuffers,
  output: WebGpuBufferLike,
  columnCount: number,
): void {
  if (handle.type === "Q8_0") {
    throw new Error("Q8_0 handle cannot use K-quant matmul dispatch");
  }
  const resource = createKMatMulBindResources(handle, q8.scale, q8.qs, q8.bsums, output, columnCount);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(handle.rowCount / 8), columnCount);
}

function dispatchQ8_0MatMul(
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  handle: QuantizedHandle,
  q8: Q8_0Buffers,
  output: WebGpuBufferLike,
  columnCount: number,
): void {
  if (handle.type !== "Q8_0") {
    throw new Error("K-quant handle cannot use Q8_0 matmul dispatch");
  }
  const resource = createQ8_0MatMulBindResources(handle, q8.scale, q8.qs, output, columnCount);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(handle.rowCount / 8), columnCount);
}

function dispatchF32MatMul(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  weight: WebGpuBufferLike,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): void {
  const resource = createF32MatMulResources(device, weight, input, output, inputSize, rowCount, columnCount);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
}

function dispatchQ8KQuantize(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  q8: Q8KBuffers,
  inputSize: number,
  columnCount: number,
): void {
  const resource = createQ8KQuantizeResources(device, input, q8.scale, q8.qs, q8.bsums, inputSize, columnCount, inputSize / 256);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(columnCount, inputSize / 256);
}

function dispatchQ8_0Quantize(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  q8: Q8_0Buffers,
  inputSize: number,
  columnCount: number,
  blockCount: number,
): void {
  const resource = createQ8_0QuantizeResources(device, input, q8.scale, q8.qs, inputSize, columnCount, blockCount);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(columnCount, blockCount);
}

function dispatchRmsNorm(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
  epsilon: number,
): void {
  const resource = createRmsNormResources(device, input, weight, output, length, epsilon);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(1);
}

function dispatchResidualAdd(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  left: WebGpuBufferLike,
  right: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
): void {
  const resource = createResidualAddResources(device, left, right, output, length);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(length / 256));
}

function dispatchFullQuery(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  qFull: WebGpuBufferLike,
  qNorm: WebGpuBufferLike,
  query: WebGpuBufferLike,
  gate: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    epsilon: number;
    freqBase: number;
    position: number;
  },
): void {
  const resource = createFullQueryResources(device, qFull, qNorm, query, gate, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.headCount);
}

function dispatchFullKvUpdate(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  kProjection: WebGpuBufferLike,
  vProjection: WebGpuBufferLike,
  kNorm: WebGpuBufferLike,
  keyCache: WebGpuBufferLike,
  valueCache: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    epsilon: number;
    freqBase: number;
    position: number;
    tokenPosition: number;
    contextLength: number;
  },
): void {
  const resource = createFullKvUpdateResources(device, kProjection, vProjection, kNorm, keyCache, valueCache, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.headCount);
}

function dispatchTopK(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  logits: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowCount: number;
    rowOffset: number;
    topK: number;
    candidateOffset: number;
  },
): void {
  for (let slot = 0; slot < options.topK; slot += 1) {
    const resource = createTopKResources(device, logits, output, { ...options, slot });
    resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(1);
  }
}

function dispatchTokenSlice(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowSize: number;
    rowIndex: number;
  },
): void {
  const resource = createTokenSliceResources(device, input, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(options.rowSize / 256));
}

function createRmsNormResources(
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

function createResidualAddResources(
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

function createFullQueryResources(
  device: WebGpuDeviceLike,
  qFull: WebGpuBufferLike,
  qNorm: WebGpuBufferLike,
  query: WebGpuBufferLike,
  gate: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    epsilon: number;
    freqBase: number;
    position: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.epsilon, options.freqBase, options.position, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[3] = options.headCount;
  const more = new Uint32Array([options.headSize, options.ropeDims, 0, 0]);
  const params = new Uint32Array(8);
  params.set(paramsU32, 0);
  params.set(more, 4);
  const paramsBuffer = device.createBuffer({ size: params.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      { binding: 2, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(3, "storage"),
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
      entries: [bindBuffer(0, qFull), bindBuffer(1, qNorm), bindBuffer(2, paramsBuffer), bindBuffer(3, query), bindBuffer(4, gate)],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

function createFullKvUpdateResources(
  device: WebGpuDeviceLike,
  kProjection: WebGpuBufferLike,
  vProjection: WebGpuBufferLike,
  kNorm: WebGpuBufferLike,
  keyCache: WebGpuBufferLike,
  valueCache: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    epsilon: number;
    freqBase: number;
    position: number;
    tokenPosition: number;
    contextLength: number;
  },
): { pipeline: unknown; bindGroup: unknown; destroy: () => void } {
  const paramsF32 = new Float32Array([options.epsilon, options.freqBase, options.position, 0]);
  const paramsU32 = new Uint32Array(paramsF32.buffer);
  paramsU32[3] = options.headCount;
  const more = new Uint32Array([options.headSize, options.ropeDims, options.tokenPosition, options.contextLength]);
  const params = new Uint32Array(8);
  params.set(paramsU32, 0);
  params.set(more, 4);
  const paramsBuffer = device.createBuffer({ size: params.byteLength, usage: GPU_UNIFORM | GPU_COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      { binding: 3, visibility: GPU_SHADER_STAGE_COMPUTE, buffer: { type: "uniform" } },
      storageEntry(4, "storage"),
      storageEntry(5, "storage"),
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
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, keyCache),
        bindBuffer(5, valueCache),
      ],
    }),
    destroy: () => paramsBuffer.destroy?.(),
  };
}

function createTopKResources(
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

function createTokenSliceResources(
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

function tokenPositionFromSingleMrope(positions: Int32Array): number {
  if (positions.length === 1 || positions.length === 4) {
    return positions[0] ?? 0;
  }
  throw new Error(`WebGPU token path expects one position, got ${positions.length}`);
}

function singleMropePosition(positions: Int32Array): Int32Array {
  if (positions.length === 4) {
    return positions;
  }
  if (positions.length === 1) {
    const position = positions[0] ?? 0;
    return new Int32Array([position, position, position, position]);
  }
  throw new Error(`WebGPU token path expects one M-RoPE position, got ${positions.length}`);
}

function tokenPositionsFromBatch(positions: Int32Array, tokenIndex: number, tokenCount: number): Int32Array {
  if (positions.length === tokenCount) {
    return new Int32Array([positions[tokenIndex] ?? 0]);
  }
  if (positions.length === tokenCount * 4) {
    return new Int32Array([
      positions[tokenIndex] ?? 0,
      positions[tokenIndex + tokenCount] ?? 0,
      positions[tokenIndex + tokenCount * 2] ?? 0,
      positions[tokenIndex + tokenCount * 3] ?? 0,
    ]);
  }
  throw new Error(`WebGPU token batch expects ${tokenCount} or ${tokenCount * 4} positions, got ${positions.length}`);
}

function mergeTopCandidates(values: Float32Array, topKCount: number): WebGpuTopToken[] {
  const best: WebGpuTopToken[] = [];
  for (let index = 0; index < values.length; index += 2) {
    const id = Math.trunc(values[index] ?? 0);
    const value = values[index + 1] ?? -Infinity;
    if (!Number.isFinite(value)) {
      continue;
    }
    best.push({ id, value });
    best.sort((left, right) => right.value - left.value);
    if (best.length > topKCount) {
      best.pop();
    }
  }
  return best;
}

function align4(value: number): number {
  return Math.max(4, Math.ceil(value / 4) * 4);
}

const TOKEN_SLICE_WGSL = `
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

const RMS_NORM_WGSL = `
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

const RESIDUAL_ADD_WGSL = `
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

const FULL_QUERY_NORM_ROPE_WGSL = `
struct Params {
  epsilon: f32,
  freqBase: f32,
  position: f32,
  headCount: u32,
  headSize: u32,
  ropeDims: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> qFullValues: array<f32>;
@group(0) @binding(1) var<storage, read> normWeights: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> queryValues: array<f32>;
@group(0) @binding(4) var<storage, read_write> gateValues: array<f32>;

fn normed(head: u32, dim: u32, scale: f32) -> f32 {
  let qFullBase = head * params.headSize * 2u;
  return qFullValues[qFullBase + dim] * scale * normWeights[dim];
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let head = id.x;
  if (head >= params.headCount) {
    return;
  }
  let qFullBase = head * params.headSize * 2u;
  let outBase = head * params.headSize;
  var meanSquare = 0.0;
  for (var dim = 0u; dim < params.headSize; dim = dim + 1u) {
    let value = qFullValues[qFullBase + dim];
    meanSquare = meanSquare + value * value;
    gateValues[outBase + dim] = qFullValues[qFullBase + params.headSize + dim];
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
    let cosTheta = cos(theta);
    let sinTheta = sin(theta);
    queryValues[outBase + ic] = x0 * cosTheta - x1 * sinTheta;
    queryValues[outBase + params.ropeDims / 2u + ic] = x0 * sinTheta + x1 * cosTheta;
    theta = theta * thetaScale;
  }
}
`;

const FULL_KV_UPDATE_WGSL = `
struct Params {
  epsilon: f32,
  freqBase: f32,
  position: f32,
  headCount: u32,
  headSize: u32,
  ropeDims: u32,
  tokenPosition: u32,
  contextLength: u32,
};

@group(0) @binding(0) var<storage, read> kProjectionValues: array<f32>;
@group(0) @binding(1) var<storage, read> vProjectionValues: array<f32>;
@group(0) @binding(2) var<storage, read> normWeights: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> keyCache: array<f32>;
@group(0) @binding(5) var<storage, read_write> valueCache: array<f32>;

fn normed(head: u32, dim: u32, scale: f32) -> f32 {
  let base = head * params.headSize;
  return kProjectionValues[base + dim] * scale * normWeights[dim];
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
    valueCache[(dim * params.headCount + head) * params.contextLength + params.tokenPosition] =
      vProjectionValues[base + dim];
  }
  let thetaScale = pow(params.freqBase, -2.0 / f32(params.ropeDims));
  var theta = params.position;
  for (var i0 = 0u; i0 < params.ropeDims; i0 = i0 + 2u) {
    let ic = i0 / 2u;
    let x0 = normed(head, ic, scale);
    let x1 = normed(head, params.ropeDims / 2u + ic, scale);
    let cosTheta = cos(theta);
    let sinTheta = sin(theta);
    keyCache[keyBase + ic] = x0 * cosTheta - x1 * sinTheta;
    keyCache[keyBase + params.ropeDims / 2u + ic] = x0 * sinTheta + x1 * cosTheta;
    theta = theta * thetaScale;
  }
}
`;

const TOPK_WGSL = `
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

const Q8_0_MATMUL_WGSL = `
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

const SWIGLU_WGSL = `
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

const F32_MATMUL_WGSL = `
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

const DELTA_GATE_WGSL = `
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

const TOP1_WGSL = `
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

const Q8_K_QUANTIZE_WGSL = `
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

const Q8_0_QUANTIZE_WGSL = `
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

const SSM_NORM_GATE_WGSL = `
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

const QKV_CONV_SPLIT_WGSL = `
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

const GATED_DELTA_NET_WGSL = `
struct Params {
  stateSize: u32,
  keyHeadCount: u32,
  valueHeadCount: u32,
  tokenCount: u32,
};

@group(0) @binding(0) var<storage, read> queryValues: array<f32>;
@group(0) @binding(1) var<storage, read> keyValues: array<f32>;
@group(0) @binding(2) var<storage, read> valueValues: array<f32>;
@group(0) @binding(3) var<storage, read> gateValues: array<f32>;
@group(0) @binding(4) var<storage, read> betaValues: array<f32>;
@group(0) @binding(5) var<storage, read> stateValues: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(8) var<storage, read_write> newStateValues: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let valueHead = id.x;
  if (valueHead >= params.valueHeadCount) {
    return;
  }
  let stateArea = params.stateSize * params.stateSize;
  let stateOffset = valueHead * stateArea;
  let keyHead = valueHead % params.keyHeadCount;
  let scale = inverseSqrt(f32(params.stateSize));

  for (var index = 0u; index < stateArea; index = index + 1u) {
    newStateValues[stateOffset + index] = stateValues[stateOffset + index];
  }

  for (var token = 0u; token < params.tokenCount; token = token + 1u) {
    let qOffset = (token * params.keyHeadCount + keyHead) * params.stateSize;
    let kOffset = qOffset;
    let vOffset = (token * params.valueHeadCount + valueHead) * params.stateSize;
    let headOffset = token * params.valueHeadCount + valueHead;
    let expGate = exp(gateValues[headOffset]);
    let beta = betaValues[headOffset];

    for (var row = 0u; row < params.stateSize; row = row + 1u) {
      let rowOffset = stateOffset + row * params.stateSize;
      var sum = 0.0;
      for (var column = 0u; column < params.stateSize; column = column + 1u) {
        let stateIndex = rowOffset + column;
        let decayed = newStateValues[stateIndex] * expGate;
        newStateValues[stateIndex] = decayed;
        sum = sum + decayed * keyValues[kOffset + column];
      }
      let delta = (valueValues[vOffset + row] - sum) * beta;
      for (var column = 0u; column < params.stateSize; column = column + 1u) {
        let stateIndex = rowOffset + column;
        newStateValues[stateIndex] = newStateValues[stateIndex] + keyValues[kOffset + column] * delta;
      }
    }

    let outputOffset = (token * params.valueHeadCount + valueHead) * params.stateSize;
    for (var row = 0u; row < params.stateSize; row = row + 1u) {
      let rowOffset = stateOffset + row * params.stateSize;
      var sum = 0.0;
      for (var column = 0u; column < params.stateSize; column = column + 1u) {
        sum = sum + newStateValues[rowOffset + column] * queryValues[qOffset + column];
      }
      outputValues[outputOffset + row] = sum * scale;
    }
  }
}
`;

const FULL_ATTENTION_SCORE_WGSL = `
struct Params {
  scale: f32,
  headSize: u32,
  queryHeadCount: u32,
  keyValueHeadCount: u32,
  keyValueTokenCount: u32,
  contextLength: u32,
  _pad0: u32,
  _pad1: u32,
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

const FULL_ATTENTION_APPLY_WGSL = `
struct Params {
  scale: f32,
  headSize: u32,
  queryHeadCount: u32,
  keyValueHeadCount: u32,
  keyValueTokenCount: u32,
  contextLength: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> valueValues: array<f32>;
@group(0) @binding(1) var<storage, read> gateValues: array<f32>;
@group(0) @binding(2) var<storage, read> probabilityValues: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let qHead = id.x;
  let dim = id.y;
  if (qHead >= params.queryHeadCount || dim >= params.headSize) {
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
  let outputIndex = qHead * params.headSize + dim;
  let gate = gateValues[outputIndex];
  outputValues[outputIndex] = weighted * (1.0 / (1.0 + exp(-gate)));
}
`;

const Q4_K_MATMUL_WGSL = `
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

const Q5_K_MATMUL_WGSL = `
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

const Q6_K_MATMUL_WGSL = `
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
