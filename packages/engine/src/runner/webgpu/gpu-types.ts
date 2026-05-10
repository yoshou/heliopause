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

export type NavigatorWithWebGpu = Navigator & {
  gpu?: {
    requestAdapter: () => Promise<WebGpuAdapterLike | null>;
  };
};

export type WebGpuAdapterLike = {
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

export type WebGpuDeviceLike = {
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

export type WebGpuBufferLike = {
  getMappedRange: () => ArrayBuffer;
  unmap: () => void;
  mapAsync: (mode: number) => Promise<void>;
  destroy?: () => void;
};

export type WebGpuCommandEncoderLike = {
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

export type WebGpuComputePassLike = {
  setPipeline: (pipeline: unknown) => void;
  setBindGroup: (index: number, bindGroup: unknown) => void;
  dispatchWorkgroups: (x: number, y?: number, z?: number) => void;
  end: () => void;
};

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

export type WebGpuTopToken = {
  id: number;
  value: number;
};

export type WebGpuQuantizedWeightHandleInternal = WebGpuQuantizedWeightHandle & {
  readonly device: WebGpuDeviceLike;
  readonly weightBuffer: WebGpuBufferLike;
  readonly blockCount: number;
  readonly rowByteLength: number;
};

export type WebGpuF32TensorHandleInternal = WebGpuF32TensorHandle & {
  readonly device: WebGpuDeviceLike;
  readonly buffer: WebGpuBufferLike;
};
