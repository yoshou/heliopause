import { GPU_COPY_DST, GPU_COPY_SRC, GPU_MAP_READ, GPU_SHADER_STAGE_COMPUTE, GPU_STORAGE } from "./gpu-constants";
import { bindBuffer, storageEntry } from "./gpu-bindings";
import type { NavigatorWithWebGpu, WebGpuAdapterLike, WebGpuDeviceLike, WebGpuSmokeTest, WebGpuSupport } from "./gpu-types";

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

export async function webGpuDevice(): Promise<WebGpuDeviceLike | undefined> {
  if (!devicePromise) {
    devicePromise = requestWebGpuDevice();
  }
  return devicePromise;
}

export async function webGpuAdapterLimits(): Promise<Pick<WebGpuAdapterLike, "limits">["limits"] | undefined> {
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

export async function assertStorageBindingFits(label: string, byteLength: number): Promise<void> {
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

function sameNumbers(left: number[], right: number[]): boolean {
  return left.length === right.length &&
    left.every((value, index) => Math.abs(value - (right[index] ?? Number.NaN)) < 1e-6);
}

function nowMs(): number {
  return globalThis.performance?.now() ?? Date.now();
}
