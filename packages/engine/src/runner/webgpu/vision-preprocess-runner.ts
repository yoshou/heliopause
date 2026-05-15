import type {
  ExecutionProviderStats,
} from "../../runtime";
import type {
  VisionPixelValues,
  VisionResize,
  VisionSession,
} from "../../vision";
import { GpuMemoryArena, type GpuResource } from "./arena";
import { GPU_COPY_DST, GPU_COPY_SRC, GPU_MAP_READ, GPU_STORAGE, WEBGPU_MEMORY_LIMIT_BYTES } from "./gpu-constants";
import { webGpuDevice } from "./gpu-device";
import type { WebGpuBufferLike } from "./gpu-types";
import {
  createVisionPreprocessRgbaResources,
} from "./vision-kernel-resources";

type VisionPreprocessStats = {
  attempts: number;
  runs: number;
  fallbacks: number;
  lastFallbackReason: string;
};

type VisionPreprocessSession = Pick<
  VisionSession,
  "manifest" | "executionProvider" | "setExecutionProviderStatsProvider" | "addDisposeCallback"
>;

type VisionPreprocessRunBuffers = {
  resources: Array<{ destroy: () => void }>;
  cleanup: GpuResource[];
};

type VisionRgbaPreprocessInput = {
  rgba: Uint8ClampedArray;
  sourceWidth: number;
  sourceHeight: number;
  resize: VisionResize;
};

const runners = new WeakMap<VisionPreprocessSession, Promise<WebGpuVisionPreprocessRunner | undefined>>();
const statsBySession = new WeakMap<VisionPreprocessSession, VisionPreprocessStats>();

export async function runWebGpuVisionPreprocessor(
  session: VisionPreprocessSession,
  input: VisionRgbaPreprocessInput,
  options: { signal?: AbortSignal } = {},
): Promise<VisionPixelValues | undefined> {
  const stats = visionPreprocessStats(session);
  stats.attempts += 1;
  const runner = await visionPreprocessRunner(session);
  if (!runner) {
    stats.fallbacks += 1;
    stats.lastFallbackReason = "webgpu-unavailable";
    return undefined;
  }
  try {
    const result = await runner.run(input, options);
    stats.runs += 1;
    stats.lastFallbackReason = "";
    return result;
  } catch (error) {
    if (!isFallbackError(error)) {
      throw error;
    }
    stats.fallbacks += 1;
    stats.lastFallbackReason = error instanceof Error ? error.message : String(error);
    return undefined;
  }
}

function visionPreprocessStats(session: VisionPreprocessSession): VisionPreprocessStats {
  let stats = statsBySession.get(session);
  if (!stats) {
    stats = {
      attempts: 0,
      runs: 0,
      fallbacks: 0,
      lastFallbackReason: "",
    };
    statsBySession.set(session, stats);
    const captured = stats;
    session.setExecutionProviderStatsProvider((): ExecutionProviderStats => ({
      webgpuVisionPreprocessAttempts: captured.attempts,
      webgpuVisionPreprocessRuns: captured.runs,
      webgpuVisionPreprocessFallbacks: captured.fallbacks,
      webgpuVisionPreprocessLastFallbackReason: captured.lastFallbackReason,
    }), "webgpu-vision-preprocess");
  }
  return stats;
}

async function visionPreprocessRunner(
  session: VisionPreprocessSession,
): Promise<WebGpuVisionPreprocessRunner | undefined> {
  let runner = runners.get(session);
  if (!runner) {
    runner = createVisionPreprocessRunner(session);
    runners.set(session, runner);
  }
  return runner;
}

async function createVisionPreprocessRunner(
  session: VisionPreprocessSession,
): Promise<WebGpuVisionPreprocessRunner | undefined> {
  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }
  const options = session.executionProvider("webgpu")?.options;
  const arena = new GpuMemoryArena(
    device,
    numberOption(options, "memoryLimitBytes") ?? WEBGPU_MEMORY_LIMIT_BYTES,
  );
  const runner = new WebGpuVisionPreprocessRunner(session, arena);
  session.addDisposeCallback(() => runner.dispose());
  return runner;
}

class WebGpuVisionPreprocessRunner {
  private readonly session: VisionPreprocessSession;
  private readonly arena: GpuMemoryArena;

  constructor(session: VisionPreprocessSession, arena: GpuMemoryArena) {
    this.session = session;
    this.arena = arena;
  }

  dispose(): void {
  }

  async run(
    input: VisionRgbaPreprocessInput,
    options: { signal?: AbortSignal } = {},
  ): Promise<VisionPixelValues> {
    throwIfAborted(options.signal);
    if (input.sourceWidth <= 0 || input.sourceHeight <= 0 || input.resize.width <= 0 || input.resize.height <= 0) {
      throw new WebGpuVisionPreprocessFallbackError("Vision preprocess dimensions must be positive.");
    }
    if (input.rgba.byteLength !== input.sourceWidth * input.sourceHeight * 4) {
      throw new WebGpuVisionPreprocessFallbackError("Vision preprocess RGBA input shape mismatch.");
    }

    const run: VisionPreprocessRunBuffers = { resources: [], cleanup: [] };
    let readback: WebGpuBufferLike | undefined;
    try {
      const outputLength = input.resize.width * input.resize.height * 3;
      const rgba = this.bufferFromRgba("vision.preprocess.rgba", input.rgba, run.cleanup);
      const output = this.scratchF32("vision.preprocess.output", outputLength, run.cleanup);
      const scopedDevice = this.arena.device as typeof this.arena.device & {
        pushErrorScope?: (filter: "validation") => void;
        popErrorScope?: () => Promise<unknown>;
      };
      scopedDevice.pushErrorScope?.("validation");
      const encoder = this.arena.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      const resource = createVisionPreprocessRgbaResources(this.arena.device, rgba, output, {
        sourceWidth: input.sourceWidth,
        sourceHeight: input.sourceHeight,
        targetWidth: input.resize.width,
        targetHeight: input.resize.height,
        mean: this.session.manifest.imageMean,
        std: this.session.manifest.imageStd,
      });
      run.resources.push(resource);
      pass.setPipeline(resource.pipeline);
      pass.setBindGroup(0, resource.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(outputLength / 256));
      pass.end();

      readback = this.arena.device.createBuffer({
        size: outputLength * Float32Array.BYTES_PER_ELEMENT,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
      encoder.copyBufferToBuffer(output, 0, readback, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
      this.arena.device.queue.submit([encoder.finish()]);
      await this.arena.device.queue.onSubmittedWorkDone?.();
      const validationError = await scopedDevice.popErrorScope?.();
      if (validationError) {
        throw new WebGpuVisionPreprocessFallbackError(String(validationError));
      }
      await readback.mapAsync(GPU_MAP_READ);
      const values = new Float32Array(readback.getMappedRange()).slice();
      readback.unmap();
      readback.destroy?.();
      readback = undefined;
      return {
        values,
        width: input.resize.width,
        height: input.resize.height,
      };
    } finally {
      readback?.destroy?.();
      for (const resource of run.resources) {
        resource.destroy();
      }
      for (const item of run.cleanup.reverse()) {
        item.destroy?.();
      }
    }
  }

  private bufferFromRgba(label: string, values: Uint8ClampedArray, cleanup: GpuResource[]): WebGpuBufferLike {
    const widened = new Uint32Array(values.length);
    for (let index = 0; index < values.length; index += 1) {
      widened[index] = values[index] ?? 0;
    }
    const buffer = this.arena.createBuffer(label, widened.byteLength, GPU_STORAGE | GPU_COPY_DST);
    this.arena.device.queue.writeBuffer(buffer, 0, widened);
    cleanup.push(buffer);
    return buffer;
  }

  private scratchF32(label: string, length: number, cleanup: GpuResource[]): WebGpuBufferLike {
    const buffer = this.arena.createBuffer(label, length * Float32Array.BYTES_PER_ELEMENT, GPU_STORAGE | GPU_COPY_SRC | GPU_COPY_DST);
    cleanup.push(buffer);
    return buffer;
  }
}

function numberOption(options: Readonly<Record<string, unknown>> | undefined, name: string): number | undefined {
  const value = options?.[name];
  return typeof value === "number" ? value : undefined;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Vision preprocessing was aborted.", "AbortError");
  }
}

function isFallbackError(error: unknown): boolean {
  return error instanceof WebGpuVisionPreprocessFallbackError ||
    (error instanceof Error && error.message.includes("WebGPU memory cap exceeded"));
}

class WebGpuVisionPreprocessFallbackError extends Error {}
