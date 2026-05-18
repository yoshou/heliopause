import type {
  SegmentRunnerProvider,
} from "./segment-runner";

export type RunnerBufferOwner = "cpu" | SegmentRunnerProvider;

export type RunnerBufferStorage =
  | {
      kind: "cpu";
      data: Float32Array;
    }
  | {
      kind: "provider";
      provider: SegmentRunnerProvider;
      handle: unknown;
      exportToCpu: () => Float32Array | Promise<Float32Array>;
      destroy?: () => void;
      destroyed?: boolean;
    };

export type RunnerBuffer = {
  dtype: "f32";
  shape: readonly number[];
  storage: RunnerBufferStorage;
};

export function cpuRunnerBuffer(data: Float32Array, shape: readonly number[]): RunnerBuffer {
  return {
    dtype: "f32",
    shape,
    storage: {
      kind: "cpu",
      data,
    },
  };
}

export function providerRunnerBuffer(
  provider: SegmentRunnerProvider,
  handle: unknown,
  shape: readonly number[],
  exportToCpu: () => Float32Array | Promise<Float32Array>,
  destroy?: () => void,
): RunnerBuffer {
  return {
    dtype: "f32",
    shape,
    storage: {
      kind: "provider",
      provider,
      handle,
      exportToCpu,
      destroy,
    },
  };
}

export function runnerBufferOwner(buffer: RunnerBuffer): RunnerBufferOwner {
  return buffer.storage.kind === "cpu" ? "cpu" : buffer.storage.provider;
}

export async function runnerBufferToCpu(buffer: RunnerBuffer): Promise<Float32Array> {
  return buffer.storage.kind === "cpu"
    ? buffer.storage.data
    : buffer.storage.exportToCpu();
}

export function destroyRunnerBuffer(buffer: RunnerBuffer): void {
  if (buffer.storage.kind === "provider") {
    if (buffer.storage.destroyed) {
      return;
    }
    buffer.storage.destroyed = true;
    buffer.storage.destroy?.();
  }
}
