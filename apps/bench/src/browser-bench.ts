import {
  checkWebGpuSupport,
  runWebGpuSmokeTest,
  type WebGpuSupport,
} from "../../../packages/engine/src/runner/webgpu/index";
import { createWebGpuQuantizedWeightHandle } from "../../../packages/engine/src/runner/webgpu/quantized-handles";
import {
  fullAttentionDecodeOutWebGpuResident,
  matMulQ4_KWebGpu,
  matMulQ5_KWebGpu,
  matMulQ6_KWebGpu,
  matMulQ8_0WebGpu,
  matMulSwiGluDownWebGpuResident,
  matMulSwiGluWebGpuResident,
  matMulTop1WebGpuQuantizedResident,
  matMulWebGpuQuantizedResident,
} from "../../../packages/engine/src/runner/webgpu/matmul";
import { GPU_COPY_DST, GPU_COPY_SRC, GPU_MAP_READ } from "../../../packages/engine/src/runner/webgpu/gpu-constants";
import { storageBuffer } from "../../../packages/engine/src/runner/webgpu/gpu-bindings";
import {
  createBatchedFullAttentionMaterializedApplyResources,
  createBatchedFullAttentionMaterializedScoreResources,
  createBatchedFullAttentionRollingTileApplyResources,
  createBatchedFullAttentionRollingTileFinalResources,
  createBatchedFullAttentionRollingTileInitResources,
  createBatchedFullAttentionRollingTileMergeResources,
  createBatchedFullAttentionRollingTileProbabilityResources,
  createKMatMulBindResources,
  createQ8_0MatMulBindResources,
} from "../../../packages/engine/src/runner/webgpu/kernel-resources";
import {
  quantizeQ8_0Columns,
  quantizeQ8_KColumns,
  quantizedWeightUploadBytes,
  webGpuQuantizedWeightLayout,
} from "../../../packages/engine/src/runner/webgpu/quantized-handles";
import type {
  NavigatorWithWebGpu,
  WebGpuBufferLike,
  WebGpuComputePassLike,
  WebGpuDeviceLike,
  WebGpuQuantizedWeightHandleInternal,
} from "../../../packages/engine/src/runner/webgpu/gpu-types";
import {
  gqaAttention,
  sigmoid,
  silu,
} from "../../../packages/engine/src/runner/reference/kernels";
import type {
  GqaAttentionOptions,
} from "../../../packages/engine/src/runner/types";
import {
  createWasmQuantizedWeightHandle,
  gqaAttentionWasm,
  matMulQuantizedWasm,
  matMulQuantizedWasmResident,
  prefillWasmBackend,
  releaseWasmQuantizedWeightHandle,
  type WasmQuantizedWeightHandle,
} from "../../../packages/engine/src/runner/wasm/wasm-kernels";
import {
  quantizeQ8_0,
  quantizeQ8_K,
  vecDotQ4_K_Q8_K,
  vecDotQ5_K_Q8_K,
  vecDotQ6_K_Q8_K,
  vecDotQ8_0_Q8_0,
  type QuantizedQ8K,
  type QuantizedQ8_0,
} from "../../../packages/engine/src/quant";

export type BrowserBenchSize = "small" | "medium" | "large";

export type BrowserBenchCaseId =
  | "matmul"
  | "gqa-attention"
  | "swiglu"
  | "swiglu-down"
  | "full-attention-decode-out"
  | "prefill-attention"
  | "top-token";

export type BrowserBenchBackend =
  | "ts-reference"
  | "wasm"
  | "wasm-resident"
  | "webgpu"
  | "webgpu-dispatch"
  | "webgpu-timestamp"
  | "webgpu-resident";

export type BrowserBenchStatus = "ok" | "failed" | "skipped";

export type BrowserBenchRunOptions = {
  caseIds?: BrowserBenchCaseId[];
  sizes?: BrowserBenchSize[];
  warmupIterations?: number;
  minimumMs?: number;
  signal?: AbortSignal;
  onResult?: (result: BrowserBenchResult) => void;
};

export type BrowserBenchEnvironment = {
  userAgent?: string;
  wasmBackend: "wasm-simd" | "ts";
  webGpuSupport: WebGpuSupport;
  webGpuSmoke?: Awaited<ReturnType<typeof runWebGpuSmokeTest>>;
};

export type BrowserBenchResult = {
  caseId: BrowserBenchCaseId;
  caseName: string;
  backend: BrowserBenchBackend;
  size: BrowserBenchSize;
  variant:
    | "reference"
    | "standalone"
    | "resident"
    | "fused"
    | "rolling-512";
  shape: string;
  status: BrowserBenchStatus;
  iterations: number;
  totalMs: number;
  meanMs: number;
  opsPerSecond: number;
  gpuMeanMs?: number;
  checksum?: number;
  maxAbsDiff?: number;
  maxRelDiff?: number;
  tolerance: number;
  relativeTolerance: number;
  tolerancePass?: boolean;
  speedupVsReference?: number;
  speedupVsWasm?: number;
  message?: string;
};

export type BrowserBenchReport = {
  environment: BrowserBenchEnvironment;
  results: BrowserBenchResult[];
};

type BenchOutput = Float32Array;
type BenchRunValue = BenchOutput | { output: BenchOutput; gpuMs?: number };

type BenchTask = {
  caseId: BrowserBenchCaseId;
  caseName: string;
  size: BrowserBenchSize;
  shape: string;
  tolerance: number;
  relativeTolerance: number;
  referenceBackend: BrowserBenchBackend;
  reference: () => Promise<BenchRunValue>;
  teardown?: () => void | Promise<void>;
  candidates: Array<{
    backend: BrowserBenchBackend;
    variant: BrowserBenchResult["variant"];
    run?: () => Promise<BenchRunValue | undefined>;
    prepare?: () => Promise<PreparedBenchRun | undefined>;
  }>;
};

type PreparedBenchRun = {
  run: () => Promise<BenchRunValue | undefined>;
  teardown?: () => void | Promise<void>;
};

type PrefillAttentionBenchMode =
  | "materialized"
  | "rolling-512";

type BenchMeasurement = {
  iterations: number;
  totalMs: number;
  gpuTotalMs?: number;
  output: BenchOutput;
};

type QuantizedType = "Q4_K" | "Q5_K" | "Q6_K" | "Q8_0";

const DEFAULT_CASES: BrowserBenchCaseId[] = [
  "matmul",
  "gqa-attention",
  "swiglu",
  "swiglu-down",
  "full-attention-decode-out",
  "prefill-attention",
  "top-token",
];
const DEFAULT_SIZES: BrowserBenchSize[] = ["small", "medium"];
const GPU_QUERY_RESOLVE = 512;

export async function runBrowserBench(
  options: BrowserBenchRunOptions = {},
): Promise<BrowserBenchReport> {
  const warmupIterations = options.warmupIterations ?? 3;
  const minimumMs = options.minimumMs ?? 250;
  const caseIds = options.caseIds?.length ? options.caseIds : DEFAULT_CASES;
  const sizes = options.sizes?.length ? options.sizes : DEFAULT_SIZES;
  const environment = await readBenchEnvironment();
  const results: BrowserBenchResult[] = [];

  for (const task of buildBenchTasks(caseIds, sizes)) {
    try {
      throwIfAborted(options.signal);
      const referenceResult = await measureTask({
        task,
        backend: task.referenceBackend,
        variant: "reference",
        run: task.reference,
        warmupIterations,
        minimumMs,
        signal: options.signal,
      });
      results.push(referenceResult);
      options.onResult?.(referenceResult);

      const referenceOutput = referenceResult.status === "ok"
        ? normalizeBenchRunValue(await task.reference())?.output
        : undefined;
      for (const candidate of task.candidates) {
        throwIfAborted(options.signal);
        const prepared = candidate.prepare
          ? await candidate.prepare()
          : candidate.run
            ? { run: candidate.run }
            : undefined;
        if (!prepared) {
          const result = skippedResult({ task, backend: candidate.backend, variant: candidate.variant }, "backend unavailable");
          results.push(result);
          options.onResult?.(result);
          continue;
        }
        try {
          const result = await measureTask({
            task,
            backend: candidate.backend,
            variant: candidate.variant,
            run: prepared.run,
            warmupIterations,
            minimumMs,
            signal: options.signal,
            referenceOutput,
            referenceMeanMs: referenceResult.status === "ok" ? referenceResult.meanMs : undefined,
          });
          results.push(result);
          options.onResult?.(result);
        } finally {
          await prepared.teardown?.();
        }
      }
    } finally {
      await task.teardown?.();
    }
  }

  logBrowserBenchTimings(results);
  return {
    environment,
    results,
  };
}

export async function readBenchEnvironment(): Promise<BrowserBenchEnvironment> {
  const [wasmBackend, webGpuSupport] = await Promise.all([
    prefillWasmBackend(),
    checkWebGpuSupport(),
  ]);
  const webGpuSmoke = webGpuSupport.available ? await runWebGpuSmokeTest() : undefined;
  return {
    userAgent: typeof navigator === "undefined" ? undefined : navigator.userAgent,
    wasmBackend,
    webGpuSupport,
    webGpuSmoke,
  };
}

export function defaultBrowserBenchCaseIds(): BrowserBenchCaseId[] {
  return [...DEFAULT_CASES];
}

export function defaultBrowserBenchSizes(): BrowserBenchSize[] {
  return [...DEFAULT_SIZES];
}

function logBrowserBenchTimings(results: readonly BrowserBenchResult[]): void {
  if (typeof console === "undefined") {
    return;
  }
  console.table(results.map((result) => ({
    case: result.caseName,
    size: result.size,
    backend: result.backend,
    variant: result.variant,
    status: result.status,
    meanMs: roundMs(result.meanMs),
    gpuMeanMs: result.gpuMeanMs === undefined ? undefined : roundMs(result.gpuMeanMs),
    totalMs: roundMs(result.totalMs),
    iterations: result.iterations,
    opsPerSecond: Math.round(result.opsPerSecond * 10) / 10,
  })));
}

function buildBenchTasks(caseIds: readonly BrowserBenchCaseId[], sizes: readonly BrowserBenchSize[]): BenchTask[] {
  const tasks: BenchTask[] = [];
  for (const size of sizes) {
    for (const caseId of caseIds) {
      if (caseId === "matmul") {
        for (const type of ["Q4_K", "Q5_K", "Q6_K", "Q8_0"] as const) {
          tasks.push(createMatMulTask(size, type, false));
          tasks.push(createMatMulTask(size, type, true));
        }
      } else if (caseId === "gqa-attention") {
        tasks.push(createGqaAttentionTask(size));
      } else if (caseId === "swiglu") {
        tasks.push(createSwiGluTask(size));
      } else if (caseId === "swiglu-down") {
        tasks.push(createSwiGluDownTask(size));
      } else if (caseId === "full-attention-decode-out") {
        tasks.push(createFullAttentionDecodeOutTask(size));
      } else if (caseId === "prefill-attention") {
        tasks.push(createPrefillAttentionTask(size));
      } else {
        tasks.push(createTopTokenTask(size));
      }
    }
  }
  return tasks;
}

function createPrefillAttentionTask(sizeName: BrowserBenchSize): BenchTask {
  const size = prefillAttentionShape(sizeName);
  const query = sequence(size.tokenCount * size.queryHeadCount * size.headSize, seedFor("prefill-attention", sizeName, 1));
  const key = sequence(size.contextLength * size.keyValueHeadCount * size.headSize, seedFor("prefill-attention", sizeName, 2));
  const value = sequence(size.contextLength * size.keyValueHeadCount * size.valueSize, seedFor("prefill-attention", sizeName, 3));
  const positions = prefillAttentionPositions(size.tokenCount, size.keyValueTokenCount);
  const shape =
    `${size.kind} T=${size.tokenCount} K=${size.keyValueTokenCount} ` +
    `head=${size.headSize} value=${size.valueSize} qHeads=${size.queryHeadCount} kvHeads=${size.keyValueHeadCount} ` +
    `oldProb=${formatBytes(prefillAttentionProbabilityBytes(size))} rolling512=${formatBytes(prefillAttentionRollingTempBytes(size, 512))}`;
  let materializedReference: PreparedBenchRun | undefined;
  return {
    caseId: "prefill-attention",
    caseName: "Prefill attention",
    size: sizeName,
    shape,
    tolerance: 1e-3,
    relativeTolerance: 1e-4,
    referenceBackend: "webgpu-dispatch",
    reference: async () => {
      materializedReference ??= await prepareWebGpuPrefillAttention("materialized", query, key, value, positions, size);
      const output = await materializedReference?.run();
      if (!output) {
        throw new Error("WebGPU materialized prefill attention is unavailable.");
      }
      return output;
    },
    teardown: async () => {
      await materializedReference?.teardown?.();
      materializedReference = undefined;
    },
    candidates: [
      {
        backend: "webgpu-dispatch",
        variant: "rolling-512",
        prepare: async () => {
          await materializedReference?.teardown?.();
          materializedReference = undefined;
          return prepareWebGpuPrefillAttention("rolling-512", query, key, value, positions, size);
        },
      },
    ],
  };
}

function createMatMulTask(sizeName: BrowserBenchSize, type: QuantizedType, resident: boolean): BenchTask {
  const size = matMulShape(sizeName);
  const weightBytes = quantizedWeightBytes(type, size.inputSize, size.rowCount);
  const inputColumns = sequence(size.inputSize * size.columnCount, seedFor(type, sizeName, resident ? 11 : 3));
  const shape = `${type} input=${size.inputSize} rows=${size.rowCount} cols=${size.columnCount}`;
  const reference = () => matMulReference(type, weightBytes, inputColumns, size.inputSize, size.rowCount, size.columnCount);
  return {
    caseId: "matmul",
    caseName: resident ? `resident quantized matmul ${type}` : `quantized matmul ${type}`,
    size: sizeName,
    shape,
    tolerance: type === "Q8_0" ? 1e-3 : 7.5e-2,
    relativeTolerance: 1e-4,
    referenceBackend: "wasm",
    reference,
    candidates: resident
      ? [
          {
            backend: "wasm-resident",
            variant: "resident",
            prepare: async () => {
              const handle = await createWasmQuantizedWeightHandle(type, weightBytes, size.inputSize, size.rowCount);
              if (!handle) {
                return undefined;
              }
              return {
                run: () => matMulQuantizedWasmResident(handle, inputColumns, size.inputSize, size.rowCount, size.columnCount),
                teardown: () => releaseWasmQuantizedWeightHandle(handle),
              };
            },
          },
          {
            backend: "webgpu-resident",
            variant: "resident",
            prepare: async () => {
              const handle = await createWebGpuQuantizedWeightHandle(type, weightBytes, size.inputSize, size.rowCount);
              if (!handle) {
                return undefined;
              }
              return {
                run: () => matMulWebGpuQuantizedResident(handle, inputColumns, size.columnCount),
                teardown: () => handle.destroy(),
              };
            },
          },
          {
            backend: "webgpu-dispatch",
            variant: "resident",
            prepare: () => prepareWebGpuMatMulDispatchOnly(type, weightBytes, inputColumns, size.inputSize, size.rowCount, size.columnCount),
          },
          {
            backend: "webgpu-timestamp",
            variant: "resident",
            prepare: () => prepareWebGpuMatMulTimestamp(type, weightBytes, inputColumns, size.inputSize, size.rowCount, size.columnCount),
          },
        ]
      : [
          {
            backend: "webgpu",
            variant: "standalone",
            run: () => runWebGpuMatMul(type, weightBytes, inputColumns, size.inputSize, size.rowCount, size.columnCount),
          },
        ],
  };
}

function createGqaAttentionTask(sizeName: BrowserBenchSize): BenchTask {
  const options = gqaShape(sizeName);
  const query = sequence(options.tokenCount * options.queryHeadCount * options.headSize, seedFor("gqa", sizeName, 1));
  const key = sequence((options.keyValueTokenCount ?? options.tokenCount) * options.keyValueHeadCount * options.headSize, seedFor("gqa", sizeName, 2));
  const value = sequence(key.length, seedFor("gqa", sizeName, 3));
  const mask = causalMask(options.tokenCount, options.keyValueTokenCount ?? options.tokenCount);
  const referenceOptions = { ...options, mask };
  return {
    caseId: "gqa-attention",
    caseName: "GQA attention",
    size: sizeName,
    shape: `head=${options.headSize} qHeads=${options.queryHeadCount} kvHeads=${options.keyValueHeadCount} tokens=${options.tokenCount}/${options.keyValueTokenCount ?? options.tokenCount}`,
    tolerance: 5e-3,
    relativeTolerance: 1e-4,
    referenceBackend: "wasm",
    reference: async () => (await gqaAttentionWasm(query, key, value, referenceOptions)) ??
      gqaAttention(query, key, value, referenceOptions),
    candidates: [
      {
        backend: "ts-reference",
        variant: "standalone",
        run: async () => gqaAttention(query, key, value, referenceOptions),
      },
    ],
  };
}

function createSwiGluTask(sizeName: BrowserBenchSize): BenchTask {
  const size = ffnShape(sizeName);
  const input = sequence(size.inputSize * size.columnCount, seedFor("swiglu", sizeName, 1));
  const gateBytes = quantizedWeightBytes("Q4_K", size.inputSize, size.hiddenSize);
  const upBytes = quantizedWeightBytes("Q5_K", size.inputSize, size.hiddenSize);
  return {
    caseId: "swiglu",
    caseName: "SwiGLU fused",
    size: sizeName,
    shape: `input=${size.inputSize} hidden=${size.hiddenSize} cols=${size.columnCount}`,
    tolerance: 1.25e-1,
    relativeTolerance: 1e-4,
    referenceBackend: "wasm",
    reference: async () => swigluReference(gateBytes, upBytes, input, size.inputSize, size.hiddenSize, size.columnCount),
    candidates: [
      {
        backend: "webgpu-resident",
        variant: "fused",
        prepare: async () => {
          const gate = await createWebGpuQuantizedWeightHandle("Q4_K", gateBytes, size.inputSize, size.hiddenSize);
          const up = await createWebGpuQuantizedWeightHandle("Q5_K", upBytes, size.inputSize, size.hiddenSize);
          if (!gate || !up) {
            gate?.destroy();
            up?.destroy();
            return undefined;
          }
          return {
            run: () => matMulSwiGluWebGpuResident(gate, up, input, size.columnCount),
            teardown: () => {
              gate.destroy();
              up.destroy();
            },
          };
        },
      },
    ],
  };
}

function createSwiGluDownTask(sizeName: BrowserBenchSize): BenchTask {
  const size = ffnShape(sizeName);
  const input = sequence(size.inputSize * size.columnCount, seedFor("ffn", sizeName, 1));
  const gateBytes = quantizedWeightBytes("Q4_K", size.inputSize, size.hiddenSize);
  const upBytes = quantizedWeightBytes("Q5_K", size.inputSize, size.hiddenSize);
  const downBytes = quantizedWeightBytes("Q6_K", size.hiddenSize, size.outputSize);
  return {
    caseId: "swiglu-down",
    caseName: "SwiGLU + down fused",
    size: sizeName,
    shape: `input=${size.inputSize} hidden=${size.hiddenSize} output=${size.outputSize} cols=${size.columnCount}`,
    tolerance: 2e-1,
    relativeTolerance: 1e-4,
    referenceBackend: "wasm",
    reference: async () => swigluDownReference(gateBytes, upBytes, downBytes, input, size),
    candidates: [
      {
        backend: "webgpu-resident",
        variant: "fused",
        prepare: async () => {
          const gate = await createWebGpuQuantizedWeightHandle("Q4_K", gateBytes, size.inputSize, size.hiddenSize);
          const up = await createWebGpuQuantizedWeightHandle("Q5_K", upBytes, size.inputSize, size.hiddenSize);
          const down = await createWebGpuQuantizedWeightHandle("Q6_K", downBytes, size.hiddenSize, size.outputSize);
          if (!gate || !up || !down) {
            gate?.destroy();
            up?.destroy();
            down?.destroy();
            return undefined;
          }
          return {
            run: () => matMulSwiGluDownWebGpuResident(gate, up, down, input, size.columnCount),
            teardown: () => {
              gate.destroy();
              up.destroy();
              down.destroy();
            },
          };
        },
      },
    ],
  };
}

function createFullAttentionDecodeOutTask(sizeName: BrowserBenchSize): BenchTask {
  const size = fullAttentionShape(sizeName);
  const hiddenSize = size.headSize * size.queryHeadCount;
  const query = sequence(hiddenSize, seedFor("full", sizeName, 1));
  const key = sequence(size.keyValueTokenCount * size.keyValueHeadCount * size.headSize, seedFor("full", sizeName, 2));
  const value = sequence(size.contextLength * size.keyValueHeadCount * size.headSize, seedFor("full", sizeName, 3));
  const gate = sequence(hiddenSize, seedFor("full", sizeName, 4));
  const outBytes = quantizedWeightBytes("Q4_K", hiddenSize, size.outputSize);
  return {
    caseId: "full-attention-decode-out",
    caseName: "full attention decode + out fused",
    size: sizeName,
    shape: `head=${size.headSize} qHeads=${size.queryHeadCount} kvHeads=${size.keyValueHeadCount} kvTokens=${size.keyValueTokenCount}`,
    tolerance: 2e-1,
    relativeTolerance: 1e-4,
    referenceBackend: "wasm",
    reference: async () => fullAttentionDecodeOutReference(outBytes, query, key, value, gate, size),
    candidates: [
      {
        backend: "webgpu-resident",
        variant: "fused",
        prepare: async () => {
          const out = await createWebGpuQuantizedWeightHandle("Q4_K", outBytes, hiddenSize, size.outputSize);
          if (!out) {
            return undefined;
          }
          return {
            run: () => fullAttentionDecodeOutWebGpuResident(out, query, key, value, gate, size),
            teardown: () => out.destroy(),
          };
        },
      },
    ],
  };
}

function createTopTokenTask(sizeName: BrowserBenchSize): BenchTask {
  const size = matMulShape(sizeName);
  const input = sequence(size.inputSize, seedFor("top", sizeName, 1));
  const weightBytes = quantizedWeightBytes("Q4_K", size.inputSize, size.rowCount);
  return {
    caseId: "top-token",
    caseName: "matmul + top token",
    size: sizeName,
    shape: `Q4_K input=${size.inputSize} rows=${size.rowCount}`,
    tolerance: 1e-3,
    relativeTolerance: 1e-6,
    referenceBackend: "wasm",
    reference: async () => topTokenReference(weightBytes, input, size.inputSize, size.rowCount),
    candidates: [
      {
        backend: "webgpu-resident",
        variant: "fused",
        prepare: async () => {
          const handle = await createWebGpuQuantizedWeightHandle("Q4_K", weightBytes, size.inputSize, size.rowCount);
          if (!handle) {
            return undefined;
          }
          return {
            run: async () => {
              const top = await matMulTop1WebGpuQuantizedResident(handle, input);
              return new Float32Array([top.id, top.value]);
            },
            teardown: () => handle.destroy(),
          };
        },
      },
    ],
  };
}

async function measureTask(params: {
  task: BenchTask;
  backend: BrowserBenchBackend;
  variant: BrowserBenchResult["variant"];
  run: () => Promise<BenchRunValue | undefined>;
  warmupIterations: number;
  minimumMs: number;
  signal?: AbortSignal;
  referenceOutput?: BenchOutput;
  referenceMeanMs?: number;
}): Promise<BrowserBenchResult> {
  try {
    const measured = await runMeasured(params.run, params.warmupIterations, params.minimumMs, params.signal);
    if (!measured) {
      return skippedResult(params, "backend unavailable");
    }
    const meanMs = measured.totalMs / measured.iterations;
    const diff = params.referenceOutput ? tensorDiff(measured.output, params.referenceOutput) : undefined;
    const tolerancePass = diff
      ? diff.maxAbs <= params.task.tolerance || diff.maxRel <= params.task.relativeTolerance
      : undefined;
    return {
      caseId: params.task.caseId,
      caseName: params.task.caseName,
      backend: params.backend,
      size: params.task.size,
      variant: params.variant,
      shape: params.task.shape,
      status: tolerancePass === false ? "failed" : "ok",
      iterations: measured.iterations,
      totalMs: measured.totalMs,
      meanMs,
      opsPerSecond: 1000 / meanMs,
      gpuMeanMs: measured.gpuTotalMs === undefined ? undefined : measured.gpuTotalMs / measured.iterations,
      checksum: checksum(measured.output),
      maxAbsDiff: diff?.maxAbs,
      maxRelDiff: diff?.maxRel,
      tolerance: params.task.tolerance,
      relativeTolerance: params.task.relativeTolerance,
      tolerancePass,
      speedupVsReference: params.referenceMeanMs ? params.referenceMeanMs / meanMs : undefined,
      speedupVsWasm: params.referenceMeanMs && params.task.referenceBackend === "wasm"
        ? params.referenceMeanMs / meanMs
        : undefined,
      message: tolerancePass === false ? "correctness diff exceeds tolerance" : undefined,
    };
  } catch (error) {
    if (params.signal?.aborted) {
      return skippedResult(params, "cancelled");
    }
    return {
      ...skippedResult(params, error instanceof Error ? error.message : String(error)),
      status: "failed",
    };
  }
}

async function runMeasured(
  run: () => Promise<BenchRunValue | undefined>,
  warmupIterations: number,
  minimumMs: number,
  signal?: AbortSignal,
): Promise<BenchMeasurement | undefined> {
  let output: BenchOutput | undefined;
  for (let index = 0; index < warmupIterations; index += 1) {
    throwIfAborted(signal);
    const value = normalizeBenchRunValue(await run());
    if (!value) {
      return undefined;
    }
    output = value.output;
  }

  let iterations = 1;
  let totalMs = 0;
  let gpuTotalMs: number | undefined;
  do {
    throwIfAborted(signal);
    const start = performance.now();
    let batchGpuTotalMs = 0;
    let hasGpuTime = false;
    for (let index = 0; index < iterations; index += 1) {
      throwIfAborted(signal);
      const value = normalizeBenchRunValue(await run());
      if (!value) {
        return undefined;
      }
      output = value.output;
      if (value.gpuMs !== undefined) {
        batchGpuTotalMs += value.gpuMs;
        hasGpuTime = true;
      }
    }
    totalMs = performance.now() - start;
    gpuTotalMs = hasGpuTime ? batchGpuTotalMs : undefined;
    if (totalMs < minimumMs) {
      iterations *= 2;
    }
  } while (totalMs < minimumMs);

  return { iterations, totalMs, gpuTotalMs, output: output ?? new Float32Array() };
}

function normalizeBenchRunValue(value: BenchRunValue | undefined): { output: BenchOutput; gpuMs?: number } | undefined {
  if (!value) {
    return undefined;
  }
  if (value instanceof Float32Array) {
    return { output: value };
  }
  return value;
}

function skippedResult(
  params: {
    task: BenchTask;
    backend: BrowserBenchBackend;
    variant: BrowserBenchResult["variant"];
  },
  message: string,
): BrowserBenchResult {
  return {
    caseId: params.task.caseId,
    caseName: params.task.caseName,
    backend: params.backend,
    size: params.task.size,
    variant: params.variant,
    shape: params.task.shape,
    status: "skipped",
    iterations: 0,
    totalMs: 0,
    meanMs: 0,
    opsPerSecond: 0,
    tolerance: params.task.tolerance,
    relativeTolerance: params.task.relativeTolerance,
    message,
  };
}

async function runWebGpuMatMul(
  type: QuantizedType,
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  if (type === "Q4_K") {
    return matMulQ4_KWebGpu(weightBytes, inputColumns, inputSize, rowCount, columnCount);
  }
  if (type === "Q5_K") {
    return matMulQ5_KWebGpu(weightBytes, inputColumns, inputSize, rowCount, columnCount);
  }
  if (type === "Q6_K") {
    return matMulQ6_KWebGpu(weightBytes, inputColumns, inputSize, rowCount, columnCount);
  }
  return matMulQ8_0WebGpu(weightBytes, inputColumns, inputSize, rowCount, columnCount);
}

async function prepareWebGpuPrefillAttention(
  mode: PrefillAttentionBenchMode,
  query: Float32Array,
  key: Float32Array,
  value: Float32Array,
  positions: Uint32Array,
  size: ReturnType<typeof prefillAttentionShape>,
): Promise<PreparedBenchRun | undefined> {
  const device = await requestPlainWebGpuDevice();
  if (!device) {
    return undefined;
  }

  const queryBuffer = uploadStorageBuffer(device, query);
  const keyBuffer = uploadStorageBuffer(device, key);
  const valueBuffer = uploadStorageBuffer(device, value);
  const positionsBuffer = uploadStorageBuffer(device, positions);
  const outputByteLength = size.tokenCount * size.queryHeadCount * size.valueSize * Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = storageBuffer(device, outputByteLength, GPU_COPY_SRC);
  const readbackBuffer = device.createBuffer({
    size: outputByteLength,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const probabilityTokenCapacity = prefillAttentionProbabilityTokenCapacity(size.keyValueTokenCount);
  const needsMaterialized = mode === "materialized";
  const needsRolling = mode === "rolling-512";
  const probabilityBuffer = needsMaterialized
    ? storageBuffer(device, prefillAttentionProbabilityBytes(size), 0)
    : undefined;
  const rollingTileSize = 512;
  const headTokenCount = size.tokenCount * size.queryHeadCount;
  const rollingProbabilityTileBuffer = needsRolling ? storageBuffer(device, headTokenCount * rollingTileSize * Float32Array.BYTES_PER_ELEMENT, 0) : undefined;
  const rollingRowMaxBuffer = needsRolling ? storageBuffer(device, headTokenCount * Float32Array.BYTES_PER_ELEMENT, 0) : undefined;
  const rollingRowSumBuffer = needsRolling ? storageBuffer(device, headTokenCount * Float32Array.BYTES_PER_ELEMENT, 0) : undefined;
  const rollingTileMaxBuffer = needsRolling ? storageBuffer(device, headTokenCount * Float32Array.BYTES_PER_ELEMENT, 0) : undefined;
  const rollingTileSumBuffer = needsRolling ? storageBuffer(device, headTokenCount * Float32Array.BYTES_PER_ELEMENT, 0) : undefined;
  const rollingOldScaleBuffer = needsRolling ? storageBuffer(device, headTokenCount * Float32Array.BYTES_PER_ELEMENT, 0) : undefined;
  const rollingTileScaleBuffer = needsRolling ? storageBuffer(device, headTokenCount * Float32Array.BYTES_PER_ELEMENT, 0) : undefined;
  const resourcesToDestroy: Array<{ destroy?: () => void }> = [
    queryBuffer,
    keyBuffer,
    valueBuffer,
    positionsBuffer,
    outputBuffer,
    readbackBuffer,
  ];
  if (probabilityBuffer) {
    resourcesToDestroy.push(probabilityBuffer);
  }
  if (
    rollingProbabilityTileBuffer && rollingRowMaxBuffer && rollingRowSumBuffer && rollingTileMaxBuffer &&
    rollingTileSumBuffer && rollingOldScaleBuffer && rollingTileScaleBuffer
  ) {
    resourcesToDestroy.push(
      rollingProbabilityTileBuffer,
      rollingRowMaxBuffer,
      rollingRowSumBuffer,
      rollingTileMaxBuffer,
      rollingTileSumBuffer,
      rollingOldScaleBuffer,
      rollingTileScaleBuffer,
    );
  }

  const commonOptions = {
    headSize: size.headSize,
    valueSize: size.valueSize,
    queryHeadCount: size.queryHeadCount,
    keyValueHeadCount: size.keyValueHeadCount,
    keyValueTokenCount: size.keyValueTokenCount,
    contextLength: size.contextLength,
    slidingWindow: size.slidingWindow,
    tokenCount: size.tokenCount,
    scale: 1,
    causal: true,
  };
  const prepared = needsMaterialized && probabilityBuffer
    ? {
        mode,
        score: createBatchedFullAttentionMaterializedScoreResources(
          device,
          queryBuffer,
          keyBuffer,
          positionsBuffer,
          probabilityBuffer,
          { ...commonOptions, probabilityTokenCapacity },
        ),
        apply: createBatchedFullAttentionMaterializedApplyResources(
          device,
          valueBuffer,
          probabilityBuffer,
          positionsBuffer,
          outputBuffer,
          { ...commonOptions, probabilityTokenCapacity },
        ),
      }
    : rollingProbabilityTileBuffer && rollingRowMaxBuffer && rollingRowSumBuffer && rollingTileMaxBuffer &&
        rollingTileSumBuffer && rollingOldScaleBuffer && rollingTileScaleBuffer
      ? {
          mode,
          init: createBatchedFullAttentionRollingTileInitResources(
            device,
            outputBuffer,
            {
              valueSize: size.valueSize,
              queryHeadCount: size.queryHeadCount,
              tokenCount: size.tokenCount,
            },
          ),
          chunks: prefillAttentionChunks(size.keyValueTokenCount, rollingTileSize).map(([tileStart, tileEnd]) => ({
            probability: createBatchedFullAttentionRollingTileProbabilityResources(
              device,
              queryBuffer,
              keyBuffer,
              positionsBuffer,
              rollingProbabilityTileBuffer,
              rollingTileMaxBuffer,
              rollingTileSumBuffer,
              { ...commonOptions, tileSize: rollingTileSize, tileStart, tileLength: tileEnd - tileStart },
            ),
            merge: createBatchedFullAttentionRollingTileMergeResources(
              device,
              rollingRowMaxBuffer,
              rollingRowSumBuffer,
              rollingTileMaxBuffer,
              rollingTileSumBuffer,
              rollingOldScaleBuffer,
              rollingTileScaleBuffer,
              {
                queryHeadCount: size.queryHeadCount,
                tokenCount: size.tokenCount,
                firstTile: tileStart === 0,
              },
            ),
            apply: createBatchedFullAttentionRollingTileApplyResources(
              device,
              valueBuffer,
              rollingProbabilityTileBuffer,
              rollingOldScaleBuffer,
              rollingTileScaleBuffer,
              outputBuffer,
              {
                valueSize: size.valueSize,
                queryHeadCount: size.queryHeadCount,
                keyValueHeadCount: size.keyValueHeadCount,
                contextLength: size.contextLength,
                tileSize: rollingTileSize,
                tileStart,
                tileLength: tileEnd - tileStart,
                tokenCount: size.tokenCount,
              },
            ),
          })),
          final: createBatchedFullAttentionRollingTileFinalResources(
            device,
            rollingRowSumBuffer,
            outputBuffer,
            {
              valueSize: size.valueSize,
              queryHeadCount: size.queryHeadCount,
              tokenCount: size.tokenCount,
            },
          ),
        }
      : undefined;
  if (!prepared) {
    for (let index = resourcesToDestroy.length - 1; index >= 0; index -= 1) {
      resourcesToDestroy[index]?.destroy?.();
    }
    device.destroy?.();
    return undefined;
  }
  if ("init" in prepared) {
    const rollingPrepared = prepared as {
      init: { destroy: () => void };
      chunks: Array<{
        probability: { destroy: () => void };
        merge: { destroy: () => void };
        apply: { destroy: () => void };
      }>;
      final: { destroy: () => void };
    };
    resourcesToDestroy.push(
      rollingPrepared.init,
      ...rollingPrepared.chunks.flatMap((chunk) => [chunk.probability, chunk.merge, chunk.apply]),
      rollingPrepared.final,
    );
  } else if ("score" in prepared) {
    if (!prepared.score || !prepared.apply) {
      for (let index = resourcesToDestroy.length - 1; index >= 0; index -= 1) {
        resourcesToDestroy[index]?.destroy?.();
      }
      device.destroy?.();
      return undefined;
    }
    resourcesToDestroy.push(prepared.score, prepared.apply);
  }

  const validationOutput = await runPreparedPrefillAttention(
    device,
    prepared,
    outputBuffer,
    readbackBuffer,
    outputByteLength,
    size,
    true,
  );
  if (!validationOutput) {
    for (let index = resourcesToDestroy.length - 1; index >= 0; index -= 1) {
      resourcesToDestroy[index]?.destroy?.();
    }
    device.destroy?.();
    return undefined;
  }

  return {
    run: () =>
      runPreparedPrefillAttention(
        device,
        prepared,
        outputBuffer,
        readbackBuffer,
        outputByteLength,
        size,
        false,
        validationOutput,
      ),
    teardown() {
      for (let index = resourcesToDestroy.length - 1; index >= 0; index -= 1) {
        resourcesToDestroy[index]?.destroy?.();
      }
      device.destroy?.();
    },
  };
}

async function runPreparedPrefillAttention(
  device: WebGpuDeviceLike,
  prepared:
    | {
        mode: PrefillAttentionBenchMode;
        score: { pipeline: unknown; bindGroup: unknown };
        apply: { pipeline: unknown; bindGroup: unknown };
      }
    | {
        mode: PrefillAttentionBenchMode;
        init: { pipeline: unknown; bindGroup: unknown };
        chunks: Array<{
          probability: { pipeline: unknown; bindGroup: unknown };
          merge: { pipeline: unknown; bindGroup: unknown };
          apply: { pipeline: unknown; bindGroup: unknown };
        }>;
        final: { pipeline: unknown; bindGroup: unknown };
      },
  outputBuffer: WebGpuBufferLike,
  readbackBuffer: WebGpuBufferLike,
  outputByteLength: number,
  size: ReturnType<typeof prefillAttentionShape>,
  readOutput: boolean,
  validationOutput?: Float32Array,
): Promise<Float32Array> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  if ("score" in prepared) {
    pass.setPipeline(prepared.score.pipeline);
    pass.setBindGroup(0, prepared.score.bindGroup);
    pass.dispatchWorkgroups(size.queryHeadCount, size.tokenCount);
    pass.setPipeline(prepared.apply.pipeline);
    pass.setBindGroup(0, prepared.apply.bindGroup);
    pass.dispatchWorkgroups(size.queryHeadCount, size.valueSize, size.tokenCount);
  } else if ("init" in prepared) {
    pass.setPipeline(prepared.init.pipeline);
    pass.setBindGroup(0, prepared.init.bindGroup);
    pass.dispatchWorkgroups(size.queryHeadCount, size.valueSize, size.tokenCount);
    for (const chunk of prepared.chunks) {
      pass.setPipeline(chunk.probability.pipeline);
      pass.setBindGroup(0, chunk.probability.bindGroup);
      pass.dispatchWorkgroups(size.queryHeadCount, size.tokenCount);
      pass.setPipeline(chunk.merge.pipeline);
      pass.setBindGroup(0, chunk.merge.bindGroup);
      pass.dispatchWorkgroups(size.queryHeadCount, size.tokenCount);
      pass.setPipeline(chunk.apply.pipeline);
      pass.setBindGroup(0, chunk.apply.bindGroup);
      pass.dispatchWorkgroups(size.queryHeadCount, size.valueSize, size.tokenCount);
    }
    pass.setPipeline(prepared.final.pipeline);
    pass.setBindGroup(0, prepared.final.bindGroup);
    pass.dispatchWorkgroups(size.queryHeadCount, size.valueSize, size.tokenCount);
  }
  pass.end();
  if (readOutput) {
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputByteLength);
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone?.();
  if (!readOutput) {
    return validationOutput ?? new Float32Array();
  }
  await readbackBuffer.mapAsync(GPU_MAP_READ);
  try {
    return new Float32Array(readbackBuffer.getMappedRange()).slice();
  } finally {
    readbackBuffer.unmap();
  }
}

function uploadStorageBuffer(device: WebGpuDeviceLike, values: Float32Array | Uint32Array): WebGpuBufferLike {
  const buffer = storageBuffer(device, values.byteLength, GPU_COPY_DST);
  device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

async function requestPlainWebGpuDevice(): Promise<(WebGpuDeviceLike & { destroy?: () => void }) | undefined> {
  const gpu = typeof navigator === "undefined" ? undefined : (navigator as NavigatorWithWebGpu).gpu;
  const adapter = await gpu?.requestAdapter();
  return adapter?.requestDevice?.();
}

async function prepareWebGpuMatMulDispatchOnly(
  type: QuantizedType,
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<PreparedBenchRun | undefined> {
  const publicHandle = await createWebGpuQuantizedWeightHandle(type, weightBytes, inputSize, rowCount);
  if (!publicHandle) {
    return undefined;
  }
  const handle = publicHandle as WebGpuQuantizedWeightHandleInternal;
  const validationOutput = await matMulWebGpuQuantizedResident(publicHandle, inputColumns, columnCount);
  const outputBuffer = storageBuffer(handle.device, rowCount * columnCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const resourcesToDestroy: Array<{ destroy?: () => void }> = [publicHandle, outputBuffer];

  let resources: { pipeline: unknown; bindGroup: unknown; destroy: () => void };
  if (type === "Q8_0") {
    const q8 = quantizeQ8_0Columns(inputColumns, inputSize, columnCount);
    const inputScaleBuffer = storageBuffer(handle.device, q8.scales.byteLength, GPU_COPY_DST);
    const inputQsBuffer = storageBuffer(handle.device, q8.qs.byteLength, GPU_COPY_DST);
    handle.device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    handle.device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    resourcesToDestroy.push(inputScaleBuffer, inputQsBuffer);
    resources = createQ8_0MatMulBindResources(handle, inputScaleBuffer, inputQsBuffer, outputBuffer, columnCount);
  } else {
    const q8 = quantizeQ8_KColumns(inputColumns, inputSize, columnCount);
    const inputScaleBuffer = storageBuffer(handle.device, q8.scales.byteLength, GPU_COPY_DST);
    const inputQsBuffer = storageBuffer(handle.device, q8.qs.byteLength, GPU_COPY_DST);
    const inputBsumsBuffer = storageBuffer(handle.device, q8.bsums.byteLength, GPU_COPY_DST);
    handle.device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    handle.device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    handle.device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);
    resourcesToDestroy.push(inputScaleBuffer, inputQsBuffer, inputBsumsBuffer);
    resources = createKMatMulBindResources(handle, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, outputBuffer, columnCount);
  }
  resourcesToDestroy.push(resources);

  return {
    async run() {
      const encoder = handle.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(resources.pipeline);
      pass.setBindGroup(0, resources.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
      pass.end();
      handle.device.queue.submit([encoder.finish()]);
      await handle.device.queue.onSubmittedWorkDone?.();
      return validationOutput;
    },
    teardown() {
      for (let index = resourcesToDestroy.length - 1; index >= 0; index -= 1) {
        resourcesToDestroy[index]?.destroy?.();
      }
    },
  };
}

async function prepareWebGpuMatMulTimestamp(
  type: QuantizedType,
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<PreparedBenchRun | undefined> {
  const device = await requestTimestampDevice();
  if (!device) {
    return undefined;
  }

  const handle = createTimestampQuantizedHandle(device, type, weightBytes, inputSize, rowCount);
  const validationOutput = await runTimestampMatMulReadback(handle, inputColumns, columnCount);
  const outputBuffer = storageBuffer(device, rowCount * columnCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolveBuffer = device.createBuffer({
    size: 2 * BigUint64Array.BYTES_PER_ELEMENT,
    usage: GPU_QUERY_RESOLVE | 4,
  });
  const readbackBuffer = device.createBuffer({
    size: 2 * BigUint64Array.BYTES_PER_ELEMENT,
    usage: 1 | GPU_COPY_DST,
  });
  const resourcesToDestroy: Array<{ destroy?: () => void }> = [handle, outputBuffer, querySet, resolveBuffer, readbackBuffer];

  let resources: { pipeline: unknown; bindGroup: unknown; destroy: () => void };
  if (type === "Q8_0") {
    const q8 = quantizeQ8_0Columns(inputColumns, inputSize, columnCount);
    const inputScaleBuffer = storageBuffer(device, q8.scales.byteLength, GPU_COPY_DST);
    const inputQsBuffer = storageBuffer(device, q8.qs.byteLength, GPU_COPY_DST);
    device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    resourcesToDestroy.push(inputScaleBuffer, inputQsBuffer);
    resources = createQ8_0MatMulBindResources(handle, inputScaleBuffer, inputQsBuffer, outputBuffer, columnCount);
  } else {
    const q8 = quantizeQ8_KColumns(inputColumns, inputSize, columnCount);
    const inputScaleBuffer = storageBuffer(device, q8.scales.byteLength, GPU_COPY_DST);
    const inputQsBuffer = storageBuffer(device, q8.qs.byteLength, GPU_COPY_DST);
    const inputBsumsBuffer = storageBuffer(device, q8.bsums.byteLength, GPU_COPY_DST);
    device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);
    resourcesToDestroy.push(inputScaleBuffer, inputQsBuffer, inputBsumsBuffer);
    resources = createKMatMulBindResources(handle, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, outputBuffer, columnCount);
  }
  resourcesToDestroy.push(resources);

  return {
    async run() {
      const encoder = device.createCommandEncoder() as TimestampCommandEncoder;
      const pass = encoder.beginComputePass({
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      });
      pass.setPipeline(resources.pipeline);
      pass.setBindGroup(0, resources.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
      pass.end();
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, readbackBuffer, 0, 2 * BigUint64Array.BYTES_PER_ELEMENT);
      device.queue.submit([encoder.finish()]);
      await readbackBuffer.mapAsync(1);
      const timestamps = new BigUint64Array(readbackBuffer.getMappedRange()).slice();
      readbackBuffer.unmap();
      const start = timestamps[0] ?? 0n;
      const end = timestamps[1] ?? start;
      return {
        output: validationOutput,
        gpuMs: Number(end - start) / 1_000_000,
      };
    },
    teardown() {
      for (let index = resourcesToDestroy.length - 1; index >= 0; index -= 1) {
        resourcesToDestroy[index]?.destroy?.();
      }
      device.destroy?.();
    },
  };
}

async function runTimestampMatMulReadback(
  handle: WebGpuQuantizedWeightHandleInternal,
  inputColumns: Float32Array,
  columnCount: number,
): Promise<Float32Array> {
  return matMulWebGpuQuantizedResident(handle, inputColumns, columnCount);
}

function createTimestampQuantizedHandle(
  device: TimestampDevice,
  type: QuantizedType,
  weightBytes: Uint8Array,
  inputSize: number,
  rowCount: number,
): WebGpuQuantizedWeightHandleInternal {
  const layout = webGpuQuantizedWeightLayout(type, inputSize);
  const packedWeight = quantizedWeightUploadBytes(weightBytes);
  const weightBuffer = storageBuffer(device, packedWeight.byteLength, GPU_COPY_DST);
  device.queue.writeBuffer(weightBuffer, 0, packedWeight);
  return {
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
}

async function requestTimestampDevice(): Promise<TimestampDevice | undefined> {
  const gpu = typeof navigator === "undefined"
    ? undefined
    : (navigator as { gpu?: { requestAdapter: () => Promise<TimestampAdapter | null> } }).gpu;
  const adapter = await gpu?.requestAdapter();
  if (!adapter?.requestDevice || !adapter.features?.has("timestamp-query")) {
    return undefined;
  }
  try {
    return await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
  } catch {
    return undefined;
  }
}

type TimestampAdapter = {
  features?: { has(feature: string): boolean };
  requestDevice?: (descriptor: { requiredFeatures: string[] }) => Promise<TimestampDevice>;
};

type TimestampDevice = WebGpuDeviceLike & {
  destroy?: () => void;
  createQuerySet(descriptor: { type: "timestamp"; count: number }): TimestampQuerySet;
  createCommandEncoder(): TimestampCommandEncoder;
};

type TimestampQuerySet = {
  destroy?: () => void;
};

type TimestampCommandEncoder = {
  beginComputePass(descriptor?: unknown): WebGpuComputePassLike;
  copyBufferToBuffer(
    source: WebGpuBufferLike,
    sourceOffset: number,
    destination: WebGpuBufferLike,
    destinationOffset: number,
    size: number,
  ): void;
  resolveQuerySet(
    querySet: TimestampQuerySet,
    firstQuery: number,
    queryCount: number,
    destination: WebGpuBufferLike,
    destinationOffset: number,
  ): void;
  finish(): unknown;
};

async function withWasmHandle<T>(
  type: QuantizedType,
  weightBytes: Uint8Array,
  inputSize: number,
  rowCount: number,
  run: (handle: WasmQuantizedWeightHandle) => Promise<T | undefined>,
): Promise<T | undefined> {
  const handle = await createWasmQuantizedWeightHandle(type, weightBytes, inputSize, rowCount);
  if (!handle) {
    return undefined;
  }
  try {
    return await run(handle);
  } finally {
    releaseWasmQuantizedWeightHandle(handle);
  }
}

async function withWebGpuHandle<T>(
  type: QuantizedType,
  weightBytes: Uint8Array,
  inputSize: number,
  rowCount: number,
  run: (handle: NonNullable<Awaited<ReturnType<typeof createWebGpuQuantizedWeightHandle>>>) => Promise<T | undefined>,
): Promise<T | undefined> {
  const handle = await createWebGpuQuantizedWeightHandle(type, weightBytes, inputSize, rowCount);
  if (!handle) {
    return undefined;
  }
  try {
    return await run(handle);
  } finally {
    handle.destroy();
  }
}

async function matMulReference(
  type: QuantizedType,
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array> {
  return (await matMulQuantizedWasm(type, weightBytes, inputColumns, inputSize, rowCount, columnCount)) ??
    matMulQuantizedTs(type, weightBytes, inputColumns, inputSize, rowCount, columnCount);
}

function matMulQuantizedTs(
  type: QuantizedType,
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Float32Array {
  const rowByteLength = quantizedRowByteLength(type, inputSize);
  const output = new Float32Array(rowCount * columnCount);
  for (let column = 0; column < columnCount; column += 1) {
    const input = inputColumns.slice(column * inputSize, (column + 1) * inputSize);
    const q8 = type === "Q8_0" ? quantizeQ8_0(input) : quantizeQ8_K(input);
    for (let row = 0; row < rowCount; row += 1) {
      const rowBytes = weightBytes.subarray(row * rowByteLength, (row + 1) * rowByteLength);
      output[column * rowCount + row] = dotQuantized(type, rowBytes, q8);
    }
  }
  return output;
}

function dotQuantized(type: QuantizedType, rowBytes: Uint8Array, q8: QuantizedQ8K | QuantizedQ8_0): number {
  if (type === "Q4_K") {
    return vecDotQ4_K_Q8_K(rowBytes, q8 as QuantizedQ8K);
  }
  if (type === "Q5_K") {
    return vecDotQ5_K_Q8_K(rowBytes, q8 as QuantizedQ8K);
  }
  if (type === "Q6_K") {
    return vecDotQ6_K_Q8_K(rowBytes, q8 as QuantizedQ8K);
  }
  return vecDotQ8_0_Q8_0(rowBytes, q8 as QuantizedQ8_0);
}

async function swigluReference(
  gateBytes: Uint8Array,
  upBytes: Uint8Array,
  input: Float32Array,
  inputSize: number,
  hiddenSize: number,
  columnCount: number,
): Promise<Float32Array> {
  const [gate, up] = await Promise.all([
    matMulReference("Q4_K", gateBytes, input, inputSize, hiddenSize, columnCount),
    matMulReference("Q5_K", upBytes, input, inputSize, hiddenSize, columnCount),
  ]);
  return multiply(silu(gate), up);
}

async function swigluDownReference(
  gateBytes: Uint8Array,
  upBytes: Uint8Array,
  downBytes: Uint8Array,
  input: Float32Array,
  size: ReturnType<typeof ffnShape>,
): Promise<Float32Array> {
  const hidden = await swigluReference(gateBytes, upBytes, input, size.inputSize, size.hiddenSize, size.columnCount);
  return matMulReference("Q6_K", downBytes, hidden, size.hiddenSize, size.outputSize, size.columnCount);
}

async function fullAttentionDecodeOutReference(
  outBytes: Uint8Array,
  query: Float32Array,
  key: Float32Array,
  valueDimHeadToken: Float32Array,
  gate: Float32Array,
  size: ReturnType<typeof fullAttentionShape>,
): Promise<Float32Array> {
  const attention = gqaAttention(query, key, valueDimHeadToken, {
    headSize: size.headSize,
    queryHeadCount: size.queryHeadCount,
    keyValueHeadCount: size.keyValueHeadCount,
    tokenCount: 1,
    keyValueTokenCount: size.keyValueTokenCount,
    scale: size.scale,
    causal: false,
    valueLayout: "dim-head-token",
    quantizeQueryForScore: "f16",
  });
  const gated = multiply(attention, sigmoid(gate));
  return matMulReference("Q4_K", outBytes, gated, size.headSize * size.queryHeadCount, size.outputSize, 1);
}

async function topTokenReference(
  weightBytes: Uint8Array,
  input: Float32Array,
  inputSize: number,
  rowCount: number,
): Promise<Float32Array> {
  const logits = await matMulReference("Q4_K", weightBytes, input, inputSize, rowCount, 1);
  let bestId = 0;
  let bestValue = -Infinity;
  for (let index = 0; index < logits.length; index += 1) {
    const value = logits[index] ?? -Infinity;
    if (value > bestValue) {
      bestValue = value;
      bestId = index;
    }
  }
  return new Float32Array([bestId, bestValue]);
}

function multiply(left: Float32Array, right: Float32Array): Float32Array {
  const output = new Float32Array(left.length);
  for (let index = 0; index < left.length; index += 1) {
    output[index] = (left[index] ?? 0) * (right[index] ?? 0);
  }
  return output;
}

function tensorDiff(actual: Float32Array, expected: Float32Array): { maxAbs: number; maxRel: number } {
  if (actual.length !== expected.length) {
    return { maxAbs: Infinity, maxRel: Infinity };
  }
  let maxAbs = 0;
  let maxRel = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const abs = Math.abs((actual[index] ?? 0) - (expected[index] ?? 0));
    const rel = abs / Math.max(Math.abs(expected[index] ?? 0), 1e-12);
    maxAbs = Math.max(maxAbs, abs);
    maxRel = Math.max(maxRel, rel);
  }
  return { maxAbs, maxRel };
}

function checksum(values: Float32Array): number {
  let sum = 0;
  const stride = Math.max(1, Math.floor(values.length / 1024));
  for (let index = 0; index < values.length; index += stride) {
    sum += (values[index] ?? 0) * ((index % 17) + 1);
  }
  return sum;
}

function roundMs(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function quantizedWeightBytes(type: QuantizedType, inputSize: number, rowCount: number): Uint8Array {
  const rowByteLength = quantizedRowByteLength(type, inputSize);
  const bytes = new Uint8Array(rowByteLength * rowCount);
  let seed = 0xd000 + inputSize + rowCount + quantizedBlockByteLength(type);
  for (let index = 0; index < bytes.length; index += 1) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    bytes[index] = seed & 0xff;
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let row = 0; row < rowCount; row += 1) {
    const blockElements = type === "Q8_0" ? 32 : 256;
    for (let block = 0; block < inputSize / blockElements; block += 1) {
      const offset = row * rowByteLength + block * quantizedBlockByteLength(type);
      view.setUint16(offset, 0x3000, true);
      if (type === "Q4_K" || type === "Q5_K") {
        view.setUint16(offset + 2, 0x2800, true);
      } else if (type === "Q6_K") {
        view.setUint16(offset + 208, 0x3000, true);
      }
    }
  }
  return bytes;
}

function quantizedRowByteLength(type: QuantizedType, inputSize: number): number {
  if (type === "Q8_0") {
    return inputSize / 32 * 34;
  }
  return inputSize / 256 * quantizedBlockByteLength(type);
}

function quantizedBlockByteLength(type: QuantizedType): number {
  if (type === "Q4_K") {
    return 144;
  }
  if (type === "Q5_K") {
    return 176;
  }
  if (type === "Q6_K") {
    return 210;
  }
  return 34;
}

function sequence(length: number, seed: number): Float32Array {
  let value = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    output[index] = ((value / 0xffffffff) * 2 - 1) * 0.075;
  }
  return output;
}

function causalMask(tokenCount: number, keyValueTokenCount: number): Float32Array {
  const output = new Float32Array(tokenCount * keyValueTokenCount);
  for (let token = 0; token < tokenCount; token += 1) {
    for (let keyToken = 0; keyToken < keyValueTokenCount; keyToken += 1) {
      output[token * keyValueTokenCount + keyToken] = keyToken <= token ? 0 : -Infinity;
    }
  }
  return output;
}

function matMulShape(size: BrowserBenchSize): { inputSize: number; rowCount: number; columnCount: number } {
  if (size === "large") {
    return { inputSize: 2048, rowCount: 2048, columnCount: 16 };
  }
  if (size === "medium") {
    return { inputSize: 1024, rowCount: 1024, columnCount: 8 };
  }
  return { inputSize: 256, rowCount: 256, columnCount: 4 };
}

function gqaShape(size: BrowserBenchSize): Required<GqaAttentionOptions> {
  if (size === "large") {
    return { headSize: 128, queryHeadCount: 16, keyValueHeadCount: 4, tokenCount: 64, keyValueTokenCount: 64, scale: 1 / Math.sqrt(128), causal: true, mask: new Float32Array(), valueLayout: "dim-head-token", quantizeQueryForScore: "f16" };
  }
  if (size === "medium") {
    return { headSize: 64, queryHeadCount: 8, keyValueHeadCount: 2, tokenCount: 32, keyValueTokenCount: 32, scale: 1 / Math.sqrt(64), causal: true, mask: new Float32Array(), valueLayout: "dim-head-token", quantizeQueryForScore: "f16" };
  }
  return { headSize: 32, queryHeadCount: 4, keyValueHeadCount: 2, tokenCount: 8, keyValueTokenCount: 8, scale: 1 / Math.sqrt(32), causal: true, mask: new Float32Array(), valueLayout: "dim-head-token", quantizeQueryForScore: "f16" };
}

function ffnShape(size: BrowserBenchSize): { inputSize: number; hiddenSize: number; outputSize: number; columnCount: number } {
  if (size === "large") {
    return { inputSize: 2048, hiddenSize: 4096, outputSize: 2048, columnCount: 8 };
  }
  if (size === "medium") {
    return { inputSize: 1024, hiddenSize: 2048, outputSize: 1024, columnCount: 8 };
  }
  return { inputSize: 256, hiddenSize: 512, outputSize: 256, columnCount: 4 };
}

function fullAttentionShape(size: BrowserBenchSize): {
  headSize: number;
  queryHeadCount: number;
  keyValueHeadCount: number;
  keyValueTokenCount: number;
  contextLength: number;
  scale: number;
  outputSize: number;
} {
  if (size === "large") {
    return { headSize: 128, queryHeadCount: 16, keyValueHeadCount: 4, keyValueTokenCount: 128, contextLength: 128, scale: 1 / Math.sqrt(128), outputSize: 2048 };
  }
  if (size === "medium") {
    return { headSize: 64, queryHeadCount: 8, keyValueHeadCount: 2, keyValueTokenCount: 64, contextLength: 64, scale: 1 / Math.sqrt(64), outputSize: 512 };
  }
  return { headSize: 64, queryHeadCount: 4, keyValueHeadCount: 2, keyValueTokenCount: 16, contextLength: 16, scale: 1 / Math.sqrt(64), outputSize: 256 };
}

function prefillAttentionShape(size: BrowserBenchSize): {
  kind: "sliding" | "full";
  tokenCount: number;
  queryHeadCount: number;
  keyValueHeadCount: number;
  headSize: number;
  valueSize: number;
  contextLength: number;
  keyValueTokenCount: number;
  slidingWindow?: number;
} {
  if (size === "large") {
    return {
      kind: "full",
      tokenCount: 64,
      queryHeadCount: 8,
      keyValueHeadCount: 2,
      headSize: 512,
      valueSize: 512,
      contextLength: 4096,
      keyValueTokenCount: 4096,
    };
  }
  if (size === "medium") {
    return {
      kind: "full",
      tokenCount: 16,
      queryHeadCount: 8,
      keyValueHeadCount: 2,
      headSize: 512,
      valueSize: 512,
      contextLength: 1024,
      keyValueTokenCount: 1024,
    };
  }
  return {
    kind: "sliding",
    tokenCount: 16,
    queryHeadCount: 8,
    keyValueHeadCount: 2,
    headSize: 256,
    valueSize: 256,
    contextLength: 512,
    keyValueTokenCount: 512,
    slidingWindow: 512,
  };
}

function prefillAttentionPositions(tokenCount: number, keyValueTokenCount: number): Uint32Array {
  const positions = new Uint32Array(tokenCount);
  const start = Math.max(0, keyValueTokenCount - tokenCount);
  for (let index = 0; index < positions.length; index += 1) {
    positions[index] = start + index;
  }
  return positions;
}

function prefillAttentionProbabilityTokenCapacity(keyValueTokenCount: number): number {
  return Math.max(1, Math.ceil(keyValueTokenCount / 256) * 256);
}

function prefillAttentionProbabilityBytes(size: ReturnType<typeof prefillAttentionShape>): number {
  return size.tokenCount * size.queryHeadCount *
    prefillAttentionProbabilityTokenCapacity(size.keyValueTokenCount) *
    Float32Array.BYTES_PER_ELEMENT;
}

function prefillAttentionRollingTempBytes(size: ReturnType<typeof prefillAttentionShape>, tileSize: number): number {
  const headTokenCount = size.tokenCount * size.queryHeadCount;
  return (
    headTokenCount * tileSize +
    headTokenCount * 6
  ) * Float32Array.BYTES_PER_ELEMENT;
}

function prefillAttentionChunks(keyValueTokenCount: number, chunkSize: number): Array<[number, number]> {
  const chunks: Array<[number, number]> = [];
  const limit = Math.max(1, keyValueTokenCount);
  for (let start = 0; start < limit; start += chunkSize) {
    chunks.push([start, Math.min(keyValueTokenCount, start + chunkSize)]);
  }
  return chunks;
}

function formatBytes(value: number): string {
  if (value >= 1024 * 1024) {
    return `${(value / 1024 / 1024).toFixed(1)}MiB`;
  }
  if (value >= 1024) {
    return `${(value / 1024).toFixed(1)}KiB`;
  }
  return `${value}B`;
}

function seedFor(name: string, size: BrowserBenchSize, salt: number): number {
  let seed = salt * 0x9e3779b1;
  for (const char of `${name}:${size}`) {
    seed = (Math.imul(seed ^ char.charCodeAt(0), 16777619)) >>> 0;
  }
  return seed;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new Error("cancelled");
  }
}
