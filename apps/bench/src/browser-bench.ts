import {
  checkWebGpuSupport,
  runWebGpuSmokeTest,
  type WebGpuSupport,
} from "../../../packages/engine/src/runner/webgpu/index";
import { createWebGpuF32TensorHandle, createWebGpuQuantizedWeightHandle } from "../../../packages/engine/src/runner/webgpu/quantized-handles";
import {
  fullAttentionDecodeOutWebGpuResident,
  gatedDeltaNetWebGpu,
  matMulQ4_KWebGpu,
  matMulQ5_KWebGpu,
  matMulQ6_KWebGpu,
  matMulQ8_0WebGpu,
  matMulQkvConvWebGpuResident,
  matMulSsmNormGateOutWebGpuResident,
  matMulSwiGluDownWebGpuResident,
  matMulSwiGluWebGpuResident,
  matMulTop1WebGpuQuantizedResident,
  matMulWebGpuQuantizedResident,
  recurrentAttentionDecodeWebGpuResident,
} from "../../../packages/engine/src/runner/webgpu/matmul";
import { GPU_COPY_DST } from "../../../packages/engine/src/runner/webgpu/gpu-constants";
import { storageBuffer } from "../../../packages/engine/src/runner/webgpu/gpu-bindings";
import { createKMatMulBindResources, createQ8_0MatMulBindResources } from "../../../packages/engine/src/runner/webgpu/kernel-resources";
import {
  quantizeQ8_0Columns,
  quantizeQ8_KColumns,
  packBytesToU32,
  webGpuQuantizedWeightLayout,
} from "../../../packages/engine/src/runner/webgpu/quantized-handles";
import type { WebGpuBufferLike, WebGpuComputePassLike, WebGpuDeviceLike, WebGpuQuantizedWeightHandleInternal } from "../../../packages/engine/src/runner/webgpu/gpu-types";
import {
  gatedDeltaNet,
  gqaAttention,
  l2NormRows,
  sigmoid,
  silu,
  ssmConv1d,
  softplus,
  type GatedDeltaNetOptions,
  type GqaAttentionOptions,
} from "../../../packages/engine/src/ops";
import {
  createWasmQuantizedWeightHandle,
  gatedDeltaNetWasm,
  gqaAttentionWasm,
  matMulQuantizedWasm,
  matMulQuantizedWasmResident,
  prefillWasmBackend,
  releaseWasmQuantizedWeightHandle,
  ssmConv1dWasm,
  type WasmQuantizedWeightHandle,
} from "../../../packages/engine/src/runner/cpu/wasm-kernels";
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
  | "ssm-conv"
  | "gated-delta-net"
  | "gqa-attention"
  | "swiglu"
  | "swiglu-down"
  | "qkv-conv"
  | "ssm-norm-gate-out"
  | "full-attention-decode-out"
  | "recurrent-decode"
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
  variant: "reference" | "standalone" | "resident" | "fused";
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

type BenchMeasurement = {
  iterations: number;
  totalMs: number;
  gpuTotalMs?: number;
  output: BenchOutput;
};

type QuantizedType = "Q4_K" | "Q5_K" | "Q6_K" | "Q8_0";

const DEFAULT_CASES: BrowserBenchCaseId[] = [
  "matmul",
  "ssm-conv",
  "gated-delta-net",
  "gqa-attention",
  "swiglu",
  "swiglu-down",
  "qkv-conv",
  "ssm-norm-gate-out",
  "full-attention-decode-out",
  "recurrent-decode",
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
      await prepared.teardown?.();
      results.push(result);
      options.onResult?.(result);
    }
  }

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

function buildBenchTasks(caseIds: readonly BrowserBenchCaseId[], sizes: readonly BrowserBenchSize[]): BenchTask[] {
  const tasks: BenchTask[] = [];
  for (const size of sizes) {
    for (const caseId of caseIds) {
      if (caseId === "matmul") {
        for (const type of ["Q4_K", "Q5_K", "Q6_K", "Q8_0"] as const) {
          tasks.push(createMatMulTask(size, type, false));
          tasks.push(createMatMulTask(size, type, true));
        }
      } else if (caseId === "ssm-conv") {
        tasks.push(createSsmConvTask(size));
      } else if (caseId === "gated-delta-net") {
        tasks.push(createGatedDeltaTask(size));
      } else if (caseId === "gqa-attention") {
        tasks.push(createGqaAttentionTask(size));
      } else if (caseId === "swiglu") {
        tasks.push(createSwiGluTask(size));
      } else if (caseId === "swiglu-down") {
        tasks.push(createSwiGluDownTask(size));
      } else if (caseId === "qkv-conv") {
        tasks.push(createQkvConvTask(size));
      } else if (caseId === "ssm-norm-gate-out") {
        tasks.push(createSsmNormGateOutTask(size));
      } else if (caseId === "full-attention-decode-out") {
        tasks.push(createFullAttentionDecodeOutTask(size));
      } else if (caseId === "recurrent-decode") {
        tasks.push(createRecurrentDecodeTask(size));
      } else {
        tasks.push(createTopTokenTask(size));
      }
    }
  }
  return tasks;
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

function createSsmConvTask(sizeName: BrowserBenchSize): BenchTask {
  const size = ssmConvShape(sizeName);
  const inputWindow = size.kernelSize - 1 + size.tokenCount;
  const convInput = sequence(size.channelCount * inputWindow, seedFor("ssm", sizeName, 1));
  const kernel = sequence(size.channelCount * size.kernelSize, seedFor("ssm", sizeName, 2));
  return {
    caseId: "ssm-conv",
    caseName: "SSM conv1d",
    size: sizeName,
    shape: `channels=${size.channelCount} tokens=${size.tokenCount} kernel=${size.kernelSize}`,
    tolerance: 1e-4,
    relativeTolerance: 1e-4,
    referenceBackend: "wasm",
    reference: async () => (await ssmConv1dWasm(convInput, kernel, size.channelCount, size.tokenCount, size.kernelSize)) ??
      ssmConv1d(convInput, kernel, size.channelCount, size.tokenCount, size.kernelSize),
    candidates: [
      {
        backend: "ts-reference",
        variant: "standalone",
        run: async () => ssmConv1d(convInput, kernel, size.channelCount, size.tokenCount, size.kernelSize),
      },
    ],
  };
}

function createGatedDeltaTask(sizeName: BrowserBenchSize): BenchTask {
  const options = gatedDeltaShape(sizeName);
  const query = sequence(options.tokenCount * options.keyHeadCount * options.stateSize, seedFor("gdn", sizeName, 1));
  const key = sequence(query.length, seedFor("gdn", sizeName, 2));
  const value = sequence(options.tokenCount * options.valueHeadCount * options.stateSize, seedFor("gdn", sizeName, 3));
  const gate = sequence(options.tokenCount * options.valueHeadCount, seedFor("gdn", sizeName, 4));
  const beta = positive(sequence(options.tokenCount * options.valueHeadCount, seedFor("gdn", sizeName, 5)));
  const state = sequence(options.valueHeadCount * options.stateSize * options.stateSize, seedFor("gdn", sizeName, 6));
  const reference = async () => flattenGatedDeltaResult(
    (await gatedDeltaNetWasm(query, key, value, gate, beta, state, options)) ??
      gatedDeltaNet(query, key, value, gate, beta, state, options),
  );
  return {
    caseId: "gated-delta-net",
    caseName: "Gated DeltaNet",
    size: sizeName,
    shape: `state=${options.stateSize} kHeads=${options.keyHeadCount} vHeads=${options.valueHeadCount} tokens=${options.tokenCount}`,
    tolerance: 5e-3,
    relativeTolerance: 1e-4,
    referenceBackend: "wasm",
    reference,
    candidates: [
      {
        backend: "webgpu",
        variant: "standalone",
        run: async () => {
          const result = await gatedDeltaNetWebGpu(query, key, value, gate, beta, state, options);
          return result ? flattenGatedDeltaResult(result) : undefined;
        },
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

function createQkvConvTask(sizeName: BrowserBenchSize): BenchTask {
  const size = qkvConvShape(sizeName);
  const input = sequence(size.inputSize * size.tokenCount, seedFor("qkv", sizeName, 1));
  const convState = sequence((size.kernelSize - 1) * size.convDim, seedFor("qkv", sizeName, 2));
  const convKernel = sequence(size.kernelSize * size.convDim, seedFor("qkv", sizeName, 3));
  const qkvBytes = quantizedWeightBytes("Q4_K", size.inputSize, size.convDim);
  return {
    caseId: "qkv-conv",
    caseName: "QKV + conv fused",
    size: sizeName,
    shape: `input=${size.inputSize} conv=${size.convDim} tokens=${size.tokenCount} kernel=${size.kernelSize}`,
    tolerance: 1.5e-1,
    relativeTolerance: 1e-4,
    referenceBackend: "wasm",
    reference: async () => flattenQkvConvResult(await qkvConvReference(qkvBytes, input, convState, convKernel, size)),
    candidates: [
      {
        backend: "webgpu-resident",
        variant: "fused",
        prepare: async () => {
          const handle = await createWebGpuQuantizedWeightHandle("Q4_K", qkvBytes, size.inputSize, size.convDim);
          if (!handle) {
            return undefined;
          }
          return {
            run: async () => {
            const result = await matMulQkvConvWebGpuResident(handle, input, convState, convKernel, size);
            return result ? flattenQkvConvResult(result) : undefined;
            },
            teardown: () => handle.destroy(),
          };
        },
      },
    ],
  };
}

function createSsmNormGateOutTask(sizeName: BrowserBenchSize): BenchTask {
  const size = ssmOutShape(sizeName);
  const attnNorm = sequence(size.inputSize * size.columnCount, seedFor("ssmout", sizeName, 1));
  const delta = sequence(size.hiddenSize * size.columnCount, seedFor("ssmout", sizeName, 2));
  const normWeight = positive(sequence(size.hiddenSize, seedFor("ssmout", sizeName, 3)));
  const zBytes = quantizedWeightBytes("Q4_K", size.inputSize, size.hiddenSize);
  const outBytes = quantizedWeightBytes("Q8_0", size.hiddenSize, size.outputSize);
  return {
    caseId: "ssm-norm-gate-out",
    caseName: "SSM norm/gate/out fused",
    size: sizeName,
    shape: `input=${size.inputSize} hidden=${size.hiddenSize} output=${size.outputSize} cols=${size.columnCount}`,
    tolerance: 1.5e-1,
    relativeTolerance: 1e-4,
    referenceBackend: "wasm",
    reference: async () => ssmNormGateOutReference(zBytes, outBytes, attnNorm, delta, normWeight, size),
    candidates: [
      {
        backend: "webgpu-resident",
        variant: "fused",
        prepare: async () => {
          const z = await createWebGpuQuantizedWeightHandle("Q4_K", zBytes, size.inputSize, size.hiddenSize);
          const out = await createWebGpuQuantizedWeightHandle("Q8_0", outBytes, size.hiddenSize, size.outputSize);
          if (!z || !out) {
            z?.destroy();
            out?.destroy();
            return undefined;
          }
          return {
            run: () => matMulSsmNormGateOutWebGpuResident(z, out, attnNorm, delta, normWeight, size.epsilon, size.columnCount),
            teardown: () => {
              z.destroy();
              out.destroy();
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

function createRecurrentDecodeTask(sizeName: BrowserBenchSize): BenchTask {
  const size = recurrentShape(sizeName);
  const input = sequence(size.inputSize, seedFor("rec", sizeName, 1));
  const convState = sequence((size.kernelSize - 1) * size.convDim, seedFor("rec", sizeName, 2));
  const recurrentState = sequence(size.valueHeadCount * size.stateSize * size.stateSize, seedFor("rec", sizeName, 3));
  const qkvBytes = quantizedWeightBytes("Q4_K", size.inputSize, size.convDim);
  const zBytes = quantizedWeightBytes("Q5_K", size.inputSize, size.valueDim);
  const outBytes = quantizedWeightBytes("Q8_0", size.valueDim, size.outputSize);
  const alpha = sequence(size.inputSize * size.valueHeadCount, seedFor("rec", sizeName, 4));
  const beta = sequence(size.inputSize * size.valueHeadCount, seedFor("rec", sizeName, 5));
  const convKernel = sequence(size.convDim * size.kernelSize, seedFor("rec", sizeName, 6));
  const dtBias = sequence(size.valueHeadCount, seedFor("rec", sizeName, 7));
  const ssmA = negative(sequence(size.valueHeadCount, seedFor("rec", sizeName, 8)));
  const normWeight = positive(sequence(size.stateSize, seedFor("rec", sizeName, 9)));
  return {
    caseId: "recurrent-decode",
    caseName: "recurrent decode fused",
    size: sizeName,
    shape: `input=${size.inputSize} state=${size.stateSize} groups=${size.groupCount} vHeads=${size.valueHeadCount}`,
    tolerance: 2.5e-1,
    relativeTolerance: 1e-4,
    referenceBackend: "wasm",
    reference: async () => flattenRecurrentResult(await recurrentDecodeReference({
      qkvBytes,
      zBytes,
      outBytes,
      alpha,
      beta,
      convKernel,
      dtBias,
      ssmA,
      normWeight,
      input,
      convState,
      recurrentState,
      size,
    })),
    candidates: [
      {
        backend: "webgpu-resident",
        variant: "fused",
        prepare: async () => {
          const qkv = await createWebGpuQuantizedWeightHandle("Q4_K", qkvBytes, size.inputSize, size.convDim);
          const z = await createWebGpuQuantizedWeightHandle("Q5_K", zBytes, size.inputSize, size.valueDim);
          const out = await createWebGpuQuantizedWeightHandle("Q8_0", outBytes, size.valueDim, size.outputSize);
          const alphaHandle = await createWebGpuF32TensorHandle(alpha);
          const betaHandle = await createWebGpuF32TensorHandle(beta);
          const convKernelHandle = await createWebGpuF32TensorHandle(convKernel);
          const dtBiasHandle = await createWebGpuF32TensorHandle(dtBias);
          const ssmAHandle = await createWebGpuF32TensorHandle(ssmA);
          const normWeightHandle = await createWebGpuF32TensorHandle(normWeight);
          const handles = [qkv, z, out, alphaHandle, betaHandle, convKernelHandle, dtBiasHandle, ssmAHandle, normWeightHandle];
          if (handles.some((handle) => !handle)) {
            for (const handle of handles) {
              handle?.destroy();
            }
            return undefined;
          }
          return {
            run: async () => {
              const result = await recurrentAttentionDecodeWebGpuResident(
                {
                  qkv: qkv!,
                  z: z!,
                  out: out!,
                  alpha: alphaHandle!,
                  beta: betaHandle!,
                  convKernel: convKernelHandle!,
                  dtBias: dtBiasHandle!,
                  ssmA: ssmAHandle!,
                  normWeight: normWeightHandle!,
                },
                input,
                convState,
                recurrentState,
                size,
              );
              return result ? flattenRecurrentResult(result) : undefined;
            },
            teardown: () => {
              for (const handle of handles) {
                handle?.destroy();
              }
            },
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
      await device.queue.onSubmittedWorkDone?.();
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
  const packedWeight = packBytesToU32(weightBytes);
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

async function qkvConvReference(
  qkvBytes: Uint8Array,
  input: Float32Array,
  convState: Float32Array,
  convKernel: Float32Array,
  size: ReturnType<typeof qkvConvShape>,
): Promise<{ q: Float32Array; k: Float32Array; v: Float32Array; newConvState: Float32Array }> {
  const qkv = await matMulReference("Q4_K", qkvBytes, input, size.inputSize, size.convDim, size.tokenCount);
  return qkvConvSplitReference(qkv, convState, convKernel, size);
}

function qkvConvSplitReference(
  qkv: Float32Array,
  convState: Float32Array,
  convKernel: Float32Array,
  size: ReturnType<typeof qkvConvShape>,
): { q: Float32Array; k: Float32Array; v: Float32Array; newConvState: Float32Array } {
  const keyDim = size.stateSize * size.groupCount;
  const valueHeadCount = size.valueDim / size.stateSize;
  const history = size.kernelSize - 1;
  const q = new Float32Array(keyDim * size.tokenCount);
  const k = new Float32Array(keyDim * size.tokenCount);
  const v = new Float32Array(size.valueDim * size.tokenCount);
  const newConvState = new Float32Array(history * size.convDim);
  const convValue = (channel: number, inputIndex: number): number => {
    if (inputIndex < history) {
      return convState[channel * history + inputIndex] ?? 0;
    }
    return qkv[(inputIndex - history) * size.convDim + channel] ?? 0;
  };
  const convSilu = (channel: number, token: number): number => {
    let sum = 0;
    for (let index = 0; index < size.kernelSize; index += 1) {
      sum += convValue(channel, token + index) * (convKernel[channel * size.kernelSize + index] ?? 0);
    }
    return sum / (1 + Math.exp(-sum));
  };
  for (let token = 0; token < size.tokenCount; token += 1) {
    for (let group = 0; group < size.groupCount; group += 1) {
      const qRow = new Float32Array(size.stateSize);
      const kRow = new Float32Array(size.stateSize);
      for (let index = 0; index < size.stateSize; index += 1) {
        qRow[index] = convSilu(group * size.stateSize + index, token);
        kRow[index] = convSilu(keyDim + group * size.stateSize + index, token);
      }
      const qNorm = l2NormRows(qRow, size.stateSize, 1e-6);
      const kNorm = l2NormRows(kRow, size.stateSize, 1e-6);
      q.set(qNorm, token * keyDim + group * size.stateSize);
      k.set(kNorm, token * keyDim + group * size.stateSize);
    }
    for (let head = 0; head < valueHeadCount; head += 1) {
      for (let index = 0; index < size.stateSize; index += 1) {
        const valueIndex = head * size.stateSize + index;
        v[token * size.valueDim + valueIndex] = convSilu(keyDim * 2 + valueIndex, token);
      }
    }
  }
  for (let channel = 0; channel < size.convDim; channel += 1) {
    for (let index = 0; index < history; index += 1) {
      newConvState[channel * history + index] = convValue(channel, size.tokenCount + index);
    }
  }
  return { q, k, v, newConvState };
}

async function ssmNormGateOutReference(
  zBytes: Uint8Array,
  outBytes: Uint8Array,
  attnNorm: Float32Array,
  delta: Float32Array,
  normWeight: Float32Array,
  size: ReturnType<typeof ssmOutShape>,
): Promise<Float32Array> {
  const z = await matMulReference("Q4_K", zBytes, attnNorm, size.inputSize, size.hiddenSize, size.columnCount);
  const gated = ssmNormGateReference(delta, z, normWeight, size.hiddenSize, size.columnCount, size.epsilon);
  return matMulReference("Q8_0", outBytes, gated, size.hiddenSize, size.outputSize, size.columnCount);
}

function ssmNormGateReference(
  delta: Float32Array,
  z: Float32Array,
  normWeight: Float32Array,
  rowCount: number,
  columnCount: number,
  epsilon: number,
): Float32Array {
  const output = new Float32Array(delta.length);
  for (let column = 0; column < columnCount; column += 1) {
    const base = column * rowCount;
    const normalized = rmsNormRow(delta.subarray(base, base + rowCount), normWeight, epsilon);
    for (let row = 0; row < rowCount; row += 1) {
      const gate = z[base + row] ?? 0;
      output[base + row] = normalized[row]! * (gate / (1 + Math.exp(-gate)));
    }
  }
  return output;
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

async function recurrentDecodeReference(params: {
  qkvBytes: Uint8Array;
  zBytes: Uint8Array;
  outBytes: Uint8Array;
  alpha: Float32Array;
  beta: Float32Array;
  convKernel: Float32Array;
  dtBias: Float32Array;
  ssmA: Float32Array;
  normWeight: Float32Array;
  input: Float32Array;
  convState: Float32Array;
  recurrentState: Float32Array;
  size: ReturnType<typeof recurrentShape>;
}): Promise<{ attention: Float32Array; newConvState: Float32Array; newRecurrentState: Float32Array }> {
  const qkv = await matMulReference("Q4_K", params.qkvBytes, params.input, params.size.inputSize, params.size.convDim, 1);
  const z = await matMulReference("Q5_K", params.zBytes, params.input, params.size.inputSize, params.size.valueDim, 1);
  const alpha = f32MatMul(params.alpha, params.input, params.size.inputSize, params.size.valueHeadCount, 1);
  const beta = f32MatMul(params.beta, params.input, params.size.inputSize, params.size.valueHeadCount, 1);
  const qkvConv = qkvConvSplitReference(qkv, params.convState, params.convKernel, {
    inputSize: params.size.inputSize,
    convDim: params.size.convDim,
    tokenCount: 1,
    kernelSize: params.size.kernelSize,
    stateSize: params.size.stateSize,
    groupCount: params.size.groupCount,
    valueDim: params.size.valueDim,
  });
  const gate = multiply(softplus(addBias(alpha, params.dtBias)), params.ssmA);
  const betaSigmoid = sigmoid(beta);
  const delta = gatedDeltaNet(qkvConv.q, qkvConv.k, qkvConv.v, gate, betaSigmoid, params.recurrentState, {
    stateSize: params.size.stateSize,
    keyHeadCount: params.size.groupCount,
    valueHeadCount: params.size.valueHeadCount,
    tokenCount: 1,
  });
  const gated = ssmNormGateReference(delta.output, z, params.normWeight, params.size.stateSize, params.size.valueHeadCount, params.size.epsilon);
  const attention = await matMulReference("Q8_0", params.outBytes, gated, params.size.valueDim, params.size.outputSize, 1);
  return {
    attention,
    newConvState: qkvConv.newConvState,
    newRecurrentState: delta.newState,
  };
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

function f32MatMul(weightRows: Float32Array, inputColumns: Float32Array, inputSize: number, rowCount: number, columnCount: number): Float32Array {
  const output = new Float32Array(rowCount * columnCount);
  for (let column = 0; column < columnCount; column += 1) {
    for (let row = 0; row < rowCount; row += 1) {
      let sum = 0;
      for (let index = 0; index < inputSize; index += 1) {
        sum += (weightRows[row * inputSize + index] ?? 0) * (inputColumns[column * inputSize + index] ?? 0);
      }
      output[column * rowCount + row] = sum;
    }
  }
  return output;
}

function addBias(values: Float32Array, bias: Float32Array): Float32Array {
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = (values[index] ?? 0) + (bias[index % bias.length] ?? 0);
  }
  return output;
}

function multiply(left: Float32Array, right: Float32Array): Float32Array {
  const output = new Float32Array(left.length);
  for (let index = 0; index < left.length; index += 1) {
    output[index] = (left[index] ?? 0) * (right[index] ?? 0);
  }
  return output;
}

function rmsNormRow(input: Float32Array, weight: Float32Array, epsilon: number): Float32Array {
  let sumSquares = 0;
  for (const value of input) {
    sumSquares += value * value;
  }
  const scale = 1 / Math.sqrt(sumSquares / input.length + epsilon);
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = (input[index] ?? 0) * scale * (weight[index] ?? 0);
  }
  return output;
}

function flattenGatedDeltaResult(result: { output: Float32Array; newState: Float32Array }): Float32Array {
  return concatFloat32([result.output, result.newState]);
}

function flattenQkvConvResult(result: { q: Float32Array; k: Float32Array; v: Float32Array; newConvState: Float32Array }): Float32Array {
  return concatFloat32([result.q, result.k, result.v, result.newConvState]);
}

function flattenRecurrentResult(result: { attention: Float32Array; newConvState: Float32Array; newRecurrentState: Float32Array }): Float32Array {
  return concatFloat32([result.attention, result.newConvState, result.newRecurrentState]);
}

function concatFloat32(values: readonly Float32Array[]): Float32Array {
  const output = new Float32Array(values.reduce((sum, value) => sum + value.length, 0));
  let offset = 0;
  for (const value of values) {
    output.set(value, offset);
    offset += value.length;
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

function positive(values: Float32Array): Float32Array {
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = Math.abs(values[index] ?? 0) + 0.01;
  }
  return output;
}

function negative(values: Float32Array): Float32Array {
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = -Math.abs(values[index] ?? 0) - 0.01;
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

function ssmConvShape(size: BrowserBenchSize): { channelCount: number; tokenCount: number; kernelSize: number } {
  if (size === "large") {
    return { channelCount: 4096, tokenCount: 128, kernelSize: 4 };
  }
  if (size === "medium") {
    return { channelCount: 2048, tokenCount: 64, kernelSize: 4 };
  }
  return { channelCount: 128, tokenCount: 8, kernelSize: 4 };
}

function gatedDeltaShape(size: BrowserBenchSize): GatedDeltaNetOptions {
  if (size === "large") {
    return { stateSize: 128, keyHeadCount: 4, valueHeadCount: 16, tokenCount: 64 };
  }
  if (size === "medium") {
    return { stateSize: 64, keyHeadCount: 4, valueHeadCount: 8, tokenCount: 32 };
  }
  return { stateSize: 16, keyHeadCount: 2, valueHeadCount: 4, tokenCount: 8 };
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

function qkvConvShape(size: BrowserBenchSize): {
  inputSize: number;
  convDim: number;
  tokenCount: number;
  kernelSize: number;
  stateSize: number;
  groupCount: number;
  valueDim: number;
} {
  if (size === "large") {
    return { inputSize: 2048, convDim: 3072, tokenCount: 64, kernelSize: 4, stateSize: 128, groupCount: 4, valueDim: 2048 };
  }
  if (size === "medium") {
    return { inputSize: 1024, convDim: 1536, tokenCount: 32, kernelSize: 4, stateSize: 64, groupCount: 4, valueDim: 1024 };
  }
  return { inputSize: 256, convDim: 256, tokenCount: 8, kernelSize: 4, stateSize: 32, groupCount: 2, valueDim: 128 };
}

function ssmOutShape(size: BrowserBenchSize): { inputSize: number; hiddenSize: number; outputSize: number; columnCount: number; epsilon: number } {
  const ffn = ffnShape(size);
  return { inputSize: ffn.inputSize, hiddenSize: ffn.inputSize, outputSize: ffn.outputSize, columnCount: ffn.columnCount, epsilon: 1e-6 };
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

function recurrentShape(size: BrowserBenchSize): {
  inputSize: number;
  outputSize: number;
  convDim: number;
  kernelSize: number;
  stateSize: number;
  groupCount: number;
  valueHeadCount: number;
  valueDim: number;
  epsilon: number;
} {
  if (size === "large") {
    return { inputSize: 2048, outputSize: 2048, convDim: 3072, kernelSize: 4, stateSize: 128, groupCount: 4, valueHeadCount: 16, valueDim: 2048, epsilon: 1e-6 };
  }
  if (size === "medium") {
    return { inputSize: 1024, outputSize: 1024, convDim: 1536, kernelSize: 4, stateSize: 64, groupCount: 4, valueHeadCount: 16, valueDim: 1024, epsilon: 1e-6 };
  }
  return { inputSize: 256, outputSize: 256, convDim: 256, kernelSize: 4, stateSize: 32, groupCount: 2, valueHeadCount: 4, valueDim: 128, epsilon: 1e-6 };
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
