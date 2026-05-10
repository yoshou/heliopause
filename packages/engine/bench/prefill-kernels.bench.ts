import { performance } from "node:perf_hooks";

import { gqaAttention } from "../src/ops.ts";
import {
  gqaAttentionWasm,
  matMulQuantizedWasm,
  prefillWasmBackend,
} from "../src/runner/cpu/wasm-kernels.ts";
import {
  quantizeQ8_0,
  quantizeQ8_K,
  vecDotIQ4_XS_Q8_K,
  vecDotQ4_K_Q8_K,
  vecDotQ6_K_Q8_K,
  vecDotQ8_0_Q8_0,
} from "../src/quant.ts";

type BenchResult = {
  name: string;
  shape: string;
  iterations: number;
  totalMs: number;
  checksum: number;
};

const minimumMs = Number(process.env.BENCH_MIN_MS ?? 750);
const warmupIterations = Number(process.env.BENCH_WARMUP ?? 8);
const backend = await prefillWasmBackend();

console.log(`prefill kernel benchmark`);
console.log(`backend=${backend} minMs=${minimumMs} warmup=${warmupIterations}`);

for (const size of [
  { name: "smoke", type: "Q4_K" as const, inputSize: 256, rowCount: 256, columnCount: 4 },
  { name: "medium", type: "Q4_K" as const, inputSize: 1024, rowCount: 1024, columnCount: 16 },
  { name: "large", type: "Q4_K" as const, inputSize: 2048, rowCount: 2048, columnCount: 32 },
  { name: "medium", type: "Q6_K" as const, inputSize: 1024, rowCount: 1024, columnCount: 16 },
  { name: "large", type: "Q6_K" as const, inputSize: 2048, rowCount: 2048, columnCount: 32 },
  { name: "medium", type: "IQ4_XS" as const, inputSize: 1024, rowCount: 1024, columnCount: 16 },
  { name: "large", type: "IQ4_XS" as const, inputSize: 2048, rowCount: 2048, columnCount: 32 },
  { name: "smoke", type: "Q8_0" as const, inputSize: 256, rowCount: 256, columnCount: 4 },
  { name: "medium", type: "Q8_0" as const, inputSize: 1024, rowCount: 1024, columnCount: 16 },
  { name: "large", type: "Q8_0" as const, inputSize: 2048, rowCount: 2048, columnCount: 32 },
]) {
  const weightBytes = quantizedWeightBytes(size.type, size.inputSize, size.rowCount);
  const inputColumns = sequence(size.inputSize * size.columnCount, 0x9000 + size.rowCount);
  const shape = `type=${size.type} in=${size.inputSize} rows=${size.rowCount} cols=${size.columnCount}`;

  printResult(await runBench(`ts:matMul:${size.name}:${size.type}`, shape, () =>
    matMulQuantizedTs(size.type, weightBytes, inputColumns, size.inputSize, size.rowCount, size.columnCount),
  ));

  printResult(await runBench(`wasm-simd:matMul:${size.name}:${size.type}`, shape, async () =>
    (await matMulQuantizedWasm(size.type, weightBytes, inputColumns, size.inputSize, size.rowCount, size.columnCount)) ??
      matMulQuantizedTs(size.type, weightBytes, inputColumns, size.inputSize, size.rowCount, size.columnCount),
  ));
}

for (const size of [
  { name: "smoke", headSize: 32, queryHeadCount: 4, keyValueHeadCount: 2, tokenCount: 8, keyValueTokenCount: 8 },
  { name: "medium", headSize: 128, queryHeadCount: 16, keyValueHeadCount: 4, tokenCount: 32, keyValueTokenCount: 32 },
  { name: "large", headSize: 128, queryHeadCount: 32, keyValueHeadCount: 8, tokenCount: 128, keyValueTokenCount: 128 },
]) {
  const query = sequence(size.tokenCount * size.queryHeadCount * size.headSize, 0xa000 + size.tokenCount);
  const key = sequence(size.keyValueTokenCount * size.keyValueHeadCount * size.headSize, 0xb000 + size.headSize);
  const value = sequence(size.headSize * size.keyValueHeadCount * size.keyValueTokenCount, 0xc000 + size.queryHeadCount);
  const mask = causalMask(size.tokenCount, size.keyValueTokenCount);
  const options = {
    headSize: size.headSize,
    queryHeadCount: size.queryHeadCount,
    keyValueHeadCount: size.keyValueHeadCount,
    tokenCount: size.tokenCount,
    keyValueTokenCount: size.keyValueTokenCount,
    scale: 1 / Math.sqrt(size.headSize),
    mask,
    valueLayout: "dim-head-token" as const,
    quantizeQueryForScore: "f16" as const,
  };
  const shape = `head=${size.headSize} qHeads=${size.queryHeadCount} kvHeads=${size.keyValueHeadCount} tokens=${size.tokenCount}/${size.keyValueTokenCount}`;

  printResult(await runBench(`ts:gqaAttention:${size.name}`, shape, () =>
    gqaAttention(query, key, value, options),
  ));

  printResult(await runBench(`wasm-simd:gqaAttention:${size.name}`, shape, async () =>
    (await gqaAttentionWasm(query, key, value, options)) ?? gqaAttention(query, key, value, options),
  ));
}

async function runBench(
  name: string,
  shape: string,
  run: () => Float32Array | Promise<Float32Array>,
): Promise<BenchResult> {
  let checksumValue = 0;
  for (let index = 0; index < warmupIterations; index += 1) {
    checksumValue += checksum(await run());
  }

  let iterations = 1;
  let totalMs = 0;
  do {
    const start = performance.now();
    checksumValue = 0;
    for (let index = 0; index < iterations; index += 1) {
      checksumValue += checksum(await run());
    }
    totalMs = performance.now() - start;
    if (totalMs < minimumMs) {
      iterations *= 2;
    }
  } while (totalMs < minimumMs);

  return {
    name,
    shape,
    iterations,
    totalMs,
    checksum: checksumValue / iterations,
  };
}

function printResult(result: BenchResult): void {
  const mean = result.totalMs / result.iterations;
  const ops = 1000 / mean;
  console.log([
    result.name.padEnd(32),
    result.shape.padEnd(48),
    `iters=${String(result.iterations).padStart(5)}`,
    `total=${result.totalMs.toFixed(1).padStart(8)}ms`,
    `mean=${mean.toFixed(4).padStart(9)}ms/op`,
    `ops=${ops.toFixed(2).padStart(9)}/s`,
    `checksum=${result.checksum.toExponential(8)}`,
  ].join("  "));
}

function sequence(length: number, seed: number): Float32Array {
  let value = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    output[index] = ((value / 0xffffffff) * 2 - 1) * 0.125;
  }
  return output;
}

function checksum(values: Float32Array): number {
  let sum = 0;
  const stride = Math.max(1, Math.floor(values.length / 1024));
  for (let index = 0; index < values.length; index += stride) {
    sum += (values[index] ?? 0) * ((index % 17) + 1);
  }
  return sum;
}

function matMulQuantizedTs(
  type: "Q4_K" | "Q6_K" | "IQ4_XS" | "Q8_0",
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
      const rowOffset = row * rowByteLength;
      const rowBytes = weightBytes.subarray(rowOffset, rowOffset + rowByteLength);
      if (type === "Q4_K") {
        output[column * rowCount + row] = vecDotQ4_K_Q8_K(rowBytes, q8 as ReturnType<typeof quantizeQ8_K>);
      } else if (type === "Q6_K") {
        output[column * rowCount + row] = vecDotQ6_K_Q8_K(rowBytes, q8 as ReturnType<typeof quantizeQ8_K>);
      } else if (type === "IQ4_XS") {
        output[column * rowCount + row] = vecDotIQ4_XS_Q8_K(rowBytes, q8 as ReturnType<typeof quantizeQ8_K>);
      } else {
        output[column * rowCount + row] = vecDotQ8_0_Q8_0(rowBytes, q8 as ReturnType<typeof quantizeQ8_0>);
      }
    }
  }
  return output;
}

function quantizedWeightBytes(
  type: "Q4_K" | "Q6_K" | "IQ4_XS" | "Q8_0",
  inputSize: number,
  rowCount: number,
): Uint8Array {
  const rowByteLength = quantizedRowByteLength(type, inputSize);
  const bytes = new Uint8Array(rowByteLength * rowCount);
  let seed = 0xd000 + inputSize + rowCount;
  for (let index = 0; index < bytes.length; index += 1) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    bytes[index] = seed & 0xff;
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let row = 0; row < rowCount; row += 1) {
    const blockElements = type === "Q8_0" ? 32 : 256;
    for (let block = 0; block < inputSize / blockElements; block += 1) {
      const offset = row * rowByteLength + block * quantizedBlockByteLength(type);
      view.setUint16(offset, 0x3c00, true);
      if (type === "Q4_K") {
        view.setUint16(offset + 2, 0x3400, true);
      } else if (type === "Q6_K") {
        view.setUint16(offset + 208, 0x3c00, true);
      } else if (type === "IQ4_XS") {
        view.setUint16(offset + 2, 0x2222, true);
      }
    }
  }
  return bytes;
}

function quantizedRowByteLength(type: "Q4_K" | "Q6_K" | "IQ4_XS" | "Q8_0", inputSize: number): number {
  if (type === "Q8_0") {
    return inputSize / 32 * 34;
  }
  return inputSize / 256 * quantizedBlockByteLength(type);
}

function quantizedBlockByteLength(type: "Q4_K" | "Q6_K" | "IQ4_XS" | "Q8_0"): number {
  if (type === "Q4_K") {
    return 144;
  }
  if (type === "Q6_K") {
    return 210;
  }
  if (type === "IQ4_XS") {
    return 136;
  }
  return 34;
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
