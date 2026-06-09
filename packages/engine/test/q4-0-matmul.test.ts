import assert from "node:assert/strict";
import test from "node:test";

import { float32ToFloat16, quantizeQ8_0, vecDotQ4_0_Q8_0 } from "../src/index.ts";
import { matMulWeight as matMulReferenceWeight } from "../src/runner/reference/layers.ts";
import {
  createWasmQuantizedWeightHandle,
  matMulQuantizedMultiWasm,
  matMulQuantizedWasm,
  matMulQuantizedWasmResident,
  matMulQuantizedWasmResidentMulti,
  releaseWasmQuantizedWeightHandle,
} from "../src/runner/wasm/wasm-kernels.ts";
import { splitRows, WasmThreadPool } from "../src/runner/wasm/thread-pool.ts";

const inputSize = 32;
const rowCount = 4;
const columnCount = 2;

test("Reference Q4_0 matmul matches Q4_0 x Q8_0 dot product", async () => {
  const weightBytes = q4_0WeightBytes(rowCount);
  const input = inputColumns();
  const output = await matMulReferenceWeight(referenceSession(weightBytes), "q4.weight", input);
  const expected = expectedMatmul(weightBytes, input);

  assertFloatClose(output, expected);
});

test("WASM Q4_0 matmul direct, resident, multi, and sharded resident match reference", async () => {
  const weightBytes = q4_0WeightBytes(rowCount);
  const secondWeightBytes = q4_0WeightBytes(rowCount, 2);
  const input = inputColumns();
  const expected = expectedMatmul(weightBytes, input);
  const expectedSecond = expectedMatmul(secondWeightBytes, input);

  const direct = await matMulQuantizedWasm("Q4_0", weightBytes, input, inputSize, rowCount, columnCount);
  assert.ok(direct);
  assertFloatClose(direct, expected);

  const multi = await matMulQuantizedMultiWasm([
    { type: "Q4_0", weightBytes, rowCount },
    { type: "Q4_0", weightBytes: secondWeightBytes, rowCount },
  ], input, inputSize, columnCount);
  assert.ok(multi);
  assertFloatClose(multi[0]!, expected);
  assertFloatClose(multi[1]!, expectedSecond);

  const handle = await createWasmQuantizedWeightHandle("Q4_0", weightBytes, inputSize, rowCount);
  const secondHandle = await createWasmQuantizedWeightHandle("Q4_0", secondWeightBytes, inputSize, rowCount);
  assert.ok(handle);
  assert.ok(secondHandle);
  try {
    const resident = await matMulQuantizedWasmResident(handle, input, inputSize, rowCount, columnCount);
    assert.ok(resident);
    assertFloatClose(resident, expected);

    const residentMulti = await matMulQuantizedWasmResidentMulti([handle, secondHandle], input, inputSize, columnCount);
    assert.ok(residentMulti);
    assertFloatClose(residentMulti[0]!, expected);
    assertFloatClose(residentMulti[1]!, expectedSecond);
  } finally {
    releaseWasmQuantizedWeightHandle(handle);
    releaseWasmQuantizedWeightHandle(secondHandle);
  }

  const pool = await WasmThreadPool.create(2);
  assert.ok(pool);
  try {
    const rowByteLength = 18;
    const shards = splitRows(rowCount, pool.workerCount).map((shard) => ({
      ...shard,
      weightBytes: weightBytes.slice(shard.rowStart * rowByteLength, (shard.rowStart + shard.rowCount) * rowByteLength),
    }));
    const sharded = await pool.prepareWeight("Q4_0", inputSize, rowCount, shards);
    assert.ok(sharded);
    const shardedOutput = await pool.matmul(sharded, input, inputSize, rowCount, columnCount);
    assert.ok(shardedOutput);
    assertFloatClose(shardedOutput, expected);
  } finally {
    pool.shutdown();
  }
});

function referenceSession(weightBytes: Uint8Array) {
  return {
    getTensor(name: string) {
      assert.equal(name, "q4.weight");
      return {
        name,
        type: "Q4_0" as const,
        typeId: 2,
        dimensions: [inputSize, rowCount],
        offset: 0n,
        dataOffset: 0n,
      };
    },
    hasProvider(name: string) {
      return name === "reference";
    },
    async readF32Tensor(): Promise<Float32Array> {
      throw new Error("Unexpected F32 tensor read");
    },
    async readWeightBytes(name: string): Promise<Uint8Array> {
      assert.equal(name, "q4.weight");
      return weightBytes;
    },
    tensorReader: undefined,
  };
}

function q4_0WeightBytes(rows: number, seed = 1): Uint8Array {
  const bytes = new Uint8Array(rows * 18);
  const view = new DataView(bytes.buffer);
  for (let row = 0; row < rows; row += 1) {
    const offset = row * 18;
    view.setUint16(offset, float32ToFloat16(0.125 * (row + seed)), true);
    for (let index = 0; index < 16; index += 1) {
      const low = (row + index + seed) % 16;
      const high = (row + 15 - index + seed) % 16;
      bytes[offset + 2 + index] = low | (high << 4);
    }
  }
  return bytes;
}

function inputColumns(): Float32Array {
  const values = new Float32Array(inputSize * columnCount);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = ((index % 9) - 4) * 0.25;
  }
  return values;
}

function expectedMatmul(weightBytes: Uint8Array, input: Float32Array): Float32Array {
  const output = new Float32Array(rowCount * columnCount);
  for (let column = 0; column < columnCount; column += 1) {
    const q8 = quantizeQ8_0(input.slice(column * inputSize, (column + 1) * inputSize));
    for (let row = 0; row < rowCount; row += 1) {
      output[column * rowCount + row] = vecDotQ4_0_Q8_0(weightBytes.subarray(row * 18, (row + 1) * 18), q8);
    }
  }
  return output;
}

function assertFloatClose(actual: Float32Array, expected: Float32Array): void {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(Math.abs((actual[index] ?? 0) - (expected[index] ?? 0)) < 1e-5, `${index}: ${actual[index]} !== ${expected[index]}`);
  }
}
