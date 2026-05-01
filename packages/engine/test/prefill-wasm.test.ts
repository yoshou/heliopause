import assert from "node:assert/strict";
import test from "node:test";

import {
  gatedDeltaNetWasm,
  createWasmQuantizedWeightHandle,
  gqaAttentionWasm,
  matMulQuantizedMultiWasm,
  matMulQuantizedWasmResidentMulti,
  matMulQuantizedWasmResident,
  matMulQuantizedWasm,
  prefillWasmBackend,
  releaseWasmQuantizedWeightHandle,
  resetPrefillWasmForTesting,
  ssmConv1dWasm,
} from "../src/prefill-wasm.ts";

test("prefill wasm falls back when SIMD module is unavailable", async () => {
  resetPrefillWasmForTesting("not valid wasm");
  assert.equal(await prefillWasmBackend(), "ts");
  assert.equal(
    await ssmConv1dWasm(new Float32Array(4), new Float32Array(4), 1, 1, 4),
    undefined,
  );
  resetPrefillWasmForTesting();
});

test("prefill wasm ssmConv1d returns fixed expected values", async (t) => {
  resetPrefillWasmForTesting();
  if (await prefillWasmBackend() !== "wasm-simd") {
    t.skip("WebAssembly SIMD module is not available in this runtime");
    return;
  }

  const values = await ssmConv1dWasm(
    new Float32Array([
      0.25, -0.5, 0.75, 1.25, -1.5, 2,
      -0.125, 0.5, -0.75, 1, 1.5, -2.25,
    ]),
    new Float32Array([0.5, -1, 0.25, 0.75, -0.5, 0.25, 1.5, -1.25]),
    2,
    3,
    4,
  );

  assert.deepEqual(Array.from(values ?? []), [1.75, -2.1875, -1.8125, -0.8125, 0.25, 5.6875]);
});

test("prefill wasm gatedDeltaNet returns fixed expected values", async (t) => {
  resetPrefillWasmForTesting();
  if (await prefillWasmBackend() !== "wasm-simd") {
    t.skip("WebAssembly SIMD module is not available in this runtime");
    return;
  }

  const result = await gatedDeltaNetWasm(
    sequence(8, 1),
    sequence(8, 2),
    sequence(16, 3),
    new Float32Array([-0.15, 0.05, 0.2, -0.1]),
    new Float32Array([0.25, 0.5, 0.75, 0.33]),
    sequence(32, 4),
    {
      stateSize: 4,
      keyHeadCount: 1,
      valueHeadCount: 2,
      tokenCount: 2,
    },
  );

  assertClose(result?.output, new Float32Array([
    0.0118371118, 0.00660444703, -0.00528165791, -0.00510338182,
    -0.000195065513, 0.00398186641, 0.0137605518, -0.0047510271,
    0.019597197, 0.00599416532, -0.00196952582, 0.0076890327,
    0.0118195312, -0.0070283832, 0.0053942916, -0.00589220878,
  ]));
  assertClose(result?.newState, new Float32Array([
    -0.147579566, 0.103487045, 0.0598437376, 0.162284493,
    -0.225323707, 0.0407634377, -0.240554214, -0.104687504,
    0.00342084002, 0.181839481, 0.063969098, -0.003385182,
    0.0199639145, 0.0720319375, 0.19087252, -0.0590300635,
    -0.0818610787, -0.0617225729, 0.039362561, -0.149209991,
    -0.103682347, 0.130055636, -0.214854792, 0.0229546241,
    -0.00932066329, -0.225598589, -0.061825715, 0.0878982246,
    0.101517826, 0.0352943204, 0.0837640837, 0.0649418458,
  ]));
});

test("prefill wasm quantized matmul returns fixed expected values", async (t) => {
  resetPrefillWasmForTesting();
  if (await prefillWasmBackend() !== "wasm-simd") {
    t.skip("WebAssembly SIMD module is not available in this runtime");
    return;
  }

  const weight = new Uint8Array(34 * 2);
  const view = new DataView(weight.buffer);
  view.setUint16(0, 0x3c00, true);
  view.setUint16(34, 0x4000, true);
  for (let index = 0; index < 32; index += 1) {
    weight[2 + index] = index < 16 ? index + 1 : 256 - (index - 15);
    weight[36 + index] = index % 2 === 0 ? 3 : 253;
  }
  const input = new Float32Array(64);
  for (let column = 0; column < 2; column += 1) {
    for (let index = 0; index < 32; index += 1) {
      input[column * 32 + index] = (index - 15.5) * (column === 0 ? 0.125 : -0.0625);
    }
  }

  const result = await matMulQuantizedWasm("Q8_0", weight, input, 32, 2, 2);

  assertClose(result, new Float32Array([-271.850586, -12.0849609, 135.925293, 6.04248047]), 1e-5);

  const q4Weight = new Uint8Array(144);
  const q4View = new DataView(q4Weight.buffer);
  q4View.setUint16(0, 0x3c00, true);
  q4View.setUint16(2, 0x3000, true);
  for (let index = 4; index < 16; index += 1) {
    q4Weight[index] = index * 3;
  }
  for (let index = 16; index < 144; index += 1) {
    q4Weight[index] = (index * 5 + 7) & 255;
  }
  const q4Input = new Float32Array(256);
  for (let index = 0; index < q4Input.length; index += 1) {
    q4Input[index] = ((index % 17) - 8) * 0.02;
  }

  assertClose(
    await matMulQuantizedWasm("Q4_K", q4Weight, q4Input, 256, 1, 1),
    new Float32Array([11.0337009]),
    1e-6,
  );
});

test("prefill wasm quantized multi matmul matches individual matmuls", async (t) => {
  resetPrefillWasmForTesting();
  if (await prefillWasmBackend() !== "wasm-simd") {
    t.skip("WebAssembly SIMD module is not available in this runtime");
    return;
  }

  const input = new Float32Array(64);
  for (let column = 0; column < 2; column += 1) {
    for (let index = 0; index < 32; index += 1) {
      input[column * 32 + index] = (index - 11.5) * (column === 0 ? 0.0625 : -0.03125);
    }
  }
  const left = q8Weight(2, 1);
  const right = q8Weight(3, 7);

  const individualLeft = await matMulQuantizedWasm("Q8_0", left, input, 32, 2, 2);
  const individualRight = await matMulQuantizedWasm("Q8_0", right, input, 32, 3, 2);
  const multi = await matMulQuantizedMultiWasm([
    { type: "Q8_0", weightBytes: left, rowCount: 2 },
    { type: "Q8_0", weightBytes: right, rowCount: 3 },
  ], input, 32, 2);

  assert.equal(multi?.length, 2);
  assertClose(multi?.[0], individualLeft, 1e-5);
  assertClose(multi?.[1], individualRight, 1e-5);
});

test("prefill wasm resident quantized matmul matches copied matmul", async (t) => {
  resetPrefillWasmForTesting();
  if (await prefillWasmBackend() !== "wasm-simd") {
    t.skip("WebAssembly SIMD module is not available in this runtime");
    return;
  }

  const input = new Float32Array(64);
  for (let column = 0; column < 2; column += 1) {
    for (let index = 0; index < 32; index += 1) {
      input[column * 32 + index] = (index - 8.5) * (column === 0 ? 0.05 : -0.075);
    }
  }
  const weight = q8Weight(3, 11);
  const expected = await matMulQuantizedWasm("Q8_0", weight, input, 32, 3, 2);
  const handle = await createWasmQuantizedWeightHandle("Q8_0", weight, 32, 3);
  assert.ok(handle);
  try {
    const actual = await matMulQuantizedWasmResident(handle, input, 32, 3, 2);
    assertClose(actual, expected, 1e-5);
  } finally {
    releaseWasmQuantizedWeightHandle(handle);
  }
});

test("prefill wasm resident multi matmul matches resident individual matmuls", async (t) => {
  resetPrefillWasmForTesting();
  if (await prefillWasmBackend() !== "wasm-simd") {
    t.skip("WebAssembly SIMD module is not available in this runtime");
    return;
  }

  const input = new Float32Array(64);
  for (let column = 0; column < 2; column += 1) {
    for (let index = 0; index < 32; index += 1) {
      input[column * 32 + index] = (index - 7.5) * (column === 0 ? 0.04 : -0.065);
    }
  }
  const leftWeight = q8Weight(2, 17);
  const rightWeight = q8Weight(3, 23);
  const leftHandle = await createWasmQuantizedWeightHandle("Q8_0", leftWeight, 32, 2);
  const rightHandle = await createWasmQuantizedWeightHandle("Q8_0", rightWeight, 32, 3);
  assert.ok(leftHandle);
  assert.ok(rightHandle);
  try {
    const left = await matMulQuantizedWasmResident(leftHandle, input, 32, 2, 2);
    const right = await matMulQuantizedWasmResident(rightHandle, input, 32, 3, 2);
    const multi = await matMulQuantizedWasmResidentMulti([leftHandle, rightHandle], input, 32, 2);
    assert.equal(multi?.length, 2);
    assertClose(multi?.[0], left, 1e-5);
    assertClose(multi?.[1], right, 1e-5);
  } finally {
    releaseWasmQuantizedWeightHandle(leftHandle);
    releaseWasmQuantizedWeightHandle(rightHandle);
  }
});

test("prefill wasm GQA attention returns fixed expected values", async (t) => {
  resetPrefillWasmForTesting();
  if (await prefillWasmBackend() !== "wasm-simd") {
    t.skip("WebAssembly SIMD module is not available in this runtime");
    return;
  }

  const result = await gqaAttentionWasm(
    new Float32Array([0.1, -0.2, 0.3, -0.4, 0.2, 0.1, -0.1, 0.05]),
    new Float32Array([0.05, 0.1, -0.2, 0.25, -0.15, 0.2, 0.1, -0.05]),
    new Float32Array([0.2, -0.1, 0.05, 0.4, -0.3, 0.25, 0.1, -0.2]),
    {
      headSize: 4,
      queryHeadCount: 1,
      keyValueHeadCount: 1,
      tokenCount: 2,
      keyValueTokenCount: 2,
      scale: 0.5,
      mask: new Float32Array([0, -Infinity, 0, 0]),
      valueLayout: "token-head-dim",
      quantizeQueryForScore: "f16",
    },
  );

  assertClose(result, new Float32Array([
    0.200000003, -0.100000001, 0.0500000007, 0.400000006,
    -0.0453142002, 0.0717199296, 0.0745314211, 0.105622977,
  ]), 1e-8);
});

function sequence(length: number, seed: number): Float32Array {
  let value = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    output[index] = ((value / 0xffffffff) * 2 - 1) * 0.25;
  }
  return output;
}

function q8Weight(rowCount: number, seed: number): Uint8Array {
  const weight = new Uint8Array(rowCount * 34);
  const view = new DataView(weight.buffer);
  for (let row = 0; row < rowCount; row += 1) {
    const offset = row * 34;
    view.setUint16(offset, 0x3c00 + row, true);
    for (let index = 0; index < 32; index += 1) {
      weight[offset + 2 + index] = (seed + row * 17 + index * 5) & 255;
    }
  }
  return weight;
}

function assertClose(actual: Float32Array | undefined, expected: Float32Array, tolerance = 1e-8): void {
  assert.ok(actual);
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    assert.ok(
      Math.abs((actual[index] ?? 0) - (expected[index] ?? 0)) <= tolerance,
      `index ${index}: actual=${actual[index]} expected=${expected[index]}`,
    );
  }
}
