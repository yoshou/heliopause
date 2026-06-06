import assert from "node:assert/strict";
import test from "node:test";

import {
  dequantizeQ8_0,
  float16ToFloat32,
  float32ToFloat16,
  tensorByteLength,
} from "../src/index.ts";

test("float16 conversion matches common exact values", () => {
  assert.equal(float16ToFloat32(0x0000), 0);
  assert.equal(float16ToFloat32(0x3c00), 1);
  assert.equal(float16ToFloat32(0xc000), -2);
  assert.equal(float16ToFloat32(0x7c00), Infinity);
});

test("float32 to float16 rounds ties to even", () => {
  assert.equal(float32ToFloat16(1 + 0.5 / 1024), 0x3c00);
  assert.equal(float32ToFloat16(1 + 1.5 / 1024), 0x3c02);
});

test("Q8_0 dequantizes one block", () => {
  const bytes = new Uint8Array(34);
  const view = new DataView(bytes.buffer);
  view.setUint16(0, 0x3c00, true);
  for (let index = 0; index < 32; index += 1) {
    bytes[2 + index] = index < 16 ? index : 256 - (index - 15);
  }

  const values = dequantizeQ8_0(bytes, 32);

  assert.equal(values[0], 0);
  assert.equal(values[15], 15);
  assert.equal(values[16], -1);
  assert.equal(values[31], -16);
});

test("tensorByteLength uses ggml block sizes for observed types", () => {
  assert.equal(tensorByteLength(tensor("f32", "F32", [4])), 16);
  assert.equal(tensorByteLength(tensor("q8", "Q8_0", [32])), 34);
  assert.equal(tensorByteLength(tensor("q4k", "Q4_K", [256])), 144);
  assert.equal(tensorByteLength(tensor("q5k", "Q5_K", [256])), 176);
  assert.equal(tensorByteLength(tensor("q6k", "Q6_K", [256])), 210);
  assert.equal(tensorByteLength(tensor("iq4xs", "IQ4_XS", [256])), 136);
});

function tensor(name, type, dimensions) {
  return {
    name,
    type,
    dimensions,
    typeId: 0,
    offset: 0n,
    dataOffset: 0n,
  };
}
