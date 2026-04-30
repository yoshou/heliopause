import type { GgmlTypeName } from "./gguf";

const QK_K = 256;
const KVALUES_IQ4_NL = new Int8Array([
  -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
]);

export function dequantizeRow(type: GgmlTypeName, bytes: Uint8Array, elements: number): Float32Array {
  switch (type) {
    case "F32":
      return new Float32Array(bytes.buffer, bytes.byteOffset, elements).slice();
    case "F16":
      return dequantizeF16(bytes, elements);
    case "Q8_0":
      return dequantizeQ8_0(bytes, elements);
    case "Q4_K":
      return dequantizeQ4_K(bytes, elements);
    case "Q5_K":
      return dequantizeQ5_K(bytes, elements);
    case "Q6_K":
      return dequantizeQ6_K(bytes, elements);
    case "IQ4_XS":
      return dequantizeIQ4_XS(bytes, elements);
    default:
      throw new Error(`Unsupported dequantization type: ${type}`);
  }
}

export function dequantizeF16(bytes: Uint8Array, elements: number): Float32Array {
  const output = new Float32Array(elements);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < elements; index += 1) {
    output[index] = float16ToFloat32(view.getUint16(index * 2, true));
  }
  return output;
}

export function dequantizeQ8_0(bytes: Uint8Array, elements: number): Float32Array {
  assertDivisible(elements, 32, "Q8_0");
  const output = new Float32Array(elements);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  let out = 0;
  for (let block = 0; block < elements / 32; block += 1) {
    const d = float16ToFloat32(view.getUint16(offset, true));
    offset += 2;
    for (let i = 0; i < 32; i += 1) {
      output[out++] = signedByte(bytes[offset + i] ?? 0) * d;
    }
    offset += 32;
  }
  return output;
}

export function dequantizeQ4_K(bytes: Uint8Array, elements: number): Float32Array {
  assertDivisible(elements, QK_K, "Q4_K");
  const output = new Float32Array(elements);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  let out = 0;
  for (let block = 0; block < elements / QK_K; block += 1) {
    const d = float16ToFloat32(view.getUint16(offset, true));
    const dmin = float16ToFloat32(view.getUint16(offset + 2, true));
    const scales = bytes.subarray(offset + 4, offset + 16);
    const qs = bytes.subarray(offset + 16, offset + 144);
    let qOffset = 0;
    let scaleIndex = 0;
    for (let j = 0; j < QK_K; j += 64) {
      const [sc1, min1] = getScaleMinK4(scaleIndex, scales);
      const [sc2, min2] = getScaleMinK4(scaleIndex + 1, scales);
      const d1 = d * sc1;
      const d2 = d * sc2;
      const m1 = dmin * min1;
      const m2 = dmin * min2;
      for (let l = 0; l < 32; l += 1) output[out++] = d1 * ((qs[qOffset + l] ?? 0) & 0x0f) - m1;
      for (let l = 0; l < 32; l += 1) output[out++] = d2 * ((qs[qOffset + l] ?? 0) >> 4) - m2;
      qOffset += 32;
      scaleIndex += 2;
    }
    offset += 144;
  }
  return output;
}

export function dequantizeQ5_K(bytes: Uint8Array, elements: number): Float32Array {
  assertDivisible(elements, QK_K, "Q5_K");
  const output = new Float32Array(elements);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  let out = 0;
  for (let block = 0; block < elements / QK_K; block += 1) {
    const d = float16ToFloat32(view.getUint16(offset, true));
    const dmin = float16ToFloat32(view.getUint16(offset + 2, true));
    const scales = bytes.subarray(offset + 4, offset + 16);
    const qh = bytes.subarray(offset + 16, offset + 48);
    const qs = bytes.subarray(offset + 48, offset + 176);
    let qOffset = 0;
    let hOffset = 0;
    let scaleIndex = 0;
    for (let j = 0; j < QK_K; j += 64) {
      const [sc1, min1] = getScaleMinK4(scaleIndex, scales);
      const [sc2, min2] = getScaleMinK4(scaleIndex + 1, scales);
      const d1 = d * sc1;
      const d2 = d * sc2;
      const m1 = dmin * min1;
      const m2 = dmin * min2;
      for (let l = 0; l < 32; l += 1) {
        const high = (qh[hOffset + l] ?? 0) & 1 ? 16 : 0;
        output[out++] = d1 * (((qs[qOffset + l] ?? 0) & 0x0f) + high) - m1;
      }
      for (let l = 0; l < 32; l += 1) {
        const high = (qh[hOffset + l] ?? 0) & 2 ? 16 : 0;
        output[out++] = d2 * (((qs[qOffset + l] ?? 0) >> 4) + high) - m2;
      }
      qOffset += 32;
      hOffset += 32;
      scaleIndex += 2;
    }
    offset += 176;
  }
  return output;
}

export function dequantizeQ6_K(bytes: Uint8Array, elements: number): Float32Array {
  assertDivisible(elements, QK_K, "Q6_K");
  const output = new Float32Array(elements);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  let outBase = 0;
  for (let block = 0; block < elements / QK_K; block += 1) {
    const ql = bytes.subarray(offset, offset + 128);
    const qh = bytes.subarray(offset + 128, offset + 192);
    const scales = bytes.subarray(offset + 192, offset + 208);
    const d = float16ToFloat32(view.getUint16(offset + 208, true));
    let qlOffset = 0;
    let qhOffset = 0;
    let scaleOffset = 0;
    for (let n = 0; n < QK_K; n += 128) {
      for (let l = 0; l < 32; l += 1) {
        const is = Math.floor(l / 16);
        const qhByte = qh[qhOffset + l] ?? 0;
        const q1 = (((ql[qlOffset + l] ?? 0) & 0x0f) | (((qhByte >> 0) & 3) << 4)) - 32;
        const q2 = (((ql[qlOffset + l + 32] ?? 0) & 0x0f) | (((qhByte >> 2) & 3) << 4)) - 32;
        const q3 = (((ql[qlOffset + l] ?? 0) >> 4) | (((qhByte >> 4) & 3) << 4)) - 32;
        const q4 = (((ql[qlOffset + l + 32] ?? 0) >> 4) | (((qhByte >> 6) & 3) << 4)) - 32;
        output[outBase + l] = d * signedByte(scales[scaleOffset + is + 0] ?? 0) * q1;
        output[outBase + l + 32] = d * signedByte(scales[scaleOffset + is + 2] ?? 0) * q2;
        output[outBase + l + 64] = d * signedByte(scales[scaleOffset + is + 4] ?? 0) * q3;
        output[outBase + l + 96] = d * signedByte(scales[scaleOffset + is + 6] ?? 0) * q4;
      }
      outBase += 128;
      qlOffset += 64;
      qhOffset += 32;
      scaleOffset += 8;
    }
    offset += 210;
  }
  return output;
}

export type QuantizedQ8K = {
  d: Float32Array;
  qs: Int8Array;
  bsums: Int16Array;
};

export type QuantizedQ8_0 = {
  d: Float32Array;
  qs: Int8Array;
};

export function quantizeQ8_K(input: Float32Array): QuantizedQ8K {
  assertDivisible(input.length, QK_K, "Q8_K");
  const blockCount = input.length / QK_K;
  const d = new Float32Array(blockCount);
  const qs = new Int8Array(input.length);
  const bsums = new Int16Array(blockCount * (QK_K / 16));

  for (let block = 0; block < blockCount; block += 1) {
    const base = block * QK_K;
    let max = 0;
    let amax = 0;
    for (let index = 0; index < QK_K; index += 1) {
      const value = input[base + index] ?? 0;
      const abs = Math.abs(value);
      if (abs > amax) {
        amax = abs;
        max = value;
      }
    }

    if (amax === 0) {
      d[block] = 0;
      continue;
    }

    const inverseScale = -127 / max;
    for (let index = 0; index < QK_K; index += 1) {
      qs[base + index] = Math.min(127, Math.round(inverseScale * (input[base + index] ?? 0)));
    }

    for (let group = 0; group < QK_K / 16; group += 1) {
      let sum = 0;
      for (let index = 0; index < 16; index += 1) {
        sum += qs[base + group * 16 + index] ?? 0;
      }
      bsums[block * (QK_K / 16) + group] = sum;
    }

    d[block] = 1 / inverseScale;
  }

  return { d, qs, bsums };
}

export function vecDotQ6_K_Q8_K(q6Bytes: Uint8Array, q8: QuantizedQ8K): number {
  const blockCount = q8.d.length;
  if (q6Bytes.length !== blockCount * 210) {
    throw new Error(`Q6_K/Q8_K block mismatch: q6=${q6Bytes.length} q8=${blockCount}`);
  }

  const view = new DataView(q6Bytes.buffer, q6Bytes.byteOffset, q6Bytes.byteLength);
  const sums = new Float32Array(8);
  const aux8 = new Int8Array(QK_K);

  for (let block = 0; block < blockCount; block += 1) {
    const offset = block * 210;
    const ql = q6Bytes.subarray(offset, offset + 128);
    const qh = q6Bytes.subarray(offset + 128, offset + 192);
    const scales = q6Bytes.subarray(offset + 192, offset + 208);
    const aux32 = new Int32Array(8);
    let qlOffset = 0;
    let qhOffset = 0;
    let auxOffset = 0;

    for (let group = 0; group < QK_K; group += 128) {
      for (let lane = 0; lane < 32; lane += 1) {
        const qhByte = qh[qhOffset + lane] ?? 0;
        aux8[auxOffset + lane] =
          (((ql[qlOffset + lane] ?? 0) & 0x0f) | (((qhByte >> 0) & 3) << 4)) - 32;
        aux8[auxOffset + lane + 32] =
          (((ql[qlOffset + lane + 32] ?? 0) & 0x0f) | (((qhByte >> 2) & 3) << 4)) - 32;
        aux8[auxOffset + lane + 64] =
          (((ql[qlOffset + lane] ?? 0) >> 4) | (((qhByte >> 4) & 3) << 4)) - 32;
        aux8[auxOffset + lane + 96] =
          (((ql[qlOffset + lane + 32] ?? 0) >> 4) | (((qhByte >> 6) & 3) << 4)) - 32;
      }
      qlOffset += 64;
      qhOffset += 32;
      auxOffset += 128;
    }

    let scaleIndex = 0;
    let valueIndex = 0;
    const q8Base = block * QK_K;
    for (let group = 0; group < QK_K / 16; group += 1) {
      const scale = signedByte(scales[scaleIndex++] ?? 0);
      for (let lane = 0; lane < 8; lane += 1) {
        aux32[lane] += scale * ((q8.qs[q8Base + valueIndex + lane] ?? 0) * (aux8[valueIndex + lane] ?? 0));
      }
      valueIndex += 8;
      for (let lane = 0; lane < 8; lane += 1) {
        aux32[lane] += scale * ((q8.qs[q8Base + valueIndex + lane] ?? 0) * (aux8[valueIndex + lane] ?? 0));
      }
      valueIndex += 8;
    }

    const d = float16ToFloat32(view.getUint16(offset + 208, true)) * (q8.d[block] ?? 0);
    for (let lane = 0; lane < 8; lane += 1) {
      sums[lane] += d * (aux32[lane] ?? 0);
    }
  }

  let sum = 0;
  for (let lane = 0; lane < 8; lane += 1) {
    sum += sums[lane] ?? 0;
  }
  return sum;
}

export function vecDotQ5_K_Q8_K(q5Bytes: Uint8Array, q8: QuantizedQ8K): number {
  const blockCount = q8.d.length;
  if (q5Bytes.length !== blockCount * 176) {
    throw new Error(`Q5_K/Q8_K block mismatch: q5=${q5Bytes.length} q8=${blockCount}`);
  }

  const view = new DataView(q5Bytes.buffer, q5Bytes.byteOffset, q5Bytes.byteLength);
  const sums = new Float32Array(8);
  const aux8 = new Int8Array(QK_K);
  let sumf = 0;

  for (let block = 0; block < blockCount; block += 1) {
    const offset = block * 176;
    const scalesMinBytes = q5Bytes.subarray(offset + 4, offset + 16);
    const qh = q5Bytes.subarray(offset + 16, offset + 48);
    const qs = q5Bytes.subarray(offset + 48, offset + 176);
    const scales = new Uint8Array(8);
    const mins = new Uint8Array(8);
    for (let index = 0; index < 8; index += 1) {
      const [scale, min] = getScaleMinK4(index, scalesMinBytes);
      scales[index] = scale;
      mins[index] = min;
    }

    let qOffset = 0;
    let out = 0;
    let highMask = 1;
    for (let group = 0; group < QK_K / 64; group += 1) {
      for (let lane = 0; lane < 32; lane += 1) {
        aux8[out + lane] = ((qs[qOffset + lane] ?? 0) & 0x0f) + ((qh[lane] ?? 0) & highMask ? 16 : 0);
      }
      out += 32;
      highMask <<= 1;
      for (let lane = 0; lane < 32; lane += 1) {
        aux8[out + lane] = ((qs[qOffset + lane] ?? 0) >> 4) + ((qh[lane] ?? 0) & highMask ? 16 : 0);
      }
      out += 32;
      highMask <<= 1;
      qOffset += 32;
    }

    let sumi = 0;
    const q8BlockOffset = block * (QK_K / 16);
    for (let group = 0; group < QK_K / 16; group += 1) {
      sumi += (q8.bsums[q8BlockOffset + group] ?? 0) * (mins[Math.floor(group / 2)] ?? 0);
    }

    const aux32 = new Int32Array(8);
    let valueIndex = 0;
    const q8Base = block * QK_K;
    for (let group = 0; group < QK_K / 32; group += 1) {
      const scale = scales[group] ?? 0;
      for (let chunk = 0; chunk < 4; chunk += 1) {
        for (let lane = 0; lane < 8; lane += 1) {
          aux32[lane] += scale *
            ((q8.qs[q8Base + valueIndex + lane] ?? 0) * (aux8[valueIndex + lane] ?? 0));
        }
        valueIndex += 8;
      }
    }

    const d = float16ToFloat32(view.getUint16(offset, true)) * (q8.d[block] ?? 0);
    for (let lane = 0; lane < 8; lane += 1) {
      sums[lane] += d * (aux32[lane] ?? 0);
    }
    const dmin = float16ToFloat32(view.getUint16(offset + 2, true)) * (q8.d[block] ?? 0);
    sumf -= dmin * sumi;
  }

  for (let lane = 0; lane < 8; lane += 1) {
    sumf += sums[lane] ?? 0;
  }
  return sumf;
}

export function vecDotQ4_K_Q8_K(q4Bytes: Uint8Array, q8: QuantizedQ8K): number {
  const blockCount = q8.d.length;
  if (q4Bytes.length !== blockCount * 144) {
    throw new Error(`Q4_K/Q8_K block mismatch: q4=${q4Bytes.length} q8=${blockCount}`);
  }

  const view = new DataView(q4Bytes.buffer, q4Bytes.byteOffset, q4Bytes.byteLength);
  const sums = new Float32Array(8);
  const aux8 = new Int8Array(QK_K);
  let sumf = 0;

  for (let block = 0; block < blockCount; block += 1) {
    const offset = block * 144;
    const scalesMinBytes = q4Bytes.subarray(offset + 4, offset + 16);
    const qs = q4Bytes.subarray(offset + 16, offset + 144);
    const scales = new Uint8Array(8);
    const mins = new Uint8Array(8);
    for (let index = 0; index < 8; index += 1) {
      const [scale, min] = getScaleMinK4(index, scalesMinBytes);
      scales[index] = scale;
      mins[index] = min;
    }

    let qOffset = 0;
    let out = 0;
    for (let group = 0; group < QK_K / 64; group += 1) {
      for (let lane = 0; lane < 32; lane += 1) {
        aux8[out + lane] = (qs[qOffset + lane] ?? 0) & 0x0f;
      }
      out += 32;
      for (let lane = 0; lane < 32; lane += 1) {
        aux8[out + lane] = (qs[qOffset + lane] ?? 0) >> 4;
      }
      out += 32;
      qOffset += 32;
    }

    let sumi = 0;
    const q8BlockOffset = block * (QK_K / 16);
    for (let group = 0; group < QK_K / 16; group += 1) {
      sumi += (q8.bsums[q8BlockOffset + group] ?? 0) * (mins[Math.floor(group / 2)] ?? 0);
    }

    const aux32 = new Int32Array(8);
    let valueIndex = 0;
    const q8Base = block * QK_K;
    for (let group = 0; group < QK_K / 32; group += 1) {
      const scale = scales[group] ?? 0;
      for (let chunk = 0; chunk < 4; chunk += 1) {
        for (let lane = 0; lane < 8; lane += 1) {
          aux32[lane] += scale *
            ((q8.qs[q8Base + valueIndex + lane] ?? 0) * (aux8[valueIndex + lane] ?? 0));
        }
        valueIndex += 8;
      }
    }

    const d = float16ToFloat32(view.getUint16(offset, true)) * (q8.d[block] ?? 0);
    for (let lane = 0; lane < 8; lane += 1) {
      sums[lane] += d * (aux32[lane] ?? 0);
    }
    const dmin = float16ToFloat32(view.getUint16(offset + 2, true)) * (q8.d[block] ?? 0);
    sumf -= dmin * sumi;
  }

  for (let lane = 0; lane < 8; lane += 1) {
    sumf += sums[lane] ?? 0;
  }
  return sumf;
}

export function vecDotIQ4_XS_Q8_K(iq4Bytes: Uint8Array, q8: QuantizedQ8K): number {
  const blockCount = q8.d.length;
  if (iq4Bytes.length !== blockCount * 136) {
    throw new Error(`IQ4_XS/Q8_K block mismatch: iq4=${iq4Bytes.length} q8=${blockCount}`);
  }

  const view = new DataView(iq4Bytes.buffer, iq4Bytes.byteOffset, iq4Bytes.byteLength);
  const accum = new Float32Array(8);

  for (let block = 0; block < blockCount; block += 1) {
    const offset = block * 136;
    const d4d8 = float16ToFloat32(view.getUint16(offset, true)) * (q8.d[block] ?? 0);
    let scalesH = view.getUint16(offset + 2, true);
    const scalesL = iq4Bytes.subarray(offset + 4, offset + 8);
    const qs = iq4Bytes.subarray(offset + 8, offset + 136);
    const q8Base = block * QK_K;
    let qsOffset = 0;
    let q8Offset = 0;
    const laneSums = new Int32Array(8);

    for (let ib = 0; ib < QK_K / 32; ib += 2) {
      const packedScale = scalesL[ib / 2] ?? 0;
      const ls1 = ((packedScale & 0x0f) | ((scalesH << 4) & 0x30)) - 32;
      const ls2 = ((packedScale >> 4) | ((scalesH << 2) & 0x30)) - 32;
      scalesH >>= 4;

      for (let j = 0; j < 32; j += 1) {
        const packed = qs[qsOffset + (j & 15)] ?? 0;
        const q4 = j < 16
          ? (KVALUES_IQ4_NL[packed & 0x0f] ?? 0)
          : (KVALUES_IQ4_NL[packed >> 4] ?? 0);
        laneSums[j >> 2] += ls1 * q4 * (q8.qs[q8Base + q8Offset + j] ?? 0);
      }
      qsOffset += 16;
      q8Offset += 32;

      for (let j = 0; j < 32; j += 1) {
        const packed = qs[qsOffset + (j & 15)] ?? 0;
        const q4 = j < 16
          ? (KVALUES_IQ4_NL[packed & 0x0f] ?? 0)
          : (KVALUES_IQ4_NL[packed >> 4] ?? 0);
        laneSums[j >> 2] += ls2 * q4 * (q8.qs[q8Base + q8Offset + j] ?? 0);
      }
      qsOffset += 16;
      q8Offset += 32;
    }

    for (let lane = 0; lane < 8; lane += 1) {
      accum[lane] = Math.fround(
        Math.fround(d4d8 * (laneSums[lane] ?? 0)) + (accum[lane] ?? 0),
      );
    }
  }

  let sumf = 0;
  for (let lane = 0; lane < 8; lane += 1) {
    sumf += accum[lane] ?? 0;
  }
  return sumf;
}

export function quantizeQ8_0(input: Float32Array): QuantizedQ8_0 {
  assertDivisible(input.length, 32, "Q8_0");
  const blockCount = input.length / 32;
  const d = new Float32Array(blockCount);
  const qs = new Int8Array(input.length);

  for (let block = 0; block < blockCount; block += 1) {
    const base = block * 32;
    let amax = 0;
    for (let index = 0; index < 32; index += 1) {
      amax = Math.max(amax, Math.abs(input[base + index] ?? 0));
    }

    const scale = float16ToFloat32(float32ToFloat16(amax / 127));
    const inverseScale = scale ? 1 / scale : 0;
    d[block] = scale;
    for (let index = 0; index < 32; index += 1) {
      qs[base + index] = Math.round((input[base + index] ?? 0) * inverseScale);
    }
  }

  return { d, qs };
}

export function vecDotQ8_0_Q8_0(q8Bytes: Uint8Array, input: QuantizedQ8_0): number {
  const blockCount = input.d.length;
  if (q8Bytes.length !== blockCount * 34) {
    throw new Error(`Q8_0 block mismatch: q8=${q8Bytes.length} input=${blockCount}`);
  }

  const view = new DataView(q8Bytes.buffer, q8Bytes.byteOffset, q8Bytes.byteLength);
  let sum = 0;
  for (let block = 0; block < blockCount; block += 1) {
    const offset = block * 34;
    let isum = 0;
    for (let index = 0; index < 32; index += 1) {
      isum += signedByte(q8Bytes[offset + 2 + index] ?? 0) * (input.qs[block * 32 + index] ?? 0);
    }
    sum += isum * (float16ToFloat32(view.getUint16(offset, true)) * (input.d[block] ?? 0));
  }
  return sum;
}

export function dequantizeIQ4_XS(bytes: Uint8Array, elements: number): Float32Array {
  assertDivisible(elements, QK_K, "IQ4_XS");
  const output = new Float32Array(elements);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  let out = 0;
  for (let block = 0; block < elements / QK_K; block += 1) {
    const d = float16ToFloat32(view.getUint16(offset, true));
    const scalesH = view.getUint16(offset + 2, true);
    const scalesL = bytes.subarray(offset + 4, offset + 8);
    const qs = bytes.subarray(offset + 8, offset + 136);
    let qOffset = 0;
    for (let ib = 0; ib < QK_K / 32; ib += 1) {
      const ls = (((scalesL[Math.floor(ib / 2)] ?? 0) >> (4 * (ib % 2))) & 0x0f) |
        (((scalesH >> (2 * ib)) & 3) << 4);
      const dl = d * (ls - 32);
      for (let j = 0; j < 16; j += 1) {
        output[out + j] = dl * (KVALUES_IQ4_NL[(qs[qOffset + j] ?? 0) & 0x0f] ?? 0);
        output[out + j + 16] = dl * (KVALUES_IQ4_NL[(qs[qOffset + j] ?? 0) >> 4] ?? 0);
      }
      out += 32;
      qOffset += 16;
    }
    offset += 136;
  }
  return output;
}

export function float16ToFloat32(value: number): number {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;

  if (exponent === 0) {
    return sign * Math.pow(2, -14) * (fraction / 1024);
  }
  if (exponent === 31) {
    return fraction === 0 ? sign * Infinity : NaN;
  }
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

export function float32ToFloat16(value: number): number {
  if (Number.isNaN(value)) {
    return 0x7e00;
  }
  if (value === Infinity) {
    return 0x7c00;
  }
  if (value === -Infinity) {
    return 0xfc00;
  }

  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const abs = Math.abs(value);
  if (abs === 0) {
    return sign;
  }
  if (abs >= 65504) {
    return sign | 0x7bff;
  }
  if (abs < 2 ** -24) {
    return sign;
  }

  let exponent = Math.floor(Math.log2(abs));
  let mantissa = abs / 2 ** exponent - 1;

  if (exponent < -14) {
    const subnormal = Math.round(abs / 2 ** -24);
    return sign | subnormal;
  }

  let halfMantissa = Math.round(mantissa * 1024);
  if (halfMantissa === 1024) {
    exponent += 1;
    halfMantissa = 0;
  }
  return sign | ((exponent + 15) << 10) | halfMantissa;
}

function getScaleMinK4(index: number, q: Uint8Array): [number, number] {
  if (index < 4) {
    return [(q[index] ?? 0) & 63, (q[index + 4] ?? 0) & 63];
  }
  return [
    ((q[index + 4] ?? 0) & 0x0f) | (((q[index - 4] ?? 0) >> 6) << 4),
    ((q[index + 4] ?? 0) >> 4) | (((q[index] ?? 0) >> 6) << 4),
  ];
}

function signedByte(value: number): number {
  return value > 127 ? value - 256 : value;
}

function assertDivisible(value: number, divisor: number, type: string): void {
  if (value % divisor !== 0) {
    throw new Error(`${type} requires element count divisible by ${divisor}, got ${value}`);
  }
}
