import type { GgmlTypeName } from "../gguf";
import { GPU_COPY_DST, GPU_STORAGE } from "./gpu-constants";
import { storageBuffer } from "./gpu-bindings";
import { assertStorageBindingFits, webGpuDevice } from "./gpu-device";
import type { WebGpuBufferLike, WebGpuDeviceLike, WebGpuF32TensorHandle, WebGpuF32TensorHandleInternal, WebGpuQuantizedMatMulType, WebGpuQuantizedWeightHandle, WebGpuQuantizedWeightHandleInternal } from "./gpu-types";

export type GpuBufferAllocator = {
  readonly device: WebGpuDeviceLike;
  createBuffer(label: string, size: number, usage: number, mappedAtCreation?: boolean): WebGpuBufferLike;
};

export async function createWebGpuQuantizedWeightHandle(
  type: WebGpuQuantizedMatMulType,
  weightBytes: Uint8Array,
  inputSize: number,
  rowCount: number,
): Promise<WebGpuQuantizedWeightHandle | undefined> {
  const layout = webGpuQuantizedWeightLayout(type, inputSize);
  if (weightBytes.byteLength !== layout.rowByteLength * rowCount) {
    throw new Error(`WebGPU ${type} weight shape mismatch: ${weightBytes.byteLength}`);
  }
  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }
  const packedWeight = packBytesToU32(weightBytes);
  await assertStorageBindingFits(`${type} weight`, packedWeight.byteLength);
  const weightBuffer = storageBuffer(device, packedWeight.byteLength, GPU_COPY_DST);
  device.queue.writeBuffer(weightBuffer, 0, packedWeight);
  const handle: WebGpuQuantizedWeightHandleInternal = {
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
  return handle;
}

export async function createWebGpuF32TensorHandle(
  values: Float32Array,
): Promise<WebGpuF32TensorHandle | undefined> {
  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }
  await assertStorageBindingFits("F32 tensor", values.byteLength);
  const buffer = storageBuffer(device, values.byteLength, GPU_COPY_DST);
  device.queue.writeBuffer(buffer, 0, values);
  const handle: WebGpuF32TensorHandleInternal = {
    length: values.length,
    byteLength: values.byteLength,
    device,
    buffer,
    destroy: () => buffer.destroy?.(),
  };
  return handle;
}

export function webGpuQuantizedWeightLayout(
  type: WebGpuQuantizedMatMulType,
  inputSize: number,
): { blockCount: number; rowByteLength: number } {
  if (type === "Q8_0") {
    if (inputSize % 32 !== 0) {
      throw new Error(`WebGPU Q8_0 matmul input size must be divisible by 32, got ${inputSize}`);
    }
    const blockCount = inputSize / 32;
    return { blockCount, rowByteLength: blockCount * 34 };
  }
  if (inputSize % 256 !== 0) {
    throw new Error(`WebGPU ${type} matmul input size must be divisible by 256, got ${inputSize}`);
  }
  const blockCount = inputSize / 256;
  if (type === "Q4_K") {
    return { blockCount, rowByteLength: blockCount * 144 };
  }
  if (type === "Q5_K") {
    return { blockCount, rowByteLength: blockCount * 176 };
  }
  return { blockCount, rowByteLength: blockCount * 210 };
}

export function quantizeQ8_0Columns(
  inputColumns: Float32Array,
  inputSize: number,
  columnCount: number,
): { scales: Float32Array; qs: Int32Array } {
  const blockCount = inputSize / 32;
  const scales = new Float32Array(columnCount * blockCount);
  const qs = new Int32Array(inputColumns.length);
  for (let column = 0; column < columnCount; column += 1) {
    for (let block = 0; block < blockCount; block += 1) {
      const base = column * inputSize + block * 32;
      let amax = 0;
      for (let index = 0; index < 32; index += 1) {
        amax = Math.max(amax, Math.abs(inputColumns[base + index] ?? 0));
      }
      const scale = float16ToFloat32(float32ToFloat16(amax / 127));
      const inverseScale = scale ? 1 / scale : 0;
      scales[column * blockCount + block] = scale;
      for (let index = 0; index < 32; index += 1) {
        qs[base + index] = Math.round((inputColumns[base + index] ?? 0) * inverseScale);
      }
    }
  }
  return { scales, qs };
}

export function quantizeQ8_KColumns(
  inputColumns: Float32Array,
  inputSize: number,
  columnCount: number,
): { scales: Float32Array; qs: Int32Array; bsums: Int32Array } {
  const blockCount = inputSize / 256;
  const scales = new Float32Array(columnCount * blockCount);
  const qs = new Int32Array(inputColumns.length);
  const bsums = new Int32Array(columnCount * blockCount * 16);
  for (let column = 0; column < columnCount; column += 1) {
    for (let block = 0; block < blockCount; block += 1) {
      const base = column * inputSize + block * 256;
      let max = 0;
      let amax = 0;
      for (let index = 0; index < 256; index += 1) {
        const value = inputColumns[base + index] ?? 0;
        const abs = Math.abs(value);
        if (abs > amax) {
          amax = abs;
          max = value;
        }
      }
      if (amax === 0) {
        continue;
      }
      const inverseScale = -127 / max;
      scales[column * blockCount + block] = 1 / inverseScale;
      for (let index = 0; index < 256; index += 1) {
        qs[base + index] = Math.min(127, Math.round(inverseScale * (inputColumns[base + index] ?? 0)));
      }
      for (let group = 0; group < 16; group += 1) {
        let sum = 0;
        for (let index = 0; index < 16; index += 1) {
          sum += qs[base + group * 16 + index] ?? 0;
        }
        bsums[(column * blockCount + block) * 16 + group] = sum;
      }
    }
  }
  return { scales, qs, bsums };
}

export function packBytesToU32(bytes: Uint8Array): Uint32Array {
  const packed = new Uint32Array(Math.ceil(bytes.byteLength / 4));
  for (let index = 0; index < bytes.byteLength; index += 1) {
    packed[index >> 2] |= (bytes[index] ?? 0) << ((index & 3) * 8);
  }
  return packed;
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
  const mantissa = abs / 2 ** exponent - 1;
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

export function createQuantizedHandleFromBytes(
  arena: GpuBufferAllocator,
  label: string,
  type: WebGpuQuantizedMatMulType,
  inputSize: number,
  rowCount: number,
  bytes: Uint8Array,
): WebGpuQuantizedWeightHandleInternal {
  const layout = webGpuQuantizedWeightLayout(type, inputSize);
  const expected = layout.rowByteLength * rowCount;
  if (bytes.byteLength !== expected) {
    throw new Error(`WebGPU ${label} weight shape mismatch: ${bytes.byteLength} !== ${expected}`);
  }
  const packed = packBytesToU32(bytes);
  const weightBuffer = arena.createBuffer(label, packed.byteLength, GPU_STORAGE | GPU_COPY_DST);
  arena.device.queue.writeBuffer(weightBuffer, 0, packed);
  return {
    type,
    inputSize,
    rowCount,
    byteLength: bytes.byteLength,
    device: arena.device,
    weightBuffer,
    blockCount: layout.blockCount,
    rowByteLength: layout.rowByteLength,
    destroy: () => weightBuffer.destroy?.(),
  };
}

export function webGpuMatMulType(type: GgmlTypeName, name: string): WebGpuQuantizedMatMulType {
  if (type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "Q8_0") {
    return type;
  }
  throw new Error(`${name} has unsupported WebGPU suffix type ${type}`);
}
