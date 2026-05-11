import type { GqaAttentionOptions } from "../../ops";
import { PREFILL_WASM_SIMD_BASE64 } from "./wasm-kernels.generated";

type KernelExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  hp_alloc(byteLength: number): number;
  hp_dealloc(ptr: number, byteLength: number): void;
  hp_matmul_quantized_f32(
    typeId: number,
    weightPtr: number,
    weightLen: number,
    inputPtr: number,
    inputLen: number,
    inputSize: number,
    rowCount: number,
    columnCount: number,
    outputPtr: number,
    outputLen: number,
  ): number;
  hp_prepare_quantized_scales_f32(
    typeId: number,
    weightPtr: number,
    weightLen: number,
    inputSize: number,
    rowCount: number,
    scalePtr: number,
    scaleLen: number,
  ): number;
  hp_matmul_quantized_prepared_f32(
    typeId: number,
    weightPtr: number,
    weightLen: number,
    scalePtr: number,
    scaleLen: number,
    inputPtr: number,
    inputLen: number,
    inputSize: number,
    rowCount: number,
    columnCount: number,
    outputPtr: number,
    outputLen: number,
  ): number;
  hp_matmul_quantized_prepared_multi_f32(
    count: number,
    inputPtr: number,
    inputLen: number,
    inputSize: number,
    columnCount: number,
    typeId0: number,
    weightPtr0: number,
    weightLen0: number,
    scalePtr0: number,
    scaleLen0: number,
    rowCount0: number,
    outputPtr0: number,
    outputLen0: number,
    typeId1: number,
    weightPtr1: number,
    weightLen1: number,
    scalePtr1: number,
    scaleLen1: number,
    rowCount1: number,
    outputPtr1: number,
    outputLen1: number,
    typeId2: number,
    weightPtr2: number,
    weightLen2: number,
    scalePtr2: number,
    scaleLen2: number,
    rowCount2: number,
    outputPtr2: number,
    outputLen2: number,
    typeId3: number,
    weightPtr3: number,
    weightLen3: number,
    scalePtr3: number,
    scaleLen3: number,
    rowCount3: number,
    outputPtr3: number,
    outputLen3: number,
  ): number;
  hp_matmul_quantized_multi_f32(
    count: number,
    inputPtr: number,
    inputLen: number,
    inputSize: number,
    columnCount: number,
    typeId0: number,
    weightPtr0: number,
    weightLen0: number,
    rowCount0: number,
    outputPtr0: number,
    outputLen0: number,
    typeId1: number,
    weightPtr1: number,
    weightLen1: number,
    rowCount1: number,
    outputPtr1: number,
    outputLen1: number,
    typeId2: number,
    weightPtr2: number,
    weightLen2: number,
    rowCount2: number,
    outputPtr2: number,
    outputLen2: number,
    typeId3: number,
    weightPtr3: number,
    weightLen3: number,
    rowCount3: number,
    outputPtr3: number,
    outputLen3: number,
  ): number;
  hp_gqa_attention_f32(
    queryPtr: number,
    queryLen: number,
    keyPtr: number,
    keyLen: number,
    valuePtr: number,
    valueLen: number,
    maskPtr: number,
    maskLen: number,
    headSize: number,
    queryHeadCount: number,
    keyValueHeadCount: number,
    tokenCount: number,
    keyValueTokenCount: number,
    scale: number,
    valueLayout: number,
    quantizeQueryF16: number,
    outputPtr: number,
    outputLen: number,
  ): number;
  hp_vision_patch_embed_f32(
    pixelsPtr: number,
    pixelsLen: number,
    weightsPtr: number,
    weightsLen: number,
    imageWidth: number,
    patchSize: number,
    patchGridX: number,
    patchGridY: number,
    embeddingLength: number,
    outputPtr: number,
    outputLen: number,
  ): number;
  hp_vision_add_position_f32(
    hiddenPtr: number,
    hiddenLen: number,
    positionsPtr: number,
    positionsLen: number,
    patchGridX: number,
    tokenCount: number,
    embeddingLength: number,
    tableSize: number,
    outputPtr: number,
    outputLen: number,
  ): number;
  hp_vision_rms_norm_f32(
    inputPtr: number,
    inputLen: number,
    weightPtr: number,
    weightLen: number,
    rowSize: number,
    epsilon: number,
    outputPtr: number,
    outputLen: number,
  ): number;
  hp_vision_rope2d_neox_f32(
    inputPtr: number,
    inputLen: number,
    patchGridX: number,
    headSize: number,
    headCount: number,
    tokenCount: number,
    freqBase: number,
    outputPtr: number,
    outputLen: number,
  ): number;
  hp_vision_clamp_f32(
    inputPtr: number,
    inputLen: number,
    min: number,
    max: number,
    outputPtr: number,
    outputLen: number,
  ): number;
  hp_vision_gelu_mul_f32(
    gatePtr: number,
    gateLen: number,
    upPtr: number,
    upLen: number,
    outputPtr: number,
    outputLen: number,
  ): number;
  hp_vision_residual_add_f32(
    leftPtr: number,
    leftLen: number,
    rightPtr: number,
    rightLen: number,
    outputPtr: number,
    outputLen: number,
  ): number;
  hp_vision_average_pool_scale_f32(
    inputPtr: number,
    inputLen: number,
    patchGridX: number,
    patchGridY: number,
    embeddingLength: number,
    kernelSize: number,
    outputScale: number,
    outputPtr: number,
    outputLen: number,
  ): number;
  hp_vision_std_normalize_f32(
    inputPtr: number,
    inputLen: number,
    biasPtr: number,
    biasLen: number,
    scalePtr: number,
    scaleLen: number,
    rowSize: number,
    outputPtr: number,
    outputLen: number,
  ): number;
};

type Allocation = {
  ptr: number;
  byteLength: number;
  exports: KernelExports;
};

export type QuantizedMatMulInput = {
  type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0";
  weightBytes: Uint8Array;
  rowCount: number;
};

export type PrefillWasmTraceEvent = {
  kernel: string;
  section: "allocation + input copy" | "kernel call" | "output copy + free" | "resident weight copy" | "resident scale prepare";
  durationMs: number;
  bytes?: number;
};

export type PrefillWasmTrace = (event: PrefillWasmTraceEvent) => void;

let wasmBase64ForTesting: string | undefined;
let instancePromise: Promise<KernelExports | undefined> | undefined;
let modulePromise: Promise<WebAssembly.Module | undefined> | undefined;
let wasmTrace: PrefillWasmTrace | undefined;
const scratchPool: Allocation[] = [];
const maxScratchPoolEntries = 16;
const maxScratchPoolBytes = 512 * 1024 * 1024;
let scratchPoolBytes = 0;
const residentInstances: ResidentWasmInstance[] = [];
let nextResidentInstanceId = 1;
const maxResidentInstanceBytes = 3 * 1024 * 1024 * 1024;

export type WasmQuantizedWeightHandle = {
  instanceId: number;
  type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0";
  ptr: number;
  byteLength: number;
  scalePtr: number;
  scaleByteLength: number;
  scaleLength: number;
  rowCount: number;
  inputSize: number;
};

export type WasmResidentWeightStats = {
  instanceCount: number;
  residentBytes: number;
};

type ResidentWasmInstance = {
  id: number;
  exports: KernelExports;
  residentBytes: number;
};

export function setPrefillWasmTrace(trace: PrefillWasmTrace | undefined): void {
  wasmTrace = trace;
}


export async function matMulQuantizedWasm(
  type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0",
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }
  const typeId = quantizedTypeId(type);
  if (!typeId) {
    return undefined;
  }
  if (inputColumns.length !== inputSize * columnCount) {
    throw new Error(`Quantized matmul input shape mismatch: ${inputColumns.length}`);
  }

  const allocations: Allocation[] = [];
  try {
    const outputLength = rowCount * columnCount;
    const { weightAlloc, inputAlloc, outputAlloc } = timedWasmSection("matMulQuantized", "allocation + input copy", () => ({
      weightAlloc: copyU8ToWasm(exports, weightBytes, allocations),
      inputAlloc: copyF32ToWasm(exports, inputColumns, allocations),
      outputAlloc: allocateBytes(exports, outputLength * Float32Array.BYTES_PER_ELEMENT, allocations),
    }), weightBytes.byteLength + inputColumns.byteLength + outputLength * Float32Array.BYTES_PER_ELEMENT);
    const code = timedWasmSection("matMulQuantized", "kernel call", () => exports.hp_matmul_quantized_f32(
      typeId,
      weightAlloc.ptr,
      weightBytes.length,
      inputAlloc.ptr,
      inputColumns.length,
      inputSize,
      rowCount,
      columnCount,
      outputAlloc.ptr,
      outputLength,
    ));
    assertWasmOk(code, "matMulQuantized");
    return timedWasmSection("matMulQuantized", "output copy + free", () => {
      const output = readF32FromWasm(exports, outputAlloc.ptr, outputLength);
      releaseAllocations(exports, allocations);
      allocations.length = 0;
      return output;
    }, outputLength * Float32Array.BYTES_PER_ELEMENT);
  } finally {
    releaseAllocations(exports, allocations);
  }
}

export async function createWasmQuantizedWeightHandle(
  type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0",
  weightBytes: Uint8Array,
  inputSize: number,
  rowCount: number,
): Promise<WasmQuantizedWeightHandle | undefined> {
  const typeId = quantizedTypeId(type);
  if (!typeId) {
    return undefined;
  }
  if (inputSize <= 0 || rowCount <= 0) {
    throw new Error(`Resident quantized weight shape mismatch: inputSize=${inputSize} rowCount=${rowCount}`);
  }
  const scaleLength = quantizedScaleValueCount(type, inputSize, rowCount);
  const scaleByteLength = scaleLength * Float32Array.BYTES_PER_ELEMENT;
  const instance = await residentWasmInstanceFor(weightBytes.byteLength + scaleByteLength);
  if (!instance) {
    return undefined;
  }

  const allocation = timedWasmSection("matMulQuantizedResident", "resident weight copy", () => {
    const ptr = unsignedWasmPtr(instance.exports.hp_alloc(weightBytes.byteLength));
    assertAllocation(ptr, weightBytes.byteLength);
    new Uint8Array(instance.exports.memory.buffer, ptr, weightBytes.length).set(weightBytes);
    return { ptr, byteLength: weightBytes.byteLength };
  }, weightBytes.byteLength);
  const scaleAllocation = timedWasmSection("matMulQuantizedResident", "resident scale prepare", () => {
    const scalePtr = unsignedWasmPtr(instance.exports.hp_alloc(scaleByteLength));
    assertAllocation(scalePtr, scaleByteLength);
    const code = instance.exports.hp_prepare_quantized_scales_f32(
      typeId,
      allocation.ptr,
      allocation.byteLength,
      inputSize,
      rowCount,
      scalePtr,
      scaleLength,
    );
    assertWasmOk(code, "prepareQuantizedScales");
    return { ptr: scalePtr, byteLength: scaleByteLength };
  }, scaleByteLength);

  instance.residentBytes += weightBytes.byteLength + scaleByteLength;
  return {
    instanceId: instance.id,
    type,
    ptr: allocation.ptr,
    byteLength: allocation.byteLength,
    scalePtr: scaleAllocation.ptr,
    scaleByteLength: scaleAllocation.byteLength,
    scaleLength,
    rowCount,
    inputSize,
  };
}

export function releaseWasmQuantizedWeightHandle(handle: WasmQuantizedWeightHandle): void {
  const instance = residentInstances.find((candidate) => candidate.id === handle.instanceId);
  if (!instance) {
    return;
  }
  instance.exports.hp_dealloc(handle.ptr, handle.byteLength);
  instance.exports.hp_dealloc(handle.scalePtr, handle.scaleByteLength);
  instance.residentBytes = Math.max(0, instance.residentBytes - handle.byteLength - handle.scaleByteLength);
}

export function wasmResidentWeightStats(): WasmResidentWeightStats {
  return {
    instanceCount: residentInstances.length,
    residentBytes: residentInstances.reduce((sum, instance) => sum + instance.residentBytes, 0),
  };
}

export async function matMulQuantizedWasmResident(
  handle: WasmQuantizedWeightHandle,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  const instance = residentInstances.find((candidate) => candidate.id === handle.instanceId);
  if (!instance) {
    throw new Error(`Resident quantized matmul instance ${handle.instanceId} is not available`);
  }
  if (handle.inputSize !== inputSize || handle.rowCount !== rowCount) {
    throw new Error(`Resident quantized matmul handle shape mismatch: ${handle.inputSize}x${handle.rowCount}`);
  }
  if (inputColumns.length !== inputSize * columnCount) {
    throw new Error(`Resident quantized matmul input shape mismatch: ${inputColumns.length}`);
  }
  const typeId = quantizedTypeId(handle.type);
  if (!typeId) {
    throw new Error(`Resident quantized matmul type ${handle.type} is not supported`);
  }

  const allocations: Allocation[] = [];
  try {
    const outputLength = rowCount * columnCount;
    const { inputAlloc, outputAlloc } = timedWasmSection("matMulQuantizedResident", "allocation + input copy", () => ({
      inputAlloc: copyF32ToWasm(instance.exports, inputColumns, allocations),
      outputAlloc: allocateBytes(instance.exports, outputLength * Float32Array.BYTES_PER_ELEMENT, allocations),
    }), inputColumns.byteLength + outputLength * Float32Array.BYTES_PER_ELEMENT);
    const code = timedWasmSection("matMulQuantizedResident", "kernel call", () => instance.exports.hp_matmul_quantized_prepared_f32(
      typeId,
      handle.ptr,
      handle.byteLength,
      handle.scalePtr,
      handle.scaleLength,
      inputAlloc.ptr,
      inputColumns.length,
      inputSize,
      rowCount,
      columnCount,
      outputAlloc.ptr,
      outputLength,
    ));
    assertWasmOk(code, "matMulQuantizedResident");
    return timedWasmSection("matMulQuantizedResident", "output copy + free", () => {
      const output = readF32FromWasm(instance.exports, outputAlloc.ptr, outputLength);
      releaseAllocations(instance.exports, allocations);
      allocations.length = 0;
      return output;
    }, outputLength * Float32Array.BYTES_PER_ELEMENT);
  } finally {
    releaseAllocations(instance.exports, allocations);
  }
}

export async function matMulQuantizedWasmResidentMulti(
  handles: readonly WasmQuantizedWeightHandle[],
  inputColumns: Float32Array,
  inputSize: number,
  columnCount: number,
): Promise<Float32Array[] | undefined> {
  if (handles.length < 2 || handles.length > 4) {
    return undefined;
  }
  const instanceId = handles[0]?.instanceId;
  if (instanceId === undefined || handles.some((handle) => handle.instanceId !== instanceId)) {
    return undefined;
  }
  if (inputColumns.length !== inputSize * columnCount) {
    throw new Error(`Resident quantized multi matmul input shape mismatch: ${inputColumns.length}`);
  }
  for (const handle of handles) {
    if (handle.inputSize !== inputSize) {
      throw new Error(`Resident quantized multi matmul handle shape mismatch: ${handle.inputSize}`);
    }
  }

  const instance = residentInstances.find((candidate) => candidate.id === instanceId);
  if (!instance) {
    throw new Error(`Resident quantized multi matmul instance ${instanceId} is not available`);
  }
  const typeIds = handles.map((handle) => quantizedTypeId(handle.type));
  if (typeIds.some((typeId) => !typeId)) {
    throw new Error("Resident quantized multi matmul has unsupported type");
  }

  const allocations: Allocation[] = [];
  try {
    const outputLengths = handles.map((handle) => handle.rowCount * columnCount);
    const allocated = timedWasmSection("matMulQuantizedResidentMulti", "allocation + input copy", () => {
      const inputAlloc = copyF32ToWasm(instance.exports, inputColumns, allocations);
      const outputAllocs = outputLengths.map((length) =>
        allocateBytes(instance.exports, length * Float32Array.BYTES_PER_ELEMENT, allocations),
      );
      return { inputAlloc, outputAllocs };
    }, inputColumns.byteLength + outputLengths.reduce((sum, length) => sum + length * Float32Array.BYTES_PER_ELEMENT, 0));

    const slots = Array.from({ length: 4 }, (_, index) => ({
      typeId: typeIds[index] ?? 0,
      handle: handles[index],
      outputAlloc: allocated.outputAllocs[index] ?? { ptr: 0, byteLength: 0, exports: instance.exports },
      outputLength: outputLengths[index] ?? 0,
    }));

    const code = timedWasmSection("matMulQuantizedResidentMulti", "kernel call", () => instance.exports.hp_matmul_quantized_prepared_multi_f32(
      handles.length,
      allocated.inputAlloc.ptr,
      inputColumns.length,
      inputSize,
      columnCount,
      slots[0].typeId,
      slots[0].handle?.ptr ?? 0,
      slots[0].handle?.byteLength ?? 0,
      slots[0].handle?.scalePtr ?? 0,
      slots[0].handle?.scaleLength ?? 0,
      slots[0].handle?.rowCount ?? 0,
      slots[0].outputAlloc.ptr,
      slots[0].outputLength,
      slots[1].typeId,
      slots[1].handle?.ptr ?? 0,
      slots[1].handle?.byteLength ?? 0,
      slots[1].handle?.scalePtr ?? 0,
      slots[1].handle?.scaleLength ?? 0,
      slots[1].handle?.rowCount ?? 0,
      slots[1].outputAlloc.ptr,
      slots[1].outputLength,
      slots[2].typeId,
      slots[2].handle?.ptr ?? 0,
      slots[2].handle?.byteLength ?? 0,
      slots[2].handle?.scalePtr ?? 0,
      slots[2].handle?.scaleLength ?? 0,
      slots[2].handle?.rowCount ?? 0,
      slots[2].outputAlloc.ptr,
      slots[2].outputLength,
      slots[3].typeId,
      slots[3].handle?.ptr ?? 0,
      slots[3].handle?.byteLength ?? 0,
      slots[3].handle?.scalePtr ?? 0,
      slots[3].handle?.scaleLength ?? 0,
      slots[3].handle?.rowCount ?? 0,
      slots[3].outputAlloc.ptr,
      slots[3].outputLength,
    ));
    assertWasmOk(code, "MatMulQuantizedResidentMulti");

    return timedWasmSection("matMulQuantizedResidentMulti", "output copy + free", () => {
      const outputs = slots.slice(0, handles.length).map((slot) =>
        readF32FromWasm(instance.exports, slot.outputAlloc.ptr, slot.outputLength),
      );
      releaseAllocations(instance.exports, allocations);
      allocations.length = 0;
      return outputs;
    }, outputLengths.reduce((sum, length) => sum + length * Float32Array.BYTES_PER_ELEMENT, 0));
  } finally {
    releaseAllocations(instance.exports, allocations);
  }
}

export async function matMulQuantizedMultiWasm(
  weights: readonly QuantizedMatMulInput[],
  inputColumns: Float32Array,
  inputSize: number,
  columnCount: number,
): Promise<Float32Array[] | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }
  if (weights.length < 2 || weights.length > 4) {
    return undefined;
  }
  if (inputColumns.length !== inputSize * columnCount) {
    throw new Error(`Quantized multi matmul input shape mismatch: ${inputColumns.length}`);
  }
  const typeIds = weights.map((weight) => quantizedTypeId(weight.type));
  if (typeIds.some((typeId) => !typeId)) {
    return undefined;
  }

  const allocations: Allocation[] = [];
  try {
    const outputLengths = weights.map((weight) => weight.rowCount * columnCount);
    const allocated = timedWasmSection("matMulQuantizedMulti", "allocation + input copy", () => {
      const weightAllocs = weights.map((weight) => copyU8ToWasm(exports, weight.weightBytes, allocations));
      const inputAlloc = copyF32ToWasm(exports, inputColumns, allocations);
      const outputAllocs = outputLengths.map((length) =>
        allocateBytes(exports, length * Float32Array.BYTES_PER_ELEMENT, allocations),
      );
      return { weightAllocs, inputAlloc, outputAllocs };
    }, weights.reduce((sum, weight) => sum + weight.weightBytes.byteLength, inputColumns.byteLength) +
      outputLengths.reduce((sum, length) => sum + length * Float32Array.BYTES_PER_ELEMENT, 0));

    const slots = Array.from({ length: 4 }, (_, index) => ({
      typeId: typeIds[index] ?? 0,
      weightAlloc: allocated.weightAllocs[index] ?? { ptr: 0, byteLength: 0, exports },
      weightBytes: weights[index]?.weightBytes,
      rowCount: weights[index]?.rowCount ?? 0,
      outputAlloc: allocated.outputAllocs[index] ?? { ptr: 0, byteLength: 0, exports },
      outputLength: outputLengths[index] ?? 0,
    }));

    const code = timedWasmSection("matMulQuantizedMulti", "kernel call", () => exports.hp_matmul_quantized_multi_f32(
      weights.length,
      allocated.inputAlloc.ptr,
      inputColumns.length,
      inputSize,
      columnCount,
      slots[0].typeId,
      slots[0].weightAlloc.ptr,
      slots[0].weightBytes?.length ?? 0,
      slots[0].rowCount,
      slots[0].outputAlloc.ptr,
      slots[0].outputLength,
      slots[1].typeId,
      slots[1].weightAlloc.ptr,
      slots[1].weightBytes?.length ?? 0,
      slots[1].rowCount,
      slots[1].outputAlloc.ptr,
      slots[1].outputLength,
      slots[2].typeId,
      slots[2].weightAlloc.ptr,
      slots[2].weightBytes?.length ?? 0,
      slots[2].rowCount,
      slots[2].outputAlloc.ptr,
      slots[2].outputLength,
      slots[3].typeId,
      slots[3].weightAlloc.ptr,
      slots[3].weightBytes?.length ?? 0,
      slots[3].rowCount,
      slots[3].outputAlloc.ptr,
      slots[3].outputLength,
    ));
    assertWasmOk(code, "matMulQuantizedMulti");

    return timedWasmSection("matMulQuantizedMulti", "output copy + free", () => {
      const outputs = slots.slice(0, weights.length).map((slot) =>
        readF32FromWasm(exports, slot.outputAlloc.ptr, slot.outputLength),
      );
      releaseAllocations(exports, allocations);
      allocations.length = 0;
      return outputs;
    }, outputLengths.reduce((sum, length) => sum + length * Float32Array.BYTES_PER_ELEMENT, 0));
  } finally {
    releaseAllocations(exports, allocations);
  }
}

export async function gqaAttentionWasm(
  query: Float32Array,
  key: Float32Array,
  value: Float32Array,
  {
    headSize,
    queryHeadCount,
    keyValueHeadCount,
    tokenCount,
    keyValueTokenCount = tokenCount,
    scale,
    causal = true,
    mask,
    valueLayout = "token-head-dim",
    quantizeQueryForScore,
  }: GqaAttentionOptions,
): Promise<Float32Array | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }
  if (causal && mask === undefined) {
    return undefined;
  }
  validateGqaShapes(query, key, value, mask, {
    headSize,
    queryHeadCount,
    keyValueHeadCount,
    tokenCount,
    keyValueTokenCount,
    scale,
    causal,
    mask,
    valueLayout,
    quantizeQueryForScore,
  });

  const allocations: Allocation[] = [];
  try {
    const outputLength = tokenCount * queryHeadCount * headSize;
    const { queryAlloc, keyAlloc, valueAlloc, maskAlloc, outputAlloc } = timedWasmSection("gqaAttention", "allocation + input copy", () => ({
      queryAlloc: copyF32ToWasm(exports, query, allocations),
      keyAlloc: copyF32ToWasm(exports, key, allocations),
      valueAlloc: copyF32ToWasm(exports, value, allocations),
      maskAlloc: mask ? copyF32ToWasm(exports, mask, allocations) : { ptr: 0, byteLength: 0 },
      outputAlloc: allocateBytes(exports, outputLength * Float32Array.BYTES_PER_ELEMENT, allocations),
    }), query.byteLength + key.byteLength + value.byteLength + (mask?.byteLength ?? 0) +
      outputLength * Float32Array.BYTES_PER_ELEMENT);
    const code = timedWasmSection("gqaAttention", "kernel call", () => exports.hp_gqa_attention_f32(
      queryAlloc.ptr,
      query.length,
      keyAlloc.ptr,
      key.length,
      valueAlloc.ptr,
      value.length,
      maskAlloc.ptr,
      mask?.length ?? 0,
      headSize,
      queryHeadCount,
      keyValueHeadCount,
      tokenCount,
      keyValueTokenCount,
      scale,
      valueLayout === "dim-head-token" ? 1 : 0,
      quantizeQueryForScore === "f16" ? 1 : 0,
      outputAlloc.ptr,
      outputLength,
    ));
    assertWasmOk(code, "gqaAttention");
    return timedWasmSection("gqaAttention", "output copy + free", () => {
      const output = readF32FromWasm(exports, outputAlloc.ptr, outputLength);
      releaseAllocations(exports, allocations);
      allocations.length = 0;
      return output;
    }, outputLength * Float32Array.BYTES_PER_ELEMENT);
  } finally {
    releaseAllocations(exports, allocations);
  }
}

export async function visionPatchEmbedWasm(
  pixels: Float32Array,
  weights: Float32Array,
  imageWidth: number,
  patchSize: number,
  patchGridX: number,
  patchGridY: number,
  embeddingLength: number,
): Promise<Float32Array | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }
  const outputLength = patchGridX * patchGridY * embeddingLength;
  const allocations: Allocation[] = [];
  try {
    const { pixelsAlloc, weightsAlloc, outputAlloc } = timedWasmSection("visionPatchEmbed", "allocation + input copy", () => ({
      pixelsAlloc: copyF32ToWasm(exports, pixels, allocations),
      weightsAlloc: copyF32ToWasm(exports, weights, allocations),
      outputAlloc: allocateBytes(exports, outputLength * Float32Array.BYTES_PER_ELEMENT, allocations),
    }), pixels.byteLength + weights.byteLength + outputLength * Float32Array.BYTES_PER_ELEMENT);
    const code = timedWasmSection("visionPatchEmbed", "kernel call", () => exports.hp_vision_patch_embed_f32(
      pixelsAlloc.ptr,
      pixels.length,
      weightsAlloc.ptr,
      weights.length,
      imageWidth,
      patchSize,
      patchGridX,
      patchGridY,
      embeddingLength,
      outputAlloc.ptr,
      outputLength,
    ));
    assertWasmOk(code, "visionPatchEmbed");
    return readVisionOutput(exports, allocations, outputAlloc.ptr, outputLength, "visionPatchEmbed");
  } finally {
    releaseAllocations(exports, allocations);
  }
}

export async function visionAddPositionEmbeddingsWasm(
  hidden: Float32Array,
  positions: Float32Array,
  patchGridX: number,
  tokenCount: number,
  embeddingLength: number,
  tableSize: number,
): Promise<Float32Array | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }
  const allocations: Allocation[] = [];
  try {
    const { hiddenAlloc, positionsAlloc, outputAlloc } = timedWasmSection("visionAddPosition", "allocation + input copy", () => ({
      hiddenAlloc: copyF32ToWasm(exports, hidden, allocations),
      positionsAlloc: copyF32ToWasm(exports, positions, allocations),
      outputAlloc: allocateBytes(exports, hidden.byteLength, allocations),
    }), hidden.byteLength + positions.byteLength + hidden.byteLength);
    const code = timedWasmSection("visionAddPosition", "kernel call", () => exports.hp_vision_add_position_f32(
      hiddenAlloc.ptr,
      hidden.length,
      positionsAlloc.ptr,
      positions.length,
      patchGridX,
      tokenCount,
      embeddingLength,
      tableSize,
      outputAlloc.ptr,
      hidden.length,
    ));
    assertWasmOk(code, "visionAddPosition");
    return readVisionOutput(exports, allocations, outputAlloc.ptr, hidden.length, "visionAddPosition");
  } finally {
    releaseAllocations(exports, allocations);
  }
}

export async function visionRmsNormWasm(
  input: Float32Array,
  rowSize: number,
  epsilon: number,
  weight?: Float32Array,
): Promise<Float32Array | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }
  const allocations: Allocation[] = [];
  try {
    const { inputAlloc, weightAlloc, outputAlloc } = timedWasmSection("visionRmsNorm", "allocation + input copy", () => ({
      inputAlloc: copyF32ToWasm(exports, input, allocations),
      weightAlloc: weight ? copyF32ToWasm(exports, weight, allocations) : { ptr: 0, byteLength: 0, exports },
      outputAlloc: allocateBytes(exports, input.byteLength, allocations),
    }), input.byteLength + (weight?.byteLength ?? 0) + input.byteLength);
    const code = timedWasmSection("visionRmsNorm", "kernel call", () => exports.hp_vision_rms_norm_f32(
      inputAlloc.ptr,
      input.length,
      weightAlloc.ptr,
      weight?.length ?? 0,
      rowSize,
      epsilon,
      outputAlloc.ptr,
      input.length,
    ));
    assertWasmOk(code, "visionRmsNorm");
    return readVisionOutput(exports, allocations, outputAlloc.ptr, input.length, "visionRmsNorm");
  } finally {
    releaseAllocations(exports, allocations);
  }
}

export async function visionRope2dNeoxWasm(
  input: Float32Array,
  patchGridX: number,
  headSize: number,
  headCount: number,
  tokenCount: number,
  freqBase: number,
): Promise<Float32Array | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }
  const allocations: Allocation[] = [];
  try {
    const { inputAlloc, outputAlloc } = timedWasmSection("visionRope2dNeox", "allocation + input copy", () => ({
      inputAlloc: copyF32ToWasm(exports, input, allocations),
      outputAlloc: allocateBytes(exports, input.byteLength, allocations),
    }), input.byteLength * 2);
    const code = timedWasmSection("visionRope2dNeox", "kernel call", () => exports.hp_vision_rope2d_neox_f32(
      inputAlloc.ptr,
      input.length,
      patchGridX,
      headSize,
      headCount,
      tokenCount,
      freqBase,
      outputAlloc.ptr,
      input.length,
    ));
    assertWasmOk(code, "visionRope2dNeox");
    return readVisionOutput(exports, allocations, outputAlloc.ptr, input.length, "visionRope2dNeox");
  } finally {
    releaseAllocations(exports, allocations);
  }
}

export async function visionClampWasm(
  input: Float32Array,
  min: number,
  max: number,
): Promise<Float32Array | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }
  const allocations: Allocation[] = [];
  try {
    const { inputAlloc, outputAlloc } = timedWasmSection("visionClamp", "allocation + input copy", () => ({
      inputAlloc: copyF32ToWasm(exports, input, allocations),
      outputAlloc: allocateBytes(exports, input.byteLength, allocations),
    }), input.byteLength * 2);
    const code = timedWasmSection("visionClamp", "kernel call", () => exports.hp_vision_clamp_f32(
      inputAlloc.ptr,
      input.length,
      min,
      max,
      outputAlloc.ptr,
      input.length,
    ));
    assertWasmOk(code, "visionClamp");
    return readVisionOutput(exports, allocations, outputAlloc.ptr, input.length, "visionClamp");
  } finally {
    releaseAllocations(exports, allocations);
  }
}

export async function visionGeluMulWasm(
  gate: Float32Array,
  up: Float32Array,
): Promise<Float32Array | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }
  const allocations: Allocation[] = [];
  try {
    const { gateAlloc, upAlloc, outputAlloc } = timedWasmSection("visionGeluMul", "allocation + input copy", () => ({
      gateAlloc: copyF32ToWasm(exports, gate, allocations),
      upAlloc: copyF32ToWasm(exports, up, allocations),
      outputAlloc: allocateBytes(exports, gate.byteLength, allocations),
    }), gate.byteLength + up.byteLength + gate.byteLength);
    const code = timedWasmSection("visionGeluMul", "kernel call", () => exports.hp_vision_gelu_mul_f32(
      gateAlloc.ptr,
      gate.length,
      upAlloc.ptr,
      up.length,
      outputAlloc.ptr,
      gate.length,
    ));
    assertWasmOk(code, "visionGeluMul");
    return readVisionOutput(exports, allocations, outputAlloc.ptr, gate.length, "visionGeluMul");
  } finally {
    releaseAllocations(exports, allocations);
  }
}

export async function visionResidualAddWasm(
  left: Float32Array,
  right: Float32Array,
): Promise<Float32Array | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }
  const allocations: Allocation[] = [];
  try {
    const { leftAlloc, rightAlloc, outputAlloc } = timedWasmSection("visionResidualAdd", "allocation + input copy", () => ({
      leftAlloc: copyF32ToWasm(exports, left, allocations),
      rightAlloc: copyF32ToWasm(exports, right, allocations),
      outputAlloc: allocateBytes(exports, left.byteLength, allocations),
    }), left.byteLength + right.byteLength + left.byteLength);
    const code = timedWasmSection("visionResidualAdd", "kernel call", () => exports.hp_vision_residual_add_f32(
      leftAlloc.ptr,
      left.length,
      rightAlloc.ptr,
      right.length,
      outputAlloc.ptr,
      left.length,
    ));
    assertWasmOk(code, "visionResidualAdd");
    return readVisionOutput(exports, allocations, outputAlloc.ptr, left.length, "visionResidualAdd");
  } finally {
    releaseAllocations(exports, allocations);
  }
}

export async function visionAveragePoolScaleWasm(
  input: Float32Array,
  patchGridX: number,
  patchGridY: number,
  embeddingLength: number,
  kernelSize: number,
  outputScale: number,
): Promise<Float32Array | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }
  const outputLength = (patchGridX / kernelSize) * (patchGridY / kernelSize) * embeddingLength;
  const allocations: Allocation[] = [];
  try {
    const { inputAlloc, outputAlloc } = timedWasmSection("visionAveragePoolScale", "allocation + input copy", () => ({
      inputAlloc: copyF32ToWasm(exports, input, allocations),
      outputAlloc: allocateBytes(exports, outputLength * Float32Array.BYTES_PER_ELEMENT, allocations),
    }), input.byteLength + outputLength * Float32Array.BYTES_PER_ELEMENT);
    const code = timedWasmSection("visionAveragePoolScale", "kernel call", () => exports.hp_vision_average_pool_scale_f32(
      inputAlloc.ptr,
      input.length,
      patchGridX,
      patchGridY,
      embeddingLength,
      kernelSize,
      outputScale,
      outputAlloc.ptr,
      outputLength,
    ));
    assertWasmOk(code, "visionAveragePoolScale");
    return readVisionOutput(exports, allocations, outputAlloc.ptr, outputLength, "visionAveragePoolScale");
  } finally {
    releaseAllocations(exports, allocations);
  }
}

export async function visionStdNormalizeWasm(
  input: Float32Array,
  bias: Float32Array,
  scale: Float32Array,
  rowSize: number,
): Promise<Float32Array | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }
  const allocations: Allocation[] = [];
  try {
    const { inputAlloc, biasAlloc, scaleAlloc, outputAlloc } = timedWasmSection("visionStdNormalize", "allocation + input copy", () => ({
      inputAlloc: copyF32ToWasm(exports, input, allocations),
      biasAlloc: copyF32ToWasm(exports, bias, allocations),
      scaleAlloc: copyF32ToWasm(exports, scale, allocations),
      outputAlloc: allocateBytes(exports, input.byteLength, allocations),
    }), input.byteLength + bias.byteLength + scale.byteLength + input.byteLength);
    const code = timedWasmSection("visionStdNormalize", "kernel call", () => exports.hp_vision_std_normalize_f32(
      inputAlloc.ptr,
      input.length,
      biasAlloc.ptr,
      bias.length,
      scaleAlloc.ptr,
      scale.length,
      rowSize,
      outputAlloc.ptr,
      input.length,
    ));
    assertWasmOk(code, "visionStdNormalize");
    return readVisionOutput(exports, allocations, outputAlloc.ptr, input.length, "visionStdNormalize");
  } finally {
    releaseAllocations(exports, allocations);
  }
}

export async function prefillWasmBackend(): Promise<"wasm-simd" | "ts"> {
  return (await prefillWasmExports()) ? "wasm-simd" : "ts";
}

export function resetPrefillWasmForTesting(base64?: string): void {
  wasmBase64ForTesting = base64;
  instancePromise = undefined;
  modulePromise = undefined;
  scratchPool.length = 0;
  scratchPoolBytes = 0;
  residentInstances.length = 0;
  nextResidentInstanceId = 1;
}

async function prefillWasmExports(): Promise<KernelExports | undefined> {
  if (!instancePromise) {
    instancePromise = instantiatePrefillWasm();
  }
  return instancePromise;
}

async function instantiatePrefillWasm(): Promise<KernelExports | undefined> {
  const module = await prefillWasmModule();
  if (!module) {
    return undefined;
  }

  try {
    return instantiateKernelExports(module);
  } catch {
    return undefined;
  }
}

async function prefillWasmModule(): Promise<WebAssembly.Module | undefined> {
  if (!modulePromise) {
    modulePromise = compilePrefillWasmModule();
  }
  return modulePromise;
}

async function compilePrefillWasmModule(): Promise<WebAssembly.Module | undefined> {
  const base64 = wasmBase64ForTesting ?? PREFILL_WASM_SIMD_BASE64;
  if (!base64 || typeof WebAssembly === "undefined") {
    return undefined;
  }

  try {
    const bytes = decodeBase64(base64);
    if (!WebAssembly.validate(bytes)) {
      return undefined;
    }
    return WebAssembly.compile(bytes);
  } catch {
    return undefined;
  }
}

async function residentWasmInstanceFor(byteLength: number): Promise<ResidentWasmInstance | undefined> {
  for (const instance of residentInstances) {
    if (instance.residentBytes + byteLength <= maxResidentInstanceBytes) {
      return instance;
    }
  }

  const module = await prefillWasmModule();
  if (!module) {
    return undefined;
  }
  try {
    const exports = await instantiateKernelExports(module);
    const instance = {
      id: nextResidentInstanceId,
      exports,
      residentBytes: 0,
    };
    nextResidentInstanceId += 1;
    residentInstances.push(instance);
    return instance;
  } catch {
    return undefined;
  }
}

async function instantiateKernelExports(module: WebAssembly.Module): Promise<KernelExports> {
  const instance = await WebAssembly.instantiate(module, {});
  const exports = instance.exports as KernelExports;
  if (!exports.memory || !exports.hp_alloc || !exports.hp_dealloc) {
    throw new Error("prefill wasm module is missing required exports");
  }
  return exports;
}

function decodeBase64(base64: string): Uint8Array<ArrayBuffer> {
  const maybeBuffer = (globalThis as { Buffer?: { from(input: string, encoding: string): Uint8Array } }).Buffer;
  if (maybeBuffer) {
    const source = maybeBuffer.from(base64, "base64");
    const bytes = new Uint8Array(source.length);
    bytes.set(source);
    return bytes;
  }

  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function allocateBytes(exports: KernelExports, byteLength: number, allocations: Allocation[]): Allocation {
  const reused = takeScratchAllocation(exports, byteLength);
  if (reused) {
    allocations.push(reused);
    return reused;
  }
  const ptr = unsignedWasmPtr(exports.hp_alloc(byteLength));
  assertAllocation(ptr, byteLength);
  const allocation = { ptr, byteLength, exports };
  allocations.push(allocation);
  return allocation;
}

function copyF32ToWasm(exports: KernelExports, input: Float32Array, allocations: Allocation[]): Allocation {
  const allocation = allocateBytes(exports, input.byteLength, allocations);
  new Float32Array(exports.memory.buffer, allocation.ptr, input.length).set(input);
  return allocation;
}

function copyU8ToWasm(exports: KernelExports, input: Uint8Array, allocations: Allocation[]): Allocation {
  const allocation = allocateBytes(exports, input.byteLength, allocations);
  new Uint8Array(exports.memory.buffer, allocation.ptr, input.length).set(input);
  return allocation;
}

function readF32FromWasm(exports: KernelExports, ptr: number, length: number): Float32Array {
  return new Float32Array(exports.memory.buffer, ptr, length).slice();
}

function readVisionOutput(
  exports: KernelExports,
  allocations: Allocation[],
  outputPtr: number,
  outputLength: number,
  kernelName: string,
): Float32Array {
  return timedWasmSection(kernelName, "output copy + free", () => {
    const output = readF32FromWasm(exports, outputPtr, outputLength);
    releaseAllocations(exports, allocations);
    allocations.length = 0;
    return output;
  }, outputLength * Float32Array.BYTES_PER_ELEMENT);
}

function releaseAllocations(exports: KernelExports, allocations: Allocation[]): void {
  for (let index = allocations.length - 1; index >= 0; index -= 1) {
    const allocation = allocations[index];
    if (allocation) {
      releaseScratchAllocation(exports, allocation);
    }
  }
}

function takeScratchAllocation(exports: KernelExports, byteLength: number): Allocation | undefined {
  for (let index = 0; index < scratchPool.length; index += 1) {
    const allocation = scratchPool[index];
    if (allocation && allocation.exports === exports && allocation.byteLength >= byteLength) {
      scratchPool.splice(index, 1);
      scratchPoolBytes -= allocation.byteLength;
      return allocation;
    }
  }
  return undefined;
}

function releaseScratchAllocation(exports: KernelExports, allocation: Allocation): void {
  if (allocation.ptr === 0 || allocation.byteLength === 0) {
    return;
  }
  if (allocation.exports !== exports) {
    allocation.exports.hp_dealloc(allocation.ptr, allocation.byteLength);
    return;
  }
  if (
    scratchPool.length >= maxScratchPoolEntries ||
    scratchPoolBytes + allocation.byteLength > maxScratchPoolBytes
  ) {
    exports.hp_dealloc(allocation.ptr, allocation.byteLength);
    return;
  }
  scratchPool.push(allocation);
  scratchPoolBytes += allocation.byteLength;
}

function timedWasmSection<T>(
  kernel: string,
  section: PrefillWasmTraceEvent["section"],
  run: () => T,
  bytes?: number,
): T {
  if (!wasmTrace) {
    return run();
  }
  const start = nowMs();
  try {
    return run();
  } finally {
    wasmTrace({
      kernel,
      section,
      durationMs: nowMs() - start,
      bytes,
    });
  }
}

function nowMs(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function assertWasmOk(code: number, kernelName: string): void {
  if (code === 0) {
    return;
  }
  if (code === 1) {
    throw new Error(`${kernelName} wasm shape mismatch`);
  }
  if (code === 2) {
    throw new Error(`${kernelName} wasm head count mismatch`);
  }
  if (code === 3) {
    throw new Error(`${kernelName} wasm unsupported type`);
  }
  throw new Error(`${kernelName} wasm failed with code ${code}`);
}

function assertAllocation(ptr: number, byteLength: number): void {
  if (byteLength !== 0 && ptr === 0) {
    throw new Error(`WASM allocation failed for ${byteLength} bytes`);
  }
}

function unsignedWasmPtr(ptr: number): number {
  return ptr >>> 0;
}

function quantizedTypeId(type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0"): number {
  switch (type) {
    case "Q4_K":
      return 1;
    case "Q5_K":
      return 2;
    case "Q6_K":
      return 3;
    case "IQ4_XS":
      return 4;
    case "Q8_0":
      return 5;
  }
}

function quantizedScaleValueCount(
  type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0",
  inputSize: number,
  rowCount: number,
): number {
  if (type === "Q8_0") {
    if (inputSize % 32 !== 0) {
      throw new Error(`Q8_0 scale shape mismatch: ${inputSize}`);
    }
    return (inputSize / 32) * rowCount;
  }
  if (inputSize % 256 !== 0) {
    throw new Error(`${type} scale shape mismatch: ${inputSize}`);
  }
  const blocks = inputSize / 256;
  return blocks * rowCount * (type === "Q4_K" || type === "Q5_K" ? 2 : 1);
}

function validateGqaShapes(
  query: Float32Array,
  key: Float32Array,
  value: Float32Array,
  mask: Float32Array | undefined,
  {
    headSize,
    queryHeadCount,
    keyValueHeadCount,
    tokenCount,
    keyValueTokenCount = tokenCount,
  }: GqaAttentionOptions,
): void {
  if (query.length !== tokenCount * queryHeadCount * headSize) {
    throw new Error(`GQA query shape mismatch: ${query.length}`);
  }
  if (key.length !== keyValueTokenCount * keyValueHeadCount * headSize) {
    throw new Error(`GQA key shape mismatch: ${key.length}`);
  }
  if (value.length !== keyValueTokenCount * keyValueHeadCount * headSize) {
    throw new Error(`GQA value shape mismatch: ${value.length}`);
  }
  if (mask && mask.length !== tokenCount * keyValueTokenCount) {
    throw new Error(`GQA mask shape mismatch: ${mask.length}`);
  }
  if (queryHeadCount % keyValueHeadCount !== 0) {
    throw new Error(`GQA head count mismatch: q=${queryHeadCount} kv=${keyValueHeadCount}`);
  }
}
