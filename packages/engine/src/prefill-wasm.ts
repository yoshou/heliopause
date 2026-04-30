import type {
  GatedDeltaNetOptions,
  GqaAttentionOptions,
} from "./ops";
import { PREFILL_WASM_SIMD_BASE64 } from "./wasm-kernels.generated";

type KernelExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  hp_alloc(byteLength: number): number;
  hp_dealloc(ptr: number, byteLength: number): void;
  hp_ssm_conv1d_f32(
    convInputPtr: number,
    convInputLen: number,
    kernelPtr: number,
    kernelLen: number,
    channelCount: number,
    tokenCount: number,
    kernelSize: number,
    outputPtr: number,
    outputLen: number,
  ): number;
  hp_gated_delta_net_f32(
    queryPtr: number,
    queryLen: number,
    keyPtr: number,
    keyLen: number,
    valuePtr: number,
    valueLen: number,
    gatePtr: number,
    gateLen: number,
    betaPtr: number,
    betaLen: number,
    statePtr: number,
    stateLen: number,
    stateSize: number,
    keyHeadCount: number,
    valueHeadCount: number,
    tokenCount: number,
    outputPtr: number,
    outputLen: number,
    newStatePtr: number,
    newStateLen: number,
  ): number;
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
};

type Allocation = {
  ptr: number;
  byteLength: number;
};

let wasmBase64ForTesting: string | undefined;
let instancePromise: Promise<KernelExports | undefined> | undefined;

export async function ssmConv1dWasm(
  convInput: Float32Array,
  kernel: Float32Array,
  channelCount: number,
  tokenCount: number,
  kernelSize: number,
): Promise<Float32Array | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }

  const expectedInputWindow = kernelSize - 1 + tokenCount;
  if (convInput.length !== expectedInputWindow * channelCount) {
    throw new Error(`SSM conv input shape mismatch: ${convInput.length}`);
  }
  if (kernel.length !== kernelSize * channelCount) {
    throw new Error(`SSM conv kernel shape mismatch: ${kernel.length}`);
  }

  const allocations: Allocation[] = [];
  try {
    const convInputAlloc = copyF32ToWasm(exports, convInput, allocations);
    const kernelAlloc = copyF32ToWasm(exports, kernel, allocations);
    const outputLength = channelCount * tokenCount;
    const outputAlloc = allocateBytes(exports, outputLength * Float32Array.BYTES_PER_ELEMENT, allocations);

    const code = exports.hp_ssm_conv1d_f32(
      convInputAlloc.ptr,
      convInput.length,
      kernelAlloc.ptr,
      kernel.length,
      channelCount,
      tokenCount,
      kernelSize,
      outputAlloc.ptr,
      outputLength,
    );
    assertWasmOk(code, "ssmConv1d");
    return readF32FromWasm(exports, outputAlloc.ptr, outputLength);
  } finally {
    freeAllocations(exports, allocations);
  }
}

export async function gatedDeltaNetWasm(
  query: Float32Array,
  key: Float32Array,
  value: Float32Array,
  gate: Float32Array,
  beta: Float32Array,
  state: Float32Array,
  {
    stateSize,
    keyHeadCount,
    valueHeadCount,
    tokenCount,
  }: GatedDeltaNetOptions,
): Promise<{ output: Float32Array; newState: Float32Array } | undefined> {
  const exports = await prefillWasmExports();
  if (!exports) {
    return undefined;
  }

  validateGatedDeltaShapes(query, key, value, gate, beta, state, {
    stateSize,
    keyHeadCount,
    valueHeadCount,
    tokenCount,
  });

  const allocations: Allocation[] = [];
  try {
    const queryAlloc = copyF32ToWasm(exports, query, allocations);
    const keyAlloc = copyF32ToWasm(exports, key, allocations);
    const valueAlloc = copyF32ToWasm(exports, value, allocations);
    const gateAlloc = copyF32ToWasm(exports, gate, allocations);
    const betaAlloc = copyF32ToWasm(exports, beta, allocations);
    const stateAlloc = copyF32ToWasm(exports, state, allocations);
    const outputAlloc = allocateBytes(exports, tokenCount * valueHeadCount * stateSize * Float32Array.BYTES_PER_ELEMENT, allocations);
    const newStateAlloc = allocateBytes(exports, state.length * Float32Array.BYTES_PER_ELEMENT, allocations);

    const code = exports.hp_gated_delta_net_f32(
      queryAlloc.ptr,
      query.length,
      keyAlloc.ptr,
      key.length,
      valueAlloc.ptr,
      value.length,
      gateAlloc.ptr,
      gate.length,
      betaAlloc.ptr,
      beta.length,
      stateAlloc.ptr,
      state.length,
      stateSize,
      keyHeadCount,
      valueHeadCount,
      tokenCount,
      outputAlloc.ptr,
      tokenCount * valueHeadCount * stateSize,
      newStateAlloc.ptr,
      state.length,
    );
    assertWasmOk(code, "gatedDeltaNet");
    return {
      output: readF32FromWasm(exports, outputAlloc.ptr, tokenCount * valueHeadCount * stateSize),
      newState: readF32FromWasm(exports, newStateAlloc.ptr, state.length),
    };
  } finally {
    freeAllocations(exports, allocations);
  }
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
    const weightAlloc = copyU8ToWasm(exports, weightBytes, allocations);
    const inputAlloc = copyF32ToWasm(exports, inputColumns, allocations);
    const outputLength = rowCount * columnCount;
    const outputAlloc = allocateBytes(exports, outputLength * Float32Array.BYTES_PER_ELEMENT, allocations);
    const code = exports.hp_matmul_quantized_f32(
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
    );
    assertWasmOk(code, "matMulQuantized");
    return readF32FromWasm(exports, outputAlloc.ptr, outputLength);
  } finally {
    freeAllocations(exports, allocations);
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
    const queryAlloc = copyF32ToWasm(exports, query, allocations);
    const keyAlloc = copyF32ToWasm(exports, key, allocations);
    const valueAlloc = copyF32ToWasm(exports, value, allocations);
    const maskAlloc = mask
      ? copyF32ToWasm(exports, mask, allocations)
      : { ptr: 0, byteLength: 0 };
    const outputLength = tokenCount * queryHeadCount * headSize;
    const outputAlloc = allocateBytes(exports, outputLength * Float32Array.BYTES_PER_ELEMENT, allocations);
    const code = exports.hp_gqa_attention_f32(
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
    );
    assertWasmOk(code, "gqaAttention");
    return readF32FromWasm(exports, outputAlloc.ptr, outputLength);
  } finally {
    freeAllocations(exports, allocations);
  }
}

export async function prefillWasmBackend(): Promise<"wasm-simd" | "ts"> {
  return (await prefillWasmExports()) ? "wasm-simd" : "ts";
}

export function resetPrefillWasmForTesting(base64?: string): void {
  wasmBase64ForTesting = base64;
  instancePromise = undefined;
}

async function prefillWasmExports(): Promise<KernelExports | undefined> {
  if (!instancePromise) {
    instancePromise = instantiatePrefillWasm();
  }
  return instancePromise;
}

async function instantiatePrefillWasm(): Promise<KernelExports | undefined> {
  const base64 = wasmBase64ForTesting ?? PREFILL_WASM_SIMD_BASE64;
  if (!base64 || typeof WebAssembly === "undefined") {
    return undefined;
  }

  try {
    const bytes = decodeBase64(base64);
    if (!WebAssembly.validate(bytes)) {
      return undefined;
    }
    const module = await WebAssembly.compile(bytes);
    const instance = await WebAssembly.instantiate(module, {});
    const exports = instance.exports as KernelExports;
    if (!exports.memory || !exports.hp_alloc || !exports.hp_dealloc) {
      return undefined;
    }
    return exports;
  } catch {
    return undefined;
  }
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
  const ptr = exports.hp_alloc(byteLength);
  const allocation = { ptr, byteLength };
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

function freeAllocations(exports: KernelExports, allocations: Allocation[]): void {
  for (let index = allocations.length - 1; index >= 0; index -= 1) {
    const allocation = allocations[index];
    if (allocation) {
      exports.hp_dealloc(allocation.ptr, allocation.byteLength);
    }
  }
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

function validateGatedDeltaShapes(
  query: Float32Array,
  key: Float32Array,
  value: Float32Array,
  gate: Float32Array,
  beta: Float32Array,
  state: Float32Array,
  {
    stateSize,
    keyHeadCount,
    valueHeadCount,
    tokenCount,
  }: GatedDeltaNetOptions,
): void {
  if (query.length !== tokenCount * keyHeadCount * stateSize) {
    throw new Error(`GDN query shape mismatch: ${query.length}`);
  }
  if (key.length !== tokenCount * keyHeadCount * stateSize) {
    throw new Error(`GDN key shape mismatch: ${key.length}`);
  }
  if (value.length !== tokenCount * valueHeadCount * stateSize) {
    throw new Error(`GDN value shape mismatch: ${value.length}`);
  }
  if (gate.length !== tokenCount * valueHeadCount) {
    throw new Error(`GDN gate shape mismatch: ${gate.length}`);
  }
  if (beta.length !== tokenCount * valueHeadCount) {
    throw new Error(`GDN beta shape mismatch: ${beta.length}`);
  }
  if (state.length !== valueHeadCount * stateSize * stateSize) {
    throw new Error(`GDN state shape mismatch: ${state.length}`);
  }
  if (valueHeadCount % keyHeadCount !== 0) {
    throw new Error(`GDN head count mismatch: k=${keyHeadCount} v=${valueHeadCount}`);
  }
}
