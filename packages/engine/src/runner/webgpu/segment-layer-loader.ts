import type { LayerKind, LayerValueProjectionMode, ModelManifest } from "../../model";
import {
  tensorByteLength,
  type GgufTensorReader,
  type TensorByteRange,
} from "../../tensor-reader";
import { GPU_COPY_DST, GPU_STORAGE } from "./gpu-constants";
import { webGpuAdapterLimits } from "./gpu-device";
import { createQuantizedHandleFromBytes, webGpuMatMulType, webGpuQuantizedWeightLayout } from "./quantized-handles";
import type { F32Handle, GpuMemoryArena, QuantizedHandle } from "./arena";

const WEBGPU_LOAD_COALESCE_MAX_GAP_BYTES = 1024 * 1024;
const WEBGPU_LOAD_COALESCE_MAX_READ_BYTES = 256 * 1024 * 1024;

export type OutputStripe = QuantizedHandle & {
  readonly rowOffset: number;
};

export type GpuLayer = {
  kind: LayerKind;
  layer: number;
  hasKv: boolean;
  kvSourceLayer: number;
  headSize: number;
  valueSize: number;
  headCountKv: number;
  valueProjectionMode: LayerValueProjectionMode;
  attnNorm: F32Handle;
  q: QuantizedHandle;
  k?: QuantizedHandle;
  v?: QuantizedHandle;
  qNorm: F32Handle;
  kNorm?: F32Handle;
  attnOut: QuantizedHandle;
  postAttentionNorm: F32Handle;
  ffnNorm: F32Handle;
  ffnGate: QuantizedHandle;
  ffnUp: QuantizedHandle;
  ffnDown: QuantizedHandle;
  postFfwNorm: F32Handle;
  perLayerInputGate?: F32Handle;
  perLayerProjection?: F32Handle;
  postNorm?: F32Handle;
  layerOutputScale: F32Handle;
};

export async function loadGpuLayer(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
  manifest: ModelManifest,
  layer: number,
): Promise<GpuLayer> {
  const hasPerLayerInput = manifest.perLayerEmbeddingLength > 0;
  const hasKv = manifest.layerHasKv[layer] === true;
  const valueProjectionMode = manifest.layerValueProjectionModes[layer] ?? "separate";
  const tensorNames = [
    `blk.${layer}.attn_norm.weight`,
    `blk.${layer}.attn_q.weight`,
    ...(hasKv ? [
      `blk.${layer}.attn_k.weight`,
      ...(valueProjectionMode === "shared-with-key" ? [] : [`blk.${layer}.attn_v.weight`]),
    ] : []),
    `blk.${layer}.attn_q_norm.weight`,
    ...(hasKv ? [`blk.${layer}.attn_k_norm.weight`] : []),
    `blk.${layer}.attn_output.weight`,
    `blk.${layer}.post_attention_norm.weight`,
    `blk.${layer}.ffn_norm.weight`,
    `blk.${layer}.ffn_gate.weight`,
    `blk.${layer}.ffn_up.weight`,
    `blk.${layer}.ffn_down.weight`,
    `blk.${layer}.post_ffw_norm.weight`,
    ...(hasPerLayerInput ? [
      `blk.${layer}.inp_gate.weight`,
      `blk.${layer}.proj.weight`,
      `blk.${layer}.post_norm.weight`,
    ] : []),
    `blk.${layer}.layer_output_scale.weight`,
  ];
  const tensorBytes = await readNamedTensorBytesCoalesced(tensorReader, tensorNames);
  return {
    kind: manifest.layerKinds[layer] ?? "sliding-attention",
    layer,
    hasKv,
    kvSourceLayer: manifest.kvSourceLayers[layer] ?? layer,
    headSize: manifest.layerKeyLengths[layer] ?? manifest.keyLength,
    valueSize: manifest.layerValueLengths[layer] ?? manifest.valueLength,
    headCountKv: manifest.layerHeadCountKv[layer] ?? manifest.headCountKv,
    valueProjectionMode,
    attnNorm: createF32HandleFromBytes(arena, tensorReader, `blk.${layer}.attn_norm.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.attn_norm.weight`)),
    q: createQuantizedHandleFromTensorBytes(arena, tensorReader, `blk.${layer}.attn_q.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.attn_q.weight`)),
    k: hasKv ? createQuantizedHandleFromTensorBytes(arena, tensorReader, `blk.${layer}.attn_k.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.attn_k.weight`)) : undefined,
    v: hasKv && valueProjectionMode !== "shared-with-key"
      ? createQuantizedHandleFromTensorBytes(arena, tensorReader, `blk.${layer}.attn_v.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.attn_v.weight`))
      : undefined,
    qNorm: createF32HandleFromBytes(arena, tensorReader, `blk.${layer}.attn_q_norm.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.attn_q_norm.weight`)),
    kNorm: hasKv ? createF32HandleFromBytes(arena, tensorReader, `blk.${layer}.attn_k_norm.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.attn_k_norm.weight`)) : undefined,
    attnOut: createQuantizedHandleFromTensorBytes(arena, tensorReader, `blk.${layer}.attn_output.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.attn_output.weight`)),
    postAttentionNorm: createF32HandleFromBytes(arena, tensorReader, `blk.${layer}.post_attention_norm.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.post_attention_norm.weight`)),
    ffnNorm: createF32HandleFromBytes(arena, tensorReader, `blk.${layer}.ffn_norm.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.ffn_norm.weight`)),
    ffnGate: createQuantizedHandleFromTensorBytes(arena, tensorReader, `blk.${layer}.ffn_gate.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.ffn_gate.weight`)),
    ffnUp: createQuantizedHandleFromTensorBytes(arena, tensorReader, `blk.${layer}.ffn_up.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.ffn_up.weight`)),
    ffnDown: createQuantizedHandleFromTensorBytes(arena, tensorReader, `blk.${layer}.ffn_down.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.ffn_down.weight`)),
    postFfwNorm: createF32HandleFromBytes(arena, tensorReader, `blk.${layer}.post_ffw_norm.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.post_ffw_norm.weight`)),
    perLayerInputGate: hasPerLayerInput
      ? createF32HandleFromBytes(arena, tensorReader, `blk.${layer}.inp_gate.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.inp_gate.weight`))
      : undefined,
    perLayerProjection: hasPerLayerInput
      ? createF32HandleFromBytes(arena, tensorReader, `blk.${layer}.proj.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.proj.weight`))
      : undefined,
    postNorm: hasPerLayerInput
      ? createF32HandleFromBytes(arena, tensorReader, `blk.${layer}.post_norm.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.post_norm.weight`))
      : undefined,
    layerOutputScale: createF32HandleFromBytes(arena, tensorReader, `blk.${layer}.layer_output_scale.weight`, requireTensorBytes(tensorBytes, `blk.${layer}.layer_output_scale.weight`)),
  };
}

export async function loadF32Handle(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
  name: string,
): Promise<F32Handle> {
  const tensor = tensorReader.getTensor(name);
  if (tensor.type !== "F32") {
    throw new Error(`${name} must be F32 for WebGPU segment execution, got ${tensor.type}`);
  }
  const bytes = await tensorReader.readTensorBytes(name);
  const buffer = arena.createBuffer(name, bytes.byteLength, GPU_STORAGE | GPU_COPY_DST);
  arena.device.queue.writeBuffer(buffer, 0, bytes);
  return {
    length: tensor.dimensions.reduce((product, dimension) => product * dimension, 1),
    byteLength: bytes.byteLength,
    device: arena.device,
    buffer,
    destroy: () => buffer.destroy?.(),
  };
}

function createF32HandleFromBytes(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
  name: string,
  bytes: Uint8Array,
): F32Handle {
  const tensor = tensorReader.getTensor(name);
  if (tensor.type !== "F32") {
    throw new Error(`${name} must be F32 for WebGPU segment execution, got ${tensor.type}`);
  }
  const buffer = arena.createBuffer(name, bytes.byteLength, GPU_STORAGE | GPU_COPY_DST);
  arena.device.queue.writeBuffer(buffer, 0, bytes);
  return {
    length: tensor.dimensions.reduce((product, dimension) => product * dimension, 1),
    byteLength: bytes.byteLength,
    device: arena.device,
    buffer,
    destroy: () => buffer.destroy?.(),
  };
}

export async function loadQuantizedHandle(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
  name: string,
): Promise<QuantizedHandle> {
  const tensor = tensorReader.getTensor(name);
  return createQuantizedHandleFromBytes(
    arena,
    name,
    webGpuMatMulType(tensor.type, name),
    tensor.dimensions[0] ?? 0,
    tensor.dimensions[1] ?? 0,
    await tensorReader.readTensorBytes(name),
  );
}

function createQuantizedHandleFromTensorBytes(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
  name: string,
  bytes: Uint8Array,
): QuantizedHandle {
  const tensor = tensorReader.getTensor(name);
  return createQuantizedHandleFromBytes(
    arena,
    name,
    webGpuMatMulType(tensor.type, name),
    tensor.dimensions[0] ?? 0,
    tensor.dimensions[1] ?? 0,
    bytes,
  );
}

export async function loadOutputStripes(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
  manifest: ModelManifest,
): Promise<OutputStripe[]> {
  const tensorName = tensorReader.metadata.tensors.some((tensor) => tensor.name === "output.weight")
    ? "output.weight"
    : "token_embd.weight";
  const tensor = tensorReader.getTensor(tensorName);
  const type = webGpuMatMulType(tensor.type, tensorName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  if (inputSize !== manifest.embeddingLength) {
    throw new Error(`${tensorName} input mismatch: ${inputSize}`);
  }
  const layout = webGpuQuantizedWeightLayout(type, inputSize);
  const limits = await webGpuAdapterLimits();
  const bindingLimit = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const targetBytes = Math.max(layout.rowByteLength, Math.min(bindingLimit, 128 * 1024 * 1024));
  const rowsPerStripe = Math.max(1, Math.floor(targetBytes / layout.rowByteLength));
  const stripeRanges: TensorByteRange[] = [];
  const stripeRowCounts: number[] = [];
  const stripeOffsets: number[] = [];
  for (let rowOffset = 0; rowOffset < rowCount; rowOffset += rowsPerStripe) {
    const stripeRowCount = Math.min(rowsPerStripe, rowCount - rowOffset);
    stripeRowCounts.push(stripeRowCount);
    stripeOffsets.push(rowOffset);
    stripeRanges.push({
      tensor,
      offset: BigInt(rowOffset * layout.rowByteLength),
      length: stripeRowCount * layout.rowByteLength,
    });
  }
  const stripeBytes = await tensorReader.readTensorRangesCoalesced(stripeRanges, {
    maxGapBytes: WEBGPU_LOAD_COALESCE_MAX_GAP_BYTES,
    maxReadBytes: WEBGPU_LOAD_COALESCE_MAX_READ_BYTES,
    copyResults: false,
  });
  return stripeBytes.map((bytes, index) => {
    const rowOffset = stripeOffsets[index] ?? 0;
    const rows = stripeRowCounts[index] ?? 0;
    return {
      ...createQuantizedHandleFromBytes(
        arena,
        `${tensorName}.${rowOffset}`,
        type,
        inputSize,
        rows,
        bytes,
      ),
      rowOffset,
    };
  });
}

async function readNamedTensorBytesCoalesced(
  tensorReader: GgufTensorReader,
  names: readonly string[],
): Promise<ReadonlyMap<string, Uint8Array>> {
  const ranges = names.map((name): TensorByteRange => {
    const tensor = tensorReader.getTensor(name);
    return {
      tensor,
      offset: 0n,
      length: tensorByteLength(tensor),
    };
  });
  const bytes = await tensorReader.readTensorRangesCoalesced(ranges, {
    maxGapBytes: WEBGPU_LOAD_COALESCE_MAX_GAP_BYTES,
    maxReadBytes: WEBGPU_LOAD_COALESCE_MAX_READ_BYTES,
    copyResults: false,
  });
  return new Map(names.map((name, index) => [name, bytes[index] ?? new Uint8Array()]));
}

function requireTensorBytes(bytesByName: ReadonlyMap<string, Uint8Array>, name: string): Uint8Array {
  const bytes = bytesByName.get(name);
  if (!bytes) {
    throw new Error(`Missing coalesced tensor bytes for ${name}`);
  }
  return bytes;
}
