import type { Gemma4LayerKind, Gemma4ModelManifest } from "../../model";
import type { GgufTensorReader } from "../../tensor-reader";
import { GPU_COPY_DST, GPU_STORAGE } from "./gpu-constants";
import { webGpuAdapterLimits } from "./gpu-device";
import { createQuantizedHandleFromBytes, webGpuMatMulType, webGpuQuantizedWeightLayout } from "./quantized-handles";
import type { F32Handle, GpuMemoryArena, QuantizedHandle } from "./arena";

export type OutputStripe = QuantizedHandle & {
  readonly rowOffset: number;
};

export type Gemma4GpuLayer = {
  kind: Gemma4LayerKind;
  layer: number;
  hasKv: boolean;
  kvSourceLayer: number;
  headSize: number;
  valueSize: number;
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
  manifest: Gemma4ModelManifest,
  layer: number,
): Promise<Gemma4GpuLayer> {
  const hasPerLayerInput = manifest.perLayerEmbeddingLength > 0;
  const hasKv = manifest.layerHasKv[layer] === true;
  return {
    kind: manifest.layerKinds[layer] ?? "sliding-attention",
    layer,
    hasKv,
    kvSourceLayer: manifest.kvSourceLayers[layer] ?? layer,
    headSize: manifest.layerKeyLengths[layer] ?? manifest.keyLength,
    valueSize: manifest.layerValueLengths[layer] ?? manifest.valueLength,
    attnNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.attn_norm.weight`),
    q: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_q.weight`),
    k: hasKv ? await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_k.weight`) : undefined,
    v: hasKv ? await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_v.weight`) : undefined,
    qNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.attn_q_norm.weight`),
    kNorm: hasKv ? await loadF32Handle(arena, tensorReader, `blk.${layer}.attn_k_norm.weight`) : undefined,
    attnOut: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_output.weight`),
    postAttentionNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.post_attention_norm.weight`),
    ffnNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.ffn_norm.weight`),
    ffnGate: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.ffn_gate.weight`),
    ffnUp: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.ffn_up.weight`),
    ffnDown: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.ffn_down.weight`),
    postFfwNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.post_ffw_norm.weight`),
    perLayerInputGate: hasPerLayerInput
      ? await loadF32Handle(arena, tensorReader, `blk.${layer}.inp_gate.weight`)
      : undefined,
    perLayerProjection: hasPerLayerInput
      ? await loadF32Handle(arena, tensorReader, `blk.${layer}.proj.weight`)
      : undefined,
    postNorm: hasPerLayerInput
      ? await loadF32Handle(arena, tensorReader, `blk.${layer}.post_norm.weight`)
      : undefined,
    layerOutputScale: await loadF32Handle(arena, tensorReader, `blk.${layer}.layer_output_scale.weight`),
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

export async function loadOutputStripes(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
  manifest: Gemma4ModelManifest,
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
  const stripes: OutputStripe[] = [];
  for (let rowOffset = 0; rowOffset < rowCount; rowOffset += rowsPerStripe) {
    const stripeRows = Math.min(rowsPerStripe, rowCount - rowOffset);
    const bytes = await tensorReader.readTensorRange({
      tensor,
      offset: BigInt(rowOffset * layout.rowByteLength),
      length: stripeRows * layout.rowByteLength,
    });
    stripes.push({
      ...createQuantizedHandleFromBytes(
        arena,
        `${tensorName}.${rowOffset}`,
        type,
        inputSize,
        stripeRows,
        bytes,
      ),
      rowOffset,
    });
  }
  return stripes;
}
