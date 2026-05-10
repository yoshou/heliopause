import type { Gemma4ModelManifest } from "../../model";
import type { GgufTensorReader } from "../../tensor-reader";
import { GPU_COPY_DST, GPU_STORAGE } from "./gpu-constants";
import { webGpuAdapterLimits } from "./gpu-device";
import { createQuantizedHandleFromBytes, webGpuMatMulType, webGpuQuantizedWeightLayout } from "./quantized-handles";
import type { F32Handle, GpuMemoryArena, QuantizedHandle } from "./arena";

export type OutputStripe = QuantizedHandle & {
  readonly rowOffset: number;
};

export type RecurrentGpuLayer = {
  kind: "recurrent";
  layer: number;
  attnNorm: F32Handle;
  qkv: QuantizedHandle;
  alpha: F32Handle;
  beta: F32Handle;
  z: QuantizedHandle;
  convKernel: F32Handle;
  dtBias: F32Handle;
  ssmA: F32Handle;
  ssmNorm: F32Handle;
  out: QuantizedHandle;
  postNorm: F32Handle;
  ffnGate: QuantizedHandle;
  ffnUp: QuantizedHandle;
  ffnDown: QuantizedHandle;
};

export type FullAttentionGpuLayer = {
  kind: "full-attention";
  layer: number;
  attnNorm: F32Handle;
  q: QuantizedHandle;
  k: QuantizedHandle;
  v: QuantizedHandle;
  out: QuantizedHandle;
  qNorm: F32Handle;
  kNorm: F32Handle;
  postNorm: F32Handle;
  ffnGate: QuantizedHandle;
  ffnUp: QuantizedHandle;
  ffnDown: QuantizedHandle;
};

export type GpuLayer = RecurrentGpuLayer | FullAttentionGpuLayer;

export async function loadGpuLayer(
  arena: GpuMemoryArena,
  tensorReader: GgufTensorReader,
  manifest: Gemma4ModelManifest,
  layer: number,
): Promise<GpuLayer> {
  const ffn = {
    postNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.post_attention_norm.weight`),
    ffnGate: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.ffn_gate.weight`),
    ffnUp: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.ffn_up.weight`),
    ffnDown: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.ffn_down.weight`),
  };
  if (manifest.fullAttentionLayers.includes(layer)) {
    return {
      kind: "full-attention",
      layer,
      attnNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.attn_norm.weight`),
      q: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_q.weight`),
      k: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_k.weight`),
      v: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_v.weight`),
      out: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_output.weight`),
      qNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.attn_q_norm.weight`),
      kNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.attn_k_norm.weight`),
      ...ffn,
    };
  }
  return {
    kind: "recurrent",
    layer,
    attnNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.attn_norm.weight`),
    qkv: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_qkv.weight`),
    alpha: await loadF32Handle(arena, tensorReader, `blk.${layer}.ssm_alpha.weight`),
    beta: await loadF32Handle(arena, tensorReader, `blk.${layer}.ssm_beta.weight`),
    z: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.attn_gate.weight`),
    convKernel: await loadF32Handle(arena, tensorReader, `blk.${layer}.ssm_conv1d.weight`),
    dtBias: await loadF32Handle(arena, tensorReader, `blk.${layer}.ssm_dt.bias`),
    ssmA: await loadF32Handle(arena, tensorReader, `blk.${layer}.ssm_a`),
    ssmNorm: await loadF32Handle(arena, tensorReader, `blk.${layer}.ssm_norm.weight`),
    out: await loadQuantizedHandle(arena, tensorReader, `blk.${layer}.ssm_out.weight`),
    ...ffn,
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
  const tensor = tensorReader.getTensor("output.weight");
  const type = webGpuMatMulType(tensor.type, "output.weight");
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  if (inputSize !== manifest.embeddingLength) {
    throw new Error(`output.weight input mismatch: ${inputSize}`);
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
        `output.weight.${rowOffset}`,
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
