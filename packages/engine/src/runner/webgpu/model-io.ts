import type {
  GgmlTypeName,
} from "../../gguf";
import {
  dequantizeRow,
} from "../../quant";
import type {
  ForwardTrace,
  ModelSession,
} from "../../runtime";
import {
  timedAsync,
} from "../../runtime";
import type {
  ModelPreparedInput,
} from "../model-runner";
import {
  GpuMemoryArena,
  scratchQ8_0,
  scratchQ8K,
  type F32Handle,
  type GpuResource,
  type QuantizedHandle,
} from "./arena";
import {
  GPU_COPY_DST,
  GPU_COPY_SRC,
  GPU_MAP_READ,
  GPU_STORAGE,
} from "./gpu-constants";
import {
  webGpuDevice,
} from "./gpu-device";
import type {
  WebGpuBufferLike,
  WebGpuComputePassLike,
  WebGpuDeviceLike,
} from "./gpu-types";
import type {
  WebGpuPreparedInput,
} from "./segment-runner";
import {
  dispatchF32GatherRowsScale,
  dispatchF32MatMul,
  dispatchKMatMul,
  dispatchPreparePerLayerInputs,
  dispatchQ8_0MatMul,
  dispatchQ8_0Quantize,
  dispatchQ8KQuantize,
  dispatchQuantizedGatherRowsScale,
} from "./dispatch";
import {
  loadF32Handle,
  loadQuantizedHandle,
} from "./segment-layer-loader";
import {
  webGpuExecutionProviderOptions,
} from "./execution-provider";

export async function prepareWebGpuInput(
  session: ModelSession,
  tokenIds: readonly number[],
  trace?: ForwardTrace,
): Promise<ModelPreparedInput> {
  if (tokenIds.length === 0) {
    return { hidden: new Float32Array() };
  }
  return timedAsync(trace, "WebGPU input preparation", () =>
    runWebGpuInputPreparation(session, tokenIds)
  );
}

export async function prepareWebGpuPreparedHiddenInputHandle(
  session: ModelSession,
  hidden: Float32Array,
  trace?: ForwardTrace,
): Promise<WebGpuPreparedInput> {
  const tokenCount = hidden.length / session.manifest.embeddingLength;
  if (!Number.isInteger(tokenCount)) {
    throw new Error(`Prepared hidden shape mismatch: ${hidden.length}`);
  }
  if (tokenCount <= 0) {
    throw new Error("WebGPU prepared hidden input requires at least one token.");
  }
  return timedAsync(trace, "WebGPU prepared hidden input handle", () =>
    createWebGpuPreparedHiddenInputHandle(session, hidden, tokenCount)
  );
}

async function runWebGpuInputPreparation(
  session: ModelSession,
  tokenIds: readonly number[],
): Promise<ModelPreparedInput> {
  const device = await requireDevice();
  const arena = new GpuMemoryArena(device, webGpuMemoryLimitBytes(session));
  const cleanup: GpuResource[] = [];
  const resources: Array<{ destroy: () => void }> = [];
  try {
    const manifest = session.manifest;
    const hiddenSize = manifest.embeddingLength;
    const tokenCount = tokenIds.length;
    if (!Number.isInteger(tokenCount) || tokenCount <= 0) {
      throw new Error(`WebGPU input token count mismatch: ${tokenCount}`);
    }

    const hidden = arena.createScratchBuffer(
      "model-input.hidden",
      tokenCount * hiddenSize * Float32Array.BYTES_PER_ELEMENT,
      GPU_STORAGE | GPU_COPY_SRC | GPU_COPY_DST,
    );
    cleanup.push(hidden);
    const inputTokenIds = Uint32Array.from(tokenIds);
    const tokenIdBuffer = arena.createScratchBuffer("model-input.token_ids", inputTokenIds.byteLength, GPU_STORAGE | GPU_COPY_DST);
    cleanup.push(tokenIdBuffer);
    device.queue.writeBuffer(tokenIdBuffer, 0, inputTokenIds);

    const hiddenReadback = device.createBuffer({
      label: "model-input.hidden.readback",
      size: tokenCount * hiddenSize * Float32Array.BYTES_PER_ELEMENT,
      usage: GPU_COPY_DST | GPU_MAP_READ,
    });
    cleanup.push(hiddenReadback);

    let perLayerInputs: WebGpuBufferLike | undefined;
    let perLayerReadback: WebGpuBufferLike | undefined;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    const tokenEmbedding = await loadEmbeddingGatherHandle(arena, session, "token_embd.weight");
    cleanup.push(tokenEmbedding);
    dispatchGatherRowsScale(device, pass, resources, tokenEmbedding, tokenIdBuffer, hidden, {
      rowSize: hiddenSize,
      tokenCount,
      scale: Math.sqrt(hiddenSize),
    });

    if (manifest.perLayerEmbeddingLength > 0) {
      const perLayerLength = manifest.perLayerEmbeddingLength;
      const totalPerLayerLength = perLayerLength * manifest.blockCount;
      const perLayerTokenEmbedding = await loadEmbeddingGatherHandle(arena, session, "per_layer_token_embd.weight");
      const modelProjection = await loadProjectionHandle(arena, session, "per_layer_model_proj.weight");
      const projectionNorm = await loadF32Handle(arena, session.tensorReader, "per_layer_proj_norm.weight");
      cleanup.push(perLayerTokenEmbedding, modelProjection, projectionNorm);

      const tokenRows = arena.createScratchBuffer(
        "model-input.per_layer_token_rows",
        tokenCount * totalPerLayerLength * Float32Array.BYTES_PER_ELEMENT,
        GPU_STORAGE,
      );
      const projected = arena.createScratchBuffer(
        "model-input.per_layer_projected",
        tokenCount * totalPerLayerLength * Float32Array.BYTES_PER_ELEMENT,
        GPU_STORAGE,
      );
      perLayerInputs = arena.createScratchBuffer(
        "model-input.per_layer_inputs",
        tokenCount * totalPerLayerLength * Float32Array.BYTES_PER_ELEMENT,
        GPU_STORAGE | GPU_COPY_SRC,
      );
      cleanup.push(tokenRows, projected, perLayerInputs);
      dispatchGatherRowsScale(device, pass, resources, perLayerTokenEmbedding, tokenIdBuffer, tokenRows, {
        rowSize: totalPerLayerLength,
        tokenCount,
        scale: Math.sqrt(perLayerLength),
      });
      dispatchMatMul(device, pass, resources, arena, cleanup, modelProjection, hidden, projected, hiddenSize, totalPerLayerLength, tokenCount);
      dispatchPreparePerLayerInputs(
        device,
        pass,
        resources,
        tokenRows,
        projected,
        projectionNorm.buffer,
        perLayerInputs,
        {
          perLayerLength,
          totalPerLayerLength,
          tokenCount,
          blockCount: manifest.blockCount,
          projectionScale: 1 / Math.sqrt(hiddenSize),
          epsilon: session.epsilon,
        },
      );
      perLayerReadback = device.createBuffer({
        label: "model-input.per_layer_inputs.readback",
        size: tokenCount * totalPerLayerLength * Float32Array.BYTES_PER_ELEMENT,
        usage: GPU_COPY_DST | GPU_MAP_READ,
      });
      cleanup.push(perLayerReadback);
    }

    pass.end();
    encoder.copyBufferToBuffer(hidden, 0, hiddenReadback, 0, tokenCount * hiddenSize * Float32Array.BYTES_PER_ELEMENT);
    if (perLayerInputs && perLayerReadback) {
      encoder.copyBufferToBuffer(perLayerInputs, 0, perLayerReadback, 0, tokenCount * manifest.blockCount * manifest.perLayerEmbeddingLength * Float32Array.BYTES_PER_ELEMENT);
    }
    device.queue.submit([encoder.finish()]);

    const hiddenValues = await readMappedF32(hiddenReadback, tokenCount * hiddenSize);
    const perLayerValues = perLayerReadback
      ? await readMappedF32(perLayerReadback, tokenCount * manifest.blockCount * manifest.perLayerEmbeddingLength)
      : undefined;
    return {
      hidden: hiddenValues,
      perLayerInputs: perLayerValues,
    };
  } finally {
    destroyAll(resources);
    destroyAll(cleanup.reverse());
    arena.destroyScratchBuffers();
  }
}

async function createWebGpuPreparedHiddenInputHandle(
  session: ModelSession,
  hiddenValues: Float32Array,
  tokenCount: number,
): Promise<WebGpuPreparedInput> {
  const device = await requireDevice();
  const arena = new GpuMemoryArena(device, webGpuMemoryLimitBytes(session));
  const cleanup: GpuResource[] = [];
  const resources: Array<{ destroy: () => void }> = [];
  try {
    const manifest = session.manifest;
    const hiddenSize = manifest.embeddingLength;
    const hidden = arena.createScratchBuffer(
      "model-input.prepared_hidden",
      hiddenValues.byteLength,
      GPU_STORAGE | GPU_COPY_SRC | GPU_COPY_DST,
    );
    cleanup.push(hidden);
    device.queue.writeBuffer(hidden, 0, hiddenValues);

    let perLayerInputs: WebGpuBufferLike | undefined;
    if (manifest.perLayerEmbeddingLength > 0) {
      const perLayerLength = manifest.perLayerEmbeddingLength;
      const totalPerLayerLength = perLayerLength * manifest.blockCount;
      const paddingTokenIds = arena.createScratchBuffer(
        "model-input.padding_token_ids",
        tokenCount * Uint32Array.BYTES_PER_ELEMENT,
        GPU_STORAGE | GPU_COPY_DST,
      );
      const tokenRows = arena.createScratchBuffer(
        "model-input.per_layer_token_rows",
        tokenCount * totalPerLayerLength * Float32Array.BYTES_PER_ELEMENT,
        GPU_STORAGE,
      );
      const projected = arena.createScratchBuffer(
        "model-input.per_layer_projected",
        tokenCount * totalPerLayerLength * Float32Array.BYTES_PER_ELEMENT,
        GPU_STORAGE,
      );
      perLayerInputs = arena.createScratchBuffer(
        "model-input.per_layer_inputs",
        tokenCount * totalPerLayerLength * Float32Array.BYTES_PER_ELEMENT,
        GPU_STORAGE | GPU_COPY_SRC,
      );
      cleanup.push(paddingTokenIds, tokenRows, projected, perLayerInputs);
      device.queue.writeBuffer(paddingTokenIds, 0, new Uint32Array(tokenCount));

      const perLayerTokenEmbedding = await loadEmbeddingGatherHandle(arena, session, "per_layer_token_embd.weight");
      const modelProjection = await loadProjectionHandle(arena, session, "per_layer_model_proj.weight");
      const projectionNorm = await loadF32Handle(arena, session.tensorReader, "per_layer_proj_norm.weight");
      cleanup.push(perLayerTokenEmbedding, modelProjection, projectionNorm);

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      dispatchGatherRowsScale(device, pass, resources, perLayerTokenEmbedding, paddingTokenIds, tokenRows, {
        rowSize: totalPerLayerLength,
        tokenCount,
        scale: Math.sqrt(perLayerLength),
      });
      dispatchMatMul(device, pass, resources, arena, cleanup, modelProjection, hidden, projected, hiddenSize, totalPerLayerLength, tokenCount);
      dispatchPreparePerLayerInputs(
        device,
        pass,
        resources,
        tokenRows,
        projected,
        projectionNorm.buffer,
        perLayerInputs,
        {
          perLayerLength,
          totalPerLayerLength,
          tokenCount,
          blockCount: manifest.blockCount,
          projectionScale: 1 / Math.sqrt(hiddenSize),
          epsilon: session.epsilon,
        },
      );
      pass.end();
      device.queue.submit([encoder.finish()]);
    }

    return {
      tokenCount,
      hidden,
      perLayerInputs,
      destroy: () => {
        destroyAll(resources);
        destroyAll(cleanup.reverse());
        arena.destroyScratchBuffers();
      },
    };
  } catch (error) {
    destroyAll(resources);
    destroyAll(cleanup.reverse());
    arena.destroyScratchBuffers();
    throw error;
  }
}

function dispatchGatherRowsScale(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  handle: F32Handle | QuantizedHandle,
  tokenIds: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: { rowSize: number; tokenCount: number; scale: number },
): void {
  if (isF32Handle(handle)) {
    dispatchF32GatherRowsScale(device, pass, resources, handle.buffer, tokenIds, output, options);
    return;
  }
  dispatchQuantizedGatherRowsScale(pass, resources, handle, tokenIds, output, options);
}

function dispatchMatMul(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  arena: GpuMemoryArena,
  cleanup: GpuResource[],
  handle: F32Handle | QuantizedHandle,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): void {
  if (isF32Handle(handle)) {
    dispatchF32MatMul(device, pass, resources, handle.buffer, input, output, inputSize, rowCount, columnCount);
    return;
  }
  if (handle.type === "Q8_0") {
    const q8 = scratchQ8_0(arena, inputSize, columnCount, inputSize / 32, cleanup, "model-input.hidden.q8_0");
    dispatchQ8_0Quantize(device, pass, resources, input, q8, inputSize, columnCount, inputSize / 32);
    dispatchQ8_0MatMul(pass, resources, handle, q8, output, columnCount);
    return;
  }
  const q8 = scratchQ8K(arena, inputSize, columnCount, cleanup, "model-input.hidden.q8k");
  dispatchQ8KQuantize(device, pass, resources, input, q8, inputSize, columnCount);
  dispatchKMatMul(pass, resources, handle, q8, output, columnCount);
}

async function loadEmbeddingGatherHandle(
  arena: GpuMemoryArena,
  session: ModelSession,
  name: string,
): Promise<F32Handle | QuantizedHandle> {
  const tensor = session.tensorReader.getTensor(name);
  if (tensor.type === "F32") {
    return loadF32Handle(arena, session.tensorReader, name);
  }
  if (isSupportedEmbeddingGatherType(tensor.type)) {
    return loadQuantizedHandle(arena, session.tensorReader, name);
  }
  throw new Error(`${name} has unsupported WebGPU gather type ${tensor.type}`);
}

async function loadProjectionHandle(
  arena: GpuMemoryArena,
  session: ModelSession,
  name: string,
): Promise<F32Handle | QuantizedHandle> {
  const tensor = session.tensorReader.getTensor(name);
  if (tensor.type === "F32") {
    return loadF32Handle(arena, session.tensorReader, name);
  }
  if (isF32CompatibleType(tensor.type)) {
    const elementCount = tensor.dimensions.reduce((product, dimension) => product * dimension, 1);
    const source = await session.tensorReader.readTensorBytes(name);
    const values = dequantizeRow(tensor.type, source, elementCount);
    const buffer = arena.createBuffer(name, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
    arena.device.queue.writeBuffer(buffer, 0, values);
    return {
      length: elementCount,
      byteLength: values.byteLength,
      device: arena.device,
      buffer,
      destroy: () => buffer.destroy?.(),
    };
  }
  if (isSupportedProjectionType(tensor.type)) {
    return loadQuantizedHandle(arena, session.tensorReader, name);
  }
  throw new Error(`${name} has unsupported WebGPU projection type ${tensor.type}`);
}

async function requireDevice(): Promise<WebGpuDeviceLike> {
  const device = await webGpuDevice();
  if (!device) {
    throw new Error("WebGPU is not available for model runner input/output.");
  }
  return device;
}

function webGpuMemoryLimitBytes(session: ModelSession): number {
  const options = webGpuExecutionProviderOptions(session);
  if (!options) {
    throw new Error("WebGPU model runner is not enabled for this session.");
  }
  return options.memoryLimitBytes;
}

async function readMappedF32(buffer: WebGpuBufferLike, length: number): Promise<Float32Array> {
  await buffer.mapAsync(GPU_MAP_READ);
  const values = new Float32Array(buffer.getMappedRange()).slice(0, length);
  buffer.unmap();
  return values;
}

function isF32Handle(value: F32Handle | QuantizedHandle): value is F32Handle {
  return "buffer" in value;
}

function isSupportedEmbeddingGatherType(type: GgmlTypeName): boolean {
  return type === "F32" || type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "Q8_0";
}

function isF32CompatibleType(type: GgmlTypeName): boolean {
  return type === "F32" || type === "F16" || type === "BF16";
}

function isSupportedProjectionType(type: GgmlTypeName): boolean {
  return isF32CompatibleType(type) || type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "Q8_0";
}

function destroyAll(items: readonly { destroy?: () => void }[]): void {
  for (const item of items) {
    item.destroy?.();
  }
}
