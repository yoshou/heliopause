import type { GgmlTypeName } from "../../gguf";
import { dequantizeRow } from "../../quant";
import {
  type VisionEncodeResult,
  type VisionPixelValues,
  type VisionSession,
} from "../../vision";
import {
  GpuMemoryArena,
  scratchQ8_0,
  scratchQ8K,
  type F32Handle,
  type GpuResource,
  type QuantizedHandle,
} from "./arena";
import { GPU_COPY_DST, GPU_COPY_SRC, GPU_MAP_READ, GPU_STORAGE, WEBGPU_MEMORY_LIMIT_BYTES } from "./gpu-constants";
import { webGpuDevice } from "./gpu-device";
import {
  dispatchF32MatMul,
  dispatchKMatMul,
  dispatchQ8_0MatMul,
  dispatchQ8_0Quantize,
  dispatchQ8KQuantize,
  dispatchResidualAdd,
} from "./dispatch";
import type { WebGpuBufferLike, WebGpuComputePassLike } from "./gpu-types";
import { createQuantizedHandleFromBytes, webGpuMatMulType } from "./quantized-handles";
import {
  createVisionAddPositionResources,
  createVisionAttentionApplyResources,
  createVisionAttentionScoreResources,
  createVisionAveragePoolResources,
  createVisionClampResources,
  createVisionGeluMulResources,
  createVisionPatchEmbedResources,
  createVisionRmsNormResources,
  createVisionRope2dResources,
  createVisionStdNormalizeResources,
} from "./vision-kernel-resources";

type VisionStats = {
  attempts: number;
  runs: number;
};

type VisionRunBuffers = {
  resources: Array<{ destroy: () => void }>;
  cleanup: GpuResource[];
};

const runners = new WeakMap<VisionSession, Promise<WebGpuVisionRunner | undefined>>();
const statsBySession = new WeakMap<VisionSession, VisionStats>();

export async function runWebGpuVisionEncoder(
  session: VisionSession,
  pixels: VisionPixelValues,
): Promise<VisionEncodeResult> {
  if (!session.executionProvider("webgpu")) {
    throw new Error("WebGPU vision encoder provider is not enabled.");
  }
  const stats = visionStats(session);
  stats.attempts += 1;
  const runner = await visionRunner(session);
  if (!runner) {
    throw new Error("WebGPU vision encoder is unavailable.");
  }
  const result = await runner.run(pixels);
  stats.runs += 1;
  return result;
}

function visionStats(session: VisionSession): VisionStats {
  let stats = statsBySession.get(session);
  if (!stats) {
    stats = {
      attempts: 0,
      runs: 0,
    };
    statsBySession.set(session, stats);
    const captured = stats;
    session.setExecutionProviderStatsProvider(() => ({
      webgpuVisionAttempts: captured.attempts,
      webgpuVisionRuns: captured.runs,
    }), "webgpu-vision");
  }
  return stats;
}

async function visionRunner(session: VisionSession): Promise<WebGpuVisionRunner | undefined> {
  let runner = runners.get(session);
  if (!runner) {
    runner = createVisionRunner(session);
    runners.set(session, runner);
  }
  return runner;
}

async function createVisionRunner(session: VisionSession): Promise<WebGpuVisionRunner | undefined> {
  const device = await webGpuDevice();
  if (!device) {
    throw new Error("WebGPU is not available for vision encoder execution.");
  }
  const options = session.executionProvider("webgpu")?.options;
  const arena = new GpuMemoryArena(
    device,
    numberOption(options, "memoryLimitBytes") ?? WEBGPU_MEMORY_LIMIT_BYTES,
  );
  const runner = new WebGpuVisionRunner(session, arena);
  session.addDisposeCallback(() => runner.dispose());
  return runner;
}

class WebGpuVisionRunner {
  private readonly session: VisionSession;
  private readonly arena: GpuMemoryArena;
  private readonly f32Handles = new Map<string, F32Handle>();
  private readonly quantizedHandles = new Map<string, QuantizedHandle>();

  constructor(session: VisionSession, arena: GpuMemoryArena) {
    this.session = session;
    this.arena = arena;
  }

  dispose(): void {
    for (const handle of this.f32Handles.values()) {
      handle.destroy();
    }
    for (const handle of this.quantizedHandles.values()) {
      handle.destroy();
    }
    this.f32Handles.clear();
    this.quantizedHandles.clear();
  }

  async run(pixels: VisionPixelValues): Promise<VisionEncodeResult> {
    const manifest = this.session.manifest;
    const patchGridX = pixels.width / manifest.patchSize;
    const patchGridY = pixels.height / manifest.patchSize;
    if (!Number.isInteger(patchGridX) || !Number.isInteger(patchGridY)) {
      throw new Error(`Vision image size must be patch-aligned, got ${pixels.width}x${pixels.height}`);
    }
    if (patchGridX % manifest.spatialMergeSize !== 0 || patchGridY % manifest.spatialMergeSize !== 0) {
      throw new Error(`Vision image size must be merge-aligned, got ${pixels.width}x${pixels.height}`);
    }

    const run: VisionRunBuffers = { resources: [], cleanup: [] };
    const outputTokenCount = (patchGridX / manifest.spatialMergeSize) * (patchGridY / manifest.spatialMergeSize);
    const outputLength = outputTokenCount * manifest.projectionDim;
    let readback: WebGpuBufferLike | undefined;
    try {
      const pixelsBuffer = this.bufferFromF32("vision.pixels", pixels.values, run.cleanup);
      const encoder = this.arena.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      let hidden = await this.dispatchPatchEmbed(pass, run, pixelsBuffer, pixels.width, patchGridX, patchGridY);
      let next = await this.dispatchAddPosition(pass, run, hidden, patchGridX, patchGridX * patchGridY);
      this.releaseScratch(hidden);
      hidden = next;
      for (let layer = 0; layer < manifest.blockCount; layer += 1) {
        next = await this.dispatchLayer(pass, run, hidden, layer, patchGridX, patchGridX * patchGridY);
        hidden = next;
      }
      next = this.dispatchAveragePool(pass, run, hidden, patchGridX, patchGridY);
      this.releaseScratch(hidden);
      hidden = next;
      if (this.session.hasTensor("v.std_bias") && this.session.hasTensor("v.std_scale")) {
        next = await this.dispatchStdNormalize(pass, run, hidden, outputTokenCount);
        this.releaseScratch(hidden);
        hidden = next;
      }
      next = await this.dispatchMatMulVisionWeight(pass, run, "mm.input_projection.weight", hidden, outputTokenCount);
      this.releaseScratch(hidden);
      hidden = next;
      next = this.dispatchRowRmsNorm(pass, run, hidden, "vision.output_norm", outputTokenCount, manifest.projectionDim, undefined, manifest.layerNormEpsilon);
      this.releaseScratch(hidden);
      hidden = next;

      pass.end();
      readback = this.arena.device.createBuffer({
        size: outputLength * Float32Array.BYTES_PER_ELEMENT,
        usage: GPU_MAP_READ | GPU_COPY_DST,
      });
      encoder.copyBufferToBuffer(hidden, 0, readback, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
      this.arena.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPU_MAP_READ);
      const result = new Float32Array(readback.getMappedRange()).slice();
      readback.unmap();
      readback.destroy?.();
      readback = undefined;
      return {
        hidden: result,
        tokenCount: outputTokenCount,
        width: pixels.width,
        height: pixels.height,
      };
    } finally {
      readback?.destroy?.();
      for (const resource of run.resources) {
        resource.destroy();
      }
      for (const item of run.cleanup.reverse()) {
        item.destroy?.();
      }
      this.arena.destroyScratchBuffers();
    }
  }

  private async dispatchPatchEmbed(
    pass: WebGpuComputePassLike,
    run: VisionRunBuffers,
    pixels: WebGpuBufferLike,
    imageWidth: number,
    patchGridX: number,
    patchGridY: number,
  ): Promise<WebGpuBufferLike> {
    const manifest = this.session.manifest;
    const weight = await this.f32Handle("v.patch_embd.weight");
    const output = this.scratchF32("vision.patch_embed", patchGridX * patchGridY * manifest.embeddingLength, run.cleanup);
    const resource = createVisionPatchEmbedResources(this.arena.device, pixels, weight.buffer, output, {
      imageWidth,
      patchSize: manifest.patchSize,
      patchGridX,
      patchGridY,
      embeddingLength: manifest.embeddingLength,
    });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil((patchGridX * patchGridY * manifest.embeddingLength) / 256));
    return output;
  }

  private async dispatchAddPosition(
    pass: WebGpuComputePassLike,
    run: VisionRunBuffers,
    input: WebGpuBufferLike,
    patchGridX: number,
    tokenCount: number,
  ): Promise<WebGpuBufferLike> {
    const manifest = this.session.manifest;
    const position = await this.f32Handle("v.position_embd.weight");
    const tensor = this.session.getTensor("v.position_embd.weight");
    const tableSize = tensor.dimensions[1] ?? 0;
    const output = this.scratchF32("vision.position", tokenCount * manifest.embeddingLength, run.cleanup);
    const resource = createVisionAddPositionResources(this.arena.device, input, position.buffer, output, {
      patchGridX,
      tokenCount,
      embeddingLength: manifest.embeddingLength,
      tableSize,
    });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil((tokenCount * manifest.embeddingLength) / 256));
    return output;
  }

  private async dispatchLayer(
    pass: WebGpuComputePassLike,
    run: VisionRunBuffers,
    input: WebGpuBufferLike,
    layer: number,
    patchGridX: number,
    tokenCount: number,
  ): Promise<WebGpuBufferLike> {
    const manifest = this.session.manifest;
    const headSize = manifest.embeddingLength / manifest.headCount;
    const norm = this.dispatchRowRmsNorm(pass, run, input, `v.blk.${layer}.ln1`, tokenCount, manifest.embeddingLength, await this.f32Handle(`v.blk.${layer}.ln1.weight`), this.session.epsilon);
    const qProjection = await this.dispatchMatMulVisionWeight(pass, run, `v.blk.${layer}.attn_q.weight`, norm, tokenCount);
    const kProjection = await this.dispatchMatMulVisionWeight(pass, run, `v.blk.${layer}.attn_k.weight`, norm, tokenCount);
    const vProjection = await this.dispatchMatMulVisionWeight(pass, run, `v.blk.${layer}.attn_v.weight`, norm, tokenCount);
    this.releaseScratch(norm);
    const qNorm = this.dispatchRowRmsNorm(pass, run, qProjection, `v.blk.${layer}.attn_q_norm`, tokenCount * manifest.headCount, headSize, await this.f32Handle(`v.blk.${layer}.attn_q_norm.weight`), this.session.epsilon);
    this.releaseScratch(qProjection);
    const kNorm = this.dispatchRowRmsNorm(pass, run, kProjection, `v.blk.${layer}.attn_k_norm`, tokenCount * manifest.headCount, headSize, await this.f32Handle(`v.blk.${layer}.attn_k_norm.weight`), this.session.epsilon);
    this.releaseScratch(kProjection);
    const vNorm = this.dispatchRowRmsNorm(pass, run, vProjection, `v.blk.${layer}.attn_v_norm`, tokenCount * manifest.headCount, headSize, undefined, this.session.epsilon);
    this.releaseScratch(vProjection);
    const q = this.dispatchRope2d(pass, run, qNorm, `v.blk.${layer}.q_rope`, patchGridX, tokenCount, headSize);
    this.releaseScratch(qNorm);
    const k = this.dispatchRope2d(pass, run, kNorm, `v.blk.${layer}.k_rope`, patchGridX, tokenCount, headSize);
    this.releaseScratch(kNorm);
    const attention = this.dispatchAttention(pass, run, q, k, vNorm, `v.blk.${layer}.attention`, tokenCount, manifest.headCount, headSize);
    this.releaseScratch(q, k, vNorm);
    const attentionOutput = await this.dispatchMatMulVisionWeight(pass, run, `v.blk.${layer}.attn_out.weight`, attention, tokenCount);
    this.releaseScratch(attention);
    const attentionNorm = this.dispatchRowRmsNorm(pass, run, attentionOutput, `v.blk.${layer}.attn_post_norm`, tokenCount, manifest.embeddingLength, await this.f32Handle(`v.blk.${layer}.attn_post_norm.weight`), this.session.epsilon);
    this.releaseScratch(attentionOutput);
    const attentionResidual = this.scratchF32(`v.blk.${layer}.attention_residual`, tokenCount * manifest.embeddingLength, run.cleanup);
    dispatchResidualAdd(this.arena.device, pass, run.resources, input, attentionNorm, attentionResidual, tokenCount * manifest.embeddingLength);
    this.releaseScratch(input, attentionNorm);
    const ffnInput = this.dispatchRowRmsNorm(pass, run, attentionResidual, `v.blk.${layer}.ln2`, tokenCount, manifest.embeddingLength, await this.f32Handle(`v.blk.${layer}.ln2.weight`), this.session.epsilon);
    const gate = await this.dispatchMatMulVisionWeight(pass, run, `v.blk.${layer}.ffn_gate.weight`, ffnInput, tokenCount);
    const up = await this.dispatchMatMulVisionWeight(pass, run, `v.blk.${layer}.ffn_up.weight`, ffnInput, tokenCount);
    this.releaseScratch(ffnInput);
    const activated = this.scratchF32(`v.blk.${layer}.ffn_geglu`, tokenCount * manifest.feedForwardLength, run.cleanup);
    this.dispatchGeluMul(pass, run, gate, up, activated, tokenCount * manifest.feedForwardLength);
    this.releaseScratch(gate, up);
    const ffnOutput = await this.dispatchMatMulVisionWeight(pass, run, `v.blk.${layer}.ffn_down.weight`, activated, tokenCount);
    this.releaseScratch(activated);
    const ffnNorm = this.dispatchRowRmsNorm(pass, run, ffnOutput, `v.blk.${layer}.ffn_post_norm`, tokenCount, manifest.embeddingLength, await this.f32Handle(`v.blk.${layer}.ffn_post_norm.weight`), this.session.epsilon);
    this.releaseScratch(ffnOutput);
    const output = this.scratchF32(`v.blk.${layer}.output`, tokenCount * manifest.embeddingLength, run.cleanup);
    dispatchResidualAdd(this.arena.device, pass, run.resources, attentionResidual, ffnNorm, output, tokenCount * manifest.embeddingLength);
    this.releaseScratch(attentionResidual, ffnNorm);
    return output;
  }

  private dispatchRowRmsNorm(
    pass: WebGpuComputePassLike,
    run: VisionRunBuffers,
    input: WebGpuBufferLike,
    label: string,
    rowCount: number,
    rowSize: number,
    weight: F32Handle | undefined,
    epsilon: number,
  ): WebGpuBufferLike {
    const output = this.scratchF32(`${label}.rms_norm`, rowCount * rowSize, run.cleanup);
    const resource = createVisionRmsNormResources(this.arena.device, input, weight?.buffer, output, {
      rowCount,
      rowSize,
      epsilon,
    });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(rowCount);
    return output;
  }

  private dispatchRope2d(
    pass: WebGpuComputePassLike,
    run: VisionRunBuffers,
    input: WebGpuBufferLike,
    label: string,
    patchGridX: number,
    tokenCount: number,
    headSize: number,
  ): WebGpuBufferLike {
    const output = this.scratchF32(label, tokenCount * this.session.manifest.embeddingLength, run.cleanup);
    const resource = createVisionRope2dResources(this.arena.device, input, output, {
      patchGridX,
      tokenCount,
      headCount: this.session.manifest.headCount,
      headSize,
      freqBase: 100,
    });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil((tokenCount * this.session.manifest.embeddingLength) / 256));
    return output;
  }

  private dispatchAttention(
    pass: WebGpuComputePassLike,
    run: VisionRunBuffers,
    q: WebGpuBufferLike,
    k: WebGpuBufferLike,
    v: WebGpuBufferLike,
    label: string,
    tokenCount: number,
    headCount: number,
    headSize: number,
  ): WebGpuBufferLike {
    const probabilities = this.scratchF32(`${label}.probabilities`, tokenCount * headCount * tokenCount, run.cleanup);
    const score = createVisionAttentionScoreResources(this.arena.device, q, k, probabilities, {
      tokenCount,
      headCount,
      headSize,
      scale: 1,
    });
    run.resources.push(score);
    pass.setPipeline(score.pipeline);
    pass.setBindGroup(0, score.bindGroup);
    pass.dispatchWorkgroups(tokenCount, headCount);

    const output = this.scratchF32(label, tokenCount * headCount * headSize, run.cleanup);
    const apply = createVisionAttentionApplyResources(this.arena.device, v, probabilities, output, {
      tokenCount,
      headCount,
      headSize,
    });
    run.resources.push(apply);
    pass.setPipeline(apply.pipeline);
    pass.setBindGroup(0, apply.bindGroup);
    pass.dispatchWorkgroups(tokenCount, headCount, headSize);
    this.releaseScratch(probabilities);
    return output;
  }

  private dispatchAveragePool(
    pass: WebGpuComputePassLike,
    run: VisionRunBuffers,
    input: WebGpuBufferLike,
    patchGridX: number,
    patchGridY: number,
  ): WebGpuBufferLike {
    const manifest = this.session.manifest;
    const outputTokenCount = (patchGridX / manifest.spatialMergeSize) * (patchGridY / manifest.spatialMergeSize);
    const output = this.scratchF32("vision.pool", outputTokenCount * manifest.embeddingLength, run.cleanup);
    const resource = createVisionAveragePoolResources(this.arena.device, input, output, {
      patchGridX,
      patchGridY,
      embeddingLength: manifest.embeddingLength,
      kernelSize: manifest.spatialMergeSize,
      outputScale: Math.sqrt(manifest.embeddingLength),
    });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil((outputTokenCount * manifest.embeddingLength) / 256));
    return output;
  }

  private async dispatchStdNormalize(
    pass: WebGpuComputePassLike,
    run: VisionRunBuffers,
    input: WebGpuBufferLike,
    tokenCount: number,
  ): Promise<WebGpuBufferLike> {
    const manifest = this.session.manifest;
    const bias = await this.f32Handle("v.std_bias");
    const scale = await this.f32Handle("v.std_scale");
    const output = this.scratchF32("vision.std_norm", tokenCount * manifest.embeddingLength, run.cleanup);
    const resource = createVisionStdNormalizeResources(this.arena.device, input, bias.buffer, scale.buffer, output, {
      length: tokenCount * manifest.embeddingLength,
      rowSize: manifest.embeddingLength,
    });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil((tokenCount * manifest.embeddingLength) / 256));
    return output;
  }

  private async dispatchMatMulVisionWeight(
    pass: WebGpuComputePassLike,
    run: VisionRunBuffers,
    weightName: string,
    input: WebGpuBufferLike,
    columnCount: number,
  ): Promise<WebGpuBufferLike> {
    const tensor = this.session.getTensor(weightName);
    const inputSize = tensor.dimensions[0] ?? 0;
    const rowCount = tensor.dimensions[1] ?? 0;
    let current = await this.dispatchClamp(pass, run, input, weightName.replace(/\.weight$/, ".input_min"), weightName.replace(/\.weight$/, ".input_max"), inputSize * columnCount);
    const output = this.scratchF32(`${weightName}.output`, rowCount * columnCount, run.cleanup);
    if (isDenseType(tensor.type)) {
      const handle = await this.f32Handle(weightName);
      dispatchF32MatMul(this.arena.device, pass, run.resources, handle.buffer, current, output, inputSize, rowCount, columnCount);
    } else {
      const handle = await this.quantizedHandle(weightName);
      if (handle.type === "Q8_0") {
        const q8 = scratchQ8_0(this.arena, inputSize, columnCount, handle.blockCount, run.cleanup, `${weightName}.q8_0`);
        dispatchQ8_0Quantize(this.arena.device, pass, run.resources, current, q8, inputSize, columnCount, handle.blockCount);
        dispatchQ8_0MatMul(pass, run.resources, handle, q8, output, columnCount);
        this.releaseScratch(q8.scale, q8.qs);
      } else {
        const q8 = scratchQ8K(this.arena, inputSize, columnCount, run.cleanup, `${weightName}.q8k`);
        dispatchQ8KQuantize(this.arena.device, pass, run.resources, current, q8, inputSize, columnCount);
        dispatchKMatMul(pass, run.resources, handle, q8, output, columnCount);
        this.releaseScratch(q8.scale, q8.qs, q8.bsums);
      }
    }
    if (current !== input) {
      this.releaseScratch(current);
    }
    current = await this.dispatchClamp(pass, run, output, weightName.replace(/\.weight$/, ".output_min"), weightName.replace(/\.weight$/, ".output_max"), rowCount * columnCount);
    if (current !== output) {
      this.releaseScratch(output);
    }
    return current;
  }

  private async dispatchClamp(
    pass: WebGpuComputePassLike,
    run: VisionRunBuffers,
    input: WebGpuBufferLike,
    minName: string,
    maxName: string,
    length: number,
  ): Promise<WebGpuBufferLike> {
    if (!this.session.hasTensor(minName) || !this.session.hasTensor(maxName)) {
      return input;
    }
    const min = (await this.session.readF32Tensor(minName))[0] ?? -Infinity;
    const max = (await this.session.readF32Tensor(maxName))[0] ?? Infinity;
    const output = this.scratchF32(`${minName}.clamp`, length, run.cleanup);
    const resource = createVisionClampResources(this.arena.device, input, output, { length, min, max });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(length / 256));
    return output;
  }

  private dispatchGeluMul(
    pass: WebGpuComputePassLike,
    run: VisionRunBuffers,
    gate: WebGpuBufferLike,
    up: WebGpuBufferLike,
    output: WebGpuBufferLike,
    length: number,
  ): void {
    const resource = createVisionGeluMulResources(this.arena.device, gate, up, output, { length });
    run.resources.push(resource);
    pass.setPipeline(resource.pipeline);
    pass.setBindGroup(0, resource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(length / 256));
  }

  private async f32Handle(name: string): Promise<F32Handle> {
    const cached = this.f32Handles.get(name);
    if (cached) {
      return cached;
    }
    const tensor = this.session.getTensor(name);
    if (!isDenseType(tensor.type)) {
      return unsupported(`${name} must be dense for WebGPU vision, got ${tensor.type}`);
    }
    const elementCount = tensor.dimensions.reduce((product, dimension) => product * dimension, 1);
    const bytes = await this.session.readWeightBytes(name);
    const values = tensor.type === "F32"
      ? new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Float32Array.BYTES_PER_ELEMENT).slice()
      : dequantizeRow(tensor.type, bytes, elementCount);
    const buffer = this.arena.createBuffer(name, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
    this.arena.device.queue.writeBuffer(buffer, 0, values);
    const handle: F32Handle = {
      length: values.length,
      byteLength: values.byteLength,
      device: this.arena.device,
      buffer,
      destroy: () => buffer.destroy?.(),
    };
    this.f32Handles.set(name, handle);
    return handle;
  }

  private async quantizedHandle(name: string): Promise<QuantizedHandle> {
    const cached = this.quantizedHandles.get(name);
    if (cached) {
      return cached;
    }
    const tensor = this.session.getTensor(name);
    let handle: QuantizedHandle;
    try {
      handle = createQuantizedHandleFromBytes(
        this.arena,
        name,
        webGpuMatMulType(tensor.type, name),
        tensor.dimensions[0] ?? 0,
        tensor.dimensions[1] ?? 0,
        await this.session.readWeightBytes(name),
      );
    } catch (error) {
      return unsupported(error instanceof Error ? error.message : String(error));
    }
    this.quantizedHandles.set(name, handle);
    return handle;
  }

  private bufferFromF32(label: string, values: Float32Array, cleanup: GpuResource[]): WebGpuBufferLike {
    const buffer = this.arena.createBuffer(label, values.byteLength, GPU_STORAGE | GPU_COPY_DST);
    this.arena.device.queue.writeBuffer(buffer, 0, values);
    cleanup.push(buffer);
    return buffer;
  }

  private scratchF32(label: string, length: number, cleanup: GpuResource[]): WebGpuBufferLike {
    const buffer = this.arena.createScratchBuffer(label, length * Float32Array.BYTES_PER_ELEMENT, GPU_STORAGE | GPU_COPY_SRC);
    cleanup.push(buffer);
    return buffer;
  }

  private releaseScratch(...buffers: WebGpuBufferLike[]): void {
    for (const buffer of buffers) {
      buffer.destroy?.();
    }
  }
}

function numberOption(options: Readonly<Record<string, unknown>> | undefined, name: string): number | undefined {
  const value = options?.[name];
  return typeof value === "number" ? value : undefined;
}

function isDenseType(type: GgmlTypeName): boolean {
  return type === "F32" || type === "F16" || type === "BF16";
}

function unsupported(message: string): never {
  throw new WebGpuVisionFallbackError(message);
}

function isFallbackError(error: unknown): boolean {
  return error instanceof WebGpuVisionFallbackError ||
    (error instanceof Error && error.message.includes("WebGPU memory cap exceeded"));
}

class WebGpuVisionFallbackError extends Error {}
