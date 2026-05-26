import type { WebGpuBufferLike, WebGpuComputePassLike, WebGpuDeviceLike } from "./gpu-types";
import type { Q8_0Buffers, Q8KBuffers, QuantizedHandle } from "./arena";
import {
  createBatchedFullAttentionMaterializedApplyResources,
  createBatchedFullAttentionMaterializedScoreResources,
  createBatchedFullAttentionRollingTileApplyResources,
  createBatchedFullAttentionRollingTileFinalResources,
  createBatchedFullAttentionRollingTileInitResources,
  createBatchedFullAttentionRollingTileMergeResources,
  createBatchedFullAttentionRollingTileProbabilityResources,
  createBatchedFullKvUpdateResources,
  createBatchedFullQueryResources,
  createBatchedGegluSliceResources,
  createBatchedRmsNormQ8KQuantizeResources,
  createBatchedRmsNormResidualAddResources,
  createBatchedRmsNormResidualAddScaleResources,
  createDualQ4KMatMulBindResources,
  createElementwiseMulResources,
  createF16CastResources,
  createF32GatherRowsScaleResources,
  createF32MatMulResources,
  createFullAttentionApplyResources,
  createFullAttentionScoreResources,
  createFullKvUpdateResources,
  createFullQueryResources,
  createGegluResources,
  createGegluSliceResources,
  createGeluResources,
  createHeadRmsNormNoWeightResources,
  createHeadRmsNormResources,
  createKeyCacheRopeResources,
  createKMatMulBindResources,
  createQ8_0MatMulBindResources,
  createQuantizedGatherRowsScaleResources,
  createQ8_0QuantizeResources,
  createQ8KQuantizeResources,
  createPreparePerLayerInputsResources,
  createResidualAddResources,
  createResidualAddScaleResources,
  createRopeResources,
  createRmsNormResources,
  createRmsNormQ8KQuantizeResources,
  createRmsNormResidualAddResources,
  createRmsNormResidualAddScaleResources,
  createScaleResources,
  createSelectTop1CandidateResources,
  createSigmoidMulResources,
  createSwiGluResources,
  createTokenSliceResources,
  createTokenWriteResources,
  createTopKChunkCandidatesResources,
  createTopKMergeCandidatesResources,
  createTop1ChunkResources,
  createValueCacheWriteResources,
} from "./kernel-resources";
import { GPU_STORAGE } from "./gpu-constants";

export function dispatchBatchedRmsNormQ8KQuantize(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  q8: Q8KBuffers,
  options: {
    length: number;
    tokenCount: number;
    epsilon: number;
  },
): void {
  const resource = createBatchedRmsNormQ8KQuantizeResources(
    device,
    input,
    weight,
    q8.scale,
    q8.qs,
    q8.bsums,
    options,
  );
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(1, options.tokenCount);
}

export function dispatchBatchedFullQuery(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  qProjection: WebGpuBufferLike,
  qNorm: WebGpuBufferLike,
  freqFactors: WebGpuBufferLike,
  positions: WebGpuBufferLike,
  query: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    epsilon: number;
    freqBase: number;
    tokenCount: number;
    hasFreqFactors: boolean;
  },
): void {
  const resource = createBatchedFullQueryResources(device, qProjection, qNorm, freqFactors, positions, query, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.headCount, options.tokenCount);
}

export function dispatchBatchedFullKvUpdate(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  kProjection: WebGpuBufferLike,
  vProjection: WebGpuBufferLike,
  kNorm: WebGpuBufferLike,
  freqFactors: WebGpuBufferLike,
  positions: WebGpuBufferLike,
  keyCache: WebGpuBufferLike,
  valueCache: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    valueSize: number;
    ropeDims: number;
    epsilon: number;
    freqBase: number;
    tokenCount: number;
    contextLength: number;
    hasFreqFactors: boolean;
  },
): void {
  const resource = createBatchedFullKvUpdateResources(
    device,
    kProjection,
    vProjection,
    kNorm,
    freqFactors,
    positions,
    keyCache,
    valueCache,
    options,
  );
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.headCount, options.tokenCount);
}

export function dispatchBatchedFullAttentionRollingTile(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  query: WebGpuBufferLike,
  key: WebGpuBufferLike,
  value: WebGpuBufferLike,
  positions: WebGpuBufferLike,
  probabilityTile: WebGpuBufferLike,
  rowMax: WebGpuBufferLike,
  rowSum: WebGpuBufferLike,
  tileMax: WebGpuBufferLike,
  tileSum: WebGpuBufferLike,
  oldScale: WebGpuBufferLike,
  tileScale: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    headSize: number;
    valueSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    slidingWindow?: number;
    tokenCount: number;
    scale: number;
    causal: boolean;
    tileSize?: number;
  },
): void {
  const tileSize = options.tileSize ?? 512;
  const init = createBatchedFullAttentionRollingTileInitResources(device, output, {
    valueSize: options.valueSize,
    queryHeadCount: options.queryHeadCount,
    tokenCount: options.tokenCount,
  });
  resources.push(init);
  pass.setPipeline(init.pipeline);
  pass.setBindGroup(0, init.bindGroup);
  pass.dispatchWorkgroups(options.queryHeadCount, options.valueSize, options.tokenCount);

  const keyIterationLimit = Math.max(1, options.keyValueTokenCount);
  for (let tileStart = 0; tileStart < keyIterationLimit; tileStart += tileSize) {
    const tileEnd = Math.min(options.keyValueTokenCount, tileStart + tileSize);
    const probability = createBatchedFullAttentionRollingTileProbabilityResources(
      device,
      query,
      key,
      positions,
      probabilityTile,
      tileMax,
      tileSum,
      { ...options, tileSize, tileStart, tileLength: tileEnd - tileStart },
    );
    resources.push(probability);
    pass.setPipeline(probability.pipeline);
    pass.setBindGroup(0, probability.bindGroup);
    pass.dispatchWorkgroups(options.queryHeadCount, options.tokenCount);

    const merge = createBatchedFullAttentionRollingTileMergeResources(device, rowMax, rowSum, tileMax, tileSum, oldScale, tileScale, {
      queryHeadCount: options.queryHeadCount,
      tokenCount: options.tokenCount,
      firstTile: tileStart === 0,
    });
    resources.push(merge);
    pass.setPipeline(merge.pipeline);
    pass.setBindGroup(0, merge.bindGroup);
    pass.dispatchWorkgroups(options.queryHeadCount, options.tokenCount);

    const apply = createBatchedFullAttentionRollingTileApplyResources(device, value, probabilityTile, oldScale, tileScale, output, {
      valueSize: options.valueSize,
      queryHeadCount: options.queryHeadCount,
      keyValueHeadCount: options.keyValueHeadCount,
      contextLength: options.contextLength,
      tileSize,
      tileStart,
      tileLength: tileEnd - tileStart,
      tokenCount: options.tokenCount,
    });
    resources.push(apply);
    pass.setPipeline(apply.pipeline);
    pass.setBindGroup(0, apply.bindGroup);
    pass.dispatchWorkgroups(options.queryHeadCount, options.valueSize, options.tokenCount);
  }

  const final = createBatchedFullAttentionRollingTileFinalResources(device, rowSum, output, {
    valueSize: options.valueSize,
    queryHeadCount: options.queryHeadCount,
    tokenCount: options.tokenCount,
  });
  resources.push(final);
  pass.setPipeline(final.pipeline);
  pass.setBindGroup(0, final.bindGroup);
  pass.dispatchWorkgroups(options.queryHeadCount, options.valueSize, options.tokenCount);
}

export function dispatchBatchedFullAttentionMaterializedScore(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  query: WebGpuBufferLike,
  key: WebGpuBufferLike,
  positions: WebGpuBufferLike,
  probabilities: WebGpuBufferLike,
  options: {
    headSize: number;
    valueSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    probabilityTokenCapacity: number;
    slidingWindow?: number;
    tokenCount: number;
    scale: number;
    causal: boolean;
  },
): void {
  const resource = createBatchedFullAttentionMaterializedScoreResources(device, query, key, positions, probabilities, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.queryHeadCount, options.tokenCount);
}

export function dispatchBatchedFullAttentionMaterializedApply(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  value: WebGpuBufferLike,
  probabilities: WebGpuBufferLike,
  positions: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    valueSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    probabilityTokenCapacity: number;
    slidingWindow?: number;
    tokenCount: number;
    scale: number;
    causal: boolean;
  },
): void {
  const resource = createBatchedFullAttentionMaterializedApplyResources(device, value, probabilities, positions, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.queryHeadCount, options.valueSize, options.tokenCount);
}

export function dispatchBatchedRmsNormResidualAdd(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  residual: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    length: number;
    tokenCount: number;
    epsilon: number;
  },
): void {
  const resource = createBatchedRmsNormResidualAddResources(device, input, weight, residual, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(1, options.tokenCount);
}

export function dispatchBatchedRmsNormResidualAddScale(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  residual: WebGpuBufferLike,
  scale: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    length: number;
    tokenCount: number;
    epsilon: number;
  },
): void {
  const resource = createBatchedRmsNormResidualAddScaleResources(device, input, weight, residual, scale, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(1, options.tokenCount);
}

export function dispatchBatchedGegluSlice(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  gate: WebGpuBufferLike,
  right: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    length: number;
    tokenCount: number;
    rightOffset: number;
    rightStride: number;
  },
): void {
  const resource = createBatchedGegluSliceResources(device, gate, right, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(options.length / 256), options.tokenCount);
}

export function dispatchKMatMul(
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  handle: QuantizedHandle,
  q8: Q8KBuffers,
  output: WebGpuBufferLike,
  columnCount: number,
): void {
  if (handle.type === "Q8_0") {
    throw new Error("Q8_0 handle cannot use K-quant matmul dispatch");
  }
  const resource = createKMatMulBindResources(handle, q8.scale, q8.qs, q8.bsums, output, columnCount);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(handle.rowCount / 8), columnCount);
}

export function dispatchDualQ4KMatMul(
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  leftHandle: QuantizedHandle,
  rightHandle: QuantizedHandle,
  q8: Q8KBuffers,
  leftOutput: WebGpuBufferLike,
  rightOutput: WebGpuBufferLike,
  columnCount: number,
): boolean {
  if (
    leftHandle.type !== "Q4_K" ||
    rightHandle.type !== "Q4_K" ||
    leftHandle.device !== rightHandle.device ||
    leftHandle.inputSize !== rightHandle.inputSize ||
    leftHandle.rowCount !== rightHandle.rowCount ||
    leftHandle.blockCount !== rightHandle.blockCount ||
    leftHandle.rowByteLength !== rightHandle.rowByteLength
  ) {
    return false;
  }
  const resource = createDualQ4KMatMulBindResources(
    leftHandle,
    rightHandle,
    q8.scale,
    q8.qs,
    q8.bsums,
    leftOutput,
    rightOutput,
    columnCount,
  );
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(leftHandle.rowCount / 8), columnCount, 2);
  return true;
}

export function dispatchQ8_0MatMul(
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  handle: QuantizedHandle,
  q8: Q8_0Buffers,
  output: WebGpuBufferLike,
  columnCount: number,
): void {
  if (handle.type !== "Q8_0") {
    throw new Error("K-quant handle cannot use Q8_0 matmul dispatch");
  }
  const resource = createQ8_0MatMulBindResources(handle, q8.scale, q8.qs, output, columnCount);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(handle.rowCount / 8), columnCount);
}

export function dispatchF32MatMul(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  weight: WebGpuBufferLike,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): void {
  const resource = createF32MatMulResources(device, weight, input, output, inputSize, rowCount, columnCount);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
}

export function dispatchQ8KQuantize(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  q8: Q8KBuffers,
  inputSize: number,
  columnCount: number,
): void {
  const resource = createQ8KQuantizeResources(device, input, q8.scale, q8.qs, q8.bsums, inputSize, columnCount, inputSize / 256);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(columnCount, inputSize / 256);
}

export function dispatchQ8_0Quantize(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  q8: Q8_0Buffers,
  inputSize: number,
  columnCount: number,
  blockCount: number,
): void {
  const resource = createQ8_0QuantizeResources(device, input, q8.scale, q8.qs, inputSize, columnCount, blockCount);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(columnCount, blockCount);
}

export function dispatchRmsNorm(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
  epsilon: number,
): void {
  const resource = createRmsNormResources(device, input, weight, output, length, epsilon);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(1);
}

export function dispatchRmsNormQ8KQuantize(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  q8: Q8KBuffers,
  length: number,
  epsilon: number,
): void {
  const resource = createRmsNormQ8KQuantizeResources(device, input, weight, q8.scale, q8.qs, q8.bsums, length, epsilon);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(1);
}

export function dispatchResidualAdd(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  left: WebGpuBufferLike,
  right: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
): void {
  const resource = createResidualAddResources(device, left, right, output, length);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(length / 256));
}

export function dispatchResidualAddScale(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  left: WebGpuBufferLike,
  right: WebGpuBufferLike,
  scale: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
): void {
  const resource = createResidualAddScaleResources(device, left, right, scale, output, length);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(length / 256));
}

export function dispatchRmsNormResidualAdd(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  residual: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
  epsilon: number,
): void {
  const resource = createRmsNormResidualAddResources(device, input, weight, residual, output, length, epsilon);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(1);
}

export function dispatchRmsNormResidualAddScale(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  residual: WebGpuBufferLike,
  scale: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
  epsilon: number,
): void {
  const resource = createRmsNormResidualAddScaleResources(device, input, weight, residual, scale, output, length, epsilon);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(1);
}

export function dispatchHeadRmsNorm(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  weight: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    epsilon: number;
  },
): void {
  const resource = createHeadRmsNormResources(device, input, weight, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.headCount);
}

export function dispatchHeadRmsNormNoWeight(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    epsilon: number;
  },
): void {
  const resource = createHeadRmsNormNoWeightResources(device, input, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.headCount);
}

export function dispatchRope(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  freqFactors: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    freqBase: number;
    position: number;
    hasFreqFactors: boolean;
  },
): void {
  const resource = createRopeResources(device, input, freqFactors, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.headCount);
}

export function dispatchKeyCacheRope(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  freqFactors: WebGpuBufferLike,
  keyCache: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    freqBase: number;
    position: number;
    tokenPosition: number;
    hasFreqFactors: boolean;
  },
): void {
  const resource = createKeyCacheRopeResources(device, input, freqFactors, keyCache, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.headCount);
}

export function dispatchValueCacheWrite(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  valueCache: WebGpuBufferLike,
  options: {
    headCount: number;
    valueSize: number;
    tokenPosition: number;
    contextLength: number;
  },
): void {
  const resource = createValueCacheWriteResources(device, input, valueCache, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.headCount);
}

export function dispatchFullQuery(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  qProjection: WebGpuBufferLike,
  qNorm: WebGpuBufferLike,
  freqFactors: WebGpuBufferLike,
  query: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    ropeDims: number;
    epsilon: number;
    freqBase: number;
    position: number;
    hasFreqFactors: boolean;
  },
): void {
  const resource = createFullQueryResources(device, qProjection, qNorm, freqFactors, query, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.headCount);
}

export function dispatchFullKvUpdate(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  kProjection: WebGpuBufferLike,
  vProjection: WebGpuBufferLike,
  kNorm: WebGpuBufferLike,
  freqFactors: WebGpuBufferLike,
  keyCache: WebGpuBufferLike,
  valueCache: WebGpuBufferLike,
  options: {
    headCount: number;
    headSize: number;
    valueSize: number;
    ropeDims: number;
    epsilon: number;
    freqBase: number;
    position: number;
    tokenPosition: number;
    contextLength: number;
    hasFreqFactors: boolean;
  },
): void {
  const resource = createFullKvUpdateResources(device, kProjection, vProjection, kNorm, freqFactors, keyCache, valueCache, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.headCount);
}

export function dispatchTopK(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  logits: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowCount: number;
    rowOffset: number;
    topK: number;
    candidateOffset: number;
  },
): void {
  if (options.topK < 1 || options.topK > TOPK_MAX) {
    throw new Error(`dispatchTopK supports topK 1..${TOPK_MAX}, got ${options.topK}`);
  }
  const chunkCount = Math.ceil(options.rowCount / TOPK_CHUNK_SIZE);
  const candidateCount = chunkCount * options.topK;
  const candidates = device.createBuffer({
    size: candidateCount * 2 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_STORAGE,
  });
  resources.push({ destroy: () => candidates.destroy?.() });

  const chunkResources = createTopKChunkCandidatesResources(device, logits, candidates, options);
  resources.push(chunkResources);
  pass.setPipeline(chunkResources.pipeline);
  pass.setBindGroup(0, chunkResources.bindGroup);
  pass.dispatchWorkgroups(chunkCount);

  const mergeResources = createTopKMergeCandidatesResources(device, candidates, output, {
    candidateCount,
    topK: options.topK,
    candidateOffset: options.candidateOffset,
  });
  resources.push(mergeResources);
  pass.setPipeline(mergeResources.pipeline);
  pass.setBindGroup(0, mergeResources.bindGroup);
  pass.dispatchWorkgroups(1);
}

export function dispatchTop1Chunks(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  logits: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowCount: number;
    rowOffset: number;
    candidateOffset: number;
  },
): number {
  const groupCount = Math.ceil(options.rowCount / TOP1_CHUNK_SIZE);
  const resource = createTop1ChunkResources(device, logits, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(groupCount);
  return groupCount;
}

export function dispatchSelectTop1Candidate(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  candidates: WebGpuBufferLike,
  selectedToken: WebGpuBufferLike,
  candidateCount: number,
): void {
  const resource = createSelectTop1CandidateResources(device, candidates, selectedToken, candidateCount);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(1);
}

export const TOP1_CHUNK_SIZE = 256;
export const TOPK_CHUNK_SIZE = 256;
export const TOPK_MAX = 64;

export function dispatchTokenSlice(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowSize: number;
    rowIndex: number;
  },
): void {
  const resource = createTokenSliceResources(device, input, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(options.rowSize / 256));
}

export function dispatchTokenWrite(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowSize: number;
    rowIndex: number;
  },
): void {
  const resource = createTokenWriteResources(device, input, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(options.rowSize / 256));
}

export function dispatchF32GatherRowsScale(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  rows: WebGpuBufferLike,
  tokenIds: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowSize: number;
    tokenCount: number;
    scale: number;
    outputTokenOffset?: number;
  },
): void {
  const resource = createF32GatherRowsScaleResources(device, rows, tokenIds, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(options.rowSize / 256), options.tokenCount);
}

export function dispatchQ8_0GatherRowsScale(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  weight: WebGpuBufferLike,
  tokenIds: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowSize: number;
    tokenCount: number;
    blockCount: number;
    rowByteLength: number;
    scale: number;
    outputTokenOffset?: number;
  },
): void {
  const resource = createQuantizedGatherRowsScaleResources(device, "Q8_0", weight, tokenIds, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(options.rowSize / 256), options.tokenCount);
}

export function dispatchQuantizedGatherRowsScale(
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  handle: QuantizedHandle,
  tokenIds: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    rowSize: number;
    tokenCount: number;
    scale: number;
    outputTokenOffset?: number;
  },
): void {
  const resource = createQuantizedGatherRowsScaleResources(
    handle.device,
    handle.type,
    handle.weightBuffer,
    tokenIds,
    output,
    {
      ...options,
      blockCount: handle.blockCount,
      rowByteLength: handle.rowByteLength,
    },
  );
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(options.rowSize / 256), options.tokenCount);
}

export function dispatchPreparePerLayerInputs(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  tokenRows: WebGpuBufferLike,
  projected: WebGpuBufferLike,
  normWeight: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    perLayerLength: number;
    totalPerLayerLength: number;
    tokenCount: number;
    blockCount: number;
    projectionScale: number;
    epsilon: number;
  },
): void {
  const resource = createPreparePerLayerInputsResources(device, tokenRows, projected, normWeight, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(options.perLayerLength / 256), options.tokenCount, options.blockCount);
}

export function dispatchSwiGlu(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  gate: WebGpuBufferLike,
  up: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
): void {
  const resource = createSwiGluResources(device, gate, up, output, length);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(length / 256));
}

export function dispatchGeglu(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  gate: WebGpuBufferLike,
  up: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
): void {
  const resource = createGegluResources(device, gate, up, output, length);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(length / 256));
}

export function dispatchGegluSlice(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  gate: WebGpuBufferLike,
  right: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
  rightOffset: number,
): void {
  const resource = createGegluSliceResources(device, gate, right, output, length, rightOffset);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(length / 256));
}

export function dispatchGelu(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
): void {
  const resource = createGeluResources(device, input, output, length);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(length / 256));
}

export function dispatchElementwiseMul(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  left: WebGpuBufferLike,
  right: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
): void {
  const resource = createElementwiseMulResources(device, left, right, output, length);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(length / 256));
}

export function dispatchSigmoidMul(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  left: WebGpuBufferLike,
  gate: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
): void {
  const resource = createSigmoidMulResources(device, left, gate, output, length);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(length / 256));
}

export function dispatchScale(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  scale: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
): void {
  const resource = createScaleResources(device, input, scale, output, length);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(length / 256));
}

export function dispatchF16Cast(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  input: WebGpuBufferLike,
  output: WebGpuBufferLike,
  length: number,
): void {
  const resource = createF16CastResources(device, input, output, length);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(length / 256));
}

export function dispatchFullAttentionScore(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  query: WebGpuBufferLike,
  key: WebGpuBufferLike,
  probabilities: WebGpuBufferLike,
  options: {
    headSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    scale: number;
    tokenPosition: number;
    slidingWindow?: number;
  },
): void {
  const resource = createFullAttentionScoreResources(device, query, key, probabilities, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.queryHeadCount);
}

export function dispatchFullAttentionApply(
  device: WebGpuDeviceLike,
  pass: WebGpuComputePassLike,
  resources: Array<{ destroy: () => void }>,
  value: WebGpuBufferLike,
  probabilities: WebGpuBufferLike,
  output: WebGpuBufferLike,
  options: {
    valueSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    scale: number;
    keyValueStart?: number;
  },
): void {
  const resource = createFullAttentionApplyResources(device, value, probabilities, output, options);
  resources.push(resource);
  pass.setPipeline(resource.pipeline);
  pass.setBindGroup(0, resource.bindGroup);
  pass.dispatchWorkgroups(options.queryHeadCount, options.valueSize);
}
