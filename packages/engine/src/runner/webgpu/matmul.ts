import { GPU_COPY_DST, GPU_COPY_SRC, GPU_MAP_READ, GPU_STORAGE, GPU_UNIFORM, GPU_SHADER_STAGE_COMPUTE } from "./gpu-constants";
import { bindBuffer, storageBuffer, storageEntry } from "./gpu-bindings";
import { assertStorageBindingFits, webGpuDevice } from "./gpu-device";
import {
  createFullAttentionApplyResources,
  createFullAttentionScoreResources,
  createKMatMulBindResources,
  createQ8_0MatMulBindResources,
  createQ8_0QuantizeResources,
  createQ8KQuantizeResources,
  createQkvConvResources,
  createSsmNormGateResources,
  createSwiGluResources,
  createTop1Resources,
} from "./kernel-resources";
import { packBytesToU32, quantizeQ8_0Columns, quantizeQ8_KColumns, webGpuQuantizedWeightLayout } from "./quantized-handles";
import { Q4_K_MATMUL_WGSL, Q5_K_MATMUL_WGSL, Q6_K_MATMUL_WGSL, Q8_0_MATMUL_WGSL } from "./shaders";
import type { WebGpuBufferLike, WebGpuComputePassLike, WebGpuDeviceLike, WebGpuQkvConvResult, WebGpuQuantizedWeightHandle, WebGpuQuantizedWeightHandleInternal, WebGpuTopToken } from "./gpu-types";

export async function matMulQ8_0WebGpu(
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  if (inputSize % 32 !== 0) {
    throw new Error(`WebGPU Q8_0 matmul input size must be divisible by 32, got ${inputSize}`);
  }
  if (inputColumns.length !== inputSize * columnCount) {
    throw new Error(`WebGPU Q8_0 matmul input shape mismatch: ${inputColumns.length}`);
  }
  const blockCount = inputSize / 32;
  const rowByteLength = blockCount * 34;
  if (weightBytes.byteLength !== rowByteLength * rowCount) {
    throw new Error(`WebGPU Q8_0 matmul weight shape mismatch: ${weightBytes.byteLength}`);
  }

  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }

  const q8 = quantizeQ8_0Columns(inputColumns, inputSize, columnCount);
  const packedWeight = packBytesToU32(weightBytes);
  await assertStorageBindingFits("Q8_0 weight", packedWeight.byteLength);
  const outputLength = rowCount * columnCount;
  const params = new Uint32Array([inputSize, rowCount, columnCount, blockCount, rowByteLength, 0, 0, 0]);

  const weightBuffer = storageBuffer(device, packedWeight.byteLength, GPU_COPY_DST);
  const inputScaleBuffer = storageBuffer(device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(device, q8.qs.byteLength, GPU_COPY_DST);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  const outputBuffer = storageBuffer(device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    device.queue.writeBuffer(weightBuffer, 0, packedWeight);
    device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const shaderModule = device.createShaderModule({ code: Q8_0_MATMUL_WGSL });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        storageEntry(0, "read-only-storage"),
        storageEntry(1, "read-only-storage"),
        storageEntry(2, "read-only-storage"),
        {
          binding: 3,
          visibility: GPU_SHADER_STAGE_COMPUTE,
          buffer: { type: "uniform" },
        },
        storageEntry(4, "storage"),
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });
    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, outputBuffer),
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    return output;
  } finally {
    weightBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    paramsBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

export async function matMulQ4_KWebGpu(
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  if (inputSize % 256 !== 0) {
    throw new Error(`WebGPU Q4_K matmul input size must be divisible by 256, got ${inputSize}`);
  }
  if (inputColumns.length !== inputSize * columnCount) {
    throw new Error(`WebGPU Q4_K matmul input shape mismatch: ${inputColumns.length}`);
  }
  const blockCount = inputSize / 256;
  const rowByteLength = blockCount * 144;
  if (weightBytes.byteLength !== rowByteLength * rowCount) {
    throw new Error(`WebGPU Q4_K matmul weight shape mismatch: ${weightBytes.byteLength}`);
  }

  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }

  const q8 = quantizeQ8_KColumns(inputColumns, inputSize, columnCount);
  const packedWeight = packBytesToU32(weightBytes);
  await assertStorageBindingFits("Q4_K weight", packedWeight.byteLength);
  const outputLength = rowCount * columnCount;
  const params = new Uint32Array([inputSize, rowCount, columnCount, blockCount, rowByteLength, 0, 0, 0]);

  const weightBuffer = storageBuffer(device, packedWeight.byteLength, GPU_COPY_DST);
  const inputScaleBuffer = storageBuffer(device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(device, q8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(device, q8.bsums.byteLength, GPU_COPY_DST);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  const outputBuffer = storageBuffer(device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    device.queue.writeBuffer(weightBuffer, 0, packedWeight);
    device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const shaderModule = device.createShaderModule({ code: Q4_K_MATMUL_WGSL });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        storageEntry(0, "read-only-storage"),
        storageEntry(1, "read-only-storage"),
        storageEntry(2, "read-only-storage"),
        storageEntry(3, "read-only-storage"),
        {
          binding: 4,
          visibility: GPU_SHADER_STAGE_COMPUTE,
          buffer: { type: "uniform" },
        },
        storageEntry(5, "storage"),
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });
    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, inputBsumsBuffer),
        bindBuffer(4, paramsBuffer),
        bindBuffer(5, outputBuffer),
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    return output;
  } finally {
    weightBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
    paramsBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

export async function matMulQ5_KWebGpu(
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  if (inputSize % 256 !== 0) {
    throw new Error(`WebGPU Q5_K matmul input size must be divisible by 256, got ${inputSize}`);
  }
  if (inputColumns.length !== inputSize * columnCount) {
    throw new Error(`WebGPU Q5_K matmul input shape mismatch: ${inputColumns.length}`);
  }
  const blockCount = inputSize / 256;
  const rowByteLength = blockCount * 176;
  if (weightBytes.byteLength !== rowByteLength * rowCount) {
    throw new Error(`WebGPU Q5_K matmul weight shape mismatch: ${weightBytes.byteLength}`);
  }

  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }

  const q8 = quantizeQ8_KColumns(inputColumns, inputSize, columnCount);
  const packedWeight = packBytesToU32(weightBytes);
  await assertStorageBindingFits("Q5_K weight", packedWeight.byteLength);
  const outputLength = rowCount * columnCount;
  const params = new Uint32Array([inputSize, rowCount, columnCount, blockCount, rowByteLength, 0, 0, 0]);

  const weightBuffer = storageBuffer(device, packedWeight.byteLength, GPU_COPY_DST);
  const inputScaleBuffer = storageBuffer(device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(device, q8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(device, q8.bsums.byteLength, GPU_COPY_DST);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  const outputBuffer = storageBuffer(device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    device.queue.writeBuffer(weightBuffer, 0, packedWeight);
    device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const shaderModule = device.createShaderModule({ code: Q5_K_MATMUL_WGSL });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        storageEntry(0, "read-only-storage"),
        storageEntry(1, "read-only-storage"),
        storageEntry(2, "read-only-storage"),
        storageEntry(3, "read-only-storage"),
        {
          binding: 4,
          visibility: GPU_SHADER_STAGE_COMPUTE,
          buffer: { type: "uniform" },
        },
        storageEntry(5, "storage"),
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });
    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, inputBsumsBuffer),
        bindBuffer(4, paramsBuffer),
        bindBuffer(5, outputBuffer),
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    return output;
  } finally {
    weightBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
    paramsBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

export async function matMulQ6_KWebGpu(
  weightBytes: Uint8Array,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  if (inputSize % 256 !== 0) {
    throw new Error(`WebGPU Q6_K matmul input size must be divisible by 256, got ${inputSize}`);
  }
  if (inputColumns.length !== inputSize * columnCount) {
    throw new Error(`WebGPU Q6_K matmul input shape mismatch: ${inputColumns.length}`);
  }
  const blockCount = inputSize / 256;
  const rowByteLength = blockCount * 210;
  if (weightBytes.byteLength !== rowByteLength * rowCount) {
    throw new Error(`WebGPU Q6_K matmul weight shape mismatch: ${weightBytes.byteLength}`);
  }

  const device = await webGpuDevice();
  if (!device) {
    return undefined;
  }

  const q8 = quantizeQ8_KColumns(inputColumns, inputSize, columnCount);
  const packedWeight = packBytesToU32(weightBytes);
  await assertStorageBindingFits("Q6_K weight", packedWeight.byteLength);
  const outputLength = rowCount * columnCount;
  const params = new Uint32Array([inputSize, rowCount, columnCount, blockCount, rowByteLength, 0, 0, 0]);

  const weightBuffer = storageBuffer(device, packedWeight.byteLength, GPU_COPY_DST);
  const inputScaleBuffer = storageBuffer(device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(device, q8.qs.byteLength, GPU_COPY_DST);
  const paramsBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  const outputBuffer = storageBuffer(device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    device.queue.writeBuffer(weightBuffer, 0, packedWeight);
    device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const shaderModule = device.createShaderModule({ code: Q6_K_MATMUL_WGSL });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        storageEntry(0, "read-only-storage"),
        storageEntry(1, "read-only-storage"),
        storageEntry(2, "read-only-storage"),
        {
          binding: 3,
          visibility: GPU_SHADER_STAGE_COMPUTE,
          buffer: { type: "uniform" },
        },
        storageEntry(4, "storage"),
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });
    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, outputBuffer),
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    return output;
  } finally {
    weightBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    paramsBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

export async function matMulWebGpuQuantizedResident(
  handle: WebGpuQuantizedWeightHandle,
  inputColumns: Float32Array,
  columnCount: number,
): Promise<Float32Array> {
  const resident = handle as WebGpuQuantizedWeightHandleInternal;
  if (inputColumns.length !== resident.inputSize * columnCount) {
    throw new Error(`WebGPU ${resident.type} resident matmul input shape mismatch: ${inputColumns.length}`);
  }

  if (resident.type === "Q8_0") {
    return matMulQ8_0Resident(resident, inputColumns, columnCount);
  }
  if (resident.type === "Q6_K") {
    return matMulKResident(resident, inputColumns, columnCount, Q6_K_MATMUL_WGSL, false);
  }
  if (resident.type === "Q5_K") {
    return matMulKResident(resident, inputColumns, columnCount, Q5_K_MATMUL_WGSL, true);
  }
  return matMulKResident(resident, inputColumns, columnCount, Q4_K_MATMUL_WGSL, true);
}

export async function matMulTop1WebGpuQuantizedResident(
  handle: WebGpuQuantizedWeightHandle,
  inputColumns: Float32Array,
): Promise<WebGpuTopToken> {
  const resident = handle as WebGpuQuantizedWeightHandleInternal;
  if (inputColumns.length !== resident.inputSize) {
    throw new Error(`WebGPU ${resident.type} top-1 input shape mismatch: ${inputColumns.length}`);
  }
  if (resident.type === "Q8_0") {
    return top1Cpu(await matMulWebGpuQuantizedResident(handle, inputColumns, 1));
  }

  const q8 = quantizeQ8_KColumns(inputColumns, resident.inputSize, 1);
  const logitsBuffer = storageBuffer(resident.device, resident.rowCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const topBuffer = storageBuffer(resident.device, 2 * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = resident.device.createBuffer({
    size: 2 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const inputScaleBuffer = storageBuffer(resident.device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(resident.device, q8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(resident.device, q8.bsums.byteLength, GPU_COPY_DST);

  try {
    resident.device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    resident.device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    resident.device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);
    const matmulResources = createKMatMulBindResources(resident, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, logitsBuffer, 1);
    const topResources = createTop1Resources(resident.device, logitsBuffer, topBuffer, resident.rowCount);
    const encoder = resident.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(matmulResources.pipeline);
    pass.setBindGroup(0, matmulResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(resident.rowCount / 8), 1);
    pass.setPipeline(topResources.pipeline);
    pass.setBindGroup(0, topResources.bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(topBuffer, 0, readbackBuffer, 0, 2 * Float32Array.BYTES_PER_ELEMENT);
    resident.device.queue.submit([encoder.finish()]);
    await resident.device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const mapped = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    matmulResources.destroy();
    topResources.destroy();
    return { id: Math.trunc(mapped[0] ?? 0), value: mapped[1] ?? -Infinity };
  } finally {
    logitsBuffer.destroy?.();
    topBuffer.destroy?.();
    readbackBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
  }
}

export async function matMulSwiGluWebGpuResident(
  gateHandle: WebGpuQuantizedWeightHandle,
  upHandle: WebGpuQuantizedWeightHandle,
  inputColumns: Float32Array,
  columnCount: number,
): Promise<Float32Array> {
  const gate = gateHandle as WebGpuQuantizedWeightHandleInternal;
  const up = upHandle as WebGpuQuantizedWeightHandleInternal;
  if (gate.device !== up.device) {
    throw new Error("WebGPU SwiGLU gate/up handles must belong to the same device");
  }
  if (gate.inputSize !== up.inputSize || gate.rowCount !== up.rowCount) {
    throw new Error("WebGPU SwiGLU gate/up handle shape mismatch");
  }
  if (inputColumns.length !== gate.inputSize * columnCount) {
    throw new Error(`WebGPU SwiGLU input shape mismatch: ${inputColumns.length}`);
  }
  if (gate.type === "Q8_0" || up.type === "Q8_0") {
    throw new Error("WebGPU SwiGLU currently supports K-quant gate/up weights only");
  }

  const q8 = quantizeQ8_KColumns(inputColumns, gate.inputSize, columnCount);
  const outputLength = gate.rowCount * columnCount;
  const gateOutputBuffer = storageBuffer(gate.device, outputLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const upOutputBuffer = storageBuffer(gate.device, outputLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const swigluBuffer = storageBuffer(gate.device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = gate.device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const inputScaleBuffer = storageBuffer(gate.device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(gate.device, q8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(gate.device, q8.bsums.byteLength, GPU_COPY_DST);

  try {
    gate.device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    gate.device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    gate.device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);

    const gateResources = createKMatMulBindResources(
      gate,
      inputScaleBuffer,
      inputQsBuffer,
      inputBsumsBuffer,
      gateOutputBuffer,
      columnCount,
    );
    const upResources = createKMatMulBindResources(
      up,
      inputScaleBuffer,
      inputQsBuffer,
      inputBsumsBuffer,
      upOutputBuffer,
      columnCount,
    );
    const swigluResources = createSwiGluResources(gate.device, gateOutputBuffer, upOutputBuffer, swigluBuffer, outputLength);

    const encoder = gate.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(gateResources.pipeline);
    pass.setBindGroup(0, gateResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(gate.rowCount / 8), columnCount);
    pass.setPipeline(upResources.pipeline);
    pass.setBindGroup(0, upResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(up.rowCount / 8), columnCount);
    pass.setPipeline(swigluResources.pipeline);
    pass.setBindGroup(0, swigluResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outputLength / 256));
    pass.end();
    encoder.copyBufferToBuffer(swigluBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    gate.device.queue.submit([encoder.finish()]);
    await gate.device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    gateResources.destroy();
    upResources.destroy();
    swigluResources.destroy();
    return output;
  } finally {
    gateOutputBuffer.destroy?.();
    upOutputBuffer.destroy?.();
    swigluBuffer.destroy?.();
    readbackBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
  }
}

export async function matMulSwiGluDownWebGpuResident(
  gateHandle: WebGpuQuantizedWeightHandle,
  upHandle: WebGpuQuantizedWeightHandle,
  downHandle: WebGpuQuantizedWeightHandle,
  inputColumns: Float32Array,
  columnCount: number,
): Promise<Float32Array> {
  const gate = gateHandle as WebGpuQuantizedWeightHandleInternal;
  const up = upHandle as WebGpuQuantizedWeightHandleInternal;
  const down = downHandle as WebGpuQuantizedWeightHandleInternal;
  if (gate.device !== up.device || gate.device !== down.device) {
    throw new Error("WebGPU FFN handles must belong to the same device");
  }
  if (gate.inputSize !== up.inputSize || gate.rowCount !== up.rowCount || down.inputSize !== gate.rowCount) {
    throw new Error("WebGPU FFN handle shape mismatch");
  }
  if (inputColumns.length !== gate.inputSize * columnCount) {
    throw new Error(`WebGPU FFN input shape mismatch: ${inputColumns.length}`);
  }
  if (gate.type === "Q8_0" || up.type === "Q8_0" || down.type === "Q8_0") {
    throw new Error("WebGPU FFN fusion currently supports K-quant weights only");
  }

  const inputQ8 = quantizeQ8_KColumns(inputColumns, gate.inputSize, columnCount);
  const hiddenLength = gate.rowCount * columnCount;
  const outputLength = down.rowCount * columnCount;
  const gateOutputBuffer = storageBuffer(gate.device, hiddenLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const upOutputBuffer = storageBuffer(gate.device, hiddenLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const swigluBuffer = storageBuffer(gate.device, hiddenLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const outputBuffer = storageBuffer(gate.device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = gate.device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const inputScaleBuffer = storageBuffer(gate.device, inputQ8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(gate.device, inputQ8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(gate.device, inputQ8.bsums.byteLength, GPU_COPY_DST);
  const downScaleBuffer = storageBuffer(gate.device, columnCount * down.blockCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const downQsBuffer = storageBuffer(gate.device, hiddenLength * Int32Array.BYTES_PER_ELEMENT, 0);
  const downBsumsBuffer = storageBuffer(gate.device, columnCount * down.blockCount * 16 * Int32Array.BYTES_PER_ELEMENT, 0);

  try {
    gate.device.queue.writeBuffer(inputScaleBuffer, 0, inputQ8.scales);
    gate.device.queue.writeBuffer(inputQsBuffer, 0, inputQ8.qs);
    gate.device.queue.writeBuffer(inputBsumsBuffer, 0, inputQ8.bsums);

    const gateResources = createKMatMulBindResources(gate, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, gateOutputBuffer, columnCount);
    const upResources = createKMatMulBindResources(up, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, upOutputBuffer, columnCount);
    const swigluResources = createSwiGluResources(gate.device, gateOutputBuffer, upOutputBuffer, swigluBuffer, hiddenLength);
    const quantizeResources = createQ8KQuantizeResources(
      gate.device,
      swigluBuffer,
      downScaleBuffer,
      downQsBuffer,
      downBsumsBuffer,
      down.inputSize,
      columnCount,
      down.blockCount,
    );
    const downResources = createKMatMulBindResources(down, downScaleBuffer, downQsBuffer, downBsumsBuffer, outputBuffer, columnCount);

    const encoder = gate.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(gateResources.pipeline);
    pass.setBindGroup(0, gateResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(gate.rowCount / 8), columnCount);
    pass.setPipeline(upResources.pipeline);
    pass.setBindGroup(0, upResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(up.rowCount / 8), columnCount);
    pass.setPipeline(swigluResources.pipeline);
    pass.setBindGroup(0, swigluResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(hiddenLength / 256));
    pass.setPipeline(quantizeResources.pipeline);
    pass.setBindGroup(0, quantizeResources.bindGroup);
    pass.dispatchWorkgroups(columnCount, down.blockCount);
    pass.setPipeline(downResources.pipeline);
    pass.setBindGroup(0, downResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(down.rowCount / 8), columnCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    gate.device.queue.submit([encoder.finish()]);
    await gate.device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();

    gateResources.destroy();
    upResources.destroy();
    swigluResources.destroy();
    quantizeResources.destroy();
    downResources.destroy();
    return output;
  } finally {
    gateOutputBuffer.destroy?.();
    upOutputBuffer.destroy?.();
    swigluBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
    downScaleBuffer.destroy?.();
    downQsBuffer.destroy?.();
    downBsumsBuffer.destroy?.();
  }
}

export async function matMulSsmNormGateOutWebGpuResident(
  zHandle: WebGpuQuantizedWeightHandle,
  outHandle: WebGpuQuantizedWeightHandle,
  attnNormColumns: Float32Array,
  deltaOutput: Float32Array,
  normWeight: Float32Array,
  epsilon: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  const z = zHandle as WebGpuQuantizedWeightHandleInternal;
  const out = outHandle as WebGpuQuantizedWeightHandleInternal;
  if (z.device !== out.device) {
    throw new Error("WebGPU SSM z/out handles must belong to the same device");
  }
  if (out.type !== "Q8_0") {
    return undefined;
  }
  if (z.type === "Q8_0") {
    throw new Error("WebGPU SSM z projection currently supports K-quant weights only");
  }
  if (attnNormColumns.length !== z.inputSize * columnCount) {
    throw new Error(`WebGPU SSM z input shape mismatch: ${attnNormColumns.length}`);
  }
  if (deltaOutput.length !== z.rowCount * columnCount || normWeight.length !== z.rowCount || out.inputSize !== z.rowCount) {
    throw new Error("WebGPU SSM norm/gate/out shape mismatch");
  }

  const q8 = quantizeQ8_KColumns(attnNormColumns, z.inputSize, columnCount);
  const hiddenLength = z.rowCount * columnCount;
  const outputLength = out.rowCount * columnCount;
  const zOutputBuffer = storageBuffer(z.device, hiddenLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const deltaBuffer = storageBuffer(z.device, deltaOutput.byteLength, GPU_COPY_DST);
  const normWeightBuffer = storageBuffer(z.device, normWeight.byteLength, GPU_COPY_DST);
  const gatedBuffer = storageBuffer(z.device, hiddenLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const outputBuffer = storageBuffer(z.device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = z.device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const inputScaleBuffer = storageBuffer(z.device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(z.device, q8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(z.device, q8.bsums.byteLength, GPU_COPY_DST);
  const outScaleBuffer = storageBuffer(z.device, columnCount * out.blockCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const outQsBuffer = storageBuffer(z.device, hiddenLength * Int32Array.BYTES_PER_ELEMENT, 0);

  try {
    z.device.queue.writeBuffer(deltaBuffer, 0, deltaOutput);
    z.device.queue.writeBuffer(normWeightBuffer, 0, normWeight);
    z.device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    z.device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    z.device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);

    const zResources = createKMatMulBindResources(z, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, zOutputBuffer, columnCount);
    const normGateResources = createSsmNormGateResources(
      z.device,
      deltaBuffer,
      zOutputBuffer,
      normWeightBuffer,
      gatedBuffer,
      z.rowCount,
      columnCount,
      epsilon,
    );
    const quantizeResources = createQ8_0QuantizeResources(
      z.device,
      gatedBuffer,
      outScaleBuffer,
      outQsBuffer,
      out.inputSize,
      columnCount,
      out.blockCount,
    );
    const outResources = createQ8_0MatMulBindResources(out, outScaleBuffer, outQsBuffer, outputBuffer, columnCount);

    const encoder = z.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(zResources.pipeline);
    pass.setBindGroup(0, zResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(z.rowCount / 8), columnCount);
    pass.setPipeline(normGateResources.pipeline);
    pass.setBindGroup(0, normGateResources.bindGroup);
    pass.dispatchWorkgroups(columnCount);
    pass.setPipeline(quantizeResources.pipeline);
    pass.setBindGroup(0, quantizeResources.bindGroup);
    pass.dispatchWorkgroups(columnCount, out.blockCount);
    pass.setPipeline(outResources.pipeline);
    pass.setBindGroup(0, outResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(out.rowCount / 8), columnCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    z.device.queue.submit([encoder.finish()]);
    await z.device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();

    zResources.destroy();
    normGateResources.destroy();
    quantizeResources.destroy();
    outResources.destroy();
    return output;
  } finally {
    zOutputBuffer.destroy?.();
    deltaBuffer.destroy?.();
    normWeightBuffer.destroy?.();
    gatedBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
    outScaleBuffer.destroy?.();
    outQsBuffer.destroy?.();
  }
}

export async function matMulQkvConvWebGpuResident(
  qkvHandle: WebGpuQuantizedWeightHandle,
  inputColumns: Float32Array,
  convState: Float32Array,
  convKernel: Float32Array,
  options: {
    tokenCount: number;
    convDim: number;
    kernelSize: number;
    stateSize: number;
    groupCount: number;
    valueDim: number;
  },
): Promise<WebGpuQkvConvResult | undefined> {
  const qkv = qkvHandle as WebGpuQuantizedWeightHandleInternal;
  if (qkv.type === "Q8_0") {
    return undefined;
  }
  const { tokenCount, convDim, kernelSize, stateSize, groupCount, valueDim } = options;
  const history = kernelSize - 1;
  const keyDim = stateSize * groupCount;
  if (
    qkv.rowCount !== convDim ||
    inputColumns.length !== qkv.inputSize * tokenCount ||
    convState.length !== history * convDim ||
    convKernel.length !== kernelSize * convDim ||
    valueDim % stateSize !== 0
  ) {
    throw new Error("WebGPU qkv/conv shape mismatch");
  }

  const inputQ8 = quantizeQ8_KColumns(inputColumns, qkv.inputSize, tokenCount);
  const qkvLength = convDim * tokenCount;
  const qLength = keyDim * tokenCount;
  const kLength = keyDim * tokenCount;
  const vLength = valueDim * tokenCount;
  const stateLength = history * convDim;
  const qkvBuffer = storageBuffer(qkv.device, qkvLength * Float32Array.BYTES_PER_ELEMENT, 0);
  const qBuffer = storageBuffer(qkv.device, qLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const kBuffer = storageBuffer(qkv.device, kLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const vBuffer = storageBuffer(qkv.device, vLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const newStateBuffer = storageBuffer(qkv.device, stateLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = qkv.device.createBuffer({
    size: (qLength + kLength + vLength + stateLength) * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const inputScaleBuffer = storageBuffer(qkv.device, inputQ8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(qkv.device, inputQ8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = storageBuffer(qkv.device, inputQ8.bsums.byteLength, GPU_COPY_DST);
  const convStateBuffer = storageBuffer(qkv.device, convState.byteLength, GPU_COPY_DST);
  const convKernelBuffer = storageBuffer(qkv.device, convKernel.byteLength, GPU_COPY_DST);

  try {
    qkv.device.queue.writeBuffer(inputScaleBuffer, 0, inputQ8.scales);
    qkv.device.queue.writeBuffer(inputQsBuffer, 0, inputQ8.qs);
    qkv.device.queue.writeBuffer(inputBsumsBuffer, 0, inputQ8.bsums);
    qkv.device.queue.writeBuffer(convStateBuffer, 0, convState);
    qkv.device.queue.writeBuffer(convKernelBuffer, 0, convKernel);

    const qkvResources = createKMatMulBindResources(qkv, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, qkvBuffer, tokenCount);
    const convResources = createQkvConvResources(
      qkv.device,
      qkvBuffer,
      convStateBuffer,
      convKernelBuffer,
      qBuffer,
      kBuffer,
      vBuffer,
      newStateBuffer,
      options,
    );

    const encoder = qkv.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(qkvResources.pipeline);
    pass.setBindGroup(0, qkvResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(qkv.rowCount / 8), tokenCount);
    pass.setPipeline(convResources.pipeline);
    pass.setBindGroup(0, convResources.bindGroup);
    pass.dispatchWorkgroups(Math.max(tokenCount * groupCount, tokenCount * (valueDim / stateSize), convDim), 3);
    pass.end();
    let offset = 0;
    encoder.copyBufferToBuffer(qBuffer, 0, readbackBuffer, offset, qLength * Float32Array.BYTES_PER_ELEMENT);
    offset += qLength * Float32Array.BYTES_PER_ELEMENT;
    encoder.copyBufferToBuffer(kBuffer, 0, readbackBuffer, offset, kLength * Float32Array.BYTES_PER_ELEMENT);
    offset += kLength * Float32Array.BYTES_PER_ELEMENT;
    encoder.copyBufferToBuffer(vBuffer, 0, readbackBuffer, offset, vLength * Float32Array.BYTES_PER_ELEMENT);
    offset += vLength * Float32Array.BYTES_PER_ELEMENT;
    encoder.copyBufferToBuffer(newStateBuffer, 0, readbackBuffer, offset, stateLength * Float32Array.BYTES_PER_ELEMENT);
    qkv.device.queue.submit([encoder.finish()]);
    await qkv.device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const mapped = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    qkvResources.destroy();
    convResources.destroy();
    return {
      q: mapped.slice(0, qLength),
      k: mapped.slice(qLength, qLength + kLength),
      v: mapped.slice(qLength + kLength, qLength + kLength + vLength),
      newConvState: mapped.slice(qLength + kLength + vLength),
    };
  } finally {
    qkvBuffer.destroy?.();
    qBuffer.destroy?.();
    kBuffer.destroy?.();
    vBuffer.destroy?.();
    newStateBuffer.destroy?.();
    readbackBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
    convStateBuffer.destroy?.();
    convKernelBuffer.destroy?.();
  }
}

export async function fullAttentionDecodeOutWebGpuResident(
  outHandle: WebGpuQuantizedWeightHandle,
  query: Float32Array,
  keyCache: Float32Array,
  valueCache: Float32Array,
  gate: Float32Array,
  options: {
    headSize: number;
    queryHeadCount: number;
    keyValueHeadCount: number;
    keyValueTokenCount: number;
    contextLength: number;
    scale: number;
  },
): Promise<Float32Array | undefined> {
  const out = outHandle as WebGpuQuantizedWeightHandleInternal;
  if (out.type === "Q8_0") {
    return undefined;
  }
  const { headSize, queryHeadCount, keyValueHeadCount, keyValueTokenCount, contextLength, scale } = options;
  const hiddenSize = headSize * queryHeadCount;
  if (
    out.inputSize !== hiddenSize ||
    query.length !== hiddenSize ||
    gate.length !== hiddenSize ||
    keyCache.length < keyValueTokenCount * keyValueHeadCount * headSize ||
    valueCache.length < headSize * keyValueHeadCount * contextLength ||
    queryHeadCount % keyValueHeadCount !== 0
  ) {
    throw new Error("WebGPU full attention decode shape mismatch");
  }

  const queryBuffer = storageBuffer(out.device, query.byteLength, GPU_COPY_DST);
  const keyBuffer = storageBuffer(out.device, keyCache.byteLength, GPU_COPY_DST);
  const valueBuffer = storageBuffer(out.device, valueCache.byteLength, GPU_COPY_DST);
  const gateBuffer = storageBuffer(out.device, gate.byteLength, GPU_COPY_DST);
  const gatedBuffer = storageBuffer(out.device, hiddenSize * Float32Array.BYTES_PER_ELEMENT, 0);
  const probabilitiesBuffer = storageBuffer(
    out.device,
    queryHeadCount * keyValueTokenCount * Float32Array.BYTES_PER_ELEMENT,
    0,
  );
  const outputBuffer = storageBuffer(out.device, out.rowCount * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = out.device.createBuffer({
    size: out.rowCount * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });
  const inputScaleBuffer = storageBuffer(out.device, out.blockCount * Float32Array.BYTES_PER_ELEMENT, 0);
  const inputQsBuffer = storageBuffer(out.device, hiddenSize * Int32Array.BYTES_PER_ELEMENT, 0);
  const inputBsumsBuffer = storageBuffer(out.device, out.blockCount * 16 * Int32Array.BYTES_PER_ELEMENT, 0);

  try {
    out.device.queue.writeBuffer(queryBuffer, 0, query);
    out.device.queue.writeBuffer(keyBuffer, 0, keyCache);
    out.device.queue.writeBuffer(valueBuffer, 0, valueCache);
    out.device.queue.writeBuffer(gateBuffer, 0, gate);

    const scoreResources = createFullAttentionScoreResources(
      out.device,
      queryBuffer,
      keyBuffer,
      probabilitiesBuffer,
      options,
    );
    const applyResources = createFullAttentionApplyResources(
      out.device,
      valueBuffer,
      gateBuffer,
      probabilitiesBuffer,
      gatedBuffer,
      options,
    );
    const quantizeResources = createQ8KQuantizeResources(
      out.device,
      gatedBuffer,
      inputScaleBuffer,
      inputQsBuffer,
      inputBsumsBuffer,
      out.inputSize,
      1,
      out.blockCount,
    );
    const outResources = createKMatMulBindResources(out, inputScaleBuffer, inputQsBuffer, inputBsumsBuffer, outputBuffer, 1);

    const encoder = out.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(scoreResources.pipeline);
    pass.setBindGroup(0, scoreResources.bindGroup);
    pass.dispatchWorkgroups(queryHeadCount);
    pass.setPipeline(applyResources.pipeline);
    pass.setBindGroup(0, applyResources.bindGroup);
    pass.dispatchWorkgroups(queryHeadCount, headSize);
    pass.setPipeline(quantizeResources.pipeline);
    pass.setBindGroup(0, quantizeResources.bindGroup);
    pass.dispatchWorkgroups(1, out.blockCount);
    pass.setPipeline(outResources.pipeline);
    pass.setBindGroup(0, outResources.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(out.rowCount / 8), 1);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, out.rowCount * Float32Array.BYTES_PER_ELEMENT);
    out.device.queue.submit([encoder.finish()]);
    await out.device.queue.onSubmittedWorkDone?.();
    await readbackBuffer.mapAsync(GPU_MAP_READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    scoreResources.destroy();
    applyResources.destroy();
    quantizeResources.destroy();
    outResources.destroy();
    return output;
  } finally {
    queryBuffer.destroy?.();
    keyBuffer.destroy?.();
    valueBuffer.destroy?.();
    gateBuffer.destroy?.();
    gatedBuffer.destroy?.();
    probabilitiesBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer.destroy?.();
  }
}

async function matMulQ8_0Resident(
  handle: WebGpuQuantizedWeightHandleInternal,
  inputColumns: Float32Array,
  columnCount: number,
): Promise<Float32Array> {
  const q8 = quantizeQ8_0Columns(inputColumns, handle.inputSize, columnCount);
  const outputLength = handle.rowCount * columnCount;
  const params = new Uint32Array([
    handle.inputSize,
    handle.rowCount,
    columnCount,
    handle.blockCount,
    handle.rowByteLength,
    0,
    0,
    0,
  ]);

  const inputScaleBuffer = storageBuffer(handle.device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(handle.device, q8.qs.byteLength, GPU_COPY_DST);
  const paramsBuffer = handle.device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  const outputBuffer = storageBuffer(handle.device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = handle.device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    handle.device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    handle.device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    handle.device.queue.writeBuffer(paramsBuffer, 0, params);

    const shaderModule = handle.device.createShaderModule({ code: Q8_0_MATMUL_WGSL });
    const bindGroupLayout = handle.device.createBindGroupLayout({
      entries: [
        storageEntry(0, "read-only-storage"),
        storageEntry(1, "read-only-storage"),
        storageEntry(2, "read-only-storage"),
        {
          binding: 3,
          visibility: GPU_SHADER_STAGE_COMPUTE,
          buffer: { type: "uniform" },
        },
        storageEntry(4, "storage"),
      ],
    });
    const pipeline = handle.device.createComputePipeline({
      layout: handle.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    const bindGroup = handle.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        bindBuffer(0, handle.weightBuffer),
        bindBuffer(1, inputScaleBuffer),
        bindBuffer(2, inputQsBuffer),
        bindBuffer(3, paramsBuffer),
        bindBuffer(4, outputBuffer),
      ],
    });

    return await dispatchMatMulReadback(handle.device, pipeline, bindGroup, outputBuffer, readbackBuffer, outputLength, handle.rowCount, columnCount);
  } finally {
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    paramsBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

async function matMulKResident(
  handle: WebGpuQuantizedWeightHandleInternal,
  inputColumns: Float32Array,
  columnCount: number,
  shaderCode: string,
  usesBsums: boolean,
): Promise<Float32Array> {
  const q8 = quantizeQ8_KColumns(inputColumns, handle.inputSize, columnCount);
  const outputLength = handle.rowCount * columnCount;
  const params = new Uint32Array([
    handle.inputSize,
    handle.rowCount,
    columnCount,
    handle.blockCount,
    handle.rowByteLength,
    0,
    0,
    0,
  ]);

  const inputScaleBuffer = storageBuffer(handle.device, q8.scales.byteLength, GPU_COPY_DST);
  const inputQsBuffer = storageBuffer(handle.device, q8.qs.byteLength, GPU_COPY_DST);
  const inputBsumsBuffer = usesBsums ? storageBuffer(handle.device, q8.bsums.byteLength, GPU_COPY_DST) : undefined;
  const paramsBuffer = handle.device.createBuffer({
    size: params.byteLength,
    usage: GPU_UNIFORM | GPU_COPY_DST,
  });
  const outputBuffer = storageBuffer(handle.device, outputLength * Float32Array.BYTES_PER_ELEMENT, GPU_COPY_SRC);
  const readbackBuffer = handle.device.createBuffer({
    size: outputLength * Float32Array.BYTES_PER_ELEMENT,
    usage: GPU_MAP_READ | GPU_COPY_DST,
  });

  try {
    handle.device.queue.writeBuffer(inputScaleBuffer, 0, q8.scales);
    handle.device.queue.writeBuffer(inputQsBuffer, 0, q8.qs);
    if (inputBsumsBuffer) {
      handle.device.queue.writeBuffer(inputBsumsBuffer, 0, q8.bsums);
    }
    handle.device.queue.writeBuffer(paramsBuffer, 0, params);

    const shaderModule = handle.device.createShaderModule({ code: shaderCode });
    const bindGroupLayout = handle.device.createBindGroupLayout({
      entries: usesBsums
        ? [
            storageEntry(0, "read-only-storage"),
            storageEntry(1, "read-only-storage"),
            storageEntry(2, "read-only-storage"),
            storageEntry(3, "read-only-storage"),
            {
              binding: 4,
              visibility: GPU_SHADER_STAGE_COMPUTE,
              buffer: { type: "uniform" },
            },
            storageEntry(5, "storage"),
          ]
        : [
            storageEntry(0, "read-only-storage"),
            storageEntry(1, "read-only-storage"),
            storageEntry(2, "read-only-storage"),
            {
              binding: 3,
              visibility: GPU_SHADER_STAGE_COMPUTE,
              buffer: { type: "uniform" },
            },
            storageEntry(4, "storage"),
          ],
    });
    const pipeline = handle.device.createComputePipeline({
      layout: handle.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    const entries = usesBsums
      ? [
          bindBuffer(0, handle.weightBuffer),
          bindBuffer(1, inputScaleBuffer),
          bindBuffer(2, inputQsBuffer),
          bindBuffer(3, inputBsumsBuffer as WebGpuBufferLike),
          bindBuffer(4, paramsBuffer),
          bindBuffer(5, outputBuffer),
        ]
      : [
          bindBuffer(0, handle.weightBuffer),
          bindBuffer(1, inputScaleBuffer),
          bindBuffer(2, inputQsBuffer),
          bindBuffer(3, paramsBuffer),
          bindBuffer(4, outputBuffer),
        ];
    const bindGroup = handle.device.createBindGroup({
      layout: bindGroupLayout,
      entries,
    });

    return await dispatchMatMulReadback(handle.device, pipeline, bindGroup, outputBuffer, readbackBuffer, outputLength, handle.rowCount, columnCount);
  } finally {
    inputScaleBuffer.destroy?.();
    inputQsBuffer.destroy?.();
    inputBsumsBuffer?.destroy?.();
    paramsBuffer.destroy?.();
    outputBuffer.destroy?.();
    readbackBuffer.destroy?.();
  }
}

async function dispatchMatMulReadback(
  device: WebGpuDeviceLike,
  pipeline: unknown,
  bindGroup: unknown,
  outputBuffer: WebGpuBufferLike,
  readbackBuffer: WebGpuBufferLike,
  outputLength: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(rowCount / 8), columnCount);
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone?.();
  await readbackBuffer.mapAsync(GPU_MAP_READ);
  const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
  readbackBuffer.unmap();
  return output;
}

function top1Cpu(values: Float32Array): WebGpuTopToken {
  let id = 0;
  let value = values[0] ?? -Infinity;
  for (let index = 1; index < values.length; index += 1) {
    const candidate = values[index] ?? -Infinity;
    if (candidate > value) {
      id = index;
      value = candidate;
    }
  }
  return { id, value };
}
