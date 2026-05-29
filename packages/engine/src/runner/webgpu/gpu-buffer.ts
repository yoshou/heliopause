import type { WebGpuBufferLike, WebGpuCommandEncoderLike, WebGpuDeviceLike } from "./gpu-types";

/**
 * Typed wrapper around a raw WebGPU buffer.
 *
 * Internal code holds `GpuBuffer` (or a subclass) everywhere; the underlying
 * `GPUBuffer` is reached only through `.raw`, and only at the WebGPU boundary
 * (bind group entries, `queue.writeBuffer`, `copyBufferToBuffer`,
 * `resolveQuerySet`). This replaces the previous approach of attaching
 * `__heliopause*` expando properties to foreign buffer objects.
 */
export class GpuBuffer implements WebGpuBufferLike {
  readonly raw: WebGpuBufferLike;
  readonly byteLength: number;

  constructor(raw: WebGpuBufferLike, byteLength: number) {
    this.raw = raw;
    this.byteLength = byteLength;
  }

  mapAsync(mode: number): Promise<void> {
    return this.raw.mapAsync(mode);
  }

  getMappedRange(): ArrayBuffer {
    return this.raw.getMappedRange();
  }

  unmap(): void {
    this.raw.unmap();
  }

  destroy(): void {
    this.raw.destroy?.();
  }
}

/** Return the raw WebGPU buffer for `buffer`, unwrapping a `GpuBuffer` if needed. */
export function unwrapGpuBuffer(buffer: WebGpuBufferLike): WebGpuBufferLike {
  return buffer instanceof GpuBuffer ? buffer.raw : buffer;
}

const adaptedDevices = new WeakSet<WebGpuDeviceLike>();

/**
 * Wrap a raw WebGPU device so that:
 * - `createBuffer` hands back a `GpuBuffer` wrapper, and
 * - the buffer-accepting boundary methods (`queue.writeBuffer`,
 *   `copyBufferToBuffer`, `resolveQuerySet`) unwrap any `GpuBuffer` argument
 *   back to its raw buffer before reaching the real device.
 *
 * The wrapper is a fresh delegating object; the original device is never
 * mutated. Idempotent: re-wrapping an already-wrapped device is a no-op.
 */
export function wrapWebGpuDevice(device: WebGpuDeviceLike): WebGpuDeviceLike {
  if (adaptedDevices.has(device)) {
    return device;
  }

  const rawCreateBuffer = device.createBuffer.bind(device);
  const rawCreateCommandEncoder = device.createCommandEncoder.bind(device);
  const rawWriteBuffer = device.queue.writeBuffer.bind(device.queue);
  const rawSubmit = device.queue.submit.bind(device.queue);
  const rawOnSubmittedWorkDone = device.queue.onSubmittedWorkDone?.bind(device.queue);

  const adapted: WebGpuDeviceLike = {
    features: device.features,
    createBuffer: (descriptor) => new GpuBuffer(rawCreateBuffer(descriptor), descriptor.size),
    createShaderModule: device.createShaderModule.bind(device),
    createBindGroupLayout: device.createBindGroupLayout.bind(device),
    createPipelineLayout: device.createPipelineLayout.bind(device),
    createComputePipeline: device.createComputePipeline.bind(device),
    createBindGroup: device.createBindGroup.bind(device),
    createQuerySet: device.createQuerySet?.bind(device),
    createCommandEncoder: () => wrapCommandEncoder(rawCreateCommandEncoder()),
    queue: {
      writeBuffer: (buffer, bufferOffset, data, dataOffset, size) =>
        rawWriteBuffer(unwrapGpuBuffer(buffer), bufferOffset, data, dataOffset, size),
      submit: rawSubmit,
      onSubmittedWorkDone: rawOnSubmittedWorkDone,
    },
  };

  adaptedDevices.add(adapted);
  return adapted;
}

function wrapCommandEncoder(encoder: WebGpuCommandEncoderLike): WebGpuCommandEncoderLike {
  const rawResolveQuerySet = encoder.resolveQuerySet?.bind(encoder);
  return {
    beginComputePass: encoder.beginComputePass.bind(encoder),
    copyBufferToBuffer: (source, sourceOffset, destination, destinationOffset, size) =>
      encoder.copyBufferToBuffer(
        unwrapGpuBuffer(source),
        sourceOffset,
        unwrapGpuBuffer(destination),
        destinationOffset,
        size,
      ),
    resolveQuerySet: rawResolveQuerySet
      ? (querySet, firstQuery, queryCount, destination, destinationOffset) =>
        rawResolveQuerySet(querySet, firstQuery, queryCount, unwrapGpuBuffer(destination), destinationOffset)
      : undefined,
    finish: encoder.finish.bind(encoder),
  };
}
