import type { WebGpuBufferLike, WebGpuDeviceLike } from "./gpu-types";
import { GpuBuffer, unwrapGpuBuffer } from "./gpu-buffer";
import { GPU_COPY_DST, GPU_MAP_READ, GPU_UNIFORM } from "./gpu-constants";

export type WebGpuRuntimeResourceStats = {
  shaderModuleHits: number;
  shaderModuleMisses: number;
  bindGroupLayoutHits: number;
  bindGroupLayoutMisses: number;
  pipelineLayoutHits: number;
  pipelineLayoutMisses: number;
  computePipelineHits: number;
  computePipelineMisses: number;
  bindGroupHits: number;
  bindGroupMisses: number;
  shaderModuleCreateMs: number;
  bindGroupLayoutCreateMs: number;
  pipelineLayoutCreateMs: number;
  computePipelineCreateMs: number;
  bindGroupCreates: number;
  bindGroupCreateMs: number;
  bufferCreates: number;
  bufferCreateMs: number;
  bufferAllocationsByLabel: string;
  bufferAllocationLabelCounts: Record<string, number>;
};

export type WebGpuRuntimeResourceCache = {
  stats: () => WebGpuRuntimeResourceStats;
};

export type WebGpuRuntimeResources = {
  device: WebGpuDeviceLike;
  cache: WebGpuRuntimeResourceCache;
};

type RuntimeResourceCacheState = {
  objectIds: WeakMap<object, number>;
  nextObjectId: number;
  shaderModules: Map<string, unknown>;
  bindGroupLayouts: Map<string, unknown>;
  pipelineLayouts: Map<string, unknown>;
  computePipelines: Map<string, unknown>;
  uniformBufferPools: Map<string, PooledUniformGpuBuffer[]>;
  readbackBufferPools: Map<string, PooledReadbackGpuBuffer[]>;
  statsValue: WebGpuRuntimeResourceStats;
  bufferAllocationLabels: Map<string, number>;
  trackBufferAllocations: boolean;
};

/**
 * Exact-size uniform buffer that returns itself to the pool on `destroy()`
 * instead of releasing the GPU buffer. Pooled uniform buffers are retained for
 * reuse (see the eviction TODO below).
 */
class PooledUniformGpuBuffer extends GpuBuffer {
  readonly poolKey: string;
  #inPool = false;
  #destroyed = false;
  #pools: Map<string, PooledUniformGpuBuffer[]>;

  constructor(
    raw: WebGpuBufferLike,
    byteLength: number,
    poolKey: string,
    pools: Map<string, PooledUniformGpuBuffer[]>,
  ) {
    super(raw, byteLength);
    this.poolKey = poolKey;
    this.#pools = pools;
  }

  markReused(): void {
    this.#inPool = false;
  }

  override destroy(): void {
    if (this.#destroyed || this.#inPool || !this.raw.destroy) {
      return;
    }
    this.#inPool = true;
    pushPooled(this.#pools, this.poolKey, this);
  }
}

/**
 * Exact-size readback buffer. Tracks map state so that `destroy()` releases the
 * GPU buffer while a mapping is pending/active and otherwise returns it to the
 * pool for reuse.
 */
class PooledReadbackGpuBuffer extends GpuBuffer {
  readonly poolKey: string;
  #inPool = false;
  #destroyed = false;
  #mapPending = false;
  #mapped = false;
  #pools: Map<string, PooledReadbackGpuBuffer[]>;

  constructor(
    raw: WebGpuBufferLike,
    byteLength: number,
    poolKey: string,
    pools: Map<string, PooledReadbackGpuBuffer[]>,
  ) {
    super(raw, byteLength);
    this.poolKey = poolKey;
    this.#pools = pools;
  }

  markReused(): void {
    this.#inPool = false;
  }

  override async mapAsync(mode: number): Promise<void> {
    this.#mapPending = true;
    try {
      await this.raw.mapAsync(mode);
      this.#mapped = true;
    } finally {
      this.#mapPending = false;
    }
  }

  override unmap(): void {
    try {
      this.raw.unmap();
    } finally {
      this.#mapped = false;
    }
  }

  override destroy(): void {
    if (this.#destroyed || this.#inPool) {
      return;
    }
    if (this.#mapPending || this.#mapped) {
      this.#destroyed = true;
      this.raw.destroy?.();
      return;
    }
    this.#inPool = true;
    pushPooled(this.#pools, this.poolKey, this);
  }
}

function pushPooled<T>(pools: Map<string, T[]>, key: string, buffer: T): void {
  const pool = pools.get(key);
  if (pool) {
    pool.push(buffer);
  } else {
    pools.set(key, [buffer]);
  }
}

const installedResources = new WeakMap<WebGpuDeviceLike, Map<string, WebGpuRuntimeResources>>();

const emptyStats = (): WebGpuRuntimeResourceStats => ({
  shaderModuleHits: 0,
  shaderModuleMisses: 0,
  bindGroupLayoutHits: 0,
  bindGroupLayoutMisses: 0,
  pipelineLayoutHits: 0,
  pipelineLayoutMisses: 0,
  computePipelineHits: 0,
  computePipelineMisses: 0,
  bindGroupHits: 0,
  bindGroupMisses: 0,
  shaderModuleCreateMs: 0,
  bindGroupLayoutCreateMs: 0,
  pipelineLayoutCreateMs: 0,
  computePipelineCreateMs: 0,
  bindGroupCreates: 0,
  bindGroupCreateMs: 0,
  bufferCreates: 0,
  bufferCreateMs: 0,
  bufferAllocationsByLabel: "",
  bufferAllocationLabelCounts: {},
});

/**
 * Layer shader/pipeline caching and exact-size buffer pooling onto `device`.
 *
 * Returns a fresh delegating device (the input device is never mutated) plus a
 * cache handle exposing accumulated stats. Idempotent per input device.
 */
export function installWebGpuRuntimeResourceCache(
  device: WebGpuDeviceLike,
  options: { trackBufferAllocations?: boolean } = {},
): WebGpuRuntimeResources {
  let resourcesByMode = installedResources.get(device);
  if (!resourcesByMode) {
    resourcesByMode = new Map();
    installedResources.set(device, resourcesByMode);
  }
  const cacheKey = options.trackBufferAllocations === true ? "tracked" : "untracked";
  const existing = resourcesByMode.get(cacheKey);
  if (existing) {
    return existing;
  }

  const state: RuntimeResourceCacheState = {
    objectIds: new WeakMap<object, number>(),
    nextObjectId: 1,
    shaderModules: new Map<string, unknown>(),
    bindGroupLayouts: new Map<string, unknown>(),
    pipelineLayouts: new Map<string, unknown>(),
    computePipelines: new Map<string, unknown>(),
    // TODO: Add bounded flush/eviction for these exact-size buffer pools once the cache policy is redesigned.
    uniformBufferPools: new Map<string, PooledUniformGpuBuffer[]>(),
    readbackBufferPools: new Map<string, PooledReadbackGpuBuffer[]>(),
    bufferAllocationLabels: new Map<string, number>(),
    trackBufferAllocations: options.trackBufferAllocations === true,
    statsValue: emptyStats(),
  };

  const createBuffer = device.createBuffer.bind(device);
  const createShaderModule = device.createShaderModule.bind(device);
  const createBindGroupLayout = device.createBindGroupLayout.bind(device);
  const createPipelineLayout = device.createPipelineLayout.bind(device);
  const createComputePipeline = device.createComputePipeline.bind(device);
  const createBindGroup = device.createBindGroup.bind(device);

  const wrapped: WebGpuDeviceLike = {
    features: device.features,
    createQuerySet: device.createQuerySet?.bind(device),
    createCommandEncoder: device.createCommandEncoder.bind(device),
    queue: device.queue,

    createBuffer: (descriptor) => {
      const uniformPoolKey = uniformBufferPoolKey(descriptor);
      const readbackPoolKey = readbackBufferPoolKey(descriptor);
      if (uniformPoolKey) {
        const pooled = state.uniformBufferPools.get(uniformPoolKey)?.pop();
        if (pooled) {
          pooled.markReused();
          return pooled;
        }
      }
      if (readbackPoolKey) {
        const pooled = state.readbackBufferPools.get(readbackPoolKey)?.pop();
        if (pooled) {
          pooled.markReused();
          return pooled;
        }
      }
      const start = nowMs();
      try {
        const created = createBuffer(descriptor);
        if (uniformPoolKey) {
          return new PooledUniformGpuBuffer(unwrapGpuBuffer(created), descriptor.size, uniformPoolKey, state.uniformBufferPools);
        }
        if (readbackPoolKey) {
          return new PooledReadbackGpuBuffer(unwrapGpuBuffer(created), descriptor.size, readbackPoolKey, state.readbackBufferPools);
        }
        return created;
      } finally {
        state.statsValue.bufferCreates += 1;
        state.statsValue.bufferCreateMs += nowMs() - start;
        recordBufferAllocationLabel(descriptor, state);
      }
    },

    createShaderModule: (descriptor) => {
      const cached = state.shaderModules.get(descriptor.code);
      if (cached) {
        state.statsValue.shaderModuleHits += 1;
        return cached;
      }
      state.statsValue.shaderModuleMisses += 1;
      const start = nowMs();
      const created = createShaderModule(descriptor);
      state.statsValue.shaderModuleCreateMs += nowMs() - start;
      state.shaderModules.set(descriptor.code, created);
      return created;
    },

    createBindGroupLayout: (descriptor) => {
      const key = stableKey(descriptor, state);
      const cached = state.bindGroupLayouts.get(key);
      if (cached) {
        state.statsValue.bindGroupLayoutHits += 1;
        return cached;
      }
      state.statsValue.bindGroupLayoutMisses += 1;
      const start = nowMs();
      const created = createBindGroupLayout(descriptor);
      state.statsValue.bindGroupLayoutCreateMs += nowMs() - start;
      state.bindGroupLayouts.set(key, created);
      return created;
    },

    createPipelineLayout: (descriptor) => {
      const key = stableKey(descriptor, state);
      const cached = state.pipelineLayouts.get(key);
      if (cached) {
        state.statsValue.pipelineLayoutHits += 1;
        return cached;
      }
      state.statsValue.pipelineLayoutMisses += 1;
      const start = nowMs();
      const created = createPipelineLayout(descriptor);
      state.statsValue.pipelineLayoutCreateMs += nowMs() - start;
      state.pipelineLayouts.set(key, created);
      return created;
    },

    createComputePipeline: (descriptor) => {
      const key = stableKey(descriptor, state);
      const cached = state.computePipelines.get(key);
      if (cached) {
        state.statsValue.computePipelineHits += 1;
        return cached;
      }
      state.statsValue.computePipelineMisses += 1;
      const start = nowMs();
      const created = createComputePipeline(descriptor);
      state.statsValue.computePipelineCreateMs += nowMs() - start;
      state.computePipelines.set(key, created);
      return created;
    },

    createBindGroup: (descriptor) => {
      state.statsValue.bindGroupMisses += 1;
      const start = nowMs();
      try {
        return createBindGroup(descriptor);
      } finally {
        state.statsValue.bindGroupCreates += 1;
        state.statsValue.bindGroupCreateMs += nowMs() - start;
      }
    },
  };

  const cache: WebGpuRuntimeResourceCache = {
    stats() {
      const bufferAllocationLabelCounts = Object.fromEntries(state.bufferAllocationLabels);
      return {
        ...state.statsValue,
        bufferAllocationsByLabel: formatBufferAllocationLabels(bufferAllocationLabelCounts),
        bufferAllocationLabelCounts,
      };
    },
  };

  const resources: WebGpuRuntimeResources = { device: wrapped, cache };
  resourcesByMode.set(cacheKey, resources);
  return resources;
}

function uniformBufferPoolKey(descriptor: {
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}): string | undefined {
  if (descriptor.mappedAtCreation) {
    return undefined;
  }
  const isUniformCopyDst = (descriptor.usage & GPU_UNIFORM) !== 0 && (descriptor.usage & GPU_COPY_DST) !== 0;
  if (!isUniformCopyDst || descriptor.size > 256) {
    return undefined;
  }
  return `${descriptor.usage}:${descriptor.size}`;
}

function readbackBufferPoolKey(descriptor: {
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}): string | undefined {
  if (descriptor.mappedAtCreation || descriptor.usage !== (GPU_MAP_READ | GPU_COPY_DST)) {
    return undefined;
  }
  return `${descriptor.usage}:${descriptor.size}`;
}

export function diffWebGpuRuntimeResourceStats(
  after: WebGpuRuntimeResourceStats,
  before: WebGpuRuntimeResourceStats,
): WebGpuRuntimeResourceStats {
  const bufferAllocationLabelCounts = diffCounts(after.bufferAllocationLabelCounts, before.bufferAllocationLabelCounts);
  return {
    shaderModuleHits: after.shaderModuleHits - before.shaderModuleHits,
    shaderModuleMisses: after.shaderModuleMisses - before.shaderModuleMisses,
    bindGroupLayoutHits: after.bindGroupLayoutHits - before.bindGroupLayoutHits,
    bindGroupLayoutMisses: after.bindGroupLayoutMisses - before.bindGroupLayoutMisses,
    pipelineLayoutHits: after.pipelineLayoutHits - before.pipelineLayoutHits,
    pipelineLayoutMisses: after.pipelineLayoutMisses - before.pipelineLayoutMisses,
    computePipelineHits: after.computePipelineHits - before.computePipelineHits,
    computePipelineMisses: after.computePipelineMisses - before.computePipelineMisses,
    bindGroupHits: after.bindGroupHits - before.bindGroupHits,
    bindGroupMisses: after.bindGroupMisses - before.bindGroupMisses,
    shaderModuleCreateMs: after.shaderModuleCreateMs - before.shaderModuleCreateMs,
    bindGroupLayoutCreateMs: after.bindGroupLayoutCreateMs - before.bindGroupLayoutCreateMs,
    pipelineLayoutCreateMs: after.pipelineLayoutCreateMs - before.pipelineLayoutCreateMs,
    computePipelineCreateMs: after.computePipelineCreateMs - before.computePipelineCreateMs,
    bindGroupCreates: after.bindGroupCreates - before.bindGroupCreates,
    bindGroupCreateMs: after.bindGroupCreateMs - before.bindGroupCreateMs,
    bufferCreates: after.bufferCreates - before.bufferCreates,
    bufferCreateMs: after.bufferCreateMs - before.bufferCreateMs,
    bufferAllocationsByLabel: formatBufferAllocationLabels(bufferAllocationLabelCounts),
    bufferAllocationLabelCounts,
  };
}

export function runtimeResourceCreateMs(stats: WebGpuRuntimeResourceStats): number {
  return stats.shaderModuleCreateMs +
    stats.bindGroupLayoutCreateMs +
    stats.pipelineLayoutCreateMs +
    stats.computePipelineCreateMs +
    stats.bindGroupCreateMs +
    stats.bufferCreateMs;
}

function stableKey(value: unknown, state: RuntimeResourceCacheState): string {
  if (value === null || value === undefined) {
    return String(value);
  }
  if (typeof value === "number" || typeof value === "boolean" || typeof value === "string") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableKey(item, state)).join(",")}]`;
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record).sort();
    if (keys.length === 0) {
      return `#${objectId(value, state)}`;
    }
    return `{${keys.map((key) => `${JSON.stringify(key)}:${stableKey(record[key], state)}`).join(",")}}`;
  }
  return String(value);
}

function recordBufferAllocationLabel(
  descriptor: {
    label?: string;
    size: number;
    usage: number;
  },
  state: RuntimeResourceCacheState,
): void {
  if (!state.trackBufferAllocations) {
    return;
  }
  const label = descriptor.label ?? `usage:${descriptor.usage}:size:${descriptor.size}`;
  state.bufferAllocationLabels.set(label, (state.bufferAllocationLabels.get(label) ?? 0) + 1);
}

function formatBufferAllocationLabels(labels: Record<string, number>): string {
  const entries = Object.entries(labels).filter(([, count]) => count > 0);
  if (entries.length === 0) {
    return "";
  }
  return entries
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, 32)
    .map(([label, count]) => `${label}=${count}`)
    .join(";");
}

function diffCounts(after: Record<string, number>, before: Record<string, number>): Record<string, number> {
  const diff: Record<string, number> = {};
  for (const [label, afterCount] of Object.entries(after)) {
    const count = afterCount - (before[label] ?? 0);
    if (count > 0) {
      diff[label] = count;
    }
  }
  return diff;
}

function objectId(value: object, state: RuntimeResourceCacheState): number {
  const existing = state.objectIds.get(value);
  if (existing !== undefined) {
    return existing;
  }
  const id = state.nextObjectId;
  state.nextObjectId += 1;
  state.objectIds.set(value, id);
  return id;
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
