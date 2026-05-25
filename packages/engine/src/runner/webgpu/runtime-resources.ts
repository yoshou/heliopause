import type { WebGpuBufferLike, WebGpuDeviceLike } from "./gpu-types";
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
  bufferCreateLabels: string;
  bufferCreateLabelCounts: Record<string, number>;
};

export type WebGpuRuntimeResourceCache = {
  stats: () => WebGpuRuntimeResourceStats;
};

type MutableWebGpuDevice = WebGpuDeviceLike & {
  __heliopauseRuntimeResources?: RuntimeResourceCacheState;
};

type RuntimeResourceCacheState = WebGpuRuntimeResourceCache & {
  objectIds: WeakMap<object, number>;
  nextObjectId: number;
  shaderModules: Map<string, unknown>;
  bindGroupLayouts: Map<string, unknown>;
  pipelineLayouts: Map<string, unknown>;
  computePipelines: Map<string, unknown>;
  uniformBufferPools: Map<string, PooledUniformBuffer[]>;
  readbackBufferPools: Map<string, PooledReadbackBuffer[]>;
  statsValue: WebGpuRuntimeResourceStats;
  bufferCreateLabels: Map<string, number>;
};

type PooledUniformBuffer = ReturnType<WebGpuDeviceLike["createBuffer"]> & {
  __heliopauseUniformPoolKey?: string;
  __heliopauseInUniformPool?: boolean;
  __heliopauseUniformPoolDestroyed?: boolean;
};

type PooledReadbackBuffer = WebGpuBufferLike & {
  __heliopauseReadbackPoolKey?: string;
  __heliopauseInReadbackPool?: boolean;
  __heliopauseReadbackPoolDestroyed?: boolean;
  __heliopauseReadbackMapPending?: boolean;
  __heliopauseReadbackMapped?: boolean;
};

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
  bufferCreateLabels: "",
  bufferCreateLabelCounts: {},
});

export function installWebGpuRuntimeResourceCache(
  device: WebGpuDeviceLike,
): WebGpuRuntimeResourceCache {
  const mutable = device as MutableWebGpuDevice;
  if (mutable.__heliopauseRuntimeResources) {
    return mutable.__heliopauseRuntimeResources;
  }

  const state: RuntimeResourceCacheState = {
    objectIds: new WeakMap<object, number>(),
    nextObjectId: 1,
    shaderModules: new Map<string, unknown>(),
    bindGroupLayouts: new Map<string, unknown>(),
    pipelineLayouts: new Map<string, unknown>(),
    computePipelines: new Map<string, unknown>(),
    // TODO: Add bounded flush/eviction for these exact-size buffer pools once the cache policy is redesigned.
    uniformBufferPools: new Map<string, PooledUniformBuffer[]>(),
    readbackBufferPools: new Map<string, PooledReadbackBuffer[]>(),
    bufferCreateLabels: new Map<string, number>(),
    statsValue: emptyStats(),
    stats() {
      const bufferCreateLabelCounts = Object.fromEntries(this.bufferCreateLabels);
      return {
        ...this.statsValue,
        bufferCreateLabels: formatBufferCreateLabels(bufferCreateLabelCounts),
        bufferCreateLabelCounts,
      };
    },
  };

  const createBuffer = device.createBuffer.bind(device);
  const createShaderModule = device.createShaderModule.bind(device);
  const createBindGroupLayout = device.createBindGroupLayout.bind(device);
  const createPipelineLayout = device.createPipelineLayout.bind(device);
  const createComputePipeline = device.createComputePipeline.bind(device);
  const createBindGroup = device.createBindGroup.bind(device);

  mutable.createBuffer = (descriptor) => {
    const uniformPoolKey = uniformBufferPoolKey(descriptor);
    const readbackPoolKey = readbackBufferPoolKey(descriptor);
    if (uniformPoolKey) {
      const pooled = state.uniformBufferPools.get(uniformPoolKey)?.pop();
      if (pooled) {
        pooled.__heliopauseInUniformPool = false;
        return pooled;
      }
    }
    if (readbackPoolKey) {
      const pooled = state.readbackBufferPools.get(readbackPoolKey)?.pop();
      if (pooled) {
        pooled.__heliopauseInReadbackPool = false;
        return pooled;
      }
    }
    const start = nowMs();
    try {
      const created = createBuffer(descriptor);
      if (uniformPoolKey) {
        return installUniformBufferPoolRelease(created, uniformPoolKey, state);
      }
      if (readbackPoolKey) {
        return installReadbackBufferPoolRelease(created, readbackPoolKey, state);
      }
      return created;
    } finally {
      state.statsValue.bufferCreates += 1;
      state.statsValue.bufferCreateMs += nowMs() - start;
      recordBufferCreateLabel(descriptor, state);
    }
  };

  mutable.createShaderModule = (descriptor) => {
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
  };

  mutable.createBindGroupLayout = (descriptor) => {
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
  };

  mutable.createPipelineLayout = (descriptor) => {
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
  };

  mutable.createComputePipeline = (descriptor) => {
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
  };

  mutable.createBindGroup = (descriptor) => {
    state.statsValue.bindGroupMisses += 1;
    const start = nowMs();
    try {
      return createBindGroup(descriptor);
    } finally {
      state.statsValue.bindGroupCreates += 1;
      state.statsValue.bindGroupCreateMs += nowMs() - start;
    }
  };

  mutable.__heliopauseRuntimeResources = state;
  return state;
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

function installUniformBufferPoolRelease(
  buffer: ReturnType<WebGpuDeviceLike["createBuffer"]>,
  key: string,
  state: RuntimeResourceCacheState,
): PooledUniformBuffer {
  const pooled = buffer as PooledUniformBuffer;
  const destroy = buffer.destroy?.bind(buffer);
  pooled.__heliopauseUniformPoolKey = key;
  pooled.destroy = () => {
    if (pooled.__heliopauseUniformPoolDestroyed || pooled.__heliopauseInUniformPool) {
      return;
    }
    if (!destroy) {
      return;
    }
    pooled.__heliopauseInUniformPool = true;
    const targetPool = state.uniformBufferPools.get(key);
    if (targetPool) {
      targetPool.push(pooled);
    } else {
      state.uniformBufferPools.set(key, [pooled]);
    }
  };
  return pooled;
}

function installReadbackBufferPoolRelease(
  buffer: ReturnType<WebGpuDeviceLike["createBuffer"]>,
  key: string,
  state: RuntimeResourceCacheState,
): PooledReadbackBuffer {
  const pooled = buffer as PooledReadbackBuffer;
  const mapAsync = buffer.mapAsync.bind(buffer);
  const unmap = buffer.unmap.bind(buffer);
  const destroy = buffer.destroy?.bind(buffer);
  pooled.__heliopauseReadbackPoolKey = key;
  pooled.mapAsync = async (mode) => {
    pooled.__heliopauseReadbackMapPending = true;
    try {
      await mapAsync(mode);
      pooled.__heliopauseReadbackMapped = true;
    } finally {
      pooled.__heliopauseReadbackMapPending = false;
    }
  };
  pooled.unmap = () => {
    try {
      unmap();
    } finally {
      pooled.__heliopauseReadbackMapped = false;
    }
  };
  pooled.destroy = () => {
    if (pooled.__heliopauseReadbackPoolDestroyed || pooled.__heliopauseInReadbackPool) {
      return;
    }
    if (pooled.__heliopauseReadbackMapPending || pooled.__heliopauseReadbackMapped) {
      pooled.__heliopauseReadbackPoolDestroyed = true;
      destroy?.();
      return;
    }
    pooled.__heliopauseInReadbackPool = true;
    const targetPool = state.readbackBufferPools.get(key);
    if (targetPool) {
      targetPool.push(pooled);
    } else {
      state.readbackBufferPools.set(key, [pooled]);
    }
  };
  return pooled;
}

export function diffWebGpuRuntimeResourceStats(
  after: WebGpuRuntimeResourceStats,
  before: WebGpuRuntimeResourceStats,
): WebGpuRuntimeResourceStats {
  const bufferCreateLabelCounts = diffCounts(after.bufferCreateLabelCounts, before.bufferCreateLabelCounts);
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
    bufferCreateLabels: formatBufferCreateLabels(bufferCreateLabelCounts),
    bufferCreateLabelCounts,
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

function recordBufferCreateLabel(
  descriptor: {
    label?: string;
    size: number;
    usage: number;
  },
  state: RuntimeResourceCacheState,
): void {
  if (!webGpuBufferCreateLabelTimingEnabled()) {
    return;
  }
  const label = descriptor.label ?? `usage:${descriptor.usage}:size:${descriptor.size}`;
  state.bufferCreateLabels.set(label, (state.bufferCreateLabels.get(label) ?? 0) + 1);
}

function formatBufferCreateLabels(labels: Record<string, number>): string {
  const entries = Object.entries(labels).filter(([, count]) => count > 0);
  if (!webGpuBufferCreateLabelTimingEnabled() || entries.length === 0) {
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

function webGpuBufferCreateLabelTimingEnabled(): boolean {
  return (globalThis as { __heliopauseDisableWebGpuBufferCreateLabels?: unknown }).__heliopauseDisableWebGpuBufferCreateLabels !== true;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
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
