import type { WebGpuDeviceLike } from "./gpu-types";

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
  bindGroups: Map<string, unknown>;
  statsValue: WebGpuRuntimeResourceStats;
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
    bindGroups: new Map<string, unknown>(),
    statsValue: emptyStats(),
    stats() {
      return { ...this.statsValue };
    },
  };

  const createBuffer = device.createBuffer.bind(device);
  const createShaderModule = device.createShaderModule.bind(device);
  const createBindGroupLayout = device.createBindGroupLayout.bind(device);
  const createPipelineLayout = device.createPipelineLayout.bind(device);
  const createComputePipeline = device.createComputePipeline.bind(device);
  const createBindGroup = device.createBindGroup.bind(device);

  mutable.createBuffer = (descriptor) => {
    const start = nowMs();
    try {
      return createBuffer(descriptor);
    } finally {
      state.statsValue.bufferCreates += 1;
      state.statsValue.bufferCreateMs += nowMs() - start;
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

export function diffWebGpuRuntimeResourceStats(
  after: WebGpuRuntimeResourceStats,
  before: WebGpuRuntimeResourceStats,
): WebGpuRuntimeResourceStats {
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
