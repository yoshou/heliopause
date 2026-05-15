import type { ExecutionProviderStats, ModelSession } from "../../runtime";
import {
  tensorByteLength,
  type TensorByteRange,
} from "../../tensor-reader";
import {
  createWasmQuantizedWeightHandle,
  releaseWasmQuantizedWeightHandle,
  type WasmQuantizedWeightHandle,
} from "./wasm-kernels";
import {
  splitRows,
  WasmThreadPool,
  type WasmShardedQuantizedWeightHandle,
} from "./thread-pool";

type WasmWeightCache = {
  handles: Map<string, WasmQuantizedWeightHandle>;
  pending: Map<string, Promise<WasmQuantizedWeightHandle | undefined>>;
  bytes: number;
  hits: number;
  misses: number;
};

const wasmWeightCaches = new WeakMap<ModelSession, WasmWeightCache>();
const wasmShardedWeightCaches = new WeakMap<ModelSession, WasmShardedWeightCache>();
const wasmThreadPools = new WeakMap<ModelSession, WasmThreadPool>();
const wasmThreadPoolPromises = new WeakMap<ModelSession, Promise<WasmThreadPool | undefined>>();
const wasmIoPrefetchStates = new WeakMap<ModelSession, WasmIoPrefetchState>();

type WasmShardedWeightCache = {
  handles: Map<string, WasmShardedQuantizedWeightHandle>;
  pending: Map<string, Promise<WasmShardedQuantizedWeightHandle | undefined>>;
  bytes: number;
  hits: number;
  misses: number;
};

type WasmIoPrefetchState = {
  active: number;
  queue: WasmPrefetchWeight[];
  scheduled: Set<string>;
  reads: number;
  bytes: number;
  workerBlobReads: number;
  workerBlobBytes: number;
};

type WasmPrefetchWeight = {
  name: string;
  type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0";
  inputSize: number;
  rowCount: number;
  byteLength: number;
};

export function registerCpuExecutionProvider(session: ModelSession): void {
  session.setExecutionProviderStatsProvider(() => cpuExecutionProviderStats(session), "cpu");
}

export function cpuExecutionProviderStats(session: ModelSession): ExecutionProviderStats {
  const cache = wasmWeightCaches.get(session);
  const shardedCache = wasmShardedWeightCaches.get(session);
  const pool = wasmThreadPools.get(session);
  const ioStats = session.tensorReader.ioStats();
  const prefetch = wasmIoPrefetchStates.get(session);
  return {
    cpuResidentWeightCacheEnabled: cpuResidentWeightCacheEnabled(session),
    cpuResidentWeightCacheCount: cache?.handles.size ?? 0,
    cpuResidentWeightCacheBytes: cache?.bytes ?? 0,
    cpuResidentWeightCacheHits: cache?.hits ?? 0,
    cpuResidentWeightCacheMisses: cache?.misses ?? 0,
    cpuThreadPoolEnabled: cpuThreadPoolEnabled(session),
    cpuThreadPoolWorkerCount: pool?.workerCount ?? 0,
    cpuShardedResidentWeightCacheCount: shardedCache?.handles.size ?? 0,
    cpuShardedResidentWeightCacheBytes: shardedCache?.bytes ?? 0,
    cpuShardedResidentWeightCacheHits: shardedCache?.hits ?? 0,
    cpuShardedResidentWeightCacheMisses: shardedCache?.misses ?? 0,
    cpuIoPrefetchEnabled: cpuIoPrefetchEnabled(session),
    cpuIoPrefetchReads: prefetch?.reads ?? 0,
    cpuIoPrefetchBytes: prefetch?.bytes ?? 0,
    cpuIoWorkerBlobReads: prefetch?.workerBlobReads ?? 0,
    cpuIoWorkerBlobBytes: prefetch?.workerBlobBytes ?? 0,
    cpuIoCoalescedReads: ioStats.coalescedReads,
    cpuIoInflightHits: ioStats.inflightHits,
    cpuIoReadMs: ioStats.readMs,
  };
}

export function shutdownCpuExecutionProvider(session: ModelSession): void {
  const cache = wasmWeightCaches.get(session);
  if (cache) {
    for (const handle of cache.handles.values()) {
      releaseWasmQuantizedWeightHandle(handle);
    }
    cache.handles.clear();
    cache.pending.clear();
    cache.bytes = 0;
  }

  wasmShardedWeightCaches.delete(session);
  wasmIoPrefetchStates.delete(session);
  const pool = wasmThreadPools.get(session);
  if (pool) {
    pool.shutdown();
    wasmThreadPools.delete(session);
  }
  wasmThreadPoolPromises.delete(session);
}

export function cpuProjectionBatchingEnabled(session: ModelSession): boolean {
  return booleanCpuOption(session, "projectionBatching");
}

export function cpuResidentWeightCacheEnabled(session: ModelSession): boolean {
  return booleanCpuOption(session, "residentWeightCache");
}

export function cpuThreadPoolEnabled(session: ModelSession): boolean {
  return cpuResidentWeightCacheEnabled(session) &&
    booleanCpuOption(session, "parallelResidentMatmul") &&
    cpuThreadPoolSize(session) > 1;
}

export function prefetchWasmShardedLayerWeights(session: ModelSession, layer: number): void {
  if (!cpuThreadPoolEnabled(session) || !cpuIoPrefetchEnabled(session)) {
    return;
  }
  if (layer < 0 || layer >= session.manifest.blockCount) {
    return;
  }
  enqueueWasmIoPrefetch(session, orderedPrefetchWeightsForLayer(session, layer));
}

export function prefetchWasmShardedOutputWeight(session: ModelSession): void {
  if (!cpuThreadPoolEnabled(session) || !cpuIoPrefetchEnabled(session)) {
    return;
  }
  enqueueWasmIoPrefetch(session, prefetchWeightForName(session, "output.weight"));
}

function enqueueWasmIoPrefetch(session: ModelSession, weights: WasmPrefetchWeight[]): void {
  if (weights.length === 0) {
    return;
  }
  const state = ensureWasmIoPrefetchState(session);
  for (const weight of weights) {
    if (state.scheduled.has(weight.name)) {
      continue;
    }
    state.scheduled.add(weight.name);
    state.queue.push(weight);
  }
  pumpWasmIoPrefetch(session, state);
}

export async function readWasmWeightHandle(
  session: ModelSession,
  name: string,
  type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0",
  inputSize: number,
  rowCount: number,
): Promise<WasmQuantizedWeightHandle | undefined> {
  if (!cpuResidentWeightCacheEnabled(session)) {
    return undefined;
  }
  const cache = ensureWasmWeightCache(session);
  const cached = cache.handles.get(name);
  if (cached) {
    cache.hits += 1;
    return cached;
  }
  const pending = cache.pending.get(name);
  if (pending) {
    cache.hits += 1;
    return pending;
  }

  cache.misses += 1;
  const pendingRead = (async () => {
    const bytes = await session.tensorReader.readTensorBytes(name);
    const handle = await createWasmQuantizedWeightHandle(type, bytes, inputSize, rowCount);
    if (!handle) {
      return undefined;
    }
    cache.handles.set(name, handle);
    cache.bytes += handle.byteLength;
    return handle;
  })();
  cache.pending.set(name, pendingRead);
  return pendingRead.finally(() => {
    cache.pending.delete(name);
  });
}

export async function readWasmShardedWeightHandle(
  session: ModelSession,
  name: string,
  type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0",
  inputSize: number,
  rowCount: number,
): Promise<WasmShardedQuantizedWeightHandle | undefined> {
  if (!cpuThreadPoolEnabled(session) || rowCount < cpuParallelMatmulMinRows(session)) {
    return undefined;
  }
  const pool = await ensureWasmThreadPool(session);
  if (!pool) {
    return undefined;
  }
  const cache = ensureWasmShardedWeightCache(session);
  const cached = cache.handles.get(name);
  if (cached) {
    cache.hits += 1;
    return cached;
  }
  const pending = cache.pending.get(name);
  if (pending) {
    cache.hits += 1;
    return pending;
  }

  cache.misses += 1;
  const pendingPrepare = prepareWasmShardedWeightHandle(
    session,
    pool,
    name,
    type,
    inputSize,
    rowCount,
    cache,
  );
  cache.pending.set(name, pendingPrepare);
  return pendingPrepare.finally(() => {
    cache.pending.delete(name);
  });
}

export async function matMulWasmShardedWeightHandle(
  handle: WasmShardedQuantizedWeightHandle,
  inputColumns: Float32Array,
  inputSize: number,
  rowCount: number,
  columnCount: number,
): Promise<Float32Array | undefined> {
  return handle.pool.matmul(handle, inputColumns, inputSize, rowCount, columnCount);
}

export async function matMulWasmShardedWeightHandleBatch(
  handles: readonly WasmShardedQuantizedWeightHandle[],
  inputColumns: Float32Array,
  inputSize: number,
  columnCount: number,
): Promise<Float32Array[] | undefined> {
  const pool = handles[0]?.pool;
  if (!pool || handles.some((handle) => handle.pool !== pool)) {
    return undefined;
  }
  return pool.matmulBatch(handles, inputColumns, inputSize, columnCount);
}

function booleanCpuOption(session: ModelSession, name: string): boolean {
  return session.executionProvider("cpu")?.options?.[name] === true;
}

function numberCpuOption(session: ModelSession, name: string): number | undefined {
  const value = session.executionProvider("cpu")?.options?.[name];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function cpuThreadPoolSize(session: ModelSession): number {
  const value = session.executionProvider("cpu")?.options?.threadPoolSize;
  if (value === "auto") {
    return Math.max(1, Math.min(8, Math.floor((globalThis.navigator?.hardwareConcurrency ?? 4) / 2)));
  }
  return typeof value === "number" && Number.isFinite(value) ? Math.max(1, Math.floor(value)) : 1;
}

function cpuParallelMatmulMinRows(session: ModelSession): number {
  return numberCpuOption(session, "parallelMatmulMinRows") ?? 512;
}

function cpuIoPrefetchEnabled(session: ModelSession): boolean {
  const value = session.executionProvider("cpu")?.options?.ioPrefetch;
  return typeof value === "boolean" ? value : cpuResidentWeightCacheEnabled(session);
}

function cpuIoPrefetchConcurrency(session: ModelSession): number {
  const value = session.executionProvider("cpu")?.options?.ioPrefetchConcurrency;
  if (value === "auto" || value === undefined) {
    const globalWithProcess = globalThis as typeof globalThis & {
      process?: { versions?: { node?: string } };
    };
    return globalWithProcess.process?.versions?.node ? 4 : 2;
  }
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(1, Math.floor(value))
    : 1;
}

function cpuIoCoalesceMaxGapBytes(session: ModelSession): number {
  return numberCpuOption(session, "ioCoalesceMaxGapBytes") ?? 1024 * 1024;
}

function cpuIoCoalesceMaxReadBytes(session: ModelSession): number {
  return numberCpuOption(session, "ioCoalesceMaxReadBytes") ?? 256 * 1024 * 1024;
}

function cpuIoWorkerBlobReadEnabled(session: ModelSession): boolean {
  const value = session.executionProvider("cpu")?.options?.ioWorkerBlobRead;
  return typeof value === "boolean" ? value : false;
}

function ensureWasmWeightCache(session: ModelSession): WasmWeightCache {
  let cache = wasmWeightCaches.get(session);
  if (!cache) {
    cache = {
      handles: new Map<string, WasmQuantizedWeightHandle>(),
      pending: new Map(),
      bytes: 0,
      hits: 0,
      misses: 0,
    };
    wasmWeightCaches.set(session, cache);
    registerCpuExecutionProvider(session);
  }
  return cache;
}

function ensureWasmShardedWeightCache(session: ModelSession): WasmShardedWeightCache {
  let cache = wasmShardedWeightCaches.get(session);
  if (!cache) {
    cache = {
      handles: new Map(),
      pending: new Map(),
      bytes: 0,
      hits: 0,
      misses: 0,
    };
    wasmShardedWeightCaches.set(session, cache);
    registerCpuExecutionProvider(session);
  }
  return cache;
}

async function ensureWasmThreadPool(session: ModelSession): Promise<WasmThreadPool | undefined> {
  let pool = wasmThreadPools.get(session);
  if (pool) {
    return pool;
  }
  const pending = wasmThreadPoolPromises.get(session);
  if (pending) {
    return pending;
  }
  const created = WasmThreadPool.create(cpuThreadPoolSize(session)).then((nextPool) => {
    if (nextPool) {
      wasmThreadPools.set(session, nextPool);
      registerCpuExecutionProvider(session);
    }
    return nextPool;
  }).finally(() => {
    wasmThreadPoolPromises.delete(session);
  });
  wasmThreadPoolPromises.set(session, created);
  return created;
}

async function prepareWasmShardedWeightHandle(
  session: ModelSession,
  pool: WasmThreadPool,
  name: string,
  type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0",
  inputSize: number,
  rowCount: number,
  cache: WasmShardedWeightCache,
): Promise<WasmShardedQuantizedWeightHandle | undefined> {
  const tensor = session.getTensor(name);
  const rowByteLength = tensorByteLength({
    ...tensor,
    dimensions: [inputSize],
  });
  const rowShards = splitRows(rowCount, pool.workerCount);
  if (rowShards.length < 2) {
    return undefined;
  }
  const sourceBlob = session.tensorReader.sourceBlob();
  if (sourceBlob && cpuIoWorkerBlobReadEnabled(session)) {
    const blobHandle = await prepareWasmShardedWeightHandleFromBlob(
      session,
      pool,
      name,
      type,
      inputSize,
      rowCount,
      sourceBlob,
      rowByteLength,
      rowShards,
      cache,
    );
    if (blobHandle) {
      return blobHandle;
    }
  }
  const ranges: TensorByteRange[] = rowShards.map((shard) => ({
    tensor,
    offset: BigInt(shard.rowStart * rowByteLength),
    length: shard.rowCount * rowByteLength,
  }));
  const rangeBytes = await session.tensorReader.readTensorRangesCoalesced(ranges, {
    maxGapBytes: cpuIoCoalesceMaxGapBytes(session),
    maxReadBytes: cpuIoCoalesceMaxReadBytes(session),
  });
  const weightShards = rowShards.map((shard, index) => ({
    ...shard,
    weightBytes: rangeBytes[index] ?? new Uint8Array(),
  }));
  const handle = await pool.prepareWeight(type, inputSize, rowCount, weightShards);
  if (!handle) {
    return undefined;
  }
  cache.handles.set(name, handle);
  cache.bytes += handle.residentBytes;
  return handle;
}

async function prepareWasmShardedWeightHandleFromBlob(
  session: ModelSession,
  pool: WasmThreadPool,
  name: string,
  type: "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" | "Q8_0",
  inputSize: number,
  rowCount: number,
  sourceBlob: Blob,
  rowByteLength: number,
  rowShards: Array<{ rowStart: number; rowCount: number }>,
  cache: WasmShardedWeightCache,
): Promise<WasmShardedQuantizedWeightHandle | undefined> {
  const tensor = session.getTensor(name);
  const blobShards = [];
  for (const shard of rowShards) {
    const absoluteOffset = tensor.dataOffset + BigInt(shard.rowStart * rowByteLength);
    const fileOffset = Number(absoluteOffset);
    const byteLength = shard.rowCount * rowByteLength;
    if (!Number.isSafeInteger(fileOffset) || !Number.isSafeInteger(byteLength)) {
      return undefined;
    }
    blobShards.push({
      rowStart: shard.rowStart,
      rowCount: shard.rowCount,
      fileOffset,
      byteLength,
    });
  }
  const state = wasmIoPrefetchStates.get(session);
  if (state) {
    state.workerBlobReads += blobShards.length;
    state.workerBlobBytes += blobShards.reduce((sum, shard) => sum + shard.byteLength, 0);
  }
  const handle = await pool.prepareWeightFromBlob(
    type,
    inputSize,
    rowCount,
    sourceBlob,
    blobShards,
  );
  if (!handle) {
    return undefined;
  }
  cache.handles.set(name, handle);
  cache.bytes += handle.residentBytes;
  return handle;
}

function ensureWasmIoPrefetchState(session: ModelSession): WasmIoPrefetchState {
  let state = wasmIoPrefetchStates.get(session);
  if (!state) {
    state = {
      active: 0,
      queue: [],
      scheduled: new Set(),
      reads: 0,
      bytes: 0,
      workerBlobReads: 0,
      workerBlobBytes: 0,
    };
    wasmIoPrefetchStates.set(session, state);
    registerCpuExecutionProvider(session);
  }
  return state;
}

function pumpWasmIoPrefetch(session: ModelSession, state: WasmIoPrefetchState): void {
  const concurrency = cpuIoPrefetchConcurrency(session);
  while (state.active < concurrency && state.queue.length > 0) {
    const weight = state.queue.shift();
    if (!weight) {
      continue;
    }
    state.active += 1;
    state.reads += 1;
    state.bytes += weight.byteLength;
    void readWasmShardedWeightHandle(
      session,
      weight.name,
      weight.type,
      weight.inputSize,
      weight.rowCount,
    ).catch(() => undefined).finally(() => {
      state.active -= 1;
      pumpWasmIoPrefetch(session, state);
    });
  }
}

function orderedPrefetchWeightsForLayer(session: ModelSession, layer: number): WasmPrefetchWeight[] {
  const names: string[] = [];
  const manifest = session.manifest;
  names.push(`blk.${layer}.attn_q.weight`);
  if (manifest.layerHasKv[layer]) {
    names.push(
      `blk.${layer}.attn_k.weight`,
      `blk.${layer}.attn_v.weight`,
    );
  }
  names.push(
    `blk.${layer}.attn_output.weight`,
    `blk.${layer}.ffn_gate.weight`,
    `blk.${layer}.ffn_up.weight`,
    `blk.${layer}.ffn_down.weight`,
  );

  const weights: WasmPrefetchWeight[] = [];
  for (const name of names) {
    weights.push(...prefetchWeightForName(session, name));
  }
  return weights;
}

function prefetchWeightForName(session: ModelSession, name: string): WasmPrefetchWeight[] {
  let tensor: ReturnType<ModelSession["getTensor"]>;
  try {
    tensor = session.getTensor(name);
  } catch {
    return [];
  }
  if (!isWasmPrefetchType(tensor.type)) {
    return [];
  }
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  if (inputSize <= 0 || rowCount < cpuParallelMatmulMinRows(session)) {
    return [];
  }
  return [{
    name,
    type: tensor.type,
    inputSize,
    rowCount,
    byteLength: tensorByteLength(tensor),
  }];
}

function isWasmPrefetchType(type: string): type is WasmPrefetchWeight["type"] {
  return type === "Q4_K" || type === "Q5_K" || type === "Q6_K" || type === "IQ4_XS" || type === "Q8_0";
}
