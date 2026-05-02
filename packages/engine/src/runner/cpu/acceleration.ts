import type { ExecutionProviderStats, Qwen35ModelSession } from "../../runtime";
import {
  tensorByteLength,
} from "../../tensor-reader";
import {
  createWasmQuantizedWeightHandle,
  type WasmQuantizedWeightHandle,
} from "./wasm-kernels";
import {
  splitRows,
  WasmThreadPool,
  type WasmShardedQuantizedWeightHandle,
} from "./thread-pool";

type WasmWeightCache = {
  handles: Map<string, WasmQuantizedWeightHandle>;
  bytes: number;
  hits: number;
  misses: number;
};

const wasmWeightCaches = new WeakMap<Qwen35ModelSession, WasmWeightCache>();
const wasmShardedWeightCaches = new WeakMap<Qwen35ModelSession, WasmShardedWeightCache>();
const wasmThreadPools = new WeakMap<Qwen35ModelSession, WasmThreadPool>();

type WasmShardedWeightCache = {
  handles: Map<string, WasmShardedQuantizedWeightHandle>;
  bytes: number;
  hits: number;
  misses: number;
};

export function registerQwen35CpuExecutionProvider(session: Qwen35ModelSession): void {
  session.setExecutionProviderStatsProvider(() => cpuExecutionProviderStats(session));
}

export function cpuExecutionProviderStats(session: Qwen35ModelSession): ExecutionProviderStats {
  const cache = wasmWeightCaches.get(session);
  const shardedCache = wasmShardedWeightCaches.get(session);
  const pool = wasmThreadPools.get(session);
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
  };
}

export function cpuProjectionBatchingEnabled(session: Qwen35ModelSession): boolean {
  return booleanCpuOption(session, "projectionBatching");
}

export function cpuResidentWeightCacheEnabled(session: Qwen35ModelSession): boolean {
  return booleanCpuOption(session, "residentWeightCache");
}

export function cpuThreadPoolEnabled(session: Qwen35ModelSession): boolean {
  return cpuResidentWeightCacheEnabled(session) &&
    booleanCpuOption(session, "parallelResidentMatmul") &&
    cpuThreadPoolSize(session) > 1;
}

export async function readWasmWeightHandle(
  session: Qwen35ModelSession,
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

  cache.misses += 1;
  const bytes = await session.tensorReader.readTensorBytes(name);
  const handle = await createWasmQuantizedWeightHandle(type, bytes, inputSize, rowCount);
  if (!handle) {
    return undefined;
  }
  cache.handles.set(name, handle);
  cache.bytes += handle.byteLength;
  return handle;
}

export async function readWasmShardedWeightHandle(
  session: Qwen35ModelSession,
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

  cache.misses += 1;
  const tensor = session.getTensor(name);
  const rowByteLength = tensorByteLength({
    ...tensor,
    dimensions: [inputSize],
  });
  const rowShards = splitRows(rowCount, pool.workerCount);
  if (rowShards.length < 2) {
    return undefined;
  }
  const weightShards = await Promise.all(rowShards.map(async (shard) => ({
    ...shard,
    weightBytes: await session.tensorReader.readTensorRange({
      tensor,
      offset: BigInt(shard.rowStart * rowByteLength),
      length: shard.rowCount * rowByteLength,
    }),
  })));
  const handle = await pool.prepareWeight(type, inputSize, rowCount, weightShards);
  if (!handle) {
    return undefined;
  }
  cache.handles.set(name, handle);
  cache.bytes += handle.residentBytes;
  return handle;
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

function booleanCpuOption(session: Qwen35ModelSession, name: string): boolean {
  return session.executionProvider("cpu")?.options?.[name] === true;
}

function numberCpuOption(session: Qwen35ModelSession, name: string): number | undefined {
  const value = session.executionProvider("cpu")?.options?.[name];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function cpuThreadPoolSize(session: Qwen35ModelSession): number {
  const value = session.executionProvider("cpu")?.options?.threadPoolSize;
  if (value === "auto") {
    return Math.max(1, Math.min(8, Math.floor((globalThis.navigator?.hardwareConcurrency ?? 4) / 2)));
  }
  return typeof value === "number" && Number.isFinite(value) ? Math.max(1, Math.floor(value)) : 1;
}

function cpuParallelMatmulMinRows(session: Qwen35ModelSession): number {
  return numberCpuOption(session, "parallelMatmulMinRows") ?? 512;
}

function ensureWasmWeightCache(session: Qwen35ModelSession): WasmWeightCache {
  let cache = wasmWeightCaches.get(session);
  if (!cache) {
    cache = {
      handles: new Map<string, WasmQuantizedWeightHandle>(),
      bytes: 0,
      hits: 0,
      misses: 0,
    };
    wasmWeightCaches.set(session, cache);
    registerQwen35CpuExecutionProvider(session);
  }
  return cache;
}

function ensureWasmShardedWeightCache(session: Qwen35ModelSession): WasmShardedWeightCache {
  let cache = wasmShardedWeightCaches.get(session);
  if (!cache) {
    cache = {
      handles: new Map(),
      bytes: 0,
      hits: 0,
      misses: 0,
    };
    wasmShardedWeightCaches.set(session, cache);
    registerQwen35CpuExecutionProvider(session);
  }
  return cache;
}

async function ensureWasmThreadPool(session: Qwen35ModelSession): Promise<WasmThreadPool | undefined> {
  let pool = wasmThreadPools.get(session);
  if (!pool) {
    pool = await WasmThreadPool.create(cpuThreadPoolSize(session));
    if (pool) {
      wasmThreadPools.set(session, pool);
      registerQwen35CpuExecutionProvider(session);
    }
  }
  return pool;
}
