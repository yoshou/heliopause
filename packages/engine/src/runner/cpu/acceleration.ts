import type { Qwen35ExecutionProviderStats, Qwen35ModelSession } from "../../runtime";
import {
  createWasmQuantizedWeightHandle,
  type WasmQuantizedWeightHandle,
} from "./wasm-kernels";

type WasmWeightCache = {
  handles: Map<string, WasmQuantizedWeightHandle>;
  bytes: number;
  hits: number;
  misses: number;
};

const wasmWeightCaches = new WeakMap<Qwen35ModelSession, WasmWeightCache>();

export function registerQwen35CpuExecutionProvider(session: Qwen35ModelSession): void {
  session.setExecutionProviderStatsProvider(() => cpuExecutionProviderStats(session));
}

export function cpuExecutionProviderStats(session: Qwen35ModelSession): Qwen35ExecutionProviderStats {
  const cache = wasmWeightCaches.get(session);
  return {
    cpuResidentWeightCacheEnabled: cpuResidentWeightCacheEnabled(session),
    cpuResidentWeightCacheCount: cache?.handles.size ?? 0,
    cpuResidentWeightCacheBytes: cache?.bytes ?? 0,
    cpuResidentWeightCacheHits: cache?.hits ?? 0,
    cpuResidentWeightCacheMisses: cache?.misses ?? 0,
  };
}

export function cpuProjectionBatchingEnabled(session: Qwen35ModelSession): boolean {
  return booleanCpuOption(session, "projectionBatching");
}

export function cpuResidentWeightCacheEnabled(session: Qwen35ModelSession): boolean {
  return booleanCpuOption(session, "residentWeightCache");
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

function booleanCpuOption(session: Qwen35ModelSession, name: string): boolean {
  return session.executionProvider("cpu")?.options?.[name] === true;
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
