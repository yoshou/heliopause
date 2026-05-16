import type {
  WasmThreadQuantizedType,
  WasmThreadWorkerRequest,
  WasmThreadWorkerResponse,
} from "./thread-worker-protocol";

type PendingRequest = {
  resolve: (response: WasmThreadWorkerResponse) => void;
  reject: (error: Error) => void;
};

type WorkerSlot = {
  worker: WorkerBridge;
  pending: Map<number, PendingRequest>;
};

type WorkerBridge = {
  postMessage(message: WasmThreadWorkerRequest, transfer?: Transferable[]): void;
  onMessage(handler: (response: WasmThreadWorkerResponse) => void): void;
  onError(handler: (error: Error) => void): void;
};

export type WasmWeightShardInput = {
  rowStart: number;
  rowCount: number;
  weightBytes: Uint8Array;
};

export type WasmWeightShardBlobInput = {
  rowStart: number;
  rowCount: number;
  fileOffset: number;
  byteLength: number;
};

export type WasmShardedWeightShard = {
  workerIndex: number;
  handleId: number;
  rowStart: number;
  rowCount: number;
  residentBytes: number;
};

export type WasmShardedQuantizedWeightHandle = {
  pool: WasmThreadPool;
  type: WasmThreadQuantizedType;
  inputSize: number;
  rowCount: number;
  residentBytes: number;
  shards: WasmShardedWeightShard[];
};

export class WasmThreadPool {
  readonly workerCount: number;

  private readonly workers: WorkerSlot[];
  private nextRequestId = 1;
  private nextHandleId = 1;
  private shutdownStarted = false;

  static async create(workerCount: number): Promise<WasmThreadPool | undefined> {
    if (workerCount < 2) {
      return undefined;
    }
    try {
      const workers = await Promise.all(Array.from({ length: workerCount }, () => createWorkerBridge()));
      return new WasmThreadPool(workerCount, workers);
    } catch {
      return undefined;
    }
  }

  private constructor(workerCount: number, workers: WorkerBridge[]) {
    this.workerCount = workerCount;
    this.workers = workers.map((worker) => {
      const slot: WorkerSlot = {
        worker,
        pending: new Map(),
      };
      worker.onMessage((response) => {
        const pending = slot.pending.get(response.requestId);
        if (!pending) {
          return;
        }
        slot.pending.delete(response.requestId);
        if (response.type === "error") {
          pending.reject(new Error(response.message));
        } else {
          pending.resolve(response);
        }
      });
      worker.onError((error) => {
        for (const pending of slot.pending.values()) {
          pending.reject(error);
        }
        slot.pending.clear();
      });
      return slot;
    });
  }

  async prepareWeight(
    type: WasmThreadQuantizedType,
    inputSize: number,
    rowCount: number,
    shards: readonly WasmWeightShardInput[],
  ): Promise<WasmShardedQuantizedWeightHandle | undefined> {
    if (this.shutdownStarted || shards.length < 2) {
      return undefined;
    }

    const prepared = await Promise.all(shards.map(async (shard, index) => {
      const workerIndex = index % this.workers.length;
      const handleId = this.nextHandleId++;
      const weightBytes = shard.weightBytes.byteOffset === 0 &&
          shard.weightBytes.byteLength === shard.weightBytes.buffer.byteLength
        ? shard.weightBytes
        : shard.weightBytes.slice();
      const response = await this.send(workerIndex, {
        type: "prepareWeight",
        requestId: this.nextRequestId++,
        handleId,
        quantizedType: type,
        weightBuffer: weightBytes.buffer as ArrayBuffer,
        inputSize,
        rowCount: shard.rowCount,
      }, [weightBytes.buffer as ArrayBuffer]);
      if (response.type !== "preparedWeight") {
        throw new Error("Unexpected WASM thread prepare response");
      }
      return {
        workerIndex,
        handleId,
        rowStart: shard.rowStart,
        rowCount: shard.rowCount,
        residentBytes: response.residentBytes,
      };
    }));

    return {
      pool: this,
      type,
      inputSize,
      rowCount,
      residentBytes: prepared.reduce((sum, shard) => sum + shard.residentBytes, 0),
      shards: prepared,
    };
  }

  async prepareWeightFromBlob(
    type: WasmThreadQuantizedType,
    inputSize: number,
    rowCount: number,
    fileBlob: Blob,
    shards: readonly WasmWeightShardBlobInput[],
  ): Promise<WasmShardedQuantizedWeightHandle | undefined> {
    if (this.shutdownStarted || shards.length < 2) {
      return undefined;
    }

    const prepared = await Promise.all(shards.map(async (shard, index) => {
      const workerIndex = index % this.workers.length;
      const handleId = this.nextHandleId++;
      const response = await this.send(workerIndex, {
        type: "prepareWeightFromBlob",
        requestId: this.nextRequestId++,
        handleId,
        quantizedType: type,
        fileBlob,
        fileOffset: shard.fileOffset,
        byteLength: shard.byteLength,
        inputSize,
        rowCount: shard.rowCount,
      });
      if (response.type !== "preparedWeight") {
        throw new Error("Unexpected WASM thread blob prepare response");
      }
      return {
        workerIndex,
        handleId,
        rowStart: shard.rowStart,
        rowCount: shard.rowCount,
        residentBytes: response.residentBytes,
      };
    }));

    return {
      pool: this,
      type,
      inputSize,
      rowCount,
      residentBytes: prepared.reduce((sum, shard) => sum + shard.residentBytes, 0),
      shards: prepared,
    };
  }

  async matmul(
    handle: WasmShardedQuantizedWeightHandle,
    inputColumns: Float32Array,
    inputSize: number,
    rowCount: number,
    columnCount: number,
  ): Promise<Float32Array | undefined> {
    if (handle.pool !== this || handle.inputSize !== inputSize || handle.rowCount !== rowCount) {
      return undefined;
    }
    const sharedInput = sharedFloat32Buffer(inputColumns);
    const shardOutputs = await Promise.all(handle.shards.map(async (shard) => {
      const input = sharedInput ?? (inputColumns.slice().buffer as ArrayBuffer);
      const response = await this.send(shard.workerIndex, {
        type: "matmul",
        requestId: this.nextRequestId++,
        handleId: shard.handleId,
        inputBuffer: input,
        inputSize,
        rowCount: shard.rowCount,
        columnCount,
      }, sharedInput ? [] : [input as ArrayBuffer]);
      if (response.type !== "matmulResult") {
        throw new Error("Unexpected WASM thread matmul response");
      }
      return {
        shard,
        output: new Float32Array(response.outputBuffer, 0, response.outputLength),
      };
    }));
    const output = new Float32Array(rowCount * columnCount);
    for (const { shard, output: shardOutput } of shardOutputs) {
      copyShardOutput(output, shardOutput, shard.rowStart, shard.rowCount, rowCount, columnCount);
    }
    return output;
  }

  async matmulBatch(
    handles: readonly WasmShardedQuantizedWeightHandle[],
    inputColumns: Float32Array,
    inputSize: number,
    columnCount: number,
  ): Promise<Float32Array[] | undefined> {
    if (handles.length < 2 || handles.length > 4 || handles.some((handle) => handle.pool !== this)) {
      return undefined;
    }
    if (handles.some((handle) => handle.inputSize !== inputSize)) {
      return undefined;
    }

    const outputs = handles.map((handle) => new Float32Array(handle.rowCount * columnCount));
    const sharedInput = sharedFloat32Buffer(inputColumns);
    await Promise.all(this.workers.map(async (_worker, workerIndex) => {
      const workerShards = handles.map((handle) =>
        handle.shards.find((shard) => shard.workerIndex === workerIndex),
      );
      if (workerShards.some((shard) => !shard)) {
        return;
      }
      const input = sharedInput ?? (inputColumns.slice().buffer as ArrayBuffer);
      const response = await this.send(workerIndex, {
        type: "matmulBatch",
        requestId: this.nextRequestId++,
        handleIds: workerShards.map((shard) => shard?.handleId ?? 0),
        inputBuffer: input,
        inputSize,
        columnCount,
      }, sharedInput ? [] : [input as ArrayBuffer]);
      if (response.type !== "matmulBatchResult") {
        throw new Error("Unexpected WASM thread batch matmul response");
      }
      for (let index = 0; index < response.outputs.length; index += 1) {
        const shard = workerShards[index];
        const target = outputs[index];
        const result = response.outputs[index];
        if (!shard || !target || !result) {
          throw new Error("WASM thread batch result shape mismatch");
        }
        copyShardOutput(
          target,
          new Float32Array(result.outputBuffer, 0, result.outputLength),
          shard.rowStart,
          shard.rowCount,
          handles[index]?.rowCount ?? 0,
          columnCount,
        );
      }
    }));
    return outputs;
  }

  releaseWeight(handle: WasmShardedQuantizedWeightHandle): void {
    for (const shard of handle.shards) {
      void this.send(shard.workerIndex, {
        type: "releaseWeight",
        requestId: this.nextRequestId++,
        handleId: shard.handleId,
      });
    }
  }

  shutdown(): void {
    if (this.shutdownStarted) {
      return;
    }
    this.shutdownStarted = true;
    for (let index = 0; index < this.workers.length; index += 1) {
      void this.send(index, {
        type: "shutdown",
        requestId: this.nextRequestId++,
      });
    }
  }

  private send(
    workerIndex: number,
    message: WasmThreadWorkerRequest,
    transfer: Transferable[] = [],
  ): Promise<WasmThreadWorkerResponse> {
    const slot = this.workers[workerIndex];
    if (!slot) {
      return Promise.reject(new Error(`Unknown WASM worker ${workerIndex}`));
    }
    return new Promise((resolve, reject) => {
      slot.pending.set(message.requestId, { resolve, reject });
      slot.worker.postMessage(message, transfer);
    });
  }
}

async function createWorkerBridge(): Promise<WorkerBridge> {
  if (typeof Worker !== "undefined") {
    const worker = new Worker(new URL("./thread-worker.ts", import.meta.url), {
      type: "module",
    });
    return {
      postMessage(message, transfer = []) {
        worker.postMessage(message, transfer);
      },
      onMessage(handler) {
        worker.onmessage = (event: MessageEvent<WasmThreadWorkerResponse>) => handler(event.data);
      },
      onError(handler) {
        worker.onerror = (event) => handler(new Error(event.message));
      },
    };
  }

  const module = await dynamicImport<{
    Worker: new (
      filename: URL,
      options?: { type?: "module"; execArgv?: string[] },
    ) => {
      postMessage(message: WasmThreadWorkerRequest, transfer?: Transferable[]): void;
      on(event: "message", handler: (response: WasmThreadWorkerResponse) => void): void;
      on(event: "error", handler: (error: Error) => void): void;
    };
  }>("node:worker_threads");
  const processLike = globalThis as typeof globalThis & { process?: { execArgv?: string[] } };
  const worker = new module.Worker(new URL("./thread-worker.ts", import.meta.url), {
    type: "module",
    execArgv: nodeWorkerExecArgv(processLike.process?.execArgv ?? []),
  });
  return {
    postMessage(message, transfer = []) {
      worker.postMessage(message, transfer);
    },
    onMessage(handler) {
      worker.on("message", handler);
    },
    onError(handler) {
      worker.on("error", handler);
    },
  };
}

async function dynamicImport<T>(specifier: string): Promise<T> {
  const importer = new Function("specifier", "return import(specifier)") as (value: string) => Promise<T>;
  return importer(specifier);
}

function nodeWorkerExecArgv(execArgv: string[]): string[] {
  const output: string[] = [];
  for (let index = 0; index < execArgv.length; index += 1) {
    const value = execArgv[index] ?? "";
    if (value === "--input-type") {
      index += 1;
      continue;
    }
    if (value.startsWith("--input-type=")) {
      continue;
    }
    output.push(value);
  }
  return output;
}

export function splitRows(rowCount: number, shardCount: number): Array<{ rowStart: number; rowCount: number }> {
  const output: Array<{ rowStart: number; rowCount: number }> = [];
  const baseRows = Math.floor(rowCount / shardCount);
  let extraRows = rowCount % shardCount;
  let rowStart = 0;
  for (let index = 0; index < shardCount; index += 1) {
    const count = baseRows + (extraRows > 0 ? 1 : 0);
    extraRows -= extraRows > 0 ? 1 : 0;
    if (count > 0) {
      output.push({ rowStart, rowCount: count });
      rowStart += count;
    }
  }
  return output;
}

function copyShardOutput(
  target: Float32Array,
  shardOutput: Float32Array,
  rowStart: number,
  shardRowCount: number,
  totalRowCount: number,
  columnCount: number,
): void {
  if (shardOutput.length !== shardRowCount * columnCount) {
    throw new Error(`WASM thread shard output shape mismatch: ${shardOutput.length}`);
  }
  for (let column = 0; column < columnCount; column += 1) {
    target.set(
      shardOutput.subarray(column * shardRowCount, (column + 1) * shardRowCount),
      column * totalRowCount + rowStart,
    );
  }
}

function sharedFloat32Buffer(input: Float32Array): SharedArrayBuffer | undefined {
  if (typeof SharedArrayBuffer === "undefined") {
    return undefined;
  }
  const buffer = new SharedArrayBuffer(input.byteLength);
  new Float32Array(buffer).set(input);
  return buffer;
}
