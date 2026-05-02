import {
  createWasmQuantizedWeightHandle,
  matMulQuantizedWasmResident,
  matMulQuantizedWasmResidentMulti,
  releaseWasmQuantizedWeightHandle,
  type WasmQuantizedWeightHandle,
} from "./wasm-kernels";
import type {
  WasmThreadWorkerRequest,
  WasmThreadWorkerResponse,
} from "./thread-worker-protocol";

const handles = new Map<number, WasmQuantizedWeightHandle>();

type WasmThreadWorkerScope = {
  postMessage(message: WasmThreadWorkerResponse, transfer?: Transferable[]): void;
  close(): void;
  onMessage(handler: (request: WasmThreadWorkerRequest) => void): void;
};

let workerScope: WasmThreadWorkerScope | undefined;

void createWorkerScope().then((scope) => {
  workerScope = scope;
  scope.onMessage((request) => {
    void handleRequest(request);
  });
});

async function handleRequest(request: WasmThreadWorkerRequest): Promise<void> {
  try {
    if (request.type === "prepareWeight") {
      const handle = await createWasmQuantizedWeightHandle(
        request.quantizedType,
        new Uint8Array(request.weightBuffer),
        request.inputSize,
        request.rowCount,
      );
      if (!handle) {
        throw new Error("WASM resident weight preparation failed");
      }
      handles.set(request.handleId, handle);
      postMessage({
        type: "preparedWeight",
        requestId: request.requestId,
        residentBytes: handle.byteLength + handle.scaleByteLength,
      });
      return;
    }

    if (request.type === "matmul") {
      const handle = requiredHandle(request.handleId);
      const output = await matMulQuantizedWasmResident(
        handle,
        new Float32Array(request.inputBuffer),
        request.inputSize,
        request.rowCount,
        request.columnCount,
      );
      if (!output) {
        throw new Error("WASM resident shard matmul failed");
      }
      postMessage({
        type: "matmulResult",
        requestId: request.requestId,
        outputBuffer: output.buffer as ArrayBuffer,
        outputLength: output.length,
      }, [output.buffer as ArrayBuffer]);
      return;
    }

    if (request.type === "matmulBatch") {
      const batchHandles = request.handleIds.map((handleId) => requiredHandle(handleId));
      const outputs = await matMulQuantizedWasmResidentMulti(
        batchHandles,
        new Float32Array(request.inputBuffer),
        request.inputSize,
        request.columnCount,
      );
      if (!outputs || outputs.length !== batchHandles.length) {
        throw new Error("WASM resident shard batch matmul failed");
      }
      postMessage({
        type: "matmulBatchResult",
        requestId: request.requestId,
        outputs: outputs.map((output) => ({
          outputBuffer: output.buffer as ArrayBuffer,
          outputLength: output.length,
        })),
      }, outputs.map((output) => output.buffer as ArrayBuffer));
      return;
    }

    if (request.type === "releaseWeight") {
      const handle = handles.get(request.handleId);
      if (handle) {
        releaseWasmQuantizedWeightHandle(handle);
        handles.delete(request.handleId);
      }
      postMessage({
        type: "releasedWeight",
        requestId: request.requestId,
      });
      return;
    }

    for (const handle of handles.values()) {
      releaseWasmQuantizedWeightHandle(handle);
    }
    handles.clear();
    postMessage({
      type: "shutdownComplete",
      requestId: request.requestId,
    });
    workerScope?.close();
  } catch (error) {
    postMessage({
      type: "error",
      requestId: request.requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
}

function requiredHandle(handleId: number): WasmQuantizedWeightHandle {
  const handle = handles.get(handleId);
  if (!handle) {
    throw new Error(`Unknown WASM resident shard handle ${handleId}`);
  }
  return handle;
}

function postMessage(message: WasmThreadWorkerResponse, transfer: Transferable[] = []): void {
  if (!workerScope) {
    throw new Error("WASM thread worker scope is not initialized");
  }
  workerScope.postMessage(message, transfer);
}

async function createWorkerScope(): Promise<WasmThreadWorkerScope> {
  if (typeof self !== "undefined" && typeof self.postMessage === "function") {
    const browserScope = self as unknown as {
      postMessage(message: WasmThreadWorkerResponse, transfer?: Transferable[]): void;
      close(): void;
      onmessage: ((event: MessageEvent<WasmThreadWorkerRequest>) => void) | null;
    };
    return {
      postMessage(message, transfer = []) {
        browserScope.postMessage(message, transfer);
      },
      close() {
        browserScope.close();
      },
      onMessage(handler) {
        browserScope.onmessage = (event) => handler(event.data);
      },
    };
  }

  const module = await dynamicImport<{
    parentPort?: {
      postMessage(message: WasmThreadWorkerResponse, transfer?: Transferable[]): void;
      on(event: "message", handler: (request: WasmThreadWorkerRequest) => void): void;
      close(): void;
    };
  }>("node:worker_threads");
  if (!module.parentPort) {
    throw new Error("node:worker_threads parentPort is not available");
  }
  return {
    postMessage(message, transfer = []) {
      module.parentPort?.postMessage(message, transfer);
    },
    close() {
      module.parentPort?.close();
    },
    onMessage(handler) {
      module.parentPort?.on("message", handler);
    },
  };
}

async function dynamicImport<T>(specifier: string): Promise<T> {
  const importer = new Function("specifier", "return import(specifier)") as (value: string) => Promise<T>;
  return importer(specifier);
}
