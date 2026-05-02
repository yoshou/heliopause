import type { WasmQuantizedWeightHandle } from "./wasm-kernels";

export type WasmThreadQuantizedType = WasmQuantizedWeightHandle["type"];
export type WasmThreadInputBuffer = ArrayBuffer | SharedArrayBuffer;

export type WasmThreadWorkerRequest =
  | {
      type: "prepareWeight";
      requestId: number;
      handleId: number;
      quantizedType: WasmThreadQuantizedType;
      weightBuffer: ArrayBuffer;
      inputSize: number;
      rowCount: number;
    }
  | {
      type: "prepareWeightFromBlob";
      requestId: number;
      handleId: number;
      quantizedType: WasmThreadQuantizedType;
      fileBlob: Blob;
      fileOffset: number;
      byteLength: number;
      inputSize: number;
      rowCount: number;
    }
  | {
      type: "matmul";
      requestId: number;
      handleId: number;
      inputBuffer: WasmThreadInputBuffer;
      inputSize: number;
      rowCount: number;
      columnCount: number;
    }
  | {
      type: "matmulBatch";
      requestId: number;
      handleIds: number[];
      inputBuffer: WasmThreadInputBuffer;
      inputSize: number;
      columnCount: number;
    }
  | {
      type: "releaseWeight";
      requestId: number;
      handleId: number;
    }
  | {
      type: "shutdown";
      requestId: number;
    };

export type WasmThreadWorkerResponse =
  | {
      type: "preparedWeight";
      requestId: number;
      residentBytes: number;
    }
  | {
      type: "matmulResult";
      requestId: number;
      outputBuffer: ArrayBuffer;
      outputLength: number;
    }
  | {
      type: "matmulBatchResult";
      requestId: number;
      outputs: Array<{
        outputBuffer: ArrayBuffer;
        outputLength: number;
      }>;
    }
  | {
      type: "releasedWeight";
      requestId: number;
    }
  | {
      type: "shutdownComplete";
      requestId: number;
    }
  | {
      type: "error";
      requestId: number;
      message: string;
    };
