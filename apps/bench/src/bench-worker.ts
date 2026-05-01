import {
  readBenchEnvironment,
  runBrowserBench,
} from "@heliopause/engine";
import type { BenchWorkerRequest, BenchWorkerResponse } from "./bench-worker-protocol";

let activeRun: { requestId: number; abortController: AbortController } | undefined;

const workerScope = self as unknown as {
  postMessage(message: BenchWorkerResponse): void;
  onmessage: ((event: MessageEvent<BenchWorkerRequest>) => void) | null;
};

workerScope.onmessage = (event) => {
  void handleRequest(event.data);
};

async function handleRequest(request: BenchWorkerRequest): Promise<void> {
  if (request.type === "cancel") {
    if (activeRun?.requestId === request.requestId) {
      activeRun.abortController.abort();
    }
    return;
  }

  try {
    if (request.type === "environment") {
      workerScope.postMessage({
        type: "environment",
        requestId: request.requestId,
        environment: await readBenchEnvironment(),
      });
      return;
    }

    if (activeRun) {
      throw new Error("Benchmark is already running.");
    }
    const abortController = new AbortController();
    activeRun = { requestId: request.requestId, abortController };

    const report = await runBrowserBench({
      caseIds: request.caseIds,
      sizes: request.sizes,
      warmupIterations: request.warmupIterations,
      minimumMs: request.minimumMs,
      signal: abortController.signal,
      onResult(result) {
        workerScope.postMessage({
          type: "progress",
          requestId: request.requestId,
          result,
        });
      },
    });

    if (abortController.signal.aborted) {
      workerScope.postMessage({ type: "cancelled", requestId: request.requestId });
    } else {
      workerScope.postMessage({ type: "done", requestId: request.requestId, report });
    }
  } catch (error) {
    if (activeRun?.abortController.signal.aborted) {
      workerScope.postMessage({ type: "cancelled", requestId: request.requestId });
    } else {
      workerScope.postMessage({
        type: "error",
        requestId: request.requestId,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  } finally {
    if (activeRun?.requestId === request.requestId) {
      activeRun = undefined;
    }
  }
}
