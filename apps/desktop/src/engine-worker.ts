import {
  buildQwen35Tokenizer,
  cloneQwen35InferenceState,
  createFileGgufTensorReader,
  createQwen35ChatSession,
  estimateQwen35WeightCacheBytes,
  generateQwen35ChatTurn,
  getGgufModelName,
  prefillQwen35ChatMessages,
  type Qwen35InferenceState,
  type Qwen35ModelSession,
  type Qwen35Tokenizer,
} from "@heliopause/engine";
import type {
  EngineWorkerRequest,
  EngineWorkerResponse,
  MemoryProfile,
  ResolvedMemoryProfile,
  SystemMemoryInfo,
} from "./engine-worker-protocol";

const CHAT_CONTEXT_LENGTH = 4096;
const LOW_WEIGHT_CACHE_BYTES = 768 * 1024 * 1024;
const FULL_WEIGHT_CACHE_LIMIT_BYTES = 32 * 1024 * 1024 * 1024;
const FULL_WEIGHT_CACHE_HEADROOM = 1.25;

let session: Qwen35ModelSession | undefined;
let tokenizer: Qwen35Tokenizer | undefined;
let currentState: Qwen35InferenceState | undefined;
let currentSystemPrompt: string | undefined;
let activeGeneration:
  | {
      requestId: number;
      abortController: AbortController;
      workingState?: Qwen35InferenceState;
    }
  | undefined;

const workerScope = self as unknown as {
  postMessage(message: EngineWorkerResponse): void;
  onmessage: ((event: MessageEvent<EngineWorkerRequest>) => void) | null;
};

workerScope.onmessage = (event) => {
  void handleRequest(event.data);
};

async function handleRequest(request: EngineWorkerRequest): Promise<void> {
  try {
    if (request.type === "loadModel") {
      await handleLoadModel(request);
      return;
    }
    if (request.type === "generateTurn") {
      await handleGenerateTurn(request);
      return;
    }
    handleCancelGeneration(request.requestId);
  } catch (error) {
    postError(request.requestId, error);
  }
}

async function handleLoadModel(
  request: Extract<EngineWorkerRequest, { type: "loadModel" }>,
): Promise<void> {
  activeGeneration?.abortController.abort();
  activeGeneration = undefined;
  session = undefined;
  tokenizer = undefined;
  currentState = undefined;
  currentSystemPrompt = undefined;

  const tensorReader = await createFileGgufTensorReader(request.file);
  const estimatedWeightCacheBytes = estimateQwen35WeightCacheBytes(tensorReader);
  const resolvedMemoryProfile = resolveMemoryProfile(
    request.memoryProfile,
    estimatedWeightCacheBytes,
    request.memoryInfo,
  );
  const nextSession = createQwen35ChatSession(tensorReader, {
    maxContextLength: CHAT_CONTEXT_LENGTH,
    maxWeightCacheBytes: resolvedMemoryProfile.maxWeightCacheBytes,
    enableWasmWeightCache: resolvedMemoryProfile.wasmResidentWeightCache,
    enableWebGpu: true,
  });
  const nextTokenizer = buildQwen35Tokenizer(tensorReader.metadata);

  session = nextSession;
  tokenizer = nextTokenizer;

  workerScope.postMessage({
    type: "modelLoaded",
    requestId: request.requestId,
    model: {
      fileName: request.fileName,
      modelName: getGgufModelName(tensorReader),
      contextLength: Math.min(nextSession.manifest.contextLength, CHAT_CONTEXT_LENGTH),
      originalContextLength: nextSession.manifest.contextLength,
      memoryProfile: resolvedMemoryProfile,
    },
  });
}

async function handleGenerateTurn(
  request: Extract<EngineWorkerRequest, { type: "generateTurn" }>,
): Promise<void> {
  if (!session || !tokenizer) {
    throw new Error("No model loaded.");
  }
  if (activeGeneration) {
    throw new Error("Generation is already running.");
  }

  const abortController = new AbortController();
  activeGeneration = { requestId: request.requestId, abortController };

  try {
    await ensureChatState(request.systemPrompt, abortController.signal);
    if (!currentState) {
      throw new Error("Chat state was not initialized.");
    }

    const workingState = session.enableWebGpu
      ? currentState
      : cloneQwen35InferenceState(currentState);
    activeGeneration.workingState = workingState;

    await generateQwen35ChatTurn(
      session,
      tokenizer,
      workingState,
      request.userContent,
      {
        maxNewTokens: request.maxNewTokens,
        signal: abortController.signal,
        onToken(chunk) {
          workerScope.postMessage({
            type: "generationChunk",
            requestId: request.requestId,
            content: chunk.content,
          });
        },
      },
    );

    if (!abortController.signal.aborted) {
      currentState = workingState;
    }

    workerScope.postMessage({
      type: abortController.signal.aborted ? "generationCancelled" : "generationDone",
      requestId: request.requestId,
    });
  } catch (error) {
    if (!abortController.signal.aborted) {
      throw error;
    }
    workerScope.postMessage({
      type: "generationCancelled",
      requestId: request.requestId,
    });
  } finally {
    if (activeGeneration?.requestId === request.requestId) {
      activeGeneration = undefined;
    }
  }
}

async function ensureChatState(systemPrompt: string, signal?: AbortSignal): Promise<void> {
  if (currentState && currentSystemPrompt === systemPrompt) {
    return;
  }
  await resetChatState(systemPrompt, signal);
}

async function resetChatState(systemPrompt: string, signal?: AbortSignal): Promise<void> {
  if (!session || !tokenizer) {
    return;
  }
  const state = session.createInferenceState();
  await prefillQwen35ChatMessages(
    session,
    tokenizer,
    state,
    [{ role: "system", content: systemPrompt }],
    { signal },
  );
  currentState = state;
  currentSystemPrompt = systemPrompt;
}

function handleCancelGeneration(requestId: number): void {
  if (!activeGeneration || activeGeneration.requestId !== requestId) {
    return;
  }
  activeGeneration.abortController.abort();
}

function resolveMemoryProfile(
  requested: MemoryProfile,
  estimatedWeightCacheBytes: number,
  memoryInfo: SystemMemoryInfo | undefined,
): ResolvedMemoryProfile {
  const fullBytes = Math.min(
    FULL_WEIGHT_CACHE_LIMIT_BYTES,
    Math.ceil(estimatedWeightCacheBytes * FULL_WEIGHT_CACHE_HEADROOM),
  );
  const availableMemoryBytes = memoryInfo?.available_bytes && memoryInfo.available_bytes > 0
    ? memoryInfo.available_bytes
    : undefined;
  const hasFullMemory = availableMemoryBytes === undefined
    ? requested === "full"
    : availableMemoryBytes > fullBytes + 2 * 1024 * 1024 * 1024;
  const resolved = requested === "full" || (requested === "auto" && hasFullMemory)
    ? "full"
    : "low";

  return {
    requested,
    resolved,
    maxWeightCacheBytes: resolved === "full" ? fullBytes : LOW_WEIGHT_CACHE_BYTES,
    estimatedWeightCacheBytes,
    wasmResidentWeightCache: resolved === "full",
    availableMemoryBytes,
  };
}

function postError(requestId: number, error: unknown): void {
  workerScope.postMessage({
    type: "error",
    requestId,
    message: error instanceof Error ? error.message : String(error),
  });
}
