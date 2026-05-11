import {
  buildGemma4Manifest,
  buildGemma4Tokenizer,
  cloneGemma4InferenceState,
  createFileGgufTensorReader,
  createGemma4ChatSession,
  createGemma4VisionSession,
  estimateWeightCacheBytes,
  generateGemma4PreparedImageChatTurn,
  generateGemma4ChatTurn,
  getGgufModelName,
  isGemma4VisionGguf,
  planGemma4RunnerPlacement,
  preprocessGemma4VisionImageFile,
  prefillGemma4ChatMessages,
  runGemma4VisionEncoder,
  type ExecutionProviderConfig,
  type Gemma4InferenceState,
  type Gemma4ModelSession,
  type Gemma4Tokenizer,
  type Gemma4VisionSession,
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

let session: Gemma4ModelSession | undefined;
let visionSession: Gemma4VisionSession | undefined;
let tokenizer: Gemma4Tokenizer | undefined;
let currentState: Gemma4InferenceState | undefined;
let currentSystemPrompt: string | undefined;
let activeGeneration:
  | {
      requestId: number;
      abortController: AbortController;
      workingState?: Gemma4InferenceState;
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
  visionSession?.dispose();
  session = undefined;
  visionSession = undefined;
  tokenizer = undefined;
  currentState = undefined;
  currentSystemPrompt = undefined;

  const tensorReader = await createFileGgufTensorReader(request.file);
  const manifest = buildGemma4Manifest(tensorReader.metadata);
  let nextVisionSession: Gemma4VisionSession | undefined;
  const estimatedWeightCacheBytes = estimateWeightCacheBytes(tensorReader);
  const resolvedMemoryProfile = resolveMemoryProfile(
    request.memoryProfile,
    estimatedWeightCacheBytes,
    request.memoryInfo,
  );
  if (request.visionFile) {
    const visionReader = await createFileGgufTensorReader(request.visionFile);
    if (!isGemma4VisionGguf(visionReader.metadata)) {
      throw new Error("Vision encoder GGUF is not a Gemma4V projector.");
    }
    nextVisionSession = createGemma4VisionSession(visionReader, {
      maxWeightCacheBytes: resolvedMemoryProfile.maxWeightCacheBytes,
      executionProviders: [{
        name: "cpu",
        options: {
          projectionBatching: true,
          residentWeightCache: resolvedMemoryProfile.wasmResidentWeightCache,
          wasmKernels: true,
        },
      }],
    });
  }
  const executionProviders: ExecutionProviderConfig[] = [{
    name: "cpu",
    options: {
      projectionBatching: true,
      residentWeightCache: resolvedMemoryProfile.wasmResidentWeightCache,
      parallelResidentMatmul: resolvedMemoryProfile.wasmResidentWeightCache,
      parallelMatmulMinRows: 512,
      threadPoolSize: resolvedMemoryProfile.wasmResidentWeightCache ? "auto" : 1,
      ioPrefetch: resolvedMemoryProfile.wasmResidentWeightCache,
      ioPrefetchConcurrency: "auto",
      ioWorkerBlobRead: false,
    },
  }];

  if (resolvedMemoryProfile.resolved === "full" && !nextVisionSession) {
    const webGpuPlan = planGemma4RunnerPlacement(tensorReader.metadata, manifest, {
      mode: "enabled",
      contextLength: CHAT_CONTEXT_LENGTH,
    });
    if (webGpuPlan.status === "planned" && webGpuPlan.segmentStartLayer !== undefined) {
      resolvedMemoryProfile.webGpuStatus = "suffix-enabled";
      resolvedMemoryProfile.webGpuSegmentStartLayer = webGpuPlan.segmentStartLayer;
      resolvedMemoryProfile.webGpuSegmentLayerCount = webGpuPlan.gpuSegmentLayerCount;
      executionProviders.push({
        name: "webgpu",
        options: {
          segmentStartLayer: webGpuPlan.segmentStartLayer,
          memoryLimitBytes: webGpuPlan.memoryLimitBytes,
        },
      });
    } else {
      resolvedMemoryProfile.webGpuStatus = "blocked";
    }
  } else if (nextVisionSession) {
    resolvedMemoryProfile.webGpuStatus = "blocked";
  }

  const nextSession = createGemma4ChatSession(tensorReader, {
    maxContextLength: CHAT_CONTEXT_LENGTH,
    maxWeightCacheBytes: resolvedMemoryProfile.maxWeightCacheBytes,
    executionProviders,
  });
  const nextTokenizer = buildGemma4Tokenizer(tensorReader.metadata);

  session = nextSession;
  visionSession = nextVisionSession;
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
      visionFileName: request.visionFileName,
      supportsImages: Boolean(nextVisionSession),
      visionImageTokens: nextVisionSession
        ? {
            min: nextVisionSession.manifest.imageMinTokens,
            max: nextVisionSession.manifest.imageMaxTokens,
          }
        : undefined,
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
  const inferenceStartedAt = performance.now();

  try {
    await ensureChatState(request.systemPrompt, abortController.signal);
    if (!currentState) {
      throw new Error("Chat state was not initialized.");
    }

    const workingState = session.executionProvider("webgpu")
      ? currentState
      : cloneGemma4InferenceState(currentState);
    activeGeneration.workingState = workingState;

    const turnOptions = {
      maxNewTokens: request.maxNewTokens,
      signal: abortController.signal,
      onToken(chunk: { content: string }) {
        workerScope.postMessage({
          type: "generationChunk",
          requestId: request.requestId,
          content: chunk.content,
        });
      },
    };
    if (request.image) {
      if (!visionSession) {
        throw new Error("No vision encoder loaded.");
      }
      const pixels = await preprocessGemma4VisionImageFile(request.image.file, visionSession.manifest);
      const encoded = await runGemma4VisionEncoder(visionSession, pixels);
      await generateGemma4PreparedImageChatTurn(
        session,
        tokenizer,
        workingState,
        request.userContent,
        {
          hidden: encoded.hidden,
          tokenCount: encoded.tokenCount,
        },
        turnOptions,
      );
    } else {
      await generateGemma4ChatTurn(
        session,
        tokenizer,
        workingState,
        request.userContent,
        turnOptions,
      );
    }

    if (!abortController.signal.aborted) {
      currentState = workingState;
    }

    if (abortController.signal.aborted) {
      workerScope.postMessage({
        type: "generationCancelled",
        requestId: request.requestId,
      });
    } else {
      workerScope.postMessage({
        type: "generationDone",
        requestId: request.requestId,
        inferenceDurationMs: performance.now() - inferenceStartedAt,
      });
    }
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
  await prefillGemma4ChatMessages(
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
    ? requested !== "low"
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
    webGpuStatus: resolved === "full" ? "blocked" : "memory-profile-disabled",
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
