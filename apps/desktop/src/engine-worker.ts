import {
  buildTokenizer,
  cloneInferenceState,
  createAudioSession,
  createFileGgufTensorReader,
  createChatSession,
  createReferenceProvider,
  createVisionSession,
  createWasmProvider,
  createWebGpuProvider,
  checkWasmSupport,
  checkWebGpuSupport,
  estimateWeightCacheBytes,
  generatePreparedAudioChatTurn,
  generatePreparedImageChatTurn,
  generateChatTurn,
  getGgufModelName,
  isAudioGguf,
  isVisionGguf,
  runAudioEncoder,
  prefillChatMessages,
  runAudioPreprocessor,
  runVisionPreprocessor,
  runVisionEncoder,
  type ExecutionProviderConfig,
  type AudioSession,
  type InferenceState,
  type ModelSession,
  type RunnerProvider,
  type Tokenizer,
  type VisionSession,
} from "@heliopause/engine";
import type {
  EngineWorkerRequest,
  EngineWorkerResponse,
  MemoryProfile,
  ResolvedRuntimeProfile,
  SystemMemoryInfo,
} from "./engine-worker-protocol";

const CHAT_CONTEXT_LENGTH = 4096;
const LOW_WEIGHT_CACHE_BYTES = 768 * 1024 * 1024;
const FULL_WEIGHT_CACHE_LIMIT_BYTES = 32 * 1024 * 1024 * 1024;
const FULL_WEIGHT_CACHE_HEADROOM = 1.25;

let session: ModelSession | undefined;
let visionSession: VisionSession | undefined;
let audioSession: AudioSession | undefined;
let tokenizer: Tokenizer | undefined;
let currentState: InferenceState | undefined;
let currentSystemPrompt: string | undefined;
let activeGeneration:
  | {
      requestId: number;
      abortController: AbortController;
      workingState?: InferenceState;
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
  audioSession?.dispose();
  session = undefined;
  visionSession = undefined;
  audioSession = undefined;
  tokenizer = undefined;
  currentState = undefined;
  currentSystemPrompt = undefined;

  const tensorReader = await createFileGgufTensorReader(request.file);
  let nextVisionSession: VisionSession | undefined;
  let nextAudioSession: AudioSession | undefined;
  const estimatedWeightCacheBytes = estimateWeightCacheBytes(tensorReader);
  const runtimeProfile = resolveRuntimeProfile(
    request.memoryProfile,
    estimatedWeightCacheBytes,
    request.memoryInfo,
  );
  const executionProviders = await resolveExecutionProviders(
    runtimeProfile,
  );
  const runnerProviders = instantiateRunnerProviderList(executionProviders);
  if (request.visionFile) {
    const visionReader = await createFileGgufTensorReader(request.visionFile);
    if (!isVisionGguf(visionReader.metadata) && !isAudioGguf(visionReader.metadata)) {
      throw new Error("Projector GGUF is not a supported multimodal projector.");
    }
    if (isVisionGguf(visionReader.metadata)) {
      nextVisionSession = createVisionSession(visionReader, {
        maxWeightCacheBytes: runtimeProfile.maxWeightCacheBytes,
        executionProviders,
        runnerProviders,
      });
    }
    if (isAudioGguf(visionReader.metadata)) {
      nextAudioSession = createAudioSession(visionReader, {
        maxWeightCacheBytes: runtimeProfile.maxWeightCacheBytes,
        executionProviders,
        runnerProviders,
      });
    }
  }
  const nextSession = createChatSession(tensorReader, {
    maxContextLength: CHAT_CONTEXT_LENGTH,
    maxWeightCacheBytes: runtimeProfile.maxWeightCacheBytes,
    executionProviders,
    runnerProviders,
  });
  const nextTokenizer = buildTokenizer(tensorReader.metadata);

  session = nextSession;
  visionSession = nextVisionSession;
  audioSession = nextAudioSession;
  tokenizer = nextTokenizer;

  workerScope.postMessage({
    type: "modelLoaded",
    requestId: request.requestId,
    model: {
      fileName: request.fileName,
      modelName: getGgufModelName(tensorReader),
      contextLength: Math.min(nextSession.manifest.contextLength, CHAT_CONTEXT_LENGTH),
      originalContextLength: nextSession.manifest.contextLength,
      runtimeProfile,
      visionFileName: request.visionFileName,
      supportsImages: Boolean(nextVisionSession),
      visionImageTokens: nextVisionSession
        ? {
            min: nextVisionSession.manifest.imageMinTokens,
            max: nextVisionSession.manifest.imageMaxTokens,
          }
        : undefined,
      supportsAudio: Boolean(nextAudioSession),
      audioFileName: nextAudioSession ? request.visionFileName : undefined,
      audioMaxSeconds: nextAudioSession?.manifest.maxSeconds,
    },
  });
}

function instantiateRunnerProviderList(
  executionProviders: readonly ExecutionProviderConfig[],
): readonly RunnerProvider[] {
  return executionProviders.map((provider) => {
    switch (provider.name) {
      case "webgpu":
        return createWebGpuProvider();
      case "wasm":
        return createWasmProvider();
      case "reference":
        return createReferenceProvider();
      default:
        throw new Error(`Unsupported execution provider: ${provider.name}`);
    }
  });
}

async function resolveExecutionProviders(
  runtimeProfile: ResolvedRuntimeProfile,
): Promise<ExecutionProviderConfig[]> {
  const providers: ExecutionProviderConfig[] = [];
  const webGpuSupport = await checkWebGpuSupport();
  if (webGpuSupport.available) {
    runtimeProfile.webGpuStatus = "enabled";
    providers.push({ name: "webgpu" });
  } else {
    runtimeProfile.webGpuStatus = "blocked";
    runtimeProfile.webGpuUnavailableReason = webGpuSupport.reason;
  }

  const wasmSupport = await checkWasmSupport();
  if (wasmSupport.available) {
    runtimeProfile.wasmStatus = "enabled";
    providers.push({
      name: "wasm",
      options: {
        projectionBatching: true,
        residentWeightCache: runtimeProfile.wasmResidentWeightCache,
        parallelResidentMatmul: runtimeProfile.wasmResidentWeightCache,
        parallelMatmulMinRows: 512,
        threadPoolSize: runtimeProfile.wasmResidentWeightCache ? "auto" : 1,
        ioPrefetch: runtimeProfile.wasmResidentWeightCache,
        ioPrefetchConcurrency: "auto",
        ioWorkerBlobRead: false,
      },
    });
  } else {
    runtimeProfile.wasmStatus = "unavailable";
    runtimeProfile.wasmUnavailableReason = wasmSupport.reason;
  }

  providers.push({ name: "reference" });
  runtimeProfile.executionProviders = providers.map((provider) => provider.name);
  return providers;
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
      : cloneInferenceState(currentState);
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
    if (request.audio) {
      if (!audioSession) {
        throw new Error("No audio encoder loaded.");
      }
      const features = await runAudioPreprocessor(audioSession, request.audio, {
        signal: abortController.signal,
      });
      if (abortController.signal.aborted) {
        throw new DOMException("Generation was aborted.", "AbortError");
      }
      const encoded = await runAudioEncoder(audioSession, features, {
        signal: abortController.signal,
      });
      await generatePreparedAudioChatTurn(
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
    } else if (request.image) {
      if (!visionSession) {
        throw new Error("No vision encoder loaded.");
      }
      const pixels = await runVisionPreprocessor(visionSession, request.image.file, {
        signal: abortController.signal,
      });
      const encoded = await runVisionEncoder(visionSession, pixels);
      await generatePreparedImageChatTurn(
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
      await generateChatTurn(
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
  await prefillChatMessages(
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

function resolveRuntimeProfile(
  requested: MemoryProfile,
  estimatedWeightCacheBytes: number,
  memoryInfo: SystemMemoryInfo | undefined,
): ResolvedRuntimeProfile {
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
    webGpuStatus: "blocked",
    executionProviders: [],
    wasmStatus: "unavailable",
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
