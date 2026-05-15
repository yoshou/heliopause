import {
  buildModelManifest,
  buildTokenizer,
  cloneInferenceState,
  createAudioSession,
  createFileGgufTensorReader,
  createChatSession,
  createVisionSession,
  estimateWeightCacheBytes,
  generatePreparedAudioChatTurn,
  generatePreparedImageChatTurn,
  generateChatTurn,
  getGgufModelName,
  isAudioGguf,
  isVisionGguf,
  preprocessAudioPcm,
  planRunnerPlacement,
  runAudioEncoder,
  preprocessVisionImageFile,
  prefillChatMessages,
  runVisionEncoder,
  type ExecutionProviderConfig,
  type AudioSession,
  type InferenceState,
  type ModelSession,
  type Tokenizer,
  type VisionSession,
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
  const manifest = buildModelManifest(tensorReader.metadata);
  let nextVisionSession: VisionSession | undefined;
  let nextAudioSession: AudioSession | undefined;
  const estimatedWeightCacheBytes = estimateWeightCacheBytes(tensorReader);
  const resolvedMemoryProfile = resolveMemoryProfile(
    request.memoryProfile,
    estimatedWeightCacheBytes,
    request.memoryInfo,
  );
  if (request.visionFile) {
    const visionReader = await createFileGgufTensorReader(request.visionFile);
    if (!isVisionGguf(visionReader.metadata) && !isAudioGguf(visionReader.metadata)) {
      throw new Error("Projector GGUF is not a supported multimodal projector.");
    }
    if (isVisionGguf(visionReader.metadata)) {
      nextVisionSession = createVisionSession(visionReader, {
        maxWeightCacheBytes: resolvedMemoryProfile.maxWeightCacheBytes,
        executionProviders: [
          {
            name: "webgpu",
          },
          {
            name: "cpu",
            options: {
              projectionBatching: true,
              residentWeightCache: resolvedMemoryProfile.wasmResidentWeightCache,
              wasmKernels: true,
            },
          },
        ],
      });
    }
    if (isAudioGguf(visionReader.metadata)) {
      nextAudioSession = createAudioSession(visionReader, {
        maxWeightCacheBytes: resolvedMemoryProfile.maxWeightCacheBytes,
        executionProviders: [
          {
            name: "cpu",
            options: {
              projectionBatching: true,
              residentWeightCache: resolvedMemoryProfile.wasmResidentWeightCache,
              wasmKernels: true,
            },
          },
        ],
      });
    }
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

  if (resolvedMemoryProfile.resolved === "full") {
    const webGpuPlan = planRunnerPlacement(tensorReader.metadata, manifest, {
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
  }

  const nextSession = createChatSession(tensorReader, {
    maxContextLength: CHAT_CONTEXT_LENGTH,
    maxWeightCacheBytes: resolvedMemoryProfile.maxWeightCacheBytes,
    executionProviders,
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
      memoryProfile: resolvedMemoryProfile,
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
      const features = preprocessAudioPcm(request.audio, audioSession.manifest);
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
      const pixels = await preprocessVisionImageFile(request.image.file, visionSession.manifest);
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
