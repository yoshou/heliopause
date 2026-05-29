import {
  DEFAULT_MAX_THINKING_CHARS,
  DEFAULT_MAX_TOOL_STEPS,
  generateAgentTurn,
  type AgentToolCall,
} from "@heliopause/agent";
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
  disposeInferenceState,
  checkWasmSupport,
  checkWebGpuSupport,
  estimateWeightCacheBytes,
  generateChatTurn,
  generatePreparedAudioChatTurn,
  generatePreparedImageChatTurn,
  getGgufModelName,
  isAudioGguf,
  isVisionGguf,
  runAudioEncoder,
  prefillChatMessages,
  runAudioPreprocessor,
  runVisionPreprocessor,
  runVisionEncoder,
  type AudioSession,
  type InferenceState,
  type ModelSession,
  type MultimodalRunnerProvider,
  type Tokenizer,
  type VisionSession,
} from "@heliopause/engine";
import { createVirtualFileSystem, normalizeVirtualPath } from "@heliopause/sandbox";
import {
  buildAgentTools,
  executeDesktopAgentTool,
} from "./agent-sandbox-tools";
import type {
  EngineWorkerRequest,
  EngineWorkerResponse,
  MemoryProfile,
  ResolvedRuntimeProfile,
  SystemMemoryInfo,
  WorkerWebFetchToolResult,
  WorkerWebSearchToolResult,
} from "./engine-worker-protocol";
import { fetchBrowserTextResource } from "./web-fetch";

const CHAT_CONTEXT_LENGTH = 16_384;
const LOW_WEIGHT_CACHE_BYTES = 768 * 1024 * 1024;
const FULL_WEIGHT_CACHE_LIMIT_BYTES = 32 * 1024 * 1024 * 1024;
const FULL_WEIGHT_CACHE_HEADROOM = 1.25;

let session: ModelSession | undefined;
let visionSession: VisionSession | undefined;
let audioSession: AudioSession | undefined;
let tokenizer: Tokenizer | undefined;
let currentState: InferenceState | undefined;
let currentSystemPrompt: string | undefined;
const sandboxFs = createVirtualFileSystem();
let activeGeneration:
  | {
      requestId: number;
      abortController: AbortController;
      workingState?: InferenceState;
      pendingWebSearches: Map<
        string,
        {
          resolve: (result: WorkerWebSearchToolResult) => void;
          reject: (error: unknown) => void;
        }
      >;
      pendingWebFetches: Map<
        string,
        {
          resolve: (approved: boolean) => void;
          reject: (error: unknown) => void;
        }
      >;
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
    if (request.type === "resolveWebSearchConfirmation") {
      handleResolveWebSearchConfirmation(request);
      return;
    }
    if (request.type === "resolveWebFetchConfirmation") {
      handleResolveWebFetchConfirmation(request);
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
  const generationWasActive = activeGeneration !== undefined;
  const previousState = currentState;
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
  if (!generationWasActive) {
    disposeInferenceState(previousState);
  }
  sandboxFs.reset();

  const tensorReader = await createFileGgufTensorReader(request.file);
  let nextVisionSession: VisionSession | undefined;
  let nextAudioSession: AudioSession | undefined;
  const estimatedWeightCacheBytes = estimateWeightCacheBytes(tensorReader);
  const runtimeProfile = resolveRuntimeProfile(
    request.memoryProfile,
    estimatedWeightCacheBytes,
    request.memoryInfo,
  );
  const providers = await resolveRunnerProviders(
    runtimeProfile,
  );
  if (request.visionFile) {
    const visionReader = await createFileGgufTensorReader(request.visionFile);
    if (!isVisionGguf(visionReader.metadata) && !isAudioGguf(visionReader.metadata)) {
      throw new Error("Projector GGUF is not a supported multimodal projector.");
    }
    if (isVisionGguf(visionReader.metadata)) {
      nextVisionSession = createVisionSession(visionReader, {
        maxWeightCacheBytes: runtimeProfile.maxWeightCacheBytes,
        providers,
      });
    }
    if (isAudioGguf(visionReader.metadata)) {
      nextAudioSession = createAudioSession(visionReader, {
        maxWeightCacheBytes: runtimeProfile.maxWeightCacheBytes,
        providers,
      });
    }
  }
  const nextSession = createChatSession(tensorReader, {
    maxContextLength: CHAT_CONTEXT_LENGTH,
    maxWeightCacheBytes: runtimeProfile.maxWeightCacheBytes,
    providers,
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

async function resolveRunnerProviders(
  runtimeProfile: ResolvedRuntimeProfile,
): Promise<MultimodalRunnerProvider[]> {
  const providers: MultimodalRunnerProvider[] = [];
  const webGpuSupport = await checkWebGpuSupport();
  if (webGpuSupport.available) {
    runtimeProfile.webGpuStatus = "enabled";
    providers.push(createWebGpuProvider());
  } else {
    runtimeProfile.webGpuStatus = "blocked";
    runtimeProfile.webGpuUnavailableReason = webGpuSupport.reason;
  }

  const wasmSupport = await checkWasmSupport();
  if (wasmSupport.available) {
    runtimeProfile.wasmStatus = "enabled";
    providers.push(createWasmProvider({
      projectionBatching: true,
      residentWeightCache: runtimeProfile.wasmResidentWeightCache,
      parallelResidentMatmul: runtimeProfile.wasmResidentWeightCache,
      parallelMatmulMinRows: 512,
      threadPoolSize: runtimeProfile.wasmResidentWeightCache ? "auto" : 1,
      ioPrefetch: runtimeProfile.wasmResidentWeightCache,
      ioPrefetchConcurrency: "auto",
      ioWorkerBlobRead: false,
    }));
  } else {
    runtimeProfile.wasmStatus = "unavailable";
    runtimeProfile.wasmUnavailableReason = wasmSupport.reason;
  }

  providers.push(createReferenceProvider());
  runtimeProfile.providerNames = providers.map((provider) => provider.name);
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
  activeGeneration = {
    requestId: request.requestId,
    abortController,
    pendingWebSearches: new Map(),
    pendingWebFetches: new Map(),
  };
  const inferenceStartedAt = performance.now();

  try {
    await ensureChatState(request.systemPrompt, abortController.signal);
    if (!currentState) {
      throw new Error("Chat state was not initialized.");
    }

    const workingState = session.hasProvider("webgpu")
      ? currentState
      : cloneInferenceState(currentState);
    activeGeneration.workingState = workingState;
    let nextState = workingState;

    const turnOptions = {
      maxNewTokens: request.maxNewTokens,
      doSample: request.doSample,
      temperature: request.temperature,
      topP: request.topP,
      topK: request.topK,
      seed: request.seed,
      signal: abortController.signal,
      onToken(chunk: { content: string }) {
        workerScope.postMessage({
          type: "generationChunk",
          requestId: request.requestId,
          content: chunk.content,
        });
      },
    };
    const tools = buildAgentTools({ webSearchAvailable: request.webSearchAvailable, webFetchAvailable: true });
    const executeTool = (call: AgentToolCall, signal: AbortSignal) =>
      executeDesktopAgentTool(sandboxFs, call, signal, {
        executeWebSearch: executeWebSearchWithUserConfirmation,
        executeWebFetch: executeWebFetchWithUserConfirmation,
      });
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
      let isFirstAgentStep = true;
      const result = await generateAgentTurn(
        session,
        tokenizer,
        workingState,
        request.userContent,
        {
          ...turnOptions,
          tools,
          executeTool,
          maxToolSteps: DEFAULT_MAX_TOOL_STEPS,
          maxThinkingChars: DEFAULT_MAX_THINKING_CHARS,
          enableThinking: request.enableThinking,
          cloneState: session.hasProvider("webgpu") ? false : cloneInferenceState,
          onAgentEvent(event) {
            workerScope.postMessage({
              type: "agentEvent",
              requestId: request.requestId,
              event,
            });
          },
          chatTurnGenerator(nextSession, nextTokenizer, nextState, userContent, options) {
            if (!isFirstAgentStep) {
              return generateChatTurn(nextSession, nextTokenizer, nextState, userContent, options);
            }
            isFirstAgentStep = false;
            return generatePreparedAudioChatTurn(
              nextSession,
              nextTokenizer,
              nextState,
              userContent,
              {
                hidden: encoded.hidden,
                tokenCount: encoded.tokenCount,
              },
              options,
            );
          },
        },
      );
      nextState = result.state;
    } else if (request.image) {
      if (!visionSession) {
        throw new Error("No vision encoder loaded.");
      }
      const pixels = await runVisionPreprocessor(visionSession, request.image.file, {
        signal: abortController.signal,
      });
      const encoded = await runVisionEncoder(visionSession, pixels);
      const result = await generatePreparedImageChatTurn(
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
      nextState = result.state;
    } else {
      const result = await generateAgentTurn(
        session,
        tokenizer,
        workingState,
        request.userContent,
        {
          ...turnOptions,
          tools,
          executeTool,
          maxToolSteps: DEFAULT_MAX_TOOL_STEPS,
          maxThinkingChars: DEFAULT_MAX_THINKING_CHARS,
          enableThinking: request.enableThinking,
          cloneState: session.hasProvider("webgpu") ? false : cloneInferenceState,
          onAgentEvent(event) {
            workerScope.postMessage({
              type: "agentEvent",
              requestId: request.requestId,
              event,
            });
          },
        },
      );
      nextState = result.state;
    }

    if (!abortController.signal.aborted) {
      currentState = nextState;
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
  const previousState = currentState;
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
  disposeInferenceState(previousState);
}

function handleCancelGeneration(requestId: number): void {
  if (!activeGeneration || activeGeneration.requestId !== requestId) {
    return;
  }
  activeGeneration.abortController.abort();
}

async function executeWebSearchWithUserConfirmation(
  call: AgentToolCall,
  signal: AbortSignal,
): Promise<WorkerWebSearchToolResult> {
  const args = requireWebSearchArguments(call.arguments);
  const maxResults = normalizeWebSearchMaxResults(args.max_results);
  const generation = activeGeneration;
  if (!generation) {
    return {
      callId: call.id,
      ok: false,
      error: {
        code: "web_search_unavailable",
        message: "No active generation can confirm web_search.",
      },
    };
  }

  return new Promise((resolve, reject) => {
    const abort = () => {
      generation.pendingWebSearches.delete(call.id);
      reject(new DOMException("web_search was aborted.", "AbortError"));
    };
    if (signal.aborted) {
      abort();
      return;
    }

    generation.pendingWebSearches.set(call.id, { resolve, reject });
    signal.addEventListener("abort", abort, { once: true });
    workerScope.postMessage({
      type: "webSearchConfirmationRequested",
      requestId: generation.requestId,
      callId: call.id,
      query: args.query,
      maxResults,
    });
  });
}

function handleResolveWebSearchConfirmation(
  request: Extract<EngineWorkerRequest, { type: "resolveWebSearchConfirmation" }>,
): void {
  if (!activeGeneration || activeGeneration.requestId !== request.requestId) {
    return;
  }
  const pending = activeGeneration.pendingWebSearches.get(request.callId);
  if (!pending) {
    return;
  }
  activeGeneration.pendingWebSearches.delete(request.callId);

  if (!request.approved) {
    pending.resolve({
      callId: request.callId,
      ok: false,
      error: {
        code: "user_denied",
        message: "The user declined the web search.",
      },
    });
    return;
  }

  pending.resolve(request.result ?? {
    callId: request.callId,
    ok: false,
    error: {
      code: "web_search_unavailable",
      message: "The approved web search did not return a result.",
    },
  });
}

async function executeWebFetchWithUserConfirmation(
  call: AgentToolCall,
  signal: AbortSignal,
): Promise<WorkerWebFetchToolResult> {
  const args = requireWebFetchArguments(call.arguments);
  const generation = activeGeneration;
  if (!generation) {
    return {
      callId: call.id,
      ok: false,
      error: {
        code: "web_fetch_unavailable",
        message: "No active generation can confirm web_fetch.",
      },
    };
  }

  const approved = await new Promise<boolean>((resolve, reject) => {
    const abort = () => {
      generation.pendingWebFetches.delete(call.id);
      reject(new DOMException("web_fetch was aborted.", "AbortError"));
    };
    if (signal.aborted) {
      abort();
      return;
    }

    generation.pendingWebFetches.set(call.id, { resolve, reject });
    signal.addEventListener("abort", abort, { once: true });
    workerScope.postMessage({
      type: "webFetchConfirmationRequested",
      requestId: generation.requestId,
      callId: call.id,
      url: args.url,
      path: args.path,
    });
  });

  if (!approved) {
    return {
      callId: call.id,
      ok: false,
      error: {
        code: "user_denied",
        message: "The user declined the web fetch.",
      },
    };
  }

  try {
    const result = await fetchBrowserTextResource(args, { signal });
    sandboxFs.writeText(result.path, result.content);
    const { content: _content, ...content } = result;
    return {
      callId: call.id,
      ok: true,
      content,
    };
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    return {
      callId: call.id,
      ok: false,
      error: normalizeWorkerToolError(error, "web_fetch_failed", "web_fetch failed."),
    };
  }
}

function handleResolveWebFetchConfirmation(
  request: Extract<EngineWorkerRequest, { type: "resolveWebFetchConfirmation" }>,
): void {
  if (!activeGeneration || activeGeneration.requestId !== request.requestId) {
    return;
  }
  const pending = activeGeneration.pendingWebFetches.get(request.callId);
  if (!pending) {
    return;
  }
  activeGeneration.pendingWebFetches.delete(request.callId);
  pending.resolve(request.approved);
}

function requireWebSearchArguments(value: unknown): { query: string; max_results?: unknown } {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw {
      code: "invalid_arguments",
      message: "web_search arguments must be an object.",
    };
  }
  const args = value as Record<string, unknown>;
  if (typeof args.query !== "string" || args.query.trim().length === 0) {
    throw {
      code: "invalid_arguments",
      message: "web_search query must be a non-empty string.",
    };
  }
  return {
    query: args.query.trim(),
    max_results: args.max_results,
  };
}

function requireWebFetchArguments(value: unknown): { url: string; path: string } {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw {
      code: "invalid_arguments",
      message: "web_fetch arguments must be an object.",
    };
  }
  const args = value as Record<string, unknown>;
  if (typeof args.url !== "string" || args.url.trim().length === 0) {
    throw {
      code: "invalid_arguments",
      message: "web_fetch url must be a non-empty string.",
    };
  }
  if (typeof args.path !== "string" || args.path.trim().length === 0) {
    throw {
      code: "invalid_arguments",
      message: "web_fetch path must be a non-empty string.",
    };
  }
  return {
    url: args.url.trim(),
    path: normalizeVirtualPath(args.path.trim()),
  };
}

function normalizeWebSearchMaxResults(value: unknown): number {
  if (value === undefined) {
    return 5;
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    return 5;
  }
  return Math.min(5, Math.max(1, value));
}

function normalizeWorkerToolError(
  error: unknown,
  fallbackCode: string,
  fallbackMessage: string,
): { code: string; message: string; retryable?: boolean } {
  if (typeof error === "object" && error !== null) {
    const record = error as Record<string, unknown>;
    const code = typeof record.code === "string" ? record.code : fallbackCode;
    const message = typeof record.message === "string" ? record.message : fallbackMessage;
    const retryable = typeof record.retryable === "boolean" ? record.retryable : undefined;
    return retryable === undefined ? { code, message } : { code, message, retryable };
  }
  if (error instanceof Error) {
    return {
      code: fallbackCode,
      message: error.message,
    };
  }
  return {
    code: fallbackCode,
    message: fallbackMessage,
  };
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
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
    providerNames: [],
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
