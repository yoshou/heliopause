import { ChangeEvent, ClipboardEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  DEFAULT_GEMMA4_SYSTEM_PROMPT,
  stripGemma4Thinking,
} from "@heliopause/engine";
import type {
  EngineWorkerRequest,
  EngineWorkerResponse,
  MemoryProfile,
  SystemMemoryInfo,
  WorkerModelInfo,
} from "./engine-worker-protocol";

type UiMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  image?: UiImageAttachment;
  inferenceDurationMs?: number;
};

type UiImageAttachment = {
  file: File;
  fileName: string;
  url: string;
};

type ModelState =
  | { status: "empty" }
  | { status: "loading"; fileName: string }
  | ({ status: "ready" } & WorkerModelInfo)
  | { status: "error"; fileName: string; message: string };

type PendingRequest =
  | {
      type: "load";
      worker: Worker;
      resolve: (model: WorkerModelInfo) => void;
      reject: (error: Error) => void;
    }
  | {
      type: "generate";
      worker: Worker;
      userId: string;
      assistantId: string;
      userContent: string;
      image?: UiImageAttachment;
      resolve: () => void;
      reject: (error: Error) => void;
    };

function App() {
  const [model, setModel] = useState<ModelState>({ status: "empty" });
  const [memoryProfile, setMemoryProfile] = useState<MemoryProfile>("auto");
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_GEMMA4_SYSTEM_PROMPT);
  const [prompt, setPrompt] = useState("");
  const [modelFile, setModelFile] = useState<File | undefined>();
  const [visionFile, setVisionFile] = useState<File | undefined>();
  const [imageAttachment, setImageAttachment] = useState<UiImageAttachment | undefined>();
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationError, setGenerationError] = useState<string | undefined>();
  const workerRef = useRef<Worker | null>(null);
  const nextRequestIdRef = useRef(1);
  const pendingRequestsRef = useRef(new Map<number, PendingRequest>());
  const generationRequestRef = useRef<{ requestId: number; worker: Worker } | null>(null);

  const canSubmit = useMemo(
    () =>
      model.status === "ready" &&
      prompt.trim().length > 0 &&
      !isGenerating &&
      (!imageAttachment || model.supportsImages),
    [imageAttachment, isGenerating, model, prompt],
  );

  useEffect(() => () => {
    const worker = workerRef.current;
    if (worker) {
      rejectWorkerRequests(worker, new Error("Application closed."));
      worker.terminate();
    }
    if (imageAttachment) {
      URL.revokeObjectURL(imageAttachment.url);
    }
  }, []);

  async function handleModelChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setModelFile(file);
    await loadSelectedModel(file, visionFile);
    event.target.value = "";
  }

  async function handleVisionModelChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setVisionFile(file);
    if (modelFile) {
      await loadSelectedModel(modelFile, file);
    }
    event.target.value = "";
  }

  async function loadSelectedModel(file: File, nextVisionFile: File | undefined) {
    const worker = createEngineWorker();
    const previousWorker = workerRef.current;
    if (previousWorker) {
      rejectWorkerRequests(previousWorker, new Error("Model was replaced."));
      previousWorker.terminate();
    }
    workerRef.current = worker;
    generationRequestRef.current = null;

    setModel({ status: "loading", fileName: file.name });
    try {
      const modelInfo = await loadModelInWorker(
        worker,
        file,
        file.name,
        nextVisionFile,
        nextVisionFile?.name,
        memoryProfile,
        await readSystemMemoryInfo(),
      );
      setMessages([]);
      setModel({
        status: "ready",
        ...modelInfo,
      });
    } catch (error) {
      if (workerRef.current === worker) {
        workerRef.current = null;
        worker.terminate();
      }
      setModel({
        status: "error",
        fileName: file.name,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (model.status !== "ready" || isGenerating) {
      return;
    }

    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) {
      return;
    }

    const userMessage: UiMessage = {
      id: createId("user"),
      role: "user",
      content: trimmedPrompt,
      image: imageAttachment,
    };
    const assistantId = createId("assistant");
    const nextMessages = [
      ...messages,
      userMessage,
      { id: assistantId, role: "assistant" as const, content: "" },
    ];

    setMessages(nextMessages);
    setPrompt("");
    setImageAttachment(undefined);
    setIsGenerating(true);
    setGenerationError(undefined);

    try {
      const worker = workerRef.current;
      if (!worker) {
        throw new Error("The model worker is not running. Reload the model.");
      }
      await generateTurnInWorker(worker, userMessage.id, assistantId, trimmedPrompt, imageAttachment, 256);
    } catch (error) {
      setGenerationError(error instanceof Error ? error.message : String(error));
      setMessages((currentMessages) =>
        currentMessages.filter((message) =>
          message.id !== userMessage.id && message.id !== assistantId
        ),
      );
      setPrompt((currentPrompt) => currentPrompt.length === 0 ? trimmedPrompt : currentPrompt);
      setImageAttachment(imageAttachment);
    } finally {
      generationRequestRef.current = null;
      setIsGenerating(false);
    }
  }

  function handleStop() {
    const activeGeneration = generationRequestRef.current;
    if (!activeGeneration) {
      return;
    }
    activeGeneration.worker.postMessage({
      type: "cancelGeneration",
      requestId: activeGeneration.requestId,
    } satisfies EngineWorkerRequest);
  }

  function handleImageFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      setNextImageAttachment(file);
    }
    event.target.value = "";
  }

  function handlePromptPaste(event: ClipboardEvent<HTMLTextAreaElement>) {
    const file = Array.from(event.clipboardData.files).find((item) => item.type.startsWith("image/"));
    if (file) {
      setNextImageAttachment(file);
    }
  }

  function clearImageAttachment() {
    if (imageAttachment) {
      URL.revokeObjectURL(imageAttachment.url);
    }
    setImageAttachment(undefined);
  }

  function setNextImageAttachment(file: File) {
    if (imageAttachment) {
      URL.revokeObjectURL(imageAttachment.url);
    }
    setImageAttachment({
      file,
      fileName: file.name || "pasted-image",
      url: URL.createObjectURL(file),
    });
    setGenerationError(undefined);
  }

  function createEngineWorker(): Worker {
    const worker = new Worker(new URL("./engine-worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (event: MessageEvent<EngineWorkerResponse>) => {
      handleWorkerMessage(worker, event.data);
    };
    worker.onerror = (event) => {
      rejectWorkerRequests(
        worker,
        new Error(event.message || "The model worker failed."),
      );
    };
    return worker;
  }

  function loadModelInWorker(
    worker: Worker,
    file: File,
    fileName: string,
    visionModelFile: File | undefined,
    visionFileName: string | undefined,
    requestedMemoryProfile: MemoryProfile,
    memoryInfo: SystemMemoryInfo | undefined,
  ): Promise<WorkerModelInfo> {
    const requestId = nextRequestIdRef.current;
    nextRequestIdRef.current += 1;

    return new Promise((resolve, reject) => {
      pendingRequestsRef.current.set(requestId, {
        type: "load",
        worker,
        resolve,
        reject,
      });
      worker.postMessage({
        type: "loadModel",
        requestId,
        file,
        fileName,
        visionFile: visionModelFile,
        visionFileName,
        memoryProfile: requestedMemoryProfile,
        memoryInfo,
      } satisfies EngineWorkerRequest);
    });
  }

  function generateTurnInWorker(
    worker: Worker,
    userId: string,
    assistantId: string,
    userContent: string,
    image: UiImageAttachment | undefined,
    maxNewTokens: number,
  ): Promise<void> {
    const requestId = nextRequestIdRef.current;
    nextRequestIdRef.current += 1;
    generationRequestRef.current = { requestId, worker };

    return new Promise((resolve, reject) => {
      pendingRequestsRef.current.set(requestId, {
        type: "generate",
        worker,
        userId,
        assistantId,
        userContent,
        image,
        resolve,
        reject,
      });
      worker.postMessage({
        type: "generateTurn",
        requestId,
        systemPrompt,
        userContent,
        image: image
          ? {
              file: image.file,
              fileName: image.fileName,
            }
          : undefined,
        maxNewTokens,
      } satisfies EngineWorkerRequest);
    });
  }

  function handleWorkerMessage(worker: Worker, message: EngineWorkerResponse) {
    const pending = pendingRequestsRef.current.get(message.requestId);
    if (!pending || pending.worker !== worker) {
      return;
    }

    if (message.type === "error") {
      pendingRequestsRef.current.delete(message.requestId);
      pending.reject(new Error(message.message));
      return;
    }

    if (message.type === "modelLoaded") {
      if (pending.type !== "load") {
        return;
      }
      pendingRequestsRef.current.delete(message.requestId);
      pending.resolve(message.model);
      return;
    }

    if (pending.type !== "generate") {
      return;
    }

    if (message.type === "generationChunk") {
      setMessages((currentMessages) =>
        currentMessages.map((uiMessage) =>
          uiMessage.id === pending.assistantId
            ? { ...uiMessage, content: message.content }
            : uiMessage,
        ),
      );
      return;
    }

    if (message.type === "generationCancelled") {
      setMessages((currentMessages) =>
        currentMessages.filter((uiMessage) =>
          uiMessage.id !== pending.userId && uiMessage.id !== pending.assistantId
        ),
      );
      setPrompt((currentPrompt) =>
        currentPrompt.length === 0 ? pending.userContent : currentPrompt
      );
      setImageAttachment(pending.image);
    }

    if (message.type === "generationDone") {
      setMessages((currentMessages) =>
        currentMessages.map((uiMessage) =>
          uiMessage.id === pending.assistantId
            ? { ...uiMessage, inferenceDurationMs: message.inferenceDurationMs }
            : uiMessage,
        ),
      );
    }

    pendingRequestsRef.current.delete(message.requestId);
    pending.resolve();
  }

  function rejectWorkerRequests(worker: Worker, error: Error) {
    for (const [requestId, pending] of pendingRequestsRef.current) {
      if (pending.worker !== worker) {
        continue;
      }
      pendingRequestsRef.current.delete(requestId);
      pending.reject(error);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar" aria-label="Model settings">
        <header>
          <h1>Heliopause</h1>
        </header>

        <section className="model-panel">
          <label className="field-label" htmlFor="memory-profile">Weight cache</label>
          <select
            id="memory-profile"
            value={memoryProfile}
            onChange={(event) => setMemoryProfile(event.target.value as MemoryProfile)}
            disabled={model.status === "loading" || isGenerating}
          >
            <option value="auto">Auto</option>
            <option value="full">Full</option>
            <option value="low">Low</option>
          </select>
          <label className="file-picker">
            <span>Load GGUF model</span>
            <input type="file" accept=".gguf" onChange={handleModelChange} />
          </label>
          <label className="file-picker">
            <span>Vision encoder GGUF</span>
            <input type="file" accept=".gguf" onChange={handleVisionModelChange} />
          </label>
          <ModelStatus model={model} />
        </section>

        <section className="system-panel">
          <label htmlFor="system-prompt">System prompt</label>
          <textarea
            id="system-prompt"
            value={systemPrompt}
            onChange={(event) => {
              setSystemPrompt(event.target.value);
              setMessages([]);
            }}
            rows={8}
            disabled={isGenerating}
          />
        </section>
      </aside>

      <section className="chat-workspace" aria-label="Chat">
        <div className="message-panel" aria-live="polite">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Load a Gemma GGUF model</h2>
              <p>Choose a local model file, then start a private on-device chat.</p>
            </div>
          ) : (
            messages.map((message) => (
              <MessageBubble
                isGenerating={isGenerating}
                key={message.id}
                message={message}
              />
            ))
          )}
        </div>

        <form className="prompt-form" onSubmit={handleSubmit}>
          <label htmlFor="prompt">Message</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(event) => {
              setPrompt(event.target.value);
              setGenerationError(undefined);
            }}
            onPaste={handlePromptPaste}
            placeholder={model.status === "ready" ? "Ask the local model" : "Load a GGUF model first"}
            rows={4}
            disabled={model.status !== "ready" || isGenerating}
          />
          {imageAttachment ? (
            <div className="attachment-preview">
              <img src={imageAttachment.url} alt="" />
              <span>{imageAttachment.fileName}</span>
              <button type="button" onClick={clearImageAttachment} disabled={isGenerating}>
                Remove
              </button>
            </div>
          ) : null}
          {generationError ? (
            <p className="generation-error">{generationError}</p>
          ) : null}
          <div className="form-actions">
            <label className="image-button">
              Image
              <input
                type="file"
                accept="image/*"
                onChange={handleImageFileChange}
                disabled={model.status !== "ready" || isGenerating || (model.status === "ready" && !model.supportsImages)}
              />
            </label>
            <p>{model.status === "ready" ? `${prompt.trim().length} characters` : "No model loaded"}</p>
            {isGenerating ? (
              <button type="button" className="secondary-button" onClick={handleStop}>
                Stop
              </button>
            ) : (
              <button type="submit" disabled={!canSubmit}>
                Send
              </button>
            )}
          </div>
        </form>
      </section>
    </main>
  );
}

function MessageBubble(
  { isGenerating, message }: { isGenerating: boolean; message: UiMessage },
) {
  const visibleContent = stripGemma4Thinking(message.content);
  const placeholder = message.role === "assistant" && isGenerating ? "Generating..." : "";
  return (
    <article className={`message message--${message.role}`}>
      <span>{message.role === "user" ? "You" : "Assistant"}</span>
      {message.image ? (
        <img className="message-image" src={message.image.url} alt="" />
      ) : null}
      <p>{visibleContent || placeholder}</p>
      {message.role === "assistant" && message.inferenceDurationMs !== undefined ? (
        <footer className="message-meta">
          {formatDuration(message.inferenceDurationMs)}
        </footer>
      ) : null}
    </article>
  );
}

function ModelStatus({ model }: { model: ModelState }) {
  if (model.status === "empty") {
    return <p className="model-status">No model loaded.</p>;
  }
  if (model.status === "loading") {
    return <p className="model-status">Loading {model.fileName}...</p>;
  }
  if (model.status === "error") {
    return (
      <p className="model-status model-status--error">
        {model.fileName}: {model.message}
      </p>
    );
  }
  return (
    <dl className="model-details">
      <div>
        <dt>Model</dt>
        <dd>{model.modelName}</dd>
      </div>
      <div>
        <dt>File</dt>
        <dd>{model.fileName}</dd>
      </div>
      <div>
        <dt>Context</dt>
        <dd>
          {model.contextLength.toLocaleString()} chat tokens
          {model.originalContextLength > model.contextLength
            ? ` (${model.originalContextLength.toLocaleString()} model max)`
            : ""}
        </dd>
      </div>
      <div>
        <dt>Weight cache</dt>
        <dd>
          {model.memoryProfile.resolved} / {formatBytes(model.memoryProfile.maxWeightCacheBytes)}
          {model.memoryProfile.requested === "auto" ? " (auto)" : ""}
          {model.memoryProfile.wasmResidentWeightCache ? " / WASM resident" : ""}
        </dd>
      </div>
      <div>
        <dt>Execution</dt>
        <dd>{executionLabel(model.memoryProfile)}</dd>
      </div>
      <div>
        <dt>Vision</dt>
        <dd>
          {model.supportsImages
            ? `${model.visionFileName ?? "Loaded"} (${model.visionImageTokens?.min ?? 0}-${model.visionImageTokens?.max ?? 0} image tokens)`
            : "Not loaded"}
        </dd>
      </div>
      <div>
        <dt>Estimated weights</dt>
        <dd>{formatBytes(model.memoryProfile.estimatedWeightCacheBytes)}</dd>
      </div>
    </dl>
  );
}

function executionLabel(memoryProfile: WorkerModelInfo["memoryProfile"]): string {
  if (memoryProfile.webGpuStatus === "suffix-enabled") {
    const start = memoryProfile.webGpuSegmentStartLayer;
    const count = memoryProfile.webGpuSegmentLayerCount;
    return start === undefined || count === undefined
      ? "CPU/WASM + WebGPU suffix"
      : `CPU/WASM + WebGPU suffix (${start}-${start + count - 1})`;
  }
  if (memoryProfile.webGpuStatus === "blocked") {
    return "CPU/WASM (WebGPU suffix blocked)";
  }
  if (memoryProfile.webGpuStatus === "memory-profile-disabled") {
    return "CPU/WASM";
  }
  return "CPU/WASM";
}

async function readSystemMemoryInfo(): Promise<SystemMemoryInfo | undefined> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const info = await invoke<SystemMemoryInfo>("system_memory_info");
    if (info.total_bytes <= 0 && info.available_bytes <= 0) {
      return undefined;
    }
    return info;
  } catch {
    return undefined;
  }
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  }
  return `${bytes.toLocaleString()} bytes`;
}

function formatDuration(durationMs: number): string {
  if (durationMs < 1000) {
    return `${Math.round(durationMs)} ms`;
  }
  if (durationMs < 60_000) {
    return `${(durationMs / 1000).toFixed(1)} s`;
  }
  const minutes = Math.floor(durationMs / 60_000);
  const seconds = Math.round((durationMs % 60_000) / 1000);
  return `${minutes} min ${seconds} s`;
}

function createId(prefix: string): string {
  return `${prefix}-${globalThis.crypto?.randomUUID?.() ?? Date.now().toString(36)}`;
}

export default App;
