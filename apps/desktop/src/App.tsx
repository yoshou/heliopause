import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  DEFAULT_QWEN35_SYSTEM_PROMPT,
  stripQwen35Thinking,
  type ChatMessage,
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
      assistantId: string;
      resolve: () => void;
      reject: (error: Error) => void;
    };

function App() {
  const [model, setModel] = useState<ModelState>({ status: "empty" });
  const [memoryProfile, setMemoryProfile] = useState<MemoryProfile>("auto");
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_QWEN35_SYSTEM_PROMPT);
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const nextRequestIdRef = useRef(1);
  const pendingRequestsRef = useRef(new Map<number, PendingRequest>());
  const generationRequestRef = useRef<{ requestId: number; worker: Worker } | null>(null);

  const canSubmit = useMemo(
    () => model.status === "ready" && prompt.trim().length > 0 && !isGenerating,
    [isGenerating, model.status, prompt],
  );

  useEffect(() => () => {
    const worker = workerRef.current;
    if (worker) {
      rejectWorkerRequests(worker, new Error("Application closed."));
      worker.terminate();
    }
  }, []);

  async function handleModelChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

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
        memoryProfile,
        await readSystemMemoryInfo(),
      );
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
    } finally {
      event.target.value = "";
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
    };
    const assistantId = createId("assistant");
    const nextMessages = [
      ...messages,
      userMessage,
      { id: assistantId, role: "assistant" as const, content: "" },
    ];

    setMessages(nextMessages);
    setPrompt("");
    setIsGenerating(true);

    const chatMessages: ChatMessage[] = [
      { role: "system", content: systemPrompt },
      ...messages.map((message) => ({
        role: message.role,
        content: message.content,
      })),
      { role: "user", content: trimmedPrompt },
    ];

    try {
      const worker = workerRef.current;
      if (!worker) {
        throw new Error("The model worker is not running. Reload the model.");
      }
      await generateInWorker(worker, assistantId, chatMessages, 256);
    } catch (error) {
      setMessages((currentMessages) =>
        currentMessages.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: `Generation failed: ${error instanceof Error ? error.message : String(error)}`,
              }
            : message,
        ),
      );
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
        memoryProfile: requestedMemoryProfile,
        memoryInfo,
      } satisfies EngineWorkerRequest);
    });
  }

  function generateInWorker(
    worker: Worker,
    assistantId: string,
    chatMessages: ChatMessage[],
    maxNewTokens: number,
  ): Promise<void> {
    const requestId = nextRequestIdRef.current;
    nextRequestIdRef.current += 1;
    generationRequestRef.current = { requestId, worker };

    return new Promise((resolve, reject) => {
      pendingRequestsRef.current.set(requestId, {
        type: "generate",
        worker,
        assistantId,
        resolve,
        reject,
      });
      worker.postMessage({
        type: "generate",
        requestId,
        messages: chatMessages,
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
        currentMessages.map((uiMessage) =>
          uiMessage.id === pending.assistantId && uiMessage.content.length === 0
            ? { ...uiMessage, content: "Generation stopped." }
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
          <p className="eyebrow">CPU / WASM mode</p>
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
          <ModelStatus model={model} />
        </section>

        <section className="system-panel">
          <label htmlFor="system-prompt">System prompt</label>
          <textarea
            id="system-prompt"
            value={systemPrompt}
            onChange={(event) => setSystemPrompt(event.target.value)}
            rows={8}
          />
        </section>
      </aside>

      <section className="chat-workspace" aria-label="Chat">
        <div className="message-panel" aria-live="polite">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Load a Qwen GGUF model</h2>
              <p>Choose a local model file, then start a private on-device chat.</p>
            </div>
          ) : (
            messages.map((message) => (
              <article
                className={`message message--${message.role}`}
                key={message.id}
              >
                <span>{message.role === "user" ? "You" : "Assistant"}</span>
                <p>{stripQwen35Thinking(message.content) || (message.role === "assistant" && isGenerating ? "Generating..." : "")}</p>
              </article>
            ))
          )}
        </div>

        <form className="prompt-form" onSubmit={handleSubmit}>
          <label htmlFor="prompt">Message</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder={model.status === "ready" ? "Ask the local model" : "Load a GGUF model first"}
            rows={4}
            disabled={model.status !== "ready" || isGenerating}
          />
          <div className="form-actions">
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
        <dt>Estimated weights</dt>
        <dd>{formatBytes(model.memoryProfile.estimatedWeightCacheBytes)}</dd>
      </div>
    </dl>
  );
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

function createId(prefix: string): string {
  return `${prefix}-${globalThis.crypto?.randomUUID?.() ?? Date.now().toString(36)}`;
}

export default App;
