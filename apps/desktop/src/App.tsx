import { ChangeEvent, FormEvent, useMemo, useRef, useState } from "react";
import {
  DEFAULT_QWEN35_SYSTEM_PROMPT,
  buildQwen35Tokenizer,
  createFileGgufTensorReader,
  createQwen35ChatSession,
  generateQwen35ChatCompletion,
  getGgufModelName,
  stripQwen35Thinking,
  type ChatMessage,
  type Qwen35ModelSession,
  type Qwen35Tokenizer,
} from "@heliopause/engine";

type UiMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type ModelState =
  | { status: "empty" }
  | { status: "loading"; fileName: string }
  | {
      status: "ready";
      fileName: string;
      modelName: string;
      contextLength: number;
      originalContextLength: number;
      session: Qwen35ModelSession;
      tokenizer: Qwen35Tokenizer;
    }
  | { status: "error"; fileName: string; message: string };

const CHAT_CONTEXT_LENGTH = 4096;

function App() {
  const [model, setModel] = useState<ModelState>({ status: "empty" });
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_QWEN35_SYSTEM_PROMPT);
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const canSubmit = useMemo(
    () => model.status === "ready" && prompt.trim().length > 0 && !isGenerating,
    [isGenerating, model.status, prompt],
  );

  async function handleModelChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setModel({ status: "loading", fileName: file.name });
    try {
      const tensorReader = await createFileGgufTensorReader(file);
      const session = createQwen35ChatSession(tensorReader, {
        maxContextLength: CHAT_CONTEXT_LENGTH,
        maxWeightCacheBytes: 256 * 1024 * 1024,
      });
      const tokenizer = buildQwen35Tokenizer(tensorReader.metadata);
      setModel({
        status: "ready",
        fileName: file.name,
        modelName: getGgufModelName(tensorReader),
        contextLength: Math.min(session.manifest.contextLength, CHAT_CONTEXT_LENGTH),
        originalContextLength: session.manifest.contextLength,
        session,
        tokenizer,
      });
    } catch (error) {
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

    const abortController = new AbortController();
    abortRef.current = abortController;

    const chatMessages: ChatMessage[] = [
      { role: "system", content: systemPrompt },
      ...messages.map((message) => ({
        role: message.role,
        content: message.content,
      })),
      { role: "user", content: trimmedPrompt },
    ];

    try {
      for await (const chunk of generateQwen35ChatCompletion(
        model.session,
        model.tokenizer,
        chatMessages,
        {
          maxNewTokens: 256,
          signal: abortController.signal,
        },
      )) {
        setMessages((currentMessages) =>
          currentMessages.map((message) =>
            message.id === assistantId
              ? { ...message, content: chunk.content }
              : message,
          ),
        );
      }
    } catch (error) {
      if (!abortController.signal.aborted) {
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
      }
    } finally {
      abortRef.current = null;
      setIsGenerating(false);
    }
  }

  function handleStop() {
    abortRef.current?.abort();
  }

  return (
    <main className="app-shell">
      <aside className="sidebar" aria-label="Model settings">
        <header>
          <p className="eyebrow">CPU / WASM mode</p>
          <h1>Heliopause</h1>
        </header>

        <section className="model-panel">
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
    </dl>
  );
}

function createId(prefix: string): string {
  return `${prefix}-${globalThis.crypto?.randomUUID?.() ?? Date.now().toString(36)}`;
}

export default App;
