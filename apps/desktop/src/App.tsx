import { FormEvent, useEffect, useMemo, useState } from "react";
import { WebGpuSupport, checkWebGpuSupport } from "@heliopause/engine";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type WebGpuStatus =
  | { state: "checking"; label: string; detail: string }
  | { state: "ready"; label: string; detail: string }
  | { state: "unavailable"; label: string; detail: string };

const initialWebGpuStatus: WebGpuStatus = {
  state: "checking",
  label: "Checking WebGPU",
  detail: "Looking for a GPU adapter in the current WebView.",
};

function App() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [webGpuStatus, setWebGpuStatus] =
    useState<WebGpuStatus>(initialWebGpuStatus);

  const canSubmit = useMemo(() => prompt.trim().length > 0, [prompt]);

  useEffect(() => {
    let cancelled = false;

    checkWebGpuSupport().then((support) => {
      if (!cancelled) {
        setWebGpuStatus(formatWebGpuStatus(support));
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) {
      return;
    }

    const submittedAt = Date.now().toString();
    setMessages((currentMessages) => [
      ...currentMessages,
      {
        id: `user-${submittedAt}`,
        role: "user",
        content: trimmedPrompt,
      },
      {
        id: `assistant-${submittedAt}`,
        role: "assistant",
        content:
          "The WebGPU inference engine is not connected yet. The next implementation step will send this prompt to a local LLM.",
      },
    ]);
    setPrompt("");
  }

  return (
    <main className="app-shell">
      <section className="workspace" aria-labelledby="app-title">
        <header className="app-header">
          <div>
            <p className="eyebrow">Local LLM Workspace</p>
            <h1 id="app-title">Heliopause</h1>
          </div>
          <div className={`status-badge status-badge--${webGpuStatus.state}`}>
            <span aria-hidden="true" />
            <div>
              <strong>{webGpuStatus.label}</strong>
              <small>{webGpuStatus.detail}</small>
            </div>
          </div>
        </header>

        <div className="message-panel" aria-live="polite">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Prompt</h2>
              <p>
                Your text will be connected to WebGPU LLM inference in a later
                phase.
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <article
                className={`message message--${message.role}`}
                key={message.id}
              >
                <span>{message.role === "user" ? "You" : "Assistant"}</span>
                <p>{message.content}</p>
              </article>
            ))
          )}
        </div>

        <form className="prompt-form" onSubmit={handleSubmit}>
          <label htmlFor="prompt">Prompt</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Enter a prompt for the local LLM"
            rows={5}
          />
          <div className="form-actions">
            <p>{prompt.trim().length} characters</p>
            <button type="submit" disabled={!canSubmit}>
              Send
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}

function formatWebGpuStatus(support: WebGpuSupport): WebGpuStatus {
  if (support.available === true) {
    return {
      state: "ready",
      label: "WebGPU ready",
      detail: "A GPU adapter is available for future local inference.",
    };
  }

  return {
    state: "unavailable",
    label: "WebGPU unavailable",
    detail: getWebGpuUnavailableDetail(support),
  };
}

function getWebGpuUnavailableDetail(
  support: Extract<WebGpuSupport, { available: false }>,
): string {
  switch (support.reason) {
    case "navigator-missing":
      return "navigator is not available in this runtime.";
    case "api-missing":
      return "This WebView does not expose navigator.gpu.";
    case "adapter-missing":
      return "No compatible GPU adapter was found.";
    case "request-failed":
      return support.error ?? "Adapter request failed.";
  }
}

export default App;
