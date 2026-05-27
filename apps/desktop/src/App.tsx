import { ChangeEvent, ClipboardEvent, DragEvent, FormEvent, KeyboardEvent, PointerEvent, RefObject, useEffect, useMemo, useRef, useState } from "react";
import { ChevronRight, FileArchive, Image, KeyRound, Lightbulb, Mic, Plus, SendHorizontal, Square, X } from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  DEFAULT_SYSTEM_PROMPT,
  stripThinking,
} from "@heliopause/engine";
import type {
  EngineWorkerRequest,
  EngineWorkerResponse,
  MemoryProfile,
  SystemMemoryInfo,
  WorkerAgentEvent,
  WorkerWebFetchContent,
  WorkerModelInfo,
  WorkerWebSearchResult,
  WorkerWebSearchToolResult,
} from "./engine-worker-protocol";

type UiMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  files?: UiFileAttachment[];
  image?: UiImageAttachment;
  audio?: UiAudioAttachment;
  agentEvents?: WorkerAgentEvent[];
  inferenceDurationMs?: number;
};

type UiFileAttachment = {
  fileName: string;
};

type UiImageAttachment = {
  file: File;
  fileName: string;
  url: string;
};

type UiAudioAttachment = {
  blob: Blob;
  url: string;
  fileName: string;
  wavBlob?: Blob;
  wavUrl?: string;
  wavFileName?: string;
  pcm: Float32Array;
  durationMs: number;
};

type ToolCardStatus = "running" | "success" | "failed" | "refused";

type ToolCardView = {
  id: string;
  step: number;
  actionLabel: string;
  status: ToolCardStatus;
  statusLabel: string;
  call?: Extract<WorkerAgentEvent, { type: "toolCall" }>["call"];
  result?: Extract<WorkerAgentEvent, { type: "toolResult" }>["result"];
  stepError?: Extract<WorkerAgentEvent, { type: "stepError" }>["error"];
};

type ThinkingView = {
  id: string;
  step: number;
  content: string;
  truncated: boolean;
};

type SandboxListFilesContent = {
  kind: "sandbox_list_files";
  entries: SandboxFileEntry[];
};

type SandboxReadFileContent = {
  kind: "sandbox_read_file";
  path: string;
  content: string;
  truncated?: boolean;
};

type SandboxWriteFileContent = {
  kind: "sandbox_write_file";
  path: string;
  bytesWritten: number;
};

type SandboxCommandContent = {
  kind: "sandbox_command";
  exitCode: number;
  stdout: string;
  stderr: string;
  truncated: boolean;
};

type WebSearchContent = {
  kind: "web_search";
  query: string;
  results: WorkerWebSearchResult[];
};

type WebFetchContent = WorkerWebFetchContent;

type SandboxFileEntry = {
  path: string;
  name: string;
  kind: "file" | "directory";
  sizeBytes: number;
};

type RecordingState = "idle" | "requesting" | "recording" | "processing";

const markdownPlugins = [remarkGfm];
const markdownComponents: Components = {
  a({ node: _node, ...props }) {
    return <a {...props} target="_blank" rel="noreferrer" />;
  },
  table({ node: _node, ...props }) {
    return (
      <div className="message-markdown-table-wrap">
        <table {...props} />
      </div>
    );
  },
};

type ModelState =
  | { status: "empty" }
  | { status: "loading"; fileName: string }
  | ({ status: "ready" } & WorkerModelInfo)
  | { status: "error"; fileName: string; message: string };

type GgufConfirmation = {
  message: string;
  mainCandidates: File[];
  projectorCandidates: File[];
  selectedMainName: string;
  selectedProjectorName: string;
};

type GgufValidation =
  | { status: "valid"; mainFile: File; projectorFile?: File }
  | { status: "invalid"; message: string }
  | {
      status: "needs-confirmation";
      message: string;
      mainCandidates: File[];
      projectorCandidates: File[];
    };

type TavilyTokenStatus = {
  available: boolean;
  configured: boolean;
  reason?: string;
};

type PendingWebSearchConfirmation = {
  worker: Worker;
  requestId: number;
  callId: string;
  query: string;
  maxResults: number;
};

type PendingWebFetchConfirmation = {
  worker: Worker;
  requestId: number;
  callId: string;
  url: string;
  path: string;
};

type TauriWebSearchResponse = {
  results: WorkerWebSearchResult[];
};

type WebSearchToolError = Extract<WorkerWebSearchToolResult, { ok: false }>["error"];

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
      audio?: UiAudioAttachment;
      resolve: () => void;
      reject: (error: Error) => void;
    };

const MIN_RECORDING_MS = 300;
const MAX_RECORDING_MS = 30_000;
const AUDIO_SAMPLE_RATE = 16_000;
const CHAT_MAX_NEW_TOKENS = 1024;
const MESSAGE_PANEL_BOTTOM_THRESHOLD_PX = 80;
const INITIAL_ASSISTANT_CONTENT = [
  "Drop your GGUF files in the message box.",
  "Add the main model and optional projector together.",
].join("\n");

function App() {
  const [isTauriRuntime] = useState(detectTauriRuntime);
  const [model, setModel] = useState<ModelState>({ status: "empty" });
  const memoryProfile: MemoryProfile = "auto";
  const systemPrompt = DEFAULT_SYSTEM_PROMPT;
  const [prompt, setPrompt] = useState("");
  const [ggufFiles, setGgufFiles] = useState<File[]>([]);
  const [ggufError, setGgufError] = useState<string | undefined>();
  const [ggufConfirmation, setGgufConfirmation] = useState<GgufConfirmation | undefined>();
  const [isDraggingFiles, setIsDraggingFiles] = useState(false);
  const [imageAttachment, setImageAttachment] = useState<UiImageAttachment | undefined>();
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [isAttachmentMenuOpen, setIsAttachmentMenuOpen] = useState(false);
  const [isTavilyMenuOpen, setIsTavilyMenuOpen] = useState(false);
  const [tavilyStatus, setTavilyStatus] = useState<TavilyTokenStatus | undefined>();
  const [tavilyError, setTavilyError] = useState<string | undefined>();
  const [pendingWebSearch, setPendingWebSearch] = useState<PendingWebSearchConfirmation | undefined>();
  const [pendingWebFetch, setPendingWebFetch] = useState<PendingWebFetchConfirmation | undefined>();
  const [isThinkingEnabled, setIsThinkingEnabled] = useState(true);
  const [messages, setMessages] = useState<UiMessage[]>(() => [
    createAssistantMessage(INITIAL_ASSISTANT_CONTENT),
  ]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatingAssistantId, setGeneratingAssistantId] = useState<string | undefined>();
  const [generationError, setGenerationError] = useState<string | undefined>();
  const workerRef = useRef<Worker | null>(null);
  const modelFilesInputRef = useRef<HTMLInputElement | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const attachmentMenuRef = useRef<HTMLDivElement | null>(null);
  const tavilyMenuRef = useRef<HTMLDivElement | null>(null);
  const tavilyTokenInputRef = useRef<HTMLInputElement | null>(null);
  const messagePanelRef = useRef<HTMLDivElement | null>(null);
  const nextRequestIdRef = useRef(1);
  const pendingRequestsRef = useRef(new Map<number, PendingRequest>());
  const generationRequestRef = useRef<{ requestId: number; worker: Worker } | null>(null);
  const imageAttachmentRef = useRef<UiImageAttachment | undefined>(undefined);
  const messagesRef = useRef<UiMessage[]>([]);
  const shouldFollowMessagesRef = useRef(true);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const recordingStartedAtRef = useRef(0);
  const recordingStopTimerRef = useRef<number | undefined>(undefined);
  const recordingStopRequestedRef = useRef(false);

  const isGgufComposerActive = model.status !== "ready" || ggufFiles.length > 0 || Boolean(ggufConfirmation);
  const canSubmit = useMemo(() => {
    if (isGgufComposerActive) {
      return model.status !== "loading" && ggufFiles.length > 0 && !isGenerating;
    }
    return (
      model.status === "ready" &&
      prompt.trim().length > 0 &&
      !isGenerating &&
      (!imageAttachment || model.supportsImages)
    );
  }, [ggufFiles.length, imageAttachment, isGenerating, isGgufComposerActive, model, prompt]);
  const canRecordAudio = useMemo(
    () =>
      model.status === "ready" &&
      model.supportsAudio &&
      !isGenerating &&
      recordingState !== "processing" &&
      Boolean(navigator.mediaDevices?.getUserMedia) &&
      typeof MediaRecorder !== "undefined",
    [isGenerating, model, recordingState],
  );
  const recordingLabel = recordingState === "recording"
    ? "Recording..."
    : recordingState === "requesting"
      ? "Requesting mic..."
      : recordingState === "processing"
        ? "Preparing audio..."
        : "Hold to record";
  const addAttachmentLabel = model.status === "ready"
    ? model.supportsImages ? "Add image" : "Images unavailable for this model"
    : model.status === "loading" ? "Loading model..." : "Choose model files";
  const canAddAttachment = model.status === "ready"
    ? model.supportsImages && !isGenerating
    : model.status !== "loading" && !isGenerating;
  const thinkingToggleLabel = isThinkingEnabled ? "Thinking on" : "Thinking off";
  const canToggleThinking = model.status === "ready" && !isGenerating && !isGgufComposerActive;

  useEffect(() => {
    imageAttachmentRef.current = imageAttachment;
  }, [imageAttachment]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    const messagePanel = messagePanelRef.current;
    if (!messagePanel) {
      return;
    }
    if (isGenerating) {
      return;
    }
    if (!shouldFollowMessagesRef.current) {
      return;
    }
    messagePanel.scrollTo({
      top: messagePanel.scrollHeight,
      behavior: "smooth",
    });
  }, [isGenerating, messages]);

  useEffect(() => {
    const messagePanel = messagePanelRef.current;
    if (!messagePanel) {
      return;
    }
    const panel = messagePanel;

    function updateShouldFollowMessages() {
      shouldFollowMessagesRef.current = isScrolledNearBottom(panel);
    }

    updateShouldFollowMessages();
    panel.addEventListener("scroll", updateShouldFollowMessages, { passive: true });
    return () => {
      panel.removeEventListener("scroll", updateShouldFollowMessages);
    };
  }, []);

  useEffect(() => () => {
    const worker = workerRef.current;
    if (worker) {
      rejectWorkerRequests(worker, new Error("Application closed."));
      worker.terminate();
    }
    const image = imageAttachmentRef.current;
    if (image) {
      URL.revokeObjectURL(image.url);
    }
    stopActiveRecording();
    revokeMessageAudioAttachments(messagesRef.current);
  }, []);

  useEffect(() => {
    function handleWindowBlur() {
      void stopRecording();
      setIsAttachmentMenuOpen(false);
      setIsTavilyMenuOpen(false);
    }
    window.addEventListener("blur", handleWindowBlur);
    return () => {
      window.removeEventListener("blur", handleWindowBlur);
    };
  }, []);

  useEffect(() => {
    function handlePointerDown(event: globalThis.PointerEvent) {
      if (!attachmentMenuRef.current?.contains(event.target as Node | null)) {
        setIsAttachmentMenuOpen(false);
      }
      if (!tavilyMenuRef.current?.contains(event.target as Node | null)) {
        setIsTavilyMenuOpen(false);
      }
    }

    function handleKeyDown(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") {
        setIsAttachmentMenuOpen(false);
        setIsTavilyMenuOpen(false);
      }
    }

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  useEffect(() => {
    if (!isTauriRuntime) {
      return;
    }
    void refreshTavilyStatus();
  }, [isTauriRuntime]);

  function handleModelFilesChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    setIsAttachmentMenuOpen(false);
    if (model.status === "ready") {
      event.target.value = "";
      setGenerationError("Model selection is locked after loading.");
      return;
    }
    if (files.length === 0) {
      return;
    }
    addGgufFiles(files);
    event.target.value = "";
  }

  function addGgufFiles(files: File[]) {
    if (model.status === "ready") {
      setGenerationError("Model selection is locked after loading.");
      return;
    }

    const ggufs = files.filter(isGgufFile);
    const rejectedImages = files.some((file) => file.type.startsWith("image/"));
    if (ggufs.length === 0) {
      const message = rejectedImages
        ? "Load a GGUF model before attaching images."
        : "Add GGUF files to load a model.";
      setGgufError(message);
      appendAssistantMessage(message);
      return;
    }

    if (imageAttachment) {
      URL.revokeObjectURL(imageAttachment.url);
    }
    setImageAttachment(undefined);
    setGgufFiles((currentFiles) => {
      const nextFiles = [...currentFiles];
      for (const file of ggufs) {
        if (!nextFiles.some((existingFile) => existingFile.name === file.name && existingFile.size === file.size)) {
          nextFiles.push(file);
        }
      }
      return nextFiles;
    });
    setGgufConfirmation(undefined);
    setGgufError(files.length === ggufs.length ? undefined : "Only GGUF files can be used to load a model.");
    setGenerationError(undefined);
  }

  function removeGgufFile(file: File) {
    setGgufFiles((currentFiles) => currentFiles.filter((currentFile) => currentFile !== file));
    setGgufConfirmation(undefined);
    setGgufError(undefined);
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
      setModel({
        status: "ready",
        ...modelInfo,
      });
      appendAssistantMessage(formatModelLoadedMessage(modelInfo));
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
      appendAssistantMessage([
        "I could not load those GGUF files.",
        "Please choose a usable Gemma GGUF model file, and optionally its projector.",
      ].join("\n"));
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isGgufComposerActive) {
      await submitGgufFiles();
      return;
    }
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
    setGeneratingAssistantId(assistantId);
    setIsGenerating(true);
    setGenerationError(undefined);

    try {
      const worker = workerRef.current;
      if (!worker) {
        throw new Error("The model worker is not running. Reload the model.");
      }
      await generateTurnInWorker(
        worker,
        userMessage.id,
        assistantId,
        trimmedPrompt,
        imageAttachment,
        undefined,
        CHAT_MAX_NEW_TOKENS,
        isTauriRuntime,
        isThinkingEnabled,
      );
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
      setGeneratingAssistantId(undefined);
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

  function handleAddAttachmentClick() {
    if (!canAddAttachment) {
      return;
    }
    setIsAttachmentMenuOpen((isOpen) => !isOpen);
  }

  function handleTavilyMenuClick() {
    if (!isTauriRuntime) {
      return;
    }
    setIsTavilyMenuOpen((isOpen) => {
      const nextIsOpen = !isOpen;
      if (nextIsOpen) {
        void refreshTavilyStatus();
      }
      return nextIsOpen;
    });
  }

  async function refreshTavilyStatus() {
    if (!isTauriRuntime) {
      return;
    }
    try {
      const status = await invokeTauri<TavilyTokenStatus>("tavily_token_status");
      setTavilyStatus(status);
      setTavilyError(undefined);
    } catch (error) {
      setTavilyStatus({ available: false, configured: false });
      setTavilyError(normalizeErrorMessage(error));
    }
  }

  async function saveTavilyToken() {
    const input = tavilyTokenInputRef.current;
    const token = input?.value.trim() ?? "";
    if (!token) {
      setTavilyError("Enter a Tavily token before saving.");
      return;
    }
    try {
      await invokeTauri("save_tavily_token", { token });
      if (input) {
        input.value = "";
      }
      await refreshTavilyStatus();
    } catch (error) {
      if (input) {
        input.value = "";
      }
      setTavilyError(normalizeErrorMessage(error));
    }
  }

  async function deleteTavilyToken() {
    try {
      await invokeTauri("delete_tavily_token");
      if (tavilyTokenInputRef.current) {
        tavilyTokenInputRef.current.value = "";
      }
      await refreshTavilyStatus();
    } catch (error) {
      setTavilyError(normalizeErrorMessage(error));
    }
  }

  async function approveWebSearch() {
    const confirmation = pendingWebSearch;
    if (!confirmation) {
      return;
    }
    setPendingWebSearch(undefined);
    const result = await runTavilyWebSearch(confirmation);
    confirmation.worker.postMessage({
      type: "resolveWebSearchConfirmation",
      requestId: confirmation.requestId,
      callId: confirmation.callId,
      approved: true,
      result,
    } satisfies EngineWorkerRequest);
  }

  function declineWebSearch() {
    const confirmation = pendingWebSearch;
    if (!confirmation) {
      return;
    }
    setPendingWebSearch(undefined);
    confirmation.worker.postMessage({
      type: "resolveWebSearchConfirmation",
      requestId: confirmation.requestId,
      callId: confirmation.callId,
      approved: false,
    } satisfies EngineWorkerRequest);
  }

  async function runTavilyWebSearch(
    confirmation: PendingWebSearchConfirmation,
  ): Promise<WorkerWebSearchToolResult> {
    try {
      const response = await invokeTauri<TauriWebSearchResponse>("web_search", {
        request: {
          query: confirmation.query,
          max_results: confirmation.maxResults,
        },
      });
      return {
        callId: confirmation.callId,
        ok: true,
        content: {
          kind: "web_search",
          query: confirmation.query,
          results: response.results,
        },
      };
    } catch (error) {
      return {
        callId: confirmation.callId,
        ok: false,
        error: normalizeToolError(error, "tavily_error"),
      };
    }
  }

  function approveWebFetch() {
    const confirmation = pendingWebFetch;
    if (!confirmation) {
      return;
    }
    setPendingWebFetch(undefined);
    confirmation.worker.postMessage({
      type: "resolveWebFetchConfirmation",
      requestId: confirmation.requestId,
      callId: confirmation.callId,
      approved: true,
    } satisfies EngineWorkerRequest);
  }

  function declineWebFetch() {
    const confirmation = pendingWebFetch;
    if (!confirmation) {
      return;
    }
    setPendingWebFetch(undefined);
    confirmation.worker.postMessage({
      type: "resolveWebFetchConfirmation",
      requestId: confirmation.requestId,
      callId: confirmation.callId,
      approved: false,
    } satisfies EngineWorkerRequest);
  }

  function handleModelMenuSelect() {
    if (model.status === "ready") {
      setIsAttachmentMenuOpen(false);
      setGenerationError("Model selection is locked after loading.");
      return;
    }
    setIsAttachmentMenuOpen(false);
    modelFilesInputRef.current?.click();
  }

  function handleImageMenuSelect() {
    if (model.status === "ready") {
      setIsAttachmentMenuOpen(false);
      imageInputRef.current?.click();
      return;
    }
    setIsAttachmentMenuOpen(false);
    const message = "Load a GGUF model before attaching images.";
    setGgufError(message);
    appendAssistantMessage(message);
  }

  function handleImageFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    setIsAttachmentMenuOpen(false);
    if (file) {
      setNextImageAttachment(file);
    }
    event.target.value = "";
  }

  function handlePromptPaste(event: ClipboardEvent<HTMLTextAreaElement>) {
    const file = Array.from(event.clipboardData.files).find((item) => item.type.startsWith("image/"));
    if (file) {
      if (model.status !== "ready") {
        const message = "Load a GGUF model before attaching images.";
        setGgufError(message);
        appendAssistantMessage(message);
        return;
      }
      setNextImageAttachment(file);
    }
  }

  function handleComposerDragOver(event: DragEvent<HTMLFormElement>) {
    if (event.dataTransfer.types.includes("Files")) {
      event.preventDefault();
      setIsDraggingFiles(true);
    }
  }

  function handleComposerDragLeave(event: DragEvent<HTMLFormElement>) {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
      setIsDraggingFiles(false);
    }
  }

  function handleComposerDrop(event: DragEvent<HTMLFormElement>) {
    const files = Array.from(event.dataTransfer.files);
    if (files.length === 0) {
      return;
    }
    event.preventDefault();
    setIsDraggingFiles(false);

    const ggufs = files.filter(isGgufFile);
    if (model.status === "ready" && ggufs.length > 0) {
      setGenerationError("Model selection is locked after loading.");
      return;
    }

    if (model.status !== "ready") {
      addGgufFiles(files);
      return;
    }

    const image = files.find((file) => file.type.startsWith("image/"));
    if (image) {
      setNextImageAttachment(image);
      return;
    }

    setGenerationError("Attach an image, or add GGUF files to replace the model.");
  }

  async function submitGgufFiles() {
    if (model.status === "loading" || ggufFiles.length === 0) {
      return;
    }

    const validation = ggufConfirmation
      ? resolveConfirmedGgufFiles(ggufFiles, ggufConfirmation)
      : validateGgufFiles(ggufFiles);

    if (validation.status === "invalid") {
      setGgufError(validation.message);
      appendAssistantMessage(validation.message);
      return;
    }

    if (validation.status === "needs-confirmation") {
      setGgufConfirmation({
        ...validation,
        selectedMainName: validation.mainCandidates[0]?.name ?? "",
        selectedProjectorName: validation.projectorCandidates[0]?.name ?? "",
      });
      setGgufError(undefined);
      appendAssistantMessage(validation.message);
      return;
    }

    const filesForMessage = [validation.mainFile, validation.projectorFile].filter(Boolean) as File[];
    setMessages((currentMessages) => [
      ...currentMessages,
      {
        id: createId("user"),
        role: "user",
        content: filesForMessage.map((file) => file.name).join("\n"),
        files: filesForMessage.map((file) => ({ fileName: file.name })),
      },
    ]);
    setGgufFiles([]);
    setGgufConfirmation(undefined);
    setGgufError(undefined);
    setPrompt("");
    await loadSelectedModel(validation.mainFile, validation.projectorFile);
  }

  function clearImageAttachment() {
    if (imageAttachment) {
      URL.revokeObjectURL(imageAttachment.url);
    }
    setImageAttachment(undefined);
  }

  function setNextImageAttachment(file: File) {
    if (model.status !== "ready") {
      const message = "Load a GGUF model before attaching images.";
      setGgufError(message);
      appendAssistantMessage(message);
      return;
    }
    if (!model.supportsImages) {
      setGenerationError("This model does not support images.");
      return;
    }
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

  async function startRecording() {
    if (!canRecordAudio || recordingState !== "idle") {
      return;
    }
    setGenerationError(undefined);
    setRecordingState("requesting");
    recordingStopRequestedRef.current = false;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (recordingStopRequestedRef.current) {
        stopMediaStream(stream);
        setRecordingState("idle");
        return;
      }
      const mimeType = selectAudioMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recordingChunksRef.current = [];
      recordingStartedAtRef.current = performance.now();
      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordingChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = () => {
        void finalizeRecording(recorder.mimeType || mimeType || "audio/webm");
      };
      recorder.start();
      setRecordingState("recording");
      recordingStopTimerRef.current = window.setTimeout(() => {
        void stopRecording();
      }, MAX_RECORDING_MS);
      if (recordingStopRequestedRef.current) {
        void stopRecording();
      }
    } catch (error) {
      setRecordingState("idle");
      setGenerationError(error instanceof Error ? error.message : String(error));
    }
  }

  async function stopRecording() {
    recordingStopRequestedRef.current = true;
    if (recordingStopTimerRef.current !== undefined) {
      window.clearTimeout(recordingStopTimerRef.current);
      recordingStopTimerRef.current = undefined;
    }
    const recorder = mediaRecorderRef.current;
    if (!recorder) {
      if (recordingState === "requesting") {
        return;
      }
      setRecordingState("idle");
      return;
    }
    if (recorder.state !== "inactive") {
      recorder.stop();
    }
  }

  function handleRecordPointerDown(event: PointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    void startRecording();
  }

  function handleRecordPointerUp(event: PointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    void stopRecording();
  }

  function handleRecordPointerCancel() {
    void stopRecording();
  }

  function handleRecordKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key !== " " && event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    if (event.repeat) {
      return;
    }
    void startRecording();
  }

  function handleRecordKeyUp(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key !== " " && event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    void stopRecording();
  }

  async function finalizeRecording(mimeType: string) {
    const chunks = recordingChunksRef.current;
    const durationMs = performance.now() - recordingStartedAtRef.current;
    stopActiveRecording();
    if (durationMs < MIN_RECORDING_MS || chunks.length === 0) {
      recordingChunksRef.current = [];
      setRecordingState("idle");
      return;
    }

    setRecordingState("processing");
    const timestamp = formatAudioFileTimestamp(new Date());
    const webmBlob = new Blob(chunks, { type: mimeType });
    const fileName = `heliopause-audio-${timestamp}.webm`;
    try {
      const { wavBlob, pcm } = await create16KhzMonoWavBlob(webmBlob);
      const nextAudio: UiAudioAttachment = {
        blob: webmBlob,
        url: URL.createObjectURL(webmBlob),
        fileName,
        wavBlob,
        wavUrl: URL.createObjectURL(wavBlob),
        wavFileName: `heliopause-audio-${timestamp}-16khz.wav`,
        pcm,
        durationMs,
      };
      await submitAudioTurn(nextAudio);
      setGenerationError(undefined);
    } catch (error) {
      setGenerationError(error instanceof Error ? error.message : String(error));
    } finally {
      recordingChunksRef.current = [];
      setRecordingState("idle");
    }
  }

  function appendAssistantMessage(content: string) {
    setMessages((currentMessages) => [
      ...currentMessages,
      createAssistantMessage(content),
    ]);
  }

  async function submitAudioTurn(audio: UiAudioAttachment) {
    if (model.status !== "ready" || isGenerating || !model.supportsAudio) {
      revokeAudioAttachment(audio);
      return;
    }

    const userMessage: UiMessage = {
      id: createId("user"),
      role: "user",
      content: "",
      audio,
    };
    const assistantId = createId("assistant");
    setMessages((currentMessages) => [
      ...currentMessages,
      userMessage,
      { id: assistantId, role: "assistant" as const, content: "" },
    ]);
    setGeneratingAssistantId(assistantId);
    setIsGenerating(true);
    setGenerationError(undefined);

    try {
      const worker = workerRef.current;
      if (!worker) {
        throw new Error("The model worker is not running. Reload the model.");
      }
      await generateTurnInWorker(
        worker,
        userMessage.id,
        assistantId,
        "",
        undefined,
        audio,
        CHAT_MAX_NEW_TOKENS,
        isTauriRuntime,
        isThinkingEnabled,
      );
    } catch (error) {
      setGenerationError(error instanceof Error ? error.message : String(error));
      setMessages((currentMessages) =>
        currentMessages.filter((message) =>
          message.id !== userMessage.id && message.id !== assistantId
        ),
      );
      revokeAudioAttachment(audio);
    } finally {
      generationRequestRef.current = null;
      setGeneratingAssistantId(undefined);
      setIsGenerating(false);
    }
  }

  function stopActiveRecording() {
    if (recordingStopTimerRef.current !== undefined) {
      window.clearTimeout(recordingStopTimerRef.current);
      recordingStopTimerRef.current = undefined;
    }
    mediaRecorderRef.current = null;
    const stream = mediaStreamRef.current;
    mediaStreamRef.current = null;
    if (stream) {
      stopMediaStream(stream);
    }
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
    audio: UiAudioAttachment | undefined,
    maxNewTokens: number,
    webSearchAvailable: boolean,
    enableThinking: boolean,
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
        audio,
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
        audio: audio
          ? {
              pcm: audio.pcm,
              sampleRate: 16000,
              durationMs: audio.durationMs,
            }
          : undefined,
        maxNewTokens,
        webSearchAvailable,
        enableThinking,
      } satisfies EngineWorkerRequest);
    });
  }

  function handleWorkerMessage(worker: Worker, message: EngineWorkerResponse) {
    const pending = pendingRequestsRef.current.get(message.requestId);
    if (!pending || pending.worker !== worker) {
      return;
    }

    if (message.type === "error") {
      setPendingWebSearch((confirmation) =>
        confirmation?.requestId === message.requestId ? undefined : confirmation
      );
      setPendingWebFetch((confirmation) =>
        confirmation?.requestId === message.requestId ? undefined : confirmation
      );
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

    if (message.type === "webSearchConfirmationRequested") {
      if (!isTauriRuntime) {
        worker.postMessage({
          type: "resolveWebSearchConfirmation",
          requestId: message.requestId,
          callId: message.callId,
          approved: true,
          result: {
            callId: message.callId,
            ok: false,
            error: {
              code: "web_search_unavailable",
              message: "web_search is only available in the desktop app.",
            },
          },
        } satisfies EngineWorkerRequest);
        return;
      }
      setPendingWebSearch({
        worker,
        requestId: message.requestId,
        callId: message.callId,
        query: message.query,
        maxResults: message.maxResults,
      });
      return;
    }

    if (message.type === "webFetchConfirmationRequested") {
      setPendingWebFetch({
        worker,
        requestId: message.requestId,
        callId: message.callId,
        url: message.url,
        path: message.path,
      });
      return;
    }

    if (message.type === "agentEvent") {
      setMessages((currentMessages) =>
        currentMessages.map((uiMessage) =>
          uiMessage.id === pending.assistantId
            ? {
                ...uiMessage,
                agentEvents: [...(uiMessage.agentEvents ?? []), message.event],
              }
            : uiMessage,
        ),
      );
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
      setPendingWebSearch((confirmation) =>
        confirmation?.requestId === message.requestId ? undefined : confirmation
      );
      setPendingWebFetch((confirmation) =>
        confirmation?.requestId === message.requestId ? undefined : confirmation
      );
      setMessages((currentMessages) =>
        currentMessages.filter((uiMessage) =>
          uiMessage.id !== pending.userId && uiMessage.id !== pending.assistantId
        ),
      );
      revokeAudioAttachment(pending.audio);
      setPrompt((currentPrompt) =>
        currentPrompt.length === 0 ? pending.userContent : currentPrompt
      );
      setImageAttachment(pending.image);
    }

    if (message.type === "generationDone") {
      setPendingWebSearch((confirmation) =>
        confirmation?.requestId === message.requestId ? undefined : confirmation
      );
      setPendingWebFetch((confirmation) =>
        confirmation?.requestId === message.requestId ? undefined : confirmation
      );
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
    setPendingWebSearch((confirmation) =>
      confirmation?.worker === worker ? undefined : confirmation
    );
    setPendingWebFetch((confirmation) =>
      confirmation?.worker === worker ? undefined : confirmation
    );
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
      <section className="chat-workspace" aria-label="Chat">
        <div ref={messagePanelRef} className="message-panel" aria-live="polite">
          {messages.map((message) => (
            <MessageBubble
              isGenerating={isGenerating && message.id === generatingAssistantId}
              key={message.id}
              message={message}
            />
          ))}
        </div>

        <form
          className={`prompt-form${isDraggingFiles ? " prompt-form--dragging" : ""}`}
          onDragLeave={handleComposerDragLeave}
          onDragOver={handleComposerDragOver}
          onDrop={handleComposerDrop}
          onSubmit={handleSubmit}
        >
          <input
            ref={modelFilesInputRef}
            className="visually-hidden-input"
            type="file"
            accept=".gguf"
            multiple
            onChange={handleModelFilesChange}
            disabled={model.status === "ready" || model.status === "loading" || isGenerating}
          />
          <input
            ref={imageInputRef}
            className="visually-hidden-input"
            type="file"
            accept="image/*"
            onChange={handleImageFileChange}
            disabled={model.status !== "ready" || isGenerating || !model.supportsImages}
          />
          {isGgufComposerActive ? (
            <GgufComposer
              confirmation={ggufConfirmation}
              disabled={model.status === "loading" || isGenerating}
              error={ggufError}
              files={ggufFiles}
              isLoading={model.status === "loading"}
              onBrowse={() => modelFilesInputRef.current?.click()}
              onConfirmationChange={setGgufConfirmation}
              onRemoveFile={removeGgufFile}
            />
          ) : (
            <>
              <label htmlFor="prompt">You</label>
              <textarea
                id="prompt"
                value={prompt}
                onChange={(event) => {
                  setPrompt(event.target.value);
                  setGenerationError(undefined);
                }}
                onPaste={handlePromptPaste}
                placeholder="Ask the local model"
                rows={4}
                disabled={isGenerating}
              />
              {imageAttachment ? (
                <div className="attachment-preview">
                  <img src={imageAttachment.url} alt="" />
                  <span>{imageAttachment.fileName}</span>
                  <button
                    type="button"
                    className="icon-button icon-button--ghost"
                    aria-label="Remove image"
                    title="Remove image"
                    onClick={clearImageAttachment}
                    disabled={isGenerating}
                  >
                    <X aria-hidden="true" size={18} />
                  </button>
                </div>
              ) : null}
              {generationError ? (
                <p className="generation-error">{generationError}</p>
              ) : null}
              {isTauriRuntime && pendingWebSearch ? (
                <WebSearchConfirmationPanel
                  confirmation={pendingWebSearch}
                  onApprove={() => void approveWebSearch()}
                  onDecline={declineWebSearch}
                />
              ) : null}
              {pendingWebFetch ? (
                <WebFetchConfirmationPanel
                  confirmation={pendingWebFetch}
                  onApprove={approveWebFetch}
                  onDecline={declineWebFetch}
                />
              ) : null}
            </>
          )}
          <div className="form-actions">
            <div className="input-actions">
              <div className="attachment-menu-wrap" ref={attachmentMenuRef}>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={addAttachmentLabel}
                  aria-haspopup="menu"
                  aria-expanded={isAttachmentMenuOpen}
                  title={addAttachmentLabel}
                  onClick={handleAddAttachmentClick}
                  disabled={!canAddAttachment}
                >
                  <Plus aria-hidden="true" size={20} />
                </button>
                {isAttachmentMenuOpen ? (
                  <div className="attachment-menu" role="menu">
                    {model.status === "ready" ? (
                      <button
                        type="button"
                        className="attachment-menu-item"
                        role="menuitem"
                        onClick={handleImageMenuSelect}
                      >
                        <Image aria-hidden="true" size={18} />
                        <span>Image</span>
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="attachment-menu-item"
                        role="menuitem"
                        onClick={handleModelMenuSelect}
                      >
                        <FileArchive aria-hidden="true" size={18} />
                        <span>Model</span>
                      </button>
                    )}
                  </div>
                ) : null}
              </div>
              {isTauriRuntime ? (
                <div className="tavily-menu-wrap" ref={tavilyMenuRef}>
                  <button
                    type="button"
                    className="icon-button"
                    aria-label="Tavily search settings"
                    aria-haspopup="dialog"
                    aria-expanded={isTavilyMenuOpen}
                    title="Tavily search settings"
                    onClick={handleTavilyMenuClick}
                  >
                    <KeyRound aria-hidden="true" size={19} />
                  </button>
                  {isTavilyMenuOpen ? (
                    <TavilyTokenPopover
                      error={tavilyError}
                      inputRef={tavilyTokenInputRef}
                      onDelete={() => void deleteTavilyToken()}
                      onSave={() => void saveTavilyToken()}
                      status={tavilyStatus}
                    />
                  ) : null}
                </div>
              ) : null}
              {!isGgufComposerActive ? (
                <button
                  type="button"
                  className={`icon-button thinking-toggle${isThinkingEnabled ? " thinking-toggle--active" : ""}`}
                  aria-label={thinkingToggleLabel}
                  aria-pressed={isThinkingEnabled}
                  title={thinkingToggleLabel}
                  onClick={() => setIsThinkingEnabled((current) => !current)}
                  disabled={!canToggleThinking}
                >
                  <Lightbulb aria-hidden="true" size={19} />
                </button>
              ) : null}
            </div>
            <p>{composerStatusLabel(model, prompt, ggufFiles.length)}</p>
            <div className="submit-actions">
              {!isGgufComposerActive ? (
                <button
                  type="button"
                  className={`icon-button record-button${recordingState === "recording" ? " record-button--active" : ""}`}
                  onPointerDown={handleRecordPointerDown}
                  onPointerUp={handleRecordPointerUp}
                  onPointerCancel={handleRecordPointerCancel}
                  onPointerLeave={handleRecordPointerCancel}
                  onKeyDown={handleRecordKeyDown}
                  onKeyUp={handleRecordKeyUp}
                  disabled={!canRecordAudio}
                  aria-label={recordingLabel}
                  aria-pressed={recordingState === "recording"}
                  title={recordingLabel}
                >
                  <Mic aria-hidden="true" size={20} />
                </button>
              ) : null}
              {isGenerating ? (
                <button
                  type="button"
                  className="icon-button icon-button--secondary"
                  aria-label="Stop generation"
                  title="Stop generation"
                  onClick={handleStop}
                >
                  <Square aria-hidden="true" size={18} />
                </button>
              ) : (
                <button
                  type="submit"
                  className="icon-button icon-button--primary"
                  aria-label={model.status === "loading" ? "Loading..." : ggufConfirmation ? "Load model" : "Send"}
                  title={model.status === "loading" ? "Loading..." : ggufConfirmation ? "Load model" : "Send"}
                  disabled={!canSubmit}
                >
                  {isGgufComposerActive ? (
                    <FileArchive aria-hidden="true" size={20} />
                  ) : (
                    <SendHorizontal aria-hidden="true" size={20} />
                  )}
                </button>
              )}
            </div>
          </div>
        </form>
      </section>
    </main>
  );
}

function MessageBubble(
  { isGenerating, message }: { isGenerating: boolean; message: UiMessage },
) {
  const toolCards = message.role === "assistant"
    ? buildToolCardViews(message.agentEvents ?? [])
    : [];
  const thinking = message.role === "assistant"
    ? buildThinkingViews(message.agentEvents ?? [])
    : [];
  const visibleContent = stripAgentArtifacts(stripThinking(message.content)).trim();
  const textContent = message.files && message.files.length > 0 ? "" : visibleContent;
  const placeholder = message.role === "assistant" &&
      isGenerating &&
      textContent.length === 0 &&
      thinking.length === 0 &&
      toolCards.length === 0
    ? "Generating..."
    : "";
  return (
    <article className={`message message--${message.role}`}>
      <span className="message-role-label">{message.role === "user" ? "You" : "Assistant"}</span>
      {message.files && message.files.length > 0 ? (
        <div className="message-files" aria-label="Attached files">
          {message.files.map((file) => (
            <div className="file-chip" key={file.fileName}>
              {file.fileName}
            </div>
          ))}
        </div>
      ) : null}
      {message.image ? (
        <img className="message-image" src={message.image.url} alt="" />
      ) : null}
      {message.audio ? (
        <AudioMessage audio={message.audio} />
      ) : null}
      {textContent ? (
        message.role === "assistant" ? (
          <AssistantMarkdown content={textContent} />
        ) : (
          <p className="message-text">{textContent}</p>
        )
      ) : null}
      {thinking.length > 0 ? <ThinkingPanel thinking={thinking} /> : null}
      {toolCards.length > 0 ? <ToolIndicators tools={toolCards} /> : null}
      {placeholder ? (
        <p className="message-text">{placeholder}</p>
      ) : null}
      {message.role === "assistant" && message.inferenceDurationMs !== undefined ? (
        <footer className="message-meta">
          {formatDuration(message.inferenceDurationMs)}
        </footer>
      ) : null}
    </article>
  );
}

function ThinkingPanel({ thinking }: { thinking: ThinkingView[] }) {
  return (
    <div className="thinking-panels" aria-label="Thinking">
      {thinking.map((item) => (
        <details className="thinking-panel" key={item.id}>
          <summary>
            <Lightbulb aria-hidden="true" size={15} />
            <span>Thinking</span>
            <small>Step {item.step}</small>
            <ChevronRight className="thinking-details-icon" aria-hidden="true" size={15} />
          </summary>
          <pre>{item.content}</pre>
          {item.truncated ? (
            <small className="thinking-truncated">Thinking truncated</small>
          ) : null}
        </details>
      ))}
    </div>
  );
}

function AssistantMarkdown({ content }: { content: string }) {
  return (
    <div className="message-markdown">
      <ReactMarkdown
        components={markdownComponents}
        remarkPlugins={markdownPlugins}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function ToolIndicators({ tools }: { tools: ToolCardView[] }) {
  return (
    <div className="tool-indicators" aria-label="Tool activity">
      {tools.map((tool) => (
        <details
          className={`tool-indicator tool-indicator--${tool.status}`}
          key={tool.id}
        >
          <summary>
            <span className="tool-status-dot" aria-hidden="true" />
            <span className="tool-action-label">{tool.actionLabel}</span>
            <span className="tool-status-label">{tool.statusLabel}</span>
            <ChevronRight className="tool-details-icon" aria-hidden="true" size={15} />
          </summary>
          <ToolIndicatorDetails tool={tool} />
        </details>
      ))}
    </div>
  );
}

function ToolIndicatorDetails({ tool }: { tool: ToolCardView }) {
  const args = tool.call?.arguments;
  return (
    <div className="tool-detail">
      <dl className="tool-detail-grid">
        <dt>Step</dt>
        <dd>{tool.step}</dd>
        <dt>Call ID</dt>
        <dd>{tool.id}</dd>
        <dt>Tool</dt>
        <dd>{tool.call?.name ?? "Tool call"}</dd>
      </dl>
      {args !== undefined ? (
        <ToolPreBlock label="Arguments" value={formatJson(args)} />
      ) : null}
      {tool.stepError ? (
        <ToolErrorDetail code={tool.stepError.code} message={tool.stepError.message} />
      ) : null}
      {tool.result ? <ToolResultDetail result={tool.result} args={args} /> : null}
    </div>
  );
}

function ToolResultDetail(
  {
    args,
    result,
  }: {
    args: unknown;
    result: Extract<WorkerAgentEvent, { type: "toolResult" }>["result"];
  },
) {
  if (!result.ok) {
    return (
      <ToolErrorDetail
        code={result.error.code}
        message={result.error.message}
      />
    );
  }

  const content = result.content;
  if (isSandboxListFilesContent(content)) {
    return (
      <div className="tool-result">
        <div className="tool-result-heading">Result</div>
        {content.entries.length > 0 ? (
          <ul className="tool-entry-list">
            {content.entries.map((entry) => (
              <li key={entry.path}>
                <span>{entry.name}</span>
                <small>{entry.kind} | {formatBytes(entry.sizeBytes)}</small>
              </li>
            ))}
          </ul>
        ) : (
          <p className="tool-empty">No entries</p>
        )}
      </div>
    );
  }

  if (isSandboxReadFileContent(content)) {
    return (
      <div className="tool-result">
        <dl className="tool-detail-grid">
          <dt>Path</dt>
          <dd>{content.path}</dd>
          <dt>Size</dt>
          <dd>{formatBytes(byteLength(content.content))}</dd>
          <dt>Truncated</dt>
          <dd>{content.truncated ? "Yes" : "No"}</dd>
        </dl>
        <ToolPreBlock label="Preview" value={previewText(content.content)} />
      </div>
    );
  }

  if (isSandboxWriteFileContent(content)) {
    const writtenContent = isRecord(args) && typeof args.content === "string"
      ? args.content
      : undefined;
    return (
      <div className="tool-result">
        <dl className="tool-detail-grid">
          <dt>Path</dt>
          <dd>{content.path}</dd>
          <dt>Written</dt>
          <dd>{formatBytes(content.bytesWritten)}</dd>
        </dl>
        {writtenContent !== undefined ? (
          <ToolPreBlock label="Preview" value={previewText(writtenContent)} />
        ) : null}
      </div>
    );
  }

  if (isSandboxCommandContent(content)) {
    return (
      <div className="tool-result">
        <dl className="tool-detail-grid">
          <dt>Exit code</dt>
          <dd>{content.exitCode}</dd>
          <dt>Truncated</dt>
          <dd>{content.truncated ? "Yes" : "No"}</dd>
        </dl>
        <ToolPreBlock label="stdout" value={previewText(content.stdout)} />
        <ToolPreBlock label="stderr" value={previewText(content.stderr)} />
      </div>
    );
  }

  if (isWebSearchContent(content)) {
    return (
      <div className="tool-result">
        <div className="tool-result-heading">Results</div>
        {content.results.length > 0 ? (
          <ul className="web-search-result-list">
            {content.results.map((result) => (
              <li key={result.url}>
                <a href={result.url} target="_blank" rel="noreferrer">
                  {result.title}
                </a>
                <small>{result.url}</small>
                <p>{result.snippet}</p>
              </li>
            ))}
          </ul>
        ) : (
          <p className="tool-empty">No results</p>
        )}
      </div>
    );
  }

  if (isWebFetchContent(content)) {
    return (
      <div className="tool-result">
        <dl className="tool-detail-grid">
          <dt>Path</dt>
          <dd>{content.path}</dd>
          <dt>Status</dt>
          <dd>{content.status}</dd>
          <dt>Type</dt>
          <dd>{content.contentType}</dd>
          <dt>Written</dt>
          <dd>{formatBytes(content.bytesWritten)}</dd>
          <dt>Truncated</dt>
          <dd>{content.truncated ? "Yes" : "No"}</dd>
        </dl>
        {content.title ? <ToolPreBlock label="Title" value={content.title} /> : null}
        <ToolPreBlock label="Final URL" value={content.finalUrl} />
      </div>
    );
  }

  return <ToolPreBlock label="Result" value={formatJson(content)} />;
}

function ToolErrorDetail({ code, message }: { code: string; message: string }) {
  return (
    <div className="tool-error-detail">
      <dl className="tool-detail-grid">
        <dt>Error</dt>
        <dd>{code}</dd>
        <dt>Message</dt>
        <dd>{message}</dd>
      </dl>
    </div>
  );
}

function ToolPreBlock({ label, value }: { label: string; value: string }) {
  const isLong = isLongToolPreview(value);
  if (isLong) {
    return (
      <details className="tool-pre-block tool-pre-block--collapsible">
        <summary>
          <span>{label}</span>
          <ChevronRight className="tool-pre-icon" aria-hidden="true" size={14} />
        </summary>
        <pre>{value}</pre>
      </details>
    );
  }

  return (
    <div className="tool-pre-block">
      <div>{label}</div>
      <pre>{value}</pre>
    </div>
  );
}

function buildThinkingViews(events: WorkerAgentEvent[]): ThinkingView[] {
  const views = new Map<number, ThinkingView>();
  const orderedSteps: number[] = [];

  for (const event of events) {
    if (event.type !== "thinking") {
      continue;
    }
    if (!views.has(event.step)) {
      orderedSteps.push(event.step);
    }
    views.set(event.step, {
      id: `thinking-${event.step}`,
      step: event.step,
      content: event.content,
      truncated: event.truncated === true,
    });
  }

  return orderedSteps.map((step) => views.get(step)).filter((view): view is ThinkingView => Boolean(view));
}

function buildToolCardViews(events: WorkerAgentEvent[]): ToolCardView[] {
  const tools = new Map<string, ToolCardView>();
  const orderedIds: string[] = [];

  function upsert(id: string, next: ToolCardView) {
    if (!tools.has(id)) {
      orderedIds.push(id);
    }
    tools.set(id, next);
  }

  for (const event of events) {
    if (event.type === "toolCall") {
      const id = event.call.id;
      upsert(id, {
        id,
        step: event.step,
        actionLabel: describeToolAction(event.call),
        status: "running",
        statusLabel: "Running",
        call: event.call,
      });
      continue;
    }

    if (event.type === "toolResult") {
      const current = tools.get(event.result.callId);
      const status = statusForToolResult(event.result);
      upsert(event.result.callId, {
        id: event.result.callId,
        step: current?.step ?? event.step,
        actionLabel: current?.call ? describeToolAction(current.call) : "Tool call",
        status,
        statusLabel: labelForStatus(status),
        call: current?.call,
        result: event.result,
      });
      continue;
    }

    if (event.type === "stepError") {
      const current = tools.get(event.callId);
      upsert(event.callId, {
        id: event.callId,
        step: current?.step ?? event.step,
        actionLabel: current?.call ? describeToolAction(current.call) : "Tool call",
        status: "failed",
        statusLabel: "Failed",
        call: current?.call,
        stepError: event.error,
      });
    }
  }

  return orderedIds.map((id) => tools.get(id)).filter((tool): tool is ToolCardView => Boolean(tool));
}

function statusForToolResult(
  result: Extract<WorkerAgentEvent, { type: "toolResult" }>["result"],
): ToolCardStatus {
  if (!result.ok) {
    return result.error.code === "user_denied" ? "refused" : "failed";
  }
  if (isSandboxCommandContent(result.content) && result.content.exitCode !== 0) {
    return "failed";
  }
  return "success";
}

function labelForStatus(status: ToolCardStatus): string {
  switch (status) {
    case "running":
      return "Running";
    case "success":
      return "Done";
    case "failed":
      return "Failed";
    case "refused":
      return "Declined";
  }
}

function describeToolAction(
  call: Extract<WorkerAgentEvent, { type: "toolCall" }>["call"],
): string {
  const args = isRecord(call.arguments) ? call.arguments : {};
  switch (call.name) {
    case "sandbox_list_files":
      return `List ${typeof args.path === "string" ? args.path : "/workspace"}`;
    case "sandbox_read_file":
      return `Read ${typeof args.path === "string" ? args.path : "file"}`;
    case "sandbox_write_file":
      return `Write ${typeof args.path === "string" ? args.path : "file"}`;
    case "sandbox_command": {
      const commandArgs = Array.isArray(args.args)
        ? args.args.filter((item): item is string => typeof item === "string")
        : [];
      return commandArgs.join(" ") || "command";
    }
    case "web_search":
      return `Search: ${typeof args.query === "string" ? args.query : "web"}`;
    case "web_fetch":
      return `Fetch: ${typeof args.url === "string" ? args.url : "web"}`;
  }
}

function stripAgentArtifacts(content: string): string {
  return content
    .replace(/<\|tool_call>[\s\S]*?<tool_call\|>/g, "")
    .replace(/<\|tool_response>[\s\S]*?<tool_response\|>/g, "")
    .replace(/<\|tool_call>[\s\S]*$/g, "")
    .replace(/<\|tool_response>[\s\S]*$/g, "");
}

function isSandboxListFilesContent(value: unknown): value is SandboxListFilesContent {
  return (
    isRecord(value) &&
    value.kind === "sandbox_list_files" &&
    Array.isArray(value.entries) &&
    value.entries.every(isSandboxFileEntry)
  );
}

function isSandboxReadFileContent(value: unknown): value is SandboxReadFileContent {
  return (
    isRecord(value) &&
    value.kind === "sandbox_read_file" &&
    typeof value.path === "string" &&
    typeof value.content === "string"
  );
}

function isSandboxWriteFileContent(value: unknown): value is SandboxWriteFileContent {
  return (
    isRecord(value) &&
    value.kind === "sandbox_write_file" &&
    typeof value.path === "string" &&
    typeof value.bytesWritten === "number"
  );
}

function isSandboxCommandContent(value: unknown): value is SandboxCommandContent {
  return (
    isRecord(value) &&
    value.kind === "sandbox_command" &&
    typeof value.exitCode === "number" &&
    typeof value.stdout === "string" &&
    typeof value.stderr === "string" &&
    typeof value.truncated === "boolean"
  );
}

function isWebSearchContent(value: unknown): value is WebSearchContent {
  return (
    isRecord(value) &&
    value.kind === "web_search" &&
    typeof value.query === "string" &&
    Array.isArray(value.results) &&
    value.results.every(isWebSearchResult)
  );
}

function isWebSearchResult(value: unknown): value is WorkerWebSearchResult {
  return (
    isRecord(value) &&
    typeof value.title === "string" &&
    typeof value.url === "string" &&
    typeof value.snippet === "string"
  );
}

function isWebFetchContent(value: unknown): value is WebFetchContent {
  return (
    isRecord(value) &&
    value.kind === "web_fetch" &&
    typeof value.url === "string" &&
    typeof value.finalUrl === "string" &&
    typeof value.path === "string" &&
    typeof value.status === "number" &&
    typeof value.contentType === "string" &&
    typeof value.bytesWritten === "number" &&
    typeof value.truncated === "boolean" &&
    (value.title === undefined || typeof value.title === "string")
  );
}

function isSandboxFileEntry(value: unknown): value is SandboxFileEntry {
  return (
    isRecord(value) &&
    typeof value.path === "string" &&
    typeof value.name === "string" &&
    (value.kind === "file" || value.kind === "directory") &&
    typeof value.sizeBytes === "number"
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatJson(value: unknown): string {
  try {
    return previewText(JSON.stringify(value, null, 2));
  } catch {
    return "Unable to format value";
  }
}

function previewText(value: string): string {
  const maxLength = 1800;
  if (value.length === 0) {
    return "(empty)";
  }
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength)}\n... truncated`;
}

function isLongToolPreview(value: string): boolean {
  return value.length > 360 || value.split("\n").length > 8;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "unknown size";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function byteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

function isScrolledNearBottom(element: HTMLElement): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= MESSAGE_PANEL_BOTTOM_THRESHOLD_PX;
}

function GgufComposer(
  {
    confirmation,
    disabled,
    error,
    files,
    isLoading,
    onBrowse,
    onConfirmationChange,
    onRemoveFile,
  }: {
    confirmation: GgufConfirmation | undefined;
    disabled: boolean;
    error: string | undefined;
    files: File[];
    isLoading: boolean;
    onBrowse: () => void;
    onConfirmationChange: (confirmation: GgufConfirmation) => void;
    onRemoveFile: (file: File) => void;
  },
) {
  return (
    <>
      <label>You</label>
      <div className="gguf-drop-box">
        {files.length > 0 ? (
          <div className="file-chip-list" aria-label="Attached GGUF files">
            {files.map((file) => (
              <div className="file-chip file-chip--removable" key={`${file.name}-${file.size}`}>
                <span>{file.name}</span>
                <button
                  type="button"
                  aria-label={`Remove ${file.name}`}
                  onClick={() => onRemoveFile(file)}
                  disabled={disabled}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        ) : (
          <button
            type="button"
            className="gguf-placeholder"
            onClick={onBrowse}
            disabled={disabled}
          >
            Drop GGUF here or choose files
          </button>
        )}
        {isLoading ? <p className="composer-hint">Loading model...</p> : null}
      </div>
      {confirmation ? (
        <div className="gguf-confirmation">
          <p>{confirmation.message}</p>
          <label>
            Main model
            <select
              value={confirmation.selectedMainName}
              onChange={(event) =>
                onConfirmationChange({
                  ...confirmation,
                  selectedMainName: event.target.value,
                })}
              disabled={disabled}
            >
              {confirmation.mainCandidates.map((file) => (
                <option key={`${file.name}-${file.size}-main`} value={file.name}>
                  {file.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Projector
            <select
              value={confirmation.selectedProjectorName}
              onChange={(event) =>
                onConfirmationChange({
                  ...confirmation,
                  selectedProjectorName: event.target.value,
                })}
              disabled={disabled}
            >
              <option value="">None</option>
              {confirmation.projectorCandidates.map((file) => (
                <option key={`${file.name}-${file.size}-projector`} value={file.name}>
                  {file.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : null}
      {error ? <p className="generation-error">{error}</p> : null}
    </>
  );
}

function AudioMessage({ audio }: { audio: UiAudioAttachment }) {
  return (
    <div className="message-audio">
      <audio controls src={audio.url} />
      <a href={audio.url} download={audio.fileName}>WebM</a>
      {audio.wavUrl && audio.wavFileName ? (
        <a href={audio.wavUrl} download={audio.wavFileName}>WAV</a>
      ) : null}
    </div>
  );
}

function TavilyTokenPopover(
  {
    error,
    inputRef,
    onDelete,
    onSave,
    status,
  }: {
    error: string | undefined;
    inputRef: RefObject<HTMLInputElement | null>;
    onDelete: () => void;
    onSave: () => void;
    status: TavilyTokenStatus | undefined;
  },
) {
  const statusLabel = !status
    ? "Checking..."
    : !status.available
      ? "Unavailable"
      : status.configured ? "Configured" : "Not configured";
  return (
    <div className="tavily-popover" role="dialog" aria-label="Tavily search settings">
      <div className="tavily-popover-heading">
        <strong>Tavily search</strong>
        <span>{statusLabel}</span>
      </div>
      {status?.reason ? <p>{status.reason}</p> : null}
      <label>
        API token
        <input
          ref={inputRef}
          type="password"
          autoComplete="off"
          spellCheck={false}
          placeholder="tvly-..."
          disabled={status?.available === false}
        />
      </label>
      {error ? <p className="tavily-error">{error}</p> : null}
      <div className="tavily-actions">
        <button type="button" onClick={onSave} disabled={status?.available === false}>
          Save
        </button>
        <button type="button" onClick={onDelete} disabled={status?.available === false || !status?.configured}>
          Delete
        </button>
      </div>
    </div>
  );
}

function WebSearchConfirmationPanel(
  {
    confirmation,
    onApprove,
    onDecline,
  }: {
    confirmation: PendingWebSearchConfirmation;
    onApprove: () => void;
    onDecline: () => void;
  },
) {
  return (
    <div className="web-search-confirmation" role="alert">
      <div>
        <strong>Web search</strong>
        <p>{confirmation.query}</p>
        <small>{confirmation.maxResults} results maximum</small>
      </div>
      <div className="web-search-confirmation-actions">
        <button type="button" onClick={onDecline}>
          Decline
        </button>
        <button type="button" onClick={onApprove}>
          Approve
        </button>
      </div>
    </div>
  );
}

function WebFetchConfirmationPanel(
  {
    confirmation,
    onApprove,
    onDecline,
  }: {
    confirmation: PendingWebFetchConfirmation;
    onApprove: () => void;
    onDecline: () => void;
  },
) {
  return (
    <div className="web-search-confirmation" role="alert">
      <div>
        <strong>Web fetch</strong>
        <p>{confirmation.url}</p>
        <small>Save to {confirmation.path}</small>
      </div>
      <div className="web-search-confirmation-actions">
        <button type="button" onClick={onDecline}>
          Decline
        </button>
        <button type="button" onClick={onApprove}>
          Approve
        </button>
      </div>
    </div>
  );
}

function validateGgufFiles(files: File[]): GgufValidation {
  const ggufFiles = files.filter(isGgufFile);
  if (ggufFiles.length === 0) {
    return {
      status: "invalid",
      message: "Add GGUF files to load a model.",
    };
  }

  const projectorCandidates = ggufFiles.filter(isProjectorFile);
  const mainCandidates = ggufFiles.filter((file) => !isProjectorFile(file));

  if (mainCandidates.length === 0 && projectorCandidates.length > 0) {
    return {
      status: "invalid",
      message: [
        "This looks like a projector file, but I still need the main model.",
        "Please add the main Gemma GGUF model and send again.",
      ].join("\n"),
    };
  }

  if (mainCandidates.length === 0) {
    return {
      status: "invalid",
      message: [
        "I could not find a usable main model in those files.",
        "Please add a Gemma GGUF model file, and optionally its projector.",
      ].join("\n"),
    };
  }

  if (mainCandidates.length > 1 || projectorCandidates.length > 1) {
    return {
      status: "needs-confirmation",
      message: mainCandidates.length > 1
        ? "I found multiple possible main models. Please choose one model to load."
        : "I found multiple possible projectors. Please choose one projector, or select None.",
      mainCandidates,
      projectorCandidates,
    };
  }

  return {
    status: "valid",
    mainFile: mainCandidates[0],
    projectorFile: projectorCandidates[0],
  };
}

function resolveConfirmedGgufFiles(files: File[], confirmation: GgufConfirmation): GgufValidation {
  const mainFile = files.find((file) => file.name === confirmation.selectedMainName);
  const projectorFile = confirmation.selectedProjectorName
    ? files.find((file) => file.name === confirmation.selectedProjectorName)
    : undefined;

  if (!mainFile) {
    return {
      status: "invalid",
      message: "Please choose one main model to load.",
    };
  }
  if (projectorFile && projectorFile === mainFile) {
    return {
      status: "invalid",
      message: "Please choose different files for the main model and projector.",
    };
  }
  return {
    status: "valid",
    mainFile,
    projectorFile,
  };
}

function isGgufFile(file: File): boolean {
  return file.name.toLowerCase().endsWith(".gguf");
}

function isProjectorFile(file: File): boolean {
  const fileName = file.name.toLowerCase();
  return fileName.includes("projector") || fileName.includes("mmproj") || fileName.includes("vision");
}

function createAssistantMessage(content: string): UiMessage {
  return {
    id: createId("assistant"),
    role: "assistant",
    content,
  };
}

function formatModelLoadedMessage(model: WorkerModelInfo): string {
  return [
    `Model loaded: ${model.modelName}`,
    `Context: ${model.contextLength.toLocaleString()} tokens`,
    `Vision: ${model.supportsImages ? "available" : "not available"}`,
    `Audio: ${model.supportsAudio ? "available" : "not available"}`,
    "",
    "I am ready. What would you like to do?",
  ].join("\n");
}

function composerStatusLabel(model: ModelState, prompt: string, ggufFileCount: number): string {
  if (model.status === "loading") {
    return `Loading ${model.fileName}`;
  }
  if (ggufFileCount > 0) {
    return `${ggufFileCount} GGUF ${ggufFileCount === 1 ? "file" : "files"}`;
  }
  if (model.status === "ready") {
    return `${prompt.trim().length} characters`;
  }
  return "No model loaded";
}

function selectAudioMimeType(): string {
  const supportedType = [
    "audio/webm;codecs=opus",
    "audio/webm",
  ].find((mimeType) => MediaRecorder.isTypeSupported(mimeType));
  return supportedType ?? "";
}

function stopMediaStream(stream: MediaStream) {
  for (const track of stream.getTracks()) {
    track.stop();
  }
}

function revokeAudioAttachment(audio: UiAudioAttachment | undefined) {
  if (!audio) {
    return;
  }
  URL.revokeObjectURL(audio.url);
  if (audio.wavUrl) {
    URL.revokeObjectURL(audio.wavUrl);
  }
}

function revokeMessageAudioAttachments(messages: UiMessage[]) {
  for (const message of messages) {
    revokeAudioAttachment(message.audio);
  }
}

function formatAudioFileTimestamp(date: Date): string {
  const pad = (value: number) => value.toString().padStart(2, "0");
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
    "-",
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds()),
  ].join("");
}

async function create16KhzMonoWavBlob(audioBlob: Blob): Promise<{ wavBlob: Blob; pcm: Float32Array }> {
  const audioContext = new AudioContext();
  try {
    const decoded = await audioContext.decodeAudioData(await audioBlob.arrayBuffer());
    const mono = mixAudioBufferToMono(decoded);
    const resampled = resampleLinear(mono, decoded.sampleRate, AUDIO_SAMPLE_RATE);
    return {
      wavBlob: encodeFloat32PcmWav(resampled, AUDIO_SAMPLE_RATE),
      pcm: resampled,
    };
  } finally {
    await audioContext.close();
  }
}

function mixAudioBufferToMono(audioBuffer: AudioBuffer): Float32Array {
  const mono = new Float32Array(audioBuffer.length);
  for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
    const channelData = audioBuffer.getChannelData(channel);
    for (let index = 0; index < channelData.length; index += 1) {
      mono[index] += channelData[index] / audioBuffer.numberOfChannels;
    }
  }
  return mono;
}

function resampleLinear(input: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (sourceRate === targetRate) {
    return input;
  }
  const outputLength = Math.max(1, Math.round(input.length * targetRate / sourceRate));
  const output = new Float32Array(outputLength);
  const ratio = sourceRate / targetRate;
  for (let index = 0; index < outputLength; index += 1) {
    const position = index * ratio;
    const leftIndex = Math.floor(position);
    const rightIndex = Math.min(leftIndex + 1, input.length - 1);
    const fraction = position - leftIndex;
    output[index] = input[leftIndex] * (1 - fraction) + input[rightIndex] * fraction;
  }
  return output;
}

function encodeFloat32PcmWav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (const sample of samples) {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
    offset += bytesPerSample;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeAscii(view: DataView, offset: number, text: string) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

async function readSystemMemoryInfo(): Promise<SystemMemoryInfo | undefined> {
  try {
    const info = await invokeTauri<SystemMemoryInfo>("system_memory_info");
    if (info.total_bytes <= 0 && info.available_bytes <= 0) {
      return undefined;
    }
    return info;
  } catch {
    return undefined;
  }
}

function detectTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function invokeTauri<T = unknown>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

function normalizeErrorMessage(error: unknown): string {
  if (isRecord(error) && typeof error.message === "string") {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function normalizeToolError(error: unknown, fallbackCode: string): WebSearchToolError {
  if (isRecord(error)) {
    const code = typeof error.code === "string" ? error.code : fallbackCode;
    const message = typeof error.message === "string" ? error.message : "Web search failed.";
    const retryable = typeof error.retryable === "boolean" ? error.retryable : undefined;
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
    message: "Web search failed.",
  };
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
