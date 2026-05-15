import { ChangeEvent, ClipboardEvent, DragEvent, FormEvent, KeyboardEvent, PointerEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  DEFAULT_SYSTEM_PROMPT,
  stripThinking,
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
  files?: UiFileAttachment[];
  image?: UiImageAttachment;
  audio?: UiAudioAttachment;
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

type RecordingState = "idle" | "requesting" | "recording" | "processing";

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
const INITIAL_ASSISTANT_CONTENT = [
  "Drop your GGUF files in the message box.",
  "Add the main model and optional projector together.",
].join("\n");

function App() {
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
  const [messages, setMessages] = useState<UiMessage[]>(() => [
    createAssistantMessage(INITIAL_ASSISTANT_CONTENT),
  ]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationError, setGenerationError] = useState<string | undefined>();
  const workerRef = useRef<Worker | null>(null);
  const modelFilesInputRef = useRef<HTMLInputElement | null>(null);
  const nextRequestIdRef = useRef(1);
  const pendingRequestsRef = useRef(new Map<number, PendingRequest>());
  const generationRequestRef = useRef<{ requestId: number; worker: Worker } | null>(null);
  const imageAttachmentRef = useRef<UiImageAttachment | undefined>(undefined);
  const messagesRef = useRef<UiMessage[]>([]);
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

  useEffect(() => {
    imageAttachmentRef.current = imageAttachment;
  }, [imageAttachment]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

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
    }
    window.addEventListener("blur", handleWindowBlur);
    return () => {
      window.removeEventListener("blur", handleWindowBlur);
    };
  }, []);

  function handleModelFilesChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    if (files.length === 0) {
      return;
    }
    addGgufFiles(files);
    event.target.value = "";
  }

  function addGgufFiles(files: File[]) {
    const ggufs = files.filter(isGgufFile);
    const rejectedImages = files.some((file) => file.type.startsWith("image/"));
    if (ggufs.length === 0) {
      const message = rejectedImages && model.status !== "ready"
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
    setIsGenerating(true);
    setGenerationError(undefined);

    try {
      const worker = workerRef.current;
      if (!worker) {
        throw new Error("The model worker is not running. Reload the model.");
      }
      await generateTurnInWorker(worker, userMessage.id, assistantId, trimmedPrompt, imageAttachment, undefined, 256);
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
    if (ggufs.length > 0 || model.status !== "ready") {
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
    setIsGenerating(true);
    setGenerationError(undefined);

    try {
      const worker = workerRef.current;
      if (!worker) {
        throw new Error("The model worker is not running. Reload the model.");
      }
      await generateTurnInWorker(worker, userMessage.id, assistantId, "", undefined, audio, 256);
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
      revokeAudioAttachment(pending.audio);
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
      <section className="chat-workspace" aria-label="Chat">
        <div className="message-panel" aria-live="polite">
          {messages.map((message) => (
            <MessageBubble
              isGenerating={isGenerating}
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
            disabled={model.status === "loading" || isGenerating}
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
                  <button type="button" onClick={clearImageAttachment} disabled={isGenerating}>
                    Remove
                  </button>
                </div>
              ) : null}
              {generationError ? (
                <p className="generation-error">{generationError}</p>
              ) : null}
            </>
          )}
          <div className="form-actions">
            <div className="input-actions">
              <button
                type="button"
                className="image-button"
                onClick={() => modelFilesInputRef.current?.click()}
                disabled={model.status === "loading" || isGenerating}
              >
                Models
              </button>
              {!isGgufComposerActive ? (
                <>
                  <label className="image-button">
                    Image
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageFileChange}
                      disabled={isGenerating || (model.status === "ready" && !model.supportsImages)}
                    />
                  </label>
                  <button
                    type="button"
                    className={`record-button${recordingState === "recording" ? " record-button--active" : ""}`}
                    onPointerDown={handleRecordPointerDown}
                    onPointerUp={handleRecordPointerUp}
                    onPointerCancel={handleRecordPointerCancel}
                    onPointerLeave={handleRecordPointerCancel}
                    onKeyDown={handleRecordKeyDown}
                    onKeyUp={handleRecordKeyUp}
                    disabled={!canRecordAudio}
                    aria-pressed={recordingState === "recording"}
                  >
                    {recordingLabel}
                  </button>
                </>
              ) : null}
            </div>
            <p>{composerStatusLabel(model, prompt, ggufFiles.length)}</p>
            {isGenerating ? (
              <button type="button" className="secondary-button" onClick={handleStop}>
                Stop
              </button>
            ) : (
              <button type="submit" disabled={!canSubmit}>
                {model.status === "loading" ? "Loading..." : ggufConfirmation ? "Load" : "Send"}
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
  const visibleContent = stripThinking(message.content);
  const textContent = message.files && message.files.length > 0 ? "" : visibleContent;
  const placeholder = message.role === "assistant" && isGenerating ? "Generating..." : "";
  return (
    <article className={`message message--${message.role}`}>
      <span>{message.role === "user" ? "You" : "Assistant"}</span>
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
      {textContent || placeholder ? <p>{textContent || placeholder}</p> : null}
      {message.role === "assistant" && message.inferenceDurationMs !== undefined ? (
        <footer className="message-meta">
          {formatDuration(message.inferenceDurationMs)}
        </footer>
      ) : null}
    </article>
  );
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
