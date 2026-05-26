import type { AgentEvent } from "@heliopause/agent";

export type MemoryProfile = "auto" | "low" | "full";

export type WorkerAgentEvent = Exclude<AgentEvent, { type: "text" }>;

export type WorkerWebSearchResult = {
  title: string;
  url: string;
  snippet: string;
};

export type WorkerWebSearchToolResult =
  | {
      callId: string;
      ok: true;
      content: {
        kind: "web_search";
        query: string;
        results: WorkerWebSearchResult[];
      };
    }
  | {
      callId: string;
      ok: false;
      error: {
        code: string;
        message: string;
        retryable?: boolean;
      };
    };

export type WorkerWebFetchContent = {
  kind: "web_fetch";
  url: string;
  finalUrl: string;
  path: string;
  status: number;
  contentType: string;
  bytesWritten: number;
  truncated: boolean;
  title?: string;
};

export type WorkerWebFetchToolResult =
  | {
      callId: string;
      ok: true;
      content: WorkerWebFetchContent;
    }
  | {
      callId: string;
      ok: false;
      error: {
        code: string;
        message: string;
        retryable?: boolean;
      };
    };

export type ResolvedRuntimeProfile = {
  requested: MemoryProfile;
  resolved: "low" | "full";
  maxWeightCacheBytes: number;
  estimatedWeightCacheBytes: number;
  wasmResidentWeightCache: boolean;
  webGpuStatus: "enabled" | "blocked";
  providerNames: string[];
  wasmStatus: "enabled" | "unavailable";
  wasmUnavailableReason?: string;
  webGpuUnavailableReason?: string;
  availableMemoryBytes?: number;
};

export type SystemMemoryInfo = {
  total_bytes: number;
  available_bytes: number;
};

export type WorkerModelInfo = {
  fileName: string;
  modelName: string;
  contextLength: number;
  originalContextLength: number;
  runtimeProfile: ResolvedRuntimeProfile;
  visionFileName?: string;
  supportsImages: boolean;
  visionImageTokens?: {
    min: number;
    max: number;
  };
  supportsAudio: boolean;
  audioFileName?: string;
  audioMaxSeconds?: number;
};

export type WorkerImageInput = {
  file: File;
  fileName: string;
};

export type WorkerAudioInput = {
  pcm: Float32Array;
  sampleRate: 16000;
  durationMs: number;
};

export type EngineWorkerRequest =
  | {
      type: "loadModel";
      requestId: number;
      file: File;
      fileName: string;
      visionFile?: File;
      visionFileName?: string;
      memoryProfile: MemoryProfile;
      memoryInfo?: SystemMemoryInfo;
    }
  | {
      type: "generateTurn";
      requestId: number;
      systemPrompt: string;
      userContent: string;
      image?: WorkerImageInput;
      audio?: WorkerAudioInput;
      maxNewTokens: number;
      doSample?: boolean;
      temperature?: number;
      topP?: number;
      topK?: number;
      seed?: number;
      webSearchAvailable: boolean;
    }
  | {
      type: "resolveWebSearchConfirmation";
      requestId: number;
      callId: string;
      approved: boolean;
      result?: WorkerWebSearchToolResult;
    }
  | {
      type: "resolveWebFetchConfirmation";
      requestId: number;
      callId: string;
      approved: boolean;
    }
  | {
      type: "cancelGeneration";
      requestId: number;
    };

export type EngineWorkerResponse =
  | {
      type: "modelLoaded";
      requestId: number;
      model: WorkerModelInfo;
    }
  | {
      type: "generationChunk";
      requestId: number;
      content: string;
    }
  | {
      type: "agentEvent";
      requestId: number;
      event: WorkerAgentEvent;
    }
  | {
      type: "webSearchConfirmationRequested";
      requestId: number;
      callId: string;
      query: string;
      maxResults: number;
    }
  | {
      type: "webFetchConfirmationRequested";
      requestId: number;
      callId: string;
      url: string;
      path: string;
    }
  | {
      type: "generationDone";
      requestId: number;
      inferenceDurationMs: number;
    }
  | {
      type: "generationCancelled";
      requestId: number;
    }
  | {
      type: "error";
      requestId: number;
      message: string;
    };
