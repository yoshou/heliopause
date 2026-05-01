import type { ChatMessage } from "@heliopause/engine";

export type MemoryProfile = "auto" | "low" | "full";

export type ResolvedMemoryProfile = {
  requested: MemoryProfile;
  resolved: "low" | "full";
  maxWeightCacheBytes: number;
  estimatedWeightCacheBytes: number;
  wasmResidentWeightCache: boolean;
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
  memoryProfile: ResolvedMemoryProfile;
};

export type EngineWorkerRequest =
  | {
      type: "loadModel";
      requestId: number;
      file: File;
      fileName: string;
      memoryProfile: MemoryProfile;
      memoryInfo?: SystemMemoryInfo;
    }
  | {
      type: "generate";
      requestId: number;
      messages: ChatMessage[];
      maxNewTokens: number;
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
      type: "generationDone";
      requestId: number;
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
