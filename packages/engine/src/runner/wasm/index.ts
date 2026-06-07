import type {
  ModelRunner,
} from "../model-runner";
import type {
  MtpAssistantRunnerProvider,
  MultimodalRunnerProvider,
} from "../provider";
import type {
  WasmProviderOptions,
} from "./options";
import {
  wasmSegmentRunner,
} from "./execution-provider";
import {
  prepareInput,
  preparePreparedHiddenInput,
} from "./layers";
import {
  WasmEmbeddingNode,
  WasmLayerSegmentNode,
  WasmOutputNode,
  WasmPreparedHiddenInputNode,
} from "./nodes";
import {
  CpuHiddenTransferNode,
} from "../buffer-nodes";
import {
  createModelResourceRequirements,
} from "../model-resources";
import {
  wasmAudioPreprocessRunner,
} from "./audio-preprocess-runner";
import {
  wasmAudioEncoderRunner,
} from "./audio-runner";
import {
  wasmVisionPreprocessRunner,
} from "./vision-preprocess-runner";
import {
  wasmVisionEncoderRunner,
} from "./vision-runner";
import {
  createWasmMtpAssistantRunners,
} from "./mtp-assistant-runner";

export function createWasmModelRunner(): ModelRunner {
  return {
    provider: "wasm",
    prepareInput,
    preparePreparedHiddenInput,
    segmentRunner: wasmSegmentRunner,
    graphNodes: {
      createEmbeddingNode: (tokenIds) => new WasmEmbeddingNode(tokenIds),
      createPreparedHiddenInputNode: (hidden) => new WasmPreparedHiddenInputNode(hidden),
      createLayerSegmentNode: (startLayer, endLayerExclusive, inputId) =>
        new WasmLayerSegmentNode(startLayer, endLayerExclusive, inputId),
      createOutputNode: (inputId, topK) => new WasmOutputNode(inputId, topK),
      createImportHiddenNode: (inputId) => new CpuHiddenTransferNode(inputId, "wasm-import-hidden"),
      createExportHiddenNode: (inputId) => new CpuHiddenTransferNode(inputId, "wasm-export-hidden"),
    },
  };
}

export function createWasmProvider(options: WasmProviderOptions = {}): MultimodalRunnerProvider & MtpAssistantRunnerProvider & {
  readonly name: "wasm";
  readonly options: Readonly<WasmProviderOptions>;
} {
  return {
    name: "wasm",
    options: { ...options },
    createModelRunner: createWasmModelRunner,
    modelResourceRequirements: (session, resourceOptions) => createModelResourceRequirements({
      provider: "wasm",
      gguf: session.tensorReader.metadata,
      manifest: session.manifest,
      contextLength: resourceOptions.contextLength,
      slidingWindowReserveTokens: resourceOptions.slidingWindowReserveTokens,
      memoryLimitBytes: Number.POSITIVE_INFINITY,
      targetResourceConstrained: false,
      canRunFullModel: true,
      plannedReason: "WASM full-model placement is planned.",
    }),
    createAudioRunners: createWasmAudioRunners,
    createVisionRunners: createWasmVisionRunners,
    createMtpAssistantRunners: createWasmMtpAssistantRunners,
  };
}

export function createWasmAudioRunners() {
  return {
    preprocess: wasmAudioPreprocessRunner,
    encoder: wasmAudioEncoderRunner,
  };
}

export function createWasmVisionRunners() {
  return {
    preprocess: wasmVisionPreprocessRunner,
    encoder: wasmVisionEncoderRunner,
  };
}

export { WasmSegmentRunner } from "./segment-runner";
export type {
  WasmProviderOptions,
} from "./options";

export type {
  WasmHiddenResult,
  WasmSegmentRunnerOptions,
} from "./segment-runner";
