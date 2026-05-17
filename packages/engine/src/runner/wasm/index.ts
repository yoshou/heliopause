import type {
  ModelRunner,
} from "../model-runner";
import type {
  MultimodalRunnerProvider,
} from "../provider";
import type {
  WasmProviderOptions,
} from "./options";
import {
  wasmOutput,
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

export function createWasmModelRunner(): ModelRunner {
  return {
    provider: "wasm",
    prepareInput,
    preparePreparedHiddenInput,
    segmentRunner: wasmSegmentRunner,
    output: wasmOutput,
    graph: {
      embeddingNode: (tokenIds) => new WasmEmbeddingNode(tokenIds),
      layerSegmentNode: (startLayer, endLayerExclusive, inputId) =>
        new WasmLayerSegmentNode(startLayer, endLayerExclusive, inputId),
      outputNode: (inputId, topK) => new WasmOutputNode(inputId, topK),
      importHiddenNode: (inputId) => new CpuHiddenTransferNode(inputId, "wasm-import-hidden"),
      exportHiddenNode: (inputId) => new CpuHiddenTransferNode(inputId, "wasm-export-hidden"),
    },
  };
}

export function createWasmProvider(options: WasmProviderOptions = {}): MultimodalRunnerProvider & {
  readonly name: "wasm";
  readonly options: Readonly<WasmProviderOptions>;
} {
  return {
    name: "wasm",
    options: { ...options },
    createModelRunner: createWasmModelRunner,
    createModelGraphRunner: () => createWasmModelRunner().graph as NonNullable<ModelRunner["graph"]>,
    modelResourceRequirements: (session, resourceOptions) => createModelResourceRequirements({
      provider: "wasm",
      gguf: session.tensorReader.metadata,
      manifest: session.manifest,
      contextLength: resourceOptions.contextLength,
      memoryLimitBytes: Number.POSITIVE_INFINITY,
      targetResourceConstrained: false,
      canRunFullModel: true,
      plannedReason: "WASM full-model placement is planned.",
    }),
    createAudioRunners: createWasmAudioRunners,
    createVisionRunners: createWasmVisionRunners,
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
