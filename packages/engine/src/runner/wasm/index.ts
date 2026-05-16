import type {
  ModelRunner,
} from "../model-runner";
import type {
  RunnerProvider,
} from "../provider";
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

export function createWasmProvider(): RunnerProvider {
  return {
    name: "wasm",
    createModelRunner: createWasmModelRunner,
    createModelGraphRunner: () => createWasmModelRunner().graph as NonNullable<ModelRunner["graph"]>,
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
  WasmHiddenResult,
  WasmSegmentRunnerOptions,
} from "./segment-runner";
