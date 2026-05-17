import type {
  ModelRunner,
} from "../model-runner";
import type {
  MultimodalRunnerProvider,
} from "../provider";
import {
  referenceAudioEncoderRunner,
  referenceAudioPreprocessRunner,
} from "./audio-runner";
import {
  referenceVisionEncoderRunner,
  referenceVisionPreprocessRunner,
} from "./vision-runner";
import {
  referenceOutput,
  referenceSegmentRunner,
} from "./execution-provider";
import {
  prepareInput,
  preparePreparedHiddenInput,
} from "./layers";
import {
  ReferenceEmbeddingNode,
  ReferenceLayerSegmentNode,
  ReferenceOutputNode,
} from "./nodes";
import {
  CpuHiddenTransferNode,
} from "../buffer-nodes";
import {
  createModelResourceRequirements,
} from "../model-resources";

export function createReferenceModelRunner(): ModelRunner {
  return {
    provider: "reference",
    prepareInput,
    preparePreparedHiddenInput,
    segmentRunner: referenceSegmentRunner,
    output: referenceOutput,
    graph: {
      embeddingNode: (tokenIds) => new ReferenceEmbeddingNode(tokenIds),
      layerSegmentNode: (startLayer, endLayerExclusive, inputId) =>
        new ReferenceLayerSegmentNode(startLayer, endLayerExclusive, inputId),
      outputNode: (inputId, topK) => new ReferenceOutputNode(inputId, topK),
      importHiddenNode: (inputId) => new CpuHiddenTransferNode(inputId, "reference-import-hidden"),
      exportHiddenNode: (inputId) => new CpuHiddenTransferNode(inputId, "reference-export-hidden"),
    },
  };
}

export function createReferenceProvider(): MultimodalRunnerProvider {
  return {
    name: "reference",
    createModelRunner: createReferenceModelRunner,
    modelResourceRequirements: (session, options) => createModelResourceRequirements({
      provider: "reference",
      gguf: session.tensorReader.metadata,
      manifest: session.manifest,
      contextLength: options.contextLength,
      memoryLimitBytes: Number.POSITIVE_INFINITY,
      targetResourceConstrained: false,
      canRunFullModel: true,
      plannedReason: "Reference full-model placement is planned.",
    }),
    createAudioRunners: createReferenceAudioRunners,
    createVisionRunners: createReferenceVisionRunners,
  };
}

export function createReferenceAudioRunners() {
  return {
    preprocess: referenceAudioPreprocessRunner,
    encoder: referenceAudioEncoderRunner,
  };
}

export function createReferenceVisionRunners() {
  return {
    preprocess: referenceVisionPreprocessRunner,
    encoder: referenceVisionEncoderRunner,
  };
}

export { ReferenceSegmentRunner } from "./segment-runner";
export type {
  ReferenceHiddenResult,
  ReferenceSegmentRunnerOptions,
} from "./segment-runner";
