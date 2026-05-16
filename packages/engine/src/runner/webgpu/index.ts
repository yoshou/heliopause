import {
  webGpuAudioEncoderRunner,
} from "./audio-execution-provider";
import {
  webGpuAudioPreprocessRunner,
} from "./audio-preprocess-runner";
import {
  webGpuVisionEncoderRunner,
} from "./vision-execution-provider";
import {
  webGpuVisionPreprocessRunner,
} from "./vision-preprocess-runner";
import {
  CpuToGpuHiddenTransferNode,
  GpuToCpuHiddenTransferNode,
  WebGpuEmbeddingNode,
  WebGpuLayerSegmentNode,
  WebGpuOutputNode,
} from "./nodes";
import type {
  RunnerProvider,
} from "../provider";
import {
  createWebGpuModelRunner,
} from "./model-runner";
import {
  webGpuExecutionProviderOptions,
} from "./execution-provider";
import {
  planRunnerPlacement,
} from "./planning";

export { createWebGpuModelRunner } from "./model-runner";

export function createWebGpuGraphRunner() {
  return {
    embeddingNode: (tokenIds: readonly number[]) => new WebGpuEmbeddingNode(tokenIds),
    layerSegmentNode: (startLayer: number, endLayerExclusive: number, inputId: string) =>
      new WebGpuLayerSegmentNode(startLayer, endLayerExclusive, inputId),
    outputNode: (inputId: string, topK?: number) => new WebGpuOutputNode(inputId, topK),
    importHiddenNode: (inputId: string) => new CpuToGpuHiddenTransferNode(inputId),
    exportHiddenNode: (inputId: string) => new GpuToCpuHiddenTransferNode(inputId),
  };
}

export function createWebGpuAudioRunners() {
  return {
    preprocess: webGpuAudioPreprocessRunner,
    encoder: webGpuAudioEncoderRunner,
  };
}

export function createWebGpuVisionRunners() {
  return {
    preprocess: webGpuVisionPreprocessRunner,
    encoder: webGpuVisionEncoderRunner,
  };
}

export function createWebGpuProvider(): RunnerProvider {
  return {
    name: "webgpu",
    createModelRunner: createWebGpuModelRunner,
    createModelGraphRunner: createWebGpuGraphRunner,
    planModelPlacement: (session, options) => {
      const providerOptions = webGpuExecutionProviderOptions(session);
      return planRunnerPlacement(session.tensorReader.metadata, session.manifest, {
        mode: "enabled",
        contextLength: options.contextLength,
        memoryLimitBytes: providerOptions?.memoryLimitBytes,
      });
    },
    createAudioRunners: createWebGpuAudioRunners,
    createVisionRunners: createWebGpuVisionRunners,
  };
}

export { checkWebGpuSupport, runWebGpuSmokeTest } from "./gpu-device";
export { WebGpuSegmentRunner } from "./segment-runner";
export type {
  WebGpuSmokeTest,
  WebGpuSupport,
} from "./gpu-types";
export type {
  WebGpuHiddenResult,
  WebGpuRuntimeStats,
  WebGpuSegmentRunnerOptions,
  WebGpuStateLike,
  WebGpuTokenResult,
} from "./segment-runner";
