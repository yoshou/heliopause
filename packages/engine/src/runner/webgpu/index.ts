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
  MultimodalRunnerProvider,
} from "../provider";
import {
  createWebGpuModelRunner,
} from "./model-runner";
import {
  type WebGpuProviderOptions,
  webGpuExecutionProviderOptions,
} from "./execution-provider";
import {
  webGpuResourceRequirements,
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

export function createWebGpuProvider(options: WebGpuProviderOptions = {}): MultimodalRunnerProvider & {
  readonly name: "webgpu";
  readonly options: Readonly<WebGpuProviderOptions>;
} {
  return {
    name: "webgpu",
    options: { ...options },
    createModelRunner: createWebGpuModelRunner,
    createModelGraphRunner: createWebGpuGraphRunner,
    modelResourceRequirements: (session, resourceOptions) => {
      const providerOptions = webGpuExecutionProviderOptions(session);
      return webGpuResourceRequirements(session.tensorReader.metadata, session.manifest, {
        mode: "enabled",
        contextLength: resourceOptions.contextLength,
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
  WebGpuProviderOptions,
} from "./execution-provider";

export type {
  WebGpuHiddenResult,
  WebGpuRuntimeStats,
  WebGpuSegmentRunnerOptions,
  WebGpuStateLike,
  WebGpuTokenResult,
} from "./segment-runner";
