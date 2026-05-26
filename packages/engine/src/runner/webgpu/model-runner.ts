import type {
  ModelDecodeTokenOptions,
  ModelDecodeTokenResult,
  ModelRunner,
  ModelSegmentRunnerOptions,
} from "../model-runner";
import type {
  SegmentRunner,
} from "../segment-runner";
import type {
  InferenceState,
  ModelSession,
} from "../../runtime";
import {
  timedAsync,
} from "../../runtime";
import {
  planModelPlacement,
} from "../planning";
import {
  webGpuExecutionProviderOptions,
  webGpuSegmentRunner,
} from "./execution-provider";
import {
  CpuToGpuHiddenTransferNode,
  GpuToCpuHiddenTransferNode,
  WebGpuEmbeddingNode,
  WebGpuLayerSegmentNode,
  WebGpuOutputNode,
  WebGpuPreparedHiddenInputNode,
} from "./nodes";
import {
  prepareWebGpuInput,
} from "./model-io";

export function createWebGpuModelRunner(): ModelRunner {
  return {
    provider: "webgpu",
    graphNodes: createWebGpuGraphNodes(),
    prepareInput: prepareWebGpuInput,
    preparePreparedHiddenInput: rejectWebGpuPreparedHiddenInput,
    segmentRunner: webGpuModelSegmentRunner,
    decodeToken: decodeWebGpuToken,
  };
}

function createWebGpuGraphNodes() {
  return {
    createEmbeddingNode: (tokenIds: readonly number[]) => new WebGpuEmbeddingNode(tokenIds),
    createPreparedHiddenInputNode: (hidden: Float32Array) => new WebGpuPreparedHiddenInputNode(hidden),
    createLayerSegmentNode: (startLayer: number, endLayerExclusive: number, inputId: string) =>
      new WebGpuLayerSegmentNode(startLayer, endLayerExclusive, inputId),
    createOutputNode: (inputId: string, topK?: number) => new WebGpuOutputNode(inputId, topK),
    createImportHiddenNode: (inputId: string) => new CpuToGpuHiddenTransferNode(inputId),
    createExportHiddenNode: (inputId: string) => new GpuToCpuHiddenTransferNode(inputId),
  };
}

async function rejectWebGpuPreparedHiddenInput(): Promise<never> {
  throw new Error(
    "WebGPU prepared hidden input must be created through graphNodes.createPreparedHiddenInputNode.",
  );
}

async function decodeWebGpuToken(
  session: ModelSession,
  tokenId: number,
  options: ModelDecodeTokenOptions,
): Promise<ModelDecodeTokenResult> {
  const positions = new Int32Array([options.position]);
  const gpu = await webGpuSegmentRunnerForForward(session, options.state);

  if (!gpu.supportsGpuInputPreparation()) {
    throw new Error(
      "WebGPU direct token decode requires GPU input preparation. " +
        "Use graph-planned forward.decode() for mixed WASM/WebGPU placement.",
    );
  }

  const result = await timedAsync(
    options.trace,
    "WebGPU token-id input segment",
    () => gpu.runTokenIds([tokenId], positions, options.state, {
      computeSelectedToken: true,
      topK: options.logitsTopK,
    }),
  );
  options.state.nextPosition = Math.max(options.state.nextPosition, options.position + 1);
  return {
    hidden: new Float32Array(),
    state: options.state,
    selectedTokenId: result.selectedTokenId,
    topTokens: result.topTokens ?? [],
  };
}

async function webGpuSegmentRunnerForForward(
  session: ModelSession,
  state: InferenceState,
) {
  const providerOptions = webGpuExecutionProviderOptions(session);
  if (!providerOptions) {
    throw new Error("WebGPU segment runner is not enabled for this session.");
  }
  const webGpuStartLayer = providerOptions.segmentStartLayer ??
    plannedWebGpuStartLayer(session, state.contextLength, providerOptions.memoryLimitBytes);
  return webGpuSegmentRunner(session, state, { segmentStartLayer: webGpuStartLayer });
}

async function webGpuModelSegmentRunner(
  options: ModelSegmentRunnerOptions,
): Promise<SegmentRunner> {
  const providerOptions = webGpuExecutionProviderOptions(options.session);
  if (!providerOptions) {
    throw new Error("WebGPU model runner is not enabled for this session.");
  }
  const webGpuStartLayer = providerOptions.segmentStartLayer ??
    plannedWebGpuStartLayer(options.session, options.state.contextLength, providerOptions.memoryLimitBytes);

  if (webGpuStartLayer === undefined || webGpuStartLayer >= options.segmentEndLayerExclusive) {
    throw new Error(
      "WebGPU model segment runner cannot execute this segment. " +
        "Use graph-planned execution for mixed WASM/WebGPU placement.",
    );
  }

  const segmentStartLayer = Math.max(options.segmentStartLayer, webGpuStartLayer);
  if (segmentStartLayer > options.segmentStartLayer) {
    throw new Error(
      "WebGPU model segment runner cannot execute a WASM prefix inside the WebGPU runner. " +
        "Use graph-planned execution for mixed WASM/WebGPU placement.",
    );
  }

  const gpu = await webGpuSegmentRunner(options.session, options.state, {
    segmentStartLayer,
    segmentEndLayerExclusive: options.segmentEndLayerExclusive,
  });
  return gpu;
}

function plannedWebGpuStartLayer(
  session: ModelSession,
  contextLength: number,
  memoryLimitBytes: number,
): number | undefined {
  const plan = planModelPlacement(
    session.providers.map((provider) =>
      provider.modelResourceRequirements(session, { contextLength })
    ),
    {
      mode: "enabled",
      memoryLimitBytes,
      providerPriority: session.providers.map((provider) => provider.name),
    },
  );
  if (plan.status !== "planned") {
    throw new Error(plan.reason ?? "WebGPU model placement could not be planned.");
  }
  return plan.segments.find((segment) => segment.provider === "webgpu")?.startLayer;
}
