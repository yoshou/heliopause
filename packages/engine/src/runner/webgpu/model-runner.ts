import type {
  ModelDecodeTokenOptions,
  ModelDecodeTokenResult,
  ModelRunner,
  ModelSegmentRunnerOptions,
} from "../model-runner";
import type {
  SegmentHiddenResult,
  SegmentRunOptions,
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
  prepareInput as prepareWasmInput,
  preparePreparedHiddenInput as prepareWasmPreparedHiddenInput,
} from "../wasm/layers";
import {
  wasmOutput,
  wasmSegmentRunner,
} from "../wasm/execution-provider";
import {
  planRunnerPlacement,
} from "./planning";
import {
  webGpuExecutionProviderOptions,
  webGpuSegmentRunner,
} from "./execution-provider";

export function createWebGpuModelRunner(): ModelRunner {
  return {
    provider: "webgpu",
    prepareInput: prepareWasmInput,
    preparePreparedHiddenInput: prepareWasmPreparedHiddenInput,
    segmentRunner: webGpuModelSegmentRunner,
    output: wasmOutput,
    decodeToken: decodeWebGpuToken,
  };
}

async function decodeWebGpuToken(
  session: ModelSession,
  tokenId: number,
  options: ModelDecodeTokenOptions,
): Promise<ModelDecodeTokenResult> {
  const positions = new Int32Array([options.position]);
  const gpu = await webGpuSegmentRunnerForForward(session, options.state);

  if (gpu.supportsGpuInputPreparation()) {
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

  const prepared = await prepareWasmInput(session, [tokenId], options.trace);
  const prefix = wasmSegmentRunner({
    session,
    manifest: session.manifest,
    epsilon: session.epsilon,
    segmentStartLayer: 0,
    segmentEndLayerExclusive: gpu.segmentStartLayer,
  });
  const segmentInputHidden = (await prefix.runTokenHidden(prepared.hidden, positions, options.state, {
    trace: options.trace,
    perLayerInputs: prepared.perLayerInputs,
  })).hidden;
  const result = await timedAsync(
    options.trace,
    "WebGPU segment",
    () => gpu.runToken(segmentInputHidden, positions, options.state, {
      computeSelectedToken: true,
      topK: options.logitsTopK,
      perLayerInputs: prepared.perLayerInputs,
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
    planRunnerPlacement(
      session.tensorReader.metadata,
      session.manifest,
      {
        mode: "enabled",
        contextLength: state.contextLength,
        memoryLimitBytes: providerOptions.memoryLimitBytes,
      },
    ).segments.find((segment) => segment.provider === "webgpu")?.startLayer;
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
    planRunnerPlacement(
      options.session.tensorReader.metadata,
      options.manifest,
      {
        mode: "enabled",
        contextLength: options.state.contextLength,
        memoryLimitBytes: providerOptions.memoryLimitBytes,
      },
    ).segments.find((segment) => segment.provider === "webgpu")?.startLayer;

  if (webGpuStartLayer === undefined || webGpuStartLayer >= options.segmentEndLayerExclusive) {
    return wasmSegmentRunner(options);
  }

  const gpu = await webGpuSegmentRunner(options.session, options.state, {
    segmentStartLayer: Math.max(options.segmentStartLayer, webGpuStartLayer),
  });
  if (gpu.segmentStartLayer <= options.segmentStartLayer) {
    return gpu;
  }

  const prefix = wasmSegmentRunner({
    ...options,
    segmentEndLayerExclusive: gpu.segmentStartLayer,
  });
  return new WasmPrefixWebGpuSegmentRunner(prefix, gpu, options.segmentStartLayer);
}

class WasmPrefixWebGpuSegmentRunner implements SegmentRunner {
  readonly provider = "webgpu" as const;
  readonly segmentEndLayerExclusive: number;
  readonly segmentStartLayer: number;
  private readonly prefix: SegmentRunner;
  private readonly gpu: SegmentRunner;

  constructor(
    prefix: SegmentRunner,
    gpu: SegmentRunner,
    segmentStartLayer: number,
  ) {
    this.prefix = prefix;
    this.gpu = gpu;
    this.segmentStartLayer = segmentStartLayer;
    this.segmentEndLayerExclusive = gpu.segmentEndLayerExclusive;
  }

  async runTokensHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: InferenceState,
    options: SegmentRunOptions = {},
  ): Promise<SegmentHiddenResult> {
    const prefix = await this.prefix.runTokensHidden(inputHidden, positions, state, options);
    return this.gpu.runTokensHidden(prefix.hidden, positions, state, options);
  }

  async runTokenHidden(
    inputHidden: Float32Array,
    positions: Int32Array,
    state: InferenceState,
    options: SegmentRunOptions = {},
  ): Promise<SegmentHiddenResult> {
    const prefix = await this.prefix.runTokenHidden(inputHidden, positions, state, options);
    return this.gpu.runTokenHidden(prefix.hidden, positions, state, options);
  }
}
