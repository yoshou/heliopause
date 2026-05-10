import {
  createForwardTrace,
  modelSession,
  timedAsync,
  type Gemma4InferenceState,
  type Gemma4ModelInput,
  type Gemma4ModelSession,
  type TimingSink,
} from "./runtime";
import { planGemma4RunnerPlacement } from "./runner/webgpu/planning";
import {
  gemma4CpuOutput,
  gemma4CpuSegmentRunner,
} from "./runner/cpu/execution-provider";
import {
  prepareGemma4Input,
} from "./runner/cpu/layers";
import {
  gemma4WebGpuSegmentRunner,
  webGpuExecutionProviderEnabled,
  webGpuExecutionProviderOptions,
} from "./runner/webgpu/execution-provider";

export type PrefillOptions = {
  positions?: Int32Array | number[];
  state?: Gemma4InferenceState;
  computeLogits?: boolean;
  logitsTopK?: number;
  onTiming?: TimingSink;
};

export type PrefillResult = {
  hidden: Float32Array;
  state: Gemma4InferenceState;
  logits?: Float32Array;
  topTokens?: Array<{ id: number; value: number }>;
};

export type DecodeOptions = {
  position?: number;
  state?: Gemma4InferenceState;
  logitsTopK?: number;
  onTiming?: TimingSink;
};

export type DecodeResult = {
  hidden: Float32Array;
  state: Gemma4InferenceState;
  logits?: Float32Array;
  topTokens: Array<{ id: number; value: number }>;
};

export type { OutputResult } from "./runtime";

export async function prefillGemma4(
  model: Gemma4ModelInput,
  tokenIds: readonly number[],
  options: PrefillOptions = {},
): Promise<PrefillResult> {
  const session = modelSession(model);
  if (webGpuExecutionProviderEnabled(session)) {
    return prefillGemma4HybridWebGpu(session, tokenIds, options);
  }
  const state = options.state ?? session.createInferenceState();
  const positions = normalizePositions(options.positions, tokenIds.length);
  const trace = createForwardTrace("prefill", options.onTiming);

  if (tokenIds.length === 0) {
    return { hidden: new Float32Array(), state };
  }

  const prepared = await prepareGemma4Input(session, tokenIds, trace);
  const runner = gemma4CpuSegmentRunner({
    session,
    manifest: session.manifest,
    epsilon: session.epsilon,
    segmentStartLayer: 0,
    segmentEndLayerExclusive: session.manifest.blockCount,
  });
  const hidden = (await runner.runTokensHidden(prepared.hidden, positions, state, {
    trace,
    perLayerInputs: prepared.perLayerInputs,
  })).hidden;
  updateNextPosition(state, positions, tokenIds.length);

  const result: PrefillResult = { hidden, state };
  if (options.computeLogits) {
    const output = await gemma4CpuOutput(session, hidden, {
      topK: options.logitsTopK ?? 10,
      trace,
    });
    result.logits = output.logits;
    result.topTokens = output.topTokens;
  }
  return result;
}

export async function decodeGemma4(
  model: Gemma4ModelInput,
  tokenId: number,
  options: DecodeOptions = {},
): Promise<DecodeResult> {
  const session = modelSession(model);
  if (webGpuExecutionProviderEnabled(session)) {
    return decodeGemma4HybridWebGpu(session, tokenId, options);
  }
  const state = options.state ?? session.createInferenceState();
  const position = options.position ?? state.nextPosition;
  const positions = new Int32Array([position]);
  const trace = createForwardTrace("decode", options.onTiming);
  const prepared = await prepareGemma4Input(session, [tokenId], trace);
  const runner = gemma4CpuSegmentRunner({
    session,
    manifest: session.manifest,
    epsilon: session.epsilon,
    segmentStartLayer: 0,
    segmentEndLayerExclusive: session.manifest.blockCount,
  });
  const hidden = (await runner.runTokenHidden(prepared.hidden, positions, state, {
    trace,
    perLayerInputs: prepared.perLayerInputs,
  })).hidden;
  state.nextPosition = Math.max(state.nextPosition, position + 1);
  const output = await gemma4CpuOutput(session, hidden, {
    topK: options.logitsTopK ?? 10,
    trace,
  });
  return {
    hidden,
    state,
    logits: output.logits,
    topTokens: output.topTokens,
  };
}

async function prefillGemma4HybridWebGpu(
  session: Gemma4ModelSession,
  tokenIds: readonly number[],
  options: PrefillOptions = {},
): Promise<PrefillResult> {
  const state = options.state ?? session.createInferenceState();
  const positions = normalizePositions(options.positions, tokenIds.length);
  const trace = createForwardTrace("prefill", options.onTiming);
  const runner = await webGpuSegmentRunnerForForward(session, state);
  if (tokenIds.length === 0) {
    return { hidden: new Float32Array(), state };
  }

  const prepared = await prepareGemma4Input(session, tokenIds, trace);
  const cpuPrefix = gemma4CpuSegmentRunner({
    session,
    manifest: session.manifest,
    epsilon: session.epsilon,
    segmentStartLayer: 0,
    segmentEndLayerExclusive: runner.segmentStartLayer,
  });
  const segmentInputHidden = (await cpuPrefix.runTokensHidden(prepared.hidden, positions, state, {
    trace,
    perLayerInputs: prepared.perLayerInputs,
  })).hidden;
  const gpu = await timedAsync(
    trace,
    "WebGPU segment",
    () => runner.runTokens(segmentInputHidden, positions, state, {
      computeTopK: options.computeLogits === true,
      topK: options.logitsTopK ?? 10,
      perLayerInputs: prepared.perLayerInputs,
    }),
  );
  updateNextPosition(state, positions, tokenIds.length);

  const result: PrefillResult = {
    hidden: new Float32Array(),
    state,
  };
  if (options.computeLogits) {
    result.topTokens = gpu.topTokens ?? [];
  }
  return result;
}

async function decodeGemma4HybridWebGpu(
  session: Gemma4ModelSession,
  tokenId: number,
  options: DecodeOptions = {},
): Promise<DecodeResult> {
  const state = options.state ?? session.createInferenceState();
  const position = options.position ?? state.nextPosition;
  const positions = new Int32Array([position]);
  const trace = createForwardTrace("decode", options.onTiming);
  const runner = await webGpuSegmentRunnerForForward(session, state);

  const prepared = await prepareGemma4Input(session, [tokenId], trace);
  const cpuPrefix = gemma4CpuSegmentRunner({
    session,
    manifest: session.manifest,
    epsilon: session.epsilon,
    segmentStartLayer: 0,
    segmentEndLayerExclusive: runner.segmentStartLayer,
  });
  const segmentInputHidden = (await cpuPrefix.runTokenHidden(prepared.hidden, positions, state, {
    trace,
    perLayerInputs: prepared.perLayerInputs,
  })).hidden;
  const gpu = await timedAsync(
    trace,
    "WebGPU segment",
    () => runner.runToken(segmentInputHidden, positions, state, {
      computeTopK: true,
      topK: options.logitsTopK ?? 10,
      perLayerInputs: prepared.perLayerInputs,
    }),
  );
  state.nextPosition = Math.max(state.nextPosition, position + 1);
  return {
    hidden: new Float32Array(),
    state,
    topTokens: gpu.topTokens ?? [],
  };
}

function normalizePositions(positions: PrefillOptions["positions"], tokenCount: number): Int32Array {
  if (!positions) {
    return Int32Array.from({ length: tokenCount }, (_, index) => index);
  }
  const output = positions instanceof Int32Array ? positions : Int32Array.from(positions);
  if (output.length !== tokenCount && output.length !== tokenCount * 4) {
    throw new Error(`Expected ${tokenCount} or ${tokenCount * 4} positions, got ${output.length}`);
  }
  return output;
}

async function webGpuSegmentRunnerForForward(
  session: Gemma4ModelSession,
  state: Gemma4InferenceState,
) {
  const providerOptions = webGpuExecutionProviderOptions(session);
  if (!providerOptions) {
    throw new Error("WebGPU segment runner is not enabled for this Gemma4 session.");
  }
  const segmentStartLayer = providerOptions.segmentStartLayer ??
    planGemma4RunnerPlacement(
      session.tensorReader.metadata,
      session.manifest,
      {
        mode: "enabled",
        contextLength: state.contextLength,
        memoryLimitBytes: providerOptions.memoryLimitBytes,
      },
    ).segmentStartLayer;
  return gemma4WebGpuSegmentRunner(session, state, { segmentStartLayer });
}

function updateNextPosition(
  state: Gemma4InferenceState,
  positions: Int32Array,
  tokenCount: number,
): void {
  if (tokenCount === 0) {
    return;
  }
  const tokenPositions = tokenPositionsFromMrope(positions, tokenCount);
  let nextPosition = state.nextPosition;
  for (const position of tokenPositions) {
    nextPosition = Math.max(nextPosition, position + 1);
  }
  state.nextPosition = nextPosition;
}

function tokenPositionsFromMrope(positions: Int32Array, tokenCount: number): Int32Array {
  if (positions.length === tokenCount) {
    return positions;
  }
  if (positions.length === tokenCount * 4) {
    return positions.slice(0, tokenCount);
  }
  throw new Error(`Expected ${tokenCount} or ${tokenCount * 4} positions, got ${positions.length}`);
}
