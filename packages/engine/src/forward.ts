import {
  createForwardTrace,
  modelSession,
  timedAsync,
  type Qwen35InferenceState,
  type Qwen35ModelInput,
  type Qwen35ModelSession,
  type TimingSink,
} from "./runtime";
import { planQwen35RunnerPlacement } from "./runner/planning";
import {
  qwen35CpuOutput,
  qwen35CpuSegmentRunner,
} from "./runner/cpu/execution-provider";
import {
  qwen35WebGpuSegmentRunner,
  webGpuExecutionProviderEnabled,
  webGpuExecutionProviderOptions,
} from "./runner/webgpu/execution-provider";

export type PrefillOptions = {
  positions?: Int32Array | number[];
  state?: Qwen35InferenceState;
  computeLogits?: boolean;
  logitsTopK?: number;
  onTiming?: TimingSink;
};

export type PrefillResult = {
  hidden: Float32Array;
  state: Qwen35InferenceState;
  logits?: Float32Array;
  topTokens?: Array<{ id: number; value: number }>;
};

export type DecodeOptions = {
  position?: number;
  state?: Qwen35InferenceState;
  logitsTopK?: number;
  onTiming?: TimingSink;
};

export type DecodeResult = {
  hidden: Float32Array;
  state: Qwen35InferenceState;
  logits?: Float32Array;
  topTokens: Array<{ id: number; value: number }>;
};

export type { OutputResult } from "./runtime";

export async function prefillQwen35(
  model: Qwen35ModelInput,
  tokenIds: readonly number[],
  options: PrefillOptions = {},
): Promise<PrefillResult> {
  const session = modelSession(model);
  if (webGpuExecutionProviderEnabled(session)) {
    return prefillQwen35HybridWebGpu(session, tokenIds, options);
  }
  const state = options.state ?? session.createInferenceState();
  const positions = normalizePositions(options.positions, tokenIds.length);
  const trace = createForwardTrace("prefill", options.onTiming);

  if (tokenIds.length === 0) {
    return { hidden: new Float32Array(), state };
  }

  const embedding = await timedAsync(trace, "embedding read", () => session.readEmbeddingRows(tokenIds));
  const runner = qwen35CpuSegmentRunner({
    session,
    manifest: session.manifest,
    epsilon: session.epsilon,
    segmentStartLayer: 0,
    segmentEndLayerExclusive: session.manifest.blockCount,
  });
  const hidden = (await runner.runTokensHidden(embedding, positions, state, { trace })).hidden;
  updateNextPosition(state, positions, tokenIds.length);

  const result: PrefillResult = { hidden, state };
  if (options.computeLogits) {
    const output = await qwen35CpuOutput(session, hidden, {
      topK: options.logitsTopK ?? 10,
      trace,
    });
    result.logits = output.logits;
    result.topTokens = output.topTokens;
  }
  return result;
}

export async function decodeQwen35(
  model: Qwen35ModelInput,
  tokenId: number,
  options: DecodeOptions = {},
): Promise<DecodeResult> {
  const session = modelSession(model);
  if (webGpuExecutionProviderEnabled(session)) {
    return decodeQwen35HybridWebGpu(session, tokenId, options);
  }
  const state = options.state ?? session.createInferenceState();
  const position = options.position ?? state.nextPosition;
  const positions = new Int32Array([position]);
  const trace = createForwardTrace("decode", options.onTiming);
  const embedding = await timedAsync(trace, "embedding read", () => session.readEmbeddingRows([tokenId]));
  const runner = qwen35CpuSegmentRunner({
    session,
    manifest: session.manifest,
    epsilon: session.epsilon,
    segmentStartLayer: 0,
    segmentEndLayerExclusive: session.manifest.blockCount,
  });
  const hidden = (await runner.runTokenHidden(embedding, positions, state, { trace })).hidden;
  state.nextPosition = Math.max(state.nextPosition, position + 1);
  const output = await qwen35CpuOutput(session, hidden, {
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

async function prefillQwen35HybridWebGpu(
  session: Qwen35ModelSession,
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

  const embedding = await timedAsync(trace, "embedding read", () => session.readEmbeddingRows(tokenIds));
  const cpuPrefix = qwen35CpuSegmentRunner({
    session,
    manifest: session.manifest,
    epsilon: session.epsilon,
    segmentStartLayer: 0,
    segmentEndLayerExclusive: runner.segmentStartLayer,
  });
  const segmentInputHidden = (await cpuPrefix.runTokensHidden(embedding, positions, state, { trace })).hidden;
  const gpu = await timedAsync(
    trace,
    "WebGPU segment",
    () => runner.runTokens(segmentInputHidden, positions, state, {
      computeTopK: options.computeLogits === true,
      topK: options.logitsTopK ?? 10,
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

async function decodeQwen35HybridWebGpu(
  session: Qwen35ModelSession,
  tokenId: number,
  options: DecodeOptions = {},
): Promise<DecodeResult> {
  const state = options.state ?? session.createInferenceState();
  const position = options.position ?? state.nextPosition;
  const positions = new Int32Array([position]);
  const trace = createForwardTrace("decode", options.onTiming);
  const runner = await webGpuSegmentRunnerForForward(session, state);

  const embedding = await timedAsync(trace, "embedding read", () => session.readEmbeddingRows([tokenId]));
  const cpuPrefix = qwen35CpuSegmentRunner({
    session,
    manifest: session.manifest,
    epsilon: session.epsilon,
    segmentStartLayer: 0,
    segmentEndLayerExclusive: runner.segmentStartLayer,
  });
  const segmentInputHidden = (await cpuPrefix.runTokenHidden(embedding, positions, state, { trace })).hidden;
  const gpu = await timedAsync(
    trace,
    "WebGPU segment",
    () => runner.runToken(segmentInputHidden, positions, state, {
      computeTopK: true,
      topK: options.logitsTopK ?? 10,
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
  session: Qwen35ModelSession,
  state: Qwen35InferenceState,
) {
  const providerOptions = webGpuExecutionProviderOptions(session);
  if (!providerOptions) {
    throw new Error("WebGPU segment runner is not enabled for this Qwen35 session.");
  }
  const segmentStartLayer = providerOptions.segmentStartLayer ??
    planQwen35RunnerPlacement(
      session.tensorReader.metadata,
      session.manifest,
      {
        mode: "enabled",
        browserGate: "passed",
        contextLength: state.contextLength,
        memoryLimitBytes: providerOptions.memoryLimitBytes,
      },
    ).segmentStartLayer;
  return qwen35WebGpuSegmentRunner(session, state, { segmentStartLayer });
}

function updateNextPosition(
  state: Qwen35InferenceState,
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
