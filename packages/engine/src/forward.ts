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
  prepareGemma4PreparedHiddenInput,
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

export type PreparedHiddenPrefillOptions = PrefillOptions & {
  attentionCausal?: boolean;
};

export type PrefillResult = {
  hidden: Float32Array;
  state: Gemma4InferenceState;
  logits?: Float32Array;
  selectedTokenId?: number;
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
  selectedTokenId?: number;
  topTokens?: Array<{ id: number; value: number }>;
};

export type { OutputResult } from "./runtime";

export async function prefillGemma4PreparedHidden(
  model: Gemma4ModelInput,
  hiddenInput: Float32Array,
  options: PreparedHiddenPrefillOptions = {},
): Promise<PrefillResult> {
  const session = modelSession(model);
  if (webGpuExecutionProviderEnabled(session)) {
    throw new Error("Prepared hidden image input is only supported on the CPU/WASM runner.");
  }
  const tokenCount = hiddenInput.length / session.manifest.embeddingLength;
  if (!Number.isInteger(tokenCount)) {
    throw new Error(`Prepared hidden shape mismatch: ${hiddenInput.length}`);
  }
  const state = options.state ?? session.createInferenceState();
  const positions = normalizePositions(options.positions, tokenCount);
  const trace = createForwardTrace("prefill", options.onTiming);

  if (tokenCount === 0) {
    return { hidden: new Float32Array(), state };
  }

  const prepared = await prepareGemma4PreparedHiddenInput(session, hiddenInput, trace);
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
    attentionCausal: options.attentionCausal ?? true,
  })).hidden;
  updateNextPosition(state, positions, tokenCount);

  const result: PrefillResult = { hidden, state };
  if (options.computeLogits) {
    const output = await gemma4CpuOutput(session, hidden, {
      topK: options.logitsTopK ?? 10,
      trace,
    });
    result.logits = output.logits;
    result.selectedTokenId = output.topTokens[0]?.id;
    result.topTokens = output.topTokens;
  }
  return result;
}

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
    result.selectedTokenId = output.topTokens[0]?.id;
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
    selectedTokenId: output.topTokens[0]?.id,
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

  if (runner.supportsGpuInputPreparation()) {
    const gpu = await timedAsync(
      trace,
      "WebGPU token-id input segment",
      () => runner.runTokenIds(tokenIds, positions, state, {
        computeSelectedToken: options.computeLogits === true,
        topK: options.logitsTopK ?? 10,
      }),
    );
    updateNextPosition(state, positions, tokenIds.length);
    logWebGpuRunnerTiming(session, "prefill");
    return {
      hidden: new Float32Array(),
      state,
      selectedTokenId: options.computeLogits ? gpu.selectedTokenId : undefined,
      topTokens: options.computeLogits ? [] : undefined,
    };
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
      computeSelectedToken: options.computeLogits === true,
      topK: options.logitsTopK ?? 10,
      perLayerInputs: prepared.perLayerInputs,
    }),
  );
  updateNextPosition(state, positions, tokenIds.length);
  logWebGpuRunnerTiming(session, "prefill");

  const result: PrefillResult = {
    hidden: new Float32Array(),
    state,
  };
  if (options.computeLogits) {
    result.selectedTokenId = gpu.selectedTokenId;
    result.topTokens = [];
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

  if (runner.supportsGpuInputPreparation()) {
    const gpu = await timedAsync(
      trace,
      "WebGPU token-id input segment",
      () => runner.runTokenIds([tokenId], positions, state, {
        computeSelectedToken: true,
        topK: options.logitsTopK ?? 10,
      }),
    );
    state.nextPosition = Math.max(state.nextPosition, position + 1);
    logWebGpuRunnerTiming(session, "decode");
    return {
      hidden: new Float32Array(),
      state,
      selectedTokenId: gpu.selectedTokenId,
      topTokens: [],
    };
  }

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
      computeSelectedToken: true,
      topK: options.logitsTopK ?? 10,
      perLayerInputs: prepared.perLayerInputs,
    }),
  );
  state.nextPosition = Math.max(state.nextPosition, position + 1);
  logWebGpuRunnerTiming(session, "decode");
  return {
    hidden: new Float32Array(),
    state,
    selectedTokenId: gpu.selectedTokenId,
    topTokens: [],
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

function logWebGpuRunnerTiming(session: Gemma4ModelSession, phase: "prefill" | "decode"): void {
  if (typeof console === "undefined") {
    return;
  }
  const stats = session.cacheStats().executionProviderStats;
  if (typeof stats.webgpuRunnerCreateMs !== "number") {
    return;
  }
  const row: WebGpuRunnerTimingRow = {
    phase,
    runnerCreateMs: roundMs(stats.webgpuRunnerCreateMs),
    runtimeInitMs: roundMs(numberStat(stats.webgpuRuntimeInitMs)),
    firstRunTotalMs: roundMs(numberStat(stats.webgpuFirstRunTotalMs)),
    steadyRunMs: roundMs(numberStat(stats.webgpuSteadyRunMs)),
    runtimeResizeMs: roundMs(numberStat(stats.webgpuRuntimeResizeMs)),
    submitCount: numberStat(stats.webgpuSubmitCount),
    blockingWaitCount: numberStat(stats.webgpuBlockingWaitCount),
    readbackCount: numberStat(stats.webgpuReadbackCount),
    pipelineHit: numberStat(stats.webgpuComputePipelineCacheHits),
    pipelineMiss: numberStat(stats.webgpuComputePipelineCacheMisses),
    bindGroupHit: numberStat(stats.webgpuBindGroupCacheHits),
    bindGroupMiss: numberStat(stats.webgpuBindGroupCacheMisses),
    lastRunDurationMs: roundMs(stats.webgpuLastRunDurationMs),
    lastRunSubmitCount: numberStat(stats.webgpuLastRunSubmitCount),
    lastRunReadbackCount: numberStat(stats.webgpuLastRunReadbackCount),
    lastRunBindGroupCreates: numberStat(stats.webgpuLastRunBindGroupCreates),
    lastRunBindGroupCreateMs: roundMs(stats.webgpuLastRunBindGroupCreateMs),
    lastRunBufferCreates: numberStat(stats.webgpuLastRunBufferCreates),
    lastRunBufferCreateMs: roundMs(stats.webgpuLastRunBufferCreateMs),
    lastRunBufferCreateLabels: stringStat(stats.webgpuLastRunBufferCreateLabels),
    lastRunEncodeMs: roundMs(stats.webgpuLastRunEncodeMs),
    lastRunSubmitMs: roundMs(stats.webgpuLastRunSubmitMs),
    lastRunReadbackWaitMs: roundMs(stats.webgpuLastRunReadbackWaitMs),
    lastRunReadbackWaitMinusGpuPassMs: roundMs(stats.webgpuLastRunReadbackWaitMinusGpuPassMs),
    lastRunTimestampReadbackWaitMs: roundMs(stats.webgpuLastRunTimestampReadbackWaitMs),
    lastRunGpuPassMs: roundMs(stats.webgpuLastRunGpuPassMs),
    lastRunGpuSections: stringStat(stats.webgpuLastRunGpuSections),
    lastRunGpuTimingStatus: stringStat(stats.webgpuLastRunGpuTimingStatus),
    lastRunReadbackBytes: numberStat(stats.webgpuLastRunReadbackBytes),
    lastRunDispatchCount: numberStat(stats.webgpuLastRunDispatchCount),
    lastRunSelectedTokenId: numberStat(stats.webgpuLastRunSelectedTokenId),
    lastRunPipelineHit: numberStat(stats.webgpuLastRunComputePipelineHits),
    lastRunPipelineMiss: numberStat(stats.webgpuLastRunComputePipelineMisses),
    lastRunBindGroupHit: numberStat(stats.webgpuLastRunBindGroupHits),
    lastRunBindGroupMiss: numberStat(stats.webgpuLastRunBindGroupMisses),
    lastRunLayoutHit: numberStat(stats.webgpuLastRunBindGroupLayoutHits),
    lastRunLayoutMiss: numberStat(stats.webgpuLastRunBindGroupLayoutMisses),
  };
  const global = webGpuTimingGlobal();
  global.__heliopauseWebGpuTimings ??= [];
  global.__heliopauseWebGpuTimings.push(row);
  global.__heliopauseWebGpuTimingsTsv = () => webGpuTimingRowsToTsv(global.__heliopauseWebGpuTimings ?? []);
  if (!global.__heliopauseWebGpuTimingCopyHintShown) {
    global.__heliopauseWebGpuTimingCopyHintShown = true;
    console.log(
      "[heliopause] WebGPU timings are buffered for copying. " +
        "Use copy(globalThis.__heliopauseWebGpuTimingsTsv()) or " +
        "copy(JSON.stringify(globalThis.__heliopauseWebGpuTimings, null, 2)).",
    );
  }
  if (global.__heliopauseWebGpuTimingFlush !== undefined) {
    clearTimeout(global.__heliopauseWebGpuTimingFlush);
  }
  global.__heliopauseWebGpuTimingFlush = setTimeout(() => {
    const rows = global.__heliopauseWebGpuTimings ?? [];
    console.log(`[heliopause-webgpu-timings.tsv]\n${webGpuTimingRowsToTsv(rows)}`);
  }, 1000);
}

type WebGpuRunnerTimingRow = {
  phase: "prefill" | "decode";
  runnerCreateMs: number;
  runtimeInitMs: number;
  firstRunTotalMs: number;
  steadyRunMs: number;
  runtimeResizeMs: number;
  submitCount: number;
  blockingWaitCount: number;
  readbackCount: number;
  pipelineHit: number;
  pipelineMiss: number;
  bindGroupHit: number;
  bindGroupMiss: number;
  lastRunDurationMs: number;
  lastRunSubmitCount: number;
  lastRunReadbackCount: number;
  lastRunBindGroupCreates: number;
  lastRunBindGroupCreateMs: number;
  lastRunBufferCreates: number;
  lastRunBufferCreateMs: number;
  lastRunBufferCreateLabels: string;
  lastRunEncodeMs: number;
  lastRunSubmitMs: number;
  lastRunReadbackWaitMs: number;
  lastRunReadbackWaitMinusGpuPassMs: number;
  lastRunTimestampReadbackWaitMs: number;
  lastRunGpuPassMs: number;
  lastRunGpuSections: string;
  lastRunGpuTimingStatus: string;
  lastRunReadbackBytes: number;
  lastRunDispatchCount: number;
  lastRunSelectedTokenId: number;
  lastRunPipelineHit: number;
  lastRunPipelineMiss: number;
  lastRunBindGroupHit: number;
  lastRunBindGroupMiss: number;
  lastRunLayoutHit: number;
  lastRunLayoutMiss: number;
};

type WebGpuTimingGlobal = typeof globalThis & {
  __heliopauseWebGpuTimings?: WebGpuRunnerTimingRow[];
  __heliopauseWebGpuTimingsTsv?: () => string;
  __heliopauseWebGpuTimingFlush?: ReturnType<typeof setTimeout>;
  __heliopauseWebGpuTimingCopyHintShown?: boolean;
};

function webGpuTimingGlobal(): WebGpuTimingGlobal {
  return globalThis as WebGpuTimingGlobal;
}

function webGpuTimingRowsToTsv(rows: readonly WebGpuRunnerTimingRow[]): string {
  const headers: Array<keyof WebGpuRunnerTimingRow> = [
    "phase",
    "runnerCreateMs",
    "runtimeInitMs",
    "firstRunTotalMs",
    "steadyRunMs",
    "runtimeResizeMs",
    "submitCount",
    "blockingWaitCount",
    "readbackCount",
    "pipelineHit",
    "pipelineMiss",
    "bindGroupHit",
    "bindGroupMiss",
    "lastRunDurationMs",
    "lastRunSubmitCount",
    "lastRunReadbackCount",
    "lastRunBindGroupCreates",
    "lastRunBindGroupCreateMs",
    "lastRunBufferCreates",
    "lastRunBufferCreateMs",
    "lastRunBufferCreateLabels",
    "lastRunEncodeMs",
    "lastRunSubmitMs",
    "lastRunReadbackWaitMs",
    "lastRunReadbackWaitMinusGpuPassMs",
    "lastRunTimestampReadbackWaitMs",
    "lastRunGpuPassMs",
    "lastRunGpuSections",
    "lastRunGpuTimingStatus",
    "lastRunReadbackBytes",
    "lastRunDispatchCount",
    "lastRunSelectedTokenId",
    "lastRunPipelineHit",
    "lastRunPipelineMiss",
    "lastRunBindGroupHit",
    "lastRunBindGroupMiss",
    "lastRunLayoutHit",
    "lastRunLayoutMiss",
  ];
  return [
    headers.join("\t"),
    ...rows.map((row) => headers.map((header) => String(row[header])).join("\t")),
  ].join("\n");
}

function numberStat(value: unknown): number {
  return typeof value === "number" ? value : 0;
}

function stringStat(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function roundMs(value: unknown): number {
  return Math.round(numberStat(value) * 1000) / 1000;
}
