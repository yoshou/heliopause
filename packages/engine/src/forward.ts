import {
  createForwardTrace,
  type InferenceState,
  type ModelSession,
  type OutputResult,
  type TimingSink,
} from "./runtime";
import type {
  ModelGraphRunner,
  ModelRunner,
} from "./runner/model-runner";
import {
  cpuRunnerBuffer,
  runnerBufferToCpu,
} from "./runner/buffer";
import {
  ForwardGraphExecutor,
  requireCpuHidden,
  type ForwardCpuHiddenValue,
  type ForwardGraphContext,
  type ForwardOutputValue,
  type ForwardRunnerNode,
  type ForwardValue,
} from "./runner/graph";
import {
  type ModelRunnerProvider,
} from "./runner/provider";
import type {
  RunnerNodePlacement,
} from "./runner/planning";
import {
  planModelPlacement,
} from "./runner/planning";
import type {
  SegmentRunnerProvider,
} from "./runner/segment-runner";

export type PrefillOptions = {
  positions?: Int32Array | number[];
  computeLogits?: boolean;
  logitsTopK?: number;
  onTiming?: TimingSink;
};

export type PrefillStateOptions = {
  positions?: Int32Array | number[];
  onTiming?: TimingSink;
};

export type PreparedHiddenPrefillStateOptions = PrefillStateOptions & {
  attentionCausal?: boolean;
};

export type NextTokenResult = {
  nextTokenId: number;
  logits?: Float32Array;
};

export type DecodeOptions = {
  position?: number;
  computeLogits?: boolean;
  logitsTopK?: number;
  onTiming?: TimingSink;
};

export type { OutputResult } from "./runtime";

export async function prefillPreparedHiddenState(
  session: ModelSession,
  state: InferenceState,
  hiddenInput: Float32Array,
  options: PreparedHiddenPrefillStateOptions = {},
): Promise<void> {
  const tokenCount = hiddenInput.length / session.manifest.embeddingLength;
  if (!Number.isInteger(tokenCount)) {
    throw new Error(`Prepared hidden shape mismatch: ${hiddenInput.length}`);
  }
  const runtime = modelRuntimeForForward(session, state);
  const positions = normalizePositions(options.positions, tokenCount);
  const trace = createForwardTrace("prefill", options.onTiming);

  if (tokenCount === 0) {
    return;
  }

  const input = new PreparedHiddenInputNode(runtime.primary.runner, hiddenInput);
  const built = buildForwardGraphFromPlan(runtime, {
    input,
    produceOutput: false,
    logitsTopK: 1,
    segmentOptions: {
      attentionCausal: options.attentionCausal ?? true,
    },
  });
  const graph = new ForwardGraphExecutor(built.nodes);
  await graph.run({
    session,
    manifest: session.manifest,
    state,
    positions,
    phase: "prefill",
    trace,
  });
  updateNextPosition(state, positions, tokenCount);
}

export async function prefillState(
  session: ModelSession,
  state: InferenceState,
  tokenIds: readonly number[],
  options: PrefillStateOptions = {},
): Promise<void> {
  const runtime = modelRuntimeForForward(session, state);
  const positions = normalizePositions(options.positions, tokenIds.length);
  const trace = createForwardTrace("prefill", options.onTiming);

  if (tokenIds.length === 0) {
    return;
  }

  const built = buildForwardGraphFromPlan(runtime, {
    tokenIds,
    produceOutput: false,
    logitsTopK: 1,
  });
  const graph = new ForwardGraphExecutor(built.nodes);
  await graph.run({
    session,
    manifest: session.manifest,
    state,
    positions,
    phase: "prefill",
    trace,
  });
  updateNextPosition(state, positions, tokenIds.length);
}

export async function prefill(
  session: ModelSession,
  state: InferenceState,
  tokenIds: readonly number[],
  options: PrefillOptions = {},
): Promise<NextTokenResult> {
  const runtime = modelRuntimeForForward(session, state);
  const positions = normalizePositions(options.positions, tokenIds.length);
  const trace = createForwardTrace("prefill", options.onTiming);

  if (tokenIds.length === 0) {
    throw new Error("Cannot prefill next token from an empty token sequence.");
  }

  const built = buildForwardGraphFromPlan(runtime, {
    tokenIds,
    produceOutput: true,
    logitsTopK: options.logitsTopK ?? 1,
  });
  const graph = new ForwardGraphExecutor(built.nodes);
  const graphResult = await graph.run({
    session,
    manifest: session.manifest,
    state,
    positions,
    phase: "prefill",
    outputTopK: options.logitsTopK ?? 1,
    trace,
  });
  updateNextPosition(state, positions, tokenIds.length);

  const output = requireGraphOutput(graphResult.values, built.outputId ?? "output").result;
  return nextTokenResultFromOutput(output, { computeLogits: options.computeLogits === true });
}

export async function decode(
  session: ModelSession,
  state: InferenceState,
  tokenId: number,
  options: DecodeOptions = {},
): Promise<NextTokenResult> {
  const runtime = modelRuntimeForForward(session, state);
  const position = options.position ?? state.nextPosition;
  const positions = new Int32Array([position]);
  const trace = createForwardTrace("decode", options.onTiming);
  const built = buildForwardGraphFromPlan(runtime, {
    tokenIds: [tokenId],
    produceOutput: true,
    logitsTopK: options.logitsTopK ?? 1,
  });
  const graph = new ForwardGraphExecutor(built.nodes);
  const graphResult = await graph.run({
    session,
    manifest: session.manifest,
    state,
    positions,
    phase: "decode",
    outputTopK: options.logitsTopK ?? 1,
    trace,
  });
  state.nextPosition = Math.max(state.nextPosition, position + 1);
  const output = requireGraphOutput(graphResult.values, built.outputId ?? "output").result;
  return nextTokenResultFromOutput(output, { computeLogits: options.computeLogits === true });
}

class PreparedHiddenInputNode implements ForwardRunnerNode {
  readonly id = "input";
  readonly deps: readonly string[] = [];
  readonly backend = "transfer" as const;
  private readonly runner: ModelRunner;
  private readonly hidden: Float32Array;

  constructor(
    runner: ModelRunner,
    hidden: Float32Array,
  ) {
    this.runner = runner;
    this.hidden = hidden;
  }

  async run(context: ForwardGraphContext): Promise<ForwardCpuHiddenValue> {
    const prepared = await this.runner.preparePreparedHiddenInput(context.session, this.hidden, context.trace);
    return {
      kind: "cpu-hidden",
      buffer: cpuRunnerBuffer(prepared.hidden, [prepared.hidden.length / context.manifest.embeddingLength, context.manifest.embeddingLength]),
      hidden: prepared.hidden,
      perLayerInputs: prepared.perLayerInputs,
    };
  }
}

class CpuHiddenAliasNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[];
  readonly backend = "transfer" as const;

  constructor(inputId: string, id = "hidden") {
    this.id = id;
    this.deps = [inputId];
  }

  run(_context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): ForwardCpuHiddenValue {
    return requireCpuHidden(inputs, this.deps[0] ?? "");
  }
}

class ModelSegmentNode implements ForwardRunnerNode {
  readonly id = "segment";
  readonly deps: readonly string[];
  readonly backend: ForwardRunnerNode["backend"];
  private readonly runner: ModelRunner;
  private readonly options: {
    attentionCausal?: boolean;
    segmentStartLayer: number;
    segmentEndLayerExclusive: number;
  };

  constructor(
    runner: ModelRunner,
    inputId: string,
    options: {
      attentionCausal?: boolean;
      segmentStartLayer?: number;
      segmentEndLayerExclusive?: number;
    } = {},
  ) {
    this.runner = runner;
    this.options = {
      ...options,
      segmentStartLayer: options.segmentStartLayer ?? 0,
      segmentEndLayerExclusive: options.segmentEndLayerExclusive ?? Number.POSITIVE_INFINITY,
    };
    this.deps = [inputId];
    this.backend = runner.provider;
  }

  async run(context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): Promise<ForwardCpuHiddenValue> {
    const input = requireCpuHidden(inputs, this.deps[0] ?? "");
    const segment = await this.runner.segmentRunner({
      session: context.session,
      state: context.state,
      manifest: context.manifest,
      epsilon: context.session.epsilon,
      segmentStartLayer: this.options.segmentStartLayer,
      segmentEndLayerExclusive: Math.min(this.options.segmentEndLayerExclusive, context.manifest.blockCount),
    });
    const inputHidden = await runnerBufferToCpu(input.buffer);
    const hidden = inputHidden.length === context.manifest.embeddingLength
      ? (await segment.runTokenHidden(inputHidden, context.positions, context.state, {
        trace: context.trace,
        perLayerInputs: input.perLayerInputs,
        attentionCausal: this.options.attentionCausal,
      })).hidden
      : (await segment.runTokensHidden(inputHidden, context.positions, context.state, {
        trace: context.trace,
        perLayerInputs: input.perLayerInputs,
        attentionCausal: this.options.attentionCausal,
      })).hidden;
    return {
      kind: "cpu-hidden",
      buffer: cpuRunnerBuffer(hidden, [hidden.length / context.manifest.embeddingLength, context.manifest.embeddingLength]),
      hidden,
      perLayerInputs: input.perLayerInputs,
    };
  }
}

class ModelOutputNode implements ForwardRunnerNode {
  readonly id = "output";
  readonly deps: readonly string[];
  readonly backend: ForwardRunnerNode["backend"];
  private readonly runner: ModelRunner;
  private readonly topK: number;

  constructor(
    runner: ModelRunner,
    inputId: string,
    topK: number,
  ) {
    this.runner = runner;
    this.topK = topK;
    this.deps = [inputId];
    this.backend = runner.provider;
  }

  async run(context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): Promise<ForwardOutputValue> {
    const input = requireCpuHidden(inputs, this.deps[0] ?? "");
    const hidden = await runnerBufferToCpu(input.buffer);
    return {
      kind: "output",
      result: await this.runner.output(context.session, hidden, {
        topK: this.topK,
        trace: context.trace,
      }),
    };
  }
}

function requireGraphOutput(values: ReadonlyMap<string, ForwardValue>, id: string): ForwardOutputValue {
  const value = values.get(id);
  if (!value || value.kind !== "output") {
    throw new Error(`Expected output graph value from ${id}`);
  }
  return value;
}

function nextTokenResultFromOutput(
  output: OutputResult,
  options: { computeLogits: boolean },
): NextTokenResult {
  const nextTokenId = output.topTokens[0]?.id ?? nextTokenFromLogits(output.logits);
  if (nextTokenId === undefined) {
    throw new Error("Forward output did not produce a next token.");
  }
  const result: NextTokenResult = { nextTokenId };
  if (options.computeLogits) {
    if (output.logits.length === 0) {
      throw new Error("Full logits were requested but the selected provider did not return them.");
    }
    result.logits = output.logits;
  }
  return result;
}

function nextTokenFromLogits(logits: Float32Array): number | undefined {
  if (logits.length === 0) {
    return undefined;
  }
  let bestId = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let id = 0; id < logits.length; id += 1) {
    const value = logits[id] ?? Number.NEGATIVE_INFINITY;
    if (value > bestValue) {
      bestId = id;
      bestValue = value;
    }
  }
  return bestId;
}

function normalizePositions(positions: PrefillStateOptions["positions"], tokenCount: number): Int32Array {
  if (!positions) {
    return Int32Array.from({ length: tokenCount }, (_, index) => index);
  }
  const output = positions instanceof Int32Array ? positions : Int32Array.from(positions);
  if (output.length !== tokenCount && output.length !== tokenCount * 4) {
    throw new Error(`Expected ${tokenCount} or ${tokenCount * 4} positions, got ${output.length}`);
  }
  return output;
}

type ModelRuntimeForForward = {
  provider: ModelRunnerProvider;
  runner: ModelRunner;
  graph: ModelGraphRunner;
};

type PlannedModelForward = {
  primary: ModelRuntimeForForward;
  providers: ReadonlyMap<SegmentRunnerProvider, ModelRuntimeForForward>;
  nodes: readonly RunnerNodePlacement[];
};

type BuiltForwardGraph = {
  nodes: ForwardRunnerNode[];
  outputId?: string;
};

function modelRuntimeForForward(session: ModelSession, state: InferenceState): PlannedModelForward {
  const providers = modelRuntimesForForward(session);
  const firstRuntime = providers.get(session.providers[0]?.name ?? "reference");
  if (!firstRuntime) {
    throw new Error("No model runner was selected.");
  }
  const plan = planModelPlacement(
    session.providers.map((provider) =>
      provider.modelResourceRequirements(session, {
        contextLength: state.contextLength,
      })
    ),
    {
      mode: "enabled",
      providerPriority: session.providers.map((provider) => provider.name),
    },
  );
  if (plan.status !== "planned" || plan.nodes.length === 0) {
    throw new Error(plan.reason ?? "No model placement was planned.");
  }
  const primaryProvider = "provider" in plan.nodes[0] ? plan.nodes[0].provider : firstRuntime.provider.name;
  return {
    primary: providers.get(primaryProvider) ?? firstRuntime,
    providers,
    nodes: plan.nodes,
  };
}

function modelRuntimesForForward(session: ModelSession): ReadonlyMap<SegmentRunnerProvider, ModelRuntimeForForward> {
  const providers = new Map<SegmentRunnerProvider, ModelRuntimeForForward>();
  for (const provider of session.providers) {
    const runner = provider.createModelRunner();
    const graph = provider.createModelGraphRunner?.() ?? runner.graph;
    if (!graph) {
      throw new Error(`Model graph runner is not available for ${provider.name}.`);
    }
    providers.set(provider.name, {
      provider,
      runner,
      graph,
    });
  }
  return providers;
}

function buildForwardGraphFromPlan(
  planned: PlannedModelForward,
  options: {
    tokenIds?: readonly number[];
    input?: ForwardRunnerNode;
    produceOutput: boolean;
    logitsTopK: number;
    segmentOptions?: { attentionCausal?: boolean };
  },
): BuiltForwardGraph {
  const nodes: ForwardRunnerNode[] = [];
  let currentId: string | undefined;
  let currentProvider: SegmentRunnerProvider | undefined;
  let outputId: string | undefined;

  if (options.input) {
    nodes.push(options.input);
    currentId = options.input.id;
  }

  for (const planNode of planned.nodes) {
    if (planNode.kind === "embedding") {
      if (!options.tokenIds) {
        continue;
      }
      const runtime = requirePlannedProvider(planned, planNode.provider);
      const embedding = runtime.graph.embeddingNode(options.tokenIds);
      nodes.push(embedding);
      currentId = embedding.id;
      currentProvider = planNode.provider;
      continue;
    }

    if (planNode.kind === "segment") {
      if (!currentId) {
        throw new Error(`Cannot run ${planNode.provider} segment without an input node.`);
      }
      if (!currentProvider && planNode.provider !== "reference" && planNode.provider !== "wasm") {
        const imported = importHiddenNode(requirePlannedProvider(planned, planNode.provider), currentId);
        nodes.push(imported);
        currentId = imported.id;
      }
      const runtime = requirePlannedProvider(planned, planNode.provider);
      const segment = segmentNode(
        runtime,
        currentId,
        planNode.startLayer,
        planNode.endLayerExclusive,
        options.segmentOptions,
      );
      nodes.push(segment);
      currentId = segment.id;
      currentProvider = planNode.provider;
      continue;
    }

    if (planNode.kind === "transfer") {
      if (!currentId) {
        throw new Error(`Cannot transfer ${planNode.from} hidden without an input node.`);
      }
      const exported = exportHiddenNode(requirePlannedProvider(planned, planNode.from), currentId);
      nodes.push(exported);
      const imported = importHiddenNode(requirePlannedProvider(planned, planNode.to), exported.id);
      nodes.push(imported);
      currentId = imported.id;
      currentProvider = planNode.to;
      continue;
    }

    if (planNode.kind === "output") {
      if (!currentId) {
        throw new Error(`Cannot produce ${planNode.provider} output without hidden input.`);
      }
      const outputRuntime = requirePlannedProvider(planned, planNode.provider);
      if (currentProvider !== planNode.provider || !outputRuntime.graph.outputNode) {
        const exported = exportHiddenNode(requirePlannedProvider(planned, currentProvider ?? planNode.provider), currentId);
        nodes.push(exported);
        currentId = exported.id;
      }
      if (options.produceOutput) {
        const output = outputNode(outputRuntime, currentId, options.logitsTopK);
        nodes.push(output);
        outputId = output.id;
      }
    }
  }

  if (!currentId) {
    throw new Error("Forward graph plan produced no hidden value.");
  }
  return { nodes, outputId };
}

function segmentNode(
  runtime: ModelRuntimeForForward,
  inputId: string,
  startLayer: number,
  endLayerExclusive: number,
  options: { attentionCausal?: boolean } = {},
): ForwardRunnerNode {
  return runtime.graph.layerSegmentNode?.(startLayer, endLayerExclusive, inputId) ??
    new ModelSegmentNode(runtime.runner, inputId, {
      ...options,
      segmentStartLayer: startLayer,
      segmentEndLayerExclusive: endLayerExclusive,
    });
}

function importHiddenNode(runtime: ModelRuntimeForForward, inputId: string): ForwardRunnerNode {
  return runtime.graph.importHiddenNode?.(inputId) ?? new CpuHiddenAliasNode(inputId, `${runtime.runner.provider}-import-hidden`);
}

function exportHiddenNode(runtime: ModelRuntimeForForward, inputId: string): ForwardRunnerNode {
  return runtime.graph.exportHiddenNode?.(inputId) ?? new CpuHiddenAliasNode(inputId, "hidden");
}

function outputNode(runtime: ModelRuntimeForForward, inputId: string, topK: number): ForwardRunnerNode {
  return runtime.graph.outputNode?.(inputId, topK) ?? new ModelOutputNode(runtime.runner, inputId, topK);
}

function requirePlannedProvider(
  planned: PlannedModelForward,
  provider: SegmentRunnerProvider,
): ModelRuntimeForForward {
  const runtime = planned.providers.get(provider);
  if (!runtime) {
    throw new Error(`No runner provider was supplied for planned ${provider} node.`);
  }
  return runtime;
}

function updateNextPosition(
  state: InferenceState,
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
