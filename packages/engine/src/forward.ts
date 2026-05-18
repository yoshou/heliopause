import {
  createForwardTrace,
  type InferenceState,
  type ModelSession,
  type OutputResult,
  type TimingSink,
} from "./runtime";
import type {
  ModelGraphNodeFactory,
  ModelRunner,
} from "./runner/model-runner";
import {
  cpuRunnerBuffer,
} from "./runner/buffer";
import {
  ForwardGraphExecutor,
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

  const output = requireGraphOutput(graphResult.outputs, built.outputId ?? "output").result;
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
  const output = requireGraphOutput(graphResult.outputs, built.outputId ?? "output").result;
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

function requireGraphOutput(values: ReadonlyMap<string, ForwardOutputValue>, id: string): ForwardOutputValue {
  const value = values.get(id);
  if (!value) {
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
  const primary = providers.get(primaryProvider);
  if (!primary) {
    throw new Error(`No model runner was supplied for planned ${primaryProvider} provider.`);
  }
  return {
    primary,
    providers,
    nodes: plan.nodes,
  };
}

function modelRuntimesForForward(session: ModelSession): ReadonlyMap<SegmentRunnerProvider, ModelRuntimeForForward> {
  const providers = new Map<SegmentRunnerProvider, ModelRuntimeForForward>();
  for (const provider of session.providers) {
    const runner = provider.createModelRunner();
    providers.set(provider.name, {
      provider,
      runner,
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
      const embedding = createEmbeddingNode(runtime, options.tokenIds);
      nodes.push(embedding);
      currentId = embedding.id;
      currentProvider = planNode.provider;
      continue;
    }

    if (planNode.kind === "segment") {
      if (!currentId) {
        throw new Error(`Cannot run ${planNode.provider} segment without an input node.`);
      }
      const runtime = requirePlannedProvider(planned, planNode.provider);
      const segment = createSegmentNode(
        runtime,
        currentId,
        planNode.startLayer,
        planNode.endLayerExclusive,
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
      if (currentProvider !== undefined && currentProvider !== planNode.from) {
        throw new Error(
          `Planned transfer ${planNode.from} -> ${planNode.to} cannot consume ${currentProvider} hidden without a matching plan.`,
        );
      }
      const exported = createExportHiddenNode(requirePlannedProvider(planned, planNode.from), currentId);
      nodes.push(exported);
      const imported = createImportHiddenNode(requirePlannedProvider(planned, planNode.to), exported.id);
      nodes.push(imported);
      currentId = imported.id;
      currentProvider = planNode.to;
      continue;
    }

    if (planNode.kind === "output") {
      if (!currentId) {
        throw new Error(`Cannot produce ${planNode.provider} output without hidden input.`);
      }
      if (!options.produceOutput) {
        continue;
      }
      if (currentProvider !== planNode.provider) {
        throw new Error(
          `Planned ${planNode.provider} output cannot consume ${currentProvider ?? "CPU"} hidden without a planned transfer.`,
        );
      }
      const output = createOutputNode(requirePlannedProvider(planned, planNode.provider), currentId, options.logitsTopK);
      nodes.push(output);
      outputId = output.id;
    }
  }

  if (!currentId) {
    throw new Error("Forward graph plan produced no hidden value.");
  }
  return { nodes, outputId };
}

function createEmbeddingNode(runtime: ModelRuntimeForForward, tokenIds: readonly number[]): ForwardRunnerNode {
  const create = graphFactory(runtime, "createEmbeddingNode", `Planned ${runtime.provider.name} embedding`);
  return create(tokenIds);
}

function createSegmentNode(
  runtime: ModelRuntimeForForward,
  inputId: string,
  startLayer: number,
  endLayerExclusive: number,
): ForwardRunnerNode {
  const create = graphFactory(runtime, "createLayerSegmentNode", `Planned ${runtime.provider.name} segment`);
  return create(startLayer, endLayerExclusive, inputId);
}

function createImportHiddenNode(runtime: ModelRuntimeForForward, inputId: string): ForwardRunnerNode {
  const create = graphFactory(runtime, "createImportHiddenNode", `Planned ${runtime.provider.name} hidden import`);
  return create(inputId);
}

function createExportHiddenNode(runtime: ModelRuntimeForForward, inputId: string): ForwardRunnerNode {
  const create = graphFactory(runtime, "createExportHiddenNode", `Planned ${runtime.provider.name} hidden export`);
  return create(inputId);
}

function createOutputNode(runtime: ModelRuntimeForForward, inputId: string, topK: number): ForwardRunnerNode {
  const create = graphFactory(runtime, "createOutputNode", `Planned ${runtime.provider.name} output`);
  return create(inputId, topK);
}

function graphFactory<TKey extends keyof ModelGraphNodeFactory>(
  runtime: ModelRuntimeForForward,
  key: TKey,
  label: string,
): ModelGraphNodeFactory[TKey] {
  const create = runtime.runner.graphNodes[key];
  if (typeof create !== "function") {
    throw new Error(`${label} requires graphNodes.${String(key)}.`);
  }
  return create;
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
