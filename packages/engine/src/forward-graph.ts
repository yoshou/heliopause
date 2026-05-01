import {
  forwardQwen35FullAttentionLayer,
  forwardQwen35Output,
  forwardQwen35RecurrentLayer,
  type Qwen35ForwardTrace,
  type Qwen35InferenceState,
  type Qwen35ModelSession,
  type Qwen35OutputResult,
} from "./qwen35-forward";
import type { Qwen35ModelManifest } from "./qwen35";
import { Qwen35WebGpuSegmentRunner } from "./webgpu/segment-runner";

export type ForwardRunnerBackend = "cpu" | "webgpu" | "transfer";

export type ForwardCpuHiddenValue = {
  kind: "cpu-hidden";
  hidden: Float32Array;
};

export type ForwardWebGpuHiddenValue = {
  kind: "webgpu-hidden";
  hidden: Float32Array;
  destroy?: () => void;
};

export type ForwardOutputValue = {
  kind: "output";
  result: Qwen35OutputResult;
};

export type ForwardValue =
  | ForwardCpuHiddenValue
  | ForwardWebGpuHiddenValue
  | ForwardOutputValue;

export type ForwardGraphContext = {
  session: Qwen35ModelSession;
  manifest: Qwen35ModelManifest;
  state: Qwen35InferenceState;
  positions: Int32Array;
  phase: "prefill" | "decode";
  trace?: Qwen35ForwardTrace;
};

export type ForwardRunnerNode = {
  id: string;
  deps: readonly string[];
  backend: ForwardRunnerBackend;
  run: (
    context: ForwardGraphContext,
    inputs: ReadonlyMap<string, ForwardValue>,
  ) => Promise<ForwardValue> | ForwardValue;
};

export type ForwardGraphExecutionResult = {
  order: string[];
  values: Map<string, ForwardValue>;
};

export class ForwardGraphExecutor {
  private readonly nodes: Map<string, ForwardRunnerNode>;
  private readonly order: ForwardRunnerNode[];

  constructor(nodes: readonly ForwardRunnerNode[]) {
    this.nodes = validateForwardGraph(nodes);
    this.order = topologicalSortForwardGraph(this.nodes);
  }

  executionOrder(): string[] {
    return this.order.map((node) => node.id);
  }

  async run(context: ForwardGraphContext): Promise<ForwardGraphExecutionResult> {
    const values = new Map<string, ForwardValue>();
    const cleanup: ForwardValue[] = [];
    try {
      for (const node of this.order) {
        const output = await node.run(context, values);
        values.set(node.id, output);
        cleanup.push(output);
      }
      return {
        order: this.executionOrder(),
        values,
      };
    } catch (error) {
      for (const value of cleanup.reverse()) {
        if (value.kind === "webgpu-hidden") {
          value.destroy?.();
        }
      }
      throw error;
    }
  }
}

export class EmbeddingCpuNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[] = [];
  readonly backend = "cpu" as const;

  private readonly tokenIds: readonly number[];

  constructor(tokenIds: readonly number[], id = "embedding") {
    this.tokenIds = tokenIds;
    this.id = id;
  }

  async run(context: ForwardGraphContext): Promise<ForwardCpuHiddenValue> {
    return {
      kind: "cpu-hidden",
      hidden: await context.session.readEmbeddingRows(this.tokenIds),
    };
  }
}

export class CpuLayerNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[];
  readonly backend = "cpu" as const;

  private readonly layer: number;

  constructor(layer: number, inputId: string, id = `cpu-layer:${layer}`) {
    this.layer = layer;
    this.id = id;
    this.deps = [inputId];
  }

  async run(context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): Promise<ForwardCpuHiddenValue> {
    const input = requireCpuHidden(inputs, this.deps[0] ?? "");
    const isFullAttention = context.manifest.fullAttentionLayers.includes(this.layer);
    const hidden = isFullAttention
      ? await forwardQwen35FullAttentionLayer(
        context.session,
        context.manifest,
        context.state,
        this.layer,
        input.hidden,
        context.positions,
        context.session.epsilon,
        context.trace,
      )
      : await forwardQwen35RecurrentLayer(
        context.session,
        context.manifest,
        context.state,
        this.layer,
        input.hidden,
        context.session.epsilon,
        context.trace,
      );
    return { kind: "cpu-hidden", hidden };
  }
}

export class WebGpuLayerSegmentNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[];
  readonly backend = "webgpu" as const;

  private readonly startLayer: number;
  private readonly endLayerExclusive: number;
  private runnerPromise?: Promise<Qwen35WebGpuSegmentRunner>;

  constructor(
    startLayer: number,
    endLayerExclusive: number,
    inputId: string,
    id = `webgpu-segment:${startLayer}:${endLayerExclusive}`,
  ) {
    this.startLayer = startLayer;
    this.endLayerExclusive = endLayerExclusive;
    this.id = id;
    this.deps = [inputId];
  }

  async run(context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): Promise<ForwardWebGpuHiddenValue> {
    const input = requireCpuHidden(inputs, this.deps[0] ?? "");
    this.runnerPromise ??= Qwen35WebGpuSegmentRunner.create({
      tensorReader: context.session.tensorReader,
      manifest: context.manifest,
      epsilon: context.session.epsilon,
      contextLength: context.state.contextLength,
      segmentStartLayer: this.startLayer,
      segmentEndLayerExclusive: this.endLayerExclusive,
      loadOutput: false,
    });
    const runner = await this.runnerPromise;
    const hidden = input.hidden.length === context.manifest.embeddingLength
      ? (await runner.runTokenHidden(input.hidden, context.positions, context.state)).hidden
      : (await runner.runTokensHidden(input.hidden, context.positions, context.state)).hidden;
    return {
      kind: "webgpu-hidden",
      hidden,
    };
  }
}

export class GpuToCpuHiddenTransferNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[];
  readonly backend = "transfer" as const;

  constructor(inputId: string, id = "gpu-to-cpu-hidden") {
    this.id = id;
    this.deps = [inputId];
  }

  run(_context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): ForwardCpuHiddenValue {
    const input = inputs.get(this.deps[0] ?? "");
    if (!input || input.kind !== "webgpu-hidden") {
      throw new Error(`Expected WebGPU hidden input from ${this.deps[0]}`);
    }
    return {
      kind: "cpu-hidden",
      hidden: input.hidden,
    };
  }
}

export class OutputCpuNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[];
  readonly backend = "cpu" as const;

  private readonly topK: number;

  constructor(
    inputId: string,
    topK = 10,
    id = "output",
  ) {
    this.topK = topK;
    this.id = id;
    this.deps = [inputId];
  }

  async run(context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): Promise<ForwardOutputValue> {
    const input = requireCpuHidden(inputs, this.deps[0] ?? "");
    return {
      kind: "output",
      result: await forwardQwen35Output(context.session, input.hidden, {
        topK: this.topK,
        trace: context.trace,
      }),
    };
  }
}

export function buildQwen35CpuOnlyForwardGraph(
  manifest: Qwen35ModelManifest,
  tokenIds: readonly number[],
  options: { includeOutput?: boolean; outputTopK?: number } = {},
): ForwardRunnerNode[] {
  const nodes: ForwardRunnerNode[] = [new EmbeddingCpuNode(tokenIds)];
  let currentId = "embedding";
  nodes.push(...cpuLayerRangeNodes(0, manifest.blockCount, currentId, (nextId) => {
    currentId = nextId;
  }));
  if (options.includeOutput ?? true) {
    nodes.push(new OutputCpuNode(currentId, options.outputTopK));
  }
  return nodes;
}

export function buildQwen35ManualSegmentForwardGraph(
  manifest: Qwen35ModelManifest,
  tokenIds: readonly number[],
  segment: { startLayer: number; endLayerExclusive: number },
  options: { includeOutput?: boolean; outputTopK?: number } = {},
): ForwardRunnerNode[] {
  validateLayerSegment(manifest, segment);
  const nodes: ForwardRunnerNode[] = [new EmbeddingCpuNode(tokenIds)];
  let currentId = "embedding";
  nodes.push(...cpuLayerRangeNodes(0, segment.startLayer, currentId, (nextId) => {
    currentId = nextId;
  }));
  const gpu = new WebGpuLayerSegmentNode(segment.startLayer, segment.endLayerExclusive, currentId);
  nodes.push(gpu);
  currentId = gpu.id;
  const transfer = new GpuToCpuHiddenTransferNode(currentId);
  nodes.push(transfer);
  currentId = transfer.id;
  nodes.push(...cpuLayerRangeNodes(segment.endLayerExclusive, manifest.blockCount, currentId, (nextId) => {
    currentId = nextId;
  }));
  if (options.includeOutput ?? true) {
    nodes.push(new OutputCpuNode(currentId, options.outputTopK));
  }
  return nodes;
}

export function topologicalSortForwardNodes(nodes: readonly ForwardRunnerNode[]): string[] {
  return topologicalSortForwardGraph(validateForwardGraph(nodes)).map((node) => node.id);
}

function cpuLayerRangeNodes(
  start: number,
  end: number,
  inputId: string,
  setCurrentId: (id: string) => void,
): ForwardRunnerNode[] {
  const nodes: ForwardRunnerNode[] = [];
  let currentId = inputId;
  for (let layer = start; layer < end; layer += 1) {
    const node = new CpuLayerNode(layer, currentId);
    nodes.push(node);
    currentId = node.id;
    setCurrentId(currentId);
  }
  return nodes;
}

function validateForwardGraph(nodes: readonly ForwardRunnerNode[]): Map<string, ForwardRunnerNode> {
  const byId = new Map<string, ForwardRunnerNode>();
  for (const node of nodes) {
    if (byId.has(node.id)) {
      throw new Error(`Duplicate forward graph node id: ${node.id}`);
    }
    byId.set(node.id, node);
  }
  for (const node of nodes) {
    for (const dep of node.deps) {
      if (!byId.has(dep)) {
        throw new Error(`Forward graph node ${node.id} depends on missing node ${dep}`);
      }
    }
  }
  return byId;
}

function topologicalSortForwardGraph(nodes: ReadonlyMap<string, ForwardRunnerNode>): ForwardRunnerNode[] {
  const sorted: ForwardRunnerNode[] = [];
  const permanent = new Set<string>();
  const temporary = new Set<string>();

  const visit = (node: ForwardRunnerNode) => {
    if (permanent.has(node.id)) {
      return;
    }
    if (temporary.has(node.id)) {
      throw new Error(`Forward graph cycle detected at ${node.id}`);
    }
    temporary.add(node.id);
    for (const dep of node.deps) {
      const dependency = nodes.get(dep);
      if (!dependency) {
        throw new Error(`Forward graph node ${node.id} depends on missing node ${dep}`);
      }
      visit(dependency);
    }
    temporary.delete(node.id);
    permanent.add(node.id);
    sorted.push(node);
  };

  for (const node of nodes.values()) {
    visit(node);
  }
  return sorted;
}

function requireCpuHidden(inputs: ReadonlyMap<string, ForwardValue>, id: string): ForwardCpuHiddenValue {
  const input = inputs.get(id);
  if (!input || input.kind !== "cpu-hidden") {
    throw new Error(`Expected CPU hidden input from ${id}`);
  }
  return input;
}

function validateLayerSegment(
  manifest: Qwen35ModelManifest,
  segment: { startLayer: number; endLayerExclusive: number },
): void {
  if (
    !Number.isInteger(segment.startLayer) ||
    !Number.isInteger(segment.endLayerExclusive) ||
    segment.startLayer < 0 ||
    segment.endLayerExclusive <= segment.startLayer ||
    segment.endLayerExclusive > manifest.blockCount
  ) {
    throw new Error(
      `Invalid WebGPU layer segment: ${segment.startLayer}..${segment.endLayerExclusive}`,
    );
  }
}
