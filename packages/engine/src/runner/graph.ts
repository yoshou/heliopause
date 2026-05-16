import type { ModelManifest } from "../model";
import type {
  ForwardTrace,
  InferenceState,
  ModelSession,
  OutputResult,
} from "../runtime";
import {
  destroyRunnerBuffer,
  type RunnerBuffer,
} from "./buffer";

export type ForwardRunnerBackend = "reference" | "wasm" | "webgpu" | "transfer";

export type ForwardCpuHiddenValue = {
  kind: "cpu-hidden";
  buffer: RunnerBuffer;
  hidden: Float32Array;
  perLayerInputs?: Float32Array;
};

export type ForwardProviderHiddenValue = {
  kind: "provider-hidden";
  provider: string;
  buffer: RunnerBuffer;
  perLayerInputs?: Float32Array;
  selectedTokenId?: number;
  topTokens?: Array<{ id: number; value: number }>;
};

export type ForwardOutputValue = {
  kind: "output";
  result: OutputResult;
};

export type ForwardDecodeValue = {
  kind: "decode";
  hidden: Float32Array;
  logits?: Float32Array;
  selectedTokenId?: number;
  topTokens?: Array<{ id: number; value: number }>;
};

export type ForwardValue =
  | ForwardCpuHiddenValue
  | ForwardProviderHiddenValue
  | ForwardOutputValue
  | ForwardDecodeValue;

export type ForwardGraphContext = {
  session: ModelSession;
  manifest: ModelManifest;
  state: InferenceState;
  positions: Int32Array;
  phase: "prefill" | "decode";
  outputTopK?: number;
  trace?: ForwardTrace;
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
        if (value.kind === "cpu-hidden" || value.kind === "provider-hidden") {
          destroyRunnerBuffer(value.buffer);
        }
      }
      throw error;
    }
  }
}

export function topologicalSortForwardNodes(nodes: readonly ForwardRunnerNode[]): string[] {
  return topologicalSortForwardGraph(validateForwardGraph(nodes)).map((node) => node.id);
}

export function requireCpuHidden(inputs: ReadonlyMap<string, ForwardValue>, id: string): ForwardCpuHiddenValue {
  const input = inputs.get(id);
  if (!input || input.kind !== "cpu-hidden") {
    throw new Error(`Expected CPU hidden input from ${id}`);
  }
  return input;
}

export function requireHidden(
  inputs: ReadonlyMap<string, ForwardValue>,
  id: string,
): ForwardCpuHiddenValue | ForwardProviderHiddenValue {
  const input = inputs.get(id);
  if (!input || (input.kind !== "cpu-hidden" && input.kind !== "provider-hidden")) {
    throw new Error(`Expected hidden input from ${id}`);
  }
  return input;
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
