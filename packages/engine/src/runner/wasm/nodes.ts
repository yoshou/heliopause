import {
  cpuRunnerBuffer,
  runnerBufferToCpu,
} from "../buffer";
import {
  type ForwardCpuHiddenValue,
  type ForwardGraphContext,
  type ForwardOutputValue,
  type ForwardRunnerNode,
  type ForwardValue,
  requireCpuHidden,
} from "../graph";
import {
  forwardOutput,
  prepareInput,
} from "./layers";
import { WasmSegmentRunner } from "./segment-runner";

export class WasmEmbeddingNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[] = [];
  readonly backend = "wasm" as const;

  private readonly tokenIds: readonly number[];

  constructor(tokenIds: readonly number[], id = "embedding") {
    this.tokenIds = tokenIds;
    this.id = id;
  }

  async run(context: ForwardGraphContext): Promise<ForwardCpuHiddenValue> {
    const prepared = await prepareInput(context.session, this.tokenIds, context.trace);
    return {
      kind: "cpu-hidden",
      buffer: cpuRunnerBuffer(prepared.hidden, [this.tokenIds.length, context.manifest.embeddingLength]),
      hidden: prepared.hidden,
      perLayerInputs: prepared.perLayerInputs,
    };
  }
}

export class WasmLayerSegmentNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[];
  readonly backend = "wasm" as const;

  private readonly startLayer: number;
  private readonly endLayerExclusive: number;
  private runner?: WasmSegmentRunner;

  constructor(
    startLayer: number,
    endLayerExclusive: number,
    inputId: string,
    id = `wasm-segment:${startLayer}:${endLayerExclusive}`,
  ) {
    this.startLayer = startLayer;
    this.endLayerExclusive = endLayerExclusive;
    this.id = id;
    this.deps = [inputId];
  }

  async run(context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): Promise<ForwardCpuHiddenValue> {
    const input = requireCpuHidden(inputs, this.deps[0] ?? "");
    this.runner ??= new WasmSegmentRunner({
      session: context.session,
      manifest: context.manifest,
      epsilon: context.session.epsilon,
      segmentStartLayer: this.startLayer,
      segmentEndLayerExclusive: this.endLayerExclusive,
    });
    const inputHidden = await runnerBufferToCpu(input.buffer);
    const hidden = inputHidden.length === context.manifest.embeddingLength
      ? (await this.runner.runTokenHidden(inputHidden, context.positions, context.state, {
        trace: context.trace,
        perLayerInputs: input.perLayerInputs,
      })).hidden
      : (await this.runner.runTokensHidden(inputHidden, context.positions, context.state, {
        trace: context.trace,
        perLayerInputs: input.perLayerInputs,
      })).hidden;
    return {
      kind: "cpu-hidden",
      buffer: cpuRunnerBuffer(hidden, [hidden.length / context.manifest.embeddingLength, context.manifest.embeddingLength]),
      hidden,
      perLayerInputs: input.perLayerInputs,
    };
  }
}

export class WasmOutputNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[];
  readonly backend = "wasm" as const;

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
    const hidden = await runnerBufferToCpu(input.buffer);
    return {
      kind: "output",
      result: await forwardOutput(context.session, hidden, {
        topK: this.topK,
        trace: context.trace,
      }),
    };
  }
}
