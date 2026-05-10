import {
  type ForwardCpuHiddenValue,
  type ForwardGraphContext,
  type ForwardOutputValue,
  type ForwardRunnerNode,
  type ForwardValue,
  requireCpuHidden,
} from "../graph";
import { forwardGemma4Output } from "./layers";
import { Gemma4CpuSegmentRunner } from "./segment-runner";

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

export class CpuLayerSegmentNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[];
  readonly backend = "cpu" as const;

  private readonly startLayer: number;
  private readonly endLayerExclusive: number;
  private runner?: Gemma4CpuSegmentRunner;

  constructor(
    startLayer: number,
    endLayerExclusive: number,
    inputId: string,
    id = `cpu-segment:${startLayer}:${endLayerExclusive}`,
  ) {
    this.startLayer = startLayer;
    this.endLayerExclusive = endLayerExclusive;
    this.id = id;
    this.deps = [inputId];
  }

  async run(context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): Promise<ForwardCpuHiddenValue> {
    const input = requireCpuHidden(inputs, this.deps[0] ?? "");
    this.runner ??= new Gemma4CpuSegmentRunner({
      session: context.session,
      manifest: context.manifest,
      epsilon: context.session.epsilon,
      segmentStartLayer: this.startLayer,
      segmentEndLayerExclusive: this.endLayerExclusive,
    });
    const hidden = input.hidden.length === context.manifest.embeddingLength
      ? (await this.runner.runTokenHidden(input.hidden, context.positions, context.state, { trace: context.trace })).hidden
      : (await this.runner.runTokensHidden(input.hidden, context.positions, context.state, { trace: context.trace })).hidden;
    return { kind: "cpu-hidden", hidden };
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
      result: await forwardGemma4Output(context.session, input.hidden, {
        topK: this.topK,
        trace: context.trace,
      }),
    };
  }
}
