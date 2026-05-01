import {
  type ForwardCpuHiddenValue,
  type ForwardGraphContext,
  type ForwardProviderHiddenValue,
  type ForwardRunnerNode,
  type ForwardValue,
  requireCpuHidden,
} from "../graph";
import { Qwen35WebGpuSegmentRunner } from "./segment-runner";

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

  async run(context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): Promise<ForwardProviderHiddenValue> {
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
      kind: "provider-hidden",
      provider: "webgpu",
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
    if (!input || input.kind !== "provider-hidden" || input.provider !== "webgpu") {
      throw new Error(`Expected WebGPU hidden input from ${this.deps[0]}`);
    }
    return {
      kind: "cpu-hidden",
      hidden: input.hidden,
    };
  }
}
