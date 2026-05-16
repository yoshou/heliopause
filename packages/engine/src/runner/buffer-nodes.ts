import {
  cpuRunnerBuffer,
  runnerBufferToCpu,
} from "./buffer";
import {
  type ForwardCpuHiddenValue,
  type ForwardGraphContext,
  type ForwardRunnerNode,
  type ForwardValue,
  requireHidden,
} from "./graph";

export class CpuHiddenTransferNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[];
  readonly backend = "transfer" as const;

  constructor(inputId: string, id = "cpu-hidden-transfer") {
    this.id = id;
    this.deps = [inputId];
  }

  async run(context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): Promise<ForwardCpuHiddenValue> {
    const input = requireHidden(inputs, this.deps[0] ?? "");
    const hidden = await runnerBufferToCpu(input.buffer);
    return {
      kind: "cpu-hidden",
      buffer: cpuRunnerBuffer(hidden, [hidden.length / context.manifest.embeddingLength, context.manifest.embeddingLength]),
      hidden,
      perLayerInputs: input.perLayerInputs,
    };
  }
}
