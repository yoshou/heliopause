import {
  cpuRunnerBuffer,
  providerRunnerBuffer,
  runnerBufferToCpu,
} from "../buffer";
import {
  type ForwardCpuHiddenValue,
  type ForwardGraphContext,
  type ForwardOutputValue,
  type ForwardProviderHiddenValue,
  type ForwardRunnerNode,
  type ForwardValue,
  requireCpuHidden,
  requireHidden,
} from "../graph";
import { WebGpuSegmentRunner } from "./segment-runner";
import type {
  WebGpuHiddenResult,
  WebGpuPreparedInput,
  WebGpuTokenResult,
} from "./segment-runner";
import {
  webGpuSegmentRunner,
} from "./execution-provider";
import {
  prepareWebGpuPreparedHiddenInputHandle,
} from "./model-io";

export class WebGpuPreparedHiddenInputNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[] = [];
  readonly backend = "webgpu" as const;

  private readonly hidden: Float32Array;

  constructor(hidden: Float32Array, id = "input") {
    this.hidden = hidden;
    this.id = id;
  }

  async run(context: ForwardGraphContext): Promise<ForwardProviderHiddenValue> {
    const prepared = await prepareWebGpuPreparedHiddenInputHandle(context.session, this.hidden, context.trace);
    return {
      kind: "provider-hidden",
      provider: "webgpu",
      buffer: providerRunnerBuffer(
        "webgpu",
        prepared,
        [prepared.tokenCount, context.manifest.embeddingLength],
        () => this.hidden.slice(),
        prepared.destroy,
      ),
    };
  }
}

export class WebGpuEmbeddingNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[] = [];
  readonly backend = "webgpu" as const;

  private readonly tokenIds: readonly number[];

  constructor(tokenIds: readonly number[], id = "embedding") {
    this.tokenIds = tokenIds;
    this.id = id;
  }

  async run(context: ForwardGraphContext): Promise<ForwardProviderHiddenValue> {
    const runner = await webGpuSegmentRunner(context.session, context.state, {
      segmentStartLayer: 0,
      segmentEndLayerExclusive: context.manifest.blockCount,
    });
    const prepared = await runner.prepareTokenIds(this.tokenIds);
    return {
      kind: "provider-hidden",
      provider: "webgpu",
      buffer: providerRunnerBuffer(
        "webgpu",
        prepared,
        [this.tokenIds.length, context.manifest.embeddingLength],
        () => runner.readPreparedInputHidden(prepared),
        prepared.destroy,
      ),
    };
  }
}

export class WebGpuLayerSegmentNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[];
  readonly backend = "webgpu" as const;

  private readonly startLayer: number;
  private readonly endLayerExclusive: number;

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
    const input = requireHidden(inputs, this.deps[0] ?? "");
    const runner = await webGpuSegmentRunner(context.session, context.state, {
      segmentStartLayer: this.startLayer,
      segmentEndLayerExclusive: this.endLayerExclusive,
    });
    const prepared = webGpuPreparedInput(input);
    const outputOptions = this.outputOptions(context);
    const result = prepared
      ? await this.runPreparedInput(runner, prepared, context, outputOptions)
      : await this.runCpuHidden(runner, context, input, outputOptions);
    const hidden = "hidden" in result ? result.hidden : new Float32Array();
    return {
      kind: "provider-hidden",
      provider: "webgpu",
      buffer: providerRunnerBuffer(
        "webgpu",
        hidden,
        [hidden.length / context.manifest.embeddingLength, context.manifest.embeddingLength],
        () => hidden,
      ),
      perLayerInputs: input.perLayerInputs,
      selectedTokenId: result.selectedTokenId,
      topTokens: result.topTokens,
    };
  }

  private async runPreparedInput(
    runner: WebGpuSegmentRunner,
    prepared: WebGpuPreparedInput,
    context: ForwardGraphContext,
    outputOptions: WebGpuOutputOptions,
  ): Promise<WebGpuSegmentNodeResult> {
    return outputOptions.computeSelectedToken
      ? await runner.runPreparedInput(prepared, context.positions, context.state, outputOptions)
      : await runner.runPreparedInputHidden(prepared, context.positions, context.state);
  }

  private async runCpuHidden(
    runner: WebGpuSegmentRunner,
    context: ForwardGraphContext,
    input: ForwardCpuHiddenValue | ForwardProviderHiddenValue,
    outputOptions: WebGpuOutputOptions,
  ): Promise<WebGpuSegmentNodeResult> {
    const inputHidden = await runnerBufferToCpu(input.buffer);
    if (inputHidden.length === context.manifest.embeddingLength) {
      return outputOptions.computeSelectedToken
        ? await runner.runToken(inputHidden, context.positions, context.state, {
          perLayerInputs: input.perLayerInputs,
          ...outputOptions,
        })
        : await runner.runTokenHidden(inputHidden, context.positions, context.state, {
          perLayerInputs: input.perLayerInputs,
        });
    }
    return outputOptions.computeSelectedToken
      ? await runner.runTokens(inputHidden, context.positions, context.state, {
        perLayerInputs: input.perLayerInputs,
        ...outputOptions,
      })
      : await runner.runTokensHidden(inputHidden, context.positions, context.state, {
        perLayerInputs: input.perLayerInputs,
      });
  }

  private outputOptions(context: ForwardGraphContext): WebGpuOutputOptions {
    const computeSelectedToken = context.outputTopK !== undefined &&
      this.endLayerExclusive === context.manifest.blockCount;
    return computeSelectedToken
      ? {
        computeSelectedToken: true,
        topK: context.outputTopK,
      }
      : {};
  }
}

type WebGpuOutputOptions = {
  computeSelectedToken?: true;
  topK?: number;
};

type WebGpuSegmentNodeResult = WebGpuHiddenResult | WebGpuTokenResult;

function webGpuPreparedInput(
  input: ForwardCpuHiddenValue | ForwardProviderHiddenValue,
): WebGpuPreparedInput | undefined {
  if (input.kind !== "provider-hidden" || input.provider !== "webgpu") {
    return undefined;
  }
  const handle = input.buffer.storage.kind === "provider" ? input.buffer.storage.handle : undefined;
  return isWebGpuPreparedInput(handle) ? handle : undefined;
}

function isWebGpuPreparedInput(value: unknown): value is WebGpuPreparedInput {
  return typeof value === "object" &&
    value !== null &&
    "hidden" in value &&
    "tokenCount" in value &&
    "destroy" in value;
}

export class CpuToGpuHiddenTransferNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[];
  readonly backend = "transfer" as const;

  constructor(inputId: string, id = "cpu-to-gpu-hidden") {
    this.id = id;
    this.deps = [inputId];
  }

  async run(context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): Promise<ForwardProviderHiddenValue> {
    const input = requireCpuHidden(inputs, this.deps[0] ?? "");
    const hidden = await runnerBufferToCpu(input.buffer);
    return {
      kind: "provider-hidden",
      provider: "webgpu",
      buffer: providerRunnerBuffer(
        "webgpu",
        hidden,
        [hidden.length / context.manifest.embeddingLength, context.manifest.embeddingLength],
        () => hidden,
      ),
      perLayerInputs: input.perLayerInputs,
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

  async run(context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): Promise<ForwardCpuHiddenValue> {
    const input = inputs.get(this.deps[0] ?? "");
    if (!input || input.kind !== "provider-hidden" || input.provider !== "webgpu") {
      throw new Error(`Expected WebGPU hidden input from ${this.deps[0]}`);
    }
    const hidden = await runnerBufferToCpu(input.buffer);
    return {
      kind: "cpu-hidden",
      buffer: cpuRunnerBuffer(hidden, [hidden.length / context.manifest.embeddingLength, context.manifest.embeddingLength]),
      hidden,
      perLayerInputs: input.perLayerInputs,
    };
  }
}

export class WebGpuOutputNode implements ForwardRunnerNode {
  readonly id: string;
  readonly deps: readonly string[];
  readonly backend = "webgpu" as const;

  constructor(inputId: string, _topK = 10, id = "output") {
    this.id = id;
    this.deps = [inputId];
  }

  run(_context: ForwardGraphContext, inputs: ReadonlyMap<string, ForwardValue>): ForwardOutputValue {
    const input = inputs.get(this.deps[0] ?? "");
    if (!input || input.kind !== "provider-hidden" || input.provider !== "webgpu") {
      throw new Error(`Expected WebGPU hidden input from ${this.deps[0]}`);
    }
    const topTokens = input.topTokens ?? (
      input.selectedTokenId === undefined
        ? []
        : [{ id: input.selectedTokenId, value: 0 }]
    );
    return {
      kind: "output",
      result: {
        logits: new Float32Array(),
        topTokens,
      },
    };
  }
}
