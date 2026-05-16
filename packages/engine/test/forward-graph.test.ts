import assert from "node:assert/strict";
import test from "node:test";

import {
  buildModelManifest,
  createModelSession,
  createReferenceProvider,
  createWasmProvider,
  createWebGpuProvider,
  GgufTensorReader,
  prefill,
  type ModelRunnerProvider,
} from "../src/index.ts";
import {
  cpuRunnerBuffer,
  providerRunnerBuffer,
} from "../src/runner/buffer.ts";
import {
  ForwardGraphExecutor,
  topologicalSortForwardNodes,
  type ForwardGraphContext,
  type ForwardRunnerNode,
} from "../src/runner/graph.ts";
import {
  CpuHiddenTransferNode,
} from "../src/runner/buffer-nodes.ts";
import {
  CpuToGpuHiddenTransferNode,
  GpuToCpuHiddenTransferNode,
} from "../src/runner/webgpu/nodes.ts";
import type {
  ModelManifest,
} from "../src/model.ts";
import type {
  ModelGraphRunner,
} from "../src/runner/model-runner.ts";
import type {
  RunnerNodePlacement,
} from "../src/runner/planning.ts";

test("forward graph topologically sorts dependency order", () => {
  const nodes = [
    node("output", ["layer"]),
    node("embedding"),
    node("layer", ["embedding"]),
  ];

  assert.deepEqual(topologicalSortForwardNodes(nodes), ["embedding", "layer", "output"]);
});

test("forward graph rejects duplicate, missing, and cyclic dependencies", () => {
  assert.throws(
    () => new ForwardGraphExecutor([node("a"), node("a")]),
    /Duplicate forward graph node id/,
  );
  assert.throws(
    () => new ForwardGraphExecutor([node("a", ["missing"])]),
    /depends on missing node/,
  );
  assert.throws(
    () => new ForwardGraphExecutor([node("a", ["b"]), node("b", ["a"])]),
    /cycle detected/,
  );
});

test("forward graph cleans produced WebGPU values when execution fails", async () => {
  let destroyed = false;
  const graph = new ForwardGraphExecutor([
    {
      id: "gpu",
      deps: [],
      backend: "webgpu",
      run() {
        const hidden = new Float32Array([1]);
        return {
          kind: "provider-hidden",
          provider: "webgpu",
          buffer: providerRunnerBuffer("webgpu", hidden, [1, 1], () => hidden, () => {
            destroyed = true;
          }),
        };
      },
    },
    {
      id: "fail",
      deps: ["gpu"],
      backend: "wasm",
      run() {
        throw new Error("boom");
      },
    },
  ]);

  await assert.rejects(
    graph.run(emptyContext()),
    /boom/,
  );
  assert.equal(destroyed, true);
});

test("hidden transfers preserve per-layer inputs across provider buffers", async () => {
  const hidden = new Float32Array([1, 2, 3, 4]);
  const perLayerInputs = new Float32Array([5, 6, 7, 8]);
  const graph = new ForwardGraphExecutor([
    {
      id: "source",
      deps: [],
      backend: "wasm",
      run() {
        return {
          kind: "cpu-hidden",
          buffer: cpuRunnerBuffer(hidden, [1, 4]),
          hidden,
          perLayerInputs,
        };
      },
    },
    new CpuToGpuHiddenTransferNode("source", "gpu-import"),
    new GpuToCpuHiddenTransferNode("gpu-import", "gpu-export"),
    new CpuHiddenTransferNode("gpu-export", "cpu-copy"),
  ]);

  const result = await graph.run(emptyContext());
  const value = result.values.get("cpu-copy");

  assert.equal(value?.kind, "cpu-hidden");
  assert.equal(value.perLayerInputs, perLayerInputs);
});

test("WASM-only forward graph produces fixed logits for synthetic tensors", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("output.weight", [4, 3], new Float32Array([
      0.2, 0.1, -0.1, 0.3,
      -0.4, 0.2, 0.5, 0.1,
      0.3, -0.2, 0.4, -0.3,
    ])),
  ]);
  const session = createModelSession(reader, {
    providers: [createWasmProvider()],
  });
  const state = session.createInferenceState();
  const tokenIds = [2];
  const positions = new Int32Array([0]);
  const graph = new ForwardGraphExecutor(buildWasmOnlyForwardGraph(session.manifest, tokenIds, {
    outputTopK: 2,
  }));
  const result = await graph.run({
    session,
    manifest: session.manifest,
    state,
    positions,
    phase: "prefill",
  });
  const output = result.values.get("output");

  assert.equal(output?.kind, "output");
  assertFloatArrayClose(output.result.logits, new Float32Array([
    -0.16329389810562134,
    0.5715285539627075,
    -0.5715285539627075,
  ]), 2e-5);
  assert.deepEqual(output.result.topTokens.map((token) => token.id), [1, 0]);
  assertFloatArrayClose(
    Float32Array.from(output.result.topTokens.map((token) => token.value)),
    new Float32Array([0.5715285539627075, -0.16329389810562134]),
    2e-5,
  );
});

test("manual WebGPU segment graph is explicit and leaves transfer test-only", () => {
  const manifest = buildModelManifest({
    ...minimalGguf(),
    metadata: {
      ...minimalGguf().metadata,
      "gemma4.block_count": 3,
      "gemma4.full_attention_interval": 2,
    },
  });
  const nodes = buildManualSegmentForwardGraph(
    manifest,
    [1],
    { startLayer: 1, endLayerExclusive: 2 },
  );

  assert.deepEqual(nodes.map((item) => item.id), [
    "embedding",
    "wasm-segment:0:1",
    "webgpu-segment:1:2",
    "gpu-to-cpu-hidden",
    "wasm-segment:2:3",
    "output",
  ]);
});

test("prefill follows planned provider graph instead of a full primary segment", async () => {
  const executed: string[] = [];
  const reader = tensorReaderFromGguf({
    ...minimalGguf(),
    metadata: {
      ...minimalGguf().metadata,
      "gemma4.block_count": 3,
      "gemma4.full_attention_interval": 2,
    },
  });
  const session = createModelSession(reader, {
    providers: [
      fakeProvider("webgpu", executed, [
        { kind: "embedding", provider: "wasm" },
        {
          kind: "segment",
          provider: "wasm",
          startLayer: 0,
          endLayerExclusive: 1,
          layerCount: 1,
          weightBytes: 0,
          cacheBytes: 0,
        },
        { kind: "transfer", from: "wasm", to: "webgpu", via: "cpu", value: "hidden" },
        {
          kind: "segment",
          provider: "webgpu",
          startLayer: 1,
          endLayerExclusive: 3,
          layerCount: 2,
          weightBytes: 0,
          cacheBytes: 0,
        },
        { kind: "output", provider: "webgpu" },
      ]),
      fakeProvider("wasm", executed),
    ],
  });

  await prefill(session, [1], { computeLogits: true });

  assert.deepEqual(executed, [
    "wasm.embedding",
    "wasm.segment:0:1",
    "wasm.export",
    "webgpu.import",
    "webgpu.segment:1:3",
    "webgpu.output",
  ]);
});

test("prefill keeps WebGPU-only planned graph on WebGPU", async () => {
  const executed: string[] = [];
  const reader = tensorReaderFromGguf({
    ...minimalGguf(),
    metadata: {
      ...minimalGguf().metadata,
      "gemma4.block_count": 3,
      "gemma4.full_attention_interval": 2,
    },
  });
  const session = createModelSession(reader, {
    providers: [
      fakeProvider("webgpu", executed, [
        { kind: "embedding", provider: "webgpu" },
        {
          kind: "segment",
          provider: "webgpu",
          startLayer: 0,
          endLayerExclusive: 3,
          layerCount: 3,
          weightBytes: 0,
          cacheBytes: 0,
        },
        { kind: "output", provider: "webgpu" },
      ]),
      fakeProvider("wasm", executed),
    ],
  });

  await prefill(session, [1], { computeLogits: true });

  assert.deepEqual(executed, [
    "webgpu.embedding",
    "webgpu.segment:0:3",
    "webgpu.output",
  ]);
});

function node(id: string, deps: string[] = []): ForwardRunnerNode {
  return {
    id,
    deps,
    backend: "wasm",
    run() {
      const hidden = new Float32Array();
      return { kind: "cpu-hidden", buffer: cpuRunnerBuffer(hidden, [0, 0]), hidden };
    },
  };
}

function fakeProvider(
  name: "wasm" | "webgpu",
  executed: string[],
  planNodes?: RunnerNodePlacement[],
): ModelRunnerProvider {
  const hidden = () => new Float32Array([1, 0, 0, 0]);
  const hiddenNode = (id: string, deps: string[] = []): ForwardRunnerNode => ({
    id,
    deps,
    backend: name,
    run() {
      executed.push(id);
      const value = hidden();
      return {
        kind: "cpu-hidden",
        buffer: cpuRunnerBuffer(value, [1, 4]),
        hidden: value,
      };
    },
  });
  return {
    name,
    createModelRunner: () => ({
      provider: name,
      async prepareInput() {
        return { hidden: hidden() };
      },
      async preparePreparedHiddenInput(_session, value) {
        return { hidden: value };
      },
      segmentRunner() {
        throw new Error("fake segment runner should not be used");
      },
      async output() {
        return {
          logits: new Float32Array([1]),
          topTokens: [{ id: 0, value: 1 }],
        };
      },
    }),
    createModelGraphRunner: () => ({
      embeddingNode: () => hiddenNode(`${name}.embedding`),
      layerSegmentNode: (startLayer, endLayerExclusive, inputId) =>
        hiddenNode(`${name}.segment:${startLayer}:${endLayerExclusive}`, [inputId]),
      importHiddenNode: (inputId) => hiddenNode(`${name}.import`, [inputId]),
      exportHiddenNode: (inputId) => hiddenNode(`${name}.export`, [inputId]),
      outputNode: (inputId) => ({
        id: `${name}.output`,
        deps: [inputId],
        backend: name,
        run() {
          executed.push(`${name}.output`);
          return {
            kind: "output",
            result: {
              logits: new Float32Array([1]),
              topTokens: [{ id: 0, value: 1 }],
            },
          };
        },
      }),
    }),
    planModelPlacement: planNodes
      ? () => ({
        status: "planned",
        mode: "enabled",
        memoryLimitBytes: 0,
        enabled: false,
        outputBytes: 0,
        fixedBytes: 0,
        scratchBytes: 0,
        selectedLayerCount: 0,
        wasmSegmentLayerCount: 1,
        webGpuSegmentLayerCount: 2,
        webGpuWeightBytes: 0,
        webGpuCacheBytes: 0,
        estimatedResidentBytes: 0,
        remainingBytes: 0,
        webGpuSelectedLayers: [],
        segments: [],
        nodes: planNodes,
        copyAuditExpectations: {
          decodeTensorReads: 0,
          segmentIntermediateReadbacks: 0,
          logitsReadbacks: 0,
          expectedBoundaryUploads: 1,
          expectedTokenReadbacks: 0,
          expectedSelectedTokenReadbacks: 1,
        },
      })
      : undefined,
  };
}

function tensorReaderFromGguf(gguf: ReturnType<typeof minimalGguf>) {
  return new GgufTensorReader(gguf, {
    async read() {
      return new Uint8Array();
    },
  });
}

function emptyContext(): ForwardGraphContext {
  const reader = tensorReaderFromTensors([]);
  const session = createModelSession(reader, { providers: [createReferenceProvider()] });
  return {
    session,
    manifest: session.manifest,
    state: session.createInferenceState(),
    positions: new Int32Array(),
    phase: "prefill",
  };
}

function buildWasmOnlyForwardGraph(
  manifest: ModelManifest,
  tokenIds: readonly number[],
  options: { includeOutput?: boolean; outputTopK?: number } = {},
): ForwardRunnerNode[] {
  const wasm = requireModelGraphRunner(createWasmProvider(), "wasm");
  const nodes: ForwardRunnerNode[] = [requireNode(wasm.embeddingNode, "wasm embedding")(tokenIds)];
  let currentId = "embedding";
  const segment = maybeLayerSegmentNode(wasm, 0, manifest.blockCount, currentId);
  if (segment) {
    nodes.push(segment);
    currentId = segment.id;
  }
  if (options.includeOutput ?? true) {
    nodes.push(requireNode(wasm.outputNode, "wasm output")(currentId, options.outputTopK));
  }
  return nodes;
}

function buildManualSegmentForwardGraph(
  manifest: ModelManifest,
  tokenIds: readonly number[],
  segment: { startLayer: number; endLayerExclusive: number },
  options: { includeOutput?: boolean; outputTopK?: number } = {},
): ForwardRunnerNode[] {
  validateLayerSegment(manifest, segment);
  const wasm = requireModelGraphRunner(createWasmProvider(), "wasm");
  const webgpu = requireModelGraphRunner(createWebGpuProvider(), "webgpu");
  const nodes: ForwardRunnerNode[] = [requireNode(wasm.embeddingNode, "wasm embedding")(tokenIds)];
  let currentId = "embedding";
  const prefix = maybeLayerSegmentNode(wasm, 0, segment.startLayer, currentId);
  if (prefix) {
    nodes.push(prefix);
    currentId = prefix.id;
  }
  const gpu = requireNode(webgpu.layerSegmentNode, "webgpu segment")(
    segment.startLayer,
    segment.endLayerExclusive,
    currentId,
  );
  nodes.push(gpu);
  currentId = gpu.id;
  const transfer = requireNode(webgpu.exportHiddenNode, "provider hidden export")(currentId);
  nodes.push(transfer);
  currentId = transfer.id;
  const suffix = maybeLayerSegmentNode(wasm, segment.endLayerExclusive, manifest.blockCount, currentId);
  if (suffix) {
    nodes.push(suffix);
    currentId = suffix.id;
  }
  if (options.includeOutput ?? true) {
    nodes.push(requireNode(wasm.outputNode, "wasm output")(currentId, options.outputTopK));
  }
  return nodes;
}

function maybeLayerSegmentNode(
  provider: ModelGraphRunner,
  start: number,
  end: number,
  inputId: string,
): ForwardRunnerNode | undefined {
  return end > start
    ? requireNode(provider.layerSegmentNode, "provider segment")(start, end, inputId)
    : undefined;
}

function requireModelGraphRunner(
  provider: ModelRunnerProvider,
  name: string,
): ModelGraphRunner {
  const graph = provider.createModelGraphRunner?.();
  if (!graph) {
    throw new Error(`Model graph provider is not available for ${name}.`);
  }
  return graph;
}

function requireNode<T extends (...args: never[]) => ForwardRunnerNode>(
  create: T | undefined,
  label: string,
): T {
  if (!create) {
    throw new Error(`Model graph node is not available for ${label}.`);
  }
  return create;
}

function validateLayerSegment(
  manifest: ModelManifest,
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

function assertFloatArrayClose(actual: Float32Array, expected: Float32Array, tolerance: number): void {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(
      Math.abs((actual[index] ?? 0) - (expected[index] ?? 0)) <= tolerance,
      `index ${index}: expected ${expected[index]}, got ${actual[index]}`,
    );
  }
}

function minimalGguf() {
  return {
    version: 3,
    tensorCount: 0,
    metadataCount: 14,
    dataStart: 0n,
    metadata: {
      "general.architecture": "gemma4",
      "gemma4.block_count": 0,
      "gemma4.embedding_length": 4,
      "gemma4.feed_forward_length": 8,
      "gemma4.attention.head_count": 1,
      "gemma4.attention.head_count_kv": 1,
      "gemma4.attention.key_length": 2,
      "gemma4.attention.value_length": 2,
      "gemma4.context_length": 16,
      "gemma4.full_attention_interval": 1,
      "gemma4.attention.layer_norm_rms_epsilon": 1e-6,
      "gemma4.rope.dimension_count": 2,
      "gemma4.rope.dimension_sections": {
        type: "int32",
        length: 4,
        sample: [1, 1, 0, 0],
        truncated: false,
      },
      "gemma4.rope.freq_base": 10000,
    },
    tensors: [],
  };
}

function tensorReaderFromTensors(
  tensors: Array<{
    name: string;
    dimensions: number[];
    type: "F32";
    bytes: Uint8Array;
  }>,
) {
  let offset = 0n;
  const bytes: Uint8Array[] = [];
  const infos = tensors.map((tensor, typeId) => {
    const dataOffset = offset;
    offset += BigInt(tensor.bytes.byteLength);
    bytes.push(tensor.bytes);
    return {
      name: tensor.name,
      dimensions: tensor.dimensions,
      type: tensor.type,
      typeId,
      offset: 0n,
      dataOffset,
    };
  });
  const data = concatBytes(bytes);
  const gguf = {
    ...minimalGguf(),
    tensorCount: infos.length,
    tensors: infos,
  };
  return new GgufTensorReader(gguf, {
    async read(readOffset, length) {
      return data.subarray(Number(readOffset), Number(readOffset) + length);
    },
  });
}

function f32Tensor(name: string, dimensions: number[], values: Float32Array) {
  return {
    name,
    dimensions,
    type: "F32" as const,
    bytes: new Uint8Array(values.buffer, values.byteOffset, values.byteLength).slice(),
  };
}

function sequence(length: number): Float32Array {
  const values = new Float32Array(length);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = (index % 7 - 3) * 0.1;
  }
  return values;
}

function concatBytes(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const output = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}
