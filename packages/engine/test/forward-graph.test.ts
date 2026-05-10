import assert from "node:assert/strict";
import test from "node:test";

import {
  buildGemma4Manifest,
  createGemma4ModelSession,
  GgufTensorReader,
} from "../src/index.ts";
import {
  ForwardGraphExecutor,
  topologicalSortForwardNodes,
  type ForwardGraphContext,
  type ForwardRunnerNode,
} from "../src/runner/graph.ts";
import {
  buildGemma4CpuOnlyForwardGraph,
  buildGemma4ManualSegmentForwardGraph,
} from "../src/runner/nodes.ts";

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
        return {
          kind: "provider-hidden",
          provider: "webgpu",
          hidden: new Float32Array([1]),
          destroy() {
            destroyed = true;
          },
        };
      },
    },
    {
      id: "fail",
      deps: ["gpu"],
      backend: "cpu",
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

test("CPU-only forward graph produces fixed Gemma4 logits for synthetic tensors", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("output.weight", [4, 3], new Float32Array([
      0.2, 0.1, -0.1, 0.3,
      -0.4, 0.2, 0.5, 0.1,
      0.3, -0.2, 0.4, -0.3,
    ])),
  ]);
  const session = createGemma4ModelSession(reader);
  const state = session.createInferenceState();
  const tokenIds = [2];
  const positions = new Int32Array([0]);
  const graph = new ForwardGraphExecutor(buildGemma4CpuOnlyForwardGraph(session.manifest, tokenIds, {
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
  const manifest = buildGemma4Manifest({
    ...minimalGguf(),
    metadata: {
      ...minimalGguf().metadata,
      "gemma4.block_count": 3,
      "gemma4.full_attention_interval": 2,
    },
  });
  const nodes = buildGemma4ManualSegmentForwardGraph(
    manifest,
    [1],
    { startLayer: 1, endLayerExclusive: 2 },
  );

  assert.deepEqual(nodes.map((item) => item.id), [
    "embedding",
    "cpu-segment:0:1",
    "webgpu-segment:1:2",
    "gpu-to-cpu-hidden",
    "cpu-segment:2:3",
    "output",
  ]);
});

function node(id: string, deps: string[] = []): ForwardRunnerNode {
  return {
    id,
    deps,
    backend: "cpu",
    run() {
      return { kind: "cpu-hidden", hidden: new Float32Array() };
    },
  };
}

function emptyContext(): ForwardGraphContext {
  const reader = tensorReaderFromTensors([]);
  const session = createGemma4ModelSession(reader);
  return {
    session,
    manifest: session.manifest,
    state: session.createInferenceState(),
    positions: new Int32Array(),
    phase: "prefill",
  };
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
