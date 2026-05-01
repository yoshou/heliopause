import assert from "node:assert/strict";
import test from "node:test";

import {
  buildQwen35Manifest,
  createQwen35ModelSession,
  GgufTensorReader,
  prefillQwen35,
} from "../src/index.ts";
import {
  ForwardGraphExecutor,
  topologicalSortForwardNodes,
  type ForwardGraphContext,
  type ForwardRunnerNode,
} from "../src/runner/graph.ts";
import {
  buildQwen35CpuOnlyForwardGraph,
  buildQwen35ManualSegmentForwardGraph,
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

test("CPU-only forward graph matches existing Qwen35 prefill output path", async () => {
  const reader = tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 8], sequence(32)),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("output.weight", [4, 3], new Float32Array([
      0.2, 0.1, -0.1, 0.3,
      -0.4, 0.2, 0.5, 0.1,
      0.3, -0.2, 0.4, -0.3,
    ])),
  ]);
  const session = createQwen35ModelSession(reader);
  const state = session.createInferenceState();
  const tokenIds = [2];
  const positions = new Int32Array([0]);
  const expected = await prefillQwen35(session, tokenIds, {
    state,
    positions,
    computeLogits: true,
    logitsTopK: 2,
  });

  const graphSession = createQwen35ModelSession(reader);
  const graphState = graphSession.createInferenceState();
  const graph = new ForwardGraphExecutor(buildQwen35CpuOnlyForwardGraph(graphSession.manifest, tokenIds, {
    outputTopK: 2,
  }));
  const result = await graph.run({
    session: graphSession,
    manifest: graphSession.manifest,
    state: graphState,
    positions,
    phase: "prefill",
  });
  const output = result.values.get("output");

  assert.equal(output?.kind, "output");
  assert.deepEqual(Array.from(output.result.logits), Array.from(expected.logits ?? []));
  assert.deepEqual(output.result.topTokens, expected.topTokens);
});

test("manual WebGPU segment graph is explicit and leaves transfer test-only", () => {
  const manifest = buildQwen35Manifest({
    ...minimalGguf(),
    metadata: {
      ...minimalGguf().metadata,
      "qwen35.block_count": 3,
      "qwen35.full_attention_interval": 2,
    },
  });
  const nodes = buildQwen35ManualSegmentForwardGraph(
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
  const session = createQwen35ModelSession(reader);
  return {
    session,
    manifest: session.manifest,
    state: session.createInferenceState(),
    positions: new Int32Array(),
    phase: "prefill",
  };
}

function minimalGguf() {
  return {
    version: 3,
    tensorCount: 0,
    metadataCount: 14,
    dataStart: 0n,
    metadata: {
      "general.architecture": "qwen35",
      "qwen35.block_count": 0,
      "qwen35.embedding_length": 4,
      "qwen35.feed_forward_length": 8,
      "qwen35.attention.head_count": 1,
      "qwen35.attention.head_count_kv": 1,
      "qwen35.attention.key_length": 2,
      "qwen35.attention.value_length": 2,
      "qwen35.context_length": 16,
      "qwen35.full_attention_interval": 1,
      "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
      "qwen35.rope.dimension_count": 2,
      "qwen35.rope.dimension_sections": {
        type: "int32",
        length: 4,
        sample: [1, 1, 0, 0],
        truncated: false,
      },
      "qwen35.rope.freq_base": 10000,
      "qwen35.ssm.conv_kernel": 4,
      "qwen35.ssm.group_count": 1,
      "qwen35.ssm.inner_size": 2,
      "qwen35.ssm.state_size": 2,
      "qwen35.ssm.time_step_rank": 1,
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
