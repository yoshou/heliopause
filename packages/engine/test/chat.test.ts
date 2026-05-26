import assert from "node:assert/strict";
import test from "node:test";

import {
  applyChatTemplate,
  buildTokenizer,
  createFileGgufTensorReader,
  createReferenceProvider,
  createModelSession,
  generateChatTurn,
  generateChatCompletion,
  GgufTensorReader,
  prefillChatMessages,
  stripThinking,
  type GgufMetadata,
  type ModelRunnerProvider,
  type ProviderResourceRequirements,
  type SegmentRunnerProvider,
} from "../src/index.ts";
import {
  cpuRunnerBuffer,
} from "../src/runner/buffer.ts";
import type {
  ForwardRunnerNode,
} from "../src/runner/graph.ts";
import type {
  ModelRunner,
} from "../src/runner/model-runner.ts";

// Gemma 4 chat template snapshot: tmp/gemma4-chat-template-2026-05-22.jinja
// sha256: 2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082

test("chat template formats history and disables thinking by default", () => {
  assert.equal(
    applyChatTemplate([
      { role: "system", content: "Be concise." },
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi" },
    ]),
    "<|turn>system\nBe concise.<turn|>\n" +
      "<|turn>user\nHello<turn|>\n" +
      "<|turn>model\nHi<turn|>\n" +
      "<|turn>model\n",
  );
});

test("chat template can leave thinking enabled", () => {
  assert.equal(
    applyChatTemplate([
      { role: "user", content: "Hello" },
    ], { enableThinking: true }),
    "<|turn>user\nHello<turn|>\n" +
      "<|think|>\n<|turn>model\n",
  );
});

test("chat template serializes Gemma 4 native tool declarations", () => {
  assert.equal(
    applyChatTemplate([
      {
        role: "system",
        content: "",
        toolDeclarations: [{
          type: "function",
          function: {
            name: "sandbox_list_files",
            description: "List files under /workspace.",
            parameters: {
              type: "object",
              properties: {
                path: {
                  type: "string",
                  description: "Directory path to list.",
                },
              },
              required: ["path"],
            },
          },
        }],
      },
    ], { addGenerationPrompt: false }),
    "<|turn>system\n" +
      '<|tool>declaration:sandbox_list_files{description:<|"|>List files under /workspace.<|"|>,parameters:{properties:{path:{description:<|"|>Directory path to list.<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>path<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>' +
      "<turn|>\n",
  );
});

test("chat template serializes Gemma 4 native tool calls and responses", () => {
  assert.equal(
    applyChatTemplate([
      {
        role: "assistant",
        tool_calls: [{
          id: "tool_1",
          type: "function",
          function: {
            name: "sandbox_list_files",
            arguments: { path: "/workspace" },
          },
        }],
      },
      {
        role: "tool",
        tool_call_id: "tool_1",
        content: {
          ok: true,
          content: { entries: ["notes.md"] },
        },
      },
    ], { addGenerationPrompt: false }),
    "<|turn>model\n" +
      '<|tool_call>call:sandbox_list_files{path:<|"|>/workspace<|"|>}<tool_call|>' +
      '<|tool_response>response:sandbox_list_files{content:{entries:[<|"|>notes.md<|"|>]},ok:true}<tool_response|>' +
      "<turn|>\n",
  );
});

test("chat template can leave a model tool turn open for incremental tool responses", () => {
  assert.equal(
    applyChatTemplate([
      {
        role: "tool",
        tool_call_id: "tool_1",
        name: "sandbox_list_files",
        content: { ok: true },
      },
    ], { addGenerationPrompt: false, closeFinalTurn: false }),
    '<|tool_response>response:sandbox_list_files{ok:true}<tool_response|>',
  );
});

test("chat template serializes consecutive tool responses in one model turn", () => {
  assert.equal(
    applyChatTemplate([
      {
        role: "tool",
        tool_call_id: "tool_1_1",
        name: "sandbox_read_file",
        content: { ok: true, content: { path: "a.txt" } },
      },
      {
        role: "tool",
        tool_call_id: "tool_1_2",
        name: "sandbox_read_file",
        content: { ok: true, content: { path: "b.txt" } },
      },
    ], { addGenerationPrompt: false }),
    '<|tool_response>response:sandbox_read_file{content:{path:<|"|>a.txt<|"|>},ok:true}<tool_response|>' +
      '<|tool_response>response:sandbox_read_file{content:{path:<|"|>b.txt<|"|>},ok:true}<tool_response|>' +
      "<turn|>\n",
  );
});

test("tokenizer preserves special chat tokens and detokenizes text", () => {
  const tokenizer = buildTokenizer(tokenizerGguf([
    "H",
    "e",
    "l",
    "o",
    ",",
    "Ġ",
    "w",
    "r",
    "d",
    "!",
    "<|im_start|>",
    "<|im_end|>",
  ]));

  assert.deepEqual(tokenizer.tokenize("Hello, world!"), [0, 1, 2, 2, 3, 4, 5, 6, 3, 7, 2, 8, 9]);
  assert.deepEqual(tokenizer.tokenize("<|im_start|>H<|im_end|>"), [10, 0, 11]);
  assert.equal(tokenizer.detokenize([0, 1, 2, 2, 3, 4, 5, 6, 3, 7, 2, 8, 9]), "Hello, world!");
  assert.equal(tokenizer.detokenize([10, 0, 11]), "<|im_start|>H<|im_end|>");
});

test("tokenizer accepts GGUF model metadata without pre field", () => {
  const tokenizer = buildTokenizer(tokenizerGguf([
    "H",
    "e",
    "l",
    "o",
  ], undefined, { model: "gemma4", includePre: false }));

  assert.deepEqual(tokenizer.tokenize("Hello"), [0, 1, 2, 2, 3]);
  assert.equal(tokenizer.detokenize([0, 1, 2, 2, 3]), "Hello");
});

test("tokenizer uses SPM-style BPE and byte fallback for model metadata", () => {
  const tokenizer = buildTokenizer(tokenizerGguf([
    "▁",
    "H",
    "i",
    "▁H",
    "▁Hi",
    "<0xE3>",
    "<0x81>",
    "<0x82>",
  ], undefined, {
    model: "gemma4",
    includePre: false,
    merges: ["▁ H", "▁H i"],
  }));

  assert.deepEqual(tokenizer.tokenize(" Hi"), [4]);
  assert.deepEqual(tokenizer.tokenize("あ"), [5, 6, 7]);
  assert.equal(tokenizer.detokenize([4, 5, 6, 7]), " Hiあ");
});

test("chat generation stops on turn token", async () => {
  const tokenizer = {
    eosTokenId: 3,
    tokenize() {
      return [2];
    },
    detokenize(tokenIds: readonly number[]) {
      return tokenIds.map((id) => id === 0 ? "A" : "").join("");
    },
    idToToken(id: number) {
      return id === 0 ? "A" : id === 3 ? "<turn|>" : undefined;
    },
    tokenToId(token: string) {
      return token === "<turn|>" ? 3 : undefined;
    },
  };
  const session = createModelSession(tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 12], new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ])),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("output.weight", [4, 12], new Float32Array([
      0, 0, 1, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      1, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ])),
  ]), { providers: [createReferenceProvider()] });

  const chunks = [];
  for await (const chunk of generateChatCompletion(
    session,
    tokenizer,
    [{ role: "user", content: "A" }],
    { maxNewTokens: 4, doSample: false },
  )) {
    chunks.push(chunk);
  }

  assert.deepEqual(chunks.map((chunk) => chunk.text), ["A"]);
});

test("chat generation stops on official Gemma 4 EOS token ids", async () => {
  const tokenizer = {
    tokenize() {
      return [0];
    },
    detokenize(tokenIds: readonly number[]) {
      return tokenIds.map((id) => `token-${id}`).join("");
    },
    idToToken(id: number) {
      return `token-${id}`;
    },
    tokenToId() {
      return undefined;
    },
  };
  const embedding = new Float32Array(4 * 51);
  embedding[0] = 1;
  const output = new Float32Array(4 * 51);
  output[50 * 4] = 10;
  const session = createModelSession(tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 51], embedding),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("output.weight", [4, 51], output),
  ]), { providers: [createReferenceProvider()] });

  const chunks = [];
  for await (const chunk of generateChatCompletion(
    session,
    tokenizer,
    [{ role: "user", content: "Stop." }],
    { maxNewTokens: 4, doSample: false },
  )) {
    chunks.push(chunk);
  }

  assert.deepEqual(chunks, []);
});

test("chat generation requests official topK by default and top-1 for greedy across providers", async () => {
  for (const provider of ["reference", "wasm", "webgpu"] as const) {
    const defaultTopKs: number[] = [];
    const defaultSession = sessionWithCapturingProvider(provider, defaultTopKs);
    await generateChatTurn(
      defaultSession,
      singleTokenTokenizer,
      defaultSession.createInferenceState(),
      "Hello",
      { maxNewTokens: 1, appendTurnEnd: false },
    );
    assert.deepEqual(defaultTopKs, [64, 64], provider);

    const greedyTopKs: number[] = [];
    const greedySession = sessionWithCapturingProvider(provider, greedyTopKs);
    await generateChatTurn(
      greedySession,
      singleTokenTokenizer,
      greedySession.createInferenceState(),
      "Hello",
      { maxNewTokens: 1, appendTurnEnd: false, doSample: false },
    );
    assert.deepEqual(greedyTopKs, [1, 1], provider);
  }
});

test("stateful chat turn pre-fills only the new turn suffix", async () => {
  const tokenizedTexts: string[] = [];
  const tokenizer = {
    eosTokenId: 3,
    tokenize(text: string) {
      tokenizedTexts.push(text);
      if (text === "<turn|>\n") {
        return [3];
      }
      if (text.startsWith("<|turn>model\n")) {
        return [2];
      }
      return [1];
    },
    detokenize(tokenIds: readonly number[]) {
      return tokenIds.map((id) => id === 0 ? "A" : "").join("");
    },
    idToToken(id: number) {
      return id === 0 ? "A" : id === 3 ? "<turn|>" : undefined;
    },
    tokenToId(token: string) {
      return token === "<turn|>" ? 3 : undefined;
    },
  };
  const session = createModelSession(tensorReaderFromTensors([
    f32Tensor("token_embd.weight", [4, 12], new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ])),
    f32Tensor("output_norm.weight", [4], new Float32Array([1, 1, 1, 1])),
    f32Tensor("output.weight", [4, 4], new Float32Array([
      0, 0, 1, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      1, 0, 0, 0,
    ])),
  ]), { providers: [createReferenceProvider()] });
  const state = session.createInferenceState();

  await prefillChatMessages(
    session,
    tokenizer,
    state,
    [{ role: "system", content: "Be concise." }],
  );
  assert.equal(state.nextPosition, 1);

  const chunks: string[] = [];
  const result = await generateChatTurn(
    session,
    tokenizer,
    state,
    "Hello",
    {
      maxNewTokens: 4,
      doSample: false,
      onToken(chunk) {
        chunks.push(chunk.text);
      },
    },
  );

  assert.equal(result.content, "A");
  assert.equal(result.finishReason, "stop");
  assert.deepEqual(chunks, ["A"]);
  assert.equal(state.nextPosition, 5);
  assert.deepEqual(tokenizedTexts, [
    "<|turn>system\nBe concise.<turn|>\n",
    "<|turn>user\nHello<turn|>\n",
    "<|turn>model\n",
    "<turn|>\n",
  ]);
});

test("file GGUF tensor reader uses File.slice ranges", async () => {
  const tensorBytes = new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer);
  const fileBytes = minimalGgufBytes(tensorBytes);
  const reads: Array<{ start: number; end: number }> = [];
  const reader = await createFileGgufTensorReader({
    slice(start = 0, end = fileBytes.byteLength) {
      reads.push({ start, end });
      return {
        async arrayBuffer() {
          return fileBytes.slice(start, end).buffer;
        },
      } as Blob;
    },
  });

  const bytes = await reader.readTensorBytes("weight");
  assert.deepEqual(Array.from(bytes), Array.from(tensorBytes));
  assert.ok(reads.some((read) => read.end - read.start === tensorBytes.byteLength));
});

test("file GGUF tensor reader fully parses tokenizer arrays", async () => {
  const fileBytes = tokenizerArrayGgufBytes(["a", "b", "c"], ["a b", "b c", "c d"]);
  const reader = await createFileGgufTensorReader({
    slice(start = 0, end = fileBytes.byteLength) {
      return {
        async arrayBuffer() {
          return fileBytes.slice(start, end).buffer;
        },
      } as Blob;
    },
  }, { maxArraySample: 1 });

  assert.deepEqual(reader.metadata.metadata["tokenizer.ggml.tokens"], {
    type: "string",
    length: 3,
    sample: ["a", "b", "c"],
    truncated: false,
  });
  assert.deepEqual(reader.metadata.metadata["tokenizer.ggml.merges"], {
    type: "string",
    length: 3,
    sample: ["a b", "b c", "c d"],
    truncated: false,
  });
});

test("stripThinking hides complete and partial thinking blocks", () => {
  assert.equal(stripThinking("<think>\nsecret\n</think>\n\nVisible"), "\n\nVisible");
  assert.equal(stripThinking("<think>\nsecret"), "");
  assert.equal(stripThinking("Visible\n\n</think>\n<|im_end|>"), "Visible\n\n\n");
  assert.equal(stripThinking("Visible<turn|>\nignored"), "Visible");
});

function tokenizerGguf(
  tokens: string[],
  eosTokenId?: number,
  options: { model?: string; includePre?: boolean; merges?: string[] } = {},
): GgufMetadata {
  const merges = options.merges ?? [];
  return {
    version: 3,
    tensorCount: 0,
    metadataCount: 0,
    dataStart: 0n,
    metadata: {
      "tokenizer.ggml.model": options.model ?? "gpt2",
      ...(options.includePre === false ? {} : { "tokenizer.ggml.pre": "gemma4" }),
      "tokenizer.ggml.tokens": {
        type: "string",
        length: tokens.length,
        sample: tokens,
        truncated: false,
      },
      "tokenizer.ggml.merges": {
        type: "string",
        length: merges.length,
        sample: merges,
        truncated: false,
      },
      ...(eosTokenId === undefined ? {} : { "tokenizer.ggml.eos_token_id": eosTokenId }),
    },
    tensors: [],
  };
}

function tensorReaderFromTensors(tensors: Array<{
  name: string;
  dimensions: number[];
  bytes: Uint8Array;
}>) {
  let offset = 0n;
  const chunks: Uint8Array[] = [];
  const infos = tensors.map((tensor) => {
    const dataOffset = offset;
    offset += BigInt(tensor.bytes.byteLength);
    chunks.push(tensor.bytes);
    return {
      name: tensor.name,
      dimensions: tensor.dimensions,
      type: "F32" as const,
      typeId: 0,
      offset: 0n,
      dataOffset,
    };
  });
  const data = concatBytes(chunks);
  return new GgufTensorReader({
    version: 3,
    tensorCount: infos.length,
    metadataCount: 0,
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
      "gemma4.context_length": 32,
      "gemma4.full_attention_interval": 1,
      "gemma4.rope.dimension_count": 2,
      "gemma4.rope.dimension_sections": {
        type: "int32",
        length: 4,
        sample: [1, 1, 0, 0],
        truncated: false,
      },
      "gemma4.rope.freq_base": 10000,
      "gemma4.attention.layer_norm_rms_epsilon": 1e-6,
    },
    tensors: infos,
  }, {
    async read(offset, length) {
      return data.subarray(Number(offset), Number(offset) + length);
    },
  });
}

const singleTokenTokenizer = {
  tokenize() {
    return [0];
  },
  detokenize(tokenIds: readonly number[]) {
    return tokenIds.map((id) => id === 5 ? "A" : "").join("");
  },
  idToToken(id: number) {
    return id === 5 ? "A" : undefined;
  },
  tokenToId(token: string) {
    return token === "<turn|>" ? 106 : undefined;
  },
};

function sessionWithCapturingProvider(provider: SegmentRunnerProvider, topKs: number[]) {
  return createModelSession(tensorReaderFromTensors([]), {
    providers: [capturingProvider(provider, topKs)],
  });
}

function capturingProvider(provider: SegmentRunnerProvider, topKs: number[]): ModelRunnerProvider {
  return {
    name: provider,
    createModelRunner(): ModelRunner {
      return {
        provider,
        graphNodes: {
          createEmbeddingNode: () => hiddenNode(provider, "embedding"),
          createPreparedHiddenInputNode: () => hiddenNode(provider, "input"),
          createLayerSegmentNode: (_start, _end, inputId) => hiddenNode(provider, `${provider}-segment`, [inputId]),
          createOutputNode: (inputId, topK = 1) => outputNode(provider, inputId, topK, topKs),
          createImportHiddenNode: (inputId) => hiddenNode(provider, `${provider}-import`, [inputId]),
          createExportHiddenNode: (inputId) => hiddenNode(provider, `${provider}-export`, [inputId]),
        },
        async prepareInput() {
          return { hidden: new Float32Array([1, 0, 0, 0]) };
        },
        async preparePreparedHiddenInput(_session, hidden) {
          return { hidden };
        },
        async segmentRunner() {
          throw new Error("capturing provider segment runner should not be called.");
        },
      };
    },
    modelResourceRequirements(): ProviderResourceRequirements {
      return {
        provider,
        mode: "enabled",
        support: { available: true },
        memoryLimitBytes: Number.POSITIVE_INFINITY,
        fixedBytes: 0,
        outputBytes: 0,
        scratchBytes: 0,
        targetResourceConstrained: false,
        canRunFullModel: true,
        offReason: `${provider} off.`,
        blockedReason: `${provider} blocked.`,
        plannedReason: `${provider} planned.`,
        layers: [],
      };
    },
  };
}

function hiddenNode(
  provider: SegmentRunnerProvider,
  id: string,
  deps: readonly string[] = [],
): ForwardRunnerNode {
  return {
    id,
    deps,
    backend: provider,
    run() {
      const hidden = new Float32Array([1, 0, 0, 0]);
      return {
        kind: "cpu-hidden",
        buffer: cpuRunnerBuffer(hidden, [1, 4]),
        hidden,
      };
    },
  };
}

function outputNode(
  provider: SegmentRunnerProvider,
  inputId: string,
  topK: number,
  topKs: number[],
): ForwardRunnerNode {
  return {
    id: "output",
    deps: [inputId],
    backend: provider,
    run() {
      topKs.push(topK);
      return {
        kind: "output",
        result: {
          topTokens: Array.from({ length: topK }, (_, index) => ({
            id: index === 0 ? 5 : 200 + index,
            value: topK - index,
          })),
        },
      };
    },
  };
}

function f32Tensor(name: string, dimensions: number[], values: Float32Array) {
  return {
    name,
    dimensions,
    bytes: new Uint8Array(values.buffer, values.byteOffset, values.byteLength).slice(),
  };
}

function minimalGgufBytes(tensorBytes: Uint8Array): Uint8Array {
  const writer = new ByteWriter();
  writer.ascii("GGUF");
  writer.u32(3);
  writer.u64(1n);
  writer.u64(0n);
  writer.string("weight");
  writer.u32(1);
  writer.u64(4n);
  writer.u32(0);
  writer.u64(0n);
  writer.align(32);
  writer.bytes(tensorBytes);
  return writer.toBytes();
}

function tokenizerArrayGgufBytes(tokens: string[], merges: string[]): Uint8Array {
  const writer = new ByteWriter();
  writer.ascii("GGUF");
  writer.u32(3);
  writer.u64(0n);
  writer.u64(2n);
  writer.string("tokenizer.ggml.tokens");
  writer.u32(9);
  writer.u32(8);
  writer.u64(BigInt(tokens.length));
  for (const token of tokens) {
    writer.string(token);
  }
  writer.string("tokenizer.ggml.merges");
  writer.u32(9);
  writer.u32(8);
  writer.u64(BigInt(merges.length));
  for (const merge of merges) {
    writer.string(merge);
  }
  writer.align(32);
  return writer.toBytes();
}

function concatBytes(chunks: Uint8Array[]): Uint8Array {
  const output = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0));
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

class ByteWriter {
  private chunks: number[] = [];

  ascii(value: string): void {
    for (const char of value) {
      this.chunks.push(char.charCodeAt(0));
    }
  }

  bytes(value: Uint8Array): void {
    this.chunks.push(...value);
  }

  string(value: string): void {
    const bytes = new TextEncoder().encode(value);
    this.u64(BigInt(bytes.byteLength));
    this.bytes(bytes);
  }

  u32(value: number): void {
    const view = new DataView(new ArrayBuffer(4));
    view.setUint32(0, value, true);
    this.bytes(new Uint8Array(view.buffer));
  }

  u64(value: bigint): void {
    const view = new DataView(new ArrayBuffer(8));
    view.setBigUint64(0, value, true);
    this.bytes(new Uint8Array(view.buffer));
  }

  align(alignment: number): void {
    while (this.chunks.length % alignment !== 0) {
      this.chunks.push(0);
    }
  }

  toBytes(): Uint8Array {
    return new Uint8Array(this.chunks);
  }
}
