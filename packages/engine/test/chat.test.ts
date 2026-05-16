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
} from "../src/index.ts";

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
  ]), { runnerProviders: [createReferenceProvider()] });

  const chunks = [];
  for await (const chunk of generateChatCompletion(
    session,
    tokenizer,
    [{ role: "user", content: "A" }],
    { maxNewTokens: 4 },
  )) {
    chunks.push(chunk);
  }

  assert.deepEqual(chunks.map((chunk) => chunk.text), ["A"]);
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
  ]), { runnerProviders: [createReferenceProvider()] });
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
