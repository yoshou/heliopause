import assert from "node:assert/strict";
import test from "node:test";

import {
  applyQwen35ChatTemplate,
  buildQwen35Tokenizer,
  createFileGgufTensorReader,
  createQwen35ModelSession,
  generateQwen35ChatCompletion,
  GgufTensorReader,
  stripQwen35Thinking,
  type GgufMetadata,
} from "../src/index.ts";

test("Qwen35 chat template formats history and disables thinking by default", () => {
  assert.equal(
    applyQwen35ChatTemplate([
      { role: "system", content: "Be concise." },
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi" },
    ]),
    "<|im_start|>system\nBe concise.<|im_end|>\n" +
      "<|im_start|>user\nHello<|im_end|>\n" +
      "<|im_start|>assistant\nHi<|im_end|>\n" +
      "<|im_start|>assistant\n<think>\n\n</think>\n\n",
  );
});

test("Qwen35 chat template can leave thinking enabled", () => {
  assert.equal(
    applyQwen35ChatTemplate([
      { role: "user", content: "Hello" },
    ], { enableThinking: true }),
    "<|im_start|>user\nHello<|im_end|>\n" +
      "<|im_start|>assistant\n",
  );
});

test("Qwen35 tokenizer preserves special chat tokens and detokenizes text", () => {
  const tokenizer = buildQwen35Tokenizer(tokenizerGguf([
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

test("Qwen35 chat generation stops on im_end token", async () => {
  const tokenizer = {
    eosTokenId: 3,
    tokenize() {
      return [2];
    },
    detokenize(tokenIds: readonly number[]) {
      return tokenIds.map((id) => id === 0 ? "A" : "").join("");
    },
    idToToken(id: number) {
      return id === 0 ? "A" : id === 3 ? "<|im_end|>" : undefined;
    },
    tokenToId(token: string) {
      return token === "<|im_end|>" ? 3 : undefined;
    },
  };
  const session = createQwen35ModelSession(tensorReaderFromTensors([
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
  ]));

  const chunks = [];
  for await (const chunk of generateQwen35ChatCompletion(
    session,
    tokenizer,
    [{ role: "user", content: "A" }],
    { maxNewTokens: 4 },
  )) {
    chunks.push(chunk);
  }

  assert.deepEqual(chunks.map((chunk) => chunk.text), ["A"]);
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

test("stripQwen35Thinking hides complete and partial thinking blocks", () => {
  assert.equal(stripQwen35Thinking("<think>\nsecret\n</think>\n\nVisible"), "\n\nVisible");
  assert.equal(stripQwen35Thinking("<think>\nsecret"), "");
});

function tokenizerGguf(tokens: string[], eosTokenId?: number): GgufMetadata {
  return {
    version: 3,
    tensorCount: 0,
    metadataCount: 0,
    dataStart: 0n,
    metadata: {
      "tokenizer.ggml.model": "gpt2",
      "tokenizer.ggml.pre": "qwen35",
      "tokenizer.ggml.tokens": {
        type: "string",
        length: tokens.length,
        sample: tokens,
        truncated: false,
      },
      "tokenizer.ggml.merges": {
        type: "string",
        length: 0,
        sample: [],
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
      "general.architecture": "qwen35",
      "qwen35.block_count": 0,
      "qwen35.embedding_length": 4,
      "qwen35.feed_forward_length": 8,
      "qwen35.attention.head_count": 1,
      "qwen35.attention.head_count_kv": 1,
      "qwen35.attention.key_length": 2,
      "qwen35.attention.value_length": 2,
      "qwen35.context_length": 32,
      "qwen35.full_attention_interval": 1,
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
      "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
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
