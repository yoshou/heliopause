import {
  getMetadataString,
  parseGguf,
  type GgufByteReader,
} from "./gguf";
import {
  createQwen35ModelSession,
  decodeQwen35,
  prefillQwen35,
  type Qwen35ModelSession,
  type Qwen35ModelSessionOptions,
} from "./qwen35-forward";
import { GgufTensorReader } from "./tensor-reader";
import type { Qwen35Tokenizer } from "./tokenizer";

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type Qwen35ChatTemplateOptions = {
  addGenerationPrompt?: boolean;
};

export type FileGgufTensorReaderOptions = {
  bufferSizeBytes?: number;
  maxArraySample?: number;
};

export type Qwen35ChatCompletionOptions = {
  maxNewTokens?: number;
  stopTokenIds?: readonly number[];
  signal?: AbortSignal;
  onToken?: (chunk: Qwen35ChatCompletionChunk) => void;
};

export type Qwen35ChatCompletionChunk = {
  tokenId: number;
  token: string;
  text: string;
  content: string;
};

export const DEFAULT_QWEN35_SYSTEM_PROMPT =
  "You are Heliopause, a helpful local assistant running entirely on this device.";

const DEFAULT_MAX_NEW_TOKENS = 256;
const DEFAULT_GGUF_PARSE_BUFFER_BYTES = 16 * 1024 * 1024;

export function applyQwen35ChatTemplate(
  messages: readonly ChatMessage[],
  options: Qwen35ChatTemplateOptions = {},
): string {
  const addGenerationPrompt = options.addGenerationPrompt ?? true;
  let output = "";

  for (const message of messages) {
    output += `<|im_start|>${message.role}\n${message.content.trim()}<|im_end|>\n`;
  }

  if (addGenerationPrompt) {
    output += "<|im_start|>assistant\n";
  }

  return output;
}

export async function createFileGgufTensorReader(
  file: Pick<File, "slice">,
  options: FileGgufTensorReaderOptions = {},
): Promise<GgufTensorReader> {
  const byteReader = fileGgufByteReader(file);
  const parseReader = new BufferedGgufByteReader(
    byteReader,
    options.bufferSizeBytes ?? DEFAULT_GGUF_PARSE_BUFFER_BYTES,
  );
  const gguf = await parseGguf(parseReader, {
    maxArraySample: options.maxArraySample ?? 300000,
  });
  return new GgufTensorReader(gguf, byteReader);
}

export function createQwen35ChatSession(
  tensorReader: GgufTensorReader,
  options: Qwen35ModelSessionOptions = {},
): Qwen35ModelSession {
  return createQwen35ModelSession(tensorReader, options);
}

export async function* generateQwen35ChatCompletion(
  session: Qwen35ModelSession,
  tokenizer: Qwen35Tokenizer,
  messages: readonly ChatMessage[],
  options: Qwen35ChatCompletionOptions = {},
): AsyncGenerator<Qwen35ChatCompletionChunk, string> {
  throwIfAborted(options.signal);

  const prompt = applyQwen35ChatTemplate(messages);
  const promptTokenIds = tokenizer.tokenize(prompt);
  const state = session.createInferenceState();
  const stopTokenIds = new Set([
    tokenizer.eosTokenId,
    tokenizer.tokenToId("<|im_end|>"),
    ...(options.stopTokenIds ?? []),
  ].filter((id): id is number => typeof id === "number"));
  const maxNewTokens = options.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS;

  if (promptTokenIds.length >= state.contextLength) {
    throw new Error(
      `Prompt uses ${promptTokenIds.length} tokens, which exceeds the configured chat context of ${state.contextLength} tokens.`,
    );
  }

  const prefill = await prefillQwen35(session, promptTokenIds, {
    state,
    computeLogits: true,
    logitsTopK: 1,
  });
  let logits = prefill.logits;
  if (!logits) {
    return "";
  }

  let content = "";
  for (let index = 0; index < maxNewTokens; index += 1) {
    throwIfAborted(options.signal);

    const tokenId = argmax(logits);
    if (stopTokenIds.has(tokenId)) {
      break;
    }

    const token = tokenizer.idToToken(tokenId) ?? "";
    const text = tokenizer.detokenize([tokenId]);
    content += text;

    const chunk = {
      tokenId,
      token,
      text,
      content,
    };
    options.onToken?.(chunk);
    yield chunk;

    const decode = await decodeQwen35(session, tokenId, {
      state,
      logitsTopK: 1,
    });
    logits = decode.logits;
  }

  return content;
}

export function stripQwen35Thinking(content: string): string {
  let output = content;
  while (true) {
    const start = output.indexOf("<think>");
    if (start < 0) {
      return output;
    }
    const end = output.indexOf("</think>", start);
    if (end < 0) {
      return output.slice(0, start).trimStart();
    }
    output = `${output.slice(0, start)}${output.slice(end + "</think>".length)}`;
  }
}

export function getGgufModelName(reader: GgufTensorReader): string {
  return getMetadataString(reader.metadata.metadata, "general.name") ?? "Unknown GGUF model";
}

function fileGgufByteReader(file: Pick<File, "slice">): GgufByteReader {
  return {
    async read(offset, length) {
      const start = Number(offset);
      if (!Number.isSafeInteger(start)) {
        throw new Error(`File offset exceeds JavaScript safe integer range: ${offset.toString()}`);
      }
      const bytes = await file.slice(start, start + length).arrayBuffer();
      return new Uint8Array(bytes);
    },
  };
}

class BufferedGgufByteReader implements GgufByteReader {
  private buffer: Uint8Array<ArrayBufferLike> = new Uint8Array();
  private bufferStart = 0n;
  private readonly reader: GgufByteReader;
  private readonly bufferSizeBytes: number;

  constructor(
    reader: GgufByteReader,
    bufferSizeBytes: number,
  ) {
    this.reader = reader;
    this.bufferSizeBytes = bufferSizeBytes;
  }

  async read(offset: bigint, length: number): Promise<Uint8Array> {
    if (length <= 0) {
      return new Uint8Array();
    }

    const bufferEnd = this.bufferStart + BigInt(this.buffer.byteLength);
    if (offset >= this.bufferStart && offset + BigInt(length) <= bufferEnd) {
      const start = Number(offset - this.bufferStart);
      return this.buffer.subarray(start, start + length);
    }

    const readLength = Math.max(length, this.bufferSizeBytes);
    this.buffer = await this.reader.read(offset, readLength);
    this.bufferStart = offset;

    if (length > this.buffer.byteLength) {
      throw new Error(`Expected ${length} bytes, got ${this.buffer.byteLength}`);
    }
    return this.buffer.subarray(0, length);
  }
}

function argmax(values: Float32Array): number {
  let bestId = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let id = 0; id < values.length; id += 1) {
    const value = values[id] ?? Number.NEGATIVE_INFINITY;
    if (value > bestValue) {
      bestValue = value;
      bestId = id;
    }
  }
  return bestId;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Generation was aborted.", "AbortError");
  }
}
