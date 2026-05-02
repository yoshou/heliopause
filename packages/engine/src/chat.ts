import {
  getMetadataString,
  parseGguf,
  type GgufByteReader,
} from "./gguf";
import {
  createQwen35ModelSession,
  type Qwen35InferenceState,
  type Qwen35ModelSession,
  type Qwen35ModelSessionOptions,
} from "./runtime";
import {
  decodeQwen35,
  prefillQwen35,
} from "./forward";
import {
  GgufTensorReader,
  type GgufTensorReadTrace,
} from "./tensor-reader";
import type { Qwen35Tokenizer } from "./tokenizer";

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type Qwen35ChatTemplateOptions = {
  addGenerationPrompt?: boolean;
  enableThinking?: boolean;
};

export type FileGgufTensorReaderOptions = {
  bufferSizeBytes?: number;
  maxArraySample?: number;
  onRead?: GgufTensorReadTrace;
};

export type Qwen35ChatCompletionOptions = {
  maxNewTokens?: number;
  stopTokenIds?: readonly number[];
  signal?: AbortSignal;
  onToken?: (chunk: Qwen35ChatCompletionChunk) => void;
};

export type Qwen35ChatPrefillOptions = {
  signal?: AbortSignal;
};

export type Qwen35ChatTurnOptions = Qwen35ChatCompletionOptions;

export type Qwen35ChatTurnResult = {
  content: string;
  finishReason: "stop" | "length";
  state: Qwen35InferenceState;
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
  const enableThinking = options.enableThinking ?? false;
  let output = "";

  for (const message of messages) {
    output += `<|im_start|>${message.role}\n${message.content.trim()}<|im_end|>\n`;
  }

  if (addGenerationPrompt) {
    output += "<|im_start|>assistant\n";
    if (!enableThinking) {
      output += "<think>\n\n</think>\n\n";
    }
  }

  return output;
}

export function applyQwen35ChatGenerationPrompt(
  options: Pick<Qwen35ChatTemplateOptions, "enableThinking"> = {},
): string {
  let output = "<|im_start|>assistant\n";
  if (!(options.enableThinking ?? false)) {
    output += "<think>\n\n</think>\n\n";
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
  return new GgufTensorReader(gguf, byteReader, { onRead: options.onRead });
}

export function createQwen35ChatSession(
  tensorReader: GgufTensorReader,
  options: Qwen35ModelSessionOptions = {},
): Qwen35ModelSession {
  return createQwen35ModelSession(tensorReader, options);
}

export async function prefillQwen35ChatMessages(
  session: Qwen35ModelSession,
  tokenizer: Qwen35Tokenizer,
  state: Qwen35InferenceState,
  messages: readonly ChatMessage[],
  options: Qwen35ChatPrefillOptions = {},
): Promise<Qwen35InferenceState> {
  const prompt = applyQwen35ChatTemplate(messages, { addGenerationPrompt: false });
  await prefillQwen35ChatText(session, tokenizer, state, prompt, {
    signal: options.signal,
    requireGenerationSlot: false,
  });
  return state;
}

export async function generateQwen35ChatTurn(
  session: Qwen35ModelSession,
  tokenizer: Qwen35Tokenizer,
  state: Qwen35InferenceState,
  userContent: string,
  options: Qwen35ChatTurnOptions = {},
): Promise<Qwen35ChatTurnResult> {
  throwIfAborted(options.signal);

  await prefillQwen35ChatText(
    session,
    tokenizer,
    state,
    applyQwen35ChatTemplate([{ role: "user", content: userContent }], {
      addGenerationPrompt: false,
    }),
    { signal: options.signal, requireGenerationSlot: true },
  );

  const promptPrefill = await prefillQwen35ChatText(
    session,
    tokenizer,
    state,
    applyQwen35ChatGenerationPrompt(),
    {
      signal: options.signal,
      computeLogits: true,
      requireGenerationSlot: true,
    },
  );
  let logits = promptPrefill.logits;
  let nextTokenId = nextTokenFrom(logits, promptPrefill.topTokens);
  if (nextTokenId === undefined) {
    return { content: "", finishReason: "stop", state };
  }

  const stopTokenIds = new Set([
    tokenizer.eosTokenId,
    tokenizer.tokenToId("<|im_end|>"),
    ...(options.stopTokenIds ?? []),
  ].filter((id): id is number => typeof id === "number"));
  const maxNewTokens = options.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS;
  let content = "";
  let finishReason: Qwen35ChatTurnResult["finishReason"] = "length";

  for (let index = 0; index < maxNewTokens; index += 1) {
    throwIfAborted(options.signal);

    const tokenId = nextTokenId;
    if (stopTokenIds.has(tokenId)) {
      finishReason = "stop";
      break;
    }
    if (state.nextPosition >= state.contextLength) {
      finishReason = "length";
      break;
    }

    const decode = await decodeQwen35(session, tokenId, {
      state,
      logitsTopK: 1,
    });
    logits = decode.logits;
    nextTokenId = nextTokenFrom(logits, decode.topTokens);
    if (nextTokenId === undefined) {
      finishReason = "stop";
      break;
    }

    const token = tokenizer.idToToken(tokenId) ?? "";
    const text = tokenizer.detokenize([tokenId]);
    content += text;

    options.onToken?.({
      tokenId,
      token,
      text,
      content,
    });
  }

  if (state.nextPosition < state.contextLength) {
    await prefillQwen35ChatText(session, tokenizer, state, "<|im_end|>\n", {
      signal: options.signal,
      requireGenerationSlot: false,
    });
  }

  return { content, finishReason, state };
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
  let nextTokenId = nextTokenFrom(logits, prefill.topTokens);
  if (nextTokenId === undefined) {
    return "";
  }

  let content = "";
  for (let index = 0; index < maxNewTokens; index += 1) {
    throwIfAborted(options.signal);

    const tokenId = nextTokenId;
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
    nextTokenId = nextTokenFrom(logits, decode.topTokens);
    if (nextTokenId === undefined) {
      break;
    }
  }

  return content;
}

async function prefillQwen35ChatText(
  session: Qwen35ModelSession,
  tokenizer: Qwen35Tokenizer,
  state: Qwen35InferenceState,
  text: string,
  options: {
    signal?: AbortSignal;
    computeLogits?: boolean;
    requireGenerationSlot?: boolean;
  } = {},
): Promise<{
  state: Qwen35InferenceState;
  logits?: Float32Array;
  topTokens?: Array<{ id: number; value: number }>;
}> {
  throwIfAborted(options.signal);

  const tokenIds = tokenizer.tokenize(text);
  if (tokenIds.length === 0) {
    return { state };
  }

  const requiredPositions = state.nextPosition + tokenIds.length;
  const limit = options.requireGenerationSlot ? state.contextLength - 1 : state.contextLength;
  if (requiredPositions > limit) {
    throw new Error(
      `Chat state would use ${requiredPositions} tokens, which exceeds the configured chat context of ${state.contextLength} tokens.`,
    );
  }

  return prefillQwen35(session, tokenIds, {
    state,
    positions: Int32Array.from(
      { length: tokenIds.length },
      (_, index) => state.nextPosition + index,
    ),
    computeLogits: options.computeLogits,
    logitsTopK: options.computeLogits ? 1 : undefined,
  });
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
    sourceBlob: typeof Blob !== "undefined" && file instanceof Blob ? file : undefined,
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

function nextTokenFrom(
  logits: Float32Array | undefined,
  topTokens: Array<{ id: number; value: number }> | undefined,
): number | undefined {
  const topToken = topTokens?.[0]?.id;
  if (topToken !== undefined) {
    return topToken;
  }
  return logits ? argmax(logits) : undefined;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Generation was aborted.", "AbortError");
  }
}
