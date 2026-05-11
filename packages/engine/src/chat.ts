import {
  getMetadataString,
  parseGguf,
  type GgufByteReader,
} from "./gguf";
import {
  createGemma4ModelSession,
  type Gemma4InferenceState,
  type Gemma4ModelSession,
  type Gemma4ModelSessionOptions,
} from "./runtime";
import {
  decodeGemma4,
  prefillGemma4,
  prefillGemma4PreparedHidden,
} from "./forward";
import {
  GgufTensorReader,
  type GgufTensorReadTrace,
} from "./tensor-reader";
import type { Gemma4Tokenizer } from "./tokenizer";

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type Gemma4PreparedImageInput = {
  hidden: Float32Array;
  tokenCount: number;
};

export type Gemma4ChatTemplateOptions = {
  addGenerationPrompt?: boolean;
  enableThinking?: boolean;
};

export type FileGgufTensorReaderOptions = {
  bufferSizeBytes?: number;
  maxArraySample?: number;
  onRead?: GgufTensorReadTrace;
};

export type Gemma4ChatCompletionOptions = {
  maxNewTokens?: number;
  stopTokenIds?: readonly number[];
  signal?: AbortSignal;
  onToken?: (chunk: Gemma4ChatCompletionChunk) => void;
};

export type Gemma4ChatPrefillOptions = {
  signal?: AbortSignal;
};

export type Gemma4ChatTurnOptions = Gemma4ChatCompletionOptions;

export type Gemma4ChatTurnResult = {
  content: string;
  finishReason: "stop" | "length";
  state: Gemma4InferenceState;
};

export type Gemma4ChatCompletionChunk = {
  tokenId: number;
  token: string;
  text: string;
  content: string;
};

export const DEFAULT_GEMMA4_SYSTEM_PROMPT =
  "You are Heliopause, a helpful local assistant running entirely on this device.";

const DEFAULT_MAX_NEW_TOKENS = 256;
const DEFAULT_GGUF_PARSE_BUFFER_BYTES = 16 * 1024 * 1024;
const TOKENIZER_ARRAY_METADATA_KEYS = [
  "tokenizer.ggml.tokens",
  "tokenizer.ggml.merges",
] as const;

export function applyGemma4ChatTemplate(
  messages: readonly ChatMessage[],
  options: Gemma4ChatTemplateOptions = {},
): string {
  const addGenerationPrompt = options.addGenerationPrompt ?? true;
  const enableThinking = options.enableThinking ?? false;
  let output = "";

  for (const message of messages) {
    const role = message.role === "assistant" ? "model" : message.role;
    output += `<|turn>${role}\n${message.content.trim()}<turn|>\n`;
  }

  if (addGenerationPrompt) {
    output += enableThinking ? "<|think|>\n<|turn>model\n" : "<|turn>model\n";
  }

  return output;
}

export function applyGemma4ChatGenerationPrompt(
  options: Pick<Gemma4ChatTemplateOptions, "enableThinking"> = {},
): string {
  return (options.enableThinking ?? false) ? "<|think|>\n<|turn>model\n" : "<|turn>model\n";
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
    completeArrayKeys: TOKENIZER_ARRAY_METADATA_KEYS,
  });
  return new GgufTensorReader(gguf, byteReader, { onRead: options.onRead });
}

export function createGemma4ChatSession(
  tensorReader: GgufTensorReader,
  options: Gemma4ModelSessionOptions = {},
): Gemma4ModelSession {
  return createGemma4ModelSession(tensorReader, options);
}

export async function prefillGemma4ChatMessages(
  session: Gemma4ModelSession,
  tokenizer: Gemma4Tokenizer,
  state: Gemma4InferenceState,
  messages: readonly ChatMessage[],
  options: Gemma4ChatPrefillOptions = {},
): Promise<Gemma4InferenceState> {
  const prompt = applyGemma4ChatTemplate(messages, { addGenerationPrompt: false });
  await prefillGemma4ChatText(session, tokenizer, state, prompt, {
    signal: options.signal,
    requireGenerationSlot: false,
  });
  return state;
}

export async function generateGemma4ChatTurn(
  session: Gemma4ModelSession,
  tokenizer: Gemma4Tokenizer,
  state: Gemma4InferenceState,
  userContent: string,
  options: Gemma4ChatTurnOptions = {},
): Promise<Gemma4ChatTurnResult> {
  throwIfAborted(options.signal);

  await prefillGemma4ChatText(
    session,
    tokenizer,
    state,
    applyGemma4ChatTemplate([{ role: "user", content: userContent }], {
      addGenerationPrompt: false,
    }),
    { signal: options.signal, requireGenerationSlot: true },
  );

  const promptPrefill = await prefillGemma4ChatText(
    session,
    tokenizer,
    state,
    applyGemma4ChatGenerationPrompt(),
    {
      signal: options.signal,
      computeLogits: true,
      requireGenerationSlot: true,
    },
  );
  let logits = promptPrefill.logits;
  let nextTokenId = nextTokenFrom(logits, promptPrefill.topTokens, promptPrefill.selectedTokenId);
  if (nextTokenId === undefined) {
    return { content: "", finishReason: "stop", state };
  }

  const stopTokenIds = new Set([
    tokenizer.eosTokenId,
    tokenizer.tokenToId("<turn|>"),
    tokenizer.tokenToId("<eos>"),
    tokenizer.tokenToId("<|im_end|>"),
    ...(options.stopTokenIds ?? []),
  ].filter((id): id is number => typeof id === "number"));
  const maxNewTokens = options.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS;
  let content = "";
  let finishReason: Gemma4ChatTurnResult["finishReason"] = "length";

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

    const decode = await decodeGemma4(session, tokenId, {
      state,
      logitsTopK: 1,
    });
    logits = decode.logits;
    nextTokenId = nextTokenFrom(logits, decode.topTokens, decode.selectedTokenId);
    if (nextTokenId === undefined) {
      finishReason = "stop";
      break;
    }

    const token = tokenizer.idToToken(tokenId) ?? "";
    const text = tokenizer.detokenize([tokenId]);
    content = sanitizeGemma4ChatOutput(content + text);

    options.onToken?.({
      tokenId,
      token,
      text,
      content,
    });
  }

  if (state.nextPosition < state.contextLength) {
    await prefillGemma4ChatText(session, tokenizer, state, "<turn|>\n", {
      signal: options.signal,
      requireGenerationSlot: false,
    });
  }

  return { content, finishReason, state };
}

export async function generateGemma4PreparedImageChatTurn(
  session: Gemma4ModelSession,
  tokenizer: Gemma4Tokenizer,
  state: Gemma4InferenceState,
  userContent: string,
  image: Gemma4PreparedImageInput,
  options: Gemma4ChatTurnOptions = {},
): Promise<Gemma4ChatTurnResult> {
  throwIfAborted(options.signal);
  if (image.tokenCount <= 0 || image.hidden.length !== image.tokenCount * session.manifest.embeddingLength) {
    throw new Error(`Prepared image hidden shape mismatch: ${image.hidden.length}`);
  }

  await prefillGemma4ChatText(session, tokenizer, state, "<|turn>user\n<|image>", {
    signal: options.signal,
    requireGenerationSlot: true,
  });
  await prefillGemma4PreparedHidden(session, image.hidden, {
    state,
    positions: Int32Array.from(
      { length: image.tokenCount },
      (_, index) => state.nextPosition + index,
    ),
    attentionCausal: false,
  });
  await prefillGemma4ChatText(session, tokenizer, state, `<image|>\n${userContent.trim()}<turn|>\n`, {
    signal: options.signal,
    requireGenerationSlot: true,
  });

  const promptPrefill = await prefillGemma4ChatText(
    session,
    tokenizer,
    state,
    applyGemma4ChatGenerationPrompt(),
    {
      signal: options.signal,
      computeLogits: true,
      requireGenerationSlot: true,
    },
  );
  let logits = promptPrefill.logits;
  let nextTokenId = nextTokenFrom(logits, promptPrefill.topTokens, promptPrefill.selectedTokenId);
  if (nextTokenId === undefined) {
    return { content: "", finishReason: "stop", state };
  }

  const stopTokenIds = new Set([
    tokenizer.eosTokenId,
    tokenizer.tokenToId("<turn|>"),
    tokenizer.tokenToId("<eos>"),
    tokenizer.tokenToId("<|im_end|>"),
    ...(options.stopTokenIds ?? []),
  ].filter((id): id is number => typeof id === "number"));
  const maxNewTokens = options.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS;
  let content = "";
  let finishReason: Gemma4ChatTurnResult["finishReason"] = "length";

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

    const decode = await decodeGemma4(session, tokenId, {
      state,
      logitsTopK: 1,
    });
    logits = decode.logits;
    nextTokenId = nextTokenFrom(logits, decode.topTokens, decode.selectedTokenId);
    if (nextTokenId === undefined) {
      finishReason = "stop";
      break;
    }

    const token = tokenizer.idToToken(tokenId) ?? "";
    const text = tokenizer.detokenize([tokenId]);
    content = sanitizeGemma4ChatOutput(content + text);

    options.onToken?.({
      tokenId,
      token,
      text,
      content,
    });
  }

  if (state.nextPosition < state.contextLength) {
    await prefillGemma4ChatText(session, tokenizer, state, "<turn|>\n", {
      signal: options.signal,
      requireGenerationSlot: false,
    });
  }

  return { content, finishReason, state };
}

export async function* generateGemma4ChatCompletion(
  session: Gemma4ModelSession,
  tokenizer: Gemma4Tokenizer,
  messages: readonly ChatMessage[],
  options: Gemma4ChatCompletionOptions = {},
): AsyncGenerator<Gemma4ChatCompletionChunk, string> {
  throwIfAborted(options.signal);

  const prompt = applyGemma4ChatTemplate(messages);
  const promptTokenIds = tokenizer.tokenize(prompt);
  const state = session.createInferenceState();
  const stopTokenIds = new Set([
    tokenizer.eosTokenId,
    tokenizer.tokenToId("<turn|>"),
    tokenizer.tokenToId("<eos>"),
    tokenizer.tokenToId("<|im_end|>"),
    ...(options.stopTokenIds ?? []),
  ].filter((id): id is number => typeof id === "number"));
  const maxNewTokens = options.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS;

  if (promptTokenIds.length >= state.contextLength) {
    throw new Error(
      `Prompt uses ${promptTokenIds.length} tokens, which exceeds the configured chat context of ${state.contextLength} tokens.`,
    );
  }

  const prefill = await prefillGemma4(session, promptTokenIds, {
    state,
    computeLogits: true,
    logitsTopK: 1,
  });
  let logits = prefill.logits;
  let nextTokenId = nextTokenFrom(logits, prefill.topTokens, prefill.selectedTokenId);
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
    content = sanitizeGemma4ChatOutput(content + text);

    const chunk = {
      tokenId,
      token,
      text,
      content,
    };
    options.onToken?.(chunk);
    yield chunk;

    const decode = await decodeGemma4(session, tokenId, {
      state,
      logitsTopK: 1,
    });
    logits = decode.logits;
    nextTokenId = nextTokenFrom(logits, decode.topTokens, decode.selectedTokenId);
    if (nextTokenId === undefined) {
      break;
    }
  }

  return content;
}

async function prefillGemma4ChatText(
  session: Gemma4ModelSession,
  tokenizer: Gemma4Tokenizer,
  state: Gemma4InferenceState,
  text: string,
  options: {
    signal?: AbortSignal;
    computeLogits?: boolean;
    requireGenerationSlot?: boolean;
  } = {},
): Promise<{
  state: Gemma4InferenceState;
  logits?: Float32Array;
  selectedTokenId?: number;
  topTokens?: Array<{ id: number; value: number }>;
}> {
  throwIfAborted(options.signal);

  const tokenIds = tokenizer.tokenize(text, { addBos: state.nextPosition === 0 });
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

  return prefillGemma4(session, tokenIds, {
    state,
    positions: Int32Array.from(
      { length: tokenIds.length },
      (_, index) => state.nextPosition + index,
    ),
    computeLogits: options.computeLogits,
    logitsTopK: options.computeLogits ? 1 : undefined,
  });
}

export function stripGemma4Thinking(content: string): string {
  let output = content;
  while (true) {
    const start = output.indexOf("<think>");
    if (start < 0) {
      return sanitizeGemma4ChatOutput(output);
    }
    const end = output.indexOf("</think>", start);
    if (end < 0) {
      return sanitizeGemma4ChatOutput(output.slice(0, start).trimStart());
    }
    output = `${output.slice(0, start)}${output.slice(end + "</think>".length)}`;
  }
}

function sanitizeGemma4ChatOutput(content: string): string {
  const stopMarkers = ["<turn|>", "<|im_end|>", "<eos>"];
  let output = content;
  for (const marker of stopMarkers) {
    const index = output.indexOf(marker);
    if (index >= 0) {
      output = output.slice(0, index);
    }
  }
  output = output.replaceAll("</think>", "");
  while (true) {
    const start = output.indexOf("<|channel>");
    if (start < 0) {
      break;
    }
    const end = output.indexOf("<channel|>", start);
    output = end < 0 ? output.slice(0, start) : `${output.slice(0, start)}${output.slice(end + "<channel|>".length)}`;
  }
  return output;
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
  selectedTokenId: number | undefined,
): number | undefined {
  if (selectedTokenId !== undefined) {
    return selectedTokenId;
  }
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
