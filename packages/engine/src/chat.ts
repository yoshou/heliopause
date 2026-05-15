import {
  getMetadataString,
  parseGguf,
  type GgufByteReader,
} from "./gguf";
import {
  createModelSession,
  type InferenceState,
  type ModelSession,
  type ModelSessionOptions,
} from "./runtime";
import {
  decode as decodeToken,
  prefill as prefillTokens,
  prefillPreparedHidden,
} from "./forward";
import {
  GgufTensorReader,
  type GgufTensorReadTrace,
} from "./tensor-reader";
import type { Tokenizer } from "./tokenizer";

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type PreparedImageInput = {
  hidden: Float32Array;
  tokenCount: number;
};

export type PreparedAudioInput = {
  hidden: Float32Array;
  tokenCount: number;
};

export type ChatTemplateOptions = {
  addGenerationPrompt?: boolean;
  enableThinking?: boolean;
};

export type FileGgufTensorReaderOptions = {
  bufferSizeBytes?: number;
  maxArraySample?: number;
  onRead?: GgufTensorReadTrace;
};

export type ChatCompletionOptions = {
  maxNewTokens?: number;
  stopTokenIds?: readonly number[];
  signal?: AbortSignal;
  onToken?: (chunk: ChatCompletionChunk) => void;
};

export type ChatPrefillOptions = {
  signal?: AbortSignal;
};

export type ChatTurnOptions = ChatCompletionOptions;

export type ChatTurnResult = {
  content: string;
  finishReason: "stop" | "length";
  state: InferenceState;
};

export type ChatCompletionChunk = {
  tokenId: number;
  token: string;
  text: string;
  content: string;
};

export const DEFAULT_SYSTEM_PROMPT =
  "You are Heliopause, a helpful local assistant running entirely on this device.";

const DEFAULT_MAX_NEW_TOKENS = 256;
const DEFAULT_GGUF_PARSE_BUFFER_BYTES = 16 * 1024 * 1024;
const TOKENIZER_ARRAY_METADATA_KEYS = [
  "tokenizer.ggml.tokens",
  "tokenizer.ggml.merges",
] as const;

export function applyChatTemplate(
  messages: readonly ChatMessage[],
  options: ChatTemplateOptions = {},
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

export function applyChatGenerationPrompt(
  options: Pick<ChatTemplateOptions, "enableThinking"> = {},
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

export function createChatSession(
  tensorReader: GgufTensorReader,
  options: ModelSessionOptions = {},
): ModelSession {
  return createModelSession(tensorReader, options);
}

export async function prefillChatMessages(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  messages: readonly ChatMessage[],
  options: ChatPrefillOptions = {},
): Promise<InferenceState> {
  const prompt = applyChatTemplate(messages, { addGenerationPrompt: false });
  await prefillChatText(session, tokenizer, state, prompt, {
    signal: options.signal,
    requireGenerationSlot: false,
  });
  return state;
}

export async function generateChatTurn(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  userContent: string,
  options: ChatTurnOptions = {},
): Promise<ChatTurnResult> {
  throwIfAborted(options.signal);

  await prefillChatText(
    session,
    tokenizer,
    state,
    applyChatTemplate([{ role: "user", content: userContent }], {
      addGenerationPrompt: false,
    }),
    { signal: options.signal, requireGenerationSlot: true },
  );

  const promptPrefill = await prefillChatText(
    session,
    tokenizer,
    state,
    applyChatGenerationPrompt(),
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
  let finishReason: ChatTurnResult["finishReason"] = "length";

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

    const decodeResult = await decodeToken(session, tokenId, {
      state,
      logitsTopK: 1,
    });
    logits = decodeResult.logits;
    nextTokenId = nextTokenFrom(logits, decodeResult.topTokens, decodeResult.selectedTokenId);
    if (nextTokenId === undefined) {
      finishReason = "stop";
      break;
    }

    const token = tokenizer.idToToken(tokenId) ?? "";
    const text = tokenizer.detokenize([tokenId]);
    content = sanitizeChatOutput(content + text);

    options.onToken?.({
      tokenId,
      token,
      text,
      content,
    });
  }

  if (state.nextPosition < state.contextLength) {
    await prefillChatText(session, tokenizer, state, "<turn|>\n", {
      signal: options.signal,
      requireGenerationSlot: false,
    });
  }

  return { content, finishReason, state };
}

export async function generatePreparedImageChatTurn(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  userContent: string,
  image: PreparedImageInput,
  options: ChatTurnOptions = {},
): Promise<ChatTurnResult> {
  throwIfAborted(options.signal);
  if (image.tokenCount <= 0 || image.hidden.length !== image.tokenCount * session.manifest.embeddingLength) {
    throw new Error(`Prepared image hidden shape mismatch: ${image.hidden.length}`);
  }

  await prefillChatText(session, tokenizer, state, "<|turn>user\n<|image>", {
    signal: options.signal,
    requireGenerationSlot: true,
  });
  await prefillPreparedHidden(session, image.hidden, {
    state,
    positions: Int32Array.from(
      { length: image.tokenCount },
      (_, index) => state.nextPosition + index,
    ),
    attentionCausal: false,
  });
  await prefillChatText(session, tokenizer, state, `<image|>\n${userContent.trim()}<turn|>\n`, {
    signal: options.signal,
    requireGenerationSlot: true,
  });

  const promptPrefill = await prefillChatText(
    session,
    tokenizer,
    state,
    applyChatGenerationPrompt(),
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
  let finishReason: ChatTurnResult["finishReason"] = "length";

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

    const decodeResult = await decodeToken(session, tokenId, {
      state,
      logitsTopK: 1,
    });
    logits = decodeResult.logits;
    nextTokenId = nextTokenFrom(logits, decodeResult.topTokens, decodeResult.selectedTokenId);
    if (nextTokenId === undefined) {
      finishReason = "stop";
      break;
    }

    const token = tokenizer.idToToken(tokenId) ?? "";
    const text = tokenizer.detokenize([tokenId]);
    content = sanitizeChatOutput(content + text);

    options.onToken?.({
      tokenId,
      token,
      text,
      content,
    });
  }

  if (state.nextPosition < state.contextLength) {
    await prefillChatText(session, tokenizer, state, "<turn|>\n", {
      signal: options.signal,
      requireGenerationSlot: false,
    });
  }

  return { content, finishReason, state };
}

export async function generatePreparedAudioChatTurn(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  userContent: string,
  audio: PreparedAudioInput,
  options: ChatTurnOptions = {},
): Promise<ChatTurnResult> {
  throwIfAborted(options.signal);
  if (audio.tokenCount <= 0 || audio.hidden.length !== audio.tokenCount * session.manifest.embeddingLength) {
    throw new Error(`Prepared audio hidden shape mismatch: ${audio.hidden.length}`);
  }

  await prefillChatText(session, tokenizer, state, "<|turn>user\n<|audio>", {
    signal: options.signal,
    requireGenerationSlot: true,
  });
  await prefillPreparedHidden(session, audio.hidden, {
    state,
    positions: Int32Array.from(
      { length: audio.tokenCount },
      (_, index) => state.nextPosition + index,
    ),
  });
  await prefillChatText(session, tokenizer, state, `<audio|>\n${userContent.trim()}<turn|>\n`, {
    signal: options.signal,
    requireGenerationSlot: true,
  });

  return generateAssistantFromState(session, tokenizer, state, options);
}

async function generateAssistantFromState(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  options: ChatTurnOptions = {},
): Promise<ChatTurnResult> {
  const promptPrefill = await prefillChatText(
    session,
    tokenizer,
    state,
    applyChatGenerationPrompt(),
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
  let finishReason: ChatTurnResult["finishReason"] = "length";

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

    const decodeResult = await decodeToken(session, tokenId, {
      state,
      logitsTopK: 1,
    });
    logits = decodeResult.logits;
    nextTokenId = nextTokenFrom(logits, decodeResult.topTokens, decodeResult.selectedTokenId);
    if (nextTokenId === undefined) {
      finishReason = "stop";
      break;
    }

    const token = tokenizer.idToToken(tokenId) ?? "";
    const text = tokenizer.detokenize([tokenId]);
    content = sanitizeChatOutput(content + text);

    options.onToken?.({
      tokenId,
      token,
      text,
      content,
    });
  }

  if (state.nextPosition < state.contextLength) {
    await prefillChatText(session, tokenizer, state, "<turn|>\n", {
      signal: options.signal,
      requireGenerationSlot: false,
    });
  }

  return { content, finishReason, state };
}

export async function* generateChatCompletion(
  session: ModelSession,
  tokenizer: Tokenizer,
  messages: readonly ChatMessage[],
  options: ChatCompletionOptions = {},
): AsyncGenerator<ChatCompletionChunk, string> {
  throwIfAborted(options.signal);

  const prompt = applyChatTemplate(messages);
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

  const prefillResult = await prefillTokens(session, promptTokenIds, {
    state,
    computeLogits: true,
    logitsTopK: 1,
  });
  let logits = prefillResult.logits;
  let nextTokenId = nextTokenFrom(logits, prefillResult.topTokens, prefillResult.selectedTokenId);
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
    content = sanitizeChatOutput(content + text);

    const chunk = {
      tokenId,
      token,
      text,
      content,
    };
    options.onToken?.(chunk);
    yield chunk;

    const decodeResult = await decodeToken(session, tokenId, {
      state,
      logitsTopK: 1,
    });
    logits = decodeResult.logits;
    nextTokenId = nextTokenFrom(logits, decodeResult.topTokens, decodeResult.selectedTokenId);
    if (nextTokenId === undefined) {
      break;
    }
  }

  return content;
}

async function prefillChatText(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  text: string,
  options: {
    signal?: AbortSignal;
    computeLogits?: boolean;
    requireGenerationSlot?: boolean;
  } = {},
): Promise<{
  state: InferenceState;
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

  return prefillTokens(session, tokenIds, {
    state,
    positions: Int32Array.from(
      { length: tokenIds.length },
      (_, index) => state.nextPosition + index,
    ),
    computeLogits: options.computeLogits,
    logitsTopK: options.computeLogits ? 1 : undefined,
  });
}

export function stripThinking(content: string): string {
  let output = content;
  while (true) {
    const start = output.indexOf("<think>");
    if (start < 0) {
      return sanitizeChatOutput(output);
    }
    const end = output.indexOf("</think>", start);
    if (end < 0) {
      return sanitizeChatOutput(output.slice(0, start).trimStart());
    }
    output = `${output.slice(0, start)}${output.slice(end + "</think>".length)}`;
  }
}

function sanitizeChatOutput(content: string): string {
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
