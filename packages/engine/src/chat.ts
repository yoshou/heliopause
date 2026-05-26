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
  prefillPreparedHiddenState,
  prefillState as prefillStateTokens,
  type NextTokenResult,
} from "./forward";
import {
  createDeterministicRng,
  DEFAULT_GENERATION_CONFIG,
  resolveGenerationSamplingOptions,
  sampleNextToken,
  type GenerationSamplingOptions,
} from "./generation";
import {
  GgufTensorReader,
  type GgufTensorReadTrace,
} from "./tensor-reader";
import type { Tokenizer } from "./tokenizer";

export type ChatToolDeclaration = {
  type?: "function";
  function: {
    name: string;
    description: string;
    parameters?: unknown;
    response?: unknown;
  };
};

export type ChatToolCall = {
  id?: string;
  type?: "function";
  function: {
    name: string;
    arguments: unknown;
  };
};

export type ChatMessage =
  | {
      role: "system" | "developer" | "user";
      content: string;
      toolDeclarations?: readonly ChatToolDeclaration[];
    }
  | {
      role: "assistant";
      content?: string;
      tool_calls?: readonly ChatToolCall[];
    }
  | {
      role: "tool";
      tool_call_id: string;
      name?: string;
      content: unknown;
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
  closeFinalTurn?: boolean;
};

export type FileGgufTensorReaderOptions = {
  bufferSizeBytes?: number;
  maxArraySample?: number;
  onRead?: GgufTensorReadTrace;
};

export type ChatCompletionOptions = GenerationSamplingOptions & {
  maxNewTokens?: number;
  stopTokenIds?: readonly number[];
  signal?: AbortSignal;
  onToken?: (chunk: ChatCompletionChunk) => void;
};

export type ChatPrefillOptions = {
  signal?: AbortSignal;
};

export type ChatTurnOptions = ChatCompletionOptions & {
  appendTurnEnd?: boolean;
  continueModelTurn?: boolean;
};

export type ChatTurnInput = string | readonly ChatMessage[];

export type ChatTurnResult = {
  content: string;
  finishReason: "stop" | "length";
  state: InferenceState;
  modelTurnClosed?: boolean;
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
  const closeFinalTurn = options.closeFinalTurn ?? true;
  let output = "";

  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];

    if (message.role === "tool") {
      output += formatToolResponseMessage(message, findToolNameForToolMessage(messages, index));
      if (messages[index + 1]?.role !== "tool" && (closeFinalTurn || index + 1 < messages.length)) {
        output += "<turn|>\n";
      }
      continue;
    }

    const role = message.role === "assistant" ? "model" : message.role;
    output += `<|turn>${role}\n`;

    if (message.role === "assistant") {
      const toolCalls = message.tool_calls ?? [];
      for (const toolCall of toolCalls) {
        output += formatToolCall(toolCall);
      }
      output += stripThinking(message.content ?? "");
      if (toolCalls.length > 0) {
        if (messages[index + 1]?.role !== "tool") {
          output += "<|tool_response>";
        }
      } else {
        if (closeFinalTurn || index + 1 < messages.length) {
          output += "<turn|>\n";
        }
      }
      continue;
    }

    output += message.content.trim();
    for (const tool of message.toolDeclarations ?? []) {
      output += `<|tool>${formatToolDeclaration(tool)}<tool|>`;
    }
    if (closeFinalTurn || index + 1 < messages.length) {
      output += "<turn|>\n";
    }
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

function formatToolDeclaration(tool: ChatToolDeclaration): string {
  const declaration = tool.function;
  let output = `declaration:${declaration.name}{description:${formatGemma4Value(declaration.description)}`;
  if (declaration.parameters !== undefined) {
    output += `,parameters:${formatToolSchema(declaration.parameters)}`;
  }
  if (declaration.response !== undefined) {
    output += `,response:${formatToolSchema(declaration.response)}`;
  }
  return `${output}}`;
}

function formatToolSchema(schema: unknown): string {
  if (!isRecord(schema)) {
    return formatGemma4Value(schema);
  }

  const parts: string[] = [];
  if (isRecord(schema.properties)) {
    parts.push(`properties:{${formatToolParameters(schema.properties)}}`);
  }
  if (Array.isArray(schema.required) && schema.required.length > 0) {
    parts.push(`required:${formatGemma4Value(schema.required)}`);
  }
  if (typeof schema.type === "string") {
    parts.push(`type:${formatGemma4Value(schema.type.toUpperCase())}`);
  }
  return `{${parts.join(",")}}`;
}

function formatToolParameters(properties: Record<string, unknown>): string {
  return Object.keys(properties)
    .sort()
    .map((key) => `${key}:${formatToolParameter(properties[key])}`)
    .join(",");
}

function formatToolParameter(parameter: unknown): string {
  if (!isRecord(parameter)) {
    return formatGemma4Value(parameter);
  }

  const parts: string[] = [];
  if (typeof parameter.description === "string") {
    parts.push(`description:${formatGemma4Value(parameter.description)}`);
  }
  if (typeof parameter.type === "string" && parameter.type.toUpperCase() === "STRING" && Array.isArray(parameter.enum)) {
    parts.push(`enum:${formatGemma4Value(parameter.enum)}`);
  }
  if (typeof parameter.type === "string" && parameter.type.toUpperCase() === "ARRAY" && isRecord(parameter.items)) {
    parts.push(`items:${formatToolSchema(parameter.items)}`);
  }
  if (parameter.nullable === true) {
    parts.push("nullable:true");
  }
  if (typeof parameter.type === "string" && parameter.type.toUpperCase() === "OBJECT") {
    if (isRecord(parameter.properties)) {
      parts.push(`properties:{${formatToolParameters(parameter.properties)}}`);
    }
    if (Array.isArray(parameter.required) && parameter.required.length > 0) {
      parts.push(`required:${formatGemma4Value(parameter.required)}`);
    }
  }
  if (typeof parameter.type === "string") {
    parts.push(`type:${formatGemma4Value(parameter.type.toUpperCase())}`);
  }
  return `{${parts.join(",")}}`;
}

function formatToolCall(toolCall: ChatToolCall): string {
  return `<|tool_call>call:${toolCall.function.name}${formatGemma4ObjectBody(toolCall.function.arguments)}<tool_call|>`;
}

function formatToolResponseMessage(message: Extract<ChatMessage, { role: "tool" }>, fallbackName: string): string {
  return `<|tool_response>response:${message.name ?? fallbackName}${formatGemma4ObjectBody(message.content)}<tool_response|>`;
}

function formatGemma4ObjectBody(value: unknown): string {
  if (isRecord(value)) {
    return formatGemma4Value(value, false);
  }
  return `{value:${formatGemma4Value(value, false)}}`;
}

function formatGemma4Value(value: unknown, escapeKeys = true): string {
  if (typeof value === "string") {
    return `<|"|>${value}<|"|>`;
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? String(value) : "null";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (value === null || value === undefined) {
    return "null";
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => formatGemma4Value(item, escapeKeys)).join(",")}]`;
  }
  if (isRecord(value)) {
    return `{${Object.keys(value)
      .sort()
      .filter((key) => value[key] !== undefined)
      .map((key) => {
        const formattedKey = escapeKeys ? formatGemma4Value(key) : key;
        return `${formattedKey}:${formatGemma4Value(value[key], escapeKeys)}`;
      })
      .join(",")}}`;
  }
  return String(value);
}

function findToolNameForToolMessage(messages: readonly ChatMessage[], toolMessageIndex: number): string {
  const toolMessage = messages[toolMessageIndex];
  if (toolMessage.role !== "tool") {
    return "unknown";
  }
  for (let index = toolMessageIndex - 1; index >= 0; index -= 1) {
    const previous = messages[index];
    if (previous.role !== "assistant") {
      continue;
    }
    const call = previous.tool_calls?.find((candidate) => candidate.id === toolMessage.tool_call_id);
    return call?.function.name ?? "unknown";
  }
  return "unknown";
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
  options: ModelSessionOptions,
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
  turn: ChatTurnInput,
  options: ChatTurnOptions = {},
): Promise<ChatTurnResult> {
  throwIfAborted(options.signal);
  resolveGenerationSamplingOptions(options);

  const messages = normalizeChatTurnInput(turn);
  const prefillText = applyChatTemplate(messages, {
    addGenerationPrompt: false,
    closeFinalTurn: !options.continueModelTurn,
  });

  if (options.continueModelTurn) {
    const sampling = resolveGenerationSamplingOptions(options);
    const prefillResult = await prefillChatText(
      session,
      tokenizer,
      state,
      prefillText,
      {
        signal: options.signal,
        returnNextToken: true,
        requireGenerationSlot: true,
        logitsTopK: sampling.logitsTopK,
      },
    );
    return generateAssistantFromNextToken(session, tokenizer, state, prefillResult, options);
  }

  await prefillChatText(
    session,
    tokenizer,
    state,
    prefillText,
    { signal: options.signal, requireGenerationSlot: true },
  );

  return generateAssistantFromState(session, tokenizer, state, options);
}

export async function closeChatModelTurn(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  options: ChatPrefillOptions = {},
): Promise<InferenceState> {
  await prefillChatText(session, tokenizer, state, "<turn|>\n", {
    signal: options.signal,
    requireGenerationSlot: false,
  });
  return state;
}

export async function generatePreparedImageChatTurn(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  turn: ChatTurnInput,
  image: PreparedImageInput,
  options: ChatTurnOptions = {},
): Promise<ChatTurnResult> {
  throwIfAborted(options.signal);
  resolveGenerationSamplingOptions(options);
  if (image.tokenCount <= 0 || image.hidden.length !== image.tokenCount * session.manifest.embeddingLength) {
    throw new Error(`Prepared image hidden shape mismatch: ${image.hidden.length}`);
  }
  const { prefillMessages, userContent } = splitPreparedTurnInput(turn);
  if (prefillMessages.length > 0) {
    await prefillChatMessages(session, tokenizer, state, prefillMessages, { signal: options.signal });
  }

  await prefillChatText(session, tokenizer, state, "<|turn>user\n<|image>", {
    signal: options.signal,
    requireGenerationSlot: true,
  });
  await prefillPreparedHiddenState(session, state, image.hidden, {
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

  return generateAssistantFromState(session, tokenizer, state, options);
}

function normalizeChatTurnInput(turn: ChatTurnInput): readonly ChatMessage[] {
  return typeof turn === "string" ? [{ role: "user", content: turn }] : turn;
}

function splitPreparedTurnInput(turn: ChatTurnInput): {
  prefillMessages: readonly ChatMessage[];
  userContent: string;
} {
  if (typeof turn === "string") {
    return { prefillMessages: [], userContent: turn };
  }
  const lastMessage = turn.at(-1);
  if (!lastMessage || lastMessage.role !== "user") {
    throw new Error("Prepared media chat turns require the final structured message to be a user text turn.");
  }
  return {
    prefillMessages: turn.slice(0, -1),
    userContent: lastMessage.content,
  };
}

export async function generatePreparedAudioChatTurn(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  turn: ChatTurnInput,
  audio: PreparedAudioInput,
  options: ChatTurnOptions = {},
): Promise<ChatTurnResult> {
  throwIfAborted(options.signal);
  resolveGenerationSamplingOptions(options);
  if (audio.tokenCount <= 0 || audio.hidden.length !== audio.tokenCount * session.manifest.embeddingLength) {
    throw new Error(`Prepared audio hidden shape mismatch: ${audio.hidden.length}`);
  }
  const { prefillMessages, userContent } = splitPreparedTurnInput(turn);
  if (prefillMessages.length > 0) {
    await prefillChatMessages(session, tokenizer, state, prefillMessages, { signal: options.signal });
  }

  await prefillChatText(session, tokenizer, state, "<|turn>user\n<|audio>", {
    signal: options.signal,
    requireGenerationSlot: true,
  });
  await prefillPreparedHiddenState(session, state, audio.hidden, {
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
  if (options.continueModelTurn) {
    throw new Error("Cannot continue a model turn without prefilled continuation text.");
  }
  const promptPrefill = await prefillChatText(
    session,
    tokenizer,
    state,
    applyChatGenerationPrompt(),
    {
      signal: options.signal,
      returnNextToken: true,
      requireGenerationSlot: true,
      logitsTopK: resolveGenerationSamplingOptions(options).logitsTopK,
    },
  );
  return generateAssistantFromNextToken(session, tokenizer, state, promptPrefill, options);
}

async function generateAssistantFromNextToken(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  firstTokenResult: NextTokenResult,
  options: ChatTurnOptions = {},
): Promise<ChatTurnResult> {
  const stopTokenIds = buildStopTokenIds(tokenizer, options.stopTokenIds);
  const sampling = resolveGenerationSamplingOptions(options);
  const rng = createDeterministicRng(sampling.seed);
  const maxNewTokens = options.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS;
  let nextTokenId = sampleNextTokenFromResult(firstTokenResult, sampling, rng);
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

    const decodeResult = await decodeToken(session, state, tokenId, {
      logitsTopK: sampling.logitsTopK,
    });
    nextTokenId = sampleNextTokenFromResult(decodeResult, sampling, rng);

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

  const appendTurnEnd = options.appendTurnEnd ?? true;
  if (appendTurnEnd && state.nextPosition < state.contextLength) {
    await prefillChatText(session, tokenizer, state, "<turn|>\n", {
      signal: options.signal,
      requireGenerationSlot: false,
    });
  }

  return { content, finishReason, state, modelTurnClosed: appendTurnEnd };
}

export async function* generateChatCompletion(
  session: ModelSession,
  tokenizer: Tokenizer,
  messages: readonly ChatMessage[],
  options: ChatCompletionOptions = {},
): AsyncGenerator<ChatCompletionChunk, string> {
  throwIfAborted(options.signal);
  resolveGenerationSamplingOptions(options);

  const prompt = applyChatTemplate(messages);
  const promptTokenIds = tokenizer.tokenize(prompt);
  const state = session.createInferenceState();
  const stopTokenIds = buildStopTokenIds(tokenizer, options.stopTokenIds);
  const sampling = resolveGenerationSamplingOptions(options);
  const rng = createDeterministicRng(sampling.seed);
  const maxNewTokens = options.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS;

  if (promptTokenIds.length >= state.contextLength) {
    throw new Error(
      `Prompt uses ${promptTokenIds.length} tokens, which exceeds the configured chat context of ${state.contextLength} tokens.`,
    );
  }

  const prefillResult = await prefillTokens(session, state, promptTokenIds, {
    logitsTopK: sampling.logitsTopK,
  });
  let nextTokenId = sampleNextTokenFromResult(prefillResult, sampling, rng);

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

    const decodeResult = await decodeToken(session, state, tokenId, {
      logitsTopK: sampling.logitsTopK,
    });
    nextTokenId = sampleNextTokenFromResult(decodeResult, sampling, rng);
  }

  return content;
}

function sampleNextTokenFromResult(
  result: NextTokenResult,
  sampling: ReturnType<typeof resolveGenerationSamplingOptions>,
  rng: () => number,
): number {
  return sampleNextToken(
    result.topTokens ?? [{ id: result.nextTokenId, value: 0 }],
    sampling,
    rng,
  );
}

function buildStopTokenIds(tokenizer: Tokenizer, extraStopTokenIds: readonly number[] | undefined): Set<number> {
  return new Set([
    tokenizer.eosTokenId,
    tokenizer.tokenToId("<turn|>"),
    tokenizer.tokenToId("<eos>"),
    tokenizer.tokenToId("<|im_end|>"),
    ...DEFAULT_GENERATION_CONFIG.eosTokenIds,
    ...(extraStopTokenIds ?? []),
  ].filter((id): id is number => typeof id === "number"));
}

async function prefillChatText(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  text: string,
  options: {
    signal?: AbortSignal;
    logitsTopK?: number;
    returnNextToken: true;
    requireGenerationSlot?: boolean;
  },
): Promise<NextTokenResult>;
async function prefillChatText(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  text: string,
  options?: {
    signal?: AbortSignal;
    returnNextToken?: false;
    requireGenerationSlot?: boolean;
  },
): Promise<void>;
async function prefillChatText(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  text: string,
  options: {
    signal?: AbortSignal;
    logitsTopK?: number;
    returnNextToken?: boolean;
    requireGenerationSlot?: boolean;
  } = {},
): Promise<NextTokenResult | void> {
  throwIfAborted(options.signal);

  const tokenIds = tokenizer.tokenize(text, { addBos: state.nextPosition === 0 });
  if (tokenIds.length === 0) {
    if (options.returnNextToken) {
      throw new Error("Cannot produce a next token from empty chat prefill text.");
    }
    return;
  }

  const requiredPositions = state.nextPosition + tokenIds.length;
  const limit = options.requireGenerationSlot ? state.contextLength - 1 : state.contextLength;
  if (requiredPositions > limit) {
    throw new Error(
      `Chat state would use ${requiredPositions} tokens, which exceeds the configured chat context of ${state.contextLength} tokens.`,
    );
  }

  const positions = Int32Array.from(
    { length: tokenIds.length },
    (_, index) => state.nextPosition + index,
  );
  if (options.returnNextToken) {
    return prefillTokens(session, state, tokenIds, {
      positions,
      logitsTopK: options.logitsTopK ?? 1,
    });
  }
  await prefillStateTokens(session, state, tokenIds, { positions });
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Generation was aborted.", "AbortError");
  }
}
