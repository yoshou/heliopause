import {
  cloneInferenceState,
  closeChatModelTurn,
  generateChatTurn,
  type ChatMessage,
  type ChatToolDeclaration,
  type ChatTurnInput,
  type ChatTurnOptions,
  type ChatTurnResult,
  type InferenceState,
  type ModelSession,
  type Tokenizer,
} from "@heliopause/engine";

export const TOOL_CALL_OPEN_TAG = "<|tool_call>";
export const TOOL_CALL_CLOSE_TAG = "<tool_call|>";
export const TOOL_RESPONSE_OPEN_TAG = "<|tool_response>";
export const TOOL_RESPONSE_CLOSE_TAG = "<tool_response|>";
export const DEFAULT_MAX_TOOL_STEPS = 3;
export const DEFAULT_MAX_THINKING_CHARS = 8000;

const MAX_TOOL_STEPS_FINAL_PROMPT =
  "You have reached the maximum number of tool steps. Answer the user using only the tool responses already provided. Do not call another tool.";
const MAX_TOOL_STEPS_FINAL_MESSAGE =
  "I reached the maximum number of tool steps and cannot call another tool. I do not have enough final text to show beyond the tool results already gathered.";
const THOUGHT_CHANNEL_OPEN = "<|channel>thought\n";
const THOUGHT_CHANNEL_CLOSE = "<channel|>";

const AGENT_TOOL_NAMES = [
  "sandbox_list_files",
  "sandbox_read_file",
  "sandbox_write_file",
  "sandbox_command",
  "web_search",
  "web_fetch",
] as const;

export type AgentToolName = (typeof AGENT_TOOL_NAMES)[number];

export type AgentToolDefinition = {
  name: AgentToolName;
  description: string;
  parametersJsonSchema: unknown;
  requiresConfirmation?: boolean;
};

export type AgentToolCall = {
  id: string;
  name: AgentToolName;
  arguments: unknown;
};

export type AgentToolError = {
  code: string;
  message: string;
  retryable?: boolean;
};

export type AgentToolResult =
  | {
      callId: string;
      ok: true;
      content: unknown;
    }
  | {
      callId: string;
      ok: false;
      error: AgentToolError;
    };

export type AgentEvent =
  | { type: "text"; step: number; content: string }
  | { type: "thinking"; step: number; content: string; truncated?: boolean }
  | { type: "toolCall"; step: number; call: AgentToolCall }
  | { type: "toolResult"; step: number; result: AgentToolResult }
  | { type: "stepError"; step: number; callId: string; error: AgentToolError }
  | { type: "done"; steps: number; finishReason: "stop" | "length" | "maxToolSteps" };

export type AgentToolExecutor = (
  call: AgentToolCall,
  signal: AbortSignal,
) => Promise<AgentToolResult>;

export type AgentChatTurnGenerator = (
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  turn: ChatTurnInput,
  options?: ChatTurnOptions,
) => Promise<ChatTurnResult>;

export type AgentModelTurnCloser = (
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  options?: Pick<ChatTurnOptions, "signal">,
) => Promise<InferenceState>;

export type AgentStateCloner = false | ((state: InferenceState) => InferenceState);

export type AgentTurnOptions = ChatTurnOptions & {
  tools: readonly AgentToolDefinition[];
  executeTool: AgentToolExecutor;
  maxToolSteps?: number;
  maxThinkingChars?: number;
  onAgentEvent?: (event: AgentEvent) => void;
  chatTurnGenerator?: AgentChatTurnGenerator;
  closeModelTurn?: AgentModelTurnCloser;
  cloneState?: AgentStateCloner;
};

export type AgentTurnResult = {
  content: string;
  finishReason: "stop" | "length" | "maxToolSteps";
  steps: number;
  state: InferenceState;
};

export type AgentToolParseResult =
  | { type: "none" }
  | { type: "items"; items: readonly AgentToolParseItem[] }
  | { type: "error"; callId: string; error: AgentToolError };

export type AgentToolParseItem =
  | { type: "call"; call: AgentToolCall }
  | { type: "error"; callId: string; toolName: AgentToolName | "unknown"; error: AgentToolError };

type JsonObject = Record<string, unknown>;

export function parseToolCalls(
  content: string,
  tools: readonly AgentToolDefinition[],
  step: number,
): AgentToolParseResult {
  const extraction = extractNativeToolCalls(content);

  if (extraction.type === "none") {
    return { type: "none" };
  }

  if (extraction.type === "malformed") {
    const callId = toolCallIdForStepIndex(step, 1);
    return invalidToolCallFormat(callId, "Native tool call tags are incomplete or malformed.");
  }

  const items = extraction.bodies.map((body, index): AgentToolParseItem => {
    const callId = toolCallIdForStepIndex(step, index + 1);
    let parsed: NativeToolCall;
    try {
      parsed = parseNativeToolCallBody(body);
    } catch {
      return invalidToolCallFormatItem(callId, "Native tool call body could not be parsed.");
    }

    const tool = tools.find((candidate) => candidate.name === parsed.name);
    if (!tool || !isAgentToolName(parsed.name)) {
      return unknownToolItem(callId, `Unknown tool: ${parsed.name}`);
    }

    if (!validateJsonSchema(parsed.arguments, tool.parametersJsonSchema)) {
      return invalidToolArgumentsItem(callId, tool.name, `Arguments for ${tool.name} do not match the tool schema.`);
    }

    return {
      type: "call",
      call: {
        id: callId,
        name: tool.name,
        arguments: parsed.arguments,
      },
    };
  });

  return { type: "items", items };
}

export async function generateAgentTurn(
  session: ModelSession,
  tokenizer: Tokenizer,
  state: InferenceState,
  userContent: string,
  options: AgentTurnOptions,
): Promise<AgentTurnResult> {
  throwIfAborted(options.signal);

  const maxToolSteps = normalizeMaxToolSteps(options.maxToolSteps);
  const maxThinkingChars = normalizeMaxThinkingChars(options.maxThinkingChars);
  const enableThinking = options.enableThinking ?? true;
  const originalOnToken = options.onToken;
  const chatTurnGenerator = options.chatTurnGenerator ?? generateChatTurn;
  const {
    chatTurnGenerator: _chatTurnGenerator,
    tools: _tools,
    executeTool: _executeTool,
    maxToolSteps: _maxToolSteps,
    maxThinkingChars: _maxThinkingChars,
    onAgentEvent: _onAgentEvent,
    closeModelTurn: _closeModelTurn,
    cloneState: _cloneState,
    ...chatOptions
  } = options;
  const modelTurnCloser = options.closeModelTurn ?? closeChatModelTurn;
  const stateCloner = options.cloneState === undefined ? cloneInferenceState : options.cloneState;
  let nextTurn: ChatTurnInput = buildInitialTurn(userContent, options.tools);
  let toolSteps = 0;
  let finalOnly = maxToolSteps === 0;
  let modelTurnOpen = false;

  while (true) {
    throwIfAborted(options.signal);

    const step = toolSteps + 1;
    const committedModelTurnOpen = modelTurnOpen;
    const stateBeforeFinalGeneration = finalOnly && stateCloner ? stateCloner(state) : undefined;
    const generationState = finalOnly && stateCloner ? stateCloner(state) : state;
    const continueModelTurn = committedModelTurnOpen && isOnlyToolResponseTurn(nextTurn);
    const generation = await chatTurnGenerator(session, tokenizer, generationState, nextTurn, {
      ...chatOptions,
      doSample: chatOptions.doSample ?? false,
      enableThinking,
      appendTurnEnd: false,
      continueModelTurn,
      onToken: () => {},
    });
    const generatedModelTurnOpen = generation.modelTurnClosed === false;
    if (!finalOnly) {
      modelTurnOpen = generatedModelTurnOpen;
    }
    const separatedContent = separateThinking(generation.content);
    emitThinking(separatedContent.thinking, step, maxThinkingChars, enableThinking, options.onAgentEvent);
    const parseResult = parseToolCalls(separatedContent.visibleContent, options.tools, step);

    if (parseResult.type === "none") {
      let closedCommittedFinalTurn = false;
      if (finalOnly && committedModelTurnOpen) {
        await modelTurnCloser(session, tokenizer, state, { signal: options.signal });
        modelTurnOpen = false;
        closedCommittedFinalTurn = true;
      }
      if (generatedModelTurnOpen && (!closedCommittedFinalTurn || generation.state !== state)) {
        await modelTurnCloser(session, tokenizer, generation.state, { signal: options.signal });
        modelTurnOpen = false;
      }
      emitVisibleText(separatedContent.visibleContent, step, originalOnToken, options.onAgentEvent);
      const finishReason = finalOnly ? "maxToolSteps" : generation.finishReason;
      options.onAgentEvent?.({ type: "done", steps: toolSteps, finishReason });
      return {
        content: separatedContent.visibleContent,
        finishReason,
        steps: toolSteps,
        state: generation.state,
      };
    }

    if (finalOnly) {
      if (committedModelTurnOpen) {
        await modelTurnCloser(session, tokenizer, state, { signal: options.signal });
        if (stateBeforeFinalGeneration && stateBeforeFinalGeneration !== state) {
          await modelTurnCloser(session, tokenizer, stateBeforeFinalGeneration, { signal: options.signal });
        }
        modelTurnOpen = false;
      }
      emitVisibleText(MAX_TOOL_STEPS_FINAL_MESSAGE, step, originalOnToken, options.onAgentEvent);
      options.onAgentEvent?.({ type: "done", steps: toolSteps, finishReason: "maxToolSteps" });
      return {
        content: MAX_TOOL_STEPS_FINAL_MESSAGE,
        finishReason: "maxToolSteps",
        steps: toolSteps,
        state: stateBeforeFinalGeneration ?? state,
      };
    }

    toolSteps += 1;

    const toolResponses = await executeParsedToolItems(parseResult, step, options);

    const toolResponseTurn = toolResponses.map(({ result, toolName }) => buildToolResponseTurn(result, toolName));
    if (toolSteps >= maxToolSteps) {
      finalOnly = true;
      nextTurn = [
        ...toolResponseTurn,
        { role: "user", content: MAX_TOOL_STEPS_FINAL_PROMPT },
      ];
    } else {
      nextTurn = toolResponseTurn;
    }
  }
}

function isOnlyToolResponseTurn(turn: ChatTurnInput): boolean {
  return Array.isArray(turn) && turn.length > 0 && turn.every((message) => message.role === "tool");
}

function separateThinking(content: string): { visibleContent: string; thinking: string[] } {
  const thinking: string[] = [];
  let visibleContent = "";
  let cursor = 0;

  while (cursor < content.length) {
    const start = content.indexOf(THOUGHT_CHANNEL_OPEN, cursor);
    if (start < 0) {
      visibleContent += content.slice(cursor);
      break;
    }

    visibleContent += content.slice(cursor, start);
    const thoughtStart = start + THOUGHT_CHANNEL_OPEN.length;
    const end = content.indexOf(THOUGHT_CHANNEL_CLOSE, thoughtStart);
    if (end < 0) {
      thinking.push(content.slice(thoughtStart));
      break;
    }

    thinking.push(content.slice(thoughtStart, end));
    cursor = end + THOUGHT_CHANNEL_CLOSE.length;
  }

  return { visibleContent, thinking };
}

function emitThinking(
  thinking: readonly string[],
  step: number,
  maxThinkingChars: number,
  enabled: boolean,
  onAgentEvent: AgentTurnOptions["onAgentEvent"],
): void {
  if (!enabled) {
    return;
  }
  for (const content of thinking) {
    const trimmed = content.trim();
    if (trimmed.length === 0) {
      continue;
    }
    if (trimmed.length > maxThinkingChars) {
      onAgentEvent?.({
        type: "thinking",
        step,
        content: trimmed.slice(0, maxThinkingChars),
        truncated: true,
      });
      continue;
    }
    onAgentEvent?.({ type: "thinking", step, content: trimmed });
  }
}

async function executeParsedTool(
  call: AgentToolCall,
  step: number,
  options: AgentTurnOptions,
): Promise<AgentToolResult> {
  options.onAgentEvent?.({ type: "toolCall", step, call });

  let result: AgentToolResult;
  try {
    result = await options.executeTool(call, options.signal ?? new AbortController().signal);
  } catch (error) {
    result = {
      callId: call.id,
      ok: false,
      error: normalizeThrownToolError(error),
    };
  }

  options.onAgentEvent?.({ type: "toolResult", step, result });
  return result;
}

async function executeParsedToolItems(
  parseResult: Exclude<AgentToolParseResult, { type: "none" }>,
  step: number,
  options: AgentTurnOptions,
): Promise<Array<{ result: AgentToolResult; toolName: AgentToolName | "unknown" }>> {
  if (parseResult.type === "error") {
    return [{
      result: toolStepErrorResult(parseResult, step, options),
      toolName: "unknown",
    }];
  }

  const responses: Array<{ result: AgentToolResult; toolName: AgentToolName | "unknown" }> = [];
  for (const item of parseResult.items) {
    if (item.type === "call") {
      responses.push({
        result: await executeParsedTool(item.call, step, options),
        toolName: item.call.name,
      });
    } else {
      responses.push({
        result: toolStepErrorResult(item, step, options),
        toolName: item.toolName,
      });
    }
  }
  return responses;
}

function toolStepErrorResult(
  parseResult: Extract<AgentToolParseResult | AgentToolParseItem, { type: "error" }>,
  step: number,
  options: AgentTurnOptions,
): AgentToolResult {
  options.onAgentEvent?.({
    type: "stepError",
    step,
    callId: parseResult.callId,
    error: parseResult.error,
  });
  return {
    callId: parseResult.callId,
    ok: false,
    error: parseResult.error,
  };
}

function emitVisibleText(
  content: string,
  step: number,
  onToken: ChatTurnOptions["onToken"],
  onAgentEvent: AgentTurnOptions["onAgentEvent"],
): void {
  if (content.length === 0) {
    return;
  }
  onAgentEvent?.({ type: "text", step, content });
  onToken?.({
    tokenId: -1,
    token: "",
    text: content,
    content,
  });
}

function buildInitialTurn(
  userContent: string,
  tools: readonly AgentToolDefinition[],
): ChatMessage[] {
  const messages: ChatMessage[] = [];
  if (tools.length > 0) {
    messages.push({
      role: "system",
      content: "",
      toolDeclarations: tools.map(toChatToolDeclaration),
    });
  }
  messages.push({ role: "user", content: userContent });
  return messages;
}

function toChatToolDeclaration(tool: AgentToolDefinition): ChatToolDeclaration {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parametersJsonSchema,
    },
  };
}

function buildToolResponseTurn(result: AgentToolResult, toolName: AgentToolName | "unknown"): ChatMessage {
  return {
    role: "tool",
    tool_call_id: result.callId,
    name: toolName,
    content: result,
  };
}

function normalizeThrownToolError(error: unknown): AgentToolError {
  if (isPlainObject(error)) {
    const code = typeof error.code === "string" ? error.code : "tool_executor_error";
    const message = typeof error.message === "string" ? error.message : "Tool executor failed.";
    const retryable = typeof error.retryable === "boolean" ? error.retryable : undefined;
    return retryable === undefined ? { code, message } : { code, message, retryable };
  }

  if (error instanceof Error) {
    return {
      code: "tool_executor_error",
      message: error.message,
    };
  }

  return {
    code: "tool_executor_error",
    message: "Tool executor failed.",
  };
}

function normalizeMaxToolSteps(value: number | undefined): number {
  if (value === undefined) {
    return DEFAULT_MAX_TOOL_STEPS;
  }
  if (!Number.isFinite(value)) {
    return DEFAULT_MAX_TOOL_STEPS;
  }
  return Math.max(0, Math.floor(value));
}

function normalizeMaxThinkingChars(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value)) {
    return DEFAULT_MAX_THINKING_CHARS;
  }
  return Math.max(0, Math.floor(value));
}

function validateJsonSchema(value: unknown, schema: unknown): boolean {
  if (!isPlainObject(schema)) {
    return false;
  }

  const enumValues = schema.enum;
  if (enumValues !== undefined) {
    if (!Array.isArray(enumValues)) {
      return false;
    }
    if (!enumValues.some((candidate) => jsonValuesEqual(candidate, value))) {
      return false;
    }
  }

  const schemaType = schema.type;
  if (schemaType !== undefined && !matchesSchemaType(value, schemaType)) {
    return false;
  }

  if (typeof value === "string" && !validateStringKeywords(value, schema)) {
    return false;
  }
  if (typeof value === "number" && !validateNumberKeywords(value, schema)) {
    return false;
  }
  if (Array.isArray(value) && !validateArrayKeywords(value, schema)) {
    return false;
  }
  if (isPlainObject(value) && !validateObjectKeywords(value, schema)) {
    return false;
  }

  return true;
}

function validateStringKeywords(value: string, schema: JsonObject): boolean {
  if (schema.minLength !== undefined && (!isNonNegativeInteger(schema.minLength) || value.length < schema.minLength)) {
    return false;
  }
  if (schema.maxLength !== undefined && (!isNonNegativeInteger(schema.maxLength) || value.length > schema.maxLength)) {
    return false;
  }
  return true;
}

function validateNumberKeywords(value: number, schema: JsonObject): boolean {
  if (schema.minimum !== undefined && (typeof schema.minimum !== "number" || value < schema.minimum)) {
    return false;
  }
  if (schema.maximum !== undefined && (typeof schema.maximum !== "number" || value > schema.maximum)) {
    return false;
  }
  return true;
}

function validateArrayKeywords(value: readonly unknown[], schema: JsonObject): boolean {
  if (schema.minItems !== undefined && (!isNonNegativeInteger(schema.minItems) || value.length < schema.minItems)) {
    return false;
  }
  if (schema.maxItems !== undefined && (!isNonNegativeInteger(schema.maxItems) || value.length > schema.maxItems)) {
    return false;
  }
  if (schema.items === undefined) {
    return true;
  }
  if (!isPlainObject(schema.items)) {
    return false;
  }
  return value.every((item) => validateJsonSchema(item, schema.items));
}

function validateObjectKeywords(value: JsonObject, schema: JsonObject): boolean {
  const properties = schema.properties;
  if (properties !== undefined && !isPlainObject(properties)) {
    return false;
  }

  const required = schema.required;
  if (required !== undefined) {
    if (!Array.isArray(required) || !required.every((item) => typeof item === "string")) {
      return false;
    }
    for (const key of required) {
      if (!Object.hasOwn(value, key)) {
        return false;
      }
    }
  }

  if (isPlainObject(properties)) {
    for (const [key, propertySchema] of Object.entries(properties)) {
      if (Object.hasOwn(value, key) && !validateJsonSchema(value[key], propertySchema)) {
        return false;
      }
    }
  }

  const additionalProperties = schema.additionalProperties;
  if (additionalProperties === false && isPlainObject(properties)) {
    for (const key of Object.keys(value)) {
      if (!Object.hasOwn(properties, key)) {
        return false;
      }
    }
  } else if (additionalProperties === false) {
    return Object.keys(value).length === 0;
  } else if (isPlainObject(additionalProperties) && isPlainObject(properties)) {
    for (const key of Object.keys(value)) {
      if (!Object.hasOwn(properties, key) && !validateJsonSchema(value[key], additionalProperties)) {
        return false;
      }
    }
  } else if (additionalProperties !== undefined && additionalProperties !== true && additionalProperties !== false) {
    return false;
  }

  return true;
}

function matchesSchemaType(value: unknown, schemaType: unknown): boolean {
  if (typeof schemaType === "string") {
    return matchesSingleSchemaType(value, schemaType);
  }
  if (Array.isArray(schemaType) && schemaType.every((item) => typeof item === "string")) {
    return schemaType.some((item) => matchesSingleSchemaType(value, item));
  }
  return false;
}

function matchesSingleSchemaType(value: unknown, schemaType: string): boolean {
  switch (schemaType) {
    case "array":
      return Array.isArray(value);
    case "boolean":
      return typeof value === "boolean";
    case "integer":
      return Number.isInteger(value);
    case "null":
      return value === null;
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "object":
      return isPlainObject(value);
    case "string":
      return typeof value === "string";
    default:
      return false;
  }
}

function jsonValuesEqual(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function invalidToolCallFormat(callId: string, message: string): AgentToolParseResult {
  return {
    type: "error",
    callId,
    error: {
      code: "invalid_tool_call_format",
      message,
    },
  };
}

function invalidToolCallFormatItem(callId: string, message: string): AgentToolParseItem {
  return {
    type: "error",
    callId,
    toolName: "unknown",
    error: {
      code: "invalid_tool_call_format",
      message,
    },
  };
}

function unknownToolItem(callId: string, message: string): AgentToolParseItem {
  return {
    type: "error",
    callId,
    toolName: "unknown",
    error: {
      code: "unknown_tool",
      message,
    },
  };
}

function invalidToolArgumentsItem(
  callId: string,
  toolName: AgentToolName,
  message: string,
): AgentToolParseItem {
  return {
    type: "error",
    callId,
    toolName,
    error: {
      code: "invalid_tool_arguments",
      message,
    },
  };
}

function isAgentToolName(value: string): value is AgentToolName {
  return (AGENT_TOOL_NAMES as readonly string[]).includes(value);
}

function toolCallIdForStepIndex(step: number, index: number): string {
  return `tool_${step}_${index}`;
}

function isPlainObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

type ToolCallExtraction =
  | { type: "none" }
  | { type: "calls"; bodies: readonly string[] }
  | { type: "malformed" };

type NativeToolCall = {
  name: string;
  arguments: JsonObject;
};

function extractNativeToolCalls(content: string): ToolCallExtraction {
  const bodies: string[] = [];
  let searchStart = 0;

  while (searchStart < content.length) {
    const openIndex = findNativeTagOutsideString(content, TOOL_CALL_OPEN_TAG, searchStart);
    const closeIndex = findNativeTagOutsideString(content, TOOL_CALL_CLOSE_TAG, searchStart);

    if (openIndex < 0) {
      if (closeIndex >= 0) {
        return { type: "malformed" };
      }
      return extractedToolCallBodies(bodies);
    }
    if (closeIndex >= 0 && closeIndex < openIndex) {
      return { type: "malformed" };
    }

    const bodyStart = openIndex + TOOL_CALL_OPEN_TAG.length;
    const bodyCloseIndex = findNativeTagOutsideString(content, TOOL_CALL_CLOSE_TAG, bodyStart);
    if (bodyCloseIndex < 0) {
      return { type: "malformed" };
    }

    bodies.push(content.slice(bodyStart, bodyCloseIndex));
    searchStart = bodyCloseIndex + TOOL_CALL_CLOSE_TAG.length;
  }

  return extractedToolCallBodies(bodies);
}

function extractedToolCallBodies(bodies: readonly string[]): ToolCallExtraction {
  return bodies.length > 0 ? { type: "calls", bodies } : { type: "none" };
}

function parseNativeToolCallBody(body: string): NativeToolCall {
  const parser = new NativeValueParser(body);
  parser.consumeLiteral("call:");
  const name = parser.consumeIdentifier();
  const parsedArguments = parser.consumeValue();
  parser.consumeWhitespace();
  if (!parser.isDone()) {
    throw new Error("Unexpected trailing native tool call text.");
  }
  if (!isPlainObject(parsedArguments)) {
    throw new Error("Native tool call arguments must be an object.");
  }
  return { name, arguments: parsedArguments };
}

function findNativeTagOutsideString(content: string, tag: string, start: number): number {
  let inString = false;

  for (let index = start; index < content.length; index += 1) {
    if (content.startsWith('<|"|>', index)) {
      inString = !inString;
      index += '<|"|>'.length - 1;
      continue;
    }

    if (!inString && content.startsWith(tag, index)) {
      return index;
    }
  }

  return -1;
}

class NativeValueParser {
  private index = 0;
  private readonly input: string;

  constructor(input: string) {
    this.input = input;
  }

  isDone(): boolean {
    return this.index >= this.input.length;
  }

  consumeWhitespace(): void {
    while (this.index < this.input.length && /\s/.test(this.input[this.index] ?? "")) {
      this.index += 1;
    }
  }

  consumeLiteral(literal: string): void {
    this.consumeWhitespace();
    if (!this.input.startsWith(literal, this.index)) {
      throw new Error(`Expected ${literal}.`);
    }
    this.index += literal.length;
  }

  consumeIdentifier(): string {
    this.consumeWhitespace();
    const match = /^[A-Za-z_][A-Za-z0-9_-]*/.exec(this.input.slice(this.index));
    if (!match) {
      throw new Error("Expected native identifier.");
    }
    this.index += match[0].length;
    return match[0];
  }

  consumeValue(): unknown {
    this.consumeWhitespace();
    if (this.input.startsWith('<|"|>', this.index)) {
      return this.consumeString();
    }
    const char = this.input[this.index];
    if (char === "{") {
      return this.consumeObject();
    }
    if (char === "[") {
      return this.consumeArray();
    }
    if (this.input.startsWith("true", this.index)) {
      this.index += 4;
      return true;
    }
    if (this.input.startsWith("false", this.index)) {
      this.index += 5;
      return false;
    }
    if (this.input.startsWith("null", this.index)) {
      this.index += 4;
      return null;
    }
    return this.consumeNumber();
  }

  private consumeString(): string {
    this.consumeLiteral('<|"|>');
    const end = this.input.indexOf('<|"|>', this.index);
    if (end < 0) {
      throw new Error("Unclosed native string delimiter.");
    }
    const value = this.input.slice(this.index, end);
    this.index = end + '<|"|>'.length;
    return value;
  }

  private consumeObject(): JsonObject {
    this.consumeLiteral("{");
    const output: JsonObject = {};
    this.consumeWhitespace();
    if (this.input[this.index] === "}") {
      this.index += 1;
      return output;
    }

    while (true) {
      const key = this.input.startsWith('<|"|>', this.index)
        ? this.consumeString()
        : this.consumeIdentifier();
      this.consumeLiteral(":");
      output[key] = this.consumeValue();
      this.consumeWhitespace();
      const char = this.input[this.index];
      if (char === ",") {
        this.index += 1;
        continue;
      }
      if (char === "}") {
        this.index += 1;
        return output;
      }
      throw new Error("Expected native object delimiter.");
    }
  }

  private consumeArray(): unknown[] {
    this.consumeLiteral("[");
    const output: unknown[] = [];
    this.consumeWhitespace();
    if (this.input[this.index] === "]") {
      this.index += 1;
      return output;
    }

    while (true) {
      output.push(this.consumeValue());
      this.consumeWhitespace();
      const char = this.input[this.index];
      if (char === ",") {
        this.index += 1;
        continue;
      }
      if (char === "]") {
        this.index += 1;
        return output;
      }
      throw new Error("Expected native array delimiter.");
    }
  }

  private consumeNumber(): number {
    this.consumeWhitespace();
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(this.input.slice(this.index));
    if (!match) {
      throw new Error("Expected native value.");
    }
    this.index += match[0].length;
    return Number(match[0]);
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Agent turn was aborted.", "AbortError");
  }
}
