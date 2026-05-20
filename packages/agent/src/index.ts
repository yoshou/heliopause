import {
  cloneInferenceState,
  generateChatTurn,
  type ChatCompletionChunk,
  type ChatTurnOptions,
  type ChatTurnResult,
  type InferenceState,
  type ModelSession,
  type Tokenizer,
} from "@heliopause/engine";

export const TOOL_CALL_TAG = "tool_call";
export const TOOL_RESPONSE_TAG = "tool_response";
export const DEFAULT_MAX_TOOL_STEPS = 3;

const TOOL_CALL_OPEN_TAG = `<${TOOL_CALL_TAG}>`;
const TOOL_CALL_CLOSE_TAG = `</${TOOL_CALL_TAG}>`;
const MAX_TOOL_STEPS_FINAL_PROMPT =
  "You have reached the maximum number of tool steps. Answer the user using only the tool responses already provided. Do not call another tool.";
const MAX_TOOL_STEPS_FALLBACK =
  "I reached the maximum number of tool steps and cannot call another tool. I do not have enough final text to show beyond the tool results already gathered.";

const AGENT_TOOL_NAMES = [
  "sandbox_list_files",
  "sandbox_read_file",
  "sandbox_write_file",
  "sandbox_command",
  "web_search",
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
  userContent: string,
  options?: ChatTurnOptions,
) => Promise<ChatTurnResult>;

export type AgentTurnOptions = ChatTurnOptions & {
  tools: readonly AgentToolDefinition[];
  executeTool: AgentToolExecutor;
  maxToolSteps?: number;
  onAgentEvent?: (event: AgentEvent) => void;
  chatTurnGenerator?: AgentChatTurnGenerator;
};

export type AgentTurnResult = {
  content: string;
  finishReason: "stop" | "length" | "maxToolSteps";
  steps: number;
  state: InferenceState;
};

export type AgentToolParseResult =
  | { type: "none" }
  | { type: "call"; call: AgentToolCall }
  | { type: "error"; callId: string; error: AgentToolError };

type JsonObject = Record<string, unknown>;

export function parseToolCall(
  content: string,
  tools: readonly AgentToolDefinition[],
  step: number,
): AgentToolParseResult {
  const callId = toolCallIdForStep(step);
  const extraction = extractToolCallBody(content);

  if (extraction.type === "none") {
    return { type: "none" };
  }

  if (extraction.type === "multiple") {
    return {
      type: "error",
      callId,
      error: {
        code: "multiple_tool_calls",
        message: "Only one tool call is allowed per assistant output.",
      },
    };
  }

  if (extraction.type === "malformed") {
    return invalidToolCallJson(callId, "Tool call tags are incomplete or malformed.");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(extraction.body);
  } catch {
    return invalidToolCallJson(callId, "Tool call JSON could not be parsed.");
  }

  if (!isPlainObject(parsed)) {
    return invalidToolCallJson(callId, "Tool call JSON must be an object.");
  }

  const toolName = parsed.tool;
  if (typeof toolName !== "string") {
    return unknownTool(callId, "Tool call must include a string tool name.");
  }

  const tool = tools.find((candidate) => candidate.name === toolName);
  if (!tool || !isAgentToolName(toolName)) {
    return unknownTool(callId, `Unknown tool: ${toolName}`);
  }

  if (!Object.hasOwn(parsed, "arguments")) {
    return invalidToolArguments(callId, "Tool call must include arguments.");
  }

  if (!validateJsonSchema(parsed.arguments, tool.parametersJsonSchema)) {
    return invalidToolArguments(callId, `Arguments for ${tool.name} do not match the tool schema.`);
  }

  return {
    type: "call",
    call: {
      id: callId,
      name: tool.name,
      arguments: parsed.arguments,
    },
  };
}

export function formatToolResponse(result: AgentToolResult): string {
  return `<${TOOL_RESPONSE_TAG}>${JSON.stringify(result)}</${TOOL_RESPONSE_TAG}>`;
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
  const originalOnToken = options.onToken;
  const chatTurnGenerator = options.chatTurnGenerator ?? generateChatTurn;
  const {
    chatTurnGenerator: _chatTurnGenerator,
    tools: _tools,
    executeTool: _executeTool,
    maxToolSteps: _maxToolSteps,
    onAgentEvent: _onAgentEvent,
    ...chatOptions
  } = options;
  let nextUserContent = buildInitialUserContent(userContent, options.tools);
  let toolSteps = 0;
  let finalOnly = maxToolSteps === 0;

  while (true) {
    throwIfAborted(options.signal);

    const step = toolSteps + 1;
    const chunks: ChatCompletionChunk[] = [];
    const stateBeforeFinalGeneration = finalOnly ? cloneInferenceState(state) : undefined;
    const generationState = finalOnly ? cloneInferenceState(state) : state;
    const generation = await chatTurnGenerator(session, tokenizer, generationState, nextUserContent, {
      ...chatOptions,
      onToken: (chunk: ChatCompletionChunk) => {
        chunks.push(chunk);
      },
    });
    const parseResult = parseToolCall(generation.content, options.tools, step);

    if (parseResult.type === "none") {
      replayVisibleText(chunks, step, originalOnToken, options.onAgentEvent);
      const finishReason = finalOnly ? "maxToolSteps" : generation.finishReason;
      options.onAgentEvent?.({ type: "done", steps: toolSteps, finishReason });
      return {
        content: generation.content,
        finishReason,
        steps: toolSteps,
        state: generation.state,
      };
    }

    if (finalOnly) {
      emitVisibleText(MAX_TOOL_STEPS_FALLBACK, step, originalOnToken, options.onAgentEvent);
      options.onAgentEvent?.({ type: "done", steps: toolSteps, finishReason: "maxToolSteps" });
      return {
        content: MAX_TOOL_STEPS_FALLBACK,
        finishReason: "maxToolSteps",
        steps: toolSteps,
        state: stateBeforeFinalGeneration ?? state,
      };
    }

    toolSteps += 1;

    const toolResult = parseResult.type === "call"
      ? await executeParsedTool(parseResult.call, step, options)
      : toolStepErrorResult(parseResult, step, options);

    const toolResponse = formatToolResponse(toolResult);
    if (toolSteps >= maxToolSteps) {
      finalOnly = true;
      nextUserContent = `${toolResponse}\n\n${MAX_TOOL_STEPS_FINAL_PROMPT}`;
    } else {
      nextUserContent = toolResponse;
    }
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

function toolStepErrorResult(
  parseResult: Extract<AgentToolParseResult, { type: "error" }>,
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

function replayVisibleText(
  chunks: readonly ChatCompletionChunk[],
  step: number,
  onToken: ChatTurnOptions["onToken"],
  onAgentEvent: AgentTurnOptions["onAgentEvent"],
): void {
  for (const chunk of chunks) {
    onAgentEvent?.({ type: "text", step, content: chunk.text });
    onToken?.(chunk);
  }
}

function emitVisibleText(
  content: string,
  step: number,
  onToken: ChatTurnOptions["onToken"],
  onAgentEvent: AgentTurnOptions["onAgentEvent"],
): void {
  onAgentEvent?.({ type: "text", step, content });
  onToken?.({
    tokenId: -1,
    token: "",
    text: content,
    content,
  });
}

function buildInitialUserContent(
  userContent: string,
  tools: readonly AgentToolDefinition[],
): string {
  const toolDescriptions = tools.length === 0
    ? "No tools are available."
    : tools
      .map((tool) => JSON.stringify({
        name: tool.name,
        description: tool.description,
        parametersJsonSchema: tool.parametersJsonSchema,
        requiresConfirmation: tool.requiresConfirmation ?? false,
      }))
      .join("\n");

  return [
    "Tool instructions:",
    `- To call a tool, output exactly one <${TOOL_CALL_TAG}> tag containing compact JSON.`,
    "- The JSON object must include tool and arguments fields.",
    "- Do not output more than one tool call in a single response.",
    "- If no tool is needed, answer normally without tool tags.",
    "Available tools:",
    toolDescriptions,
    "",
    "User request:",
    userContent,
  ].join("\n");
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

function invalidToolCallJson(callId: string, message: string): AgentToolParseResult {
  return {
    type: "error",
    callId,
    error: {
      code: "invalid_tool_call_json",
      message,
    },
  };
}

function unknownTool(callId: string, message: string): AgentToolParseResult {
  return {
    type: "error",
    callId,
    error: {
      code: "unknown_tool",
      message,
    },
  };
}

function invalidToolArguments(callId: string, message: string): AgentToolParseResult {
  return {
    type: "error",
    callId,
    error: {
      code: "invalid_tool_arguments",
      message,
    },
  };
}

function isAgentToolName(value: string): value is AgentToolName {
  return (AGENT_TOOL_NAMES as readonly string[]).includes(value);
}

function toolCallIdForStep(step: number): string {
  return `tool_${step}`;
}

function isPlainObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

type ToolCallExtraction =
  | { type: "none" }
  | { type: "single"; body: string }
  | { type: "multiple" }
  | { type: "malformed" };

function extractToolCallBody(content: string): ToolCallExtraction {
  const firstOpen = content.indexOf(TOOL_CALL_OPEN_TAG);
  const firstClose = content.indexOf(TOOL_CALL_CLOSE_TAG);

  if (firstOpen < 0) {
    return firstClose < 0 ? { type: "none" } : { type: "malformed" };
  }
  if (firstClose >= 0 && firstClose < firstOpen) {
    return { type: "malformed" };
  }

  const bodyStart = firstOpen + TOOL_CALL_OPEN_TAG.length;
  const closeIndex = findTagOutsideJsonString(content, TOOL_CALL_CLOSE_TAG, bodyStart);
  if (closeIndex < 0) {
    return { type: "malformed" };
  }

  const afterClose = closeIndex + TOOL_CALL_CLOSE_TAG.length;
  const nextOpen = content.indexOf(TOOL_CALL_OPEN_TAG, afterClose);
  if (nextOpen >= 0) {
    const nextClose = findTagOutsideJsonString(content, TOOL_CALL_CLOSE_TAG, nextOpen + TOOL_CALL_OPEN_TAG.length);
    return nextClose >= 0 ? { type: "multiple" } : { type: "malformed" };
  }
  if (content.indexOf(TOOL_CALL_CLOSE_TAG, afterClose) >= 0) {
    return { type: "malformed" };
  }

  return {
    type: "single",
    body: content.slice(bodyStart, closeIndex),
  };
}

function findTagOutsideJsonString(content: string, tag: string, start: number): number {
  let inString = false;
  let escaped = false;

  for (let index = start; index < content.length; index += 1) {
    const char = content[index];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }

    if (char === "\"") {
      inString = true;
      continue;
    }

    if (content.startsWith(tag, index)) {
      return index;
    }
  }

  return -1;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Agent turn was aborted.", "AbortError");
  }
}
