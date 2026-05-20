import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_MAX_TOOL_STEPS,
  formatToolResponse,
  generateAgentTurn,
  parseToolCall,
  type AgentChatTurnGenerator,
  type AgentEvent,
  type AgentToolDefinition,
  type AgentToolResult,
} from "../src/index.ts";
import type {
  ChatCompletionChunk,
  InferenceState,
  ModelSession,
  Tokenizer,
} from "@heliopause/engine";

const sandboxCommandTool: AgentToolDefinition = {
  name: "sandbox_command",
  description: "Run an allowed sandbox command.",
  parametersJsonSchema: {
    type: "object",
    required: ["cmd", "args"],
    additionalProperties: false,
    properties: {
      cmd: { type: "string", enum: ["ls", "cat"] },
      args: { type: "array", items: { type: "string" } },
    },
  },
};

const readFileTool: AgentToolDefinition = {
  name: "sandbox_read_file",
  description: "Read a virtual file.",
  parametersJsonSchema: {
    type: "object",
    required: ["path"],
    additionalProperties: false,
    properties: {
      path: { type: "string", minLength: 1 },
    },
  },
};

test("parseToolCall returns none for normal assistant text", () => {
  assert.deepEqual(parseToolCall("Hello there.", [sandboxCommandTool], 1), { type: "none" });
});

test("parseToolCall accepts a valid tool call and assigns a step id", () => {
  assert.deepEqual(
    parseToolCall(
      '<tool_call>{"tool":"sandbox_command","arguments":{"cmd":"ls","args":["/workspace"]}}</tool_call>',
      [sandboxCommandTool],
      2,
    ),
    {
      type: "call",
      call: {
        id: "tool_2",
        name: "sandbox_command",
        arguments: {
          cmd: "ls",
          args: ["/workspace"],
        },
      },
    },
  );
});

test("parseToolCall accepts tag text inside JSON string values", () => {
  const writeFileTool: AgentToolDefinition = {
    name: "sandbox_write_file",
    description: "Write a virtual file.",
    parametersJsonSchema: {
      type: "object",
      required: ["path", "content"],
      additionalProperties: false,
      properties: {
        path: { type: "string" },
        content: { type: "string" },
      },
    },
  };

  assert.deepEqual(
    parseToolCall(
      '<tool_call>{"tool":"sandbox_write_file","arguments":{"path":"a.txt","content":"literal </tool_call> text"}}</tool_call>',
      [writeFileTool],
      1,
    ),
    {
      type: "call",
      call: {
        id: "tool_1",
        name: "sandbox_write_file",
        arguments: {
          path: "a.txt",
          content: "literal </tool_call> text",
        },
      },
    },
  );
});

test("parseToolCall rejects malformed JSON and unmatched tags", () => {
  assert.equal(
    parseToolCall('<tool_call>{"tool":</tool_call>', [sandboxCommandTool], 1).type,
    "error",
  );
  assert.deepEqual(
    parseToolCall("<tool_call>{}", [sandboxCommandTool], 1),
    {
      type: "error",
      callId: "tool_1",
      error: {
        code: "invalid_tool_call_json",
        message: "Tool call tags are incomplete or malformed.",
      },
    },
  );
});

test("parseToolCall rejects multiple complete tool calls", () => {
  const result = parseToolCall(
    '<tool_call>{"tool":"sandbox_read_file","arguments":{"path":"a.txt"}}</tool_call><tool_call>{"tool":"sandbox_read_file","arguments":{"path":"b.txt"}}</tool_call>',
    [readFileTool],
    1,
  );
  assert.equal(result.type, "error");
  assert.equal(result.type === "error" ? result.error.code : "", "multiple_tool_calls");
});

test("parseToolCall rejects unknown tools, missing arguments, and schema mismatches", () => {
  const unknown = parseToolCall(
    '<tool_call>{"tool":"missing_tool","arguments":{}}</tool_call>',
    [sandboxCommandTool],
    1,
  );
  assert.equal(unknown.type === "error" ? unknown.error.code : "", "unknown_tool");

  const missingArguments = parseToolCall(
    '<tool_call>{"tool":"sandbox_command"}</tool_call>',
    [sandboxCommandTool],
    1,
  );
  assert.equal(missingArguments.type === "error" ? missingArguments.error.code : "", "invalid_tool_arguments");

  const invalidArguments = parseToolCall(
    '<tool_call>{"tool":"sandbox_command","arguments":{"cmd":"rm","args":[]}}</tool_call>',
    [sandboxCommandTool],
    1,
  );
  assert.equal(invalidArguments.type === "error" ? invalidArguments.error.code : "", "invalid_tool_arguments");
});

test("parseToolCall treats invalid schemas as argument validation failures", () => {
  const invalidSchemaTool: AgentToolDefinition = {
    name: "sandbox_read_file",
    description: "Read a virtual file.",
    parametersJsonSchema: {
      type: 42,
    },
  };

  const result = parseToolCall(
    '<tool_call>{"tool":"sandbox_read_file","arguments":{"path":"a.txt"}}</tool_call>',
    [invalidSchemaTool],
    1,
  );

  assert.equal(result.type === "error" ? result.error.code : "", "invalid_tool_arguments");
});

test("parseToolCall rejects additional properties when properties are omitted", () => {
  const noPropertiesTool: AgentToolDefinition = {
    name: "sandbox_read_file",
    description: "Read a virtual file.",
    parametersJsonSchema: {
      type: "object",
      additionalProperties: false,
    },
  };

  const result = parseToolCall(
    '<tool_call>{"tool":"sandbox_read_file","arguments":{"path":"a.txt"}}</tool_call>',
    [noPropertiesTool],
    1,
  );

  assert.equal(result.type === "error" ? result.error.code : "", "invalid_tool_arguments");
});

test("formatToolResponse wraps success and failure envelopes", () => {
  assert.equal(
    formatToolResponse({ callId: "tool_1", ok: true, content: { answer: 42 } }),
    '<tool_response>{"callId":"tool_1","ok":true,"content":{"answer":42}}</tool_response>',
  );
  assert.equal(
    formatToolResponse({ callId: "tool_2", ok: false, error: { code: "nope", message: "Nope." } }),
    '<tool_response>{"callId":"tool_2","ok":false,"error":{"code":"nope","message":"Nope."}}</tool_response>',
  );
});

test("generateAgentTurn returns a normal answer without executing tools", async () => {
  const events: AgentEvent[] = [];
  const tokenTexts: string[] = [];
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "Say hi.", {
    tools: [sandboxCommandTool],
    chatTurnGenerator: mockGenerator(["Hello."]),
    executeTool: async () => {
      throw new Error("executeTool should not be called");
    },
    onAgentEvent: (event) => events.push(event),
    onToken: (chunk) => tokenTexts.push(chunk.text),
  });

  assert.equal(result.content, "Hello.");
  assert.equal(result.finishReason, "stop");
  assert.equal(result.steps, 0);
  assert.deepEqual(tokenTexts, ["Hello."]);
  assert.deepEqual(events.map((event) => event.type), ["text", "done"]);
});

test("generateAgentTurn executes a valid tool call and then emits final text", async () => {
  const calls: string[] = [];
  const events: AgentEvent[] = [];
  const tokenTexts: string[] = [];
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "List files.", {
    tools: [sandboxCommandTool],
    chatTurnGenerator: mockGenerator([
      '<tool_call>{"tool":"sandbox_command","arguments":{"cmd":"ls","args":["/workspace"]}}</tool_call>',
      "notes.md",
    ], calls),
    executeTool: async (call) => ({
      callId: call.id,
      ok: true,
      content: { kind: "sandbox_command", exitCode: 0, stdout: "notes.md\n", stderr: "", truncated: false },
    }),
    onAgentEvent: (event) => events.push(event),
    onToken: (chunk) => tokenTexts.push(chunk.text),
  });

  assert.equal(result.content, "notes.md");
  assert.equal(result.steps, 1);
  assert.deepEqual(events.map((event) => event.type), ["toolCall", "toolResult", "text", "done"]);
  assert.deepEqual(tokenTexts, ["notes.md"]);
  assert.match(calls[0] ?? "", /Tool instructions:/);
  assert.match(calls[1] ?? "", /^<tool_response>/);
});

test("generateAgentTurn returns parser errors as tool responses without toolResult events", async () => {
  const calls: string[] = [];
  const events: AgentEvent[] = [];
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "Use a tool.", {
    tools: [sandboxCommandTool],
    chatTurnGenerator: mockGenerator([
      '<tool_call>{"tool":</tool_call>',
      "Recovered.",
    ], calls),
    executeTool: async () => {
      throw new Error("executeTool should not be called");
    },
    onAgentEvent: (event) => events.push(event),
  });

  assert.equal(result.content, "Recovered.");
  assert.deepEqual(events.map((event) => event.type), ["stepError", "text", "done"]);
  assert.match(calls[1] ?? "", /"code":"invalid_tool_call_json"/);
});

test("generateAgentTurn normalizes executor throws into tool results", async () => {
  const calls: string[] = [];
  const events: AgentEvent[] = [];
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "List files.", {
    tools: [sandboxCommandTool],
    chatTurnGenerator: mockGenerator([
      '<tool_call>{"tool":"sandbox_command","arguments":{"cmd":"ls","args":[]}}</tool_call>',
      "Recovered after failure.",
    ], calls),
    executeTool: async () => {
      throw { code: "custom_failure", message: "Custom failure.", retryable: true };
    },
    onAgentEvent: (event) => events.push(event),
  });

  assert.equal(result.content, "Recovered after failure.");
  const toolResult = events.find((event): event is Extract<AgentEvent, { type: "toolResult" }> => event.type === "toolResult");
  assert.deepEqual(toolResult?.result, {
    callId: "tool_1",
    ok: false,
    error: {
      code: "custom_failure",
      message: "Custom failure.",
      retryable: true,
    },
  });
  assert.match(calls[1] ?? "", /"code":"custom_failure"/);
});

test("generateAgentTurn stops executing tools after maxToolSteps and hides final raw tool JSON", async () => {
  const events: AgentEvent[] = [];
  const tokenTexts: string[] = [];
  let executeCount = 0;
  const state = fakeState();
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, state, "Keep using tools.", {
    tools: [sandboxCommandTool],
    maxToolSteps: 1,
    chatTurnGenerator: mockGenerator([
      '<tool_call>{"tool":"sandbox_command","arguments":{"cmd":"ls","args":[]}}</tool_call>',
      '<tool_call>{"tool":"sandbox_command","arguments":{"cmd":"cat","args":["a.txt"]}}</tool_call>',
    ]),
    executeTool: async (call): Promise<AgentToolResult> => {
      executeCount += 1;
      return { callId: call.id, ok: true, content: { ok: true } };
    },
    onAgentEvent: (event) => events.push(event),
    onToken: (chunk) => tokenTexts.push(chunk.text),
  });

  assert.equal(executeCount, 1);
  assert.equal(result.finishReason, "maxToolSteps");
  assert.equal(result.steps, 1);
  assert.equal(result.content.includes("<tool_call>"), false);
  assert.deepEqual(tokenTexts, [result.content]);
  assert.deepEqual(events.map((event) => event.type), ["toolCall", "toolResult", "text", "done"]);
  const doneEvent = events.at(-1);
  assert.equal(doneEvent?.type === "done" ? doneEvent.finishReason : "", "maxToolSteps");
});

test("generateAgentTurn propagates the same AbortSignal to generation and executor", async () => {
  const controller = new AbortController();
  let generationSignal: AbortSignal | undefined;
  let executorSignal: AbortSignal | undefined;
  const chatTurnGenerator: AgentChatTurnGenerator = async (_session, _tokenizer, state, _userContent, options = {}) => {
    generationSignal = options.signal;
    options.onToken?.(chunk('<tool_call>{"tool":"sandbox_command","arguments":{"cmd":"ls","args":[]}}</tool_call>'));
    return {
      content: '<tool_call>{"tool":"sandbox_command","arguments":{"cmd":"ls","args":[]}}</tool_call>',
      finishReason: "stop",
      state,
    };
  };

  await generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "List.", {
    tools: [sandboxCommandTool],
    maxToolSteps: 1,
    signal: controller.signal,
    chatTurnGenerator,
    executeTool: async (call, signal) => {
      executorSignal = signal;
      return { callId: call.id, ok: true, content: { ok: true } };
    },
  });

  assert.equal(generationSignal, controller.signal);
  assert.equal(executorSignal, controller.signal);
});

test("generateAgentTurn returns the pre-final state when max-step fallback discards a tool call", async () => {
  const state = fakeState();
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, state, "Keep using tools.", {
    tools: [sandboxCommandTool],
    maxToolSteps: 1,
    chatTurnGenerator: mutatingMockGenerator([
      '<tool_call>{"tool":"sandbox_command","arguments":{"cmd":"ls","args":[]}}</tool_call>',
      '<tool_call>{"tool":"sandbox_command","arguments":{"cmd":"cat","args":["a.txt"]}}</tool_call>',
    ]),
    executeTool: async (call): Promise<AgentToolResult> => ({ callId: call.id, ok: true, content: { ok: true } }),
  });

  assert.equal(state.nextPosition, 1);
  assert.equal(result.state.nextPosition, 1);
  assert.notEqual(result.state, state);
});

test("generateAgentTurn rejects immediately for pre-aborted signals", async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "Nope.", {
      tools: [sandboxCommandTool],
      signal: controller.signal,
      executeTool: async () => ({ callId: "tool_1", ok: true, content: {} }),
    }),
    (error) => error instanceof DOMException && error.name === "AbortError",
  );
});

test("DEFAULT_MAX_TOOL_STEPS is three", () => {
  assert.equal(DEFAULT_MAX_TOOL_STEPS, 3);
});

function mockGenerator(outputs: readonly string[], calls: string[] = []): AgentChatTurnGenerator {
  let index = 0;
  return async (_session, _tokenizer, state, userContent, options = {}) => {
    calls.push(userContent);
    const content = outputs[index] ?? outputs.at(-1) ?? "";
    index += 1;
    options.onToken?.(chunk(content));
    return {
      content,
      finishReason: "stop",
      state,
    };
  };
}

function mutatingMockGenerator(outputs: readonly string[]): AgentChatTurnGenerator {
  const generator = mockGenerator(outputs);
  return async (session, tokenizer, state, userContent, options) => {
    state.nextPosition += 1;
    return generator(session, tokenizer, state, userContent, options);
  };
}

function chunk(content: string): ChatCompletionChunk {
  return {
    tokenId: 0,
    token: "",
    text: content,
    content,
  };
}

function fakeState(): InferenceState {
  return {
    fullAttention: new Map(),
    contextLength: 128,
    nextPosition: 0,
  };
}

const fakeSession = {} as ModelSession;
const fakeTokenizer = {} as Tokenizer;
