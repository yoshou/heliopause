import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_MAX_TOOL_STEPS,
  generateAgentTurn,
  parseToolCalls,
  type AgentChatTurnGenerator,
  type AgentEvent,
  type AgentModelTurnCloser,
  type AgentToolDefinition,
  type AgentToolResult,
} from "../src/index.ts";
import type {
  ChatCompletionChunk,
  ChatTurnOptions,
  ChatTurnInput,
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

const webFetchTool: AgentToolDefinition = {
  name: "web_fetch",
  description: "Fetch a CORS-readable text resource to the virtual filesystem.",
  parametersJsonSchema: {
    type: "object",
    required: ["url", "path"],
    additionalProperties: false,
    properties: {
      url: { type: "string", minLength: 1, maxLength: 2048 },
      path: { type: "string", minLength: 1, maxLength: 512 },
    },
  },
  requiresConfirmation: true,
};

test("parseToolCalls returns none for normal assistant text", () => {
  assert.deepEqual(parseToolCalls("Hello there.", [sandboxCommandTool], 1), { type: "none" });
});

test("generateAgentTurn continues an open model turn for tool responses", async () => {
  const callOptions: Array<Pick<ChatTurnOptions, "appendTurnEnd" | "continueModelTurn" | "doSample">> = [];
  let index = 0;
  const chatTurnGenerator: AgentChatTurnGenerator = async (_session, _tokenizer, state, _turn, options = {}) => {
    callOptions.push({
      appendTurnEnd: options.appendTurnEnd,
      continueModelTurn: options.continueModelTurn,
      doSample: options.doSample,
    });
    index += 1;
    const content = index === 1
      ? '<|tool_call>call:sandbox_command{cmd:<|"|>ls<|"|>,args:[]}<tool_call|>'
      : "done";
    options.onToken?.(chunk(content));
    return {
      content,
      finishReason: "stop",
      state,
      modelTurnClosed: index !== 1,
    };
  };

  await generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "List files.", {
    tools: [sandboxCommandTool],
    chatTurnGenerator,
    executeTool: async (call) => ({
      callId: call.id,
      ok: true,
      content: { stdout: "notes.md\n" },
    }),
  });

  assert.deepEqual(callOptions, [
    { appendTurnEnd: false, continueModelTurn: false, doSample: false },
    { appendTurnEnd: false, continueModelTurn: true, doSample: false },
  ]);
});

test("parseToolCalls accepts a valid tool call and assigns a step/index id", () => {
  assert.deepEqual(
    parseToolCalls(
      '<|tool_call>call:sandbox_command{cmd:<|"|>ls<|"|>,args:[<|"|>/workspace<|"|>]}<tool_call|>',
      [sandboxCommandTool],
      2,
    ),
    {
      type: "items",
      items: [{
        type: "call",
        call: {
          id: "tool_2_1",
          name: "sandbox_command",
          arguments: {
            cmd: "ls",
            args: ["/workspace"],
          },
        },
      }],
    },
  );
});

test("parseToolCalls accepts web_fetch and validates its schema", () => {
  assert.deepEqual(
    parseToolCalls(
      '<|tool_call>call:web_fetch{url:<|"|>https://example.com/<|"|>,path:<|"|>/workspace/example.txt<|"|>}<tool_call|>',
      [webFetchTool],
      3,
    ),
    {
      type: "items",
      items: [{
        type: "call",
        call: {
          id: "tool_3_1",
          name: "web_fetch",
          arguments: {
            url: "https://example.com/",
            path: "/workspace/example.txt",
          },
        },
      }],
    },
  );

  for (const body of [
    'call:web_fetch{path:<|"|>/workspace/example.txt<|"|>}',
    'call:web_fetch{url:<|"|>https://example.com/<|"|>}',
    'call:web_fetch{url:<|"|>https://example.com/<|"|>,path:<|"|>/workspace/example.txt<|"|>,extra:true}',
    'call:web_fetch{url:42,path:<|"|>/workspace/example.txt<|"|>}',
  ]) {
    const result = parseToolCalls(`<|tool_call>${body}<tool_call|>`, [webFetchTool], 1);
    assert.equal(result.type === "items" ? result.items[0]?.type : "", "error");
    assert.equal(result.type === "items" && result.items[0]?.type === "error" ? result.items[0].error.code : "", "invalid_tool_arguments");
  }
});

test("parseToolCalls accepts tag text inside native string values", () => {
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
    parseToolCalls(
      '<|tool_call>call:sandbox_write_file{path:<|"|>a.txt<|"|>,content:<|"|>literal <tool_call|> text<|"|>}<tool_call|>',
      [writeFileTool],
      1,
    ),
    {
      type: "items",
      items: [{
        type: "call",
        call: {
          id: "tool_1_1",
          name: "sandbox_write_file",
          arguments: {
            path: "a.txt",
            content: "literal <tool_call|> text",
          },
        },
      }],
    },
  );
});

test("parseToolCalls returns per-call errors for malformed bodies and batch errors for malformed tags", () => {
  const malformedBody = parseToolCalls(
    '<|tool_call>call:sandbox_command{cmd:<|"|>ls<|"|><tool_call|>',
    [sandboxCommandTool],
    1,
  );
  assert.equal(malformedBody.type === "items" ? malformedBody.items[0]?.type : "", "error");
  assert.equal(
    malformedBody.type === "items" && malformedBody.items[0]?.type === "error"
      ? malformedBody.items[0].error.code
      : "",
    "invalid_tool_call_format",
  );
  assert.deepEqual(
    parseToolCalls("<|tool_call>call:sandbox_command{}", [sandboxCommandTool], 1),
    {
      type: "error",
      callId: "tool_1_1",
      error: {
        code: "invalid_tool_call_format",
        message: "Native tool call tags are incomplete or malformed.",
      },
    },
  );
});

test("parseToolCalls accepts multiple complete tool calls in order", () => {
  const result = parseToolCalls(
    '<|tool_call>call:sandbox_read_file{path:<|"|>a.txt<|"|>}<tool_call|><|tool_call>call:sandbox_read_file{path:<|"|>b.txt<|"|>}<tool_call|>',
    [readFileTool],
    1,
  );
  assert.equal(result.type, "items");
  assert.deepEqual(result.type === "items" ? result.items : [], [
    {
      type: "call",
      call: {
        id: "tool_1_1",
        name: "sandbox_read_file",
        arguments: { path: "a.txt" },
      },
    },
    {
      type: "call",
      call: {
        id: "tool_1_2",
        name: "sandbox_read_file",
        arguments: { path: "b.txt" },
      },
    },
  ]);
});

test("parseToolCalls returns per-call errors for unknown tools, missing arguments, and schema mismatches", () => {
  const unknown = parseToolCalls(
    '<|tool_call>call:missing_tool{}<tool_call|>',
    [sandboxCommandTool],
    1,
  );
  assert.equal(unknown.type === "items" && unknown.items[0]?.type === "error" ? unknown.items[0].error.code : "", "unknown_tool");

  const malformedArguments = parseToolCalls(
    '<|tool_call>call:sandbox_command[]<tool_call|>',
    [sandboxCommandTool],
    1,
  );
  assert.equal(
    malformedArguments.type === "items" && malformedArguments.items[0]?.type === "error"
      ? malformedArguments.items[0].error.code
      : "",
    "invalid_tool_call_format",
  );

  const invalidArguments = parseToolCalls(
    '<|tool_call>call:sandbox_command{cmd:<|"|>rm<|"|>,args:[]}<tool_call|>',
    [sandboxCommandTool],
    1,
  );
  assert.equal(
    invalidArguments.type === "items" && invalidArguments.items[0]?.type === "error"
      ? invalidArguments.items[0].error.code
      : "",
    "invalid_tool_arguments",
  );
});

test("parseToolCalls keeps valid calls when other calls have per-call errors", () => {
  const result = parseToolCalls(
    [
      '<|tool_call>call:sandbox_read_file{path:<|"|>a.txt<|"|>}<tool_call|>',
      '<|tool_call>call:missing_tool{}<tool_call|>',
      '<|tool_call>call:sandbox_read_file{path:<|"|>b.txt<|"|>}<tool_call|>',
    ].join(""),
    [readFileTool],
    4,
  );

  assert.equal(result.type, "items");
  assert.deepEqual(result.type === "items" ? result.items.map((item) => item.type) : [], ["call", "error", "call"]);
  assert.equal(result.type === "items" && result.items[0]?.type === "call" ? result.items[0].call.id : "", "tool_4_1");
  assert.equal(result.type === "items" && result.items[1]?.type === "error" ? result.items[1].callId : "", "tool_4_2");
  assert.equal(result.type === "items" && result.items[2]?.type === "call" ? result.items[2].call.id : "", "tool_4_3");
});

test("parseToolCalls treats invalid schemas as argument validation failures", () => {
  const invalidSchemaTool: AgentToolDefinition = {
    name: "sandbox_read_file",
    description: "Read a virtual file.",
    parametersJsonSchema: {
      type: 42,
    },
  };

  const result = parseToolCalls(
    '<|tool_call>call:sandbox_read_file{path:<|"|>a.txt<|"|>}<tool_call|>',
    [invalidSchemaTool],
    1,
  );

  assert.equal(result.type === "items" && result.items[0]?.type === "error" ? result.items[0].error.code : "", "invalid_tool_arguments");
});

test("parseToolCalls rejects additional properties when properties are omitted", () => {
  const noPropertiesTool: AgentToolDefinition = {
    name: "sandbox_read_file",
    description: "Read a virtual file.",
    parametersJsonSchema: {
      type: "object",
      additionalProperties: false,
    },
  };

  const result = parseToolCalls(
    '<|tool_call>call:sandbox_read_file{path:<|"|>a.txt<|"|>}<tool_call|>',
    [noPropertiesTool],
    1,
  );

  assert.equal(result.type === "items" && result.items[0]?.type === "error" ? result.items[0].error.code : "", "invalid_tool_arguments");
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
  const calls: ChatTurnInput[] = [];
  const callOptions: Array<Pick<ChatTurnOptions, "appendTurnEnd" | "continueModelTurn">> = [];
  const events: AgentEvent[] = [];
  const tokenTexts: string[] = [];
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "List files.", {
    tools: [sandboxCommandTool],
    chatTurnGenerator: mockGenerator([
      '<|tool_call>call:sandbox_command{cmd:<|"|>ls<|"|>,args:[<|"|>/workspace<|"|>]}<tool_call|>',
      "notes.md",
    ], calls, callOptions),
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
  assert.deepEqual(calls[0], [
    {
      role: "system",
      content: "",
      toolDeclarations: [{
        type: "function",
        function: {
          name: "sandbox_command",
          description: "Run an allowed sandbox command.",
          parameters: sandboxCommandTool.parametersJsonSchema,
        },
      }],
    },
    { role: "user", content: "List files." },
  ]);
  assert.deepEqual(calls[1], [{
    role: "tool",
    tool_call_id: "tool_1_1",
    name: "sandbox_command",
    content: {
      callId: "tool_1_1",
      ok: true,
      content: { kind: "sandbox_command", exitCode: 0, stdout: "notes.md\n", stderr: "", truncated: false },
    },
  }]);
  assert.deepEqual(callOptions, [
    { appendTurnEnd: false, continueModelTurn: false },
    { appendTurnEnd: false, continueModelTurn: false },
  ]);
});

test("generateAgentTurn executes multiple tool calls sequentially in one step", async () => {
  const calls: ChatTurnInput[] = [];
  const events: AgentEvent[] = [];
  const executedCallIds: string[] = [];
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "Read two files.", {
    tools: [readFileTool],
    chatTurnGenerator: mockGenerator([
      [
        '<|tool_call>call:sandbox_read_file{path:<|"|>a.txt<|"|>}<tool_call|>',
        '<|tool_call>call:sandbox_read_file{path:<|"|>b.txt<|"|>}<tool_call|>',
      ].join(""),
      "Read both files.",
    ], calls),
    executeTool: async (call) => {
      executedCallIds.push(call.id);
      const args = call.arguments as { path: string };
      return {
        callId: call.id,
        ok: true,
        content: { kind: "sandbox_read_file", path: args.path, content: args.path.toUpperCase(), truncated: false },
      };
    },
    onAgentEvent: (event) => events.push(event),
  });

  assert.equal(result.content, "Read both files.");
  assert.equal(result.steps, 1);
  assert.deepEqual(executedCallIds, ["tool_1_1", "tool_1_2"]);
  assert.deepEqual(events.map((event) => event.type), ["toolCall", "toolResult", "toolCall", "toolResult", "text", "done"]);
  assert.deepEqual(calls[1], [
    {
      role: "tool",
      tool_call_id: "tool_1_1",
      name: "sandbox_read_file",
      content: {
        callId: "tool_1_1",
        ok: true,
        content: { kind: "sandbox_read_file", path: "a.txt", content: "A.TXT", truncated: false },
      },
    },
    {
      role: "tool",
      tool_call_id: "tool_1_2",
      name: "sandbox_read_file",
      content: {
        callId: "tool_1_2",
        ok: true,
        content: { kind: "sandbox_read_file", path: "b.txt", content: "B.TXT", truncated: false },
      },
    },
  ]);
});

test("generateAgentTurn continues executing remaining calls after a per-call failure", async () => {
  const calls: ChatTurnInput[] = [];
  const events: AgentEvent[] = [];
  const executedCallIds: string[] = [];
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "Use mixed tools.", {
    tools: [readFileTool],
    chatTurnGenerator: mockGenerator([
      [
        '<|tool_call>call:sandbox_read_file{path:<|"|>a.txt<|"|>}<tool_call|>',
        "<|tool_call>call:missing_tool{}<tool_call|>",
        '<|tool_call>call:sandbox_read_file{path:<|"|>b.txt<|"|>}<tool_call|>',
      ].join(""),
      "Recovered with partial results.",
    ], calls),
    executeTool: async (call) => {
      executedCallIds.push(call.id);
      return { callId: call.id, ok: true, content: { ok: true } };
    },
    onAgentEvent: (event) => events.push(event),
  });

  assert.equal(result.content, "Recovered with partial results.");
  assert.deepEqual(executedCallIds, ["tool_1_1", "tool_1_3"]);
  assert.deepEqual(events.map((event) => event.type), [
    "toolCall",
    "toolResult",
    "stepError",
    "toolCall",
    "toolResult",
    "text",
    "done",
  ]);
  assert.match(JSON.stringify(calls[1]), /tool_1_2/);
  assert.match(JSON.stringify(calls[1]), /unknown_tool/);
});

test("generateAgentTurn returns parser errors as tool responses without toolResult events", async () => {
  const calls: ChatTurnInput[] = [];
  const events: AgentEvent[] = [];
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "Use a tool.", {
    tools: [sandboxCommandTool],
    chatTurnGenerator: mockGenerator([
      '<|tool_call>call:sandbox_command{cmd:<|"|>ls<|"|><tool_call|>',
      "Recovered.",
    ], calls),
    executeTool: async () => {
      throw new Error("executeTool should not be called");
    },
    onAgentEvent: (event) => events.push(event),
  });

  assert.equal(result.content, "Recovered.");
  assert.deepEqual(events.map((event) => event.type), ["stepError", "text", "done"]);
  assert.equal(Array.isArray(calls[1]), true);
  const recoveryTurn = calls[1] as unknown as Array<{ content?: unknown }>;
  assert.match(JSON.stringify(recoveryTurn[0]?.content), /invalid_tool_call_format/);
});

test("generateAgentTurn normalizes executor throws into tool results", async () => {
  const calls: ChatTurnInput[] = [];
  const events: AgentEvent[] = [];
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "List files.", {
    tools: [sandboxCommandTool],
    chatTurnGenerator: mockGenerator([
      '<|tool_call>call:sandbox_command{cmd:<|"|>ls<|"|>,args:[]}<tool_call|>',
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
    callId: "tool_1_1",
    ok: false,
    error: {
      code: "custom_failure",
      message: "Custom failure.",
      retryable: true,
    },
  });
  assert.match(JSON.stringify(calls[1]), /custom_failure/);
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
      '<|tool_call>call:sandbox_command{cmd:<|"|>ls<|"|>,args:[]}<tool_call|>',
      '<|tool_call>call:sandbox_command{cmd:<|"|>cat<|"|>,args:[<|"|>a.txt<|"|>]}<tool_call|>',
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
  assert.equal(result.content.includes("<|tool_call>"), false);
  assert.deepEqual(tokenTexts, [result.content]);
  assert.deepEqual(events.map((event) => event.type), ["toolCall", "toolResult", "text", "done"]);
  const doneEvent = events.at(-1);
  assert.equal(doneEvent?.type === "done" ? doneEvent.finishReason : "", "maxToolSteps");
});

test("generateAgentTurn executes an entire multiple-call batch before maxToolSteps final fallback", async () => {
  const events: AgentEvent[] = [];
  const tokenTexts: string[] = [];
  let executeCount = 0;
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, fakeState(), "Read multiple files.", {
    tools: [readFileTool],
    maxToolSteps: 1,
    chatTurnGenerator: mockGenerator([
      [
        '<|tool_call>call:sandbox_read_file{path:<|"|>a.txt<|"|>}<tool_call|>',
        '<|tool_call>call:sandbox_read_file{path:<|"|>b.txt<|"|>}<tool_call|>',
      ].join(""),
      '<|tool_call>call:sandbox_read_file{path:<|"|>c.txt<|"|>}<tool_call|>',
    ]),
    executeTool: async (call): Promise<AgentToolResult> => {
      executeCount += 1;
      return { callId: call.id, ok: true, content: { ok: true } };
    },
    onAgentEvent: (event) => events.push(event),
    onToken: (chunk) => tokenTexts.push(chunk.text),
  });

  assert.equal(executeCount, 2);
  assert.equal(result.finishReason, "maxToolSteps");
  assert.equal(result.steps, 1);
  assert.equal(result.content.includes("<|tool_call>"), false);
  assert.deepEqual(tokenTexts, [result.content]);
  assert.deepEqual(events.map((event) => event.type), ["toolCall", "toolResult", "toolCall", "toolResult", "text", "done"]);
});

test("generateAgentTurn propagates the same AbortSignal to generation and executor", async () => {
  const controller = new AbortController();
  let generationSignal: AbortSignal | undefined;
  let executorSignal: AbortSignal | undefined;
  const chatTurnGenerator: AgentChatTurnGenerator = async (_session, _tokenizer, state, _userContent, options = {}) => {
    generationSignal = options.signal;
    options.onToken?.(chunk('<|tool_call>call:sandbox_command{cmd:<|"|>ls<|"|>,args:[]}<tool_call|>'));
    return {
      content: '<|tool_call>call:sandbox_command{cmd:<|"|>ls<|"|>,args:[]}<tool_call|>',
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
  const closedStates: InferenceState[] = [];
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, state, "Keep using tools.", {
    tools: [sandboxCommandTool],
    maxToolSteps: 1,
    chatTurnGenerator: mutatingOpenTurnMockGenerator([
      '<|tool_call>call:sandbox_command{cmd:<|"|>ls<|"|>,args:[]}<tool_call|>',
      '<|tool_call>call:sandbox_command{cmd:<|"|>cat<|"|>,args:[<|"|>a.txt<|"|>]}<tool_call|>',
    ]),
    closeModelTurn: markingCloseModelTurn(closedStates),
    executeTool: async (call): Promise<AgentToolResult> => ({ callId: call.id, ok: true, content: { ok: true } }),
  });

  assert.equal(state.nextPosition, 2);
  assert.equal(result.state.nextPosition, 2);
  assert.notEqual(result.state, state);
  assert.deepEqual(closedStates, [state, result.state]);
});

test("generateAgentTurn closes the committed state when max-step final text is generated on a clone", async () => {
  const state = fakeState();
  const closedStates: InferenceState[] = [];
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, state, "Keep using tools.", {
    tools: [sandboxCommandTool],
    maxToolSteps: 1,
    chatTurnGenerator: mutatingOpenTurnMockGenerator([
      '<|tool_call>call:sandbox_command{cmd:<|"|>ls<|"|>,args:[]}<tool_call|>',
      "Enough information.",
    ]),
    closeModelTurn: markingCloseModelTurn(closedStates),
    executeTool: async (call): Promise<AgentToolResult> => ({ callId: call.id, ok: true, content: { ok: true } }),
  });

  assert.equal(result.content, "Enough information.");
  assert.equal(result.finishReason, "maxToolSteps");
  assert.equal(state.nextPosition, 2);
  assert.equal(result.state.nextPosition, 3);
  assert.deepEqual(closedStates, [state, result.state]);
});

test("generateAgentTurn can avoid state cloning for provider-owned state", async () => {
  const state = fakeState();
  const closedStates: InferenceState[] = [];
  const result = await generateAgentTurn(fakeSession, fakeTokenizer, state, "Keep using tools.", {
    tools: [sandboxCommandTool],
    maxToolSteps: 1,
    cloneState: false,
    chatTurnGenerator: mutatingOpenTurnMockGenerator([
      '<|tool_call>call:sandbox_command{cmd:<|"|>ls<|"|>,args:[]}<tool_call|>',
      "Enough information.",
    ]),
    closeModelTurn: markingCloseModelTurn(closedStates),
    executeTool: async (call): Promise<AgentToolResult> => ({ callId: call.id, ok: true, content: { ok: true } }),
  });

  assert.equal(result.content, "Enough information.");
  assert.equal(result.finishReason, "maxToolSteps");
  assert.equal(result.state, state);
  assert.equal(state.nextPosition, 3);
  assert.deepEqual(closedStates, [state]);
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

function mockGenerator(
  outputs: readonly string[],
  calls: ChatTurnInput[] = [],
  callOptions: Array<Pick<ChatTurnOptions, "appendTurnEnd" | "continueModelTurn">> = [],
): AgentChatTurnGenerator {
  let index = 0;
  return async (_session, _tokenizer, state, turn, options = {}) => {
    calls.push(turn);
    callOptions.push({
      appendTurnEnd: options.appendTurnEnd,
      continueModelTurn: options.continueModelTurn,
    });
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
  return async (session, tokenizer, state, turn, options) => {
    state.nextPosition += 1;
    return generator(session, tokenizer, state, turn, options);
  };
}

function mutatingOpenTurnMockGenerator(outputs: readonly string[]): AgentChatTurnGenerator {
  const generator = mutatingMockGenerator(outputs);
  return async (session, tokenizer, state, turn, options) => {
    const result = await generator(session, tokenizer, state, turn, options);
    return { ...result, modelTurnClosed: false };
  };
}

function markingCloseModelTurn(closedStates: InferenceState[]): AgentModelTurnCloser {
  return async (_session, _tokenizer, state) => {
    closedStates.push(state);
    state.nextPosition += 1;
    return state;
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
