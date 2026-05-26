import assert from "node:assert/strict";
import test from "node:test";
import type { AgentToolCall } from "@heliopause/agent";
import { createVirtualFileSystem } from "@heliopause/sandbox";
import {
  buildAgentTools,
  executeDesktopAgentTool,
  executeSandboxAgentTool,
  SANDBOX_AGENT_TOOLS,
  WEB_FETCH_AGENT_TOOL,
  WEB_SEARCH_AGENT_TOOL,
} from "../src/agent-sandbox-tools";

test("SANDBOX_AGENT_TOOLS exposes only Phase 3 sandbox tools", () => {
  assert.deepEqual(
    SANDBOX_AGENT_TOOLS.map((tool) => tool.name),
    [
      "sandbox_list_files",
      "sandbox_read_file",
      "sandbox_write_file",
      "sandbox_command",
    ],
  );
});

test("buildAgentTools hides web_search when the desktop runtime is unavailable", () => {
  assert.deepEqual(
    buildAgentTools({ webSearchAvailable: false }).map((tool) => tool.name),
    [
      "sandbox_list_files",
      "sandbox_read_file",
      "sandbox_write_file",
      "sandbox_command",
      "web_fetch",
    ],
  );
});

test("buildAgentTools exposes web_search only when enabled and always exposes web_fetch by default", () => {
  assert.deepEqual(
    buildAgentTools({ webSearchAvailable: true }).map((tool) => tool.name),
    [
      "sandbox_list_files",
      "sandbox_read_file",
      "sandbox_write_file",
      "sandbox_command",
      "web_search",
      "web_fetch",
    ],
  );
});

test("WEB_SEARCH_AGENT_TOOL requires a query and limits max_results", () => {
  assert.equal(WEB_SEARCH_AGENT_TOOL.requiresConfirmation, true);
  assert.match(WEB_SEARCH_AGENT_TOOL.description, /recent conversation/i);
  assert.deepEqual(WEB_SEARCH_AGENT_TOOL.parametersJsonSchema, {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "Search query text.",
        minLength: 1,
        maxLength: 500,
      },
      max_results: {
        type: "integer",
        description: "Maximum number of search results to return.",
        minimum: 1,
        maximum: 5,
      },
    },
    required: ["query"],
    additionalProperties: false,
  });
});

test("WEB_FETCH_AGENT_TOOL requires a url and destination path", () => {
  assert.equal(WEB_FETCH_AGENT_TOOL.requiresConfirmation, true);
  assert.match(WEB_FETCH_AGENT_TOOL.description, /browser sandbox/i);
  assert.match(WEB_FETCH_AGENT_TOOL.description, /virtual \/workspace/i);
  assert.deepEqual(WEB_FETCH_AGENT_TOOL.parametersJsonSchema, {
    type: "object",
    properties: {
      url: {
        type: "string",
        description: "Public http:// or https:// URL to fetch. The browser must be allowed to read it by CORS.",
        minLength: 1,
        maxLength: 2048,
      },
      path: {
        type: "string",
        description: "Destination file path inside /workspace, such as /workspace/fetched/page.txt.",
        minLength: 1,
        maxLength: 512,
      },
    },
    required: ["url", "path"],
    additionalProperties: false,
  });
});

test("agent tool descriptions do not include legacy XML+JSON examples", () => {
  for (const tool of buildAgentTools({ webSearchAvailable: true })) {
    assert.equal(tool.description.includes("<tool_call>"), false);
    assert.equal(tool.description.includes("</tool_call>"), false);
  }
});

test("sandbox_list_files lists the default workspace and explicit paths", async () => {
  const fs = createVirtualFileSystem();
  fs.writeText("/workspace/notes.md", "hello\n");
  fs.writeText("/workspace/sub/todo.txt", "ship it\n");

  const defaultResult = await executeSandboxAgentTool(
    fs,
    call("sandbox_list_files", {}),
    new AbortController().signal,
  );
  assert.equal(defaultResult.ok, true);
  assert.deepEqual(
    defaultResult.content.entries.map((entry) => entry.path),
    ["/workspace/notes.md", "/workspace/sub"],
  );

  const nestedResult = await executeSandboxAgentTool(
    fs,
    call("sandbox_list_files", { path: "/workspace/sub" }),
    new AbortController().signal,
  );
  assert.equal(nestedResult.ok, true);
  assert.deepEqual(
    nestedResult.content.entries.map((entry) => entry.path),
    ["/workspace/sub/todo.txt"],
  );
});

test("sandbox_read_file returns a content envelope", async () => {
  const fs = createVirtualFileSystem();
  fs.writeText("/workspace/notes.md", "alpha\nbeta\n");

  const result = await executeSandboxAgentTool(
    fs,
    call("sandbox_read_file", { path: "notes.md" }),
    new AbortController().signal,
  );

  assert.deepEqual(result, {
    callId: "tool_1",
    ok: true,
    content: {
      kind: "sandbox_read_file",
      path: "/workspace/notes.md",
      content: "alpha\nbeta\n",
      truncated: false,
    },
  });
});

test("sandbox_write_file writes text and reports bytes", async () => {
  const fs = createVirtualFileSystem();

  const result = await executeSandboxAgentTool(
    fs,
    call("sandbox_write_file", { path: "/workspace/notes.md", content: "alpha\n" }),
    new AbortController().signal,
  );

  assert.deepEqual(result, {
    callId: "tool_1",
    ok: true,
    content: {
      kind: "sandbox_write_file",
      path: "/workspace/notes.md",
      bytesWritten: 6,
    },
  });
  assert.equal(fs.readText("/workspace/notes.md"), "alpha\n");
});

test("sandbox_command routes allowed commands through the virtual filesystem", async () => {
  const fs = createVirtualFileSystem();
  fs.writeText("/workspace/notes.md", "alpha\n");

  const result = await executeSandboxAgentTool(
    fs,
    call("sandbox_command", { args: ["ls", "/workspace"] }),
    new AbortController().signal,
  );

  assert.deepEqual(result, {
    callId: "tool_1",
    ok: true,
    content: {
      kind: "sandbox_command",
      exitCode: 0,
      stdout: "notes.md\n",
      stderr: "",
      truncated: false,
    },
  });
});

test("sandbox tool errors become failed tool results", async () => {
  const fs = createVirtualFileSystem();

  const invalidPath = await executeSandboxAgentTool(
    fs,
    call("sandbox_list_files", { path: "/etc" }),
    new AbortController().signal,
  );
  assert.equal(invalidPath.ok, false);
  assert.equal(invalidPath.error.code, "path_outside_workspace");

  const invalidArguments = await executeSandboxAgentTool(
    fs,
    call("sandbox_command", { args: "--help" }),
    new AbortController().signal,
  );
  assert.equal(invalidArguments.ok, false);
  assert.equal(invalidArguments.error.code, "invalid_arguments");
});

test("desktop executor reports web_search unavailable without a host executor", async () => {
  const fs = createVirtualFileSystem();

  const result = await executeDesktopAgentTool(
    fs,
    call("web_search", { query: "OpenAI latest model release", max_results: 5 }),
    new AbortController().signal,
  );

  assert.equal(result.ok, false);
  assert.equal(result.error.code, "web_search_unavailable");
});

test("desktop executor reports web_fetch unavailable without a host executor", async () => {
  const fs = createVirtualFileSystem();

  const result = await executeDesktopAgentTool(
    fs,
    call("web_fetch", { url: "https://example.com/", path: "/workspace/example.txt" }),
    new AbortController().signal,
  );

  assert.equal(result.ok, false);
  assert.equal(result.error.code, "web_fetch_unavailable");
});

test("desktop executor delegates web_search to the host executor", async () => {
  const fs = createVirtualFileSystem();

  const result = await executeDesktopAgentTool(
    fs,
    call("web_search", { query: "OpenAI latest model release", max_results: 5 }),
    new AbortController().signal,
    {
      executeWebSearch: async (toolCall) => ({
        callId: toolCall.id,
        ok: true,
        content: {
          kind: "web_search",
          query: "OpenAI latest model release",
          results: [{
            title: "OpenAI",
            url: "https://openai.com/",
            snippet: "OpenAI news",
          }],
        },
      }),
    },
  );

  assert.deepEqual(result, {
    callId: "tool_1",
    ok: true,
    content: {
      kind: "web_search",
      query: "OpenAI latest model release",
      results: [{
        title: "OpenAI",
        url: "https://openai.com/",
        snippet: "OpenAI news",
      }],
    },
  });
});

test("desktop executor delegates web_fetch to the host executor", async () => {
  const fs = createVirtualFileSystem();

  const result = await executeDesktopAgentTool(
    fs,
    call("web_fetch", { url: "https://example.com/", path: "/workspace/example.txt" }),
    new AbortController().signal,
    {
      executeWebFetch: async (toolCall) => ({
        callId: toolCall.id,
        ok: true,
        content: {
          kind: "web_fetch",
          url: "https://example.com/",
          finalUrl: "https://example.com/",
          path: "/workspace/example.txt",
          status: 200,
          contentType: "text/html; charset=utf-8",
          bytesWritten: 11,
          truncated: false,
          title: "Example",
        },
      }),
    },
  );

  assert.deepEqual(result, {
    callId: "tool_1",
    ok: true,
    content: {
      kind: "web_fetch",
      url: "https://example.com/",
      finalUrl: "https://example.com/",
      path: "/workspace/example.txt",
      status: 200,
      contentType: "text/html; charset=utf-8",
      bytesWritten: 11,
      truncated: false,
      title: "Example",
    },
  });
});

test("pre-aborted sandbox commands reject instead of becoming tool results", async () => {
  const fs = createVirtualFileSystem();
  const abortController = new AbortController();
  abortController.abort();

  await assert.rejects(
    () =>
      executeSandboxAgentTool(
        fs,
        call("sandbox_command", { args: ["ls", "/workspace"] }),
        abortController.signal,
      ),
    { name: "AbortError" },
  );
});

function call(name: AgentToolCall["name"], args: unknown): AgentToolCall {
  return {
    id: "tool_1",
    name,
    arguments: args,
  };
}
