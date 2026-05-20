import assert from "node:assert/strict";
import test from "node:test";
import type { AgentToolCall } from "@heliopause/agent";
import { createVirtualFileSystem } from "@heliopause/sandbox";
import {
  executeSandboxAgentTool,
  SANDBOX_AGENT_TOOLS,
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
    call("sandbox_command", { cmd: "ls", args: ["/workspace"] }),
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
    call("sandbox_command", { cmd: "ls", args: "--help" }),
    new AbortController().signal,
  );
  assert.equal(invalidArguments.ok, false);
  assert.equal(invalidArguments.error.code, "invalid_arguments");
});

test("pre-aborted sandbox commands reject instead of becoming tool results", async () => {
  const fs = createVirtualFileSystem();
  const abortController = new AbortController();
  abortController.abort();

  await assert.rejects(
    () =>
      executeSandboxAgentTool(
        fs,
        call("sandbox_command", { cmd: "ls", args: ["/workspace"] }),
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
