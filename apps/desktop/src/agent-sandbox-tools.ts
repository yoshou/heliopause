import type {
  AgentToolCall,
  AgentToolDefinition,
  AgentToolError,
  AgentToolResult,
} from "@heliopause/agent";
import {
  normalizeVirtualPath,
  runSandboxCommand,
  SANDBOX_ROOT,
  SandboxError,
  type SandboxCommandName,
  type SandboxCommandRequest,
  type VirtualFileSystem,
} from "@heliopause/sandbox";

const SANDBOX_COMMAND_NAMES = [
  "ls",
  "cat",
  "grep",
  "wc",
  "head",
  "tail",
  "sort",
  "uniq",
] as const satisfies readonly SandboxCommandName[];

export const SANDBOX_AGENT_TOOLS: readonly AgentToolDefinition[] = [
  {
    name: "sandbox_list_files",
    description:
      'List files under /workspace. Example: <tool_call>{"tool":"sandbox_list_files","arguments":{"path":"/workspace"}}</tool_call>',
    parametersJsonSchema: {
      type: "object",
      properties: {
        path: { type: "string" },
      },
      additionalProperties: false,
    },
  },
  {
    name: "sandbox_read_file",
    description:
      'Read a text file from the virtual /workspace filesystem. Example: <tool_call>{"tool":"sandbox_read_file","arguments":{"path":"/workspace/notes.md"}}</tool_call>',
    parametersJsonSchema: {
      type: "object",
      properties: {
        path: { type: "string" },
      },
      required: ["path"],
      additionalProperties: false,
    },
  },
  {
    name: "sandbox_write_file",
    description:
      'Write a text file into the virtual /workspace filesystem. Example: <tool_call>{"tool":"sandbox_write_file","arguments":{"path":"/workspace/notes.md","content":"hello\\n"}}</tool_call>',
    parametersJsonSchema: {
      type: "object",
      properties: {
        path: { type: "string" },
        content: { type: "string" },
      },
      required: ["path", "content"],
      additionalProperties: false,
    },
  },
  {
    name: "sandbox_command",
    description:
      'Run an allowed virtual sandbox command. Use structured arguments only, never a shell string. Example: <tool_call>{"tool":"sandbox_command","arguments":{"cmd":"ls","args":["/workspace"]}}</tool_call>',
    parametersJsonSchema: {
      type: "object",
      properties: {
        cmd: { type: "string", enum: SANDBOX_COMMAND_NAMES },
        args: {
          type: "array",
          items: { type: "string" },
        },
      },
      required: ["cmd", "args"],
      additionalProperties: false,
    },
  },
];

export async function executeSandboxAgentTool(
  fs: VirtualFileSystem,
  call: AgentToolCall,
  signal: AbortSignal,
): Promise<AgentToolResult> {
  throwIfAborted(signal);

  try {
    switch (call.name) {
      case "sandbox_list_files": {
        const args = requireObjectArguments(call);
        const path = typeof args.path === "string" ? args.path : SANDBOX_ROOT;
        return {
          callId: call.id,
          ok: true,
          content: {
            kind: "sandbox_list_files",
            entries: fs.list(path),
          },
        };
      }

      case "sandbox_read_file": {
        const args = requireObjectArguments(call);
        const path = requireString(args.path, "path");
        const normalizedPath = normalizeVirtualPath(path);
        return {
          callId: call.id,
          ok: true,
          content: {
            kind: "sandbox_read_file",
            path: normalizedPath,
            content: fs.readText(normalizedPath),
            truncated: false,
          },
        };
      }

      case "sandbox_write_file": {
        const args = requireObjectArguments(call);
        const path = requireString(args.path, "path");
        const content = requireString(args.content, "content");
        const normalizedPath = normalizeVirtualPath(path);
        fs.writeText(normalizedPath, content);
        return {
          callId: call.id,
          ok: true,
          content: {
            kind: "sandbox_write_file",
            path: normalizedPath,
            bytesWritten: new TextEncoder().encode(content).byteLength,
          },
        };
      }

      case "sandbox_command": {
        const request = requireSandboxCommandRequest(call.arguments);
        const result = await runSandboxCommand(fs, request, { signal });
        return {
          callId: call.id,
          ok: true,
          content: {
            kind: "sandbox_command",
            ...result,
          },
        };
      }

      case "web_search":
        return toolError(call.id, {
          code: "unknown_tool",
          message: "web_search is not available in Phase 3.",
        });
    }
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    return toolError(call.id, normalizeToolError(error));
  }
}

function requireSandboxCommandRequest(value: unknown): SandboxCommandRequest {
  const args = requireObject(value);
  const cmd = requireString(args.cmd, "cmd");
  if (!isSandboxCommandName(cmd)) {
    throw new SandboxError("unknown_command", `unknown command: ${cmd}`);
  }
  if (!Array.isArray(args.args) || !args.args.every((item) => typeof item === "string")) {
    throw new SandboxError("invalid_arguments", "args must be an array of strings.");
  }
  return {
    cmd,
    args: args.args,
  };
}

function requireObjectArguments(call: AgentToolCall): Record<string, unknown> {
  return requireObject(call.arguments);
}

function requireObject(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new SandboxError("invalid_arguments", "Tool arguments must be an object.");
  }
  return value as Record<string, unknown>;
}

function requireString(value: unknown, name: string): string {
  if (typeof value !== "string") {
    throw new SandboxError("invalid_arguments", `${name} must be a string.`);
  }
  return value;
}

function isSandboxCommandName(value: string): value is SandboxCommandName {
  return (SANDBOX_COMMAND_NAMES as readonly string[]).includes(value);
}

function normalizeToolError(error: unknown): AgentToolError {
  if (error instanceof SandboxError) {
    return {
      code: error.code,
      message: error.message,
    };
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

function toolError(callId: string, error: AgentToolError): AgentToolResult {
  return {
    callId,
    ok: false,
    error,
  };
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) {
    throw new DOMException("Tool execution was aborted.", "AbortError");
  }
}

function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException && error.name === "AbortError") ||
    (error instanceof SandboxError && error.code === "aborted")
  );
}
