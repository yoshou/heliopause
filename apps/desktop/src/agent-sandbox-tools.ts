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
      "List files and directories under the virtual /workspace filesystem.",
    parametersJsonSchema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Directory path to list. Defaults to /workspace when omitted.",
        },
      },
      additionalProperties: false,
    },
  },
  {
    name: "sandbox_read_file",
    description:
      "Read the full UTF-8 text content of a file from the virtual /workspace filesystem.",
    parametersJsonSchema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "File path to read, relative to /workspace or an absolute /workspace path.",
        },
      },
      required: ["path"],
      additionalProperties: false,
    },
  },
  {
    name: "sandbox_write_file",
    description:
      "Write UTF-8 text content to a file in the virtual /workspace filesystem, creating parent directories as needed.",
    parametersJsonSchema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Destination file path, relative to /workspace or an absolute /workspace path.",
        },
        content: {
          type: "string",
          description: "Complete file content to write.",
        },
      },
      required: ["path", "content"],
      additionalProperties: false,
    },
  },
  {
    name: "sandbox_command",
    description:
      "Run one allowed virtual sandbox command against the /workspace filesystem. Use structured arguments only, never a shell string.",
    parametersJsonSchema: {
      type: "object",
      properties: {
        cmd: {
          type: "string",
          enum: SANDBOX_COMMAND_NAMES,
          description: "Allowed command name to execute.",
        },
        args: {
          type: "array",
          description: "Command arguments, such as paths or flags, as separate strings.",
          items: { type: "string" },
        },
      },
      required: ["cmd", "args"],
      additionalProperties: false,
    },
  },
];

export const WEB_SEARCH_AGENT_TOOL: AgentToolDefinition = {
  name: "web_search",
  description:
    "Request a public web search with Tavily. Use the best query from the current request or recent conversation. Results include title, url, and snippet only; raw page content, images, cookies, credentials, and private URLs are not available.",
  parametersJsonSchema: {
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
  },
  requiresConfirmation: true,
};

export function buildAgentTools(
  options: { webSearchAvailable: boolean },
): readonly AgentToolDefinition[] {
  return options.webSearchAvailable
    ? [...SANDBOX_AGENT_TOOLS, WEB_SEARCH_AGENT_TOOL]
    : SANDBOX_AGENT_TOOLS;
}

export type WebSearchToolExecutor = (
  call: AgentToolCall,
  signal: AbortSignal,
) => Promise<AgentToolResult>;

export async function executeDesktopAgentTool(
  fs: VirtualFileSystem,
  call: AgentToolCall,
  signal: AbortSignal,
  options?: {
    executeWebSearch?: WebSearchToolExecutor;
  },
): Promise<AgentToolResult> {
  if (call.name === "web_search") {
    if (!options?.executeWebSearch) {
      return toolError(call.id, {
        code: "web_search_unavailable",
        message: "web_search is not available in this runtime.",
      });
    }
    return options.executeWebSearch(call, signal);
  }

  return executeSandboxAgentTool(fs, call, signal);
}

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
          code: "web_search_unavailable",
          message: "web_search is not available in the sandbox executor.",
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
