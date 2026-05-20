export const SANDBOX_ROOT = "/workspace";
export const DEFAULT_MAX_FILE_BYTES = 1024 * 1024;
export const DEFAULT_MAX_TOTAL_BYTES = 16 * 1024 * 1024;
export const DEFAULT_MAX_COMMAND_OUTPUT_BYTES = 16 * 1024;
export const DEFAULT_COMMAND_TIMEOUT_MS = 2000;

export type VirtualFileEntry = {
  path: string;
  name: string;
  kind: "file" | "directory";
  sizeBytes: number;
  updatedAtMs: number;
};

export type VirtualFileStat = VirtualFileEntry & {
  createdAtMs: number;
};

export type VirtualFileSystem = {
  list(path?: string): VirtualFileEntry[];
  readText(path: string): string;
  writeText(path: string, content: string): void;
  delete(path: string): void;
  stat(path: string): VirtualFileStat;
  reset(): void;
};

export type VirtualFileSystemOptions = {
  maxFileBytes?: number;
  maxTotalBytes?: number;
};

export type SandboxCommandName =
  | "ls"
  | "cat"
  | "grep"
  | "wc"
  | "head"
  | "tail"
  | "sort"
  | "uniq";

export type SandboxCommandRequest = {
  cmd: SandboxCommandName;
  args: readonly string[];
};

export type SandboxCommandResult = {
  exitCode: number;
  stdout: string;
  stderr: string;
  truncated: boolean;
};

export type SandboxCommandOptions = {
  timeoutMs?: number;
  maxOutputBytes?: number;
  signal?: AbortSignal;
};

export type SandboxErrorCode =
  | "invalid_path"
  | "path_outside_workspace"
  | "path_not_found"
  | "path_is_directory"
  | "path_is_file"
  | "file_too_large"
  | "total_size_exceeded"
  | "unknown_command"
  | "invalid_arguments"
  | "command_timeout"
  | "aborted";

export class SandboxError extends Error {
  readonly code: SandboxErrorCode;

  constructor(code: SandboxErrorCode, message: string) {
    super(message);
    this.name = code === "aborted" ? "AbortError" : "SandboxError";
    this.code = code;
  }
}

type FileNode = {
  kind: "file";
  content: string;
  sizeBytes: number;
  createdAtMs: number;
  updatedAtMs: number;
};

type DirectoryNode = {
  kind: "directory";
  sizeBytes: 0;
  createdAtMs: number;
  updatedAtMs: number;
};

type VirtualNode = FileNode | DirectoryNode;

type CommandContext = {
  deadlineMs: number;
  signal?: AbortSignal;
  check(): void;
};

const textEncoder = new TextEncoder();

const ALLOWED_COMMANDS = new Set<string>([
  "ls",
  "cat",
  "grep",
  "wc",
  "head",
  "tail",
  "sort",
  "uniq",
]);

export function normalizeVirtualPath(path: string): string {
  if (typeof path !== "string" || path.length === 0) {
    throw new SandboxError("invalid_path", "Path must be a non-empty string.");
  }
  if (path.includes("\0")) {
    throw new SandboxError("invalid_path", "Path must not contain NUL bytes.");
  }
  if (path.includes("\\")) {
    throw new SandboxError("invalid_path", "Backslash paths are not supported.");
  }
  if (path.startsWith("~")) {
    throw new SandboxError("invalid_path", "Home shorthand is not supported.");
  }
  if (/^[A-Za-z]:(?:\/|$)/.test(path)) {
    throw new SandboxError("invalid_path", "Windows drive paths are not supported.");
  }
  if (/^[A-Za-z][A-Za-z0-9+.-]*:\/\//.test(path)) {
    throw new SandboxError("invalid_path", "URL paths are not supported.");
  }

  const absolute = path.startsWith("/");
  const rawSegments = path.split("/");
  const segments: string[] = [];

  for (const segment of rawSegments) {
    if (segment === "" || segment === ".") {
      continue;
    }
    if (segment === "..") {
      throw new SandboxError("path_outside_workspace", "Path traversal is not allowed.");
    }
    segments.push(segment);
  }

  if (absolute) {
    if (segments[0] !== "workspace") {
      throw new SandboxError("path_outside_workspace", "Path must be inside /workspace.");
    }
    segments.shift();
  }

  return segments.length === 0 ? SANDBOX_ROOT : `${SANDBOX_ROOT}/${segments.join("/")}`;
}

export function createVirtualFileSystem(options: VirtualFileSystemOptions = {}): VirtualFileSystem {
  const maxFileBytes = options.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;
  const maxTotalBytes = options.maxTotalBytes ?? DEFAULT_MAX_TOTAL_BYTES;
  const nodes = new Map<string, VirtualNode>();
  let totalBytes = 0;

  function now(): number {
    return Date.now();
  }

  function reset(): void {
    nodes.clear();
    const time = now();
    nodes.set(SANDBOX_ROOT, {
      kind: "directory",
      sizeBytes: 0,
      createdAtMs: time,
      updatedAtMs: time,
    });
    totalBytes = 0;
  }

  function getNode(path: string): VirtualNode {
    const normalized = normalizeVirtualPath(path);
    const node = nodes.get(normalized);
    if (!node) {
      throw new SandboxError("path_not_found", `Path not found: ${normalized}`);
    }
    return node;
  }

  function toEntry(path: string, node: VirtualNode): VirtualFileEntry {
    return {
      path,
      name: path === SANDBOX_ROOT ? "workspace" : path.slice(path.lastIndexOf("/") + 1),
      kind: node.kind,
      sizeBytes: node.sizeBytes,
      updatedAtMs: node.updatedAtMs,
    };
  }

  function touchDirectory(path: string, time: number): void {
    const node = nodes.get(path);
    if (node?.kind === "directory") {
      node.updatedAtMs = time;
    }
  }

  function ensureParentDirectories(path: string, time: number): void {
    const parts = path.slice(SANDBOX_ROOT.length + 1).split("/");
    parts.pop();
    let current = SANDBOX_ROOT;

    for (const part of parts) {
      current = `${current}/${part}`;
      const existing = nodes.get(current);
      if (existing?.kind === "file") {
        throw new SandboxError("path_is_file", `Parent path is a file: ${current}`);
      }
      if (!existing) {
        nodes.set(current, {
          kind: "directory",
          sizeBytes: 0,
          createdAtMs: time,
          updatedAtMs: time,
        });
      }
    }
  }

  function list(path = SANDBOX_ROOT): VirtualFileEntry[] {
    const normalized = normalizeVirtualPath(path);
    const node = getNode(normalized);
    if (node.kind === "file") {
      return [toEntry(normalized, node)];
    }

    const prefix = normalized === SANDBOX_ROOT ? `${SANDBOX_ROOT}/` : `${normalized}/`;
    const entries: VirtualFileEntry[] = [];

    for (const [entryPath, entryNode] of nodes) {
      if (entryPath === normalized || !entryPath.startsWith(prefix)) {
        continue;
      }
      const remainder = entryPath.slice(prefix.length);
      if (!remainder.includes("/")) {
        entries.push(toEntry(entryPath, entryNode));
      }
    }

    return entries.sort((a, b) => a.name.localeCompare(b.name));
  }

  function readText(path: string): string {
    const normalized = normalizeVirtualPath(path);
    const node = getNode(normalized);
    if (node.kind === "directory") {
      throw new SandboxError("path_is_directory", `Cannot read directory: ${normalized}`);
    }
    return node.content;
  }

  function writeText(path: string, content: string): void {
    if (typeof content !== "string") {
      throw new SandboxError("invalid_arguments", "File content must be a string.");
    }

    const normalized = normalizeVirtualPath(path);
    if (normalized === SANDBOX_ROOT) {
      throw new SandboxError("path_is_directory", "Cannot write to /workspace.");
    }

    const sizeBytes = byteLength(content);
    if (sizeBytes > maxFileBytes) {
      throw new SandboxError("file_too_large", `File exceeds ${maxFileBytes} bytes.`);
    }

    const existing = nodes.get(normalized);
    if (existing?.kind === "directory") {
      throw new SandboxError("path_is_directory", `Cannot overwrite directory: ${normalized}`);
    }

    const previousSize = existing?.kind === "file" ? existing.sizeBytes : 0;
    const nextTotalBytes = totalBytes - previousSize + sizeBytes;
    if (nextTotalBytes > maxTotalBytes) {
      throw new SandboxError("total_size_exceeded", `Virtual filesystem exceeds ${maxTotalBytes} bytes.`);
    }

    const time = now();
    ensureParentDirectories(normalized, time);
    nodes.set(normalized, {
      kind: "file",
      content,
      sizeBytes,
      createdAtMs: existing?.createdAtMs ?? time,
      updatedAtMs: time,
    });
    totalBytes = nextTotalBytes;
    touchDirectory(parentPath(normalized), time);
  }

  function deletePath(path: string): void {
    const normalized = normalizeVirtualPath(path);
    if (normalized === SANDBOX_ROOT) {
      reset();
      return;
    }

    const node = getNode(normalized);
    const prefix = `${normalized}/`;
    let removedBytes = node.kind === "file" ? node.sizeBytes : 0;

    for (const [entryPath, entryNode] of [...nodes.entries()]) {
      if (entryPath.startsWith(prefix)) {
        removedBytes += entryNode.kind === "file" ? entryNode.sizeBytes : 0;
        nodes.delete(entryPath);
      }
    }

    nodes.delete(normalized);
    totalBytes -= removedBytes;
    touchDirectory(parentPath(normalized), now());
  }

  function stat(path: string): VirtualFileStat {
    const normalized = normalizeVirtualPath(path);
    const node = getNode(normalized);
    return {
      ...toEntry(normalized, node),
      createdAtMs: node.createdAtMs,
    };
  }

  reset();

  return {
    list,
    readText,
    writeText,
    delete: deletePath,
    stat,
    reset,
  };
}

export async function runSandboxCommand(
  fs: VirtualFileSystem,
  request: SandboxCommandRequest,
  options: SandboxCommandOptions = {},
): Promise<SandboxCommandResult> {
  const maxOutputBytes = options.maxOutputBytes ?? DEFAULT_MAX_COMMAND_OUTPUT_BYTES;
  const timeoutMs = options.timeoutMs ?? DEFAULT_COMMAND_TIMEOUT_MS;
  const context = createCommandContext(timeoutMs, options.signal);

  try {
    context.check();
    // Tool calls cross package/runtime boundaries, so keep a runtime allowlist
    // even though SandboxCommandRequest narrows this at TypeScript compile time.
    const cmd = (request as { cmd: string }).cmd;
    const args = Array.isArray(request.args) ? request.args : [];

    if (!ALLOWED_COMMANDS.has(cmd)) {
      return limitCommandOutput({
        exitCode: 127,
        stdout: "",
        stderr: `unknown command: ${cmd}\n`,
        truncated: false,
      }, maxOutputBytes);
    }

    const result = runAllowedCommand(fs, cmd as SandboxCommandName, args, context);
    context.check();
    return limitCommandOutput(result, maxOutputBytes);
  } catch (error) {
    if (error instanceof SandboxError && error.code === "aborted") {
      throw error;
    }
    if (error instanceof SandboxError && error.code === "command_timeout") {
      return limitCommandOutput({
        exitCode: 124,
        stdout: "",
        stderr: "command timed out\n",
        truncated: false,
      }, maxOutputBytes);
    }
    if (error instanceof SandboxError && error.code === "invalid_arguments") {
      return limitCommandOutput({
        exitCode: 2,
        stdout: "",
        stderr: `${error.message}\n`,
        truncated: false,
      }, maxOutputBytes);
    }
    if (error instanceof SandboxError) {
      return limitCommandOutput({
        exitCode: 1,
        stdout: "",
        stderr: `${error.message}\n`,
        truncated: false,
      }, maxOutputBytes);
    }
    throw error;
  }
}

function runAllowedCommand(
  fs: VirtualFileSystem,
  cmd: SandboxCommandName,
  args: readonly string[],
  context: CommandContext,
): SandboxCommandResult {
  switch (cmd) {
    case "ls":
      return commandLs(fs, args, context);
    case "cat":
      return commandCat(fs, args, context);
    case "grep":
      return commandGrep(fs, args, context);
    case "wc":
      return commandWc(fs, args, context);
    case "head":
      return commandHeadTail(fs, args, context, "head");
    case "tail":
      return commandHeadTail(fs, args, context, "tail");
    case "sort":
      return commandSort(fs, args, context);
    case "uniq":
      return commandUniq(fs, args, context);
  }
}

function commandLs(fs: VirtualFileSystem, args: readonly string[], context: CommandContext): SandboxCommandResult {
  let long = false;
  let onePerLine = false;
  let all = false;
  const paths: string[] = [];

  for (const arg of args) {
    context.check();
    if (arg === "-l") {
      long = true;
    } else if (arg === "-1") {
      onePerLine = true;
    } else if (arg === "-a") {
      all = true;
    } else if (arg.startsWith("-")) {
      throw new SandboxError("invalid_arguments", `ls: unsupported option: ${arg}`);
    } else {
      paths.push(arg);
    }
  }

  if (paths.length > 1) {
    throw new SandboxError("invalid_arguments", "ls: expected at most one path");
  }

  const target = paths[0] ?? SANDBOX_ROOT;
  const entries = fs.list(target).filter((entry) => all || !entry.name.startsWith("."));
  const lines = entries.map((entry) => {
    if (long) {
      return `${entry.kind === "directory" ? "d" : "-"} ${entry.sizeBytes} ${entry.path}`;
    }
    return entry.name;
  });

  return ok(lines.length === 0 ? "" : `${lines.join(onePerLine || long ? "\n" : "  ")}\n`);
}

function commandCat(fs: VirtualFileSystem, args: readonly string[], context: CommandContext): SandboxCommandResult {
  if (args.length === 0) {
    throw new SandboxError("invalid_arguments", "cat: expected at least one file");
  }

  let stdout = "";
  for (const path of args) {
    context.check();
    rejectOption(path, "cat");
    stdout += fs.readText(path);
  }
  return ok(stdout);
}

function commandGrep(fs: VirtualFileSystem, args: readonly string[], context: CommandContext): SandboxCommandResult {
  let lineNumbers = false;
  let ignoreCase = false;
  let countOnly = false;
  const rest: string[] = [];

  for (const arg of args) {
    context.check();
    if (arg === "-n") {
      lineNumbers = true;
    } else if (arg === "-i") {
      ignoreCase = true;
    } else if (arg === "-c") {
      countOnly = true;
    } else if (arg.startsWith("-")) {
      throw new SandboxError("invalid_arguments", `grep: unsupported option: ${arg}`);
    } else {
      rest.push(arg);
    }
  }

  if (rest.length < 2) {
    throw new SandboxError("invalid_arguments", "grep: expected a pattern and at least one file");
  }

  const [pattern, ...paths] = rest;
  const needle = ignoreCase ? pattern.toLowerCase() : pattern;
  const output: string[] = [];
  let totalMatches = 0;

  for (const path of paths) {
    const content = fs.readText(path);
    const lines = splitLines(content);
    let fileMatches = 0;

    for (let index = 0; index < lines.length; index += 1) {
      context.check();
      const line = lines[index];
      const haystack = ignoreCase ? line.toLowerCase() : line;
      if (!haystack.includes(needle)) {
        continue;
      }
      fileMatches += 1;
      totalMatches += 1;
      if (!countOnly) {
        const prefix = paths.length > 1 ? `${normalizeVirtualPath(path)}:` : "";
        const numberPrefix = lineNumbers ? `${index + 1}:` : "";
        output.push(`${prefix}${numberPrefix}${line}`);
      }
    }

    if (countOnly) {
      output.push(paths.length > 1 ? `${normalizeVirtualPath(path)}:${fileMatches}` : `${fileMatches}`);
    }
  }

  return {
    exitCode: totalMatches > 0 ? 0 : 1,
    stdout: output.length === 0 ? "" : `${output.join("\n")}\n`,
    stderr: "",
    truncated: false,
  };
}

function commandWc(fs: VirtualFileSystem, args: readonly string[], context: CommandContext): SandboxCommandResult {
  const selected = new Set<"lines" | "words" | "bytes">();
  const paths: string[] = [];

  for (const arg of args) {
    context.check();
    if (arg === "-l") {
      selected.add("lines");
    } else if (arg === "-w") {
      selected.add("words");
    } else if (arg === "-c") {
      selected.add("bytes");
    } else if (arg.startsWith("-")) {
      throw new SandboxError("invalid_arguments", `wc: unsupported option: ${arg}`);
    } else {
      paths.push(arg);
    }
  }

  if (paths.length === 0) {
    throw new SandboxError("invalid_arguments", "wc: expected at least one file");
  }
  if (selected.size === 0) {
    selected.add("lines");
    selected.add("words");
    selected.add("bytes");
  }

  const lines = paths.map((path) => {
    context.check();
    const content = fs.readText(path);
    const counts = {
      lines: splitLines(content).length,
      words: countWords(content),
      bytes: byteLength(content),
    };
    const fields = [...selected].map((key) => String(counts[key]));
    return `${fields.join(" ")} ${normalizeVirtualPath(path)}`;
  });

  return ok(`${lines.join("\n")}\n`);
}

function commandHeadTail(
  fs: VirtualFileSystem,
  args: readonly string[],
  context: CommandContext,
  mode: "head" | "tail",
): SandboxCommandResult {
  let count = 10;
  const paths: string[] = [];

  for (let index = 0; index < args.length; index += 1) {
    context.check();
    const arg = args[index];
    if (arg === "-n") {
      const value = args[index + 1];
      if (!value) {
        throw new SandboxError("invalid_arguments", `${mode}: -n requires a count`);
      }
      count = parseLineCount(value, mode);
      index += 1;
    } else if (/^-\d+$/.test(arg)) {
      count = parseLineCount(arg.slice(1), mode);
    } else if (arg.startsWith("-")) {
      throw new SandboxError("invalid_arguments", `${mode}: unsupported option: ${arg}`);
    } else {
      paths.push(arg);
    }
  }

  if (paths.length !== 1) {
    throw new SandboxError("invalid_arguments", `${mode}: expected exactly one file`);
  }

  const lines = splitLines(fs.readText(paths[0]));
  const selected = mode === "head" ? lines.slice(0, count) : lines.slice(Math.max(0, lines.length - count));
  return ok(selected.length === 0 ? "" : `${selected.join("\n")}\n`);
}

function commandSort(fs: VirtualFileSystem, args: readonly string[], context: CommandContext): SandboxCommandResult {
  let reverse = false;
  let unique = false;
  const paths: string[] = [];

  for (const arg of args) {
    context.check();
    if (arg === "-r") {
      reverse = true;
    } else if (arg === "-u") {
      unique = true;
    } else if (arg.startsWith("-")) {
      throw new SandboxError("invalid_arguments", `sort: unsupported option: ${arg}`);
    } else {
      paths.push(arg);
    }
  }

  if (paths.length !== 1) {
    throw new SandboxError("invalid_arguments", "sort: expected exactly one file");
  }

  let lines = splitLines(fs.readText(paths[0])).sort(compareSandboxLines);
  if (unique) {
    lines = [...new Set(lines)];
  }
  if (reverse) {
    lines.reverse();
  }

  return ok(lines.length === 0 ? "" : `${lines.join("\n")}\n`);
}

function commandUniq(fs: VirtualFileSystem, args: readonly string[], context: CommandContext): SandboxCommandResult {
  let count = false;
  const paths: string[] = [];

  for (const arg of args) {
    context.check();
    if (arg === "-c") {
      count = true;
    } else if (arg.startsWith("-")) {
      throw new SandboxError("invalid_arguments", `uniq: unsupported option: ${arg}`);
    } else {
      paths.push(arg);
    }
  }

  if (paths.length !== 1) {
    throw new SandboxError("invalid_arguments", "uniq: expected exactly one file");
  }

  const input = splitLines(fs.readText(paths[0]));
  const output: string[] = [];
  let previous: string | undefined;
  let runLength = 0;

  function flush(): void {
    if (previous === undefined) {
      return;
    }
    output.push(count ? `${runLength} ${previous}` : previous);
  }

  for (const line of input) {
    context.check();
    if (line === previous) {
      runLength += 1;
      continue;
    }
    flush();
    previous = line;
    runLength = 1;
  }
  flush();

  return ok(output.length === 0 ? "" : `${output.join("\n")}\n`);
}

function ok(stdout: string): SandboxCommandResult {
  return {
    exitCode: 0,
    stdout,
    stderr: "",
    truncated: false,
  };
}

function rejectOption(arg: string, command: string): void {
  if (arg.startsWith("-")) {
    throw new SandboxError("invalid_arguments", `${command}: unsupported option: ${arg}`);
  }
}

function parseLineCount(value: string, command: string): number {
  if (!/^\d+$/.test(value)) {
    throw new SandboxError("invalid_arguments", `${command}: invalid line count: ${value}`);
  }
  return Number(value);
}

function createCommandContext(timeoutMs: number, signal?: AbortSignal): CommandContext {
  const startMs = Date.now();
  const deadlineMs = timeoutMs <= 0 ? startMs - 1 : startMs + timeoutMs;

  return {
    deadlineMs,
    signal,
    check() {
      if (signal?.aborted) {
        throw new SandboxError("aborted", "Command was aborted.");
      }
      if (Date.now() > deadlineMs) {
        throw new SandboxError("command_timeout", "Command timed out.");
      }
    },
  };
}

function limitCommandOutput(result: SandboxCommandResult, maxOutputBytes: number): SandboxCommandResult {
  const stdout = truncateUtf8(result.stdout, maxOutputBytes);
  const stderr = truncateUtf8(result.stderr, maxOutputBytes);
  return {
    ...result,
    stdout: stdout.text,
    stderr: stderr.text,
    truncated: result.truncated || stdout.truncated || stderr.truncated,
  };
}

function truncateUtf8(text: string, maxBytes: number): { text: string; truncated: boolean } {
  if (maxBytes < 0) {
    throw new SandboxError("invalid_arguments", "maxOutputBytes must be non-negative.");
  }
  const bytes = textEncoder.encode(text);
  if (bytes.byteLength <= maxBytes) {
    return { text, truncated: false };
  }
  let usedBytes = 0;
  let output = "";
  for (const char of text) {
    const charBytes = byteLength(char);
    if (usedBytes + charBytes > maxBytes) {
      break;
    }
    output += char;
    usedBytes += charBytes;
  }
  return { text: output, truncated: true };
}

function compareSandboxLines(a: string, b: string): number {
  const lowerA = a.toLowerCase();
  const lowerB = b.toLowerCase();
  if (lowerA < lowerB) {
    return -1;
  }
  if (lowerA > lowerB) {
    return 1;
  }
  if (a < b) {
    return -1;
  }
  if (a > b) {
    return 1;
  }
  return 0;
}

function byteLength(text: string): number {
  return textEncoder.encode(text).byteLength;
}

function parentPath(path: string): string {
  const lastSlash = path.lastIndexOf("/");
  return lastSlash <= SANDBOX_ROOT.length ? SANDBOX_ROOT : path.slice(0, lastSlash);
}

function splitLines(content: string): string[] {
  if (content.length === 0) {
    return [];
  }
  const lines = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
  if (lines.at(-1) === "") {
    lines.pop();
  }
  return lines;
}

function countWords(content: string): number {
  const trimmed = content.trim();
  return trimmed.length === 0 ? 0 : trimmed.split(/\s+/).length;
}
