import assert from "node:assert/strict";
import test from "node:test";

import {
  SANDBOX_ROOT,
  SandboxError,
  createVirtualFileSystem,
  normalizeVirtualPath,
  runSandboxCommand,
  type SandboxCommandRequest,
} from "../src/index.ts";

test("normalizeVirtualPath accepts workspace paths and relative paths", () => {
  assert.equal(normalizeVirtualPath("/workspace"), SANDBOX_ROOT);
  assert.equal(normalizeVirtualPath("/workspace//notes/./todo.md"), "/workspace/notes/todo.md");
  assert.equal(normalizeVirtualPath("notes/todo.md"), "/workspace/notes/todo.md");
  assert.equal(normalizeVirtualPath("./notes/todo.md"), "/workspace/notes/todo.md");
});

test("normalizeVirtualPath rejects paths outside the workspace", () => {
  const rejected = [
    "",
    "..",
    "notes/../todo.md",
    "/tmp/file.txt",
    "/workspace2/file.txt",
    "C:/Users/me/file.txt",
    "file:///workspace/file.txt",
    "~/file.txt",
    "notes\\todo.md",
    "bad\0path",
  ];

  for (const path of rejected) {
    assert.throws(() => normalizeVirtualPath(path), SandboxError, path);
  }
});

test("virtual filesystem reads, writes, lists, stats, deletes, and resets", () => {
  const fs = createVirtualFileSystem();
  fs.writeText("docs/readme.md", "hello");
  fs.writeText("/workspace/docs/todo.md", "todo");

  assert.equal(fs.readText("docs/readme.md"), "hello");
  assert.deepEqual(fs.list("/workspace").map((entry) => [entry.kind, entry.path]), [
    ["directory", "/workspace/docs"],
  ]);
  assert.deepEqual(fs.list("/workspace/docs").map((entry) => entry.name), ["readme.md", "todo.md"]);
  assert.equal(fs.stat("docs/readme.md").sizeBytes, 5);

  fs.delete("docs/readme.md");
  assert.throws(() => fs.readText("docs/readme.md"), /Path not found/);

  fs.reset();
  assert.deepEqual(fs.list("/workspace"), []);
});

test("virtual filesystem enforces file and total size limits", () => {
  const fileLimited = createVirtualFileSystem({ maxFileBytes: 3 });
  assert.throws(() => fileLimited.writeText("a.txt", "abcd"), /File exceeds/);

  const totalLimited = createVirtualFileSystem({ maxTotalBytes: 5 });
  totalLimited.writeText("a.txt", "abc");
  assert.throws(() => totalLimited.writeText("b.txt", "def"), /Virtual filesystem exceeds/);

  totalLimited.writeText("a.txt", "a");
  totalLimited.writeText("b.txt", "def");
  assert.equal(totalLimited.readText("b.txt"), "def");
});

test("virtual filesystem rejects invalid file and directory operations", () => {
  const fs = createVirtualFileSystem();
  fs.writeText("dir/file.txt", "content");

  assert.throws(() => fs.readText("dir"), /Cannot read directory/);
  assert.throws(() => fs.writeText("dir", "replacement"), /Cannot overwrite directory/);
  assert.throws(() => fs.writeText("dir/file.txt/child.txt", "bad"), /Parent path is a file/);
  assert.throws(() => fs.stat("missing.txt"), /Path not found/);
});

test("sandbox commands implement basic text utilities", async () => {
  const fs = createVirtualFileSystem();
  fs.writeText("notes.txt", "beta\nalpha\nalpha\nGamma\n");

  assert.deepEqual(await runSandboxCommand(fs, { cmd: "ls", args: ["-1", "/workspace"] }), {
    exitCode: 0,
    stdout: "notes.txt\n",
    stderr: "",
    truncated: false,
  });
  assert.equal((await runSandboxCommand(fs, { cmd: "cat", args: ["notes.txt"] })).stdout, "beta\nalpha\nalpha\nGamma\n");
  assert.equal((await runSandboxCommand(fs, { cmd: "grep", args: ["-n", "-i", "gamma", "notes.txt"] })).stdout, "4:Gamma\n");
  assert.equal((await runSandboxCommand(fs, { cmd: "grep", args: ["-c", "alpha", "notes.txt"] })).stdout, "2\n");
  assert.equal((await runSandboxCommand(fs, { cmd: "wc", args: ["notes.txt"] })).stdout, "4 4 23 /workspace/notes.txt\n");
  assert.equal((await runSandboxCommand(fs, { cmd: "head", args: ["-n", "2", "notes.txt"] })).stdout, "beta\nalpha\n");
  assert.equal((await runSandboxCommand(fs, { cmd: "tail", args: ["-2", "notes.txt"] })).stdout, "alpha\nGamma\n");
  assert.equal((await runSandboxCommand(fs, { cmd: "sort", args: ["notes.txt"] })).stdout, "alpha\nalpha\nbeta\nGamma\n");
  assert.equal((await runSandboxCommand(fs, { cmd: "sort", args: ["-r", "-u", "notes.txt"] })).stdout, "Gamma\nbeta\nalpha\n");
  assert.equal((await runSandboxCommand(fs, { cmd: "uniq", args: ["-c", "notes.txt"] })).stdout, "1 beta\n2 alpha\n1 Gamma\n");
});

test("sandbox commands report invalid commands, arguments, and paths", async () => {
  const fs = createVirtualFileSystem();
  fs.writeText("a.txt", "a");

  const unknown = await runSandboxCommand(fs, { cmd: "rm", args: ["-rf", "/"] } as unknown as SandboxCommandRequest);
  assert.equal(unknown.exitCode, 127);
  assert.match(unknown.stderr, /unknown command/);

  const invalidArgs = await runSandboxCommand(fs, { cmd: "grep", args: ["pattern"] });
  assert.equal(invalidArgs.exitCode, 2);
  assert.match(invalidArgs.stderr, /expected a pattern/);

  const missing = await runSandboxCommand(fs, { cmd: "cat", args: ["missing.txt"] });
  assert.equal(missing.exitCode, 1);
  assert.match(missing.stderr, /Path not found/);

  const directoryRead = await runSandboxCommand(fs, { cmd: "cat", args: ["/workspace"] });
  assert.equal(directoryRead.exitCode, 1);
  assert.match(directoryRead.stderr, /Cannot read directory/);
});

test("sandbox commands do not interpret shell metacharacters", async () => {
  const fs = createVirtualFileSystem();
  fs.writeText("safe.txt", "safe");

  const result = await runSandboxCommand(fs, { cmd: "cat", args: ["/workspace/a;rm -rf /"] });
  assert.equal(result.exitCode, 1);
  assert.match(result.stderr, /Path not found: \/workspace\/a;rm -rf/);
  assert.equal(fs.readText("safe.txt"), "safe");
});

test("sandbox commands truncate stdout and stderr independently", async () => {
  const fs = createVirtualFileSystem();
  fs.writeText("long.txt", "abcdef");
  fs.writeText("japanese.txt", "あいう");

  const stdout = await runSandboxCommand(fs, { cmd: "cat", args: ["long.txt"] }, { maxOutputBytes: 3 });
  assert.deepEqual(stdout, {
    exitCode: 0,
    stdout: "abc",
    stderr: "",
    truncated: true,
  });

  const multibyte = await runSandboxCommand(fs, { cmd: "cat", args: ["japanese.txt"] }, { maxOutputBytes: 4 });
  assert.equal(multibyte.stdout, "あ");
  assert.equal(multibyte.truncated, true);
  assert.equal(multibyte.stdout.includes("\uFFFD"), false);

  const stderr = await runSandboxCommand(fs, { cmd: "cat", args: ["missing.txt"] }, { maxOutputBytes: 4 });
  assert.equal(stderr.exitCode, 1);
  assert.equal(stderr.stderr, "Path");
  assert.equal(stderr.truncated, true);
});

test("sandbox commands time out and respect pre-aborted signals", async () => {
  const fs = createVirtualFileSystem();
  fs.writeText("a.txt", "a");

  const timedOut = await runSandboxCommand(fs, { cmd: "cat", args: ["a.txt"] }, { timeoutMs: 0 });
  assert.equal(timedOut.exitCode, 124);
  assert.equal(timedOut.stderr, "command timed out\n");

  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    runSandboxCommand(fs, { cmd: "cat", args: ["a.txt"] }, { signal: controller.signal }),
    (error) => error instanceof SandboxError && error.code === "aborted" && error.name === "AbortError",
  );
});
