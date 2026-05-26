import assert from "node:assert/strict";
import test from "node:test";

import {
  fetchBrowserTextResource,
  isSupportedWebFetchContentType,
  normalizeFetchedText,
  validateWebFetchUrl,
} from "../src/web-fetch";

test("validateWebFetchUrl rejects non-http, credential, localhost, and private hosts", () => {
  for (const url of [
    "file:///tmp/a.txt",
    "ftp://example.com/file.txt",
    "https://user:pass@example.com/",
    "http://localhost/",
    "http://127.0.0.1/",
    "http://10.0.0.1/",
    "http://172.16.0.1/",
    "http://192.168.0.1/",
    "http://169.254.1.1/",
    "http://[::1]/",
    "http://[fd00::1]/",
    "http://[fe80::1]/",
  ]) {
    assert.throws(
      () => validateWebFetchUrl(url),
      (error) =>
        isErrorCode(error, "invalid_url") ||
        isErrorCode(error, "blocked_url"),
      url,
    );
  }
  assert.equal(validateWebFetchUrl("https://example.com/a.txt").href, "https://example.com/a.txt");
});

test("isSupportedWebFetchContentType allows only text resources", () => {
  assert.equal(isSupportedWebFetchContentType("text/html; charset=utf-8"), true);
  assert.equal(isSupportedWebFetchContentType("text/markdown"), true);
  assert.equal(isSupportedWebFetchContentType("application/json"), true);
  assert.equal(isSupportedWebFetchContentType("application/xml"), true);
  assert.equal(isSupportedWebFetchContentType("application/zip"), false);
  assert.equal(isSupportedWebFetchContentType("image/png"), false);
  assert.equal(isSupportedWebFetchContentType("application/pdf"), false);
});

test("normalizeFetchedText strips active HTML content", () => {
  const normalized = normalizeFetchedText(
    [
      "<!doctype html>",
      "<title>Example</title>",
      "<style>body{display:none}</style>",
      "<script>alert(1)</script>",
      "<body><h1>Hello</h1><p>World</p></body>",
    ].join(""),
    "text/html",
  );

  assert.equal(normalized.title, "Example");
  assert.equal(normalized.content.includes("alert"), false);
  assert.equal(normalized.content.includes("display:none"), false);
  assert.match(normalized.content, /Hello/);
  assert.match(normalized.content, /World/);
});

test("fetchBrowserTextResource saves supported text metadata and truncates by bytes", async () => {
  const result = await fetchBrowserTextResource(
    { url: "https://example.com/long.txt", path: "/workspace/long.txt" },
    {
      maxBytes: 3,
      fetchImpl: async () => response("abcdef", {
        contentType: "text/plain",
        url: "https://example.com/long.txt",
      }),
    },
  );

  assert.deepEqual(result, {
    kind: "web_fetch",
    url: "https://example.com/long.txt",
    finalUrl: "https://example.com/long.txt",
    path: "/workspace/long.txt",
    status: 200,
    contentType: "text/plain",
    bytesWritten: 3,
    truncated: true,
    title: undefined,
    content: "abc",
  });
});

test("fetchBrowserTextResource rejects unsupported content types and blocked final URLs", async () => {
  await assert.rejects(
    () => fetchBrowserTextResource(
      { url: "https://example.com/archive.zip", path: "/workspace/archive.zip" },
      {
        fetchImpl: async () => response("zip", {
          contentType: "application/zip",
          url: "https://example.com/archive.zip",
        }),
      },
    ),
    (error) => isErrorCode(error, "unsupported_content_type"),
  );

  await assert.rejects(
    () => fetchBrowserTextResource(
      { url: "https://example.com/redirect", path: "/workspace/redirect.txt" },
      {
        fetchImpl: async () => response("private", {
          contentType: "text/plain",
          url: "http://127.0.0.1/private",
        }),
      },
    ),
    (error) => isErrorCode(error, "blocked_url"),
  );
});

test("fetchBrowserTextResource reports browser fetch failures as CORS or policy failures", async () => {
  await assert.rejects(
    () => fetchBrowserTextResource(
      { url: "https://example.com/blocked", path: "/workspace/blocked.txt" },
      {
        fetchImpl: async () => {
          throw new TypeError("Failed to fetch");
        },
      },
    ),
    (error) => isErrorCode(error, "web_fetch_cors_blocked"),
  );
});

function response(
  body: BodyInit,
  options: { contentType: string; url: string; status?: number },
): Response {
  const result = new Response(body, {
    status: options.status ?? 200,
    headers: { "content-type": options.contentType },
  });
  Object.defineProperty(result, "url", {
    value: options.url,
  });
  return result;
}

function isErrorCode(error: unknown, code: string): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as { code?: unknown }).code === code
  );
}
