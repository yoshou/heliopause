import type { WorkerWebFetchContent } from "./engine-worker-protocol";

export const WEB_FETCH_MAX_BYTES = 256 * 1024;
export const WEB_FETCH_TIMEOUT_MS = 10_000;

export type BrowserWebFetchRequest = {
  url: string;
  path: string;
};

export type BrowserWebFetchResult = WorkerWebFetchContent & {
  content: string;
};

export type BrowserWebFetchOptions = {
  fetchImpl?: typeof fetch;
  signal?: AbortSignal;
  maxBytes?: number;
  timeoutMs?: number;
};

type ToolLikeError = {
  code: string;
  message: string;
  retryable?: boolean;
};

const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();

export async function fetchBrowserTextResource(
  request: BrowserWebFetchRequest,
  options: BrowserWebFetchOptions = {},
): Promise<BrowserWebFetchResult> {
  const url = validateWebFetchUrl(request.url);
  const fetchImpl = options.fetchImpl ?? fetch;
  const maxBytes = options.maxBytes ?? WEB_FETCH_MAX_BYTES;
  const timeoutMs = options.timeoutMs ?? WEB_FETCH_TIMEOUT_MS;
  const abortController = new AbortController();
  const timeout = setTimeout(() => abortController.abort(), timeoutMs);
  const abortFromParent = () => abortController.abort();
  options.signal?.addEventListener("abort", abortFromParent, { once: true });

  try {
    throwIfAborted(options.signal);
    const response = await fetchImpl(url.href, {
      method: "GET",
      credentials: "omit",
      mode: "cors",
      redirect: "follow",
      signal: abortController.signal,
    });
    throwIfAborted(options.signal);

    const finalUrl = validateWebFetchUrl(response.url || url.href);
    const contentType = response.headers.get("content-type")?.trim() ?? "";
    if (!isSupportedWebFetchContentType(contentType)) {
      throw toolError("unsupported_content_type", `web_fetch only supports text resources, got ${contentType || "unknown content-type"}.`);
    }
    if (!response.ok) {
      throw toolError("web_fetch_http_error", `web_fetch failed with HTTP status ${response.status}.`, response.status >= 500);
    }

    const { bytes, truncated } = await readResponseBytes(response, maxBytes, options.signal);
    const rawText = textDecoder.decode(bytes);
    const normalized = normalizeFetchedText(rawText, contentType);
    return {
      kind: "web_fetch",
      url: url.href,
      finalUrl: finalUrl.href,
      path: request.path,
      status: response.status,
      contentType,
      bytesWritten: textEncoder.encode(normalized.content).byteLength,
      truncated,
      title: normalized.title,
      content: normalized.content,
    };
  } catch (error) {
    throw normalizeWebFetchError(error, options.signal);
  } finally {
    clearTimeout(timeout);
    options.signal?.removeEventListener("abort", abortFromParent);
  }
}

export function validateWebFetchUrl(value: string): URL {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw toolError("invalid_url", "web_fetch url must be a valid URL.");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw toolError("invalid_url", "web_fetch only supports http:// and https:// URLs.");
  }
  if (url.username || url.password) {
    throw toolError("invalid_url", "web_fetch does not allow URLs with embedded credentials.");
  }
  if (isBlockedHostname(url.hostname)) {
    throw toolError("blocked_url", "web_fetch does not allow localhost, loopback, private, or link-local hosts.");
  }
  return url;
}

export function isSupportedWebFetchContentType(contentType: string): boolean {
  const type = contentType.split(";")[0]?.trim().toLowerCase() ?? "";
  return (
    type === "text/html" ||
    type === "text/plain" ||
    type.startsWith("text/") ||
    type === "application/json" ||
    type === "application/xml" ||
    type === "application/xhtml+xml"
  );
}

export function normalizeFetchedText(
  text: string,
  contentType: string,
): { content: string; title?: string } {
  const type = contentType.split(";")[0]?.trim().toLowerCase() ?? "";
  if (type !== "text/html" && type !== "application/xhtml+xml") {
    return { content: text };
  }

  const parser = typeof DOMParser !== "undefined" ? new DOMParser() : undefined;
  if (!parser) {
    return {
      content: fallbackHtmlToText(text),
      title: fallbackHtmlTitle(text),
    };
  }

  const doc = parser.parseFromString(text, "text/html");
  for (const node of doc.querySelectorAll("script,style,noscript,svg,meta,link")) {
    node.remove();
  }
  const title = doc.querySelector("title")?.textContent?.trim() || undefined;
  return {
    content: normalizeWhitespace(doc.body?.textContent ?? doc.documentElement.textContent ?? ""),
    title,
  };
}

function isBlockedHostname(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[(.*)]$/, "$1").replace(/\.$/, "");
  if (host === "localhost" || host.endsWith(".localhost")) {
    return true;
  }
  if (host.includes(":")) {
    return isBlockedIpv6(host);
  }
  return isBlockedIpv4(host);
}

function isBlockedIpv4(host: string): boolean {
  const parts = host.split(".");
  if (parts.length !== 4) {
    return false;
  }
  const octets = parts.map((part) => /^\d+$/.test(part) ? Number(part) : Number.NaN);
  if (!octets.every((octet) => Number.isInteger(octet) && octet >= 0 && octet <= 255)) {
    return false;
  }
  const [a, b] = octets;
  return (
    a === 0 ||
    a === 10 ||
    a === 127 ||
    (a === 100 && b >= 64 && b <= 127) ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168)
  );
}

function isBlockedIpv6(host: string): boolean {
  return (
    host === "::1" ||
    host.startsWith("fc") ||
    host.startsWith("fd") ||
    host.startsWith("fe8") ||
    host.startsWith("fe9") ||
    host.startsWith("fea") ||
    host.startsWith("feb")
  );
}

async function readResponseBytes(
  response: Response,
  maxBytes: number,
  signal: AbortSignal | undefined,
): Promise<{ bytes: Uint8Array; truncated: boolean }> {
  throwIfAborted(signal);
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    return bytes.length > maxBytes
      ? { bytes: bytes.slice(0, maxBytes), truncated: true }
      : { bytes, truncated: false };
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  let truncated = false;

  try {
    while (true) {
      throwIfAborted(signal);
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (!value) {
        continue;
      }
      const remainingBytes = maxBytes - totalBytes;
      if (value.length > remainingBytes) {
        if (remainingBytes > 0) {
          chunks.push(value.slice(0, remainingBytes));
          totalBytes += remainingBytes;
        }
        truncated = true;
        await reader.cancel();
        break;
      }
      chunks.push(value);
      totalBytes += value.length;
    }
  } finally {
    reader.releaseLock();
  }

  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return { bytes, truncated };
}

function fallbackHtmlTitle(html: string): string | undefined {
  return html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]?.replace(/<[^>]*>/g, "").trim() || undefined;
}

function fallbackHtmlToText(html: string): string {
  return normalizeWhitespace(
    html
      .replace(/<script\b[\s\S]*?<\/script>/gi, "")
      .replace(/<style\b[\s\S]*?<\/style>/gi, "")
      .replace(/<noscript\b[\s\S]*?<\/noscript>/gi, "")
      .replace(/<svg\b[\s\S]*?<\/svg>/gi, "")
      .replace(/<[^>]+>/g, " "),
  );
}

function normalizeWhitespace(value: string): string {
  return value
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((line) => line.replace(/[ \t\f\v]+/g, " ").trim())
    .filter(Boolean)
    .join("\n");
}

function normalizeWebFetchError(error: unknown, signal: AbortSignal | undefined): unknown {
  if (signal?.aborted || (error instanceof DOMException && error.name === "AbortError")) {
    return new DOMException("web_fetch was aborted.", "AbortError");
  }
  if (isToolLikeError(error)) {
    return error;
  }
  if (error instanceof TypeError) {
    return toolError("web_fetch_cors_blocked", "The browser could not read this URL. It may be blocked by CORS or network policy.", true);
  }
  if (error instanceof Error) {
    return toolError("web_fetch_failed", error.message, true);
  }
  return toolError("web_fetch_failed", "web_fetch failed.", true);
}

function toolError(code: string, message: string, retryable?: boolean): ToolLikeError {
  return retryable === undefined ? { code, message } : { code, message, retryable };
}

function isToolLikeError(value: unknown): value is ToolLikeError {
  return (
    typeof value === "object" &&
    value !== null &&
    "code" in value &&
    typeof (value as { code?: unknown }).code === "string" &&
    "message" in value &&
    typeof (value as { message?: unknown }).message === "string"
  );
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("web_fetch was aborted.", "AbortError");
  }
}
