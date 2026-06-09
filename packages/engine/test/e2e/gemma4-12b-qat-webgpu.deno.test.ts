import { createHash } from "node:crypto";

const MODEL = {
  path: "models/gemma4/unsloth-12b-qat/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
  size: 6_716_355_328,
  sha256: "cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165",
};
const PROMPT = "Answer with exactly this natural sentence: The sky is blue.";
const EMPTY_THOUGHT_CHANNEL = "<|channel>thought\n<channel|>";
const EXPECTED_TOKENS = [818, 7217, 563, 3730, 236761];
const EXPECTED_TEXT = "The sky is blue.";
const DEFAULT_WEBGPU_MEMORY_LIMIT_BYTES = 16 * 1024 ** 3;

Deno.test({
  name: "Gemma 4 12B QAT Q4_0 GGUF WebGPU e2e generates natural text with sliding KV ring cache",
  sanitizeOps: false,
  sanitizeResources: false,
  async fn() {
    await assertPinnedModel(MODEL.path);
    const capture = installWebGpuCapture();
    const {
      applyChatGenerationPrompt,
      applyChatTemplate,
      buildTokenizer,
      checkWebGpuSupport,
      createChatSession,
      createFileGgufTensorReader,
      createWebGpuProvider,
      decode,
      prefill,
    } = await import("../../src/index.ts");

    const support = await checkWebGpuSupport();
    assert(support.available, `WebGPU unavailable: ${JSON.stringify(support)}`);
    const adapter = await navigator.gpu?.requestAdapter();
    assert(adapter?.features.has("shader-f16"), "WebGPU native f16 support is required for KV cache storage");

    const reader = await createFileGgufTensorReader(new RangeFile(MODEL.path));
    assertQatTensorTypes(reader.metadata.tensors);
    const tokenizer = buildTokenizer(reader.metadata);
    assertEquals(tokenizer.tokenToId("<turn|>"), 106, "Gemma 4 tokenizer should resolve <turn|> to official EOS id 106");
    const session = createChatSession(reader, {
      maxContextLength: 2048,
      providers: [createWebGpuProvider({
        memoryLimitBytes: Number(
          Deno.env.get("HELIOPAUSE_WEBGPU_MEMORY_LIMIT_BYTES") ?? DEFAULT_WEBGPU_MEMORY_LIMIT_BYTES,
        ),
        prefillChunkSize: 16,
      })],
    });
    const state = session.createInferenceState();
    const prompt = applyChatTemplate([{ role: "user", content: PROMPT }], { addGenerationPrompt: false }) +
      applyChatGenerationPrompt() +
      EMPTY_THOUGHT_CHANNEL;
    const promptTokenIds = tokenizer.tokenize(prompt, { addBos: true });

    let result = await prefill(session, state, promptTokenIds, { logitsTopK: 8 });
    const generated: number[] = [];
    let nextTokenId = result.nextTokenId;
    for (let index = 0; index < EXPECTED_TOKENS.length; index += 1) {
      generated.push(nextTokenId);
      if (index < EXPECTED_TOKENS.length - 1) {
        result = await decode(session, state, nextTokenId, { logitsTopK: 8 });
        nextTokenId = result.nextTokenId;
      }
    }

    await capture.device?.queue?.onSubmittedWorkDone?.();
    const validationError = await capture.device?.popErrorScope?.();
    assert(!validationError, `WebGPU validation error: ${formatWebGpuError(validationError)}`);

    const text = generated.map((id) => tokenizer.detokenize([id])).join("");
    assertArrayEquals(generated, EXPECTED_TOKENS, `generated text: ${JSON.stringify(text)}`);
    assertNaturalSentence(text);
    assertWebGpuKvCacheMatchesRingCapacity(capture.buffers, session.manifest, state.contextLength);
  },
});

class RangeFile {
  constructor(private readonly path: string) {}

  slice(start = 0, end?: number): { arrayBuffer: () => Promise<ArrayBuffer> } {
    return {
      arrayBuffer: async () => {
        const file = await Deno.open(this.path, { read: true });
        try {
          await file.seek(start, Deno.SeekMode.Start);
          const length = Math.max(0, (end ?? (await file.stat()).size) - start);
          const bytes = new Uint8Array(length);
          let offset = 0;
          while (offset < length) {
            const read = await file.read(bytes.subarray(offset));
            if (read === null) {
              break;
            }
            offset += read;
          }
          return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + offset);
        } finally {
          file.close();
        }
      },
    };
  }
}

async function assertPinnedModel(path: string): Promise<void> {
  const stat = await Deno.stat(path);
  assert(stat.isFile, `${path} is not a file`);
  assertEquals(stat.size, MODEL.size, `${path} has wrong byte size`);
  assertEquals(await sha256File(path), MODEL.sha256, `${path} has wrong SHA-256`);
}

async function sha256File(path: string): Promise<string> {
  const file = await Deno.open(path, { read: true });
  const hash = createHash("sha256");
  const buffer = new Uint8Array(1024 * 1024);
  try {
    while (true) {
      const read = await file.read(buffer);
      if (read === null) {
        break;
      }
      hash.update(buffer.subarray(0, read));
    }
  } finally {
    file.close();
  }
  return hash.digest("hex");
}

type CapturedDevice = {
  queue?: { onSubmittedWorkDone?: () => Promise<void> };
  pushErrorScope?: (filter: "validation") => void;
  popErrorScope?: () => Promise<unknown>;
};

type CapturedBuffer = {
  label: string;
  size: number;
};

function installWebGpuCapture(): {
  device?: CapturedDevice;
  requiredLimits: Array<Record<string, number>>;
  requiredFeatures: string[][];
  buffers: CapturedBuffer[];
} {
  const originalGpu = navigator.gpu;
  const capture: {
    device?: CapturedDevice;
    requiredLimits: Array<Record<string, number>>;
    requiredFeatures: string[][];
    buffers: CapturedBuffer[];
  } = { requiredLimits: [], requiredFeatures: [], buffers: [] };
  if (!originalGpu?.requestAdapter) {
    return capture;
  }
  Object.defineProperty(navigator, "gpu", {
    configurable: true,
    value: {
      ...originalGpu,
      requestAdapter: async (...args: Parameters<typeof originalGpu.requestAdapter>) => {
        const adapter = await originalGpu.requestAdapter(...args);
        if (!adapter) {
          return adapter;
        }
        return {
          ...adapter,
          features: adapter.features,
          info: adapter.info,
          limits: adapter.limits,
          requestDevice: async (...deviceArgs: Parameters<typeof adapter.requestDevice>) => {
            const descriptor = deviceArgs[0] as {
              requiredFeatures?: Iterable<string>;
              requiredLimits?: Record<string, number>;
            } | undefined;
            if (descriptor?.requiredFeatures) {
              capture.requiredFeatures.push(Array.from(descriptor.requiredFeatures));
            }
            if (descriptor?.requiredLimits) {
              capture.requiredLimits.push({ ...descriptor.requiredLimits });
            }
            const device = await adapter.requestDevice(...deviceArgs);
            const rawCreateBuffer = device.createBuffer.bind(device);
            const overrides = new Map<PropertyKey, unknown>();
            const wrappedDevice = new Proxy(device, {
              get(target, property) {
                if (property === "createBuffer") {
                  const override = overrides.get(property) as ((descriptor: GPUBufferDescriptor) => GPUBuffer) | undefined;
                  const create = override ?? rawCreateBuffer;
                  return (bufferDescriptor: GPUBufferDescriptor) => {
                    capture.buffers.push({
                      label: String(bufferDescriptor.label ?? ""),
                      size: Number(bufferDescriptor.size),
                    });
                    return create(bufferDescriptor);
                  };
                }
                if (overrides.has(property)) {
                  return overrides.get(property);
                }
                const value = Reflect.get(target, property, target);
                return typeof value === "function" ? value.bind(target) : value;
              },
              set(_target, property, value) {
                overrides.set(property, value);
                return true;
              },
            });
            capture.device = wrappedDevice as CapturedDevice;
            capture.device.pushErrorScope?.("validation");
            return wrappedDevice;
          },
        };
      },
    },
  });
  return capture;
}

function assertQatTensorTypes(tensors: readonly { name: string; type: string }[]): void {
  const byName = new Map(tensors.map((tensor) => [tensor.name, tensor.type]));
  assertEquals(byName.get("token_embd.weight"), "Q4_0", "token_embd.weight should be Q4_0");
  assertEquals(byName.get("blk.0.attn_q.weight"), "Q4_0", "blk.0.attn_q.weight should be Q4_0");
  assertEquals(byName.get("blk.0.attn_k.weight"), "Q4_0", "blk.0.attn_k.weight should be Q4_0");
  assertEquals(byName.get("blk.0.attn_v.weight"), "Q4_0", "blk.0.attn_v.weight should be Q4_0");
  assertEquals(byName.get("blk.0.ffn_gate.weight"), "Q4_0", "blk.0.ffn_gate.weight should be Q4_0");
}

function assertWebGpuKvCacheMatchesRingCapacity(
  buffers: readonly CapturedBuffer[],
  manifest: {
    headCountKv: number;
    keyLength: number;
    valueLength: number;
    layerHeadCountKv: number[];
    layerKeyLengths: number[];
    layerValueLengths: number[];
    layerKinds: string[];
    slidingWindow: number;
  },
  contextLength: number,
): void {
  const keyBuffers = buffers.filter((buffer) => /\.gpu\.key_cache$/.test(buffer.label));
  const valueBuffers = buffers.filter((buffer) => /\.gpu\.value_cache$/.test(buffer.label));
  assert(keyBuffers.length > 0, `WebGPU key cache buffers were not created: ${JSON.stringify(buffers)}`);
  assertEquals(valueBuffers.length, keyBuffers.length, "WebGPU key/value cache buffer counts should match");

  const actualBytes = sumSizes(keyBuffers) + sumSizes(valueBuffers);
  const expectedF16Bytes = expectedCacheBytes(keyBuffers, manifest, contextLength, "key") +
    expectedCacheBytes(valueBuffers, manifest, contextLength, "value");
  const naiveF16Bytes = naiveCacheBytes(keyBuffers, manifest, contextLength, "key") +
    naiveCacheBytes(valueBuffers, manifest, contextLength, "value");
  assertEquals(
    actualBytes,
    expectedF16Bytes,
    `WebGPU KV cache buffers should use sliding ring capacity; actual=${actualBytes}, ring=${expectedF16Bytes}, naive=${naiveF16Bytes}`,
  );
  assert(
    actualBytes < naiveF16Bytes,
    `12B sliding KV cache did not shrink below full-context allocation: actual=${actualBytes}, naive=${naiveF16Bytes}`,
  );
}

function expectedCacheBytes(
  buffers: readonly CapturedBuffer[],
  manifest: {
    headCountKv: number;
    keyLength: number;
    valueLength: number;
    layerHeadCountKv: number[];
    layerKeyLengths: number[];
    layerValueLengths: number[];
    layerKinds: string[];
    slidingWindow: number;
  },
  contextLength: number,
  kind: "key" | "value",
): number {
  return buffers.reduce((total, buffer) => {
    const layer = layerFromCacheLabel(buffer.label);
    const capacity = manifest.layerKinds[layer] === "sliding-attention"
      ? Math.min(contextLength, manifest.slidingWindow)
      : contextLength;
    return total + cacheBytesForLayer(manifest, layer, capacity, kind);
  }, 0);
}

function naiveCacheBytes(
  buffers: readonly CapturedBuffer[],
  manifest: {
    headCountKv: number;
    keyLength: number;
    valueLength: number;
    layerHeadCountKv: number[];
    layerKeyLengths: number[];
    layerValueLengths: number[];
  },
  contextLength: number,
  kind: "key" | "value",
): number {
  return buffers.reduce((total, buffer) => total + cacheBytesForLayer(
    manifest,
    layerFromCacheLabel(buffer.label),
    contextLength,
    kind,
  ), 0);
}

function cacheBytesForLayer(
  manifest: {
    headCountKv: number;
    keyLength: number;
    valueLength: number;
    layerHeadCountKv: number[];
    layerKeyLengths: number[];
    layerValueLengths: number[];
  },
  layer: number,
  capacity: number,
  kind: "key" | "value",
): number {
  const headCountKv = manifest.layerHeadCountKv[layer] ?? manifest.headCountKv;
  const size = kind === "key"
    ? (manifest.layerKeyLengths[layer] ?? manifest.keyLength)
    : (manifest.layerValueLengths[layer] ?? manifest.valueLength);
  return capacity * headCountKv * size * 2;
}

function layerFromCacheLabel(label: string): number {
  const match = /blk\.(\d+)\.gpu\.(?:key|value)_cache$/.exec(label);
  assert(match, `Unexpected WebGPU KV cache label: ${label}`);
  return Number(match[1]);
}

function sumSizes(buffers: readonly CapturedBuffer[]): number {
  return buffers.reduce((total, buffer) => total + buffer.size, 0);
}

function assertNaturalSentence(text: string): void {
  const trimmed = text.trim();
  assert(trimmed.length > 0, "generated output is empty");
  assertEquals(trimmed, EXPECTED_TEXT, `generated output is not the expected natural sentence`);
  assert(/[A-Za-z]/.test(trimmed), `generated output has no alphabetic characters: ${JSON.stringify(text)}`);
  assert(/\bblue\b/i.test(trimmed), `generated output does not mention blue: ${JSON.stringify(text)}`);
  assert(
    trimmed.split(/\s+/).filter(Boolean).length >= 4,
    `generated output has too few words: ${JSON.stringify(text)}`,
  );
  assert(
    !/^(?:<[^>]+>|\d+|\W)+$/.test(trimmed),
    `generated output is only special tokens, ids, or markup: ${JSON.stringify(text)}`,
  );
  assert(
    !/\b(\w+)(?:\s+\1){3,}\b/i.test(trimmed),
    `generated output contains repeated-token artifacts: ${JSON.stringify(text)}`,
  );
}

function formatWebGpuError(error: unknown): string {
  if (error && typeof error === "object") {
    const message = "message" in error ? String((error as { message?: unknown }).message) : String(error);
    return `${error.constructor?.name ?? "GPUError"}: ${message}`;
  }
  return String(error);
}

function assert(value: unknown, message: string): asserts value {
  if (!value) {
    throw new Error(message);
  }
}

function assertEquals<T>(actual: T, expected: T, message?: string): void {
  if (actual !== expected) {
    throw new Error(message ?? `expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function assertArrayEquals(actual: readonly number[], expected: readonly number[], message?: string): void {
  if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index])) {
    throw new Error(message ?? `expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}
