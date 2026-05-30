import { createHash } from "node:crypto";

const MODEL = {
  path: "models/gemma4/unsloth/gemma-4-E4B-it-Q4_K_M.gguf",
  size: 4_977_169_568,
  sha256: "519b9793ed6ce0ff530f1b7c96e848e08e49e7af4d57bb97f76215963a54146d",
};
const PROMPT = "Answer with exactly this natural sentence: The sky is blue.";
const EXPECTED_TOKENS = [818, 7217, 563, 3730, 236761];
const EXPECTED_TEXT = "The sky is blue.";
const DEFAULT_WEBGPU_MEMORY_LIMIT_BYTES = 12 * 1024 ** 3;

Deno.test({
  name: "Gemma 4 GGUF WebGPU e2e generates natural text",
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
      createDeterministicRng,
      createFileGgufTensorReader,
      createWebGpuProvider,
      DEFAULT_GENERATION_CONFIG,
      decode,
      prefill,
      resolveGenerationSamplingOptions,
      sampleNextToken,
    } = await import("../../src/index.ts");

    const support = await checkWebGpuSupport();
    assert(support.available, `WebGPU unavailable: ${JSON.stringify(support)}`);
    const adapter = await navigator.gpu?.requestAdapter();
    assert(adapter?.features.has("shader-f16"), "WebGPU native f16 support is required for KV cache storage");

    const reader = await createFileGgufTensorReader(new RangeFile(MODEL.path));
    const tokenizer = buildTokenizer(reader.metadata);
    assertEquals(tokenizer.tokenToId("<turn|>"), 106, "Gemma 4 tokenizer should resolve <turn|> to official EOS id 106");
    const session = createChatSession(reader, {
      maxContextLength: 256,
      providers: [createWebGpuProvider({
        memoryLimitBytes: Number(
          Deno.env.get("HELIOPAUSE_WEBGPU_MEMORY_LIMIT_BYTES") ?? DEFAULT_WEBGPU_MEMORY_LIMIT_BYTES,
        ),
        prefillChunkSize: 16,
      })],
    });
    const state = session.createInferenceState();
    const prompt = applyChatTemplate([{ role: "user", content: PROMPT }], { addGenerationPrompt: false }) +
      applyChatGenerationPrompt();
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
    assert(!validationError, `WebGPU validation error: ${String(validationError)}`);

    const text = generated.map((id) => tokenizer.detokenize([id])).join("");
    assertArrayEquals(generated, EXPECTED_TOKENS, `generated text: ${JSON.stringify(text)}`);
    assertNaturalSentence(text);

    const sampledState = session.createInferenceState();
    const sampling = resolveGenerationSamplingOptions({ seed: 0 });
    assertEquals(sampling.doSample, DEFAULT_GENERATION_CONFIG.doSample);
    assertEquals(sampling.temperature, DEFAULT_GENERATION_CONFIG.temperature);
    assertEquals(sampling.topP, DEFAULT_GENERATION_CONFIG.topP);
    assertEquals(sampling.topK, DEFAULT_GENERATION_CONFIG.topK);
    assertEquals(sampling.logitsTopK, 64);

    let sampledResult = await prefill(session, sampledState, promptTokenIds, { logitsTopK: sampling.logitsTopK });
    assertEquals(sampledResult.topTokens?.length, 64, "official sampling prefill should request topK=64 candidates");
    const sampled: number[] = [];
    const rng = createDeterministicRng(sampling.seed);
    let sampledTokenId = sampleNextToken(sampledResult.topTokens ?? [], sampling, rng);
    const stopTokenIds = new Set([
      tokenizer.eosTokenId,
      tokenizer.tokenToId("<turn|>"),
      tokenizer.tokenToId("<eos>"),
      tokenizer.tokenToId("<|im_end|>"),
      ...DEFAULT_GENERATION_CONFIG.eosTokenIds,
    ].filter((id): id is number => typeof id === "number"));
    let sampledStopped = false;
    for (let index = 0; index < 128; index += 1) {
      if (stopTokenIds.has(sampledTokenId)) {
        sampledStopped = true;
        break;
      }
      sampled.push(sampledTokenId);
      sampledResult = await decode(session, sampledState, sampledTokenId, { logitsTopK: sampling.logitsTopK });
      assertEquals(sampledResult.topTokens?.length, 64, "official sampling decode should request topK=64 candidates");
      sampledTokenId = sampleNextToken(sampledResult.topTokens ?? [], sampling, rng);
    }
    assert(sampled.length > 0, "official sampling path should generate at least one token");
    assert(sampledStopped, "official sampling path should eventually reach an official stop token");

    const stats = session.cacheStats().executionProviderStats;
    assert(
      typeof stats.webgpuTokenIdInputTokens === "number" && stats.webgpuTokenIdInputTokens > 0,
      `WebGPU token-id input path was not observed: ${JSON.stringify(stats)}`,
    );
    assert(
      capture.requiredLimits.some((limits) =>
        typeof limits.maxBufferSize === "number" && typeof limits.maxStorageBufferBindingSize === "number"
      ),
      `WebGPU requiredLimits did not include raised buffer limits: ${JSON.stringify(capture.requiredLimits)}`,
    );
    assert(
      capture.requiredFeatures.some((features) => features.includes("shader-f16")),
      `WebGPU requiredFeatures did not include shader-f16: ${JSON.stringify(capture.requiredFeatures)}`,
    );
    assertKvCacheBuffersUseF16(capture.buffers, session.manifest, state.contextLength);
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

function assertKvCacheBuffersUseF16(
  buffers: readonly CapturedBuffer[],
  manifest: { headCountKv: number; keyLength: number; valueLength: number; layerKeyLengths: number[]; layerValueLengths: number[] },
  contextLength: number,
): void {
  const keyBuffers = buffers.filter((buffer) => /\.gpu\.key_cache$/.test(buffer.label));
  const valueBuffers = buffers.filter((buffer) => /\.gpu\.value_cache$/.test(buffer.label));
  assert(keyBuffers.length > 0, `WebGPU key cache buffers were not created: ${JSON.stringify(buffers)}`);
  assertEquals(valueBuffers.length, keyBuffers.length, "WebGPU key/value cache buffer counts should match");

  const actualBytes = sumSizes(keyBuffers) + sumSizes(valueBuffers);
  const expectedF16Bytes = expectedCacheBytes(keyBuffers, manifest, contextLength, "key") +
    expectedCacheBytes(valueBuffers, manifest, contextLength, "value");
  const expectedF32Bytes = expectedF16Bytes * 2;
  assertEquals(
    actualBytes,
    expectedF16Bytes,
    `WebGPU KV cache buffers should be native f16 sized; actual=${actualBytes}, f16=${expectedF16Bytes}, f32=${expectedF32Bytes}`,
  );
}

function expectedCacheBytes(
  buffers: readonly CapturedBuffer[],
  manifest: { headCountKv: number; keyLength: number; valueLength: number; layerKeyLengths: number[]; layerValueLengths: number[] },
  contextLength: number,
  kind: "key" | "value",
): number {
  return buffers.reduce((total, buffer) => {
    const match = /blk\.(\d+)\.gpu\.(?:key|value)_cache$/.exec(buffer.label);
    assert(match, `Unexpected WebGPU KV cache label: ${buffer.label}`);
    const layer = Number(match[1]);
    const elementCount = contextLength * manifest.headCountKv *
      (kind === "key"
        ? (manifest.layerKeyLengths[layer] ?? manifest.keyLength)
        : (manifest.layerValueLengths[layer] ?? manifest.valueLength));
    return total + elementCount * 2;
  }, 0);
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
