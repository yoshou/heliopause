import { createHash } from "node:crypto";

const TARGET_MODEL = {
  path: "models/gemma4/unsloth/gemma-4-E4B-it-Q4_K_M.gguf",
  size: 4_977_169_568,
  sha256: "519b9793ed6ce0ff530f1b7c96e848e08e49e7af4d57bb97f76215963a54146d",
};
const ASSISTANT_MODEL = {
  path: "models/gemma4/AtomicChat/gemma-4-E4B-it-assistant.Q4_K_M.gguf",
  size: 78_575_008,
  sha256: "6c93075cefa2902887afd7e341b32f3710fb3ecc13e3d7f31b272927cb30dacd",
};
const PROMPT = "Answer with exactly this natural sentence: The sky is blue.";
const EXPECTED_TOKENS = [818, 7217, 563, 3730, 236761];
const EXPECTED_TEXT = "The sky is blue.";
const EXPECTED_ASSISTANT_LAYER_MAPPING = [9, 20, 22, 23];
const DEFAULT_WEBGPU_MEMORY_LIMIT_BYTES = 12 * 1024 ** 3;
const ACCEPTANCE_PROMPTS = [
  {
    name: "english-sky",
    prompt: "Answer with exactly this natural sentence: The sky is blue.",
    maxNewTokens: 8,
    expectedProposed: 4,
    expectedAccepted: 3,
  },
  {
    name: "japanese-sky",
    prompt: "次の自然な日本語文だけで答えてください: 空は青いです。",
    maxNewTokens: 8,
    expectedProposed: 5,
    expectedAccepted: 3,
  },
] as const;

Deno.test({
  name: "Gemma 4 MTP WebGPU e2e matches greedy target tokens",
  sanitizeOps: false,
  sanitizeResources: false,
  async fn() {
    await assertPinnedModel(TARGET_MODEL);
    await assertPinnedModel(ASSISTANT_MODEL);
    const capture = installWebGpuCapture();
    const {
      acceptMtpDraft,
      acceptGreedyMtpDraft,
      applyChatGenerationPrompt,
      applyChatTemplate,
      buildTokenizer,
      checkWebGpuSupport,
      createChatSession,
      createDeterministicRng,
      createFileGgufTensorReader,
      createMtpAssistantSession,
      createWasmProvider,
      createWebGpuProvider,
      decode,
      finalizeMtpVerification,
      generateChatTurn,
      mapMtpAssistantLayerToTargetKvLayer,
      prefill,
      prefillMtpTarget,
      prefillState,
      proposeMtpDraft,
      resolveGenerationSamplingOptions,
      sampleMtpTokenDistribution,
      verifyMtpDraft,
    } = await import("../../src/index.ts");

    const support = await checkWebGpuSupport();
    assert(support.available, `WebGPU unavailable: ${JSON.stringify(support)}`);
    const adapter = await navigator.gpu?.requestAdapter();
    assert(adapter?.features.has("shader-f16"), "WebGPU native f16 support is required for KV cache storage");

    const targetReader = await createFileGgufTensorReader(new RangeFile(TARGET_MODEL.path));
    const assistantReader = await createFileGgufTensorReader(new RangeFile(ASSISTANT_MODEL.path));
    const tokenizer = buildTokenizer(targetReader.metadata);
    const assistantSession = createMtpAssistantSession(assistantReader, {
      providers: [createWasmProvider()],
    });

    const baselineSession = createChatSession(targetReader, {
      maxContextLength: 256,
      providers: [createWebGpuProvider({
        memoryLimitBytes: Number(
          Deno.env.get("HELIOPAUSE_WEBGPU_MEMORY_LIMIT_BYTES") ?? DEFAULT_WEBGPU_MEMORY_LIMIT_BYTES,
        ),
        prefillChunkSize: 16,
      })],
    });
    const mapping = Array.from(
      { length: assistantSession.manifest.blockCount },
      (_, layer) => mapMtpAssistantLayerToTargetKvLayer(
        baselineSession.manifest,
        assistantSession.manifest,
        layer,
      ),
    );
    assertArrayEquals(mapping, EXPECTED_ASSISTANT_LAYER_MAPPING);

    const baselineTokens: number[] = [];
    const baseline = await generateChatTurn(
      baselineSession,
      tokenizer,
      baselineSession.createInferenceState(),
      PROMPT,
      {
        appendTurnEnd: false,
        doSample: false,
        maxNewTokens: EXPECTED_TOKENS.length,
        onToken: (chunk) => baselineTokens.push(chunk.tokenId),
      },
    );
    assertArrayEquals(baselineTokens, EXPECTED_TOKENS, `baseline content: ${JSON.stringify(baseline.content)}`);
    assertNaturalSentence(baseline.content);

    const mtpSession = createChatSession(targetReader, {
      maxContextLength: 256,
      providers: [createWebGpuProvider({
        memoryLimitBytes: Number(
          Deno.env.get("HELIOPAUSE_WEBGPU_MEMORY_LIMIT_BYTES") ?? DEFAULT_WEBGPU_MEMORY_LIMIT_BYTES,
        ),
        prefillChunkSize: 16,
      })],
    });
    const mtpTokens: number[] = [];
    const mtp = await generateChatTurn(
      mtpSession,
      tokenizer,
      mtpSession.createInferenceState(),
      PROMPT,
      {
        appendTurnEnd: false,
        doSample: false,
        maxNewTokens: EXPECTED_TOKENS.length,
        mtp: {
          assistantSession,
          numSpeculativeTokens: 1,
        },
        onToken: (chunk) => mtpTokens.push(chunk.tokenId),
      },
    );

    await capture.device?.queue?.onSubmittedWorkDone?.();
    const validationError = await capture.device?.popErrorScope?.();
    assert(!validationError, `WebGPU validation error: ${formatWebGpuError(validationError)}`);

    assertArrayEquals(mtpTokens, EXPECTED_TOKENS, `MTP content: ${JSON.stringify(mtp.content)}`);
    assertArrayEquals(mtpTokens, baselineTokens);
    assertNaturalSentence(mtp.content);

    const sampledMtpSession = createChatSession(targetReader, {
      maxContextLength: 256,
      providers: [createWebGpuProvider({
        memoryLimitBytes: Number(
          Deno.env.get("HELIOPAUSE_WEBGPU_MEMORY_LIMIT_BYTES") ?? DEFAULT_WEBGPU_MEMORY_LIMIT_BYTES,
        ),
        prefillChunkSize: 16,
      })],
    });
    const sampledMtpTokens: number[] = [];
    await generateChatTurn(
      sampledMtpSession,
      tokenizer,
      sampledMtpSession.createInferenceState(),
      PROMPT,
      {
        appendTurnEnd: false,
        doSample: true,
        temperature: 1,
        topP: 1,
        topK: 1,
        seed: 42,
        maxNewTokens: EXPECTED_TOKENS.length,
        mtp: {
          assistantSession,
          numSpeculativeTokens: 1,
        },
        onToken: (chunk) => sampledMtpTokens.push(chunk.tokenId),
      },
    );
    assertArrayEquals(sampledMtpTokens, EXPECTED_TOKENS, "MTP sampling path with topK=1 should match greedy target tokens");

    const samplingTopK = resolveGenerationSamplingOptions({ doSample: true, temperature: 1, topP: 1, topK: 4, seed: 7 });
    const samplingStep = await measureMtpSamplingStep(
      targetReader,
      tokenizer,
      applyChatGenerationPrompt,
      applyChatTemplate,
      createChatSession,
      createWebGpuProvider,
      createDeterministicRng,
      prefillState,
      prefillMtpTarget,
      proposeMtpDraft,
      verifyMtpDraft,
      finalizeMtpVerification,
      acceptMtpDraft,
      sampleMtpTokenDistribution,
      assistantSession,
      samplingTopK,
      PROMPT,
    );
    assert(
      samplingStep.targetDistributionWidths.every((width) => width > 1),
      `MTP sampling verification did not expose per-position target topK distributions: ${
        JSON.stringify(samplingStep.targetDistributionWidths)
      }`,
    );
    assert(samplingStep.acceptance.committedLength >= 1, "MTP sampling acceptance did not commit the seed token");

    const sampling = resolveGenerationSamplingOptions({ doSample: false });
    let proposedDraftTokens = 0;
    let acceptedDraftTokens = 0;
    for (const acceptanceCase of ACCEPTANCE_PROMPTS) {
      const baselineForPrompt = await generateGreedyTokens(
        targetReader,
        tokenizer,
        applyChatGenerationPrompt,
        applyChatTemplate,
        createChatSession,
        createWebGpuProvider,
        prefill,
        decode,
        acceptanceCase.prompt,
        acceptanceCase.maxNewTokens,
      );
      const measured = await measureMtpAcceptance(
        targetReader,
        tokenizer,
        applyChatGenerationPrompt,
        applyChatTemplate,
        createChatSession,
        createWebGpuProvider,
        createDeterministicRng,
        prefillState,
        prefillMtpTarget,
        proposeMtpDraft,
        verifyMtpDraft,
        finalizeMtpVerification,
        acceptGreedyMtpDraft,
        assistantSession,
        sampling,
        acceptanceCase.prompt,
        acceptanceCase.maxNewTokens,
      );
      assertArrayEquals(measured.tokens, baselineForPrompt, `${acceptanceCase.name} MTP tokens should match greedy target`);
      assertEquals(
        measured.proposed,
        acceptanceCase.expectedProposed,
        `${acceptanceCase.name} proposed draft count changed`,
      );
      assertEquals(
        measured.accepted,
        acceptanceCase.expectedAccepted,
        `${acceptanceCase.name} accepted draft count changed`,
      );
      proposedDraftTokens += measured.proposed;
      acceptedDraftTokens += measured.accepted;
    }
    assertEquals(proposedDraftTokens, 9, "pinned MTP acceptance suite should propose nine draft tokens");
    assertEquals(acceptedDraftTokens, 6, "pinned MTP acceptance suite should accept six draft tokens");
    assert(
      acceptedDraftTokens / proposedDraftTokens >= 0.5,
      `MTP acceptance rate too low: accepted=${acceptedDraftTokens}, proposed=${proposedDraftTokens}`,
    );

    const targetStats = mtpSession.cacheStats().executionProviderStats;
    assert(
      typeof targetStats.webgpuTokenIdInputTokens === "number" && targetStats.webgpuTokenIdInputTokens > 0,
      `MTP target WebGPU token-id path was not observed: ${JSON.stringify(targetStats)}`,
    );
    const assistantStats = assistantSession.cacheStats().executionProviderStats;
    assert(
      typeof assistantStats.wasmMtpAssistantRuns === "number" && assistantStats.wasmMtpAssistantRuns > 0,
      `MTP assistant WASM runner was not observed: ${JSON.stringify(assistantStats)}`,
    );
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

async function generateGreedyTokens(
  targetReader: unknown,
  tokenizer: {
    tokenize: (text: string, options?: { addBos?: boolean }) => number[];
  },
  applyChatGenerationPrompt: () => string,
  applyChatTemplate: (messages: readonly { role: "user"; content: string }[], options: { addGenerationPrompt: false }) => string,
  createChatSession: (...args: any[]) => any,
  createWebGpuProvider: (...args: any[]) => any,
  prefill: (...args: any[]) => Promise<{ nextTokenId: number }>,
  decode: (...args: any[]) => Promise<{ nextTokenId: number }>,
  prompt: string,
  maxNewTokens: number,
): Promise<number[]> {
  const session = createMeasuredChatSession(targetReader, createChatSession, createWebGpuProvider);
  const state = session.createInferenceState();
  const text = applyChatTemplate([{ role: "user", content: prompt }], { addGenerationPrompt: false }) +
    applyChatGenerationPrompt();
  const tokenIds = tokenizer.tokenize(text, { addBos: true });
  let result = await prefill(session, state, tokenIds, { logitsTopK: 1 });
  const tokens: number[] = [];
  for (let index = 0; index < maxNewTokens; index += 1) {
    const tokenId = result.nextTokenId;
    tokens.push(tokenId);
    if (index < maxNewTokens - 1) {
      result = await decode(session, state, tokenId, { logitsTopK: 1 });
    }
  }
  return tokens;
}

async function measureMtpAcceptance(
  targetReader: unknown,
  tokenizer: {
    tokenize: (text: string, options?: { addBos?: boolean }) => number[];
  },
  applyChatGenerationPrompt: () => string,
  applyChatTemplate: (messages: readonly { role: "user"; content: string }[], options: { addGenerationPrompt: false }) => string,
  createChatSession: (...args: any[]) => any,
  createWebGpuProvider: (...args: any[]) => any,
  createDeterministicRng: (seed: number) => () => number,
  prefillState: (...args: any[]) => Promise<void>,
  prefillMtpTarget: (...args: any[]) => Promise<{ firstTokenId: number; context: unknown }>,
  proposeMtpDraft: (...args: any[]) => Promise<{ draftTokenIds: number[] }>,
  verifyMtpDraft: (...args: any[]) => Promise<{ targetTokenIds: number[] }>,
  finalizeMtpVerification: (...args: any[]) => unknown,
  acceptGreedyMtpDraft: (...args: any[]) => {
    acceptedDraftLength: number;
    nextTokenId: number;
    committedLength: number;
  },
  assistantSession: { manifest: unknown },
  sampling: { seed: number; logitsTopK: number },
  prompt: string,
  maxNewTokens: number,
): Promise<{ tokens: number[]; proposed: number; accepted: number }> {
  const session = createMeasuredChatSession(targetReader, createChatSession, createWebGpuProvider);
  const state = session.createInferenceState();
  const rng = createDeterministicRng(sampling.seed);
  const userPrompt = applyChatTemplate([{ role: "user", content: prompt }], { addGenerationPrompt: false });
  const userTokenIds = tokenizer.tokenize(userPrompt, { addBos: true });
  await prefillState(session, state, userTokenIds, {
    positions: Int32Array.from({ length: userTokenIds.length }, (_, index) => index),
  });
  const generationPromptTokenIds = tokenizer.tokenize(applyChatGenerationPrompt(), { addBos: false });
  const generationPromptStart = state.nextPosition;
  const promptPrefill = await prefillMtpTarget(session, state, generationPromptTokenIds, {
    positions: Int32Array.from(
      { length: generationPromptTokenIds.length },
      (_, index) => generationPromptStart + index,
    ),
    logitsTopK: sampling.logitsTopK,
    assistantManifest: assistantSession.manifest,
  });

  let pendingTokenId = promptPrefill.firstTokenId;
  let context = promptPrefill.context;
  let emitted = 0;
  let proposed = 0;
  let accepted = 0;
  const tokens: number[] = [];
  while (emitted < maxNewTokens) {
    const remainingOutput = maxNewTokens - emitted;
    const remainingContext = state.contextLength - state.nextPosition;
    const draftBudget = Math.max(0, Math.min(1, remainingOutput - 1, remainingContext - 1));
    const proposal = draftBudget > 0
      ? await proposeMtpDraft(
        session,
        { assistantSession, numSpeculativeTokens: draftBudget },
        pendingTokenId,
        context,
        sampling,
        rng,
      )
      : { draftTokenIds: [] };
    proposed += proposal.draftTokenIds.length;
    const verification = await verifyMtpDraft(session, state, [pendingTokenId, ...proposal.draftTokenIds], {
      logitsTopK: sampling.logitsTopK,
      assistantManifest: assistantSession.manifest,
    });
    const acceptance = acceptGreedyMtpDraft(proposal.draftTokenIds, verification.targetTokenIds);
    accepted += acceptance.acceptedDraftLength;
    context = finalizeMtpVerification(session, state, verification, acceptance.committedLength);
    for (const tokenId of [pendingTokenId, ...proposal.draftTokenIds.slice(0, acceptance.acceptedDraftLength)]) {
      if (emitted >= maxNewTokens) {
        break;
      }
      tokens.push(tokenId);
      emitted += 1;
    }
    pendingTokenId = acceptance.nextTokenId;
  }
  return { tokens, proposed, accepted };
}

async function measureMtpSamplingStep(
  targetReader: unknown,
  tokenizer: {
    tokenize: (text: string, options?: { addBos?: boolean }) => number[];
  },
  applyChatGenerationPrompt: () => string,
  applyChatTemplate: (messages: readonly { role: "user"; content: string }[], options: { addGenerationPrompt: false }) => string,
  createChatSession: (...args: any[]) => any,
  createWebGpuProvider: (...args: any[]) => any,
  createDeterministicRng: (seed: number) => () => number,
  prefillState: (...args: any[]) => Promise<void>,
  prefillMtpTarget: (...args: any[]) => Promise<{ firstTokenDistribution: unknown; context: unknown }>,
  proposeMtpDraft: (...args: any[]) => Promise<{ draftTokenIds: number[]; draftDistributions: unknown[] }>,
  verifyMtpDraft: (...args: any[]) => Promise<{ targetDistributions: Array<{ tokens: unknown[] }> }>,
  finalizeMtpVerification: (...args: any[]) => unknown,
  acceptMtpDraft: (...args: any[]) => { committedLength: number },
  sampleMtpTokenDistribution: (...args: any[]) => number,
  assistantSession: { manifest: unknown },
  sampling: { seed: number; logitsTopK: number },
  prompt: string,
): Promise<{ targetDistributionWidths: number[]; acceptance: { committedLength: number } }> {
  const session = createMeasuredChatSession(targetReader, createChatSession, createWebGpuProvider);
  const state = session.createInferenceState();
  const rng = createDeterministicRng(sampling.seed);
  const userPrompt = applyChatTemplate([{ role: "user", content: prompt }], { addGenerationPrompt: false });
  const userTokenIds = tokenizer.tokenize(userPrompt, { addBos: true });
  await prefillState(session, state, userTokenIds, {
    positions: Int32Array.from({ length: userTokenIds.length }, (_, index) => index),
  });
  const generationPromptTokenIds = tokenizer.tokenize(applyChatGenerationPrompt(), { addBos: false });
  const generationPromptStart = state.nextPosition;
  const promptPrefill = await prefillMtpTarget(session, state, generationPromptTokenIds, {
    positions: Int32Array.from(
      { length: generationPromptTokenIds.length },
      (_, index) => generationPromptStart + index,
    ),
    logitsTopK: sampling.logitsTopK,
    assistantManifest: assistantSession.manifest,
  });
  const pendingTokenId = sampleMtpTokenDistribution(promptPrefill.firstTokenDistribution, sampling, rng);
  const proposal = await proposeMtpDraft(
    session,
    { assistantSession, numSpeculativeTokens: 1 },
    pendingTokenId,
    promptPrefill.context,
    sampling,
    rng,
  );
  const verification = await verifyMtpDraft(session, state, [pendingTokenId, ...proposal.draftTokenIds], {
    logitsTopK: sampling.logitsTopK,
    assistantManifest: assistantSession.manifest,
  });
  const acceptance = acceptMtpDraft(proposal, verification, sampling, rng);
  finalizeMtpVerification(session, state, verification, acceptance.committedLength);
  return {
    targetDistributionWidths: verification.targetDistributions.map((distribution) => distribution.tokens.length),
    acceptance,
  };
}

function createMeasuredChatSession(
  targetReader: unknown,
  createChatSession: (...args: any[]) => any,
  createWebGpuProvider: (...args: any[]) => any,
) {
  return createChatSession(targetReader, {
    maxContextLength: 256,
    providers: [createWebGpuProvider({
      memoryLimitBytes: Number(
        Deno.env.get("HELIOPAUSE_WEBGPU_MEMORY_LIMIT_BYTES") ?? DEFAULT_WEBGPU_MEMORY_LIMIT_BYTES,
      ),
      prefillChunkSize: 16,
    })],
  });
}

async function assertPinnedModel(model: typeof TARGET_MODEL | typeof ASSISTANT_MODEL): Promise<void> {
  const stat = await Deno.stat(model.path);
  assert(stat.isFile, `${model.path} is not a file`);
  assertEquals(stat.size, model.size, `${model.path} has wrong byte size`);
  assertEquals(await sha256File(model.path), model.sha256, `${model.path} has wrong SHA-256`);
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

function installWebGpuCapture(): { device?: CapturedDevice } {
  const originalGpu = navigator.gpu;
  const capture: { device?: CapturedDevice } = {};
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
            const device = await adapter.requestDevice(...deviceArgs);
            capture.device = device as CapturedDevice;
            capture.device.pushErrorScope?.("validation");
            return device;
          },
        };
      },
    },
  });
  return capture;
}

function formatWebGpuError(error: unknown): string {
  if (error && typeof error === "object") {
    const record = error as Record<string, unknown>;
    return JSON.stringify({
      name: record.name,
      message: record.message,
      stack: record.stack,
    });
  }
  return String(error);
}

function assertNaturalSentence(text: string): void {
  const trimmed = text.trim();
  assert(trimmed.length > 0, "generated output is empty");
  assertEquals(trimmed, EXPECTED_TEXT, `generated output is not the expected natural sentence`);
  assert(/[A-Za-z]/.test(trimmed), `generated output has no alphabetic characters: ${JSON.stringify(text)}`);
  assert(/\bblue\b/i.test(trimmed), `generated output does not mention blue: ${JSON.stringify(text)}`);
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
