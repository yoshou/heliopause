export {
  parseGguf,
  serializeGgufMetadata,
  type GgufArraySummary,
  type GgufByteReader,
  type GgufMetadata,
  type GgufMetadataValue,
  type GgufTensorInfo,
  type GgmlTypeName,
} from "./gguf";

export {
  auditMtpAssistantTensorCoverage,
  auditTensorCoverage,
  buildAudioManifest,
  buildMtpAssistantManifest,
  buildModelManifest,
  buildVisionManifest,
  isAudioGguf,
  isMtpAssistantGguf,
  isVisionGguf,
  type AudioManifest,
  type ExpectedTensor,
  type LayerKind,
  type ModelManifest,
  type MtpAssistantManifest,
  type TensorCoverageAudit,
  type VisionManifest,
} from "./model";

export {
  GgufTensorReader,
  ggmlTypeStorage,
  tensorByteLength,
  type GgufTensorRangeCoalesceOptions,
  type GgufTensorReaderIoStats,
  type GgufTensorReaderOptions,
  type GgufTensorReadKind,
  type GgufTensorReadSource,
  type GgufTensorReadTrace,
  type GgufTensorReadTraceEvent,
  type TensorByteRange,
} from "./tensor-reader";

export {
  dequantizeF16,
  dequantizeIQ4_XS,
  dequantizeQ4_K,
  dequantizeQ5_K,
  dequantizeQ6_K,
  dequantizeQ8_0,
  dequantizeRow,
  float16ToFloat32,
  float32ToFloat16,
  quantizeQ8_0,
  quantizeQ8_K,
  vecDotIQ4_XS_Q8_K,
  vecDotQ4_K_Q8_K,
  vecDotQ5_K_Q8_K,
  vecDotQ6_K_Q8_K,
  vecDotQ8_0_Q8_0,
  type QuantizedQ8_0,
  type QuantizedQ8K,
} from "./quant";

export {
  buildTokenizer,
  type Tokenizer,
} from "./tokenizer";

export {
  cloneInferenceState,
  createInferenceState,
  createModelSession,
  disposeInferenceState,
  estimateWeightCacheBytes,
  ModelSession,
  type CacheStats,
  type ForwardTrace,
  type FullAttentionCache,
  type InferenceState,
  type ModelSessionOptions,
  type TimingEvent,
  type TimingPhase,
  type TimingSink,
} from "./runtime";

export {
  decode,
  prefill,
  prefillPreparedHiddenState,
  prefillState,
  type DecodeOptions,
  type NextTokenResult,
  type PrefillOptions,
  type PrefillStateOptions,
  type PreparedHiddenPrefillStateOptions,
} from "./forward";

export {
  DEFAULT_SAMPLING_SEED,
  DEFAULT_GENERATION_CONFIG,
  createDeterministicRng,
  resolveGenerationSamplingOptions,
  sampleNextToken,
  type GenerationSamplingOptions,
  type ResolvedGenerationSamplingOptions,
  type TopTokenCandidate,
} from "./generation";

export {
  applyChatGenerationPrompt,
  applyChatTemplate,
  closeChatModelTurn,
  createChatSession,
  createFileGgufTensorReader,
  DEFAULT_SYSTEM_PROMPT,
  generateChatCompletion,
  generateChatTurn,
  generatePreparedAudioChatTurn,
  generatePreparedImageChatTurn,
  getGgufModelName,
  prefillChatMessages,
  stripThinking,
  type ChatCompletionChunk,
  type ChatCompletionOptions,
  type ChatMessage,
  type ChatPrefillOptions,
  type ChatTemplateOptions,
  type ChatToolCall,
  type ChatToolDeclaration,
  type ChatTurnInput,
  type ChatTurnOptions,
  type ChatTurnResult,
  type FileGgufTensorReaderOptions,
  type PreparedAudioInput,
  type PreparedImageInput,
} from "./chat";

export {
  AudioSession,
  createAudioSession,
  preprocessAudioPcm,
  runAudioEncoder,
  runAudioPreprocessor,
  type AudioEncodeResult,
  type AudioFeatures,
  type AudioPcmInput,
  type AudioPreprocessOptions,
  type AudioSessionOptions,
} from "./audio";

export {
  calculateVisionResize,
  createVisionSession,
  preprocessVisionImageFile,
  runVisionEncoder,
  runVisionPreprocessor,
  VisionSession,
  type VisionEncodeResult,
  type VisionPixelValues,
  type VisionPreprocessOptions,
  type VisionResize,
  type VisionSessionOptions,
} from "./vision";

export {
  createMtpAssistantSession,
  MtpAssistantSession,
  runMtpAssistant,
  type MtpAssistantForwardIntermediates,
  type MtpAssistantRunInput,
  type MtpAssistantRunResult,
  type MtpAssistantRunner,
  type MtpAssistantRunners,
  type MtpAssistantSessionOptions,
  type MtpTargetKvLayerView,
  type MtpTargetKvView,
} from "./mtp-assistant";

export {
  createReferenceProvider,
  ReferenceSegmentRunner,
  type ReferenceHiddenResult,
  type ReferenceSegmentRunnerOptions,
} from "./runner/reference/index";

export {
  createWasmProvider,
  WasmSegmentRunner,
  type WasmHiddenResult,
  type WasmProviderOptions,
  type WasmSegmentRunnerOptions,
} from "./runner/wasm/index";

export {
  checkWasmSupport,
  type WasmSupport,
} from "./runner/wasm/wasm-kernels";

export {
  checkWebGpuSupport,
  createWebGpuProvider,
  WebGpuSegmentRunner,
  type WebGpuRuntimeStats,
  type WebGpuProviderOptions,
  type WebGpuSegmentRunnerOptions,
  type WebGpuSupport,
} from "./runner/webgpu/index";

export type {
  ModelGraphNodeFactory,
  ModelRunner,
} from "./runner/model-runner";

export type {
  AudioRunnerProvider,
  ModelRunnerProvider,
  MultimodalRunnerProvider,
  MtpAssistantRunnerProvider,
  RunnerProvider,
  VisionRunnerProvider,
} from "./runner/provider";

export type {
  SegmentHiddenResult,
  SegmentRunner,
  SegmentRunnerProvider,
  SegmentRunOptions,
} from "./runner/segment-runner";

export {
  auditRunnerPlacementCopies,
  planModelPlacement,
  type LayerResourceRequirement,
  type ProviderResourceRequirements,
  type ResourceBudget,
  type RunnerCopyAuditObservation,
  type RunnerCopyAuditResult,
  type RunnerCopyExpectations,
  type RunnerLayerPlacement,
  type RunnerNodePlacement,
  type RunnerPlacementPlan,
  type RunnerResourceUsage,
  type RunnerSegmentPlacement,
} from "./runner/planning";

export {
  ForwardGraphExecutor,
  topologicalSortForwardNodes,
  type ForwardGraphContext,
  type ForwardGraphExecutionResult,
  type ForwardRunnerBackend,
  type ForwardRunnerNode,
  type ForwardValue,
} from "./runner/graph";

export {
  cpuRunnerBuffer,
  destroyRunnerBuffer,
  providerRunnerBuffer,
  runnerBufferOwner,
  runnerBufferToCpu,
  type RunnerBuffer,
  type RunnerBufferOwner,
  type RunnerBufferStorage,
} from "./runner/buffer";

export {
  gqaAttention,
  l2NormRows,
  matMulRows,
  maxTensorDiff,
  rmsNorm,
  ropeMultiMropeNeox,
  sigmoid,
  silu,
  softplus,
  type RopeMultiOptions,
} from "./runner/reference/kernels";

export type {
  GqaAttentionOptions,
} from "./runner/types";
