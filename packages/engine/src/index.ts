export {
  checkWebGpuSupport,
  planGemma4RunnerPlacement,
  gemma4WebGpuPlanningProvider,
  Gemma4WebGpuSegmentRunner,
  type WebGpuSupport,
  type Gemma4WebGpuRuntimeStats,
  type Gemma4WebGpuSegmentRunnerOptions,
} from "./runner/webgpu/index";

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
  auditGemma4TensorCoverage,
  buildGemma4AudioManifest,
  buildGemma4Manifest,
  buildGemma4VisionManifest,
  isGemma4AudioGguf,
  isGemma4VisionGguf,
  type ExpectedTensor,
  type Gemma4AudioManifest,
  type Gemma4LayerKind,
  type Gemma4ModelManifest,
  type Gemma4VisionManifest,
  type TensorCoverageAudit,
} from "./model";

export {
  decodeGemma4,
  prefillGemma4,
  prefillGemma4PreparedHidden,
  type DecodeOptions,
  type DecodeResult,
  type PreparedHiddenPrefillOptions,
  type PrefillOptions,
  type PrefillResult,
} from "./forward";

export {
  calculateGemma4VisionResize,
  createGemma4VisionSession,
  preprocessGemma4VisionImageFile,
  runGemma4VisionEncoder,
  Gemma4VisionSession,
  type Gemma4VisionEncodeResult,
  type Gemma4VisionPixelValues,
  type Gemma4VisionResize,
  type Gemma4VisionSessionOptions,
} from "./vision";

export {
  createGemma4AudioSession,
  preprocessGemma4AudioPcm,
  runGemma4AudioEncoder,
  Gemma4AudioSession,
  type Gemma4AudioEncodeResult,
  type Gemma4AudioFeatures,
  type Gemma4AudioPcmInput,
  type Gemma4AudioSessionOptions,
} from "./audio";

export {
  createGemma4InferenceState,
  createGemma4ModelSession,
  cloneGemma4InferenceState,
  estimateWeightCacheBytes,
  Gemma4ModelSession,
  type CacheStats,
  type ExecutionProviderConfig,
  type ForwardTrace,
  type Gemma4FullAttentionCache,
  type Gemma4InferenceState,
  type Gemma4ModelInput,
  type Gemma4ModelSessionOptions,
  type TimingEvent,
  type TimingPhase,
  type TimingSink,
} from "./runtime";

export { Gemma4CpuSegmentRunner } from "./runner/cpu/index";
export type {
  Gemma4CpuHiddenResult,
  Gemma4CpuSegmentRunnerOptions,
} from "./runner/cpu/index";

export {
  GgufTensorReader,
  ggmlTypeStorage,
  tensorByteLength,
  type GgufTensorRangeCoalesceOptions,
  type GgufTensorReaderOptions,
  type GgufTensorReaderIoStats,
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
  float32ToFloat16,
  float16ToFloat32,
  quantizeQ8_0,
  quantizeQ8_K,
  vecDotQ4_K_Q8_K,
  vecDotQ5_K_Q8_K,
  vecDotQ6_K_Q8_K,
  vecDotQ8_0_Q8_0,
  vecDotIQ4_XS_Q8_K,
  type QuantizedQ8_0,
  type QuantizedQ8K,
} from "./quant";

export {
  buildGemma4Tokenizer,
  type Gemma4Tokenizer,
} from "./tokenizer";

export {
  applyGemma4ChatTemplate,
  applyGemma4ChatGenerationPrompt,
  createFileGgufTensorReader,
  createGemma4ChatSession,
  DEFAULT_GEMMA4_SYSTEM_PROMPT,
  generateGemma4PreparedAudioChatTurn,
  generateGemma4PreparedImageChatTurn,
  generateGemma4ChatTurn,
  generateGemma4ChatCompletion,
  getGgufModelName,
  prefillGemma4ChatMessages,
  stripGemma4Thinking,
  type ChatMessage,
  type FileGgufTensorReaderOptions,
  type Gemma4ChatCompletionChunk,
  type Gemma4ChatCompletionOptions,
  type Gemma4ChatPrefillOptions,
  type Gemma4ChatTemplateOptions,
  type Gemma4ChatTurnOptions,
  type Gemma4ChatTurnResult,
  type Gemma4PreparedAudioInput,
  type Gemma4PreparedImageInput,
} from "./chat";

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
  buildGemma4CpuOnlyForwardGraph,
  buildGemma4ManualSegmentForwardGraph,
} from "./runner/nodes";

export {
  auditGemma4RunnerPlacementCopies,
  planGemma4ProviderPlacement,
  type Gemma4RunnerCopyAuditObservation,
  type Gemma4RunnerCopyAuditResult,
  type Gemma4RunnerLayerPlacement,
  type Gemma4RunnerPlacementPlan,
  type Gemma4RunnerPlanningOptions,
  type Gemma4RunnerPlanningProvider,
} from "./runner/planning";

export {
  l2NormRows,
  gqaAttention,
  matMulRows,
  maxTensorDiff,
  ropeMultiMropeNeox,
  rmsNorm,
  sigmoid,
  silu,
  softplus,
  type GqaAttentionOptions,
  type RopeMultiOptions,
} from "./ops";
