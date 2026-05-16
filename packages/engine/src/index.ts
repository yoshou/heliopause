export {
  checkWebGpuSupport,
  planRunnerPlacement,
  webGpuPlanningProvider,
  WebGpuSegmentRunner,
  type WebGpuSupport,
  type WebGpuRuntimeStats,
  type WebGpuSegmentRunnerOptions,
} from "./runner/webgpu/index";

export {
  checkWasmSupport,
  type WasmSupport,
} from "./runner/cpu/wasm-kernels";

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
  auditTensorCoverage,
  buildAudioManifest,
  buildModelManifest,
  buildVisionManifest,
  isAudioGguf,
  isVisionGguf,
  type ExpectedTensor,
  type AudioManifest,
  type LayerKind,
  type ModelManifest,
  type VisionManifest,
  type TensorCoverageAudit,
} from "./model";

export {
  decode,
  prefill,
  prefillPreparedHidden,
  type DecodeOptions,
  type DecodeResult,
  type PreparedHiddenPrefillOptions,
  type PrefillOptions,
  type PrefillResult,
} from "./forward";

export {
  calculateVisionResize,
  createVisionSession,
  preprocessVisionImageFile,
  runVisionPreprocessor,
  runVisionEncoder,
  VisionSession,
  type VisionEncodeResult,
  type VisionPixelValues,
  type VisionPreprocessOptions,
  type VisionResize,
  type VisionSessionOptions,
} from "./vision";

export {
  createAudioSession,
  preprocessAudioPcm,
  runAudioPreprocessor,
  runAudioEncoder,
  AudioSession,
  type AudioEncodeResult,
  type AudioFeatures,
  type AudioPcmInput,
  type AudioPreprocessOptions,
  type AudioSessionOptions,
} from "./audio";

export {
  createInferenceState,
  createModelSession,
  cloneInferenceState,
  estimateWeightCacheBytes,
  resolvePreprocessProviders,
  ModelSession,
  type CacheStats,
  type ExecutionProviderConfig,
  type ForwardTrace,
  type FullAttentionCache,
  type InferenceState,
  type ModelInput,
  type ModelSessionOptions,
  type TimingEvent,
  type TimingPhase,
  type TimingSink,
} from "./runtime";

export { CpuSegmentRunner } from "./runner/cpu/index";
export type {
  CpuHiddenResult,
  CpuSegmentRunnerOptions,
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
  buildTokenizer,
  type Tokenizer,
} from "./tokenizer";

export {
  applyChatTemplate,
  applyChatGenerationPrompt,
  createFileGgufTensorReader,
  createChatSession,
  DEFAULT_SYSTEM_PROMPT,
  generatePreparedAudioChatTurn,
  generatePreparedImageChatTurn,
  generateChatTurn,
  generateChatCompletion,
  getGgufModelName,
  prefillChatMessages,
  stripThinking,
  type ChatMessage,
  type FileGgufTensorReaderOptions,
  type ChatCompletionChunk,
  type ChatCompletionOptions,
  type ChatPrefillOptions,
  type ChatTemplateOptions,
  type ChatTurnOptions,
  type ChatTurnResult,
  type PreparedAudioInput,
  type PreparedImageInput,
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
  buildCpuOnlyForwardGraph,
  buildManualSegmentForwardGraph,
} from "./runner/nodes";

export {
  auditRunnerPlacementCopies,
  planProviderPlacement,
  type RunnerCopyAuditObservation,
  type RunnerCopyAuditResult,
  type RunnerLayerPlacement,
  type RunnerPlacementPlan,
  type RunnerPlanningOptions,
  type RunnerPlanningProvider,
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
