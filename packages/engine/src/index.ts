export {
  checkWebGpuSupport,
  planQwen35RunnerPlacement,
  qwen35WebGpuPlanningProvider,
  Qwen35WebGpuSegmentRunner,
  type WebGpuSupport,
  type Qwen35WebGpuSegmentRunnerOptions,
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
  auditQwen35TensorCoverage,
  buildQwen35Manifest,
  type ExpectedTensor,
  type Qwen35LayerKind,
  type Qwen35ModelManifest,
  type TensorCoverageAudit,
} from "./model";

export {
  decodeQwen35,
  prefillQwen35,
  type DecodeOptions,
  type DecodeResult,
  type PrefillOptions,
  type PrefillResult,
} from "./forward";

export {
  createQwen35InferenceState,
  createQwen35ModelSession,
  cloneQwen35InferenceState,
  estimateWeightCacheBytes,
  Qwen35ModelSession,
  type CacheStats,
  type ExecutionProviderConfig,
  type ForwardTrace,
  type Qwen35FullAttentionCache,
  type Qwen35InferenceState,
  type Qwen35ModelInput,
  type Qwen35ModelSessionOptions,
  type Qwen35RecurrentCache,
  type TimingEvent,
  type TimingPhase,
  type TimingSink,
} from "./runtime";

export { Qwen35CpuSegmentRunner } from "./runner/cpu/index";
export type {
  Qwen35CpuHiddenResult,
  Qwen35CpuSegmentRunnerOptions,
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
  buildQwen35Tokenizer,
  type Qwen35Tokenizer,
} from "./tokenizer";

export {
  applyQwen35ChatTemplate,
  applyQwen35ChatGenerationPrompt,
  createFileGgufTensorReader,
  createQwen35ChatSession,
  DEFAULT_QWEN35_SYSTEM_PROMPT,
  generateQwen35ChatTurn,
  generateQwen35ChatCompletion,
  getGgufModelName,
  prefillQwen35ChatMessages,
  stripQwen35Thinking,
  type ChatMessage,
  type FileGgufTensorReaderOptions,
  type Qwen35ChatCompletionChunk,
  type Qwen35ChatCompletionOptions,
  type Qwen35ChatPrefillOptions,
  type Qwen35ChatTemplateOptions,
  type Qwen35ChatTurnOptions,
  type Qwen35ChatTurnResult,
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
  buildQwen35CpuOnlyForwardGraph,
  buildQwen35ManualSegmentForwardGraph,
} from "./runner/nodes";

export {
  auditQwen35RunnerPlacementCopies,
  planQwen35ProviderPlacement,
  type Qwen35RunnerCopyAuditObservation,
  type Qwen35RunnerCopyAuditResult,
  type Qwen35RunnerLayerPlacement,
  type Qwen35RunnerPlacementPlan,
  type Qwen35RunnerPlanningOptions,
  type Qwen35RunnerPlanningProvider,
} from "./runner/planning";

export {
  gatedDeltaNet,
  l2NormRows,
  gqaAttention,
  matMulRows,
  maxTensorDiff,
  ropeMultiMropeNeox,
  rmsNorm,
  sigmoid,
  silu,
  softplus,
  ssmConv1d,
  type GatedDeltaNetOptions,
  type GqaAttentionOptions,
  type RopeMultiOptions,
} from "./ops";
