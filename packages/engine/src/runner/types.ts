export type GqaAttentionOptions = {
  headSize: number;
  queryHeadCount: number;
  keyValueHeadCount: number;
  tokenCount: number;
  keyValueTokenCount?: number;
  scale: number;
  causal?: boolean;
  mask?: Float32Array;
  valueLayout?: "token-head-dim" | "dim-head-token";
  quantizeQueryForScore?: "f16";
};
