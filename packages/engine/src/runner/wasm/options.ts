export type WasmProviderOptions = {
  projectionBatching?: boolean;
  residentWeightCache?: boolean;
  parallelResidentMatmul?: boolean;
  parallelMatmulMinRows?: number;
  threadPoolSize?: number | "auto";
  ioPrefetch?: boolean;
  ioPrefetchConcurrency?: number | "auto";
  ioCoalesceMaxGapBytes?: number;
  ioCoalesceMaxReadBytes?: number;
  ioWorkerBlobRead?: boolean;
};

export type WasmConfiguredProvider = {
  readonly name: "wasm";
  readonly options: Readonly<WasmProviderOptions>;
};
