# Heliopause

Heliopause is a local LLM runtime experiment for Gemma4 E4B GGUF models. It includes a TypeScript engine, a desktop chat app, and a browser benchmark app for WebGPU and WASM execution paths.

## Packages

- `packages/engine` - core model loading, tensor reading, tokenization, CPU/WASM runners, and WebGPU runners.
- `apps/desktop` - React + Tauri desktop chat app.
- `apps/bench` - browser benchmark app for engine kernels and model operations.

## Supported Models

- `models/gemma4/unsloth/gemma-4-E4B-it-Q4_K_M.gguf`

## Requirements

- Node.js
- pnpm
- Rust, for the Tauri app and WASM kernel builds

## Getting Started

```sh
pnpm install
```

Run the desktop app in development mode:

```sh
pnpm dev
```

Run the benchmark app:

```sh
pnpm dev:bench
```

## Common Commands

```sh
pnpm typecheck
pnpm build
pnpm build:bench
pnpm --filter @heliopause/engine test
pnpm --filter @heliopause/engine build:wasm
```

## Performance Work

### CPU/WASM

- Use WASM SIMD kernels for quantized matmul and other model operations.
- Keep quantized weights in WASM-resident handles when the full cache profile is enabled.
- Batch related projection matmuls to share input handling across multiple weights.
- Split resident matmul rows across a worker pool when parallel resident matmul is enabled.
- Prefetch layer and output weights for the threaded resident path.

### WebGPU

- Use CPU-prefix / GPU-suffix execution to fit WebGPU memory limits.
- Select suffix layers from the end of the model using weight and cache size estimates.
- Expand the suffix when selected layers depend on shared KV source layers.
- Enforce a 12 GiB WebGPU memory cap with `GpuMemoryArena`.
- Keep selected weights and layer state resident in GPU buffers.
- Keep suffix activations on GPU within each segment-token call.
- Run top-k on GPU and read back token candidates instead of full logits.

## Notes

The desktop app loads a local GGUF model file from your machine. WebGPU support depends on the browser, GPU, driver, and platform.

## License

MIT
