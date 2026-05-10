# Heliopause

Heliopause is a local LLM runtime experiment for Gemma4 E4B GGUF models. It includes a TypeScript engine, a desktop chat app, and a browser benchmark app. The current Gemma4 inference path is CPU/WASM; WebGPU execution is disabled until its Gemma4 kernels pass logits parity.

## Packages

- `packages/engine` - core model loading, tensor reading, tokenization, CPU/WASM runners, and WebGPU planning.
- `apps/desktop` - React + Tauri desktop chat app.
- `apps/bench` - browser benchmark app for engine kernels and model operations.

## Supported Models

- `gemma-4-E4B-it-Q4_K_M.gguf`

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

- WebGPU placement planning remains available for memory sizing and copy-audit work.
- Gemma4 WebGPU execution is intentionally disabled until native Gemma4 attention, per-layer input, GEGLU, post-norm, and logits-softcap kernels pass logits parity.
- Do not enable WebGPU generation for Gemma4 until that parity work is complete.

## Notes

The desktop app loads a local GGUF model file from your machine. WebGPU availability still depends on the browser, GPU, driver, and platform, but Gemma4 generation currently uses the CPU/WASM path.

## License

MIT
