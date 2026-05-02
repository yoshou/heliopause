# Heliopause

Heliopause is a local LLM runtime experiment for Qwen 3.5 style GGUF models. It includes a TypeScript engine, a desktop chat app, and a browser benchmark app for WebGPU and WASM execution paths.

## Packages

- `packages/engine` - core model loading, tensor reading, tokenization, CPU/WASM runners, and WebGPU runners.
- `apps/desktop` - React + Tauri desktop chat app.
- `apps/bench` - browser benchmark app for engine kernels and model operations.

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

## Notes

The desktop app loads a local GGUF model file from your machine. WebGPU support depends on the browser, GPU, driver, and platform.

## License

MIT
