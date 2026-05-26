import {
  WasmSegmentRunner,
  type WasmSegmentRunnerOptions,
} from "./segment-runner";

export function wasmSegmentRunner(
  options: WasmSegmentRunnerOptions,
): WasmSegmentRunner {
  return new WasmSegmentRunner(options);
}
