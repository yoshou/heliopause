import type {
  ForwardTrace,
  ModelSession,
  OutputResult,
} from "../../runtime";
import { forwardOutput } from "./layers";
import {
  WasmSegmentRunner,
  type WasmSegmentRunnerOptions,
} from "./segment-runner";

export function wasmSegmentRunner(
  options: WasmSegmentRunnerOptions,
): WasmSegmentRunner {
  return new WasmSegmentRunner(options);
}

export function wasmOutput(
  session: ModelSession,
  hidden: Float32Array,
  options: { topK: number; trace?: ForwardTrace },
): Promise<OutputResult> {
  return forwardOutput(session, hidden, options);
}
