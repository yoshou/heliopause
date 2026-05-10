import type { Gemma4ModelManifest } from "../model";
import type { ForwardRunnerNode } from "./graph";
import {
  CpuLayerSegmentNode,
  EmbeddingCpuNode,
  OutputCpuNode,
} from "./cpu/nodes";
import {
  GpuToCpuHiddenTransferNode,
  WebGpuLayerSegmentNode,
} from "./webgpu/nodes";

export function buildGemma4CpuOnlyForwardGraph(
  manifest: Gemma4ModelManifest,
  tokenIds: readonly number[],
  options: { includeOutput?: boolean; outputTopK?: number } = {},
): ForwardRunnerNode[] {
  const nodes: ForwardRunnerNode[] = [new EmbeddingCpuNode(tokenIds)];
  let currentId = "embedding";
  const cpu = maybeCpuLayerSegmentNode(0, manifest.blockCount, currentId);
  if (cpu) {
    nodes.push(cpu);
    currentId = cpu.id;
  }
  if (options.includeOutput ?? true) {
    nodes.push(new OutputCpuNode(currentId, options.outputTopK));
  }
  return nodes;
}

export function buildGemma4ManualSegmentForwardGraph(
  manifest: Gemma4ModelManifest,
  tokenIds: readonly number[],
  segment: { startLayer: number; endLayerExclusive: number },
  options: { includeOutput?: boolean; outputTopK?: number } = {},
): ForwardRunnerNode[] {
  validateLayerSegment(manifest, segment);
  const nodes: ForwardRunnerNode[] = [new EmbeddingCpuNode(tokenIds)];
  let currentId = "embedding";
  const prefix = maybeCpuLayerSegmentNode(0, segment.startLayer, currentId);
  if (prefix) {
    nodes.push(prefix);
    currentId = prefix.id;
  }
  const gpu = new WebGpuLayerSegmentNode(segment.startLayer, segment.endLayerExclusive, currentId);
  nodes.push(gpu);
  currentId = gpu.id;
  const transfer = new GpuToCpuHiddenTransferNode(currentId);
  nodes.push(transfer);
  currentId = transfer.id;
  const suffix = maybeCpuLayerSegmentNode(segment.endLayerExclusive, manifest.blockCount, currentId);
  if (suffix) {
    nodes.push(suffix);
    currentId = suffix.id;
  }
  if (options.includeOutput ?? true) {
    nodes.push(new OutputCpuNode(currentId, options.outputTopK));
  }
  return nodes;
}

function maybeCpuLayerSegmentNode(
  start: number,
  end: number,
  inputId: string,
): CpuLayerSegmentNode | undefined {
  return end > start ? new CpuLayerSegmentNode(start, end, inputId) : undefined;
}

function validateLayerSegment(
  manifest: Gemma4ModelManifest,
  segment: { startLayer: number; endLayerExclusive: number },
): void {
  if (
    !Number.isInteger(segment.startLayer) ||
    !Number.isInteger(segment.endLayerExclusive) ||
    segment.startLayer < 0 ||
    segment.endLayerExclusive <= segment.startLayer ||
    segment.endLayerExclusive > manifest.blockCount
  ) {
    throw new Error(
      `Invalid WebGPU layer segment: ${segment.startLayer}..${segment.endLayerExclusive}`,
    );
  }
}
