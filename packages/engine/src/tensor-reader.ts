import type { GgufByteReader, GgufMetadata, GgufTensorInfo } from "./gguf";

export type TensorByteRange = {
  tensor: GgufTensorInfo;
  offset: bigint;
  length: number;
};

export type GgufTensorReadKind = "tensor" | "range";
export type GgufTensorReadSource = "direct" | "inflight" | "coalesced";

export type GgufTensorReadTraceEvent = {
  kind: GgufTensorReadKind;
  tensorName: string;
  offset: bigint;
  length: number;
  durationMs: number;
  queuedMs?: number;
  readMs?: number;
  coalescedRangeCount?: number;
  source?: GgufTensorReadSource;
};

export type GgufTensorReadTrace = (event: GgufTensorReadTraceEvent) => void;

export type GgufTensorReaderOptions = {
  onRead?: GgufTensorReadTrace;
};

export type GgufTensorRangeCoalesceOptions = {
  maxGapBytes?: number;
  maxReadBytes?: number;
};

export type GgufTensorReaderIoStats = {
  inflightHits: number;
  coalescedReads: number;
  readMs: number;
};

export class GgufTensorReader {
  private readonly gguf: GgufMetadata;
  private readonly reader: GgufByteReader;
  private readonly tensorsByName: Map<string, GgufTensorInfo>;
  private onRead?: GgufTensorReadTrace;
  private readonly inflightTensorReads = new Map<string, Promise<Uint8Array>>();
  private readonly inflightRangeReads = new Map<string, Promise<Uint8Array>>();
  private inflightHits = 0;
  private coalescedReads = 0;
  private readMs = 0;

  constructor(
    gguf: GgufMetadata,
    reader: GgufByteReader,
    options: GgufTensorReaderOptions = {},
  ) {
    this.gguf = gguf;
    this.reader = reader;
    this.tensorsByName = new Map(gguf.tensors.map((tensor) => [tensor.name, tensor]));
    this.onRead = options.onRead;
  }

  setReadTrace(onRead: GgufTensorReadTrace | undefined): void {
    this.onRead = onRead;
  }

  getTensor(name: string): GgufTensorInfo {
    const tensor = this.tensorsByName.get(name);
    if (!tensor) {
      throw new Error(`Unknown tensor: ${name}`);
    }
    return tensor;
  }

  async readTensorBytes(name: string): Promise<Uint8Array> {
    const tensor = this.getTensor(name);
    const length = tensorByteLength(tensor);
    const start = nowMs();
    const cached = this.inflightTensorReads.get(name);
    if (cached) {
      this.inflightHits += 1;
      const bytes = await cached;
      const durationMs = nowMs() - start;
      this.onRead?.({
        kind: "tensor",
        tensorName: name,
        offset: tensor.dataOffset,
        length,
        durationMs,
        queuedMs: durationMs,
        readMs: 0,
        source: "inflight",
      });
      return bytes;
    }

    const read = this.readBytes(tensor.dataOffset, length);
    this.inflightTensorReads.set(name, read);
    const bytes = await read.finally(() => {
      this.inflightTensorReads.delete(name);
    });
    const durationMs = nowMs() - start;
    this.onRead?.({
      kind: "tensor",
      tensorName: name,
      offset: tensor.dataOffset,
      length,
      durationMs,
      queuedMs: 0,
      readMs: durationMs,
      source: "direct",
    });
    return bytes;
  }

  async readTensorRange(range: TensorByteRange): Promise<Uint8Array> {
    const absoluteOffset = range.tensor.dataOffset + range.offset;
    const key = rangeKey(range.tensor.name, absoluteOffset, range.length);
    const start = nowMs();
    const cached = this.inflightRangeReads.get(key);
    if (cached) {
      this.inflightHits += 1;
      const bytes = await cached;
      const durationMs = nowMs() - start;
      this.onRead?.({
        kind: "range",
        tensorName: range.tensor.name,
        offset: absoluteOffset,
        length: range.length,
        durationMs,
        queuedMs: durationMs,
        readMs: 0,
        source: "inflight",
      });
      return bytes;
    }

    const read = this.readBytes(absoluteOffset, range.length);
    this.inflightRangeReads.set(key, read);
    const bytes = await read.finally(() => {
      this.inflightRangeReads.delete(key);
    });
    const durationMs = nowMs() - start;
    this.onRead?.({
      kind: "range",
      tensorName: range.tensor.name,
      offset: absoluteOffset,
      length: range.length,
      durationMs,
      queuedMs: 0,
      readMs: durationMs,
      source: "direct",
    });
    return bytes;
  }

  async readTensorRangesCoalesced(
    ranges: readonly TensorByteRange[],
    options: GgufTensorRangeCoalesceOptions = {},
  ): Promise<Uint8Array[]> {
    if (ranges.length === 0) {
      return [];
    }
    const maxGapBytes = BigInt(Math.max(0, Math.floor(options.maxGapBytes ?? 1024 * 1024)));
    const maxReadBytes = Math.max(1, Math.floor(options.maxReadBytes ?? 256 * 1024 * 1024));
    const entries = ranges.map((range, index) => ({
      index,
      range,
      absoluteOffset: range.tensor.dataOffset + range.offset,
      endOffset: range.tensor.dataOffset + range.offset + BigInt(range.length),
    })).sort((left, right) => compareBigInt(left.absoluteOffset, right.absoluteOffset));

    const output = new Array<Uint8Array>(ranges.length);
    let group: typeof entries = [];
    const flush = async () => {
      if (group.length === 0) {
        return;
      }
      const groupStart = group[0]!.absoluteOffset;
      const groupEnd = group.reduce((end, item) => item.endOffset > end ? item.endOffset : end, group[0]!.endOffset);
      const readLength = Number(groupEnd - groupStart);
      const startedAt = nowMs();
      const bytes = await this.readBytes(groupStart, readLength);
      const readMs = nowMs() - startedAt;
      if (group.length > 1) {
        this.coalescedReads += 1;
      }
      for (const item of group) {
        const relativeStart = Number(item.absoluteOffset - groupStart);
        const result = bytes.slice(relativeStart, relativeStart + item.range.length);
        output[item.index] = result;
        this.onRead?.({
          kind: "range",
          tensorName: item.range.tensor.name,
          offset: item.absoluteOffset,
          length: item.range.length,
          durationMs: readMs,
          queuedMs: 0,
          readMs,
          coalescedRangeCount: group.length,
          source: group.length > 1 ? "coalesced" : "direct",
        });
      }
      group = [];
    };

    for (const entry of entries) {
      if (group.length === 0) {
        group.push(entry);
        continue;
      }
      const groupStart = group[0]!.absoluteOffset;
      const groupEnd = group.reduce((end, item) => item.endOffset > end ? item.endOffset : end, group[0]!.endOffset);
      const gap = entry.absoluteOffset > groupEnd ? entry.absoluteOffset - groupEnd : 0n;
      const mergedEnd = entry.endOffset > groupEnd ? entry.endOffset : groupEnd;
      const mergedLength = Number(mergedEnd - groupStart);
      if (gap <= maxGapBytes && mergedLength <= maxReadBytes) {
        group.push(entry);
      } else {
        await flush();
        group.push(entry);
      }
    }
    await flush();
    return output;
  }

  ioStats(): GgufTensorReaderIoStats {
    return {
      inflightHits: this.inflightHits,
      coalescedReads: this.coalescedReads,
      readMs: this.readMs,
    };
  }

  sourceBlob(): Blob | undefined {
    return this.reader.sourceBlob;
  }

  get metadata(): GgufMetadata {
    return this.gguf;
  }

  private async readBytes(offset: bigint, length: number): Promise<Uint8Array> {
    const start = nowMs();
    const bytes = await this.reader.read(offset, length);
    this.readMs += nowMs() - start;
    return bytes;
  }
}

function nowMs(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function rangeKey(tensorName: string, offset: bigint, length: number): string {
  return `${tensorName}:${offset.toString()}:${length}`;
}

function compareBigInt(left: bigint, right: bigint): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

export function tensorByteLength(tensor: GgufTensorInfo): number {
  const elements = tensor.dimensions.reduce((product, dimension) => product * dimension, 1);
  const typeInfo = ggmlTypeStorage[tensor.type];
  if (!typeInfo) {
    throw new Error(`Unsupported tensor storage type: ${tensor.type}`);
  }
  if (elements % typeInfo.blockSize !== 0) {
    throw new Error(
      `Tensor ${tensor.name} element count ${elements} is not divisible by block size ${typeInfo.blockSize}`,
    );
  }
  return (elements / typeInfo.blockSize) * typeInfo.typeSize;
}

export const ggmlTypeStorage: Record<string, { blockSize: number; typeSize: number }> = {
  F32: { blockSize: 1, typeSize: 4 },
  F16: { blockSize: 1, typeSize: 2 },
  Q8_0: { blockSize: 32, typeSize: 34 },
  Q4_K: { blockSize: 256, typeSize: 144 },
  Q5_K: { blockSize: 256, typeSize: 176 },
  Q6_K: { blockSize: 256, typeSize: 210 },
  IQ4_XS: { blockSize: 256, typeSize: 136 },
};
