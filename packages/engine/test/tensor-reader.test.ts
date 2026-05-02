import assert from "node:assert/strict";
import test from "node:test";

import {
  GgufTensorReader,
  type GgufMetadata,
} from "../src/index.ts";

test("tensor reader shares concurrent full tensor reads", async () => {
  const bytes = new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer);
  let readCount = 0;
  let releaseRead: (() => void) | undefined;
  const reader = tensorReader(bytes, {
    async read(offset, length) {
      readCount += 1;
      assert.equal(offset, 0n);
      assert.equal(length, bytes.byteLength);
      await new Promise<void>((resolve) => {
        releaseRead = resolve;
      });
      return bytes.slice();
    },
  });

  const left = reader.readTensorBytes("weight");
  const right = reader.readTensorBytes("weight");
  releaseRead?.();
  const [leftBytes, rightBytes] = await Promise.all([left, right]);

  assert.equal(readCount, 1);
  assert.deepEqual(Array.from(leftBytes), Array.from(bytes));
  assert.deepEqual(Array.from(rightBytes), Array.from(bytes));
  assert.equal(reader.ioStats().inflightHits, 1);
});

test("tensor reader coalesces nearby ranges", async () => {
  const bytes = Uint8Array.from({ length: 64 }, (_value, index) => index);
  const reads: Array<{ offset: bigint; length: number }> = [];
  const reader = tensorReader(bytes, {
    async read(offset, length) {
      reads.push({ offset, length });
      return bytes.subarray(Number(offset), Number(offset) + length).slice();
    },
  });
  const tensor = reader.getTensor("weight");

  const ranges = await reader.readTensorRangesCoalesced([
    { tensor, offset: 0n, length: 8 },
    { tensor, offset: 8n, length: 8 },
    { tensor, offset: 24n, length: 4 },
  ], {
    maxGapBytes: 8,
    maxReadBytes: 64,
  });

  assert.deepEqual(reads, [{ offset: 0n, length: 28 }]);
  assert.deepEqual(ranges.map((range) => Array.from(range)), [
    Array.from(bytes.subarray(0, 8)),
    Array.from(bytes.subarray(8, 16)),
    Array.from(bytes.subarray(24, 28)),
  ]);
  assert.equal(reader.ioStats().coalescedReads, 1);
});

test("coalesced range results do not share transferred buffers", async () => {
  const bytes = Uint8Array.from({ length: 32 }, (_value, index) => index + 1);
  const reader = tensorReader(bytes);
  const tensor = reader.getTensor("weight");

  const ranges = await reader.readTensorRangesCoalesced([
    { tensor, offset: 0n, length: 8 },
    { tensor, offset: 8n, length: 8 },
  ]);

  structuredClone(ranges[0]?.buffer, { transfer: [ranges[0]!.buffer] });

  assert.equal(ranges[0]?.byteLength, 0);
  assert.deepEqual(Array.from(ranges[1] ?? []), Array.from(bytes.subarray(8, 16)));
});

function tensorReader(
  bytes: Uint8Array,
  reader: { read(offset: bigint, length: number): Promise<Uint8Array> } = {
    async read(offset, length) {
      return bytes.subarray(Number(offset), Number(offset) + length).slice();
    },
  },
): GgufTensorReader {
  const gguf: GgufMetadata = {
    version: 3,
    tensorCount: 1,
    metadataCount: 0,
    dataStart: 0n,
    metadata: {},
    tensors: [{
      name: "weight",
      dimensions: [bytes.byteLength / 4],
      type: "F32",
      typeId: 0,
      offset: 0n,
      dataOffset: 0n,
    }],
  };
  return new GgufTensorReader(gguf, reader);
}
