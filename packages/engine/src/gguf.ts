export type GgufByteReader = {
  read(offset: bigint, length: number): Promise<Uint8Array>;
};

export type GgufArraySummary = {
  type: GgufMetadataTypeName;
  length: number;
  sample: GgufMetadataValue[];
  truncated: boolean;
};

export type GgufMetadataValue =
  | number
  | boolean
  | string
  | GgufArraySummary;

export type GgufTensorInfo = {
  name: string;
  dimensions: number[];
  type: GgmlTypeName;
  typeId: number;
  offset: bigint;
  dataOffset: bigint;
};

export type GgufMetadata = {
  version: number;
  tensorCount: number;
  metadataCount: number;
  metadata: Record<string, GgufMetadataValue>;
  tensors: GgufTensorInfo[];
  dataStart: bigint;
};

export type ParseGgufOptions = {
  maxArraySample?: number;
};

type GgufMetadataTypeName =
  | "uint8"
  | "int8"
  | "uint16"
  | "int16"
  | "uint32"
  | "int32"
  | "float32"
  | "bool"
  | "string"
  | "array"
  | "uint64"
  | "int64"
  | "float64";

export type GgmlTypeName =
  | "F32"
  | "F16"
  | "Q4_0"
  | "Q4_1"
  | "Q5_0"
  | "Q5_1"
  | "Q8_0"
  | "Q8_1"
  | "Q2_K"
  | "Q3_K"
  | "Q4_K"
  | "Q5_K"
  | "Q6_K"
  | "Q8_K"
  | "IQ2_XXS"
  | "IQ2_XS"
  | "IQ3_XXS"
  | "IQ1_S"
  | "IQ4_NL"
  | "IQ3_S"
  | "IQ2_S"
  | "IQ4_XS"
  | "I8"
  | "I16"
  | "I32"
  | "I64"
  | "F64"
  | "IQ1_M"
  | "BF16"
  | "TQ1_0"
  | "TQ2_0"
  | "MXFP4"
  | "NVFP4"
  | "Q1_0"
  | `UNKNOWN_${number}`;

const GGUF_MAGIC = "GGUF";
const DEFAULT_ALIGNMENT = 32;

const metadataTypeNames: Record<number, GgufMetadataTypeName> = {
  0: "uint8",
  1: "int8",
  2: "uint16",
  3: "int16",
  4: "uint32",
  5: "int32",
  6: "float32",
  7: "bool",
  8: "string",
  9: "array",
  10: "uint64",
  11: "int64",
  12: "float64",
};

const ggmlTypeNames: Record<number, GgmlTypeName> = {
  0: "F32",
  1: "F16",
  2: "Q4_0",
  3: "Q4_1",
  6: "Q5_0",
  7: "Q5_1",
  8: "Q8_0",
  9: "Q8_1",
  10: "Q2_K",
  11: "Q3_K",
  12: "Q4_K",
  13: "Q5_K",
  14: "Q6_K",
  15: "Q8_K",
  16: "IQ2_XXS",
  17: "IQ2_XS",
  18: "IQ3_XXS",
  19: "IQ1_S",
  20: "IQ4_NL",
  21: "IQ3_S",
  22: "IQ2_S",
  23: "IQ4_XS",
  24: "I8",
  25: "I16",
  26: "I32",
  27: "I64",
  28: "F64",
  29: "IQ1_M",
  30: "BF16",
  34: "TQ1_0",
  35: "TQ2_0",
  39: "MXFP4",
  40: "NVFP4",
  41: "Q1_0",
};

export async function parseGguf(
  reader: GgufByteReader,
  options: ParseGgufOptions = {},
): Promise<GgufMetadata> {
  const cursor = new GgufCursor(reader);
  const magic = await cursor.readAscii(4);

  if (magic !== GGUF_MAGIC) {
    throw new Error(`Invalid GGUF magic: expected ${GGUF_MAGIC}, got ${magic}`);
  }

  const version = await cursor.readUint32();
  const tensorCount = bigintToSafeNumber(await cursor.readUint64(), "tensor count");
  const metadataCount = bigintToSafeNumber(await cursor.readUint64(), "metadata count");
  const metadata: Record<string, GgufMetadataValue> = {};

  for (let index = 0; index < metadataCount; index += 1) {
    const key = await cursor.readString();
    const typeId = await cursor.readUint32();
    metadata[key] = await readMetadataValue(cursor, typeId, options.maxArraySample ?? 16);
  }

  const tensors: GgufTensorInfo[] = [];

  for (let index = 0; index < tensorCount; index += 1) {
    const name = await cursor.readString();
    const dimensionCount = await cursor.readUint32();
    const dimensions: number[] = [];

    for (let dimension = 0; dimension < dimensionCount; dimension += 1) {
      dimensions.push(bigintToSafeNumber(await cursor.readUint64(), `dimension for ${name}`));
    }

    const typeId = await cursor.readUint32();
    const offset = await cursor.readUint64();
    tensors.push({
      name,
      dimensions,
      type: ggmlTypeNames[typeId] ?? `UNKNOWN_${typeId}`,
      typeId,
      offset,
      dataOffset: 0n,
    });
  }

  const alignment = getMetadataNumber(metadata, "general.alignment") ?? DEFAULT_ALIGNMENT;
  const dataStart = alignOffset(cursor.offset, BigInt(alignment));

  for (const tensor of tensors) {
    tensor.dataOffset = dataStart + tensor.offset;
  }

  return {
    version,
    tensorCount,
    metadataCount,
    metadata,
    tensors,
    dataStart,
  };
}

export function getMetadataNumber(
  metadata: Record<string, GgufMetadataValue>,
  key: string,
): number | undefined {
  const value = metadata[key];
  return typeof value === "number" ? value : undefined;
}

export function getMetadataString(
  metadata: Record<string, GgufMetadataValue>,
  key: string,
): string | undefined {
  const value = metadata[key];
  return typeof value === "string" ? value : undefined;
}

export function getMetadataNumberArray(
  metadata: Record<string, GgufMetadataValue>,
  key: string,
): number[] | undefined {
  const value = metadata[key];
  if (!isArraySummary(value) || value.truncated) {
    return undefined;
  }

  const numbers = value.sample.filter((item): item is number => typeof item === "number");
  return numbers.length === value.length ? numbers : undefined;
}

export function serializeGgufMetadata(metadata: GgufMetadata): unknown {
  return {
    version: metadata.version,
    tensorCount: metadata.tensorCount,
    metadataCount: metadata.metadataCount,
    dataStart: metadata.dataStart.toString(),
    metadata: metadata.metadata,
    tensors: metadata.tensors.map((tensor) => ({
      ...tensor,
      offset: tensor.offset.toString(),
      dataOffset: tensor.dataOffset.toString(),
    })),
  };
}

function isArraySummary(value: GgufMetadataValue | undefined): value is GgufArraySummary {
  return typeof value === "object" && value !== null && "sample" in value;
}

async function readMetadataValue(
  cursor: GgufCursor,
  typeId: number,
  maxArraySample: number,
): Promise<GgufMetadataValue> {
  switch (typeId) {
    case 0:
      return cursor.readUint8();
    case 1:
      return cursor.readInt8();
    case 2:
      return cursor.readUint16();
    case 3:
      return cursor.readInt16();
    case 4:
      return cursor.readUint32();
    case 5:
      return cursor.readInt32();
    case 6:
      return cursor.readFloat32();
    case 7:
      return (await cursor.readUint8()) !== 0;
    case 8:
      return cursor.readString();
    case 9:
      return readArraySummary(cursor, maxArraySample);
    case 10:
      return bigintToSafeNumber(await cursor.readUint64(), "uint64 metadata value");
    case 11:
      return bigintToSafeNumber(await cursor.readInt64(), "int64 metadata value");
    case 12:
      return cursor.readFloat64();
    default:
      throw new Error(`Unsupported GGUF metadata type id: ${typeId}`);
  }
}

async function readArraySummary(
  cursor: GgufCursor,
  maxArraySample: number,
): Promise<GgufArraySummary> {
  const elementTypeId = await cursor.readUint32();
  const length = bigintToSafeNumber(await cursor.readUint64(), "array length");
  const sampleLength = Math.min(length, maxArraySample);
  const sample: GgufMetadataValue[] = [];

  for (let index = 0; index < length; index += 1) {
    if (index < sampleLength) {
      const value = await readMetadataValue(cursor, elementTypeId, maxArraySample);
      sample.push(value);
    } else {
      await skipMetadataValue(cursor, elementTypeId);
    }
  }

  return {
    type: metadataTypeNames[elementTypeId] ?? "array",
    length,
    sample,
    truncated: length > sampleLength,
  };
}

async function skipMetadataValue(cursor: GgufCursor, typeId: number): Promise<void> {
  switch (typeId) {
    case 0:
    case 1:
    case 7:
      cursor.skipBytes(1);
      return;
    case 2:
    case 3:
      cursor.skipBytes(2);
      return;
    case 4:
    case 5:
    case 6:
      cursor.skipBytes(4);
      return;
    case 8: {
      const length = bigintToSafeNumber(await cursor.readUint64(), "string length");
      cursor.skipBytes(length);
      return;
    }
    case 9: {
      const elementTypeId = await cursor.readUint32();
      const length = bigintToSafeNumber(await cursor.readUint64(), "array length");
      for (let index = 0; index < length; index += 1) {
        await skipMetadataValue(cursor, elementTypeId);
      }
      return;
    }
    case 10:
    case 11:
    case 12:
      cursor.skipBytes(8);
      return;
    default:
      throw new Error(`Unsupported GGUF metadata type id: ${typeId}`);
  }
}

function alignOffset(offset: bigint, alignment: bigint): bigint {
  const remainder = offset % alignment;
  return remainder === 0n ? offset : offset + alignment - remainder;
}

function bigintToSafeNumber(value: bigint, label: string): number {
  if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) {
    throw new Error(`${label} exceeds JavaScript safe integer range: ${value.toString()}`);
  }

  return Number(value);
}

class GgufCursor {
  offset = 0n;
  private readonly reader: GgufByteReader;

  constructor(reader: GgufByteReader) {
    this.reader = reader;
  }

  async readAscii(length: number): Promise<string> {
    const bytes = await this.readBytes(length);
    return String.fromCharCode(...bytes);
  }

  async readString(): Promise<string> {
    const length = bigintToSafeNumber(await this.readUint64(), "string length");
    const bytes = await this.readBytes(length);
    return new TextDecoder().decode(bytes);
  }

  async readUint8(): Promise<number> {
    return this.readNumber(1, (view) => view.getUint8(0));
  }

  async readInt8(): Promise<number> {
    return this.readNumber(1, (view) => view.getInt8(0));
  }

  async readUint16(): Promise<number> {
    return this.readNumber(2, (view) => view.getUint16(0, true));
  }

  async readInt16(): Promise<number> {
    return this.readNumber(2, (view) => view.getInt16(0, true));
  }

  async readUint32(): Promise<number> {
    return this.readNumber(4, (view) => view.getUint32(0, true));
  }

  async readInt32(): Promise<number> {
    return this.readNumber(4, (view) => view.getInt32(0, true));
  }

  async readFloat32(): Promise<number> {
    return this.readNumber(4, (view) => view.getFloat32(0, true));
  }

  async readUint64(): Promise<bigint> {
    return this.readBigInt(8, (view) => view.getBigUint64(0, true));
  }

  async readInt64(): Promise<bigint> {
    return this.readBigInt(8, (view) => view.getBigInt64(0, true));
  }

  async readFloat64(): Promise<number> {
    return this.readNumber(8, (view) => view.getFloat64(0, true));
  }

  private async readNumber(length: number, parse: (view: DataView) => number): Promise<number> {
    const bytes = await this.readBytes(length);
    return parse(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  }

  private async readBigInt(length: number, parse: (view: DataView) => bigint): Promise<bigint> {
    const bytes = await this.readBytes(length);
    return parse(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  }

  private async readBytes(length: number): Promise<Uint8Array> {
    const bytes = await this.reader.read(this.offset, length);
    if (bytes.byteLength !== length) {
      throw new Error(`Unexpected EOF at offset ${this.offset.toString()}`);
    }

    this.offset += BigInt(length);
    return bytes;
  }

  skipBytes(length: number): void {
    this.offset += BigInt(length);
  }
}
