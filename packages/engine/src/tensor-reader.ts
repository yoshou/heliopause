import type { GgufByteReader, GgufMetadata, GgufTensorInfo } from "./gguf";

export type TensorByteRange = {
  tensor: GgufTensorInfo;
  offset: bigint;
  length: number;
};

export class GgufTensorReader {
  private readonly gguf: GgufMetadata;
  private readonly reader: GgufByteReader;
  private readonly tensorsByName: Map<string, GgufTensorInfo>;

  constructor(
    gguf: GgufMetadata,
    reader: GgufByteReader,
  ) {
    this.gguf = gguf;
    this.reader = reader;
    this.tensorsByName = new Map(gguf.tensors.map((tensor) => [tensor.name, tensor]));
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
    return this.reader.read(tensor.dataOffset, tensorByteLength(tensor));
  }

  async readTensorRange(range: TensorByteRange): Promise<Uint8Array> {
    return this.reader.read(range.tensor.dataOffset + range.offset, range.length);
  }

  get metadata(): GgufMetadata {
    return this.gguf;
  }
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
