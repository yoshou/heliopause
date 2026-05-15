import {
  buildVisionManifest,
  type VisionManifest,
} from "./model";
import type {
  CacheStats,
  ExecutionProviderConfig,
  ExecutionProviderStats,
} from "./runtime";
import {
  runCpuVisionEncoder,
} from "./runner/cpu/vision-runner";
import {
  runWebGpuVisionEncoder,
} from "./runner/webgpu/vision-execution-provider";
import {
  GgufTensorReader,
} from "./tensor-reader";

export type VisionPixelValues = {
  values: Float32Array;
  width: number;
  height: number;
};

export type VisionResize = {
  width: number;
  height: number;
  outputTokenCount: number;
};

export type VisionEncodeResult = {
  hidden: Float32Array;
  tokenCount: number;
  width: number;
  height: number;
};

export type VisionSessionOptions = {
  maxWeightCacheBytes?: number;
  executionProviders?: readonly ExecutionProviderConfig[];
};

export class VisionSession {
  readonly tensorReader: GgufTensorReader;
  readonly manifest: VisionManifest;
  readonly epsilon: number;
  readonly executionProviders: readonly ExecutionProviderConfig[];

  private readonly maxWeightCacheBytes: number;
  private readonly f32TensorCache = new Map<string, Float32Array>();
  private readonly weightBytesCache = new Map<string, Uint8Array>();
  private readonly executionProviderStatsProviders = new Map<string, () => ExecutionProviderStats>();
  private readonly disposeCallbacks = new Set<() => void>();
  private weightCacheBytes = 0;
  private weightCacheHits = 0;
  private weightCacheMisses = 0;
  private weightCacheEvictions = 0;

  constructor(tensorReader: GgufTensorReader, options: VisionSessionOptions = {}) {
    this.tensorReader = tensorReader;
    this.manifest = buildVisionManifest(tensorReader.metadata);
    this.epsilon = this.manifest.layerNormEpsilon;
    this.maxWeightCacheBytes = options.maxWeightCacheBytes ?? 256 * 1024 * 1024;
    this.executionProviders = (options.executionProviders ?? [{ name: "cpu" }]).map((provider) => ({
      name: provider.name,
      options: provider.options ? { ...provider.options } : undefined,
    }));
  }

  getTensor(name: string) {
    return this.tensorReader.getTensor(name);
  }

  hasTensor(name: string): boolean {
    return this.tensorReader.metadata.tensors.some((tensor) => tensor.name === name);
  }

  async readF32Tensor(name: string): Promise<Float32Array> {
    const cached = this.f32TensorCache.get(name);
    if (cached) {
      return cached;
    }
    const tensor = this.getTensor(name);
    if (tensor.type !== "F32") {
      throw new Error(`${name} must be F32, got ${tensor.type}`);
    }
    const bytes = await this.tensorReader.readTensorBytes(name);
    const value = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
    this.f32TensorCache.set(name, value);
    return value;
  }

  async readWeightBytes(name: string): Promise<Uint8Array> {
    const cached = this.weightBytesCache.get(name);
    if (cached) {
      this.weightCacheHits += 1;
      this.weightBytesCache.delete(name);
      this.weightBytesCache.set(name, cached);
      return cached;
    }
    this.weightCacheMisses += 1;
    const bytes = await this.tensorReader.readTensorBytes(name);
    if (bytes.byteLength <= this.maxWeightCacheBytes) {
      this.weightBytesCache.set(name, bytes);
      this.weightCacheBytes += bytes.byteLength;
      this.evictWeightBytes();
    }
    return bytes;
  }

  cacheStats(): CacheStats {
    return {
      f32TensorCount: this.f32TensorCache.size,
      weightTensorCount: this.weightBytesCache.size,
      weightCacheBytes: this.weightCacheBytes,
      maxWeightCacheBytes: this.maxWeightCacheBytes,
      weightCacheHits: this.weightCacheHits,
      weightCacheMisses: this.weightCacheMisses,
      weightCacheEvictions: this.weightCacheEvictions,
      embeddingRowCount: 0,
      executionProviderStats: this.executionProviderStats(),
    };
  }

  executionProvider(name: string): ExecutionProviderConfig | undefined {
    return this.executionProviders.find((provider) => provider.name === name);
  }

  setExecutionProviderStatsProvider(
    provider: (() => ExecutionProviderStats) | undefined,
    name = "default",
  ): void {
    if (!provider) {
      this.executionProviderStatsProviders.delete(name);
      return;
    }
    this.executionProviderStatsProviders.set(name, provider);
  }

  addDisposeCallback(callback: () => void): void {
    this.disposeCallbacks.add(callback);
  }

  dispose(): void {
    for (const callback of this.disposeCallbacks) {
      callback();
    }
    this.disposeCallbacks.clear();
    this.executionProviderStatsProviders.clear();
    this.f32TensorCache.clear();
    this.weightBytesCache.clear();
    this.weightCacheBytes = 0;
  }

  private executionProviderStats(): ExecutionProviderStats {
    const stats: ExecutionProviderStats = {};
    for (const provider of this.executionProviderStatsProviders.values()) {
      Object.assign(stats, provider());
    }
    return stats;
  }

  private evictWeightBytes(): void {
    while (this.weightCacheBytes > this.maxWeightCacheBytes) {
      const oldest = this.weightBytesCache.entries().next().value as [string, Uint8Array] | undefined;
      if (!oldest) {
        this.weightCacheBytes = 0;
        return;
      }
      this.weightBytesCache.delete(oldest[0]);
      this.weightCacheBytes -= oldest[1].byteLength;
      this.weightCacheEvictions += 1;
    }
  }
}

export function createVisionSession(
  tensorReader: GgufTensorReader,
  options: VisionSessionOptions = {},
): VisionSession {
  return new VisionSession(tensorReader, options);
}

export function calculateVisionResize(
  manifest: Pick<VisionManifest, "patchSize" | "spatialMergeSize" | "imageMinTokens" | "imageMaxTokens">,
  sourceWidth: number,
  sourceHeight: number,
): VisionResize {
  const alignSize = manifest.patchSize * manifest.spatialMergeSize;
  const minPixels = manifest.imageMinTokens * alignSize * alignSize;
  const maxPixels = manifest.imageMaxTokens * alignSize * alignSize;
  const width = Math.max(1, Math.floor(sourceWidth));
  const height = Math.max(1, Math.floor(sourceHeight));

  let resizedHeight = Math.max(alignSize, roundByFactor(height, alignSize));
  let resizedWidth = Math.max(alignSize, roundByFactor(width, alignSize));
  if (resizedHeight * resizedWidth > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    resizedHeight = Math.max(alignSize, floorByFactor(height / beta, alignSize));
    resizedWidth = Math.max(alignSize, floorByFactor(width / beta, alignSize));
  } else if (resizedHeight * resizedWidth < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    resizedHeight = ceilByFactor(height * beta, alignSize);
    resizedWidth = ceilByFactor(width * beta, alignSize);
  }

  return {
    width: resizedWidth,
    height: resizedHeight,
    outputTokenCount: (resizedWidth / alignSize) * (resizedHeight / alignSize),
  };
}

export async function preprocessVisionImageFile(
  file: Blob,
  manifest: VisionManifest,
): Promise<VisionPixelValues> {
  const bitmap = await createImageBitmap(file);
  try {
    const rgba = imageBitmapToRgba(bitmap);
    const resize = calculateVisionResize(manifest, bitmap.width, bitmap.height);
    const resized = resizeRgbaBilinear(rgba, bitmap.width, bitmap.height, resize.width, resize.height);
    const values = new Float32Array(resize.width * resize.height * 3);
    for (let pixel = 0; pixel < resize.width * resize.height; pixel += 1) {
      for (let channel = 0; channel < 3; channel += 1) {
        values[pixel * 3 + channel] =
          ((resized[pixel * 4 + channel] ?? 0) / 255 - (manifest.imageMean[channel] ?? 0)) /
          (manifest.imageStd[channel] ?? 1);
      }
    }
    return { values, width: resize.width, height: resize.height };
  } finally {
    bitmap.close();
  }
}

export async function runVisionEncoder(
  session: VisionSession,
  pixels: VisionPixelValues,
): Promise<VisionEncodeResult> {
  if (session.executionProvider("webgpu")) {
    const webgpu = await runWebGpuVisionEncoder(session, pixels);
    if (webgpu) {
      return webgpu;
    }
  }
  return runCpuVisionEncoder(session, pixels);
}

function imageBitmapToRgba(bitmap: ImageBitmap): Uint8ClampedArray {
  const canvas = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(bitmap.width, bitmap.height)
    : document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Could not create image decode canvas.");
  }
  context.drawImage(bitmap, 0, 0);
  return context.getImageData(0, 0, bitmap.width, bitmap.height).data;
}

function resizeRgbaBilinear(
  src: Uint8ClampedArray,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): Uint8ClampedArray {
  const output = new Uint8ClampedArray(targetWidth * targetHeight * 4);
  const xRatio = targetWidth > 1 ? (sourceWidth - 1) / (targetWidth - 1) : 0;
  const yRatio = targetHeight > 1 ? (sourceHeight - 1) / (targetHeight - 1) : 0;
  for (let y = 0; y < targetHeight; y += 1) {
    const py = y * yRatio;
    const y0 = Math.min(Math.trunc(py), sourceHeight - 1);
    const y1 = Math.min(y0 + 1, sourceHeight - 1);
    const yf = py - y0;
    for (let x = 0; x < targetWidth; x += 1) {
      const px = x * xRatio;
      const x0 = Math.min(Math.trunc(px), sourceWidth - 1);
      const x1 = Math.min(x0 + 1, sourceWidth - 1);
      const xf = px - x0;
      const dst = (y * targetWidth + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        const top = lerp(
          src[(y0 * sourceWidth + x0) * 4 + channel] ?? 0,
          src[(y0 * sourceWidth + x1) * 4 + channel] ?? 0,
          xf,
        );
        const bottom = lerp(
          src[(y1 * sourceWidth + x0) * 4 + channel] ?? 0,
          src[(y1 * sourceWidth + x1) * 4 + channel] ?? 0,
          xf,
        );
        output[dst + channel] = Math.trunc(lerp(top, bottom, yf));
      }
      output[dst + 3] = 255;
    }
  }
  return output;
}

function roundByFactor(value: number, factor: number): number {
  return Math.round(value / factor) * factor;
}

function ceilByFactor(value: number, factor: number): number {
  return Math.ceil(value / factor) * factor;
}

function floorByFactor(value: number, factor: number): number {
  return Math.floor(value / factor) * factor;
}

function lerp(left: number, right: number, amount: number): number {
  return left + (right - left) * amount;
}
