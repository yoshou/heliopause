import type {
  AudioEncodeResult,
  AudioFeatures,
  AudioSession,
} from "../../audio";
import type {
  GgmlTypeName,
} from "../../gguf";
import type {
  AudioManifest,
} from "../../model";
import {
  dequantizeRow,
} from "../../quant";
import {
  tensorByteLength,
} from "../../tensor-reader";
import type {
  AudioEncoderRunner,
  AudioPreprocessRunner,
} from "../audio-runner";
import {
  rmsNorm,
} from "./kernels";

export const referenceAudioPreprocessRunner: AudioPreprocessRunner = {
  provider: "reference",
  run: async (session, audio, audioPreprocess) =>
    audioPreprocess(audio, session.manifest),
};

export const referenceAudioEncoderRunner: AudioEncoderRunner = {
  provider: "reference",
  run: (session, features, options) => runReferenceAudioEncoder(session, features, options),
};

export async function runReferenceAudioEncoder(
  session: AudioSession,
  features: AudioFeatures,
  options: { signal?: AbortSignal } = {},
): Promise<AudioEncodeResult> {
  requireReferenceProvider(session);
  throwIfAborted(options.signal);
  const manifest = session.manifest;
  if (features.featureSize !== manifest.featureSize) {
    throw new Error(`Audio feature size mismatch: ${features.featureSize}`);
  }
  if (manifest.blockCount !== 0) {
    throw new Error("Reference audio encoder currently supports projection-only audio projectors.");
  }

  let hidden = await subsampleConvProjection(session, features);
  hidden = await matMulAudioWeight(session, "a.pre_encode.out.weight", hidden);
  if (session.hasTensor("a.pre_encode.out.bias")) {
    hidden = addBiasRows(hidden, await session.readF32Tensor("a.pre_encode.out.bias"));
  }
  hidden = rmsNormRows(
    hidden,
    await optionalTensor(session, "mm.a.soft_emb_norm.weight", manifest.outputProjectionDim),
    session.epsilon,
  );
  hidden = await matMulAudioWeight(session, "mm.a.input_projection.weight", hidden);
  return {
    hidden,
    tokenCount: hidden.length / manifest.projectionDim,
    durationMs: features.durationMs,
  };
}

function requireReferenceProvider(session: AudioSession): void {
  if (!session.executionProvider("reference")) {
    throw new Error("Audio reference execution requires an enabled reference provider.");
  }
}

async function subsampleConvProjection(
  session: AudioSession,
  features: AudioFeatures,
): Promise<Float32Array> {
  const first = await conv2dSubsampleLayer(session, features.values, features.attentionMask, features.frameCount, features.featureSize, 1, 128, "a.conv1d.0");
  const second = await conv2dSubsampleLayer(session, first.values, first.mask, first.time, first.frequency, 128, 32, "a.conv1d.1");
  const flattened = flattenChannelsLast(second.values, second.time, second.frequency, 32);
  const projected = await matMulAudioWeight(session, "a.input_projection.weight", flattened);
  return session.hasTensor("a.input_projection.bias")
    ? addBiasRows(projected, await session.readF32Tensor("a.input_projection.bias"))
    : projected;
}

async function conv2dSubsampleLayer(
  session: AudioSession,
  input: Float32Array,
  mask: Uint8Array,
  time: number,
  frequency: number,
  inChannels: number,
  outChannels: number,
  prefix: string,
): Promise<{ values: Float32Array; mask: Uint8Array; time: number; frequency: number }> {
  const weight = await session.readF32Tensor(`${prefix}.weight`);
  const bias = session.hasTensor(`${prefix}.bias`) ? await session.readF32Tensor(`${prefix}.bias`) : undefined;
  const norm = await session.readF32Tensor(`${prefix}.norm.weight`);
  const outTime = Math.floor((time + 1) / 2);
  const outFrequency = Math.floor((frequency + 1) / 2);
  const output = new Float32Array(outTime * outChannels * outFrequency);

  for (let tOut = 0; tOut < outTime; tOut += 1) {
    for (let fOut = 0; fOut < outFrequency; fOut += 1) {
      const channelValues = new Float32Array(outChannels);
      for (let outChannel = 0; outChannel < outChannels; outChannel += 1) {
        let sum = 0;
        for (let kt = 0; kt < 3; kt += 1) {
          const tIn = tOut * 2 + kt - 1;
          if (tIn < 0 || tIn >= time || (mask[tIn] ?? 0) === 0) {
            continue;
          }
          for (let kf = 0; kf < 3; kf += 1) {
            const fIn = fOut * 2 + kf - 1;
            if (fIn < 0 || fIn >= frequency) {
              continue;
            }
            for (let inChannel = 0; inChannel < inChannels; inChannel += 1) {
              const inputValue = input[(tIn * inChannels + inChannel) * frequency + fIn] ?? 0;
              const weightValue = weight[kf + kt * 3 + inChannel * 9 + outChannel * 9 * inChannels] ?? 0;
              sum = Math.fround(sum + Math.fround(inputValue * weightValue));
            }
          }
        }
        channelValues[outChannel] = bias ? Math.fround(sum + (bias[outChannel] ?? 0)) : sum;
      }
      layerNormInPlace(channelValues, norm, session.epsilon);
      for (let outChannel = 0; outChannel < outChannels; outChannel += 1) {
        output[(tOut * outChannels + outChannel) * outFrequency + fOut] = Math.max(0, channelValues[outChannel] ?? 0);
      }
    }
  }

  return {
    values: output,
    mask: downsampleMaskByTwo(mask, outTime),
    time: outTime,
    frequency: outFrequency,
  };
}

async function matMulAudioWeight(
  session: AudioSession,
  weightName: string,
  inputColumns: Float32Array,
): Promise<Float32Array> {
  const tensor = session.getTensor(weightName);
  const inputSize = tensor.dimensions[0] ?? 0;
  const rowCount = tensor.dimensions[1] ?? 0;
  const columnCount = inputColumns.length / inputSize;
  if (!Number.isInteger(columnCount)) {
    throw new Error(`${weightName} input shape mismatch`);
  }
  const weights = await readWeightRows(session, weightName, tensor.type, inputSize, rowCount);
  const output = new Float32Array(rowCount * columnCount);
  for (let column = 0; column < columnCount; column += 1) {
    const inputOffset = column * inputSize;
    const outputOffset = column * rowCount;
    for (let row = 0; row < rowCount; row += 1) {
      const weight = weights[row];
      if (!weight) {
        throw new Error(`${weightName} row ${row} is missing`);
      }
      let sum = 0;
      for (let index = 0; index < inputSize; index += 1) {
        sum = Math.fround(sum + Math.fround((weight[index] ?? 0) * (inputColumns[inputOffset + index] ?? 0)));
      }
      output[outputOffset + row] = sum;
    }
  }
  return output;
}

async function readWeightRows(
  session: AudioSession,
  weightName: string,
  type: GgmlTypeName,
  inputSize: number,
  rowCount: number,
): Promise<Float32Array[]> {
  if (type === "F32") {
    const values = await session.readF32Tensor(weightName);
    return Array.from({ length: rowCount }, (_, row) => values.slice(row * inputSize, (row + 1) * inputSize));
  }

  const bytes = await session.readWeightBytes(weightName);
  const rowByteLength = Number(tensorByteLength({
    name: weightName,
    dimensions: [inputSize, 1],
    type,
    typeId: 0,
    offset: 0n,
    dataOffset: 0n,
  }));
  return Array.from({ length: rowCount }, (_, row) => {
    return dequantizeRow(type, bytes.subarray(row * rowByteLength, (row + 1) * rowByteLength), inputSize);
  });
}

function flattenChannelsLast(
  input: Float32Array,
  timeCount: number,
  frequencyCount: number,
  channelCount: number,
): Float32Array {
  const output = new Float32Array(timeCount * frequencyCount * channelCount);
  for (let time = 0; time < timeCount; time += 1) {
    for (let frequency = 0; frequency < frequencyCount; frequency += 1) {
      for (let channel = 0; channel < channelCount; channel += 1) {
        output[time * frequencyCount * channelCount + frequency * channelCount + channel] =
          input[(time * channelCount + channel) * frequencyCount + frequency] ?? 0;
      }
    }
  }
  return output;
}

function addBiasRows(input: Float32Array, bias: Float32Array): Float32Array {
  const output = new Float32Array(input);
  for (let token = 0; token < input.length / bias.length; token += 1) {
    const offset = token * bias.length;
    for (let index = 0; index < bias.length; index += 1) {
      output[offset + index] = Math.fround((output[offset + index] ?? 0) + (bias[index] ?? 0));
    }
  }
  return output;
}

function rmsNormRows(input: Float32Array, weight: Float32Array, epsilon: number): Float32Array {
  if (input.length % weight.length !== 0) {
    throw new Error(`Audio RMSNorm row shape mismatch: input=${input.length} weight=${weight.length}`);
  }
  const output = new Float32Array(input.length);
  for (let row = 0; row < input.length / weight.length; row += 1) {
    const offset = row * weight.length;
    output.set(rmsNorm(input.slice(offset, offset + weight.length), weight, epsilon), offset);
  }
  return output;
}

function layerNormInPlace(values: Float32Array, weight: Float32Array, epsilon: number): void {
  let mean = 0;
  for (const value of values) {
    mean += value;
  }
  mean /= values.length;

  let variance = 0;
  for (const value of values) {
    const centered = value - mean;
    variance += centered * centered;
  }
  const scale = 1 / Math.sqrt(variance / values.length + epsilon);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = Math.fround(((values[index] ?? 0) - mean) * scale * (weight[index] ?? 1));
  }
}

async function optionalTensor(
  session: AudioSession,
  name: string,
  fallbackLength: number,
): Promise<Float32Array> {
  if (session.hasTensor(name)) {
    return session.readF32Tensor(name);
  }
  const fallback = new Float32Array(fallbackLength);
  fallback.fill(1);
  return fallback;
}

function downsampleMaskByTwo(mask: Uint8Array, outTime: number): Uint8Array {
  const output = new Uint8Array(outTime);
  for (let index = 0; index < outTime; index += 1) {
    output[index] = mask[Math.min(index * 2, mask.length - 1)] ?? 0;
  }
  return output;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new DOMException("Generation was aborted.", "AbortError");
  }
}
