import {
  type GgufMetadata,
  type GgufMetadataValue,
  getMetadataNumber,
  getMetadataString,
} from "./gguf";

export type Qwen35Tokenizer = {
  tokenize(input: string): number[];
  tokenToId(token: string): number | undefined;
};

const QWEN35_PATTERN =
  /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

export function buildQwen35Tokenizer(gguf: GgufMetadata): Qwen35Tokenizer {
  const model = getMetadataString(gguf.metadata, "tokenizer.ggml.model");
  const pre = getMetadataString(gguf.metadata, "tokenizer.ggml.pre");
  const addBos = gguf.metadata["tokenizer.ggml.add_bos_token"] === true;
  const bosId = getMetadataNumber(gguf.metadata, "tokenizer.ggml.bos_token_id");

  if (model !== "gpt2" || pre !== "qwen35") {
    throw new Error(`Unsupported tokenizer metadata: model=${model ?? "missing"} pre=${pre ?? "missing"}`);
  }

  const tokens = requiredStringArray(gguf.metadata["tokenizer.ggml.tokens"], "tokenizer.ggml.tokens");
  const merges = requiredStringArray(gguf.metadata["tokenizer.ggml.merges"], "tokenizer.ggml.merges");
  const tokenToId = new Map(tokens.map((token, index) => [token, index]));
  const ranks = new Map<string, number>();

  merges.forEach((merge, index) => {
    const split = merge.split(" ");
    if (split.length === 2) {
      ranks.set(pairKey(split[0] ?? "", split[1] ?? ""), index);
    }
  });

  const byteEncoder = buildByteEncoder();

  return {
    tokenize(input) {
      const ids: number[] = [];
      if (addBos) {
        if (bosId === undefined) {
          throw new Error("tokenizer.ggml.add_bos_token is true but bos id is missing");
        }
        ids.push(bosId);
      }

      for (const piece of input.match(QWEN35_PATTERN) ?? []) {
        const encoded = Array.from(new TextEncoder().encode(piece), (byte) => byteEncoder[byte]).join("");
        for (const token of applyBpe(encoded, ranks)) {
          const id = tokenToId.get(token);
          if (id === undefined) {
            throw new Error(`BPE produced unknown token: ${token}`);
          }
          ids.push(id);
        }
      }

      return ids;
    },
    tokenToId(token) {
      return tokenToId.get(token);
    },
  };
}

function applyBpe(input: string, ranks: Map<string, number>): string[] {
  let parts = Array.from(input);
  if (parts.length <= 1) {
    return parts;
  }

  while (true) {
    let bestRank = Number.POSITIVE_INFINITY;
    let bestIndex = -1;

    for (let index = 0; index < parts.length - 1; index += 1) {
      const rank = ranks.get(pairKey(parts[index] ?? "", parts[index + 1] ?? ""));
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestIndex = index;
      }
    }

    if (bestIndex < 0) {
      return parts;
    }

    parts = [
      ...parts.slice(0, bestIndex),
      `${parts[bestIndex] ?? ""}${parts[bestIndex + 1] ?? ""}`,
      ...parts.slice(bestIndex + 2),
    ];
  }
}

function requiredStringArray(value: GgufMetadataValue | undefined, key: string): string[] {
  if (
    typeof value !== "object" ||
    value === null ||
    !("sample" in value) ||
    value.truncated ||
    value.sample.length !== value.length
  ) {
    throw new Error(`${key} was not parsed completely; parse GGUF with a large enough maxArraySample`);
  }

  const strings = value.sample.filter((item): item is string => typeof item === "string");
  if (strings.length !== value.length) {
    throw new Error(`${key} contains non-string values`);
  }
  return strings;
}

function pairKey(left: string, right: string): string {
  return `${left}\u0000${right}`;
}

function buildByteEncoder(): string[] {
  const bs: number[] = [];
  for (let value = "!".charCodeAt(0); value <= "~".charCodeAt(0); value += 1) bs.push(value);
  for (let value = "¡".charCodeAt(0); value <= "¬".charCodeAt(0); value += 1) bs.push(value);
  for (let value = "®".charCodeAt(0); value <= "ÿ".charCodeAt(0); value += 1) bs.push(value);

  const cs = [...bs];
  let extra = 0;
  for (let byte = 0; byte < 256; byte += 1) {
    if (!bs.includes(byte)) {
      bs.push(byte);
      cs.push(256 + extra);
      extra += 1;
    }
  }

  const result: string[] = Array.from({ length: 256 });
  for (let index = 0; index < bs.length; index += 1) {
    result[bs[index] ?? 0] = String.fromCodePoint(cs[index] ?? 0);
  }
  return result;
}
