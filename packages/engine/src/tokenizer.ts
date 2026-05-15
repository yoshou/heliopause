import {
  type GgufMetadata,
  type GgufMetadataValue,
  getMetadataNumber,
  getMetadataString,
} from "./gguf";

export type Tokenizer = {
  bosTokenId?: number;
  eosTokenId?: number;
  tokenize(input: string, options?: { addBos?: boolean }): number[];
  detokenize(tokenIds: readonly number[]): string;
  idToToken(id: number): string | undefined;
  tokenToId(token: string): number | undefined;
};

const TOKENIZER_PATTERN =
  /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

const TOKENIZER_SPM_BPE_PATTERN = /[^\n]+|\n+/gu;

type TokenizerKind = "gpt2-byte-bpe" | "spm-bpe";

export function buildTokenizer(gguf: GgufMetadata): Tokenizer {
  const model = getMetadataString(gguf.metadata, "tokenizer.ggml.model");
  const pre = getMetadataString(gguf.metadata, "tokenizer.ggml.pre");
  const addBos = gguf.metadata["tokenizer.ggml.add_bos_token"] === true;
  const bosId = getMetadataNumber(gguf.metadata, "tokenizer.ggml.bos_token_id");
  const eosId = getMetadataNumber(gguf.metadata, "tokenizer.ggml.eos_token_id");

  const tokenizerKind = detectTokenizerKind(model, pre);
  if (!tokenizerKind) {
    throw new Error(`Unsupported tokenizer metadata: model=${model ?? "missing"} pre=${pre ?? "missing"}`);
  }

  const tokens = requiredStringArray(gguf.metadata["tokenizer.ggml.tokens"], "tokenizer.ggml.tokens");
  const merges = requiredStringArray(gguf.metadata["tokenizer.ggml.merges"], "tokenizer.ggml.merges");
  const tokenTypes = optionalNumberArray(gguf.metadata["tokenizer.ggml.token_type"]);
  const tokenToId = new Map(tokens.map((token, index) => [token, index]));
  const specialTokens = tokens
    .filter((token, index) => isSpecialToken(token, tokenTypes?.[index]))
    .sort((left, right) => right.length - left.length);
  const ranks = new Map<string, number>();

  merges.forEach((merge, index) => {
    const pair = parseMerge(merge, tokenizerKind);
    if (pair) {
      ranks.set(pairKey(pair[0], pair[1]), index);
    }
  });

  const byteEncoder = buildByteEncoder();
  const byteDecoder = buildByteDecoder(byteEncoder);

  return {
    bosTokenId: bosId,
    eosTokenId: eosId,
    tokenize(input, options = {}) {
      const ids: number[] = [];
      if (options.addBos ?? addBos) {
        if (bosId === undefined) {
          throw new Error("tokenizer.ggml.add_bos_token is true but bos id is missing");
        }
        ids.push(bosId);
      }

      ids.push(...(
        tokenizerKind === "spm-bpe"
          ? tokenizeSpmBpeText(input, tokenToId, ranks, specialTokens)
          : tokenizeGpt2ByteBpeText(input, tokenToId, ranks, byteEncoder, specialTokens)
      ));

      return ids;
    },
    detokenize(tokenIds) {
      const output: string[] = [];
      const pendingBytes: number[] = [];

      function flushBytes() {
        if (pendingBytes.length > 0) {
          output.push(new TextDecoder().decode(new Uint8Array(pendingBytes)));
          pendingBytes.length = 0;
        }
      }

      for (const id of tokenIds) {
        const token = tokens[id];
        if (token === undefined) {
          throw new Error(`Unknown token id: ${id}`);
        }
        if (isSpecialToken(token, tokenTypes?.[id])) {
          flushBytes();
          output.push(token);
          continue;
        }
        const fallbackByte = tokenizerKind === "spm-bpe" ? byteFallbackValue(token) : undefined;
        if (fallbackByte !== undefined) {
          pendingBytes.push(fallbackByte);
          continue;
        }
        const text = tokenizerKind === "spm-bpe" ? token.replaceAll("▁", " ") : token;
        for (const char of Array.from(text)) {
          if (tokenizerKind === "gpt2-byte-bpe") {
            const byte = byteDecoder.get(char);
            if (byte !== undefined) {
              pendingBytes.push(byte);
              continue;
            }
          }
          flushBytes();
          output.push(char);
        }
      }
      flushBytes();
      return output.join("");
    },
    idToToken(id) {
      return tokens[id];
    },
    tokenToId(token) {
      return tokenToId.get(token);
    },
  };
}

function detectTokenizerKind(model: string | undefined, pre: string | undefined): TokenizerKind | undefined {
  if (model === "gpt2" && pre === "gemma4") {
    return "gpt2-byte-bpe";
  }
  if (model === "gemma4" && pre === undefined) {
    return "spm-bpe";
  }
  return undefined;
}

function parseMerge(merge: string, tokenizerKind: TokenizerKind): [string, string] | undefined {
  const separatorIndex = tokenizerKind === "spm-bpe" ? merge.indexOf(" ", 1) : merge.indexOf(" ");
  if (separatorIndex < 0) {
    return undefined;
  }
  return [merge.slice(0, separatorIndex), merge.slice(separatorIndex + 1)];
}

function tokenizeGpt2ByteBpeText(
  input: string,
  tokenToId: Map<string, number>,
  ranks: Map<string, number>,
  byteEncoder: string[],
  specialTokens: readonly string[],
): number[] {
  const ids: number[] = [];
  let offset = 0;

  while (offset < input.length) {
    const special = specialTokens.find((token) => input.startsWith(token, offset));
    if (special) {
      const id = tokenToId.get(special);
      if (id === undefined) {
        throw new Error(`Unknown special token: ${special}`);
      }
      ids.push(id);
      offset += special.length;
      continue;
    }

    let nextSpecial = input.length;
    for (const token of specialTokens) {
      const index = input.indexOf(token, offset);
      if (index >= 0 && index < nextSpecial) {
        nextSpecial = index;
      }
    }

    const chunk = input.slice(offset, nextSpecial);
    for (const piece of chunk.match(TOKENIZER_PATTERN) ?? []) {
      const encoded = Array.from(new TextEncoder().encode(piece), (byte) => byteEncoder[byte]).join("");
      for (const token of applyBpe(encoded, ranks)) {
        const id = tokenToId.get(token);
        if (id === undefined) {
          throw new Error(`BPE produced unknown token: ${token}`);
        }
        ids.push(id);
      }
    }
    offset = nextSpecial;
  }

  return ids;
}

function tokenizeSpmBpeText(
  input: string,
  tokenToId: Map<string, number>,
  ranks: Map<string, number>,
  specialTokens: readonly string[],
): number[] {
  const ids: number[] = [];
  let offset = 0;

  while (offset < input.length) {
    const special = specialTokens.find((token) => input.startsWith(token, offset));
    if (special) {
      const id = tokenToId.get(special);
      if (id === undefined) {
        throw new Error(`Unknown special token: ${special}`);
      }
      ids.push(id);
      offset += special.length;
      continue;
    }

    let nextSpecial = input.length;
    for (const token of specialTokens) {
      const index = input.indexOf(token, offset);
      if (index >= 0 && index < nextSpecial) {
        nextSpecial = index;
      }
    }

    const chunk = input.slice(offset, nextSpecial).replaceAll(" ", "▁");
    for (const piece of chunk.match(TOKENIZER_SPM_BPE_PATTERN) ?? []) {
      const tokens = piece.indexOf("\n") >= 0 && piece.replaceAll("\n", "") === "" && tokenToId.has(piece)
        ? [piece]
        : applyBpe(Array.from(piece).join(""), ranks);
      for (const token of tokens) {
        const id = tokenToId.get(token);
        if (id !== undefined) {
          ids.push(id);
          continue;
        }
        ids.push(...byteFallbackTokenIds(token, tokenToId));
      }
    }
    offset = nextSpecial;
  }

  return ids;
}

function byteFallbackTokenIds(token: string, tokenToId: Map<string, number>): number[] {
  const ids: number[] = [];
  for (const byte of new TextEncoder().encode(token)) {
    const fallbackToken = `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
    const id = tokenToId.get(fallbackToken);
    if (id === undefined) {
      throw new Error(`BPE produced unknown token and byte fallback is missing: ${token}`);
    }
    ids.push(id);
  }
  return ids;
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

function optionalNumberArray(value: GgufMetadataValue | undefined): number[] | undefined {
  if (
    typeof value !== "object" ||
    value === null ||
    !("sample" in value) ||
    value.truncated ||
    value.sample.length !== value.length
  ) {
    return undefined;
  }

  const numbers = value.sample.filter((item): item is number => typeof item === "number");
  return numbers.length === value.length ? numbers : undefined;
}

function pairKey(left: string, right: string): string {
  return `${left}\u0000${right}`;
}

function isSpecialToken(token: string, tokenType?: number): boolean {
  if (tokenType !== undefined) {
    return (tokenType === 3 || tokenType === 4) && byteFallbackValue(token) === undefined;
  }
  return ((token.startsWith("<") && token.endsWith(">")) || (token.startsWith("[") && token.endsWith("]"))) &&
    byteFallbackValue(token) === undefined;
}

function byteFallbackValue(token: string): number | undefined {
  const match = /^<0x([0-9A-Fa-f]{2})>$/.exec(token);
  return match ? Number.parseInt(match[1] ?? "0", 16) : undefined;
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

function buildByteDecoder(byteEncoder: string[]): Map<string, number> {
  const result = new Map<string, number>();
  for (let byte = 0; byte < byteEncoder.length; byte += 1) {
    result.set(byteEncoder[byte] ?? "", byte);
  }
  return result;
}
