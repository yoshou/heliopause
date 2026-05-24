import { createHash } from "node:crypto";

const MODEL = {
  url:
    "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/653803f092503c04a65164346f3208a36e707693/gemma-4-E4B-it-Q4_K_M.gguf",
  path: "models/gemma4/gemma-4-E4B-it-Q4_K_M.gguf",
  size: 4_977_169_568,
  sha256: "519b9793ed6ce0ff530f1b7c96e848e08e49e7af4d57bb97f76215963a54146d",
};

async function main(): Promise<void> {
  await Deno.mkdir(dirname(MODEL.path), { recursive: true });
  if (await validated(MODEL.path)) {
    console.log(`already validated: ${MODEL.path}`);
    return;
  }

  const partPath = `${MODEL.path}.part`;
  await removeIfExists(partPath);
  const response = await fetch(MODEL.url);
  if (!response.ok || !response.body) {
    throw new Error(`download failed: ${response.status} ${response.statusText}`);
  }

  const file = await Deno.open(partPath, { create: true, write: true, truncate: true });
  const hash = createHash("sha256");
  let size = 0;
  try {
    for await (const chunk of response.body) {
      size += chunk.byteLength;
      hash.update(chunk);
      await file.write(chunk);
    }
  } finally {
    file.close();
  }

  const sha256 = hash.digest("hex");
  if (size !== MODEL.size || sha256 !== MODEL.sha256) {
    await removeIfExists(partPath);
    throw new Error(
      `download integrity check failed: size ${size}/${MODEL.size}, sha256 ${sha256}/${MODEL.sha256}`,
    );
  }

  await Deno.rename(partPath, MODEL.path);
  console.log(`downloaded and validated: ${MODEL.path}`);
}

async function validated(path: string): Promise<boolean> {
  try {
    const stat = await Deno.stat(path);
    if (!stat.isFile || stat.size !== MODEL.size) {
      return false;
    }
    return await sha256File(path) === MODEL.sha256;
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) {
      return false;
    }
    throw error;
  }
}

async function sha256File(path: string): Promise<string> {
  const file = await Deno.open(path, { read: true });
  const hash = createHash("sha256");
  const buffer = new Uint8Array(1024 * 1024);
  try {
    while (true) {
      const read = await file.read(buffer);
      if (read === null) {
        break;
      }
      hash.update(buffer.subarray(0, read));
    }
  } finally {
    file.close();
  }
  return hash.digest("hex");
}

async function removeIfExists(path: string): Promise<void> {
  try {
    await Deno.remove(path);
  } catch (error) {
    if (!(error instanceof Deno.errors.NotFound)) {
      throw error;
    }
  }
}

function dirname(path: string): string {
  const index = path.lastIndexOf("/");
  return index < 0 ? "." : path.slice(0, index);
}

if (import.meta.main) {
  await main();
}
