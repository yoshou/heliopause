import assert from "node:assert/strict";
import test from "node:test";

import {
  calculateGemma4VisionResize,
} from "../src/vision.ts";

test("Gemma4V dynamic resize follows llama.cpp token limits", () => {
  const manifest = {
    patchSize: 16,
    spatialMergeSize: 3,
    imageMinTokens: 252,
    imageMaxTokens: 280,
  };

  assert.deepEqual(calculateGemma4VisionResize(manifest, 512, 512), {
    width: 768,
    height: 768,
    outputTokenCount: 256,
  });

  const wide = calculateGemma4VisionResize(manifest, 389, 244);
  assert.equal(wide.width % 48, 0);
  assert.equal(wide.height % 48, 0);
  assert.ok(wide.outputTokenCount >= 252);
  assert.ok(wide.outputTokenCount <= 280);

  const tall = calculateGemma4VisionResize(manifest, 215, 330);
  assert.equal(tall.width % 48, 0);
  assert.equal(tall.height % 48, 0);
  assert.ok(tall.outputTokenCount >= 252);
  assert.ok(tall.outputTokenCount <= 280);
});
