import assert from "node:assert/strict";
import test from "node:test";

import {
  checkWebGpuSupport,
  WEBGPU_REQUIRED_FEATURES,
  webGpuAdapterSupportsRequiredFeatures,
  webGpuRequiredDeviceFeatures,
} from "../src/runner/webgpu/gpu-device.ts";
import type { WebGpuAdapterLike } from "../src/runner/webgpu/gpu-types.ts";

test("WebGPU support requires shader-f16", async () => {
  await withNavigatorGpu({
    requestAdapter: async () => fakeAdapter([]),
  }, async () => {
    const support = await checkWebGpuSupport();

    assert.equal(support.available, false);
    assert.equal(support.reason, "shader-f16-missing");
  });

  const adapter = fakeAdapter(["shader-f16"]);
  assert.equal(webGpuAdapterSupportsRequiredFeatures(adapter), true);
});

test("WebGPU device required features always keep shader-f16", () => {
  assert.deepEqual([...WEBGPU_REQUIRED_FEATURES], ["shader-f16"]);
  assert.deepEqual(webGpuRequiredDeviceFeatures(fakeAdapter(["shader-f16"])), ["shader-f16"]);
  assert.deepEqual(
    webGpuRequiredDeviceFeatures(fakeAdapter(["shader-f16", "timestamp-query"]), {
      includeTimestampQuery: true,
    }),
    ["shader-f16", "timestamp-query"],
  );
  assert.deepEqual(
    webGpuRequiredDeviceFeatures(fakeAdapter(["shader-f16"]), {
      includeTimestampQuery: true,
    }),
    ["shader-f16"],
  );
});

function fakeAdapter(features: readonly string[]): WebGpuAdapterLike {
  return {
    features: {
      has: (feature) => features.includes(feature),
    },
    requestDevice: async () => {
      throw new Error("fake device is not used by this test");
    },
  };
}

async function withNavigatorGpu(
  gpu: { requestAdapter: () => Promise<WebGpuAdapterLike | null> },
  fn: () => Promise<void>,
): Promise<void> {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { gpu },
  });
  try {
    await fn();
  } finally {
    if (descriptor) {
      Object.defineProperty(globalThis, "navigator", descriptor);
    } else {
      Reflect.deleteProperty(globalThis, "navigator");
    }
  }
}
