import assert from "node:assert/strict";
import test from "node:test";

import {
  checkWebGpuSupport,
  WEBGPU_REQUIRED_FEATURES,
  webGpuAdapterSupportsRequiredFeatures,
  webGpuDevice,
  webGpuRequiredDeviceFeatures,
} from "../src/runner/webgpu/gpu-device.ts";
import type { WebGpuAdapterLike, WebGpuDeviceLike } from "../src/runner/webgpu/gpu-types.ts";

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

test("WebGPU device requests timestamp-query only for GPU profiling", async () => {
  const requestedFeatures: string[][] = [];
  await withNavigatorGpu({
    requestAdapter: async () => ({
      ...fakeAdapter(["shader-f16", "timestamp-query"]),
      requestDevice: async (descriptor) => {
        requestedFeatures.push(descriptor?.requiredFeatures ?? []);
        return fakeDevice();
      },
    }),
  }, async () => {
    assert.ok(await webGpuDevice());
    assert.ok(await webGpuDevice({ gpuProfiling: true }));
  });

  assert.deepEqual(requestedFeatures, [
    ["shader-f16"],
    ["shader-f16", "timestamp-query"],
  ]);
});

test("WebGPU GPU profiling rejects when timestamp-query is unavailable", async () => {
  await withNavigatorGpu({
    requestAdapter: async () => ({
      ...fakeAdapter(["shader-f16"]),
      requestDevice: async () => fakeDevice(),
    }),
  }, async () => {
    await assert.rejects(
      () => webGpuDevice({ gpuProfiling: true }),
      /requires timestamp-query support/,
    );
  });
});

test("WebGPU GPU profiling rejects when timestamp-query device request fails", async () => {
  const requestedFeatures: string[][] = [];
  await withNavigatorGpu({
    requestAdapter: async () => ({
      ...fakeAdapter(["shader-f16", "timestamp-query"]),
      requestDevice: async (descriptor) => {
        requestedFeatures.push(descriptor?.requiredFeatures ?? []);
        throw new Error("timestamp disabled by adapter");
      },
    }),
  }, async () => {
    await assert.rejects(
      () => webGpuDevice({ gpuProfiling: true }),
      /GPU profiling device request failed: timestamp disabled by adapter/,
    );
  });

  assert.deepEqual(requestedFeatures, [["shader-f16", "timestamp-query"]]);
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

function fakeDevice(): WebGpuDeviceLike {
  return {
    features: {
      has: (feature) => feature === "timestamp-query",
    },
    createBuffer: () => ({
      getMappedRange: () => new ArrayBuffer(0),
      unmap() {},
      mapAsync: async () => {},
      destroy() {},
    }),
    createShaderModule: () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createComputePipeline: () => ({}),
    createBindGroup: () => ({}),
    createQuerySet: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline() {},
        setBindGroup() {},
        dispatchWorkgroups() {},
        end() {},
      }),
      copyBufferToBuffer() {},
      resolveQuerySet() {},
      finish: () => ({}),
    }),
    queue: {
      writeBuffer() {},
      submit() {},
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
