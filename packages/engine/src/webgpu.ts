export type WebGpuSupport =
  | {
      available: true;
    }
  | {
      available: false;
      reason: "navigator-missing" | "api-missing" | "adapter-missing" | "request-failed";
      error?: string;
    };

type NavigatorWithWebGpu = Navigator & {
  gpu?: {
    requestAdapter: () => Promise<unknown | null>;
  };
};

export async function checkWebGpuSupport(): Promise<WebGpuSupport> {
  if (typeof navigator === "undefined") {
    return {
      available: false,
      reason: "navigator-missing",
    };
  }

  const gpu = (navigator as NavigatorWithWebGpu).gpu;

  if (!gpu) {
    return {
      available: false,
      reason: "api-missing",
    };
  }

  try {
    const adapter = await gpu.requestAdapter();

    if (!adapter) {
      return {
        available: false,
        reason: "adapter-missing",
      };
    }

    return {
      available: true,
    };
  } catch (error) {
    return {
      available: false,
      reason: "request-failed",
      error: error instanceof Error ? error.message : undefined,
    };
  }
}
