import type {
  AudioRunners,
} from "./audio-runner";
import type {
  ModelRunner,
} from "./model-runner";
import type {
  MtpAssistantRunners,
} from "./mtp-assistant-runner";
import type {
  MtpTargetRunners,
} from "./mtp-target-runner";
import type {
  ProviderResourceRequirements,
} from "./planning";
import type {
  ModelSession,
} from "../runtime";
import type {
  SegmentRunnerProvider,
} from "./segment-runner";
import type {
  VisionRunners,
} from "./vision-runner";

export type RunnerProvider = {
  readonly name: SegmentRunnerProvider;
};

export type ModelRunnerProvider = RunnerProvider & {
  createModelRunner(): ModelRunner;
  modelResourceRequirements(
    session: ModelSession,
    options: { contextLength: number; slidingWindowReserveTokens?: number },
  ): ProviderResourceRequirements;
};

export type AudioRunnerProvider = RunnerProvider & {
  createAudioRunners(): AudioRunners;
};

export type VisionRunnerProvider = RunnerProvider & {
  createVisionRunners(): VisionRunners;
};

export type MtpAssistantRunnerProvider = RunnerProvider & {
  createMtpAssistantRunners(): MtpAssistantRunners;
};

export type MtpTargetRunnerProvider = ModelRunnerProvider & {
  createMtpTargetRunners(): MtpTargetRunners;
};

export type MultimodalRunnerProvider = ModelRunnerProvider & AudioRunnerProvider & VisionRunnerProvider;

export function validateProviderList<TProvider extends RunnerProvider>(
  providers: readonly TProvider[],
  capability:
    | keyof ModelRunnerProvider
    | keyof AudioRunnerProvider
    | keyof VisionRunnerProvider
    | keyof MtpAssistantRunnerProvider
    | keyof MtpTargetRunnerProvider,
): readonly TProvider[] {
  if (providers.length === 0) {
    throw new Error("At least one runner provider is required.");
  }

  const names = new Set<SegmentRunnerProvider>();
  for (const provider of providers) {
    if (names.has(provider.name)) {
      throw new Error(`Duplicate runner provider: ${provider.name}`);
    }
    names.add(provider.name);
    if (typeof (provider as Record<string, unknown>)[capability] !== "function") {
      throw new Error(`Runner provider ${provider.name} is missing ${String(capability)}.`);
    }
  }

  return providers.slice();
}

export function requireMtpTargetRunnerProvider(
  provider: ModelRunnerProvider | undefined,
): MtpTargetRunnerProvider {
  if (!provider) {
    throw new Error("MTP target execution requires a target runner provider.");
  }
  if (typeof (provider as Partial<MtpTargetRunnerProvider>).createMtpTargetRunners !== "function") {
    throw new Error(`Runner provider ${provider.name} does not support Gemma 4 MTP target execution.`);
  }
  return provider as MtpTargetRunnerProvider;
}

export function resolveProviderOrder<TProvider extends RunnerProvider>(
  providers: readonly TProvider[],
  order: readonly SegmentRunnerProvider[] | undefined,
): readonly TProvider[] {
  if (!order) {
    return providers;
  }
  if (order.length === 0) {
    throw new Error("At least one preprocess provider is required.");
  }

  const providerByName = new Map(providers.map((provider) => [provider.name, provider]));
  return order.map((name) => {
    const provider = providerByName.get(name);
    if (!provider) {
      throw new Error(`Unknown preprocess provider: ${name}`);
    }
    return provider;
  });
}
