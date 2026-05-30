import type {
  MtpAssistantRunResult,
  MtpAssistantSession,
} from "../mtp-assistant";
import type {
  MtpAssistantRunInput,
} from "./mtp-assistant-runner";

export async function runMtpAssistant(
  session: MtpAssistantSession,
  input: MtpAssistantRunInput,
  options: { signal?: AbortSignal } = {},
): Promise<MtpAssistantRunResult> {
  const runners = session.assistantRunners[0];
  if (!runners) {
    throw new Error("No MTP assistant runner provider configured.");
  }
  return runners.runner.run(session, input, options);
}
