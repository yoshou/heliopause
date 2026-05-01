import type {
  BrowserBenchCaseId,
  BrowserBenchEnvironment,
  BrowserBenchReport,
  BrowserBenchResult,
  BrowserBenchSize,
} from "./browser-bench";

export type BenchWorkerRequest =
  | {
      type: "run";
      requestId: number;
      caseIds: BrowserBenchCaseId[];
      sizes: BrowserBenchSize[];
      warmupIterations: number;
      minimumMs: number;
    }
  | {
      type: "cancel";
      requestId: number;
    }
  | {
      type: "environment";
      requestId: number;
    };

export type BenchWorkerResponse =
  | {
      type: "environment";
      requestId: number;
      environment: BrowserBenchEnvironment;
    }
  | {
      type: "progress";
      requestId: number;
      result: BrowserBenchResult;
    }
  | {
      type: "done";
      requestId: number;
      report: BrowserBenchReport;
    }
  | {
      type: "cancelled";
      requestId: number;
    }
  | {
      type: "error";
      requestId: number;
      message: string;
    };
