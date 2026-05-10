import { useEffect, useRef, useState } from "react";
import {
  defaultBrowserBenchCaseIds,
  defaultBrowserBenchSizes,
  type BrowserBenchCaseId,
  type BrowserBenchEnvironment,
  type BrowserBenchReport,
  type BrowserBenchResult,
  type BrowserBenchSize,
} from "./browser-bench";
import type { BenchWorkerRequest, BenchWorkerResponse } from "./bench-worker-protocol";

type RunState =
  | { status: "idle" }
  | { status: "running"; requestId: number }
  | { status: "error"; message: string }
  | { status: "done" }
  | { status: "cancelled" };

const CASE_LABELS: Record<BrowserBenchCaseId, string> = {
  "matmul": "Matmul",
  "gqa-attention": "GQA attention",
  "swiglu": "SwiGLU",
  "swiglu-down": "SwiGLU + down",
  "full-attention-decode-out": "Full attention decode/out",
  "top-token": "Top token",
};

const SIZE_LABELS: Record<BrowserBenchSize, string> = {
  small: "Small",
  medium: "Medium",
  large: "Large",
};

function App() {
  const [environment, setEnvironment] = useState<BrowserBenchEnvironment | undefined>();
  const [selectedCases, setSelectedCases] = useState<BrowserBenchCaseId[]>(
    defaultBrowserBenchCaseIds(),
  );
  const [selectedSizes, setSelectedSizes] = useState<BrowserBenchSize[]>(
    defaultBrowserBenchSizes(),
  );
  const [warmupIterations, setWarmupIterations] = useState(3);
  const [minimumMs, setMinimumMs] = useState(250);
  const [results, setResults] = useState<BrowserBenchResult[]>([]);
  const [lastReport, setLastReport] = useState<BrowserBenchReport | undefined>();
  const [runState, setRunState] = useState<RunState>({ status: "idle" });
  const workerRef = useRef<Worker | undefined>(undefined);
  const requestIdRef = useRef(1);

  const summary = summarize(results);
  const isRunning = runState.status === "running";

  useEffect(() => {
    const worker = createBenchWorker();
    workerRef.current = worker;
    const requestId = requestIdRef.current++;
    worker.postMessage({ type: "environment", requestId } satisfies BenchWorkerRequest);
    return () => {
      worker.terminate();
      workerRef.current = undefined;
    };
  }, []);

  function createBenchWorker(): Worker {
    const worker = new Worker(new URL("./bench-worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (event: MessageEvent<BenchWorkerResponse>) => {
      handleWorkerMessage(event.data);
    };
    worker.onerror = (event) => {
      setRunState({ status: "error", message: event.message || "Worker failed." });
    };
    return worker;
  }

  function handleWorkerMessage(message: BenchWorkerResponse): void {
    if (message.type === "environment") {
      setEnvironment(message.environment);
      return;
    }
    if (message.type === "progress") {
      setResults((current) => [...current, message.result]);
      return;
    }
    if (message.type === "done") {
      setEnvironment(message.report.environment);
      setLastReport(message.report);
      setRunState({ status: "done" });
      return;
    }
    if (message.type === "cancelled") {
      setRunState({ status: "cancelled" });
      return;
    }
    if (message.type === "error") {
      setRunState({ status: "error", message: message.message });
    }
  }

  function startRun(): void {
    const worker = workerRef.current;
    if (!worker || selectedCases.length === 0 || selectedSizes.length === 0 || isRunning) {
      return;
    }
    const requestId = requestIdRef.current++;
    setResults([]);
    setLastReport(undefined);
    setRunState({ status: "running", requestId });
    worker.postMessage({
      type: "run",
      requestId,
      caseIds: selectedCases,
      sizes: selectedSizes,
      warmupIterations,
      minimumMs,
    } satisfies BenchWorkerRequest);
  }

  function stopRun(): void {
    if (runState.status !== "running") {
      return;
    }
    workerRef.current?.postMessage({
      type: "cancel",
      requestId: runState.requestId,
    } satisfies BenchWorkerRequest);
  }

  return (
    <main className="bench-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">WebGPU / WASM synthetic bench</p>
          <h1>Heliopause Bench</h1>
        </div>
        <div className="topbar-actions">
          {isRunning ? (
            <button type="button" className="secondary-button" onClick={stopRun}>
              Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={startRun}
              disabled={selectedCases.length === 0 || selectedSizes.length === 0}
            >
              Run
            </button>
          )}
          <button
            type="button"
            className="ghost-button"
            disabled={results.length === 0}
            onClick={() => downloadJson(currentReport(lastReport, environment, results))}
          >
            JSON
          </button>
          <button
            type="button"
            className="ghost-button"
            disabled={results.length === 0}
            onClick={() => downloadCsv(results)}
          >
            CSV
          </button>
        </div>
      </header>

      <section className="summary-grid" aria-label="Summary">
        <MetricCard label="Completed" value={String(results.length)} />
        <MetricCard label="Pass" value={String(summary.ok)} tone="good" />
        <MetricCard label="Failed" value={String(summary.failed)} tone={summary.failed > 0 ? "bad" : "neutral"} />
      </section>

      <section className="bench-layout">
        <aside className="control-panel" aria-label="Controls">
          <EnvironmentPanel environment={environment} />

          <fieldset>
            <legend>Blocks</legend>
            <div className="option-list">
              {defaultBrowserBenchCaseIds().map((caseId) => (
                <label key={caseId} className="check-row">
                  <input
                    type="checkbox"
                    checked={selectedCases.includes(caseId)}
                    disabled={isRunning}
                    onChange={() => setSelectedCases((current) => toggleValue(current, caseId))}
                  />
                  <span>{CASE_LABELS[caseId]}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset>
            <legend>Sizes</legend>
            <div className="segmented">
              {(["small", "medium", "large"] as const).map((size) => (
                <label key={size}>
                  <input
                    type="checkbox"
                    checked={selectedSizes.includes(size)}
                    disabled={isRunning}
                    onChange={() => setSelectedSizes((current) => toggleValue(current, size))}
                  />
                  <span>{SIZE_LABELS[size]}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset>
            <legend>Timing</legend>
            <label className="number-field">
              <span>Warmup</span>
              <input
                type="number"
                min={0}
                max={32}
                value={warmupIterations}
                disabled={isRunning}
                onChange={(event) => setWarmupIterations(clampNumber(event.target.value, 0, 32))}
              />
            </label>
            <label className="number-field">
              <span>Min ms</span>
              <input
                type="number"
                min={50}
                max={5000}
                step={50}
                value={minimumMs}
                disabled={isRunning}
                onChange={(event) => setMinimumMs(clampNumber(event.target.value, 50, 5000))}
              />
            </label>
          </fieldset>
        </aside>

        <section className="results-panel" aria-label="Results">
          <div className="status-row">
            <RunStatus state={runState} />
          </div>
          <ResultsTable results={results} />
        </section>
      </section>
    </main>
  );
}

function EnvironmentPanel({ environment }: { environment?: BrowserBenchEnvironment }) {
  return (
    <section className="environment-panel">
      <h2>Environment</h2>
      <dl>
        <div>
          <dt>WASM</dt>
          <dd>{environment?.wasmBackend ?? "Checking"}</dd>
        </div>
        <div>
          <dt>WebGPU</dt>
          <dd>{formatWebGpu(environment)}</dd>
        </div>
        <div>
          <dt>Smoke</dt>
          <dd>{environment?.webGpuSmoke ? (environment.webGpuSmoke.ok ? "ok" : environment.webGpuSmoke.reason) : "n/a"}</dd>
        </div>
      </dl>
    </section>
  );
}

function MetricCard(
  { label, value, tone = "neutral" }: { label: string; value: string; tone?: "neutral" | "good" | "bad" },
) {
  return (
    <article className={`metric metric--${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function ResultsTable({ results }: { results: BrowserBenchResult[] }) {
  if (results.length === 0) {
    return (
      <div className="empty-results">
        <h2>No results</h2>
      </div>
    );
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Block</th>
            <th>Size</th>
            <th>Backend</th>
            <th>Status</th>
            <th>Mean</th>
            <th>GPU mean</th>
            <th>Ops/s</th>
            <th>Speedup</th>
            <th>Diff</th>
            <th>Shape</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result, index) => (
            <tr key={`${result.caseName}-${result.size}-${result.backend}-${result.variant}-${index}`}>
              <td>
                <strong>{result.caseName}</strong>
                <span>{result.variant}</span>
              </td>
              <td>{SIZE_LABELS[result.size]}</td>
              <td>{result.backend}</td>
              <td><StatusPill result={result} /></td>
              <td>{result.meanMs ? `${result.meanMs.toFixed(3)} ms` : "-"}</td>
              <td>{result.gpuMeanMs === undefined ? "-" : `${result.gpuMeanMs.toFixed(3)} ms`}</td>
              <td>{result.opsPerSecond ? result.opsPerSecond.toFixed(1) : "-"}</td>
              <td>{formatSpeedup(result.speedupVsWasm ?? result.speedupVsReference)}</td>
              <td>{formatDiff(result)}</td>
              <td className="shape-cell">{result.shape}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusPill({ result }: { result: BrowserBenchResult }) {
  return (
    <span className={`status-pill status-pill--${result.status}`} title={result.message}>
      {result.status}
    </span>
  );
}

function RunStatus({ state }: { state: RunState }) {
  if (state.status === "running") {
    return <p className="run-status run-status--running">Running</p>;
  }
  if (state.status === "error") {
    return <p className="run-status run-status--error">{state.message}</p>;
  }
  if (state.status === "cancelled") {
    return <p className="run-status">Cancelled</p>;
  }
  if (state.status === "done") {
    return <p className="run-status run-status--done">Done</p>;
  }
  return <p className="run-status">Idle</p>;
}

function summarize(results: BrowserBenchResult[]): { ok: number; failed: number } {
  let ok = 0;
  let failed = 0;
  for (const result of results) {
    if (result.status === "ok") {
      ok += 1;
    }
    if (result.status === "failed") {
      failed += 1;
    }
  }
  return { ok, failed };
}

function formatWebGpu(environment?: BrowserBenchEnvironment): string {
  const support = environment?.webGpuSupport;
  if (!support) {
    return "Checking";
  }
  if (!support.available) {
    return support.reason;
  }
  return support.adapterName || "available";
}

function formatDiff(result: BrowserBenchResult): string {
  if (result.maxAbsDiff === undefined) {
    return "-";
  }
  const rel = result.maxRelDiff === undefined ? "-" : result.maxRelDiff.toExponential(2);
  return `${result.maxAbsDiff.toExponential(2)} abs / ${rel} rel`;
}

function formatSpeedup(value?: number): string {
  return value ? `${value.toFixed(2)}x` : "-";
}

function toggleValue<T>(values: T[], value: T): T[] {
  return values.includes(value)
    ? values.filter((candidate) => candidate !== value)
    : [...values, value];
}

function clampNumber(value: string, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return min;
  }
  return Math.min(max, Math.max(min, Math.trunc(parsed)));
}

function currentReport(
  lastReport: BrowserBenchReport | undefined,
  environment: BrowserBenchEnvironment | undefined,
  results: BrowserBenchResult[],
): BrowserBenchReport {
  if (lastReport && lastReport.results.length === results.length) {
    return lastReport;
  }
  return {
    environment: environment ?? {
      wasmBackend: "ts",
      webGpuSupport: { available: false, reason: "api-missing" },
    },
    results,
  };
}

function downloadJson(report: BrowserBenchReport): void {
  downloadText(
    `heliopause-bench-${timestampForFile()}.json`,
    "application/json",
    `${JSON.stringify(report, null, 2)}\n`,
  );
}

function downloadCsv(results: BrowserBenchResult[]): void {
  const headers = [
    "caseId",
    "caseName",
    "backend",
    "size",
    "variant",
    "shape",
    "status",
    "iterations",
    "totalMs",
    "meanMs",
    "opsPerSecond",
    "gpuMeanMs",
    "checksum",
    "maxAbsDiff",
    "maxRelDiff",
    "tolerance",
    "relativeTolerance",
    "tolerancePass",
    "speedupVsReference",
    "speedupVsWasm",
    "message",
  ];
  const rows = results.map((result) =>
    headers.map((header) => csvCell(result[header as keyof BrowserBenchResult])).join(",")
  );
  downloadText(
    `heliopause-bench-${timestampForFile()}.csv`,
    "text/csv",
    `${headers.join(",")}\n${rows.join("\n")}\n`,
  );
}

function downloadText(fileName: string, type: string, content: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function csvCell(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }
  const text = String(value);
  if (!/[",\n\r]/.test(text)) {
    return text;
  }
  return `"${text.replaceAll("\"", "\"\"")}"`;
}

function timestampForFile(): string {
  return new Date().toISOString().replaceAll(":", "-").replace(/\.\d{3}Z$/, "Z");
}

export default App;
