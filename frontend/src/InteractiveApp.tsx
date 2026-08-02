import { useState } from "react";
import { queryXcds } from "./api";
import type { QueryResponse } from "./api";
import { AnswerPanel } from "./components/AnswerPanel";
import { EvidencePanel } from "./components/EvidencePanel";
import { PRESETS } from "./presets";
import { GuidedTour, type TourStep } from "./components/GuidedTour";

// Reranker model options
const MODEL_OPTIONS = [
  { value: "cross-encoder/ms-marco-MiniLM-L-6-v2", label: "MiniLM-L-6-v2 (Default / 90MB)" },
  { value: "BAAI/bge-reranker-v2-m3", label: "BGE-Reranker-v2-m3 (Upgraded / 567MB)" }
];

export function InteractiveApp() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);

  // Configuration options
  const [selectedModel, setSelectedModel] = useState(MODEL_OPTIONS[0].value);
  const [nValue, setNValue] = useState(0.25);

  const [autoSubmit, setAutoSubmit] = useState(true);
  const [tourActive, setTourActive] = useState(false);

  const TOUR_STEPS: TourStep[] = [
    {
      targetId: "tour-interactive-header",
      title: "Welcome to X-CDS Playground!",
      description: "Here you can customize RAG settings, choose the reranking model, and adjust citation verification strictness.",
    },
    {
      targetId: "tour-interactive-config",
      title: "Parameters Configuration",
      description: "Choose the reranking model and modify the Min Token Overlap ($T_{min}$). Higher overlap values enforce strict citation check iterations.",
    },
    {
      targetId: "tour-interactive-query",
      title: "Symptom & Preset Selection",
      description: "Select standard clinical preset questions or write custom queries. The 'Auto-submit' toggle lets you run the demo query instantly.",
    },
    {
      targetId: "tour-interactive-telemetry",
      title: "Self-Correction & Refusal Telemetry",
      description: "Audits the execution pipeline in real-time. Highlights validation attempts, first-pass validation status, and clinical abstention.",
    },
    {
      targetId: "tour-interactive-answer",
      title: "Attributed Diagnostic Answer",
      description: "Outputs the generated clinical guidelines response, marked with active citation brackets pointing to the source passages.",
    },
    {
      targetId: "tour-interactive-evidence",
      title: "Retrieved Source Chunks",
      description: "Visualizes the retrieved reference documents along with the exact semantic similarity score returned by the reranker.",
    },
  ];

  const startTour = () => {
    if (!response) {
      handleDemoLoad();
    }
    setTourActive(true);
  };

  async function runQuery(q: string, model: string, n: number) {
    setLoading(true);
    setElapsed(0);
    setError(null);
    setResponse(null);
    setActiveCitation(null);

    const timer = setInterval(() => {
      setElapsed((prev) => prev + 0.1);
    }, 100);

    try {
      const res = await queryXcds(q, model, n);
      setResponse(res);
      if (res.cited_indices.length > 0) {
        setActiveCitation(res.cited_indices[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      clearInterval(timer);
      setLoading(false);
    }
  }

  async function handleQuerySubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    await runQuery(trimmed, selectedModel, nValue);
  }

  function handleDemoLoad() {
    const demoQuery = "A pregnant patient in her first trimester is diagnosed with Zika virus. What fetal complications should be screened for?";
    const demoModel = "cross-encoder/ms-marco-MiniLM-L-6-v2";
    const demoN = 0.10;
    
    setQuery(demoQuery);
    setSelectedModel(demoModel);
    setNValue(demoN);
    
    if (autoSubmit) {
      void runQuery(demoQuery, demoModel, demoN);
    }
  }

  const isAbstention = response && (
    response.answer.toLowerCase().includes("insufficient evidence") || 
    response.answer.toLowerCase().includes("no evidence")
  );

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 py-6 sm:px-6 lg:px-8 lg:py-8 pb-16">
      {/* Header */}
      <header id="tour-interactive-header" className="rise-in mb-6 lg:mb-8">
        <div className="flex items-start justify-between">
          <div>
            <p className="mb-2 text-sm font-medium tracking-[0.18em] text-[var(--accent-deep)] uppercase">
              X-CDS Playground
            </p>
            <h1 className="text-[clamp(2.6rem,7vw,4.6rem)] leading-[0.95] text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>
              Interactive RAG
            </h1>
          </div>
          <div className="flex gap-2">
            <button
              onClick={startTour}
              className="rounded-xl border border-[var(--accent)] bg-[var(--highlight)] px-4 py-2.5 text-xs font-bold text-[var(--accent-deep)] transition hover:bg-[var(--accent)] hover:text-white animate-pulse"
            >
              ✨ Help Tour
            </button>
            <a
              href="/"
              className="rounded-xl border border-[var(--line)] bg-white/60 px-4 py-2.5 text-xs font-semibold text-[var(--muted)] transition hover:border-[var(--accent)] hover:bg-white hover:text-[var(--accent-deep)]"
            >
              &larr; Clinical Portal
            </a>
            <a
              href="/report"
              className="rounded-xl border border-[var(--line)] bg-white/60 px-4 py-2.5 text-xs font-semibold text-[var(--accent-deep)] transition hover:border-[var(--accent)] hover:bg-white"
            >
              Evaluation Report
            </a>
          </div>
        </div>
        <p className="mt-3 max-w-2xl text-[1.05rem] leading-7 text-[var(--muted)]">
          Experiment with different rerankers and citation validation thresholds (n values) in real-time.
        </p>
      </header>

      {/* Main Grid Layout */}
      <main className="grid flex-1 gap-5 lg:grid-cols-2 lg:gap-6">
        
        {/* Left Column: Query & Controls */}
        <section className="rise-in rounded-[1.4rem] border border-[var(--line)] bg-[var(--panel)] p-5 backdrop-blur-sm sm:p-6">
          <div className="mb-5">
            <h2 className="text-2xl text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>Query settings</h2>
            <p className="mt-1 text-sm text-[var(--muted)]">Configure parameters before submitting your query.</p>
          </div>

          <form onSubmit={handleQuerySubmit} className="space-y-5">
            <div id="tour-interactive-config" className="space-y-5">
              {/* Model Selector */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-[var(--muted)]">
                  Cross-Encoder Reranker
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full rounded-xl border border-[var(--line)] bg-white/70 px-4 py-2.5 text-[0.95rem] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
                >
                  {MODEL_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>

              {/* N Value Slider */}
              <div className="space-y-2">
                <div className="flex items-baseline justify-between">
                  <label className="block text-sm font-semibold text-[var(--muted)]">
                    Min Token Overlap (n value)
                  </label>
                  <span className="text-sm font-bold text-[var(--accent-deep)]">{nValue.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.75"
                  step="0.05"
                  value={nValue}
                  onChange={(e) => setNValue(parseFloat(e.target.value))}
                  className="h-2 w-full cursor-pointer rounded-lg bg-[var(--line)] accent-[var(--accent)]"
                />
                <p className="text-xs text-[var(--muted)]">
                  Lower value is more lenient; higher value enforces stricter citation matching.
                </p>
              </div>
            </div>

            <div id="tour-interactive-query" className="space-y-5">
              {/* Preset Selector */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-[var(--muted)]">
                  Clinical Presets
                </label>
                <select
                  onChange={(e) => {
                    if (e.target.value) {
                      setQuery(e.target.value);
                    }
                  }}
                  value={PRESETS.find(p => p.value === query) ? query : ""}
                  className="w-full rounded-xl border border-[var(--line)] bg-white/70 px-4 py-2.5 text-[0.92rem] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
                >
                  {PRESETS.map((p, idx) => (
                    <option key={idx} value={p.value}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Query Input */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-[var(--muted)]">
                  Symptom/Query Text
                </label>
                <textarea
                  className="min-h-24 w-full resize-y rounded-xl border border-[var(--line)] bg-white/70 px-4 py-3 text-[0.98rem] leading-7 text-[var(--ink)] outline-none transition focus:border-[var(--accent)]"
                  placeholder="Type your symptom or clinical question..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </div>

              {/* Submit & Demo Buttons */}
              <div className="flex flex-wrap items-center gap-3 pt-2">
                <button
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="rounded-xl bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[var(--accent-deep)] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {loading ? "Processing..." : "Submit Query"}
                </button>
                <button
                  type="button"
                  onClick={handleDemoLoad}
                  disabled={loading}
                  className="rounded-xl border border-[var(--line)] bg-white/60 px-5 py-2.5 text-sm font-semibold text-[var(--ink)] transition hover:border-[var(--accent)] disabled:opacity-50"
                >
                  Load Demo Setting
                </button>
                <label className="flex items-center gap-2 text-xs font-semibold text-[var(--muted)] cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={autoSubmit}
                    onChange={(e) => setAutoSubmit(e.target.checked)}
                    className="rounded border-[var(--line)] text-[var(--accent)] focus:ring-[var(--accent)]"
                  />
                  Auto-submit on load
                </label>
              </div>
            </div>
          </form>

          {loading && (
            <div className="mt-4 rounded-xl border border-[var(--line)] bg-slate-50/50 p-4 flex flex-col items-center justify-center space-y-3">
              <div className="flex items-center space-x-3">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--accent)] opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-[var(--accent-deep)]"></span>
                </span>
                <span className="text-sm font-semibold text-[var(--accent-deep)]">
                  {elapsed < 1.5
                    ? "Retrieving clinical literature..."
                    : elapsed < 3.0
                    ? "Reranking context via Cross-Encoder..."
                    : elapsed < 6.0
                    ? "Generating clinical response via Gemini..."
                    : "Verifying citations & running self-correction guardrails..."}
                </span>
              </div>
              <span className="text-xs text-[var(--muted)] font-mono">
                Elapsed time: {elapsed.toFixed(1)}s
              </span>
            </div>
          )}

          {error && (
            <p className="mt-4 rounded-xl bg-red-50 border border-red-200 p-3 text-sm text-[var(--danger)]" role="alert">
              <strong>Error:</strong> {error}
            </p>
          )}

          {response && (
            <div id="tour-interactive-telemetry" className="mt-8 border-t border-[var(--line)] pt-6">
              <h3 className="mb-4 text-lg font-bold text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>
                Pipeline Telemetry
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="rounded-xl border border-[var(--line)] bg-white/40 p-4">
                  <span className="block text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">
                    Generation Attempts
                  </span>
                  <span className="block mt-1 text-2xl font-bold text-[var(--ink)]">
                    {response.generation_attempts} {response.generation_attempts === 1 ? "attempt" : "attempts"}
                  </span>
                </div>
                <div className="rounded-xl border border-[var(--line)] bg-white/40 p-4">
                  <span className="block text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">
                    First-Pass Validation
                  </span>
                  <span className={`mt-1 inline-flex items-center gap-1.5 text-lg font-semibold ${response.generation_attempts === 1 && response.validation_passed ? "text-green-600" : "text-amber-600"}`}>
                    <span className={`h-2.5 w-2.5 rounded-full ${response.generation_attempts === 1 && response.validation_passed ? "bg-green-600" : "bg-amber-600"}`} />
                    {response.generation_attempts === 1 && response.validation_passed ? "PASSED" : "FAILED"}
                  </span>
                </div>
                <div className="rounded-xl border border-[var(--line)] bg-white/40 p-4">
                  <span className="block text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">
                    Clinical Abstention
                  </span>
                  <span className={`mt-1 inline-flex items-center gap-1.5 text-lg font-semibold ${isAbstention ? "text-amber-600" : "text-green-600"}`}>
                    <span className={`h-2.5 w-2.5 rounded-full ${isAbstention ? "bg-amber-600" : "bg-green-600"}`} />
                    {isAbstention ? "REFUSED" : "COMPLETED"}
                  </span>
                </div>
              </div>
              
              {response.validation_issues.length > 0 && (
                <div className="mt-4 rounded-xl bg-amber-50 border border-amber-200 p-4">
                  <span className="block text-sm font-bold text-amber-800 mb-2">Validation Issues Logged:</span>
                  <ul className="list-disc pl-5 space-y-1.5 text-sm text-amber-700">
                    {response.validation_issues.map((issue: any, index: number) => (
                      <li key={index}>
                        {issue.message}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </section>

        {/* Right Column: Answer & Evidence Visualizers */}
        <section className="flex flex-col gap-6">
          
          {/* Answer Output */}
          <div id="tour-interactive-answer" className="rise-in rounded-[1.4rem] border border-[var(--line)] bg-[var(--panel)] p-5 backdrop-blur-sm sm:p-6">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-2xl text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>Generated Answer</h2>
              {response && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-[var(--muted)] bg-white/60 px-2 py-0.5 rounded border border-[var(--line)] font-semibold">
                    Gemini-3.5-Flash
                  </span>
                  <span className="text-[10px] tracking-wide text-[var(--muted)] uppercase font-semibold">
                    {response.validation_passed ? "Verified" : "Unverified"}
                  </span>
                </div>
              )}
            </div>

            {isAbstention && (
              <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50/50 p-3.5 text-xs text-amber-800 flex items-start space-x-2">
                <span className="text-sm">⚠️</span>
                <div>
                  <strong>Safe Abstention Triggered:</strong> The referenced clinical guidelines do not contain direct evidence for this clinical query. To prevent hallucination, the system abstains from generating diagnostic assertions.
                </div>
              </div>
            )}

            <AnswerPanel
              answer={response?.answer ?? ""}
              activeCitation={activeCitation}
              onCitationClick={setActiveCitation}
            />
          </div>

          {/* Evidence Chunks with Score Visualizer */}
          <div id="tour-interactive-evidence" className="rise-in flex-1 rounded-[1.4rem] border border-[var(--line)] bg-[var(--panel)] p-5 backdrop-blur-sm sm:p-6" style={{ animationDelay: "80ms" }}>
            <div className="mb-5">
              <h2 className="text-2xl text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>Retrieved Evidence</h2>
              <p className="mt-1 text-sm text-[var(--muted)]">Source passages matched and scored by the cross-encoder.</p>
            </div>

            <EvidencePanel
              contexts={response?.contexts ?? []}
              activeCitation={activeCitation}
            />
          </div>
        </section>
      </main>

      {/* Sticky Medical Disclaimer Banner */}
      <footer className="fixed bottom-0 left-0 right-0 z-50 border-t border-amber-200 bg-amber-50 py-2.5 text-center text-xs font-semibold text-amber-800 backdrop-blur-md">
        ⚠️ <strong>Research Prototype:</strong> This tool is for demonstration purposes only and must not be used for medical diagnosis, treatment, or clinical decision-making.
      </footer>

      {/* Guided Help Tour Overlay */}
      <GuidedTour
        steps={TOUR_STEPS}
        active={tourActive}
        onClose={() => setTourActive(false)}
      />
    </div>
  );
}
