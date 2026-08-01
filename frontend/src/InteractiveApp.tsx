import { useState, useRef, useEffect } from "react";
import { queryXcds } from "./api";
import type { QueryResponse } from "./api";

// Reranker model options
const MODEL_OPTIONS = [
  { value: "cross-encoder/ms-marco-MiniLM-L-6-v2", label: "MiniLM-L-6-v2 (Default / 90MB)" },
  { value: "BAAI/bge-reranker-v2-m3", label: "BGE-Reranker-v2-m3 (Upgraded / 567MB)" }
];

export function InteractiveApp() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);

  // Configuration options
  const [selectedModel, setSelectedModel] = useState(MODEL_OPTIONS[0].value);
  const [nValue, setNValue] = useState(0.25);

  const citationRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (activeCitation !== null) {
      citationRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [activeCitation]);

  async function handleQuerySubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);
    setResponse(null);
    setActiveCitation(null);

    try {
      const res = await queryXcds(trimmed, selectedModel, nValue);
      setResponse(res);
      if (res.cited_indices.length > 0) {
        setActiveCitation(res.cited_indices[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  function handleDemoLoad() {
    setQuery("What specific neurodevelopmental processes does Zika virus disrupt according to mass spectrometry?");
    setSelectedModel("BAAI/bge-reranker-v2-m3");
    setNValue(0.25);
  }

  // Regex pattern to extract citation markers in response
  const citationRegex = /(\[\d+\])/g;

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
      {/* Header */}
      <header className="rise-in mb-6 lg:mb-8">
        <div className="flex items-start justify-between">
          <div>
            <p className="mb-2 text-sm font-medium tracking-[0.18em] text-[var(--accent-deep)] uppercase">
              X-CDS Playground
            </p>
            <h1 className="text-[clamp(2.6rem,7vw,4.6rem)] leading-[0.95] text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>
              Interactive RAG
            </h1>
          </div>
          <a
            href="/"
            className="rounded-xl border border-[var(--line)] bg-white/60 px-4 py-2.5 text-xs font-semibold text-[var(--muted)] transition hover:border-[var(--accent)] hover:bg-white hover:text-[var(--accent-deep)]"
          >
            &larr; Back to Clinical Portal
          </a>
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
            <div className="flex flex-wrap gap-3 pt-2">
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
                Load Demo
              </button>
            </div>
          </form>

          {error && (
            <p className="mt-4 rounded-xl bg-red-50 border border-red-200 p-3 text-sm text-[var(--danger)]" role="alert">
              <strong>Error:</strong> {error}
            </p>
          )}

          {/* RAG Loop Verification Dashboard */}
          {response && (
            <div className="mt-8 border-t border-[var(--line)] pt-6">
              <h3 className="mb-4 text-lg font-bold text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>
                Pipeline Telemetry
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-xl border border-[var(--line)] bg-white/40 p-4">
                  <span className="block text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">
                    Validation Status
                  </span>
                  <span className={`mt-1 inline-flex items-center gap-1.5 text-lg font-semibold ${response.validation_passed ? "text-green-600" : "text-amber-600"}`}>
                    <span className={`h-2.5 w-2.5 rounded-full ${response.validation_passed ? "bg-green-600" : "bg-amber-600"}`} />
                    {response.validation_passed ? "PASSED" : "FAILED"}
                  </span>
                </div>
                <div className="rounded-xl border border-[var(--line)] bg-white/40 p-4">
                  <span className="block text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">
                    Self-Correction Loops
                  </span>
                  <span className="block mt-1 text-2xl font-bold text-[var(--ink)]">
                    {response.generation_attempts} {response.generation_attempts === 1 ? "attempt" : "attempts"}
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
          <div className="rise-in rounded-[1.4rem] border border-[var(--line)] bg-[var(--panel)] p-5 backdrop-blur-sm sm:p-6">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-2xl text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>Generated Answer</h2>
              {response && (
                <span className="text-xs text-[var(--muted)] bg-white/60 px-2 py-0.5 rounded-md border border-[var(--line)]">
                  Gemini-3.5-Flash
                </span>
              )}
            </div>

            {response ? (
              <div className="space-y-4">
                <p className="text-[1.02rem] leading-8 text-[var(--ink)]">
                  {response.answer.split(citationRegex).map((part, idx) => {
                    const isMatch = part.match(/^\[(\d+)\]$/);
                    if (!isMatch) return part;
                    const index = Number(isMatch[1]);
                    const isActive = activeCitation === index;
                    return (
                      <button
                        key={idx}
                        type="button"
                        className={`mx-0.5 inline-flex min-w-7 items-center justify-center rounded-md px-1.5 py-0.5 align-baseline text-[0.8rem] font-bold transition duration-200 ${
                          isActive
                            ? "bg-[var(--accent)] text-white"
                            : "bg-[var(--highlight)] text-[var(--accent-deep)] hover:bg-[var(--accent)] hover:text-white"
                        }`}
                        onClick={() => setActiveCitation(index)}
                      >
                        {part}
                      </button>
                    );
                  })}
                </p>
                <p className="text-xs text-[var(--muted)] italic">
                  Click on citation markers to inspect and highlight the corresponding source passage.
                </p>
              </div>
            ) : (
              <p className="text-[var(--muted)] text-[0.95rem]">
                Answers will appear here along with verified citations.
              </p>
            )}
          </div>

          {/* Evidence Chunks with Score Visualizer */}
          <div className="rise-in flex-1 rounded-[1.4rem] border border-[var(--line)] bg-[var(--panel)] p-5 backdrop-blur-sm sm:p-6" style={{ animationDelay: "80ms" }}>
            <div className="mb-5">
              <h2 className="text-2xl text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>Retrieved Evidence</h2>
              <p className="mt-1 text-sm text-[var(--muted)]">Source passages matched and scored by the cross-encoder.</p>
            </div>

            {response && response.contexts.length > 0 ? (
              <div className="flex flex-col gap-4 max-h-[500px] overflow-y-auto pr-1">
                {response.contexts.map((context) => {
                  const isActive = activeCitation === context.index;
                  
                  
                  return (
                    <article
                      key={context.chunk_id}
                      ref={isActive ? citationRef : undefined}
                      className={`rounded-xl border px-4 py-4 transition duration-300 ${
                        isActive
                          ? "chunk-flash border-[var(--highlight-ring)] bg-[var(--highlight)]"
                          : "border-[var(--line)] bg-white/55"
                      }`}
                    >
                      <header className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
                        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                          <span className={`inline-flex min-w-8 items-center justify-center rounded-md px-2 py-0.5 text-sm font-semibold ${
                            isActive ? "bg-[var(--accent)] text-white" : "bg-[var(--panel-strong)] text-[var(--accent-deep)]"
                          }`}>
                            [{context.index}]
                          </span>
                          <span className="text-sm font-bold text-[var(--ink)]">
                            {context.section || "Passage"}
                          </span>
                          {context.pmcid && (
                            <span className="text-xs text-[var(--muted)]">
                              {context.pmcid}
                            </span>
                          )}
                        </div>
                        {context.score !== undefined && (
                          <div className="flex items-center gap-1.5">
                            <span className="text-[0.7rem] font-bold text-[var(--muted)] uppercase tracking-wider">Match:</span>
                            <span className="text-xs font-bold text-[var(--accent-deep)]">{(context.score * 100).toFixed(1)}%</span>
                          </div>
                        )}
                      </header>
                      
                      {context.score !== undefined && (
                        <div className="mb-3 h-1 w-full rounded-full bg-[var(--line)] overflow-hidden">
                          <div
                            className="h-full bg-[var(--accent)]"
                            style={{ width: `${Math.max(0, Math.min(100, context.score * 100))}%` }}
                          />
                        </div>
                      )}
                      
                      <p className="text-[0.95rem] leading-7 text-[var(--ink)]">
                        {context.text}
                      </p>
                      
                      {context.source_url && (
                        <a
                          className="mt-3 inline-block text-sm text-[var(--accent-deep)] underline-offset-2 hover:underline"
                          href={context.source_url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open PMC source
                        </a>
                      )}
                    </article>
                  );
                })}
              </div>
            ) : (
              <p className="text-[var(--muted)] text-[0.95rem]">
                Supporting passages will appear here with cross-encoder scores.
              </p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
