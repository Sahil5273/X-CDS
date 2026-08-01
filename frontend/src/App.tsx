import { useState } from "react";
import {
  DEMO_RESPONSE,
  queryXcds,
  type QueryResponse,
} from "./api";
import { AnswerPanel } from "./components/AnswerPanel";
import { EvidencePanel } from "./components/EvidencePanel";
import { QueryForm } from "./components/QueryForm";

export default function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);

  async function handleSubmit() {
    const cleaned = query.trim();
    if (!cleaned) {
      return;
    }

    setLoading(true);
    setElapsed(0);
    setError(null);
    setResult(null);
    setActiveCitation(null);

    const timer = setInterval(() => {
      setElapsed((prev) => prev + 0.1);
    }, 100);

    try {
      const response = await queryXcds(cleaned);
      setResult(response);
      if (response.cited_indices.length > 0) {
        setActiveCitation(response.cited_indices[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      clearInterval(timer);
      setLoading(false);
    }
  }

  function handleDemo() {
    setQuery(DEMO_RESPONSE.query);
    setResult(DEMO_RESPONSE);
    setActiveCitation(1);
    setError(null);
  }

  const isAbstention = result && (
    result.answer.toLowerCase().includes("insufficient evidence") || 
    result.answer.toLowerCase().includes("no evidence")
  );

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 py-6 sm:px-6 lg:px-8 lg:py-8 pb-16">
      <header className="rise-in mb-6 lg:mb-8">
        <div className="flex items-start justify-between">
          <div>
            <p className="mb-2 text-sm font-medium tracking-[0.18em] text-[var(--accent-deep)] uppercase">
              Explainable RAG
            </p>
            <h1
              className="text-[clamp(2.6rem,7vw,4.6rem)] leading-[0.95] text-[var(--ink)]"
              style={{ fontFamily: "var(--font-display)" }}
            >
              X-CDS
            </h1>
          </div>
          <div className="flex gap-2">
            <a
              href="/report"
              className="rounded-xl border border-[var(--line)] bg-white/60 px-4 py-2.5 text-xs font-semibold text-[var(--muted)] transition hover:border-[var(--accent)] hover:bg-white hover:text-[var(--accent-deep)]"
            >
              Evaluation Report
            </a>
            <a
              href="/interactive"
              className="rounded-xl border border-[var(--line)] bg-white/60 px-4 py-2.5 text-xs font-semibold text-[var(--accent-deep)] transition hover:border-[var(--accent)] hover:bg-white"
            >
              Playground &rarr;
            </a>
          </div>
        </div>
        <p className="mt-3 max-w-2xl text-[1.05rem] leading-7 text-[var(--muted)]">
          Clinical decision support with citation-linked evidence. Ask a symptom
          question, then click citation numbers to inspect their source chunks.
        </p>
      </header>

      <main className="grid flex-1 gap-5 lg:grid-cols-2 lg:gap-6">
        <section className="rise-in rounded-[1.4rem] border border-[var(--line)] bg-[var(--panel)] p-5 backdrop-blur-sm sm:p-6">
          <div className="mb-5">
            <h2
              className="text-2xl text-[var(--ink)]"
              style={{ fontFamily: "var(--font-display)" }}
            >
              Query
            </h2>
            <p className="mt-1 text-sm text-[var(--muted)]">
              Enter symptoms or a focused clinical question.
            </p>
          </div>

          <QueryForm
            value={query}
            loading={loading}
            onChange={setQuery}
            onSubmit={() => {
              void handleSubmit();
            }}
            onDemo={handleDemo}
          />

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

          {error ? (
            <p className="mt-4 text-sm text-[var(--danger)]" role="alert">
              {error}
            </p>
          ) : null}

          {/* Unified Telemetry Panel */}
          {result && (
            <div className="mt-6 border-t border-[var(--line)] pt-5">
              <h3 className="text-xs font-bold tracking-wider text-[var(--muted)] uppercase mb-3" style={{ fontFamily: "var(--font-display)" }}>
                Pipeline Telemetry
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-xl border border-[var(--line)] bg-white/40 p-3">
                  <span className="block text-[10px] font-semibold text-[var(--muted)] uppercase tracking-wider">
                    Validation Status
                  </span>
                  <span className={`mt-1 inline-flex items-center gap-1.5 text-sm font-bold ${result.validation_passed ? "text-green-600" : "text-amber-600"}`}>
                    <span className={`h-2 w-2 rounded-full ${result.validation_passed ? "bg-green-600" : "bg-amber-600"}`} />
                    {result.validation_passed ? "PASSED" : "FAILED"}
                  </span>
                </div>
                <div className="rounded-xl border border-[var(--line)] bg-white/40 p-3">
                  <span className="block text-[10px] font-semibold text-[var(--muted)] uppercase tracking-wider">
                    Self-Correction Loops
                  </span>
                  <span className="block mt-1 text-sm font-bold text-[var(--ink)]">
                    {result.generation_attempts} {result.generation_attempts === 1 ? "attempt" : "attempts"}
                  </span>
                </div>
              </div>

              {result.validation_issues.length > 0 && (
                <div className="mt-3 rounded-xl bg-amber-50 border border-amber-200 p-3 text-xs text-amber-800">
                  <span className="block font-bold mb-1">Alignment Issues Resolved:</span>
                  <ul className="list-disc pl-4 space-y-0.5">
                    {result.validation_issues.map((issue: any, idx: number) => (
                      <li key={idx}>{issue.message || JSON.stringify(issue)}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          <div className="mt-8 border-t border-[var(--line)] pt-6">
            <div className="mb-3 flex items-center justify-between gap-3">
              <h2
                className="text-2xl text-[var(--ink)]"
                style={{ fontFamily: "var(--font-display)" }}
              >
                Answer
              </h2>
              {result && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-[var(--muted)] bg-white/60 px-2 py-0.5 rounded border border-[var(--line)] font-semibold">
                    Gemini-3.5-Flash
                  </span>
                  <span className="text-[10px] tracking-wide text-[var(--muted)] uppercase font-semibold">
                    {result.validation_passed ? "Citations verified" : "Needs review"}
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
              answer={result?.answer ?? ""}
              activeCitation={activeCitation}
              onCitationClick={setActiveCitation}
            />
          </div>
        </section>

        <section
          className="rise-in rounded-[1.4rem] border border-[var(--line)] bg-[var(--panel)] p-5 backdrop-blur-sm sm:p-6"
          style={{ animationDelay: "80ms" }}
        >
          <div className="mb-5">
            <h2
              className="text-2xl text-[var(--ink)]"
              style={{ fontFamily: "var(--font-display)" }}
            >
              Evidence
            </h2>
            <p className="mt-1 text-sm text-[var(--muted)]">
              Source chunks mapped to citation numbers in the answer.
            </p>
          </div>
          <EvidencePanel
            contexts={result?.contexts ?? []}
            activeCitation={activeCitation}
          />
        </section>
      </main>

      {/* Sticky Medical Disclaimer Banner */}
      <footer className="fixed bottom-0 left-0 right-0 z-50 border-t border-amber-200 bg-amber-50 py-2.5 text-center text-xs font-semibold text-amber-800 backdrop-blur-md">
        ⚠️ <strong>Research Prototype:</strong> This tool is for demonstration purposes only and must not be used for medical diagnosis, treatment, or clinical decision-making.
      </footer>
    </div>
  );
}
