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
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);

  async function handleSubmit() {
    const cleaned = query.trim();
    if (!cleaned) {
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await queryXcds(cleaned);
      setResult(response);
      setActiveCitation(response.cited_indices[0] ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  function handleDemo() {
    setQuery(DEMO_RESPONSE.query);
    setResult(DEMO_RESPONSE);
    setActiveCitation(1);
    setError(null);
  }

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
      <header className="rise-in mb-6 lg:mb-8">
        <p className="mb-2 text-sm font-medium tracking-[0.18em] text-[var(--accent-deep)] uppercase">
          Explainable RAG
        </p>
        <h1
          className="text-[clamp(2.6rem,7vw,4.6rem)] leading-[0.95] text-[var(--ink)]"
          style={{ fontFamily: "var(--font-display)" }}
        >
          X-CDS
        </h1>
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

          {error ? (
            <p className="mt-4 text-sm text-[var(--danger)]" role="alert">
              {error}
            </p>
          ) : null}

          <div className="mt-8 border-t border-[var(--line)] pt-6">
            <div className="mb-3 flex items-end justify-between gap-3">
              <h2
                className="text-2xl text-[var(--ink)]"
                style={{ fontFamily: "var(--font-display)" }}
              >
                Answer
              </h2>
              {result ? (
                <span className="text-xs tracking-wide text-[var(--muted)]">
                  {result.validation_passed ? "Citations verified" : "Needs review"}
                </span>
              ) : null}
            </div>
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
    </div>
  );
}
