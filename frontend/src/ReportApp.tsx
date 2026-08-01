export function ReportApp() {
  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
      {/* Header */}
      <header className="rise-in mb-8 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <p className="mb-2 text-sm font-medium tracking-[0.18em] text-[var(--accent-deep)] uppercase">
            X-CDS Benchmarks
          </p>
          <h1 className="text-[clamp(2.6rem,6vw,4rem)] leading-[0.95] text-[var(--ink)] font-bold" style={{ fontFamily: "var(--font-display)" }}>
            Evaluation Report
          </h1>
          <p className="mt-3 max-w-2xl text-[1.02rem] leading-7 text-[var(--muted)]">
            Factual alignment diagnostics, corpus scaling impact, and parametric sweeps across emerging viral pathogen datasets.
          </p>
        </div>
        
        {/* Navigation Actions */}
        <div className="flex flex-wrap gap-3">
          <a
            href="/"
            className="rounded-xl border border-[var(--line)] bg-white/60 px-4 py-2.5 text-xs font-semibold text-[var(--muted)] transition hover:border-[var(--accent)] hover:bg-white hover:text-[var(--accent-deep)]"
          >
            &larr; Clinical Portal
          </a>
          <a
            href="/interactive"
            className="rounded-xl border border-[var(--line)] bg-white/60 px-4 py-2.5 text-xs font-semibold text-[var(--accent-deep)] transition hover:border-[var(--accent)] hover:bg-white"
          >
            Playground &rarr;
          </a>
        </div>
      </header>

      {/* Main Content Layout */}
      <main className="space-y-8">
        
        {/* Section 1: Database Scale Scaling Impact */}
        <section className="rise-in rounded-[1.4rem] border border-[var(--line)] bg-[var(--panel)] p-5 backdrop-blur-sm sm:p-6">
          <header className="mb-6">
            <h2 className="text-2xl font-bold text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>
              Corpus Scale Scaling Impact
            </h2>
            <p className="text-sm text-[var(--muted)] mt-1">
              Comparing RAG performance when scaling the clinical literature database from 900 passages (v1.0) to 6,900+ passages (v2.0).
            </p>
          </header>

          {/* Cards for scaling highlights */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-6">
            <div className="rounded-2xl border border-[var(--line)] bg-white/50 p-5">
              <span className="block text-xs font-bold text-[var(--muted)] uppercase tracking-wider">Context Precision</span>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-3xl font-extrabold text-[var(--ink)]">68.9%</span>
                <span className="text-sm font-semibold text-green-600">+33.3%</span>
              </div>
              <p className="mt-1 text-xs text-[var(--muted)]">RRF and Cross-Encoder noise filtering efficiency.</p>
            </div>
            
            <div className="rounded-2xl border border-[var(--line)] bg-white/50 p-5">
              <span className="block text-xs font-bold text-[var(--muted)] uppercase tracking-wider">Context Recall</span>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-3xl font-extrabold text-[var(--ink)]">71.8%</span>
                <span className="text-sm font-semibold text-green-600">+6.6%</span>
              </div>
              <p className="mt-1 text-xs text-[var(--muted)]">Broad coverage of complex clinical recommendations.</p>
            </div>

            <div className="rounded-2xl border border-[var(--line)] bg-white/50 p-5">
              <span className="block text-xs font-bold text-[var(--muted)] uppercase tracking-wider">Answer Relevancy</span>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-3xl font-extrabold text-[var(--ink)]">57.8%</span>
                <span className="text-sm font-semibold text-green-600">+7.9%</span>
              </div>
              <p className="mt-1 text-xs text-[var(--muted)]">Precision and focus in diagnostic generation.</p>
            </div>

            <div className="rounded-2xl border border-[var(--line)] bg-white/50 p-5">
              <span className="block text-xs font-bold text-[var(--muted)] uppercase tracking-wider">Faithfulness</span>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-3xl font-extrabold text-[var(--ink)]">93.4%</span>
                <span className="text-sm font-semibold text-green-600">+2.6%</span>
              </div>
              <p className="mt-1 text-xs text-[var(--muted)]">Reduction of medical hallucinations and factual errors.</p>
            </div>
          </div>

          {/* Detailed table */}
          <div className="overflow-x-auto rounded-xl border border-[var(--line)] bg-white/30">
            <table className="w-full border-collapse text-left text-sm text-[var(--ink)]">
              <thead>
                <tr className="border-b border-[var(--line)] bg-white/40 text-xs font-semibold uppercase text-[var(--muted)]">
                  <th className="px-4 py-3">Ragas Metric</th>
                  <th className="px-4 py-3 text-center">v1.0 Baseline RAG (900 Chunks)</th>
                  <th className="px-4 py-3 text-center">v1.0 X-CDS RAG (900 Chunks)</th>
                  <th className="px-4 py-3 text-center">v2.0 Baseline RAG (6,900 Chunks)</th>
                  <th className="px-4 py-3 text-center bg-[var(--highlight)] text-[var(--accent-deep)]">v2.0 X-CDS RAG (6,900 Chunks, n=0.10)</th>
                  <th className="px-4 py-3 text-right">Net Growth (X-CDS)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--line)]">
                <tr>
                  <td className="px-4 py-3.5 font-bold">Faithfulness</td>
                  <td className="px-4 py-3.5 text-center">90.40%</td>
                  <td className="px-4 py-3.5 text-center">90.70%</td>
                  <td className="px-4 py-3.5 text-center">89.78%</td>
                  <td className="px-4 py-3.5 text-center bg-[var(--highlight)] font-bold text-[var(--accent-deep)]">93.37%</td>
                  <td className="px-4 py-3.5 text-right font-semibold text-green-600">+2.67%</td>
                </tr>
                <tr>
                  <td className="px-4 py-3.5 font-bold">Answer Relevancy</td>
                  <td className="px-4 py-3.5 text-center">47.80%</td>
                  <td className="px-4 py-3.5 text-center">49.90%</td>
                  <td className="px-4 py-3.5 text-center">61.17%</td>
                  <td className="px-4 py-3.5 text-center bg-[var(--highlight)] font-bold text-[var(--accent-deep)]">57.81%</td>
                  <td className="px-4 py-3.5 text-right font-semibold text-green-600">+7.91%</td>
                </tr>
                <tr>
                  <td className="px-4 py-3.5 font-bold">Context Precision</td>
                  <td className="px-4 py-3.5 text-center">22.80%</td>
                  <td className="px-4 py-3.5 text-center">35.60%</td>
                  <td className="px-4 py-3.5 text-center">74.09%</td>
                  <td className="px-4 py-3.5 text-center bg-[var(--highlight)] font-bold text-[var(--accent-deep)]">68.94%</td>
                  <td className="px-4 py-3.5 text-right font-semibold text-green-600">+33.34%</td>
                </tr>
                <tr>
                  <td className="px-4 py-3.5 font-bold">Context Recall</td>
                  <td className="px-4 py-3.5 text-center">63.00%</td>
                  <td className="px-4 py-3.5 text-center">65.20%</td>
                  <td className="px-4 py-3.5 text-center">74.25%</td>
                  <td className="px-4 py-3.5 text-center bg-[var(--highlight)] font-bold text-[var(--accent-deep)]">71.83%</td>
                  <td className="px-4 py-3.5 text-right font-semibold text-green-600">+6.63%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Section 2: Parametric Threshold Sweep */}
        <section className="rise-in rounded-[1.4rem] border border-[var(--line)] bg-[var(--panel)] p-5 backdrop-blur-sm sm:p-6" style={{ animationDelay: "60ms" }}>
          <header className="mb-6">
            <h2 className="text-2xl font-bold text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>
              Parametric Threshold Sweep (n = 0.10 to 0.50)
            </h2>
            <p className="text-sm text-[var(--muted)] mt-1">
              Analyzing how citation overlap requirements (n value) affect faithfulness, loop counts, and query failure rates.
            </p>
          </header>

          <div className="grid gap-6 lg:grid-cols-3">
            {/* Table Column */}
            <div className="lg:col-span-2 overflow-x-auto rounded-xl border border-[var(--line)] bg-white/30 h-fit">
              <table className="w-full border-collapse text-left text-sm text-[var(--ink)]">
                <thead>
                  <tr className="border-b border-[var(--line)] bg-white/40 text-xs font-semibold uppercase text-[var(--muted)]">
                    <th className="px-4 py-3">Overlap Threshold (n)</th>
                    <th className="px-4 py-3 text-center">Faithfulness</th>
                    <th className="px-4 py-3 text-center">Answer Relevancy</th>
                    <th className="px-4 py-3 text-center">Avg. Verification Loops</th>
                    <th className="px-4 py-3 text-right">Validation Failures</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)]">
                  <tr>
                    <td className="px-4 py-3.5 font-bold">n = 0.00 (Baseline)</td>
                    <td className="px-4 py-3.5 text-center">89.78%</td>
                    <td className="px-4 py-3.5 text-center">61.17%</td>
                    <td className="px-4 py-3.5 text-center">1.00 attempt</td>
                    <td className="px-4 py-3.5 text-right font-semibold text-green-600">0% (0 / 100)</td>
                  </tr>
                  <tr className="bg-[var(--highlight)] text-[var(--accent-deep)]">
                    <td className="px-4 py-3.5 font-bold">n = 0.10 (Light)</td>
                    <td className="px-4 py-3.5 text-center font-bold">93.37% (Peak)</td>
                    <td className="px-4 py-3.5 text-center">57.81%</td>
                    <td className="px-4 py-3.5 text-center font-bold">1.10 attempts</td>
                    <td className="px-4 py-3.5 text-right font-bold">1% (1 / 100)</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-3.5 font-bold">n = 0.15 (Mild)</td>
                    <td className="px-4 py-3.5 text-center">90.20%</td>
                    <td className="px-4 py-3.5 text-center">59.07%</td>
                    <td className="px-4 py-3.5 text-center">1.25 attempts</td>
                    <td className="px-4 py-3.5 text-right font-semibold text-amber-600">3% (3 / 100)</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-3.5 font-bold">n = 0.25 (Default)</td>
                    <td className="px-4 py-3.5 text-center">89.49%</td>
                    <td className="px-4 py-3.5 text-center">57.31%</td>
                    <td className="px-4 py-3.5 text-center">1.45 attempts</td>
                    <td className="px-4 py-3.5 text-right font-semibold text-amber-700">8% (8 / 100)</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-3.5 font-bold">n = 0.50 (Strict)</td>
                    <td className="px-4 py-3.5 text-center">92.41%</td>
                    <td className="px-4 py-3.5 text-center">57.82%</td>
                    <td className="px-4 py-3.5 text-center">1.95 attempts</td>
                    <td className="px-4 py-3.5 text-right font-semibold text-red-600">15% (15 / 100)</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Analysis Box */}
            <div className="rounded-xl border border-[var(--line)] bg-white/40 p-5 space-y-4">
              <h3 className="font-bold text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>Sweep Insights</h3>
              <p className="text-sm leading-6 text-[var(--muted)]">
                <strong>Optimal Balance (n = 0.10):</strong> Lenient overlap validation triggers self-correction loops when the model creates completely ungrounded facts, while preserving clinical fluid phrasing. It features the lowest loops (1.10) and only a 1% failure rate.
              </p>
              <p className="text-sm leading-6 text-[var(--muted)]">
                <strong>Strictness Penalties (n = 0.25):</strong> Higher overlap thresholds force the LLM into repetitive synomym matching retries, disrupting textual structure and increasing failures to 8%.
              </p>
              <p className="text-sm leading-6 text-[var(--muted)]">
                <strong>Copy-Paste Trap (n = 0.50):</strong> Forces the LLM to copy guidelines verbatim. While this boosts faithfulness, it triggers high failure rates (15%) and latency (1.95 attempts).
              </p>
            </div>
          </div>
        </section>

        {/* Section 3: Reranker Comparison */}
        <section className="rise-in rounded-[1.4rem] border border-[var(--line)] bg-[var(--panel)] p-5 backdrop-blur-sm sm:p-6" style={{ animationDelay: "120ms" }}>
          <header className="mb-6">
            <h2 className="text-2xl font-bold text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>
              Reranker Comparison: MiniLM vs. BGE-Reranker-v2-m3
            </h2>
            <p className="text-sm text-[var(--muted)] mt-1">
              Evaluating the retrieval accuracy benefits of moving from a lightweight 22M parameter model to a high-capacity multilingual reranker.
            </p>
          </header>

          <div className="grid gap-6 sm:grid-cols-2">
            {/* Precision card */}
            <div className="rounded-xl border border-[var(--line)] bg-white/50 p-5 flex flex-col justify-between">
              <div>
                <span className="block text-xs font-bold text-[var(--muted)] uppercase tracking-wider">Context Precision (N = 20)</span>
                <p className="mt-2 text-sm text-[var(--muted)] leading-relaxed">
                  Measures whether the most relevant guidelines are pushed to the top of the context window.
                </p>
              </div>
              <div className="mt-6 flex items-baseline gap-4">
                <div className="space-y-1">
                  <span className="block text-[0.7rem] uppercase tracking-wide text-[var(--muted)]">MiniLM:</span>
                  <span className="text-xl font-bold text-[var(--ink)]">66.86%</span>
                </div>
                <div className="space-y-1">
                  <span className="block text-[0.7rem] uppercase tracking-wide text-[var(--muted)]">BGE-v2-m3:</span>
                  <span className="text-2xl font-extrabold text-[var(--accent-deep)]">69.38%</span>
                </div>
                <span className="text-xs font-bold text-green-600 bg-green-50 px-2 py-1 rounded-md">+2.52%</span>
              </div>
            </div>

            {/* Recall card */}
            <div className="rounded-xl border border-[var(--line)] bg-white/50 p-5 flex flex-col justify-between">
              <div>
                <span className="block text-xs font-bold text-[var(--muted)] uppercase tracking-wider">Context Recall (N = 20)</span>
                <p className="mt-2 text-sm text-[var(--muted)] leading-relaxed">
                  Measures whether all necessary facts from the clinical guidelines are successfully found.
                </p>
              </div>
              <div className="mt-6 flex items-baseline gap-4">
                <div className="space-y-1">
                  <span className="block text-[0.7rem] uppercase tracking-wide text-[var(--muted)]">MiniLM:</span>
                  <span className="text-xl font-bold text-[var(--accent-deep)]">66.67%</span>
                </div>
                <div className="space-y-1">
                  <span className="block text-[0.7rem] uppercase tracking-wide text-[var(--muted)]">BGE-v2-m3:</span>
                  <span className="text-xl font-bold text-[var(--ink)]">63.33%</span>
                </div>
                <span className="text-xs font-bold text-amber-600 bg-amber-50 px-2 py-1 rounded-md">-3.34%</span>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
