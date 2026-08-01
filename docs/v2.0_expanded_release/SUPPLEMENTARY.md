# Supplementary Materials — X-CDS

This directory contains the supplementary materials, replication code instructions, and evaluation dataset artifacts for the paper: **"Explainable Clinical Decision Support (X-CDS): Mitigating LLM Hallucinations in High-Stakes Medicine via Hybrid Retrieval and Deterministic Citation Verification Frameworks"**.

---

## 1. Project Repository & Demo Routes

*   **GitHub Repository:** [https://github.com/Sahil5273/X-CDS](https://github.com/Sahil5273/X-CDS)
*   **Web Application Local Access Routes:**
    *   `/` — **Clinical Portal**: Main user-facing interface with preset clinical queries, elapsed timing step loader, and citation-to-evidence panel mapping.
    *   `/interactive` — **Interactive Playground**: Parameter adjustment dashboard allowing real-time tuning of the Reranker model and citation verification token overlap threshold ($T_{min}$).
    *   `/report` — **Evaluation Report**: Dynamic analytical dashboard displaying pipeline metrics across Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

---

## 2. Evaluation Data Artifacts

The benchmarks in the paper are derived from the following persistent JSON logs in the `data/` directory:
*   `data/my_eval_set_large.jsonl` — The primary clinical dataset comprising $N=100$ arboviral patient scenarios (Zika, Dengue, Chikungunya, and West Nile Virus).
*   `data/naive_ragas_report.json` — Evaluator report for the **Naive RAG** baseline (Dense embeddings only, no re-ranking, no guardrails).
*   `data/baseline_ragas_report.json` — Evaluator report for the **Hybrid RAG** baseline (Sparse BM25 + Dense Chroma + RRF + MiniLM re-ranking, no guardrails).
*   `data/ragas_report_t10.json` — Evaluator report for **X-CDS RAG** at the optimal safety default ($T_{min}=0.10$).
*   `data/guardrail_metrics_summary.json` — Aggregated pipeline telemetry metrics (generation loops, abstention rates, mean citations).
*   `data/high_risk_safety_subset.json` — Domain safety proxy evaluation logs over 12 curated clinical scenarios.
*   `data/faithfulness_wilcoxon_summary.json` — Wilcoxon signed-rank significance outputs comparing faithfulness scores.

---

## 3. Reproduction Shell Commands

Ensure your environment is set up (dependencies installed via `pip install -r requirements.txt` and `.env` configured). Run the following commands in the project root:

```bash
# 1. Reproduce Naive & Hybrid Baselines (Table I Columns 1 & 2)
python scripts/evaluate_baseline.py

# 2. Reproduce X-CDS RAG Metrics (Table I Column 3 & Table II Peak row)
python scripts/evaluate_ragas.py --use-pipeline

# 3. Reproduce Wilcoxon statistics and Pipeline metrics (Table IV)
python -m scripts.compute_faithfulness_stats

# 4. Reproduce High-Risk Safety Proxy Metrics JSON log
python -m scripts.curate_safety_subset
```

---

## 4. 3–5 Minute Demo Video Script Outline

This script provides an outline for a video walk-through demonstrating X-CDS to a technical/medical audience:

### Phase 1: The Clinical Safety Challenge (0:00 - 1:00)
*   **Action:** Show the landing page of the **Clinical Portal** (`/`). Type a raw clinical query or highlight the "Dengue vs Chikungunya Differential" dilemma.
*   **Voice-over:** *"In high-stakes medicine, Large Language Models often generate plausible but dangerous hallucinations. For instance, recommending Ibuprofen for joint pain when Dengue has not been ruled out can cause fatal hemorrhages. Explainable Clinical Decision Support (X-CDS) mitigates this risk by grounding every sentence in verified source text."*

### Phase 2: Preset Vignettes & Verifiable Citations (1:00 - 2:15)
*   **Action:** Click the **Clinical Presets** dropdown. Select the *Dengue Shock Syndrome Critical Warning Signs* preset and click "Ask X-CDS". Highlight the pulsing timing steps (`Retrieving...` &rarr; `Reranking...` &rarr; `Generating...` &rarr; `Verifying...`).
*   **Voice-over:** *"Using our presets, clinicians can instantly trigger typical diagnostic queries. X-CDS demonstrates a timing step loader that shows the backend hybrid retrieval, re-ranking, and citation-verification stages in real-time. Notice how each sentence is marked with a bracketed citation. Clicking a citation directly highlights and scrolls to its corresponding verified passage in the evidence panel."*

### Phase 3: Guardrail Self-Correction Loops & Telemetry (2:15 - 3:30)
*   **Action:** Scroll down to display the **Pipeline Telemetry** panel. Point to the *Validation Status* badge (PASSED) and the *Self-Correction Loops* attempt counter.
*   **Voice-over:** *"Behind the scenes, LangGraph statefully orchestrates a verification loop. If the LLM generates a claim that falls below our token-overlap threshold ($T_{min}$), the guardrail intercepts it, appends the verbatim mismatch logs, and sends it back to the generator node for self-correction. The telemetry panel shows this retry count and lists resolved alignment errors."*

### Phase 4: Interactive Playground & Parameter Sweeps (3:30 - 5:00)
*   **Action:** Navigate to the **Interactive Playground** (`/interactive`). Show the *Min Token Overlap* slider and the *Cross-Encoder* model selector.
*   **Voice-over:** *"For researchers, the interactive playground allows real-time adjustment of thresholds. Changing the n-value slider shows how strictness impacts generation retry loops and safe abstentions, finding the mathematical sweet spot of 0.10 for medical decision-making."*
