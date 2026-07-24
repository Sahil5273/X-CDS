# X-CDS: Explainable Clinical Decision Support System

X-CDS is a state-of-the-art Clinical Decision Support System designed to assist medical practitioners by retrieving and synthesizing clinical literature with mathematically guaranteed factual grounding. 

By integrating a stateful self-correction agentic loop, X-CDS addresses the core challenge of large language model (LLM) hallucinations in critical medical environments.

## Core Features & Architecture

*   **Hybrid Ingestion & Retrieval:** Merges dense semantic embeddings (ChromaDB + `text-embedding-004`) with sparse lexical search (BM25) using Reciprocal Rank Fusion (RRF).
*   **Neural Re-ranking:** Prioritizes retrieved passages using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to maximize context precision.
*   **Stateful Agentic Guardrails (LangGraph):** Orchestrates a multi-turn evaluation loop that programmatically cross-references generated answers against source literature, retrying automatically if a citation cannot be strictly verified.
*   **Explainable UI:** Displays medical responses with inline interactive citations mapped directly to source evidence chunks.

## Tech Stack
*   **Backend:** FastAPI, Python 3.11, LangChain, LangGraph
*   **Frontend:** React, TypeScript, TailwindCSS, Vite
*   **Database:** Chroma Vector Store
*   **LLMs:** Google Cloud Vertex AI (Gemini 2.5 Pro & 3.5 Flash)


## Architecture

```text
BioC ingest -> Chroma (dense) + BM25 (sparse) -> RRF fusion
  -> cross-encoder re-rank (top 5) -> LangGraph + Gemini generation
  -> citation guardrails / self-correction -> FastAPI -> React UI
```

## Prerequisites

- Python 3.11+
- Node.js 20+ (for local frontend dev)
- Docker + Docker Compose (optional, recommended for one-command startup)
- A Google Gemini API key (`GOOGLE_API_KEY`) for live generation

## Quick start (Docker Compose)

1. Copy environment template and set your API key:

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=...
```

2. Build and start backend + frontend:

```bash
docker compose up --build
```

3. Open the UI:

- Frontend: http://localhost:5173
- Backend health: http://localhost:8000/api/v1/health

On first startup the backend bootstraps local indexes from the bundled BioC fixture when `data/` is empty.

Run the offline smoke test inside Docker after services are up:

```bash
docker compose --profile smoke run --rm smoke
```

## Local development runbook

### 1. Backend setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
cp .env.example .env
```

Set `GOOGLE_API_KEY` in `.env` before running live queries.

### 2. Ingest biomedical passages

Offline fixture (recommended for local/dev):

```bash
python -m scripts.ingest_bioc --mock-file tests/fixtures/bioc_sample.json
```

Live PMC BioC fetch (respects 3 req/s limit):

```bash
python -m scripts.ingest_bioc --pmcid PMC1234567
```

Output: `data/bioc_chunks.jsonl`

### 3. Build retrieval indexes

Dense Chroma index:

```bash
python -m scripts.index_chroma --reset
```

Sparse BM25 corpus:

```bash
python -m scripts.index_bm25
```

Or run both via bootstrap helper (requires Hugging Face model download):

```bash
python -m scripts.bootstrap_indexes --reset
```

### 4. Start API server

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Query example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"How does hybrid retrieval support clinical decision support?\"}"
```

(macOS/Linux: replace `^` line continuations with `\`.)

### 5. Start frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Vite proxies `/api` to the backend.

### 6. Evaluate with Ragas and Baseline RAG

We run evaluations against the large clinical dataset using the standard Ragas framework (using Gemini 2.5 Pro as the judge and Gemini 3.5 Flash as the pipeline generator).

**A. Generate the Clinical Dataset:**
Synthesize clinical queries and ground-truth answers from the ingested BioC corpus:
```bash
python -m scripts.generate_clinical_dataset --count 45 --output data/my_eval_set_large.jsonl
```

**B. Evaluate the X-CDS Pipeline (with Stateful Guardrails):**
Run the 45-case evaluation. This script caches pipeline answers in `data/materialized_predictions.jsonl` to avoid redundant LLM calls on subsequent runs:
```bash
python -m scripts.evaluate_ragas --dataset data/my_eval_set_large.jsonl --use-pipeline
```
*Report output:* `data/ragas_report.json`

**C. Evaluate the Baseline RAG Pipeline (without Guardrail Loop):**
Run the evaluation with the LangGraph self-correction loop bypassed (forces `max_generation_attempts=1`):
```bash
python -m scripts.evaluate_baseline --dataset data/my_eval_set_large.jsonl
```
*Report output:* `data/baseline_ragas_report.json`

**D. Compile and Open the Comparison Dashboard:**
Merge the results and open the interactive comparison interface in your browser:
```bash
# Compile the HTML file
python -m scripts.build_dashboard

# Open in default browser
# Windows (PowerShell):
Start-Process "docs/evaluation_dashboard.html"
```

## End-to-end smoke test

Offline smoke test (no Gemini key required; uses fake LLM + fixture indexes):

```bash
python -m scripts.run_smoke_test
```

Unit test form:

```bash
python -m unittest tests.test_e2e_smoke -v
```

Optional live smoke against real Gemini (requires `GOOGLE_API_KEY`, indexes, and network):

```bash
set RUN_LIVE_SMOKE=1
python -m scripts.run_smoke_test --live
```

## API endpoints

| Method | Path               | Description                                  |
|--------|--------------------|----------------------------------------------|
| GET    | `/api/v1/health`   | Service health metadata                      |
| POST   | `/api/v1/query`    | Run full X-RAG pipeline for a clinical query |

`POST /api/v1/query` body:

```json
{ "query": "symptom or clinical question" }
```

Response includes `answer`, `citations`, `contexts`, `validation_passed`, and `generation_attempts`.

## Project layout

```text
backend/app/
  ingestion/   BioC parsing + rate-limited fetch
  vector/      Chroma dense store + embeddings
  search/      BM25 + RRF hybrid retrieval
  rerank/      cross-encoder context refinement
  llm/         LangGraph + Gemini generation
  guardrail/   citation validation + retry loop
  pipeline/    end-to-end orchestration service
  api/         FastAPI routes
  eval/        Ragas helpers
frontend/      React + Tailwind split-screen UI
scripts/       ingest, index, evaluate, smoke test
tests/         unit + integration + e2e smoke tests
```

## Testing

Run full backend test suite:

```bash
python -m unittest discover -s tests -v
```

Build frontend:

```bash
cd frontend
npm run build
```

## Troubleshooting

- **`No supporting passages were retrieved`**: run ingest + index bootstrap steps.
- **`GOOGLE_API_KEY is required`**: set key in `.env` for live generation.
- **First Chroma/BM25 run is slow**: Hugging Face models download on first use.
- **Docker startup delay**: backend healthcheck waits for bootstrap + model warmup.

## License

Internal research / development use. Verify clinical outputs with qualified professionals before operational use.
