# X-CDS

**Explainable RAG (X-RAG) for clinical decision support.**

X-CDS ingests biomedical literature (PubMed Central BioC JSON), indexes it with dense + sparse retrieval, re-ranks evidence with a cross-encoder, generates Gemini answers through LangGraph, and enforces deterministic citation guardrails before returning markdown answers with clickable source mappings in the React UI.

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

### 6. Evaluate with Ragas

Using pre-filled sample predictions:

```bash
python -m scripts.evaluate_ragas --dataset tests/fixtures/ragas_eval_sample.jsonl
```

Using live pipeline predictions (requires indexes + `GOOGLE_API_KEY`):

```bash
python -m scripts.evaluate_ragas --dataset tests/fixtures/ragas_eval_sample.jsonl --use-pipeline
```

Report output: `data/ragas_report.json`

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
