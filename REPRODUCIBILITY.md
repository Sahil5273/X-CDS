# X-CDS Reproducibility Guide

This guide provides step-by-step instructions to replicate the benchmarks and evaluation sweeps for the X-CDS (Explainable Clinical Decision Support) system.

---

## 1. System Environment & Prerequisites

### A. Python Environment
All experiments are run using **Python 3.11**.
To set up the environment and install dependencies:
```powershell
# Create virtual environment
python -m venv .venv

# Activate and install packages
.venv\Scripts\activate
.venv\Scripts\pip install -r requirements.txt
```

### B. Environment Variables (`.env`)
Create a `.env` file in the root of the repository matching the following configuration:
```ini
APP_NAME=X-CDS
APP_ENV=local

# Google Cloud Platform credentials for Vertex AI (Gemini and embeddings)
GCP_PROJECT_ID=x-cds-502821
GCP_REGION=global

# Decoupled generation and evaluation models
GEMINI_MODEL=gemini-3.5-flash
EVAL_LLM_MODEL=gemini-2.5-pro
EVAL_EMBEDDING_MODEL=models/text-embedding-004

# Dense indexing model
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
CROSS_ENCODER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## 2. Core Model Specifications & Versions

| Component | Model Name / Version | Hosting / Provider |
| :--- | :--- | :--- |
| **Generation Model** | `gemini-3.5-flash` | Google Cloud Vertex AI (`global` region) |
| **Evaluator Judge Model** | `gemini-2.5-pro` | Google Cloud Vertex AI (`global` / `us-central1`) |
| **Evaluator Embedding** | `models/text-embedding-004` | Google Cloud Vertex AI |
| **Dense Vector Embeddings** | `BAAI/bge-small-en-v1.5` | Local `sentence-transformers` (PyTorch) |
| **Baseline Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local `sentence-transformers` (PyTorch) |
| **Upgraded Reranker** | `BAAI/bge-reranker-v2-m3` | Local `sentence-transformers` (PyTorch) |

---

## 3. Running the Evaluations

### A. Quick 3-Case Smoke Test Run
To run a fast, low-cost verification of the entire evaluation pipeline (RAG generation + Ragas scoring) for both rerankers:
```powershell
.venv\Scripts\python.exe docs/retrieval_evaluation/run_full_evaluation.py --limit 3 --n-value 0.25
```
*Expected execution time: ~1.5 minutes. Generates `docs/retrieval_evaluation/evaluation_summary_report_3.md`.*

### B. Full 100-Case Evaluation Sweep
To run the full clinical differential diagnosis suite (100 scenarios) comparing the two rerankers:
```powershell
.venv\Scripts\python.exe docs/retrieval_evaluation/run_full_evaluation.py --limit 100 --n-value 0.10
```
*Expected execution time: ~10 minutes. Generates `docs/retrieval_evaluation/evaluation_summary_report_100.md`.*

---

## 4. Replicating the PDF Report
To generate the formatted clinical benchmarks PDF report (`docs/evaluation_report.pdf`):
```powershell
.venv\Scripts\python.exe scripts/generate_pdf_report.py
```
