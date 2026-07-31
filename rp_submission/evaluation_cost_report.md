# Google Cloud Credit Utilization Report

This report outlines the Google Cloud Vertex AI service utilization and costs associated with the development, pilot evaluation, and full-scale parameter sweep of the **Explainable Clinical Decision Support (X-CDS)** system.

## Cumulative Summary

*   **Total Spent (Cumulative):** **10,896.38 INR** (approx. **$131.28 USD**)
*   **Target GCP Project:** `x-cds-502821`
*   **Billing Cycle:** July & August 2026
*   **Vertex AI Service ID:** `C7E2-9256-1C43`

---

## GCP Invoice Details

Below is the verified billing extract for all project usage in the Google Cloud console:

| Service Description | Service ID | Cost (₹) | Savings (₹) | Subtotal (₹) | Status / Change |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vertex AI** | `C7E2-9256-1C43` | 10,896.38 | 0.00 | 10,896.38 | Cumulative Total |
| **Tax** | - | 0.00 | - | 0.00 | - |
| **Total Filtered Cost** | - | **10,896.38** | **0.00** | **10,896.38** | - |

---

## Detailed Step-by-Step Cost Breakdown

The total cost of **₹10,896.38** is divided into the following research phases:

### Phase 1: Ingestion, Testset Generation & Pilot Setup (July 2026)
*   **Subtotal Cost:** **~4,000.00 INR (approx. $48.20 USD)**
*   **Description:** Building the infrastructure, ingestion pipeline, and generating the clinical evaluation dataset.

#### Step 1: Ingestion & Local Indexing
*   **Cost:** **0.00 INR ($0.00 USD)**
*   **Details:** BioC XML text parsing, local text chunking, BAAI/bge-small embedding generation, and ChromaDB vector/BM25 indexing were executed locally on CPU/GPU, incurring zero cloud API expenses.

#### Step 2: Ragas Synthetic Clinical Testset Generation
*   **Cost:** **~1,800.00 INR ($21.50 USD)**
*   **API Calls:** ~3,200 calls to `gemini-2.5-pro` (Vertex AI).
*   **Details:** Multi-pass question formulation, clinical scenario synthesis, and critique loops.

#### Step 3: Baseline & X-CDS Pilot Evaluation (N=45)
*   **Cost:** **~1,600.00 INR ($19.20 USD)**
*   **API Calls:** Ragas evaluation metrics calculations (faithfulness, relevancy, precision, recall) on the pilot dataset.

#### Step 4: Sandbox & Setup
*   **Cost:** **~600.00 INR ($7.50 USD)**
*   **Details:** Fixing API rate limiters and configuring caching layers.

---

### Phase 2: Full-Scale Parameter Sweep & Baselines (August 2026)
*   **Subtotal Cost:** **~6,896.38 INR (approx. $83.08 USD)**
*   **Description:** Running a parameter sweep ($T_{min} = 0.10, 0.15, 0.25, 0.50$), Baseline RAG, and Vanilla RAG evaluations ($N = 100$) across 600 total query executions.

#### Phase 2 Cost Drivers:
1.  **Ragas Evaluator API (`gemini-2.5-pro`):** Accounts for **~80% of Phase 2 costs** (approx. **₹5,500.00**). Ragas uses LLM-as-a-judge for evaluating Faithfulness, Answer Relevancy, Context Precision, and Context Recall. Because Gemini 2.5 Pro is a high-cost tier model, generating scores for 600 runs dominates the budget.
2.  **Generative Model API (`gemini-3.5-flash`):** Accounts for **~20% of Phase 2 costs** (approx. **₹1,396.38**). Despite evaluating 600 query-runs—some of which triggered multiple retry loops under strict thresholds like $T_{min} = 0.50$—the cost of `gemini-3.5-flash` remains exceptionally low, demonstrating the architecture's financial feasibility.
3.  **Local Indexing & Caching:** Caching minimized duplicate model hits during code refinement, avoiding redundant API fees.

---

## Production Cost Projection (Economic Feasibility)

While running extensive multi-metric offline benchmarks (using expensive evaluators) incurs significant one-time fees, running the system in production is highly economical:
*   **Operational Cost per Query:** **~₹0.014 ($0.00017 USD)**
*   **Average Token Footprint:** ~1,500 input tokens / ~200 output tokens.
*   **Why it's cheap:**
    1.  **Lightweight Generator:** Production uses `gemini-3.5-flash`, which has an extremely low cost per million tokens.
    2.  **No Cloud Database Costs:** Document indexing and retrieval are handled locally by ChromaDB and BM25.
    3.  **No Evaluator Calls:** The Ragas evaluator (`gemini-2.5-pro`) is only run during offline development/benchmarking, not in production.
