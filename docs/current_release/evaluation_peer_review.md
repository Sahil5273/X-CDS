# Academic Peer Review: X-CDS RAG Evaluation Methodology and Codebase

This document compiles a rigorous peer-review analysis of the Explainable Clinical Decision Support (X-CDS) RAG evaluation pipeline. It highlights potential flags that a peer reviewer for a medical informatics journal (e.g., *Journal of Biomedical Informatics*, *JAMIA*) would raise, along with recommendations to address them.

---

## 1. Evaluation Methodology & Metrics

### 🚩 Flag A: Missing "Context Recall" Metric
* **Issue:** The current evaluation in `ragas_eval.py` measures `faithfulness` (hallucination rate), `answer_relevancy` (question alignment), and `context_precision` (retrieval ranking quality). It completely omits **Context Recall**.
* **Reviewer Concern:** In clinical decision support, omitting crucial evidence (false negatives) is as dangerous as hallucinations (false positives). Without measuring *Context Recall*, you cannot statistically prove that the hybrid search (ChromaDB + BM25) is successfully capturing all necessary clinical facts present in the reference ground truth guidelines.
* **Recommendation:** Import and include `context_recall` in [ragas_eval.py](file:///d:/X-CDS/backend/app/eval/ragas_eval.py#L123-L125).

### 🚩 Flag B: LLM-as-a-Judge Bias (Self-Evaluation Bias)
* **Issue:** The pipeline uses `gemini-3.5-flash` for answer generation, and the Ragas evaluator also uses `gemini-3.5-flash` (via `settings.gemini_model`) to judge those answers.
* **Reviewer Concern:** Evaluating a model using the same model family introduces "self-evaluation bias." Models tend to score their own outputs higher and fail to catch their own stylistic anomalies.
* **Recommendation:** 
  1. Configure the evaluator LLM in [ragas_eval.py](file:///d:/X-CDS/backend/app/eval/ragas_eval.py#L131-L138) to use a larger, more objective model (such as `gemini-3.5-pro`) regardless of the generator model.
  2. Implement a small clinical audit: Have a human expert grade 30–50 samples and compute the correlation (e.g., Cohen's Kappa or Pearson correlation) between Ragas scores and human expert grades to validate Ragas as a proxy.

---

## 2. Codebase Design & Hardcoded Variables

### 🚩 Flag C: Hardcoded Evaluator Embeddings
* **Issue:** In `ragas_eval.py`, the embedding model is hardcoded as `models/text-embedding-004` on line 141:
  ```python
  GoogleGenerativeAIEmbeddings(
      model="models/text-embedding-004",
      vertexai=True,
      ...
  )
  ```
* **Reviewer Concern:** This violates modular configuration principles. If the user changes `EMBEDDING_MODEL_NAME` in settings, the Ragas evaluator will still run on Google's cloud embeddings instead of matching the pipeline's active embedding model.
* **Recommendation:** Dynamically map the embedding model from `Settings` if possible, or make it a configurable environment variable `EVAL_EMBEDDING_MODEL` in `.env`.

### 🚩 Flag D: Validation Error Cascading (Fixed)
* **Issue:** Previously, when the generator node failed to output markdown citations, it threw a Pydantic `ValidationError` which terminated the entire execution thread.
* **Reviewer Concern:** Academic papers evaluating LangGraph loops must show *why* the loop ran and *how* it corrected errors. Programmatic crashes prevent the self-correction loop from executing.
* **Recommendation:** (Now fixed) We wrapped output assembly in a try-except block so that formatting validation errors are treated as soft state errors, routing them cleanly to the LangGraph correction edge.

---

## 3. Reporting Standards & Clinical Validity (TRIPOD+AI)

### 🚩 Flag E: Sample Size Insignificance
* **Issue:** The codebase currently defaults to benchmarking against `ragas_eval_sample.jsonl`, which contains only $N=3$ cases.
* **Reviewer Concern:** A sample size of $N=3$ has zero statistical significance. Reviewers will reject any claim of "hallucination mitigation" or "faithfulness benchmarking" without a dataset size of at least $N=50$ (preferably $N=100+$) sourced from clinical datasets like MedQA or PubMedQA.
* **Recommendation:** Clearly state in your paper that the $N=3$ run is a pipeline validation smoke test, and provide the results of a larger scale run ($N \ge 100$) using the provided scripts.

### 🚩 Flag F: Citation Overlap vs Semantic Truth (A key thesis discussion)
* **Issue:** Your pipeline uses a hard overlap validation check (`CITATION_MIN_TOKEN_OVERLAP = 0.25`), whereas Ragas uses LLM-based NLI (Natural Language Inference) to score `faithfulness`.
* **Reviewer Concern:** What happens when the LLM generates a semantically true statement using synonyms (0% token overlap) but the validator rejects it?
* **Recommendation:** Use this as a key discussion point in your manuscript. Highlight how the deterministic 25% overlap threshold enforces strict verbatim safety at the cost of slight generation flexibility, and compare your programmatic loop failures against the semantic scores calculated by Ragas.
