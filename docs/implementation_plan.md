# Add Context Recall and Configure Dynamic Evaluator Embeddings

## Problem Description
To satisfy peer review requirements:
1. **Context Recall:** We need to include the `context_recall` metric in our Ragas evaluation pipeline to verify context completeness.
2. **Dynamic Embeddings:** The evaluator embeddings model name must be loaded dynamically from the project's settings or environment variables instead of using the hardcoded `"models/text-embedding-004"` string.

## Proposed Changes

### Configuration Updates

#### [MODIFY] [settings.py](file:///d:/X-CDS/backend/app/config/settings.py)
* Add `eval_embedding_model` field to the `Settings` class to specify the embedding model used for evaluation. It defaults to `"models/text-embedding-004"`.

#### [MODIFY] [.env](file:///d:/X-CDS/.env) & [.env.example](file:///d:/X-CDS/.env.example)
* Add `EVAL_EMBEDDING_MODEL=models/text-embedding-004` to the environment templates.

---

### Implementation Code

#### [MODIFY] [ragas_eval.py](file:///d:/X-CDS/backend/app/eval/ragas_eval.py)
* Import `context_recall` from `ragas.metrics` and add it to the `evaluate()` metrics list.
* Update `GoogleGenerativeAIEmbeddings` constructor to use `settings.eval_embedding_model` instead of the hardcoded string.

## Verification Plan

### Automated Tests
* Run the existing unit tests:
  ```powershell
  .venv\Scripts\python -m unittest discover -s tests
  ```

### Manual Verification
* Run the Ragas evaluation script to verify it calculates `context_recall` and runs successfully using the dynamic settings:
  ```powershell
  $env:PYTHONPATH="d:\X-CDS"
  .venv\Scripts\python scripts/evaluate_ragas.py --use-pipeline --dataset data/my_eval_set.jsonl
  ```
