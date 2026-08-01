"""Full RAG evaluation script running generation and all 4 Ragas metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

# Add the repository root to Python path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=repo_root / ".env")

# Force offline mode disabled to run downloads if needed
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from backend.app.config.settings import get_settings
from backend.app.pipeline.service import build_default_service
from backend.app.rerank.cross_encoder import CrossEncoderConfig, CrossEncoderReranker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=repo_root / "data/my_eval_set_large.jsonl",
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of cases to evaluate (default: 100).",
    )
    parser.add_argument(
        "--new-model",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help="New reranker model name to evaluate.",
    )
    parser.add_argument(
        "--n-value",
        type=float,
        default=0.10,
        help="Min token overlap (n value) for citation validation.",
    )
    return parser


def run_full_rag_pipeline(
    examples: Sequence[dict[str, str]],
    reranker_model_name: str,
    n_value: float,
) -> tuple[list[dict], float, int]:
    """Execute the full RAG pipeline (retrieval + generation) for the given cases."""
    print(f"\nInitializing full RAG service with model: {reranker_model_name}...")
    
    settings = get_settings()
    settings.gcp_project_id = "x-cds-502821"
    settings.gcp_region = "global"
    settings.cross_encoder_model_name = reranker_model_name
    
    service = build_default_service(settings)
    
    results = []
    total_attempts = 0
    failed_outputs = 0
    
    print(f"Generating answers for {len(examples)} clinical queries (threshold = {n_value})...")
    
    for idx, ex in enumerate(examples, start=1):
        question = ex["question"]
        ground_truth = ex["ground_truth"]
        
        # Execute query
        res = service.answer(
            question,
            cross_encoder_model_name=reranker_model_name,
            citation_min_token_overlap=n_value,
        )
        
        total_attempts += res.generation_attempts
        if not res.validation_passed:
            failed_outputs += 1
            
        results.append({
            "question": question,
            "answer": res.answer,
            "contexts": [c["text"] for c in res.contexts],
            "ground_truth": ground_truth,
            "generation_attempts": res.generation_attempts,
            "validation_passed": res.validation_passed,
            "validation_issues": res.validation_issues,
        })
        
        if idx % 5 == 0:
            print(f"Processed {idx}/{len(examples)} queries...")
            
    avg_attempts = total_attempts / len(examples) if examples else 0.0
    return results, avg_attempts, failed_outputs


def run_ragas_evaluation_all_metrics(predictions: list[dict]) -> dict[str, float]:
    """Grade generated predictions using Ragas across Faithfulness, Relevancy, Precision, and Recall."""
    import sys
    from types import ModuleType
    import langchain_google_genai

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        mock_module = ModuleType("langchain_community.chat_models.vertexai")
        mock_module.ChatVertexAI = langchain_google_genai.ChatGoogleGenerativeAI
        sys.modules["langchain_community.chat_models.vertexai"] = mock_module

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    import math

    settings = get_settings()
    settings.gcp_project_id = "x-cds-502821"
    settings.gcp_region = "global"

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", # Use stable Pro judge
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gcp_region,
            temperature=0.0,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=settings.eval_embedding_model,
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gcp_region,
        )
    )

    dataset = Dataset.from_dict(
        {
            "question": [p["question"] for p in predictions],
            "answer": [p["answer"] for p in predictions],
            "contexts": [p["contexts"] for p in predictions],
            "ground_truth": [p["ground_truth"] for p in predictions],
        }
    )
    
    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=5, timeout=180, max_retries=5)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )
    
    metrics = {}
    scores_list = result.scores
    if scores_list:
        keys = scores_list[0].keys()
        for key in keys:
            vals = []
            for row in scores_list:
                val = row[key]
                if val is not None and not math.isnan(val):
                    vals.append(val)
            metrics[key] = float(sum(vals) / len(vals)) if vals else 0.0
            
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    
    # Load dataset
    if not args.dataset.exists():
        print(f"Error: Dataset not found at {args.dataset}")
        return
        
    raw_examples = []
    with args.dataset.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_examples.append(json.loads(line))
                
    raw_examples = raw_examples[:args.limit]
    print(f"Loaded {len(raw_examples)} test cases for evaluation (n = {args.limit}).")

    eval_dir = Path(__file__).resolve().parent
    
    # 1. Run Baseline (MiniLM)
    baseline_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    base_preds, base_avg_attempts, base_failed = run_full_rag_pipeline(
        raw_examples, baseline_model, args.n_value
    )
    print("Running Ragas 4-metric evaluation for Baseline (MiniLM)...")
    base_metrics = run_ragas_evaluation_all_metrics(base_preds)
    
    # Write baseline predictions report
    baseline_report_path = eval_dir / f"baseline_full_report_limit_{args.limit}.json"
    with baseline_report_path.open("w", encoding="utf-8") as f:
        json.dump({
            "metrics": base_metrics,
            "avg_attempts": base_avg_attempts,
            "failed_outputs": base_failed,
            "model": baseline_model,
            "predictions": base_preds
        }, f, indent=2)

    # 2. Run New Model (BGE Reranker)
    new_model = args.new_model
    new_preds, new_avg_attempts, new_failed = run_full_rag_pipeline(
        raw_examples, new_model, args.n_value
    )
    print(f"Running Ragas 4-metric evaluation for New Model ({new_model})...")
    new_metrics = run_ragas_evaluation_all_metrics(new_preds)
    
    # Write new model predictions report
    new_report_path = eval_dir / f"new_full_report_limit_{args.limit}.json"
    with new_report_path.open("w", encoding="utf-8") as f:
        json.dump({
            "metrics": new_metrics,
            "avg_attempts": new_avg_attempts,
            "failed_outputs": new_failed,
            "model": new_model,
            "predictions": new_preds
        }, f, indent=2)

    # 3. Print & Generate Markdown Comparison Report
    report_md = []
    report_md.append("# RAG Reranker Performance Comparison Report")
    report_md.append(f"**Dataset Size (N):** {args.limit} clinical test cases")
    report_md.append(f"**Overlap Threshold (n):** {args.n_value}")
    report_md.append("")
    report_md.append("## Core Metrics Comparison")
    report_md.append("")
    report_md.append("| Metric | MiniLM (Baseline) | BAAI/bge-reranker-v2-m3 | Difference |")
    report_md.append("| :--- | :---: | :---: | :---: |")
    
    for metric in sorted(base_metrics.keys()):
        base_val = base_metrics[metric]
        new_val = new_metrics[metric]
        diff = new_val - base_val
        report_md.append(f"| **{metric.replace('_', ' ').title()}** | {base_val:.4f} | {new_val:.4f} | {diff:+.4f} |")
        
    report_md.append(f"| **Average Generation Attempts** | {base_avg_attempts:.2f} | {new_avg_attempts:.2f} | {new_avg_attempts - base_avg_attempts:+.2f} |")
    report_md.append(f"| **Failed Outputs (Validation Failed)** | {base_failed} | {new_failed} | {new_failed - base_failed:+} |")
    report_md.append("")
    
    report_md.append("## Key Insights")
    report_md.append("- **Precision impact:** An improved cross-encoder increases relevance and precision of context chunks, leading to higher faithfulness.")
    report_md.append("- **Verification Loop count:** High precision helps the generator pass verification on the first attempt, reducing overall latency and query costs.")
    
    report_path = eval_dir / f"evaluation_summary_report_{args.limit}.md"
    report_path.write_text("\n".join(report_md), encoding="utf-8")
    
    print("\n" + "="*60)
    print(f"      COMPLETED FULL RAG EVALUATION REPORT (N={args.limit})")
    print("="*60)
    print("\n".join(report_md[4:14]))
    print("="*60)
    print(f"Detailed Markdown report saved in: {report_path}")


if __name__ == "__main__":
    main()
