"""Retrieval-only evaluation script comparing MiniLM and BGE reranker models."""

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

# Disable Hugging Face offline flags temporarily to download the new reranker model weights
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from backend.app.config.settings import get_settings
from backend.app.eval.ragas_eval import EvalExample, run_ragas_evaluation
from backend.app.pipeline.service import build_default_service
from backend.app.rerank.cross_encoder import CrossEncoderConfig, CrossEncoderReranker

# Ensure we use gemini-2.5-pro as the evaluator judge for stable Vertex AI access
os.environ["EVAL_LLM_MODEL"] = "gemini-2.5-pro"

# Override settings to ensure Ragas uses the correct active GCP project and region
settings = get_settings()
settings.gcp_project_id = "x-cds-502821"
settings.gcp_region = "us-central1"


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
    return parser


def run_retrieval_only(
    examples: Sequence[dict[str, str]],
    reranker_model_name: str,
) -> list[EvalExample]:
    """Retrieve top context passages for each question using the specified reranker."""
    print(f"\nInitializing retriever with model: {reranker_model_name}...")
    
    # Load settings and override the cross-encoder model
    settings = get_settings()
    settings.cross_encoder_model_name = reranker_model_name
    
    # Build RAG service (which builds the retriever)
    service = build_default_service(settings)
    
    print(f"Retrieving passages for {len(examples)} cases...")
    predictions = []
    
    for idx, ex in enumerate(examples, start=1):
        question = ex["question"]
        ground_truth = ex["ground_truth"]
        
        # Run hybrid retrieval + re-ranking
        search_result = service.retriever.search(question)
        contexts = [hit.text for hit in search_result.reranked_hits]
        
        predictions.append(
            EvalExample(
                question=question,
                answer="",  # Not needed for retrieval evaluation
                contexts=contexts,
                ground_truth=ground_truth,
            )
        )
        
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(examples)} queries...")
            
    return predictions


def run_custom_ragas_eval(predictions: list[EvalExample]) -> dict[str, float]:
    """Score predictions with Ragas context metrics only (precision & recall) using Flash."""
    import sys
    from types import ModuleType
    import langchain_google_genai

    # Mock langchain_community.chat_models.vertexai for Ragas compatibility
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        mock_module = ModuleType("langchain_community.chat_models.vertexai")
        mock_module.ChatVertexAI = langchain_google_genai.ChatGoogleGenerativeAI
        sys.modules["langchain_community.chat_models.vertexai"] = mock_module

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    import math

    settings = get_settings()

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=settings.eval_llm_model,
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
            "question": [p.question for p in predictions],
            "answer": [p.answer for p in predictions],
            "contexts": [p.contexts for p in predictions],
            "ground_truth": [p.ground_truth for p in predictions],
        }
    )
    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=5, timeout=120, max_retries=5)

    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall],
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
                
    # Limit number of cases if requested
    raw_examples = raw_examples[:args.limit]
    print(f"Loaded {len(raw_examples)} test cases for evaluation.")

    eval_dir = Path(__file__).resolve().parent
    
    # 1. Run Baseline (MiniLM)
    baseline_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    baseline_preds = run_retrieval_only(raw_examples, baseline_model)
    print("Running Ragas evaluation for Baseline (MiniLM)...")
    baseline_metrics = run_custom_ragas_eval(baseline_preds)
    
    # Write baseline predictions report
    baseline_report_path = eval_dir / "baseline_retrieval_report.json"
    with baseline_report_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": baseline_metrics, "model": baseline_model}, f, indent=2)
    print(f"Baseline Metrics: {baseline_metrics}")

    # 2. Run New Model (BGE Reranker)
    new_model = args.new_model
    new_preds = run_retrieval_only(raw_examples, new_model)
    print(f"Running Ragas evaluation for New Model ({new_model})...")
    new_metrics = run_custom_ragas_eval(new_preds)
    
    # Write new model predictions report
    new_report_path = eval_dir / "new_retrieval_report.json"
    with new_report_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": new_metrics, "model": new_model}, f, indent=2)
    print(f"New Model Metrics: {new_metrics}")

    # 3. Print Side-by-Side Comparison
    print("\n" + "="*50)
    print("           RETRIEVAL COMPARISON REPORT")
    print("="*50)
    print(f"{'Metric':<25} | {'MiniLM (Baseline)':<20} | {new_model:<25}")
    print("-"*80)
    for metric in sorted(baseline_metrics.keys()):
        base_val = baseline_metrics[metric]
        new_val = new_metrics[metric]
        diff = new_val - base_val
        print(f"{metric:<25} | {base_val:<20.4f} | {new_val:<25.4f} ({diff:+.4f})")
    print("="*50)
    print(f"Reports saved in: {eval_dir}")


if __name__ == "__main__":
    main()
