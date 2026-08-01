import json
import math
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.app.config.settings import get_settings

data_dir = Path("d:/X-CDS/data")

class EvalExample:
    def __init__(self, question, answer, ground_truth):
        self.question = question
        self.answer = answer
        self.ground_truth = ground_truth

def load_cached_predictions(filename):
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Cached predictions file not found: {filename}")
        
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            examples.append(
                EvalExample(
                    question=record["question"],
                    answer=record["answer"],
                    ground_truth=record["ground_truth"]
                )
            )
    return examples

def evaluate_system_correctness(name, examples, limit=None):
    if limit:
        examples = examples[:limit]
        
    print(f"\nEvaluating Ragas answer_correctness for {name} on {len(examples)} cases...")
    
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
    
    dataset = Dataset.from_dict(
        {
            "question": [ex.question for ex in examples],
            "answer": [ex.answer for ex in examples],
            "ground_truth": [ex.ground_truth for ex in examples],
        }
    )
    
    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=3, timeout=180, max_retries=10)
    
    result = evaluate(
        dataset,
        metrics=[answer_correctness],
        llm=evaluator_llm,
        run_config=run_config,
    )
    
    # Process scores
    scores_list = result.scores
    details = []
    total_val = 0.0
    valid_count = 0
    
    for idx, ex in enumerate(examples):
        score = None
        if scores_list and idx < len(scores_list):
            val = scores_list[idx].get("answer_correctness")
            if val is not None and not math.isnan(val):
                score = float(val)
                total_val += score
                valid_count += 1
                
        details.append({
            "question": ex.question,
            "answer": ex.answer,
            "ground_truth": ex.ground_truth,
            "answer_correctness": score
        })
        
    mean_score = total_val / valid_count if valid_count > 0 else 0.0
    
    report = {
        "system": name,
        "sample_count": len(examples),
        "mean_answer_correctness": mean_score,
        "details": details
    }
    
    # Save full system report
    system_report_path = data_dir / f"answer_correctness_{name}.json"
    with system_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved full {name} report to {system_report_path} with mean score: {mean_score:.4f}")
    
    return mean_score

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to evaluate.")
    args = parser.parse_args()
    
    naive_examples = load_cached_predictions("naive_materialized_predictions.jsonl")
    hybrid_examples = load_cached_predictions("baseline_materialized_predictions.jsonl")
    xcds_examples = load_cached_predictions("predictions_t10.jsonl")
    
    summary = {}
    
    # Run Naive
    summary["naive"] = evaluate_system_correctness("naive", naive_examples, args.limit)
    
    # Run Hybrid
    summary["hybrid"] = evaluate_system_correctness("hybrid", hybrid_examples, args.limit)
    
    # Run X-CDS
    summary["xcds"] = evaluate_system_correctness("xcds", xcds_examples, args.limit)
    
    # Save overall summary report
    summary_path = data_dir / "answer_correctness_report.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "metrics": summary,
            "model": "gemini-2.5-pro",
            "sample_count": args.limit or len(xcds_examples)
        }, f, indent=2)
        
    print(f"\nSaved overall summary to {summary_path}")

if __name__ == "__main__":
    main()
