"""Run Ragas benchmarking against an X-CDS evaluation JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.eval.ragas_eval import (
    load_eval_dataset,
    materialize_predictions,
    run_ragas_evaluation,
    write_report,
)
from backend.app.pipeline.service import build_default_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/fixtures/ragas_eval_sample.jsonl"),
        help="JSONL evaluation dataset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ragas_report.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--use-pipeline",
        action="store_true",
        help="Generate answers/contexts with the live X-RAG pipeline.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cached_preds_path = Path("data/materialized_predictions.jsonl")
    predictions = []

    # 1. Load cached predictions if they exist
    cached_map = {}
    if cached_preds_path.exists():
        print(f"Loading cached predictions from {cached_preds_path}...")
        with cached_preds_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                from backend.app.eval.ragas_eval import EvalExample
                example = EvalExample(
                    question=record["question"],
                    answer=record["answer"],
                    contexts=record["contexts"],
                    ground_truth=record["ground_truth"],
                )
                cached_map[example.question] = example
        print(f"Loaded {len(cached_map)} cached predictions from file.")

    # 2. Load the full dataset
    examples = load_eval_dataset(args.dataset)

    # 3. Identify missing examples that need to be materialized
    missing_examples = []
    for ex in examples:
        if ex.question in cached_map:
            predictions.append(cached_map[ex.question])
        else:
            missing_examples.append(ex)

    # 4. Materialize only the missing ones
    if missing_examples:
        print(f"Materializing {len(missing_examples)} new predictions...")
        answer_fn = None
        if args.use_pipeline:
            service = build_default_service()
            answer_fn = lambda question: service.answer(question).to_dict()

        new_predictions = materialize_predictions(missing_examples, answer_fn=answer_fn)
        predictions.extend(new_predictions)

        # Update cache file with all predictions
        print(f"Caching {len(predictions)} predictions to {cached_preds_path}...")
        cached_preds_path.parent.mkdir(parents=True, exist_ok=True)
        with cached_preds_path.open("w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    print(f"Running Ragas evaluation on {len(predictions)} predictions...")
    report = run_ragas_evaluation(predictions)
    write_report(report, args.output)
    print(json.dumps(report.metrics, indent=2, ensure_ascii=False))
    print(f"Wrote Ragas report to {args.output}")


if __name__ == "__main__":
    main()
