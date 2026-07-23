"""Run Baseline RAG benchmarking (no self-correction loop) against the Zika clinical dataset."""

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
        default=Path("data/my_eval_set_large.jsonl"),
        help="JSONL evaluation dataset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/baseline_ragas_report.json"),
        help="Destination JSON report path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cached_preds_path = Path("data/baseline_materialized_predictions.jsonl")
    predictions = []

    if cached_preds_path.exists():
        print(f"Loading cached baseline predictions from {cached_preds_path}...")
        with cached_preds_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                from backend.app.eval.ragas_eval import EvalExample
                predictions.append(
                    EvalExample(
                        question=record["question"],
                        answer=record["answer"],
                        contexts=record["contexts"],
                        ground_truth=record["ground_truth"],
                    )
                )
        print(f"Loaded {len(predictions)} baseline predictions.")
    else:
        examples = load_eval_dataset(args.dataset)
        service = build_default_service()

        # Monkeypatch generator to force max_generation_attempts=1 (baseline)
        original_run = service.generator.run
        service.generator.run = lambda q, c: original_run(q, c, max_generation_attempts=1)

        answer_fn = lambda question: service.answer(question).to_dict()

        print("Materializing baseline answers (no self-correction loop)...")
        predictions = materialize_predictions(examples, answer_fn=answer_fn)

        print(f"Caching baseline predictions to {cached_preds_path}...")
        cached_preds_path.parent.mkdir(parents=True, exist_ok=True)
        with cached_preds_path.open("w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    print("Running Ragas evaluation on baseline predictions...")
    report = run_ragas_evaluation(predictions)
    write_report(report, args.output)

    print("\n--- Baseline RAG ($N=45$) Metrics ---")
    print(json.dumps(report.metrics, indent=2, ensure_ascii=False))
    print(f"Wrote baseline Ragas report to {args.output}")


if __name__ == "__main__":
    main()
