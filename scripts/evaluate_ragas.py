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
    examples = load_eval_dataset(args.dataset)

    answer_fn = None
    if args.use_pipeline:
        service = build_default_service()
        answer_fn = lambda question: service.answer(question).to_dict()

    predictions = materialize_predictions(examples, answer_fn=answer_fn)
    report = run_ragas_evaluation(predictions)
    write_report(report, args.output)
    print(json.dumps(report.metrics, indent=2, ensure_ascii=False))
    print(f"Wrote Ragas report to {args.output}")


if __name__ == "__main__":
    main()
