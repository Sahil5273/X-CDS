"""Run a parameter sweep on the citation overlap threshold T_min in [0.10, 0.15, 0.50] over the N=100 dataset, then plot the trade-off chart."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt

from backend.app.eval.ragas_eval import (
    load_eval_dataset,
    materialize_predictions,
    run_ragas_evaluation,
    write_report,
)
from backend.app.pipeline.service import build_default_service


def run_sweep_for_threshold(threshold: float, dataset_path: Path) -> dict[str, float]:
    """Execute the full Ragas evaluation pipeline for a specific overlap threshold."""
    t_key = int(threshold * 100)
    cached_preds_path = Path(f"data/predictions_t{t_key}.jsonl")
    report_path = Path(f"data/ragas_report_t{t_key}.json")

    # If report already exists, just load and return it
    if report_path.exists():
        print(f"\n[Sweep] Report for T_min={threshold} already exists at {report_path}. Loading...")
        with report_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    predictions = []
    cached_map = {}

    # 1. Load cached predictions if they exist
    if cached_preds_path.exists():
        print(f"\n[Sweep] Loading cached predictions for T_min={threshold} from {cached_preds_path}...")
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
        print(f"[Sweep] Loaded {len(cached_map)} cached predictions.")

    # 2. Load the full dataset
    examples = load_eval_dataset(dataset_path)

    # 3. Identify missing examples that need to be materialized
    missing_examples = []
    for ex in examples:
        if ex.question in cached_map:
            predictions.append(cached_map[ex.question])
        else:
            missing_examples.append(ex)

    # 4. Materialize only the missing ones
    if missing_examples:
        print(f"\n[Sweep] Materializing {len(missing_examples)} new predictions for T_min={threshold}...")
        service = build_default_service()

        # Monkeypatch the validation function inside the LangGraph generation loop
        import backend.app.guardrail.loop as loop
        from backend.app.guardrail.validator import validate_citation_alignment as original_validate_citation_alignment
        
        # Override the verification node function
        loop.validate_citation_alignment = lambda answer, contexts: original_validate_citation_alignment(
            answer, contexts, min_token_overlap=threshold
        )

        answer_fn = lambda question: service.answer(question).to_dict()
        new_predictions = materialize_predictions(missing_examples, answer_fn=answer_fn)
        predictions.extend(new_predictions)

        # Update cache file with all predictions
        print(f"[Sweep] Caching {len(predictions)} predictions to {cached_preds_path}...")
        cached_preds_path.parent.mkdir(parents=True, exist_ok=True)
        with cached_preds_path.open("w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    print(f"\n[Sweep] Running Ragas evaluation on {len(predictions)} predictions for T_min={threshold}...")
    report = run_ragas_evaluation(predictions)
    write_report(report, report_path)
    print(f"[Sweep] Wrote Ragas report to {report_path}")
    return report.metrics


def plot_threshold_sweep() -> None:
    """Read all reports and generate a publication-quality line chart."""
    thresholds = [0.10, 0.15, 0.25, 0.50]
    faithfulness_scores = []
    relevancy_scores = []

    report_map = {
        0.10: Path("data/ragas_report_t10.json"),
        0.15: Path("data/ragas_report_t15.json"),
        0.25: Path("data/ragas_report.json"),  # Default run
        0.50: Path("data/ragas_report_t50.json"),
    }

    # Verify that all files exist before plotting
    for t in thresholds:
        p = report_map[t]
        if not p.exists():
            print(f"[Plotting] Warning: Missing report file {p}. Cannot generate chart yet.")
            return
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            metrics = data.get("metrics", {})
            faithfulness_scores.append(metrics.get("faithfulness", 0.0) * 100)
            relevancy_scores.append(metrics.get("answer_relevancy", 0.0) * 100)

    print("\n[Plotting] Generating threshold sweep chart...")
    plt.figure(figsize=(8, 5.5))
    
    # Custom premium styling
    plt.plot(
        thresholds,
        faithfulness_scores,
        marker='o',
        linewidth=2.5,
        color="#1f77b4",
        label="Faithfulness (Grounding)",
    )
    plt.plot(
        thresholds,
        relevancy_scores,
        marker='s',
        linewidth=2.5,
        color="#2ca02c",
        label="Answer Relevancy",
    )

    # Label styling
    plt.title("X-CDS: Overlap Threshold ($T_{min}$) Sweep", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Citation Overlap Threshold ($T_{min}$)", fontsize=12, labelpad=10)
    plt.ylabel("Ragas Metric Score (%)", fontsize=12, labelpad=10)
    
    plt.xticks(thresholds)
    plt.ylim(50, 100)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=11)
    
    plt.tight_layout()
    chart_path = Path("docs/threshold_sweep_chart.png")
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_path, dpi=300)
    print(f"[Plotting] Successfully saved trade-off line chart to {chart_path}")


def main() -> None:
    dataset_path = Path("data/my_eval_set_large.jsonl")

    # Run sweeps for the other thresholds
    for threshold in [0.10, 0.15, 0.50]:
        try:
            metrics = run_sweep_for_threshold(threshold, dataset_path)
            print(f"Metrics for T_min={threshold}: {metrics}")
        except Exception as e:
            print(f"Error executing sweep for T_min={threshold}: {e}")

    # Generate plot combining all data
    plot_threshold_sweep()


if __name__ == "__main__":
    main()
