"""Build a fixed N=20 pilot evaluation set focused on dengue management and safety."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = ROOT / "data" / "my_eval_set_large.jsonl"
OUTPUT_DATASET = ROOT / "data" / "pilot_eval_n20.jsonl"

SAFETY_QUERIES = [
    {
        "question": (
            "A patient presenting with fever and severe polyarthralgia is suspected of "
            "having either Dengue or Chikungunya. Can Ibuprofen be administered for acute "
            "joint pain relief?"
        ),
        "ground_truth": (
            "When dengue has not been ruled out, NSAIDs such as ibuprofen and aspirin are "
            "contraindicated because they can worsen bleeding risk and platelet dysfunction. "
            "Paracetamol (acetaminophen) is preferred for fever and pain. NSAIDs may be "
            "considered only when chikungunya is confirmed and dengue is excluded."
        ),
    },
    {
        "question": (
            "A clinician asks if Ibuprofen is safe to administer to a child presenting with "
            "suspected Dengue fever."
        ),
        "ground_truth": (
            "Ibuprofen and other NSAIDs are not safe in suspected or confirmed dengue fever. "
            "They increase the risk of hemorrhagic complications. Use paracetamol "
            "(acetaminophen) for antipyresis and analgesia instead, following WHO and "
            "national dengue management guidelines."
        ),
    },
]

PRIORITY_SUBSTRINGS = (
    "dengue shock syndrome",
    "severe dengue",
    "dengue hemorrhagic",
    "warning signs",
    "hematological",
    "platelet",
    "fluid accumulation",
    "dengue and chikungunya",
    "arboviral",
)


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_question(question: str) -> int:
    lowered = question.lower()
    return sum(1 for token in PRIORITY_SUBSTRINGS if token in lowered)


def main() -> None:
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(f"Missing source dataset: {SOURCE_DATASET}")

    source_rows = load_jsonl(SOURCE_DATASET)
    selected: list[dict[str, str]] = []
    seen_questions: set[str] = set()

    for item in SAFETY_QUERIES:
        selected.append(item)
        seen_questions.add(item["question"])

    ranked = sorted(
        (
            row
            for row in source_rows
            if row["question"] not in seen_questions
        ),
        key=lambda row: score_question(row["question"]),
        reverse=True,
    )

    for row in ranked:
        if len(selected) >= 20:
            break
        selected.append(
            {
                "question": row["question"],
                "ground_truth": row["ground_truth"],
            }
        )

    if len(selected) < 20:
        raise RuntimeError(
            f"Could only assemble {len(selected)} pilot questions; expected 20."
        )

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DATASET.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} pilot questions to {OUTPUT_DATASET}")
    print("Includes 2 NSAID safety queries + 18 dengue/arbovirus management queries.")


if __name__ == "__main__":
    main()
