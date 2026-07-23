"""Generate a high-quality clinical evaluation dataset from local Zika passages using Gemini 2.5 Pro."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

# Load local environment variables
load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/bioc_chunks.jsonl"),
        help="Path to the BioC JSONL knowledge base passages.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/my_eval_set_large.jsonl"),
        help="Path to write the generated JSONL dataset.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Total number of clinical evaluation cases to generate.",
    )
    return parser


def load_bioc_chunks(path: Path) -> list[dict[str, any]]:
    if not path.exists():
        raise SystemExit(f"Input JSONL does not exist: {path}")

    chunks: list[dict[str, any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def generate_batch(
    llm: ChatGoogleGenerativeAI,
    passages: list[str],
    num_questions: int,
) -> list[dict[str, str]]:
    context_text = "\n\n".join(
        [f"Passage {i+1}:\n{p}" for i, p in enumerate(passages)]
    )

    prompt = f"""You are an expert clinical research evaluator.
Given the following medical literature passages regarding the Zika virus, generate exactly {num_questions} diverse, high-quality clinical queries and their corresponding expert ground-truth answers.

Each generated case must:
1. Represent a realistic clinical scenario (e.g., patient presenting with symptoms, diagnosis decisions, pregnancy guidelines, transplacental transmission, congenital complications, or neurological issues).
2. Have a ground-truth answer that is fully supported by the provided passages.
3. Be highly specific (avoid generic questions; mention diagnostics, cellular targets, or pathobiology details present in the text).

Passages:
{context_text}

Output the result strictly as a valid JSON array of objects, with no markdown code blocks, text wrapper, or commentary. Use this exact schema:
[
  {{"question": "clinical question here", "ground_truth": "expert grounded recommendation here"}},
  ...
]
"""

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        # Clean up any potential markdown wrapper
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        return json.loads(text)
    except Exception as exc:
        print(f"Error generating batch: {exc}")
        return []


def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading chunks from {args.input}...")
    all_chunks = load_bioc_chunks(args.input)
    print(f"Loaded {len(all_chunks)} passages.")

    # Initialize Gemini 2.5 Pro via langchain-google-genai
    project_id = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION", "global")

    print(
        f"Initializing Gemini 2.5 Pro (Project: {project_id}, Region: {region})..."
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        vertexai=True,
        project=project_id,
        location=region,
        temperature=0.3,
    )

    # We want to generate args.count questions (default 50)
    # We will run batches of 5 questions each
    questions_per_batch = 5
    num_batches = (args.count + questions_per_batch - 1) // questions_per_batch

    generated_cases: list[dict[str, str]] = []

    # Group chunks by PMCID to ensure we pull context from diverse papers
    pmcid_groups: dict[str, list[str]] = {}
    for chunk in all_chunks:
        pmcid = chunk["pmcid"]
        if pmcid not in pmcid_groups:
            pmcid_groups[pmcid] = []
        pmcid_groups[pmcid].append(chunk["text"])

    pmcids = list(pmcid_groups.keys())

    print(
        f"Starting generation of {args.count} cases in {num_batches} batches..."
    )
    for batch_idx in range(num_batches):
        # Pick a random PMCID and draw 5 random passages from it for context
        selected_pmcid = pmcids[batch_idx % len(pmcids)]
        passages = random.sample(
            pmcid_groups[selected_pmcid],
            min(6, len(pmcid_groups[selected_pmcid])),
        )

        print(
            f"Generating batch {batch_idx + 1}/{num_batches} using context from {selected_pmcid}..."
        )
        batch_cases = generate_batch(llm, passages, questions_per_batch)
        generated_cases.extend(batch_cases)

        # Stop early if we have enough
        if len(generated_cases) >= args.count:
            generated_cases = generated_cases[: args.count]
            break

    print(f"Writing {len(generated_cases)} clinical cases to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for case in generated_cases:
            destination.write(json.dumps(case, ensure_ascii=False) + "\n")

    print("Success! Large evaluation dataset generated.")


if __name__ == "__main__":
    main()
