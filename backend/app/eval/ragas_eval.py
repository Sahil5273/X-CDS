"""Ragas benchmarking utilities for X-CDS."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True, slots=True)
class EvalExample:
    """One question/answer/context record for Ragas evaluation."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregated Ragas metric scores."""

    sample_count: int
    metrics: dict[str, float]
    details: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "metrics": self.metrics,
            "details": self.details,
        }


AnswerFn = Callable[[str], dict[str, Any]]


def load_eval_dataset(path: Path) -> list[EvalExample]:
    """Load a JSONL evaluation set with question/ground_truth fields."""

    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {path}")

    examples: list[EvalExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
                examples.append(
                    EvalExample(
                        question=str(record["question"]).strip(),
                        answer=str(record.get("answer", "")).strip(),
                        contexts=[str(item) for item in record.get("contexts", [])],
                        ground_truth=str(record["ground_truth"]).strip(),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid evaluation row on line {line_number} of {path}"
                ) from exc
    return examples


def materialize_predictions(
    examples: Sequence[EvalExample],
    answer_fn: AnswerFn | None = None,
) -> list[EvalExample]:
    """Fill answers/contexts by calling the X-RAG pipeline when needed."""

    materialized: list[EvalExample] = []
    for example in examples:
        if example.answer and example.contexts:
            materialized.append(example)
            continue
        if answer_fn is None:
            raise ValueError(
                "answer_fn is required when evaluation rows omit answer/contexts"
            )
        prediction = answer_fn(example.question)
        answer = str(prediction.get("answer", "")).strip()
        contexts = [
            str(context.get("text", "")).strip()
            for context in list(prediction.get("contexts") or [])
            if str(context.get("text", "")).strip()
        ]
        materialized.append(
            EvalExample(
                question=example.question,
                answer=answer or example.answer,
                contexts=contexts or list(example.contexts),
                ground_truth=example.ground_truth,
            )
        )
    return materialized


def run_ragas_evaluation(examples: Sequence[EvalExample]) -> EvalReport:
    """Score answers with Ragas faithfulness / relevancy / context precision."""

    if not examples:
        raise ValueError("evaluation examples cannot be empty")

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
    from ragas.metrics import answer_relevancy, context_precision, faithfulness, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from backend.app.config.settings import get_settings

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
            "question": [example.question for example in examples],
            "answer": [example.answer for example in examples],
            "contexts": [example.contexts for example in examples],
            "ground_truth": [example.ground_truth for example in examples],
        }
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    # Compute aggregate mean for each metric from row-by-row scores
    metrics = {}
    scores_list = result.scores
    if scores_list:
        keys = scores_list[0].keys()
        for key in keys:
            vals = [row[key] for row in scores_list if row[key] is not None]
            if vals:
                metrics[key] = float(sum(vals) / len(vals))
    details = [
        {
            "question": example.question,
            "answer": example.answer,
            "ground_truth": example.ground_truth,
            "context_count": len(example.contexts),
        }
        for example in examples
    ]
    return EvalReport(
        sample_count=len(examples),
        metrics=metrics,
        details=details,
    )


def write_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
