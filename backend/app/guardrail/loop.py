"""LangGraph validation node and self-correction routing."""

from __future__ import annotations

from typing import Any

from backend.app.config.settings import Settings, get_settings
from backend.app.guardrail.validator import validate_citation_alignment
from backend.app.llm.citations import SourceContext


class CitationGuardrailNode:
    """Deterministic validation node that drives correction feedback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        answer = str(state.get("answer", "")).strip()
        contexts = [
            SourceContext.model_validate(context)
            for context in list(state.get("contexts") or [])
        ]
        result = validate_citation_alignment(answer, contexts)

        if result.is_valid:
            return {
                "validation_passed": True,
                "validation_issues": [],
                "correction_feedback": None,
                "error": None,
            }

        feedback = result.feedback_message()
        return {
            "validation_passed": False,
            "validation_issues": [issue.to_dict() for issue in result.issues],
            "correction_feedback": feedback,
            "error": feedback,
        }


def should_retry_generation(state: dict[str, Any]) -> str:
    """Route validated answers to END, otherwise retry or fail closed."""

    if state.get("validation_passed"):
        return "end"

    attempts = int(state.get("generation_attempts", 0))
    max_attempts = int(
        state.get("max_generation_attempts")
        or get_settings().langgraph_max_generation_attempts
    )
    if attempts < max_attempts:
        return "retry"
    return "end"


def guardrail_node(state: dict[str, Any]) -> dict[str, Any]:
    """Default LangGraph validation entrypoint."""

    return CitationGuardrailNode()(state)
