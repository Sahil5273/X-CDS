"""LangGraph workflow assembly for X-CDS generation with guardrails."""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from backend.app.config.settings import Settings, get_settings
from backend.app.guardrail.loop import (
    CitationGuardrailNode,
    should_retry_generation,
)
from backend.app.llm.citations import XRAGState
from backend.app.llm.generation import GeminiGenerationNode, generation_node


def build_generation_graph(
    node: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Any:
    """Build LangGraph with generate -> validate -> optional self-correction."""

    graph = StateGraph(XRAGState)
    graph.add_node("generate", node or generation_node)
    graph.add_node("validate", validator or CitationGuardrailNode())
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        should_retry_generation,
        {
            "retry": "generate",
            "end": END,
        },
    )
    return graph.compile()


class XRAGGenerationWorkflow:
    """Stateful wrapper around the LangGraph generation pipeline."""

    def __init__(
        self,
        settings: Settings | None = None,
        node: GeminiGenerationNode | None = None,
        validator: CitationGuardrailNode | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        active_node = node or GeminiGenerationNode(self.settings)
        active_validator = validator or CitationGuardrailNode(self.settings)
        self.graph = build_generation_graph(active_node, active_validator)

    def run(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        *,
        generation_attempts: int = 0,
        max_generation_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Execute generation with deterministic citation verification."""

        return self.graph.invoke(
            {
                "query": query,
                "contexts": contexts,
                "generation_attempts": generation_attempts,
                "max_generation_attempts": (
                    max_generation_attempts
                    if max_generation_attempts is not None
                    else self.settings.langgraph_max_generation_attempts
                ),
                "validation_passed": False,
                "validation_issues": [],
                "correction_feedback": None,
                "error": None,
            }
        )
