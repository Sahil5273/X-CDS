"""LangGraph workflow assembly for X-CDS generation."""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from backend.app.config.settings import Settings, get_settings
from backend.app.llm.citations import XRAGState
from backend.app.llm.generation import GeminiGenerationNode, generation_node


def build_generation_graph(
    node: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Any:
    """Build a minimal LangGraph with a single Gemini generation node."""

    graph = StateGraph(XRAGState)
    graph.add_node("generate", node or generation_node)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)
    return graph.compile()


class XRAGGenerationWorkflow:
    """Stateful wrapper around the LangGraph generation pipeline."""

    def __init__(
        self,
        settings: Settings | None = None,
        node: GeminiGenerationNode | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        active_node = node or GeminiGenerationNode(self.settings)
        self.graph = build_generation_graph(active_node)

    def run(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        *,
        generation_attempts: int = 0,
    ) -> dict[str, Any]:
        """Execute generation over reranked contexts."""

        return self.graph.invoke(
            {
                "query": query,
                "contexts": contexts,
                "generation_attempts": generation_attempts,
                "error": None,
            }
        )
