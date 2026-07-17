"""Gemini generation node for the X-CDS LangGraph workflow."""

from __future__ import annotations

from typing import Any, Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.app.config.settings import Settings, get_settings
from backend.app.llm.citations import (
    SourceContext,
    build_generation_output,
)
from backend.app.llm.prompts import build_generation_messages


class ChatModel(Protocol):
    """Minimal chat-model interface used by the generation node."""

    def invoke(self, messages: list[Any]) -> Any:
        """Generate a response for the supplied chat messages."""


class GeminiGenerationNode:
    """LangGraph-compatible generation node backed by Gemini."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: ChatModel | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or _build_gemini_chat_model(self.settings)

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        query = str(state.get("query", "")).strip()
        if not query:
            raise ValueError("query is required for generation")

        raw_contexts = list(state.get("contexts") or [])
        contexts = [
            SourceContext.model_validate(context) for context in raw_contexts
        ]
        if not contexts:
            raise ValueError("contexts are required for generation")

        correction_feedback = state.get("correction_feedback")
        messages = _to_langchain_messages(
            build_generation_messages(
                query,
                contexts,
                correction_feedback=str(correction_feedback or "") or None,
            )
        )
        response = self.llm.invoke(messages)
        answer = _extract_text(response)
        output = build_generation_output(query, answer, contexts)
        attempts = int(state.get("generation_attempts", 0)) + 1

        return {
            "answer": output.answer_markdown,
            "citations": [citation.model_dump() for citation in output.citations],
            "cited_indices": output.cited_indices,
            "generation_attempts": attempts,
            "validation_passed": False,
            "error": None,
        }


def generation_node(state: dict[str, Any]) -> dict[str, Any]:
    """Default LangGraph node entrypoint using environment-backed settings."""

    return GeminiGenerationNode()(state)


def _build_gemini_chat_model(settings: Settings) -> ChatModel:
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required for Gemini generation")

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.gemini_temperature,
    )


def _to_langchain_messages(messages: list[tuple[str, str]]) -> list[Any]:
    converted: list[Any] = []
    for role, content in messages:
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "user":
            converted.append(HumanMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def _extract_text(response: Any) -> str:
    if isinstance(response, AIMessage):
        content = response.content
    else:
        content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(part.strip() for part in parts if part.strip())
    return str(content).strip()
