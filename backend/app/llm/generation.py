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


class RobustChatVertexAI:
    """Wrapper that handles retries and falls back to lighter models on 503/429 errors using Vertex AI."""

    def __init__(self, primary_model: str, settings: Settings) -> None:
        self.primary_model = primary_model
        self.settings = settings
        self.fallback_model = "gemini-3.1-flash-lite"

    def invoke(self, messages: list[Any]) -> Any:
        import logging
        import time
        from langchain_google_vertexai import ChatVertexAI

        logger = logging.getLogger("backend.app.llm.generation")
        models_to_try = [self.primary_model]
        if self.primary_model != self.fallback_model:
            models_to_try.append(self.fallback_model)

        last_error = None
        for model in models_to_try:
            for attempt in range(4):
                try:
                    llm = ChatVertexAI(
                        model=model,
                        project=self.settings.gcp_project_id,
                        location=self.settings.gcp_region,
                        temperature=self.settings.gemini_temperature,
                        max_retries=1,
                    )
                    if hasattr(llm, "model"):
                        llm.model = model
                    return llm.invoke(messages)
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    is_transient = (
                        "503" in err_str
                        or "429" in err_str
                        or "temporarily unavailable" in err_str.lower()
                        or "high demand" in err_str.lower()
                    )
                    if not is_transient:
                        raise e

                    logger.warning(
                        f"Attempt {attempt + 1} failed for model '{model}' with transient error: {e}. "
                        f"Retrying..."
                    )
                    if attempt < 3:
                        time.sleep(2 ** attempt)

            logger.error(
                f"Model '{model}' failed all retry attempts with transient error: {last_error}. "
                f"Trying fallback model..."
            )

        if last_error:
            raise last_error
        raise RuntimeError("Generation failed with unknown error")


def generation_node(state: dict[str, Any]) -> dict[str, Any]:
    """Default LangGraph node entrypoint using environment-backed settings."""

    return GeminiGenerationNode()(state)


def _build_gemini_chat_model(settings: Settings) -> ChatModel:
    model_name = settings.gemini_model.strip() or "gemini-3.5-flash"
    if any(deprecated in model_name for deprecated in ["1.5-flash", "2.0-flash", "2.5-flash"]):
        model_name = "gemini-3.5-flash"

    return RobustChatVertexAI(model_name, settings)


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
