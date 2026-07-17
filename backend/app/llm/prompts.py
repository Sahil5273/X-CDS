"""Prompt templates for Gemini generation with strict markdown citations."""

from __future__ import annotations

from backend.app.llm.citations import SourceContext

SYSTEM_PROMPT = """You are X-CDS, a clinical decision support assistant.

Rules:
1. Answer only from the numbered source passages provided.
2. Every factual claim must include an explicit markdown citation like [1] or [2].
3. Use concise clinical language in markdown.
4. If evidence is insufficient, say so and cite the closest relevant passage.
5. Do not invent citations or reference passages that were not provided.
"""


def format_context_block(contexts: list[SourceContext]) -> str:
    """Render reranked passages as numbered evidence blocks."""

    blocks: list[str] = []
    for context in contexts:
        header = f"[{context.index}] chunk_id={context.chunk_id}"
        if context.pmcid:
            header += f" | pmcid={context.pmcid}"
        if context.section:
            header += f" | section={context.section}"
        if context.source_url:
            header += f" | source={context.source_url}"
        blocks.append(f"{header}\n{context.text.strip()}")
    return "\n\n".join(blocks)


def build_generation_messages(
    query: str,
    contexts: list[SourceContext],
) -> list[tuple[str, str]]:
    """Return role/content pairs for LangChain chat models."""

    context_block = format_context_block(contexts)
    user_prompt = (
        "Clinical query:\n"
        f"{query.strip()}\n\n"
        "Source passages:\n"
        f"{context_block}\n\n"
        "Write a markdown answer with explicit inline citations such as [1]."
    )
    return [("system", SYSTEM_PROMPT), ("user", user_prompt)]
