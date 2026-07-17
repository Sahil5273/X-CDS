"""LangGraph state and citation schemas for X-CDS generation."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from pydantic import BaseModel, Field, field_validator


CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class SourceContext(BaseModel):
    """One reranked passage exposed to the generation node."""

    index: int = Field(ge=1)
    chunk_id: str
    text: str
    pmcid: str = ""
    section: str = ""
    source_url: str = ""

    @classmethod
    def from_reranked_hit(cls, index: int, hit: Any) -> SourceContext:
        metadata = dict(getattr(hit, "metadata", {}) or {})
        return cls(
            index=index,
            chunk_id=str(getattr(hit, "chunk_id", "")),
            text=str(getattr(hit, "text", "")),
            pmcid=str(metadata.get("pmcid", "")),
            section=str(metadata.get("section", "")),
            source_url=str(metadata.get("source_url", "")),
        )


class MarkdownCitation(BaseModel):
    """Resolved markdown citation mapped to a source passage."""

    index: int = Field(ge=1)
    label: str
    chunk_id: str
    pmcid: str = ""
    section: str = ""
    source_url: str = ""
    excerpt: str = ""

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if not CITATION_PATTERN.fullmatch(value):
            raise ValueError("citation label must use markdown form like [1]")
        return value


class GenerationOutput(BaseModel):
    """Structured generation payload with explicit markdown citations."""

    query: str
    answer_markdown: str
    citations: list[MarkdownCitation]
    cited_indices: list[int] = Field(default_factory=list)

    @field_validator("answer_markdown")
    @classmethod
    def require_markdown_citations(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer_markdown cannot be empty")
        if not CITATION_PATTERN.search(value):
            raise ValueError("answer_markdown must include explicit markdown citations")
        return value


class XRAGState(TypedDict, total=False):
    """LangGraph state carried through the generation workflow."""

    query: str
    contexts: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    cited_indices: list[int]
    generation_attempts: int
    error: str | None


def extract_citation_indices(answer_markdown: str) -> list[int]:
    """Return unique citation indices referenced in markdown like [1]."""

    seen: set[int] = set()
    ordered: list[int] = []
    for match in CITATION_PATTERN.finditer(answer_markdown):
        index = int(match.group(1))
        if index not in seen:
            seen.add(index)
            ordered.append(index)
    return ordered


def resolve_citations(
    answer_markdown: str,
    contexts: list[SourceContext],
) -> list[MarkdownCitation]:
    """Map markdown citation labels in the answer to source passages."""

    by_index = {context.index: context for context in contexts}
    citations: list[MarkdownCitation] = []
    for index in extract_citation_indices(answer_markdown):
        context = by_index.get(index)
        if context is None:
            continue
        citations.append(
            MarkdownCitation(
                index=index,
                label=f"[{index}]",
                chunk_id=context.chunk_id,
                pmcid=context.pmcid,
                section=context.section,
                source_url=context.source_url,
                excerpt=context.text[:240],
            )
        )
    return citations


def build_generation_output(
    query: str,
    answer_markdown: str,
    contexts: list[SourceContext],
) -> GenerationOutput:
    """Validate and normalize a generated answer with citation metadata."""

    citations = resolve_citations(answer_markdown, contexts)
    return GenerationOutput(
        query=query,
        answer_markdown=answer_markdown,
        citations=citations,
        cited_indices=extract_citation_indices(answer_markdown),
    )
