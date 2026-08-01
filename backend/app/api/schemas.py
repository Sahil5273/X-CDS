"""Pydantic request and response models for the X-CDS API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Clinical symptom or decision-support query."""

    query: str = Field(min_length=1, description="Free-text clinical query.")
    cross_encoder_model_name: str | None = Field(
        default=None,
        description="Optional custom cross-encoder model name.",
    )
    citation_min_token_overlap: float | None = Field(
        default=None,
        description="Optional custom citation min token overlap threshold (n value).",
    )

    @field_validator("query")
    @classmethod
    def strip_and_require_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query cannot be empty")
        return cleaned


class CitationResponse(BaseModel):
    index: int
    label: str
    chunk_id: str
    pmcid: str = ""
    section: str = ""
    source_url: str = ""
    excerpt: str = ""


class ContextResponse(BaseModel):
    index: int
    chunk_id: str
    text: str
    pmcid: str = ""
    section: str = ""
    source_url: str = ""


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationResponse]
    contexts: list[ContextResponse]
    cited_indices: list[int]
    validation_passed: bool
    validation_issues: list[dict[str, Any]]
    generation_attempts: int
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
