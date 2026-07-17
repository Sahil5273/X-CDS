"""Asynchronous FastAPI routes for X-CDS."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.api.deps import get_app_settings, get_service
from backend.app.api.schemas import (
    ContextResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from backend.app.config.settings import Settings
from backend.app.pipeline.service import XRAGService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
    )


@router.post("/query", response_model=QueryResponse)
async def query_xrag(
    payload: QueryRequest,
    service: XRAGService = Depends(get_service),
) -> QueryResponse:
    """Execute the asynchronous X-RAG pipeline for a clinical query."""

    try:
        result = await asyncio.to_thread(service.answer, payload.query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface pipeline failures to clients
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"X-RAG pipeline failed: {exc}",
        ) from exc

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        citations=result.citations,
        contexts=[ContextResponse.model_validate(context) for context in result.contexts],
        cited_indices=result.cited_indices,
        validation_passed=result.validation_passed,
        validation_issues=result.validation_issues,
        generation_attempts=result.generation_attempts,
        error=result.error,
    )
