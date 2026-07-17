"""FastAPI dependency providers for the X-CDS API."""

from __future__ import annotations

from backend.app.config.settings import Settings, get_settings
from backend.app.pipeline.service import XRAGService, build_default_service

_service: XRAGService | None = None


def get_app_settings() -> Settings:
    return get_settings()


def get_service() -> XRAGService:
    """Return the process-wide X-RAG service, creating it on first use."""

    global _service
    if _service is None:
        _service = build_default_service(get_settings())
    return _service


def set_service(service: XRAGService | None) -> None:
    """Replace or clear the cached service (primarily for tests)."""

    global _service
    _service = service
