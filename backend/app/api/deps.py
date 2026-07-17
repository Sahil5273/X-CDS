"""FastAPI dependency providers for the X-CDS API."""

from __future__ import annotations

from pathlib import Path

from backend.app.config.settings import (
    Settings,
    clear_settings_cache,
    get_settings,
)
from backend.app.pipeline.service import XRAGService, build_default_service

_service: XRAGService | None = None
_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def get_app_settings() -> Settings:
    return get_settings()


def get_service() -> XRAGService:
    """Return the process-wide X-RAG service, creating it on first use."""

    global _service
    if _service is None:
        clear_settings_cache()
        settings = get_settings()
        print(
            f"[X-CDS] Using Gemini model: {settings.gemini_model} "
            f"(env file: {_ENV_FILE})"
        )
        _service = build_default_service(settings)
    return _service


def set_service(service: XRAGService | None) -> None:
    """Replace or clear the cached service (primarily for tests)."""

    global _service
    _service = service
