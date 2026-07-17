"""ASGI entrypoint for uvicorn: `uvicorn backend.app.main:app`."""

from backend.app.api.main import app, create_app

__all__ = ["app", "create_app"]
