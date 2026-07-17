"""Composable X-RAG execution pipeline."""

from .service import QueryResult, XRAGService, build_default_service

__all__ = ["QueryResult", "XRAGService", "build_default_service"]
