"""Shared helpers for end-to-end smoke testing."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.config.settings import Settings
from backend.app.ingestion.bioc import BiomedicalChunk, parse_bioc_json
from backend.app.llm.generation import GeminiGenerationNode
from backend.app.llm.graph import XRAGGenerationWorkflow
from backend.app.pipeline.service import XRAGService
from backend.app.rerank.cross_encoder import CrossEncoderReranker
from backend.app.rerank.pipeline import RefinedHybridRetriever
from backend.app.search.bm25 import BM25Index
from backend.app.search.hybrid import HybridRetriever
from backend.app.vector.chroma_store import ChromaVectorStore


FIXTURE_PATH = Path("tests/fixtures/bioc_sample.json")
DEFAULT_QUERY = "How does hybrid retrieval support clinical decision support?"


class FakeEmbeddingFunction:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float("hybrid" in lowered),
            float("retrieval" in lowered),
            float("clinical" in lowered),
            float("decision" in lowered),
            float(len(lowered.split())),
        ]


class FakePairScorer:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for _, passage in pairs:
            lowered = passage.lower()
            score = 0.1
            if "hybrid" in lowered:
                score += 2.0
            if "clinical" in lowered or "decision support" in lowered:
                score += 1.5
            scores.append(score)
        return scores


class FakeChatModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def invoke(self, messages: list[Any]) -> Any:
        class Response:
            content = self.answer

        return Response()


@dataclass
class SmokeIndexes:
    dense_store: ChromaVectorStore
    sparse_index: BM25Index
    chunks: list[BiomedicalChunk]

    def close(self) -> None:
        self.dense_store.close()


def load_fixture_chunks() -> list[BiomedicalChunk]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return parse_bioc_json(json.load(handle))


def build_indexes() -> SmokeIndexes:
    """Build in-memory/ephemeral indexes without Hugging Face downloads."""

    chunks = load_fixture_chunks()
    dense_store = ChromaVectorStore(
        collection_name="smoke_xcds",
        embedding_function=FakeEmbeddingFunction(),
        ephemeral=True,
    )
    dense_store.upsert_chunks(chunks)

    sparse_index = BM25Index()
    sparse_index.index_chunks(chunks)
    return SmokeIndexes(dense_store=dense_store, sparse_index=sparse_index, chunks=chunks)


def build_smoke_service(
    indexes: SmokeIndexes,
    *,
    answer: str = (
        "Hybrid retrieval combines dense and sparse evidence for clinical "
        "decision support [1]."
    ),
) -> XRAGService:
    """Construct a fully wired X-RAG service for smoke testing."""

    settings = Settings(
        gcp_project_id=os.getenv("GCP_PROJECT_ID", "test-project"),
        gcp_region=os.getenv("GCP_REGION", "us-central1"),
        langgraph_max_generation_attempts=2,
        rerank_top_k=2,
    )
    hybrid = HybridRetriever(indexes.dense_store, indexes.sparse_index, fused_top_k=3)
    refined = RefinedHybridRetriever(
        hybrid,
        reranker=CrossEncoderReranker(scorer=FakePairScorer()),
        rerank_top_k=settings.rerank_top_k,
    )
    generator = XRAGGenerationWorkflow(
        settings=settings,
        node=GeminiGenerationNode(
            settings=settings,
            llm=FakeChatModel(answer),
        ),
    )
    return XRAGService(
        retriever=refined,
        generator=generator,
        settings=settings,
    )


class SmokeWorkspace:
    """Temporary smoke-test workspace backed by ephemeral indexes."""

    def __init__(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.base_dir = Path(self._tempdir.name)
        self.indexes: SmokeIndexes | None = None
        self.service: XRAGService | None = None

    def bootstrap(self) -> SmokeIndexes:
        self.indexes = build_indexes()
        return self.indexes

    def build_service(self, **kwargs: Any) -> XRAGService:
        if self.indexes is None:
            self.bootstrap()
        assert self.indexes is not None
        self.service = build_smoke_service(self.indexes, **kwargs)
        return self.service

    def cleanup(self) -> None:
        if self.indexes is not None:
            self.indexes.close()
            self.indexes = None
        self.service = None
        self._tempdir.cleanup()

    def __enter__(self) -> SmokeWorkspace:
        self.bootstrap()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()
