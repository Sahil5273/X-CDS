"""End-to-end X-RAG orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from backend.app.config.settings import Settings, get_settings
from backend.app.llm.citations import SourceContext
from backend.app.llm.graph import XRAGGenerationWorkflow
from backend.app.rerank.cross_encoder import (
    CrossEncoderConfig,
    CrossEncoderReranker,
    RerankedHit,
)
from backend.app.rerank.pipeline import RefinedHybridRetriever
from backend.app.search.bm25 import BM25Index
from backend.app.search.hybrid import HybridRetriever
from backend.app.vector.chroma_store import ChromaVectorStore
from backend.app.vector.embeddings import EmbeddingConfig


class RetrievalPipeline(Protocol):
    def search(self, query: str) -> Any:
        """Return an object exposing reranked_hits."""


class GenerationPipeline(Protocol):
    def run(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        *,
        generation_attempts: int = 0,
        max_generation_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Generate a guarded answer for the supplied contexts."""


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Normalized API/service payload for one clinical query."""

    query: str
    answer: str
    citations: list[dict[str, Any]]
    contexts: list[dict[str, Any]]
    cited_indices: list[int]
    validation_passed: bool
    validation_issues: list[dict[str, Any]]
    generation_attempts: int
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "contexts": self.contexts,
            "cited_indices": self.cited_indices,
            "validation_passed": self.validation_passed,
            "validation_issues": self.validation_issues,
            "generation_attempts": self.generation_attempts,
            "error": self.error,
        }


class XRAGService:
    """Compose retrieval refinement and LangGraph generation."""

    def __init__(
        self,
        retriever: RetrievalPipeline,
        generator: GenerationPipeline,
        settings: Settings | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.settings = settings or get_settings()

    def answer(self, query: str) -> QueryResult:
        """Run the synchronous X-RAG pipeline for one query."""

        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query cannot be empty")

        retrieval = self.retriever.search(cleaned)
        reranked_hits: list[RerankedHit] = list(getattr(retrieval, "reranked_hits", []))
        contexts = [
            SourceContext.from_reranked_hit(index, hit).model_dump()
            for index, hit in enumerate(reranked_hits, start=1)
        ]
        if not contexts:
            return QueryResult(
                query=cleaned,
                answer="",
                citations=[],
                contexts=[],
                cited_indices=[],
                validation_passed=False,
                validation_issues=[
                    {
                        "code": "no_contexts",
                        "message": "No supporting passages were retrieved.",
                    }
                ],
                generation_attempts=0,
                error="No supporting passages were retrieved.",
            )

        generation = self.generator.run(cleaned, contexts)
        return QueryResult(
            query=cleaned,
            answer=str(generation.get("answer", "")),
            citations=list(generation.get("citations") or []),
            contexts=contexts,
            cited_indices=list(generation.get("cited_indices") or []),
            validation_passed=bool(generation.get("validation_passed")),
            validation_issues=list(generation.get("validation_issues") or []),
            generation_attempts=int(generation.get("generation_attempts", 0)),
            error=generation.get("error"),
        )


def build_default_service(settings: Settings | None = None) -> XRAGService:
    """Construct the production service from local indexes and Gemini."""

    active = settings or get_settings()
    dense_store = ChromaVectorStore(
        persist_directory=active.chroma_persist_dir,
        collection_name=active.chroma_collection_name,
        embedding_config=EmbeddingConfig(
            model_name=active.embedding_model_name,
            device=active.embedding_device,
        ),
    )
    sparse_index = _load_bm25_index(Path(active.bm25_corpus_path))
    hybrid = HybridRetriever(dense_store, sparse_index)
    refined = RefinedHybridRetriever(
        hybrid,
        reranker=CrossEncoderReranker(
            CrossEncoderConfig(
                model_name=active.cross_encoder_model_name,
                device=active.cross_encoder_device,
                top_k=active.rerank_top_k,
            )
        ),
        rerank_top_k=active.rerank_top_k,
    )
    generator = XRAGGenerationWorkflow(settings=active)
    return XRAGService(retriever=refined, generator=generator, settings=active)


def _load_bm25_index(path: Path) -> BM25Index:
    if path.exists():
        return BM25Index.load(path)
    return BM25Index()
