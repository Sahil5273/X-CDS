"""Retrieval pipeline with optional cross-encoder context refinement."""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.rerank.cross_encoder import (
    CrossEncoderReranker,
    RerankedHit,
)
from backend.app.search.hybrid import HybridRetriever, HybridSearchResult


@dataclass(frozen=True, slots=True)
class RefinedRetrievalResult:
    """Hybrid retrieval output after cross-encoder top-k filtering."""

    hybrid: HybridSearchResult
    reranked_hits: list[RerankedHit]


class RefinedHybridRetriever:
    """Compose hybrid RRF retrieval with cross-encoder context refinement."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker | None = None,
        *,
        rerank_top_k: int = 5,
    ) -> None:
        if rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be greater than 0")
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.rerank_top_k = rerank_top_k

    def search(self, query: str) -> RefinedRetrievalResult:
        """Retrieve fused candidates, then keep the top cross-encoder hits."""

        hybrid = self.hybrid_retriever.search(query)
        reranked_hits = self.reranker.rerank(
            query,
            hybrid.fused_hits,
            top_k=self.rerank_top_k,
        )
        return RefinedRetrievalResult(hybrid=hybrid, reranked_hits=reranked_hits)
