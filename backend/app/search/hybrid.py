"""Hybrid dense + sparse retrieval fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from backend.app.ingestion.bioc import BiomedicalChunk
from backend.app.search.bm25 import BM25Index, SparseHit
from backend.app.search.rrf import FusedHit, reciprocal_rank_fusion
from backend.app.vector.chroma_store import DenseHit


class DenseSearcher(Protocol):
    """Minimal dense retrieval interface required by the hybrid pipeline."""

    def similarity_search(self, query: str, *, top_k: int = 5) -> list[DenseHit]:
        """Return dense vector matches for a query."""


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    """Full hybrid search payload for debugging and downstream reranking."""

    query: str
    dense_hits: list[DenseHit]
    sparse_hits: list[SparseHit]
    fused_hits: list[FusedHit]


class HybridRetriever:
    """Run dense Chroma search and BM25 in parallel, then fuse with RRF."""

    def __init__(
        self,
        dense_store: DenseSearcher,
        sparse_index: BM25Index | None = None,
        *,
        rrf_k: int = 60,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        fused_top_k: int = 10,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")
        self.dense_store = dense_store
        self.sparse_index = sparse_index or BM25Index()
        self.rrf_k = rrf_k
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.fused_top_k = fused_top_k

    def index_chunks(self, chunks: Sequence[BiomedicalChunk]) -> int:
        """Refresh the sparse BM25 corpus from BiomedicalChunk passages."""

        return self.sparse_index.index_chunks(chunks)

    def search(self, query: str) -> HybridSearchResult:
        """Execute dense + sparse retrieval and return the RRF-fused ranking."""

        dense_hits = self.dense_store.similarity_search(query, top_k=self.dense_top_k)
        sparse_hits = self.sparse_index.search(query, top_k=self.sparse_top_k)
        fused_hits = reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            ranking_names=("dense", "bm25"),
            k=self.rrf_k,
            top_k=self.fused_top_k,
        )
        return HybridSearchResult(
            query=query,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            fused_hits=fused_hits,
        )
