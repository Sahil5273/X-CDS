"""Sparse and hybrid retrieval utilities for X-CDS."""

from .bm25 import BM25Index, SparseHit
from .hybrid import HybridRetriever, HybridSearchResult
from .rrf import FusedHit, reciprocal_rank_fusion

__all__ = [
    "BM25Index",
    "FusedHit",
    "HybridRetriever",
    "HybridSearchResult",
    "SparseHit",
    "reciprocal_rank_fusion",
]
