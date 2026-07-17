"""Sentence-transformer cross-encoder re-ranking for top-k context selection."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Protocol, Sequence


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_TOP_K = 5


@dataclass(frozen=True, slots=True)
class CrossEncoderConfig:
    """Runtime configuration for the local cross-encoder."""

    model_name: str = DEFAULT_CROSS_ENCODER_MODEL
    device: str = "cpu"
    top_k: int = DEFAULT_RERANK_TOP_K


@dataclass(frozen=True, slots=True)
class RerankedHit:
    """A candidate passage after cross-encoder scoring."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    previous_rank: int
    previous_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RankedCandidate(Protocol):
    """Minimal candidate interface accepted by the reranker."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class PairScorer(Protocol):
    """Scores (query, passage) pairs for re-ranking."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return relevance scores aligned with the input pairs."""


class SentenceTransformerPairScorer:
    """Local Hugging Face CrossEncoder backed by sentence-transformers."""

    def __init__(self, config: CrossEncoderConfig | None = None) -> None:
        self.config = config or CrossEncoderConfig()
        self._model = _load_cross_encoder(self.config.model_name, self.config.device)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]


class CrossEncoderReranker:
    """Re-rank fused retrieval hits and keep only the strongest contexts."""

    def __init__(
        self,
        config: CrossEncoderConfig | None = None,
        scorer: PairScorer | None = None,
    ) -> None:
        self.config = config or CrossEncoderConfig(
            model_name=os.getenv(
                "CROSS_ENCODER_MODEL_NAME",
                DEFAULT_CROSS_ENCODER_MODEL,
            ),
            device=os.getenv("CROSS_ENCODER_DEVICE", "cpu"),
            top_k=int(os.getenv("RERANK_TOP_K", str(DEFAULT_RERANK_TOP_K))),
        )
        if self.config.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        self.scorer = scorer or SentenceTransformerPairScorer(self.config)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RankedCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankedHit]:
        """Score candidates with a cross-encoder and return the top-k passages."""

        if not query.strip():
            raise ValueError("query cannot be empty")
        limit = self.config.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than 0")
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.scorer.predict(pairs)
        if len(scores) != len(candidates):
            raise RuntimeError("cross-encoder returned a mismatched score count")

        ranked = sorted(
            (
                (previous_rank, candidate, float(score))
                for previous_rank, (candidate, score) in enumerate(
                    zip(candidates, scores, strict=True),
                    start=1,
                )
            ),
            key=lambda item: item[2],
            reverse=True,
        )
        return [
            RerankedHit(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                score=score,
                metadata=dict(candidate.metadata),
                previous_rank=previous_rank,
                previous_score=float(candidate.score),
            )
            for previous_rank, candidate, score in ranked[:limit]
        ]


def refine_context(
    query: str,
    candidates: Sequence[RankedCandidate],
    *,
    top_k: int = DEFAULT_RERANK_TOP_K,
    reranker: CrossEncoderReranker | None = None,
) -> list[RerankedHit]:
    """Convenience helper that filters retrieval output to the top-k contexts."""

    active_reranker = reranker or CrossEncoderReranker(
        CrossEncoderConfig(top_k=top_k)
    )
    return active_reranker.rerank(query, candidates, top_k=top_k)


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, device: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)
