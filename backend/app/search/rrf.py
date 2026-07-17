"""Reciprocal Rank Fusion for merging dense and sparse rankings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence


class RankedHit(Protocol):
    """Minimal ranked-document interface accepted by RRF."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FusedHit:
    """A document after Reciprocal Rank Fusion across one or more rankings."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    sources: tuple[str, ...]
    ranks: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = list(self.sources)
        return payload


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedHit]],
    *,
    ranking_names: Sequence[str] | None = None,
    k: int = 60,
    top_k: int | None = None,
) -> list[FusedHit]:
    """Merge ranked lists with classic RRF: sum(1 / (k + rank))."""

    if k <= 0:
        raise ValueError("k must be greater than 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if ranking_names is None:
        names = [f"ranker_{index}" for index in range(len(rankings))]
    else:
        names = list(ranking_names)
        if len(names) != len(rankings):
            raise ValueError("ranking_names must match the number of rankings")

    fused_scores: dict[str, float] = {}
    payloads: dict[str, RankedHit] = {}
    sources: dict[str, set[str]] = {}
    ranks: dict[str, dict[str, int]] = {}

    for ranking_name, ranking in zip(names, rankings, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk_id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
            payloads.setdefault(chunk_id, hit)
            sources.setdefault(chunk_id, set()).add(ranking_name)
            ranks.setdefault(chunk_id, {})[ranking_name] = rank

    ordered_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
    )
    if top_k is not None:
        ordered_ids = ordered_ids[:top_k]

    return [
        FusedHit(
            chunk_id=chunk_id,
            text=payloads[chunk_id].text,
            score=fused_scores[chunk_id],
            metadata=dict(payloads[chunk_id].metadata),
            sources=tuple(sorted(sources[chunk_id])),
            ranks=dict(ranks[chunk_id]),
        )
        for chunk_id in ordered_ids
    ]
