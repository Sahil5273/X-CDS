"""BM25 sparse keyword index for biomedical passages."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rank_bm25 import BM25Okapi

from backend.app.ingestion.bioc import BiomedicalChunk
from backend.app.search.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class SparseHit:
    """A BM25 keyword match."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str | int | float | bool]
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _IndexedDocument:
    chunk_id: str
    text: str
    metadata: dict[str, str | int | float | bool]
    tokens: tuple[str, ...]


class PositiveIdfBM25(BM25Okapi):
    """BM25Okapi with Lucene-style IDF to avoid zero scores on small corpora."""

    def _calc_idf(self, nd: dict[str, int]) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(
                1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5)
            )
        if self.idf:
            self.average_idf = sum(self.idf.values()) / len(self.idf)
        else:
            self.average_idf = 0.0


class BM25Index:
    """In-memory BM25 index with optional JSONL corpus persistence."""

    def __init__(self) -> None:
        self._documents: list[_IndexedDocument] = []
        self._bm25: PositiveIdfBM25 | None = None

    def __len__(self) -> int:
        return len(self._documents)

    def index_chunks(self, chunks: Sequence[BiomedicalChunk]) -> int:
        """Replace the index with BiomedicalChunk passages."""

        documents = [
            _IndexedDocument(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata={
                    "pmcid": chunk.pmcid,
                    "section": chunk.section,
                    "passage_type": chunk.passage_type,
                    "offset": chunk.offset,
                    "source_url": chunk.source_url,
                },
                tokens=tuple(tokenize(chunk.text)),
            )
            for chunk in chunks
            if chunk.text.strip()
        ]
        return self._rebuild(documents)

    def index_texts(
        self,
        *,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        """Replace the index with raw text passages."""

        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas must match the number of ids")

        prepared_metadatas = list(metadatas or [{}] * len(ids))
        documents = [
            _IndexedDocument(
                chunk_id=str(chunk_id),
                text=text,
                metadata={
                    str(key): value
                    for key, value in dict(metadata).items()
                    if isinstance(value, (bool, int, float, str))
                },
                tokens=tuple(tokenize(text)),
            )
            for chunk_id, text, metadata in zip(ids, texts, prepared_metadatas, strict=True)
            if str(text).strip()
        ]
        return self._rebuild(documents)

    def search(self, query: str, *, top_k: int = 10) -> list[SparseHit]:
        """Return the top-k BM25 matches for a keyword query."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not query.strip():
            raise ValueError("query cannot be empty")
        if not self._documents or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        hits: list[SparseHit] = []
        for rank, document_index in enumerate(ranked_indices[:top_k], start=1):
            score = float(scores[document_index])
            if score <= 0.0:
                continue
            document = self._documents[document_index]
            hits.append(
                SparseHit(
                    chunk_id=document.chunk_id,
                    text=document.text,
                    score=score,
                    metadata=dict(document.metadata),
                    rank=rank,
                )
            )
        return hits

    def save(self, path: str | Path) -> None:
        """Persist the tokenized corpus as JSONL for later reload."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for document in self._documents:
                handle.write(
                    json.dumps(
                        {
                            "chunk_id": document.chunk_id,
                            "text": document.text,
                            "metadata": document.metadata,
                            "tokens": list(document.tokens),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        """Rebuild a BM25 index from a previously saved JSONL corpus."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"BM25 corpus does not exist: {source}")

        documents: list[_IndexedDocument] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    record = json.loads(payload)
                    documents.append(
                        _IndexedDocument(
                            chunk_id=str(record["chunk_id"]),
                            text=str(record["text"]),
                            metadata=dict(record.get("metadata") or {}),
                            tokens=tuple(str(token) for token in record.get("tokens", [])),
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid BM25 corpus row on line {line_number} of {source}"
                    ) from exc

        index = cls()
        index._rebuild(documents)
        return index

    def _rebuild(self, documents: list[_IndexedDocument]) -> int:
        self._documents = documents
        if documents:
            self._bm25 = PositiveIdfBM25(
                [list(document.tokens) for document in documents]
            )
        else:
            self._bm25 = None
        return len(documents)
