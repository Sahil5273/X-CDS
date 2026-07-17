"""Persistent ChromaDB wrapper for local dense retrieval."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import chromadb
from chromadb.api.models.Collection import Collection

from backend.app.ingestion.bioc import BiomedicalChunk
from backend.app.vector.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingConfig,
    EmbeddingFunction,
    HuggingFaceEmbeddingFunction,
)

DEFAULT_COLLECTION_NAME = "xcds_biomedical"
DEFAULT_PERSIST_DIR = Path("./data/chroma")


@dataclass(frozen=True, slots=True)
class DenseHit:
    """A dense similarity match returned by the local Chroma index."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str | int | float | bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


class ChromaVectorStore:
    """Thin persistence layer over a local Chroma collection."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str | None = None,
        embedding_function: EmbeddingFunction | None = None,
        embedding_config: EmbeddingConfig | None = None,
        *,
        ephemeral: bool = False,
    ) -> None:
        self.persist_directory = Path(
            persist_directory
            or os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
        )
        self.collection_name = (
            collection_name
            or os.getenv("CHROMA_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
        )
        self.embedding_function = embedding_function or HuggingFaceEmbeddingFunction(
            embedding_config
            or EmbeddingConfig(
                model_name=os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL),
                device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            )
        )
        self._ephemeral = ephemeral
        if ephemeral:
            self._client = chromadb.EphemeralClient()
        else:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))

        embedding_model_name = DEFAULT_EMBEDDING_MODEL
        config = getattr(self.embedding_function, "config", None)
        if isinstance(config, EmbeddingConfig):
            embedding_model_name = config.model_name
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_model_name,
            },
        )

    @property
    def collection(self) -> Collection:
        return self._collection

    def count(self) -> int:
        return int(self._collection.count())

    def upsert_chunks(self, chunks: Sequence[BiomedicalChunk]) -> int:
        """Embed and upsert BiomedicalChunk records into the local index."""

        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [_chunk_metadata(chunk) for chunk in chunks]
        embeddings = self.embedding_function.embed_documents(documents)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(ids)

    def upsert_texts(
        self,
        *,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        """Embed and upsert raw text passages with optional metadata."""

        if not (len(ids) == len(texts)):
            raise ValueError("ids and texts must have the same length")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas must match the number of ids")
        if not ids:
            return 0

        prepared_metadatas = [
            _sanitize_metadata(dict(metadata)) for metadata in (metadatas or [{}] * len(ids))
        ]
        embeddings = self.embedding_function.embed_documents(list(texts))
        self._collection.upsert(
            ids=list(ids),
            documents=list(texts),
            metadatas=prepared_metadatas,
            embeddings=embeddings,
        )
        return len(ids)

    def similarity_search(self, query: str, *, top_k: int = 5) -> list[DenseHit]:
        """Return the top-k dense matches for a free-text clinical query."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not query.strip():
            raise ValueError("query cannot be empty")

        query_embedding = self.embedding_function.embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        return _hits_from_query_result(result)

    def reset(self) -> None:
        """Drop and recreate the configured collection."""

        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        """Release local client handles so temporary directories can be deleted."""

        self._collection = None  # type: ignore[assignment]
        self._client = None  # type: ignore[assignment]


def _chunk_metadata(chunk: BiomedicalChunk) -> dict[str, str | int]:
    return _sanitize_metadata(
        {
            "pmcid": chunk.pmcid,
            "section": chunk.section,
            "passage_type": chunk.passage_type,
            "offset": chunk.offset,
            "source_url": chunk.source_url,
        }
    )


def _sanitize_metadata(metadata: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            sanitized[str(key)] = value
        else:
            sanitized[str(key)] = str(value)
    return sanitized


def _hits_from_query_result(result: Mapping[str, Any]) -> list[DenseHit]:
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[DenseHit] = []
    for chunk_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances, strict=False
    ):
        if chunk_id is None or document is None:
            continue
        hits.append(
            DenseHit(
                chunk_id=str(chunk_id),
                text=str(document),
                score=_distance_to_similarity(distance),
                metadata=dict(metadata or {}),
            )
        )
    return hits


def _distance_to_similarity(distance: Any) -> float:
    try:
        return 1.0 - float(distance)
    except (TypeError, ValueError):
        return 0.0
