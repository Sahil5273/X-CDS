"""Local Hugging Face dense embedding helpers for ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Runtime configuration for the local sentence-transformer encoder."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    device: str = "cpu"
    normalize_embeddings: bool = True


class EmbeddingFunction(Protocol):
    """Minimal embedding interface expected by the Chroma wrapper."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode corpus passages for indexing."""

    def embed_query(self, text: str) -> list[float]:
        """Encode a single retrieval query."""


class HuggingFaceEmbeddingFunction:
    """Dense encoder backed by a local Hugging Face sentence-transformer."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = _load_sentence_transformer(
            self.config.model_name,
            self.config.device,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)
