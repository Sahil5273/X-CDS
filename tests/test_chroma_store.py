"""Tests for the local Chroma dense vector store wrapper."""

from __future__ import annotations

import unittest

from backend.app.ingestion.bioc import BiomedicalChunk
from backend.app.vector.chroma_store import ChromaVectorStore


class FakeEmbeddingFunction:
    """Deterministic bag-of-characters embedding for offline unit tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("clinical")),
            float(lowered.count("decision")),
            float(lowered.count("support")),
            float(len(lowered.split())),
        ]


class ChromaVectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ChromaVectorStore(
            collection_name="test_xcds",
            embedding_function=FakeEmbeddingFunction(),
            ephemeral=True,
        )

    def tearDown(self) -> None:
        self.store.close()

    def test_upserts_chunks_and_returns_dense_hits(self) -> None:
        chunks = [
            BiomedicalChunk(
                chunk_id="PMC1:passage:0",
                pmcid="PMC1",
                text="Clinical decision support improves triage.",
                section="Abstract",
                passage_type="abstract",
                offset=0,
                source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
            ),
            BiomedicalChunk(
                chunk_id="PMC1:passage:1",
                pmcid="PMC1",
                text="Unrelated laboratory assay calibration notes.",
                section="Methods",
                passage_type="methods",
                offset=40,
                source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
            ),
        ]

        upserted = self.store.upsert_chunks(chunks)
        hits = self.store.similarity_search("clinical decision support", top_k=1)

        self.assertEqual(upserted, 2)
        self.assertEqual(self.store.count(), 2)
        self.assertEqual(hits[0].chunk_id, "PMC1:passage:0")
        self.assertEqual(hits[0].metadata["pmcid"], "PMC1")
        self.assertGreater(hits[0].score, 0.0)

    def test_rejects_empty_query(self) -> None:
        with self.assertRaisesRegex(ValueError, "query cannot be empty"):
            self.store.similarity_search("   ")


if __name__ == "__main__":
    unittest.main()
