"""Tests for BM25 sparse retrieval and Reciprocal Rank Fusion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.ingestion.bioc import BiomedicalChunk
from backend.app.search.bm25 import BM25Index
from backend.app.search.hybrid import HybridRetriever
from backend.app.search.rrf import reciprocal_rank_fusion
from backend.app.vector.chroma_store import DenseHit


def _chunk(chunk_id: str, text: str) -> BiomedicalChunk:
    return BiomedicalChunk(
        chunk_id=chunk_id,
        pmcid="PMC1",
        text=text,
        section="Abstract",
        passage_type="abstract",
        offset=0,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
    )


class FakeDenseStore:
    def __init__(self, hits: list[DenseHit]) -> None:
        self._hits = hits

    def similarity_search(self, query: str, *, top_k: int = 5) -> list[DenseHit]:
        return self._hits[:top_k]


class BM25IndexTests(unittest.TestCase):
    def test_ranks_keyword_overlap_above_unrelated_passage(self) -> None:
        index = BM25Index()
        index.index_chunks(
            [
                _chunk("a", "Hypertension management with ACE inhibitors."),
                _chunk("b", "Orthopedic casting techniques for ankle fractures."),
            ]
        )

        hits = index.search("hypertension ACE inhibitors", top_k=2)

        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk_id, "a")
        self.assertGreater(hits[0].score, 0.0)

    def test_save_and_load_round_trip(self) -> None:
        index = BM25Index()
        index.index_chunks([_chunk("a", "Clinical decision support systems.")])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bm25_corpus.jsonl"
            index.save(path)
            restored = BM25Index.load(path)

        hits = restored.search("decision support", top_k=1)
        self.assertEqual(len(restored), 1)
        self.assertEqual(hits[0].chunk_id, "a")


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_promotes_documents_appearing_in_both_rankings(self) -> None:
        dense = [
            DenseHit("shared", "Shared evidence.", 0.9, {}),
            DenseHit("dense-only", "Dense only.", 0.8, {}),
        ]
        sparse = [
            DenseHit("sparse-only", "Sparse only.", 4.0, {}),
            DenseHit("shared", "Shared evidence.", 3.5, {}),
        ]

        fused = reciprocal_rank_fusion(
            [dense, sparse],
            ranking_names=("dense", "bm25"),
            k=60,
            top_k=3,
        )

        self.assertEqual(fused[0].chunk_id, "shared")
        self.assertEqual(set(fused[0].sources), {"bm25", "dense"})
        self.assertEqual(fused[0].ranks["dense"], 1)
        self.assertEqual(fused[0].ranks["bm25"], 2)


class HybridRetrieverTests(unittest.TestCase):
    def test_returns_fused_payload_from_dense_and_sparse_channels(self) -> None:
        dense_store = FakeDenseStore(
            [
                DenseHit(
                    "shared",
                    "ACE inhibitors treat hypertension.",
                    0.91,
                    {"pmcid": "PMC1"},
                ),
                DenseHit(
                    "dense-only",
                    "Semantic match without keywords.",
                    0.80,
                    {"pmcid": "PMC1"},
                ),
            ]
        )
        retriever = HybridRetriever(dense_store, fused_top_k=3)
        retriever.index_chunks(
            [
                _chunk("shared", "ACE inhibitors treat hypertension."),
                _chunk("bm25-only", "Hypertension ACE inhibitor dosing tables."),
            ]
        )

        result = retriever.search("hypertension ACE inhibitors")

        self.assertEqual(result.query, "hypertension ACE inhibitors")
        self.assertTrue(result.dense_hits)
        self.assertTrue(result.sparse_hits)
        self.assertTrue(result.fused_hits)
        self.assertIn("shared", {hit.chunk_id for hit in result.fused_hits})


if __name__ == "__main__":
    unittest.main()
