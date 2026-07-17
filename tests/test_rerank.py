"""Tests for cross-encoder context refinement."""

from __future__ import annotations

import unittest

from backend.app.rerank.cross_encoder import (
    CrossEncoderConfig,
    CrossEncoderReranker,
    refine_context,
)
from backend.app.rerank.pipeline import RefinedHybridRetriever
from backend.app.search.hybrid import HybridRetriever
from backend.app.search.rrf import FusedHit
from backend.app.vector.chroma_store import DenseHit


class FakePairScorer:
    """Deterministic scorer that prefers passages containing 'hypertension'."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for _, passage in pairs:
            lowered = passage.lower()
            score = 0.1
            if "hypertension" in lowered:
                score += 2.0
            if "ace" in lowered:
                score += 1.0
            if "fracture" in lowered:
                score -= 1.0
            scores.append(score)
        return scores


class FakeDenseStore:
    def similarity_search(self, query: str, *, top_k: int = 5) -> list[DenseHit]:
        return [
            DenseHit("dense-1", "Semantic hypertension overview.", 0.9, {}),
            DenseHit("dense-2", "Unrelated orthopedic fracture notes.", 0.8, {}),
        ][:top_k]


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_filters_to_configured_top_k(self) -> None:
        candidates = [
            FusedHit("a", "Hypertension treated with ACE inhibitors.", 0.03, {}, ("dense",), {"dense": 1}),
            FusedHit("b", "Ankle fracture casting protocol.", 0.02, {}, ("bm25",), {"bm25": 1}),
            FusedHit("c", "General ACE pharmacology review.", 0.01, {}, ("dense",), {"dense": 2}),
            FusedHit("d", "Background nutrition guidance.", 0.005, {}, ("bm25",), {"bm25": 2}),
            FusedHit("e", "Hypertension lifestyle counseling.", 0.004, {}, ("dense", "bm25"), {"dense": 3, "bm25": 3}),
            FusedHit("f", "Dental prophylaxis checklist.", 0.003, {}, ("bm25",), {"bm25": 4}),
        ]
        reranker = CrossEncoderReranker(
            CrossEncoderConfig(top_k=5),
            scorer=FakePairScorer(),
        )

        hits = reranker.rerank("hypertension ACE therapy", candidates)

        self.assertEqual(len(hits), 5)
        self.assertEqual(hits[0].chunk_id, "a")
        self.assertNotIn("b", {hit.chunk_id for hit in hits})
        self.assertGreater(hits[0].score, hits[-1].score)
        self.assertEqual(hits[0].previous_rank, 1)

    def test_refine_context_helper_uses_top_k(self) -> None:
        candidates = [
            FusedHit("a", "Hypertension treated with ACE inhibitors.", 0.03, {}, ("dense",), {"dense": 1}),
            FusedHit("b", "Ankle fracture casting protocol.", 0.02, {}, ("bm25",), {"bm25": 1}),
        ]
        hits = refine_context(
            "hypertension",
            candidates,
            top_k=1,
            reranker=CrossEncoderReranker(scorer=FakePairScorer()),
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk_id, "a")


class RefinedHybridRetrieverTests(unittest.TestCase):
    def test_pipeline_returns_reranked_top_contexts(self) -> None:
        from backend.app.ingestion.bioc import BiomedicalChunk

        hybrid = HybridRetriever(FakeDenseStore(), fused_top_k=4)
        hybrid.index_chunks(
            [
                BiomedicalChunk(
                    chunk_id="shared",
                    pmcid="PMC1",
                    text="Hypertension treated with ACE inhibitors.",
                    section="Abstract",
                    passage_type="abstract",
                    offset=0,
                    source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
                ),
                BiomedicalChunk(
                    chunk_id="noise",
                    pmcid="PMC1",
                    text="Orthopedic fracture rehabilitation notes.",
                    section="Methods",
                    passage_type="methods",
                    offset=10,
                    source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
                ),
            ]
        )
        pipeline = RefinedHybridRetriever(
            hybrid,
            reranker=CrossEncoderReranker(scorer=FakePairScorer()),
            rerank_top_k=1,
        )

        result = pipeline.search("hypertension ACE inhibitors")

        self.assertEqual(len(result.reranked_hits), 1)
        self.assertTrue(result.hybrid.fused_hits)
        self.assertIn("hypertension", result.reranked_hits[0].text.lower())


if __name__ == "__main__":
    unittest.main()
