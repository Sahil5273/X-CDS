"""Tests for FastAPI routes and Ragas evaluation helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.api.deps import set_service
from backend.app.api.main import create_app
from backend.app.eval.ragas_eval import (
    EvalExample,
    load_eval_dataset,
    materialize_predictions,
)
from backend.app.pipeline.service import QueryResult, XRAGService
from backend.app.rerank.cross_encoder import RerankedHit
from backend.app.rerank.pipeline import RefinedRetrievalResult
from backend.app.search.hybrid import HybridSearchResult


class FakeRetriever:
    def search(self, query: str) -> RefinedRetrievalResult:
        hit = RerankedHit(
            chunk_id="PMC1:passage:0",
            text="ACE inhibitors reduce blood pressure in hypertension.",
            score=0.9,
            metadata={
                "pmcid": "PMC1",
                "section": "Abstract",
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
            },
            previous_rank=1,
            previous_score=0.4,
        )
        hybrid = HybridSearchResult(
            query=query,
            dense_hits=[],
            sparse_hits=[],
            fused_hits=[],
        )
        return RefinedRetrievalResult(hybrid=hybrid, reranked_hits=[hit])


class FakeGenerator:
    def run(
        self,
        query: str,
        contexts: list[dict],
        *,
        generation_attempts: int = 0,
        max_generation_attempts: int | None = None,
    ) -> dict:
        return {
            "answer": "ACE inhibitors reduce blood pressure in hypertension [1].",
            "citations": [
                {
                    "index": 1,
                    "label": "[1]",
                    "chunk_id": contexts[0]["chunk_id"],
                    "pmcid": contexts[0].get("pmcid", ""),
                    "section": contexts[0].get("section", ""),
                    "source_url": contexts[0].get("source_url", ""),
                    "excerpt": contexts[0]["text"][:240],
                }
            ],
            "cited_indices": [1],
            "validation_passed": True,
            "validation_issues": [],
            "generation_attempts": 1,
            "error": None,
        }


class APIRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        set_service(XRAGService(FakeRetriever(), FakeGenerator()))
        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        set_service(None)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["app_name"], "X-CDS")

    def test_query_endpoint_returns_answer_and_citations(self) -> None:
        response = self.client.post(
            "/api/v1/query",
            json={"query": "hypertension ACE inhibitors"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("[1]", payload["answer"])
        self.assertTrue(payload["validation_passed"])
        self.assertEqual(payload["contexts"][0]["chunk_id"], "PMC1:passage:0")
        self.assertEqual(payload["citations"][0]["label"], "[1]")

    def test_query_endpoint_rejects_empty_query(self) -> None:
        response = self.client.post("/api/v1/query", json={"query": "   "})
        self.assertEqual(response.status_code, 422)


class RagasHelperTests(unittest.TestCase):
    def test_load_eval_dataset(self) -> None:
        path = Path("tests/fixtures/ragas_eval_sample.jsonl")
        examples = load_eval_dataset(path)
        self.assertEqual(len(examples), 2)
        self.assertTrue(examples[0].question)
        self.assertTrue(examples[0].ground_truth)

    def test_materialize_predictions_uses_pipeline_when_needed(self) -> None:
        examples = [
            EvalExample(
                question="What do ACE inhibitors do?",
                answer="",
                contexts=[],
                ground_truth="They reduce blood pressure.",
            )
        ]

        def answer_fn(question: str) -> dict:
            return QueryResult(
                query=question,
                answer="ACE inhibitors reduce blood pressure [1].",
                citations=[],
                contexts=[{"text": "ACE inhibitors reduce blood pressure."}],
                cited_indices=[1],
                validation_passed=True,
                validation_issues=[],
                generation_attempts=1,
                error=None,
            ).to_dict()

        materialized = materialize_predictions(examples, answer_fn=answer_fn)
        self.assertEqual(len(materialized), 1)
        self.assertIn("ACE inhibitors", materialized[0].answer)
        self.assertEqual(len(materialized[0].contexts), 1)


if __name__ == "__main__":
    unittest.main()
