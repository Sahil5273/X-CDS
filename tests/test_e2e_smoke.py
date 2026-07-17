"""End-to-end smoke tests for the X-CDS live pipeline wiring."""

from __future__ import annotations

import os
import unittest

from fastapi.testclient import TestClient

from backend.app.api.deps import set_service
from backend.app.api.main import create_app
from backend.app.pipeline.service import build_default_service
from backend.app.testing.e2e_harness import DEFAULT_QUERY, SmokeWorkspace


class EndToEndSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = SmokeWorkspace()
        self.workspace.bootstrap()
        self.service = self.workspace.build_service()

    def tearDown(self) -> None:
        set_service(None)
        self.workspace.cleanup()
    def test_pipeline_returns_verified_answer_with_contexts(self) -> None:
        result = self.service.answer(DEFAULT_QUERY)

        self.assertTrue(result.contexts)
        self.assertIn("[1]", result.answer)
        self.assertTrue(result.validation_passed)
        self.assertEqual(result.cited_indices[0], 1)
        self.assertEqual(result.contexts[0]["index"], 1)
        self.assertTrue(result.citations)
        self.assertIsNone(result.error)

    def test_http_query_endpoint_runs_smoke_pipeline(self) -> None:
        set_service(self.service)
        client = TestClient(create_app())

        response = client.post("/api/v1/query", json={"query": DEFAULT_QUERY})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("[1]", payload["answer"])
        self.assertTrue(payload["validation_passed"])
        self.assertTrue(payload["contexts"])
        self.assertEqual(payload["contexts"][0]["index"], 1)

    def test_citation_indices_map_to_returned_contexts(self) -> None:
        result = self.service.answer(DEFAULT_QUERY)
        context_indices = {context["index"] for context in result.contexts}

        for index in result.cited_indices:
            self.assertIn(index, context_indices)


@unittest.skipUnless(
    os.getenv("RUN_LIVE_SMOKE") == "1",
    "Set RUN_LIVE_SMOKE=1 for live Gemini smoke test",
)
class LiveEndToEndSmokeTests(unittest.TestCase):
    def test_live_default_service_answer(self) -> None:
        service = build_default_service()
        result = service.answer(DEFAULT_QUERY)

        self.assertTrue(result.contexts)
        self.assertTrue(result.answer.strip())
        self.assertTrue(result.cited_indices)


if __name__ == "__main__":
    unittest.main()
