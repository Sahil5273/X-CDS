"""Tests for deterministic citation guardrails and correction loops."""

from __future__ import annotations

import unittest
from typing import Any

from backend.app.config.settings import Settings
from backend.app.guardrail.loop import should_retry_generation
from backend.app.guardrail.validator import validate_citation_alignment
from backend.app.llm.citations import SourceContext
from backend.app.llm.generation import GeminiGenerationNode
from backend.app.llm.graph import XRAGGenerationWorkflow


def _context(
    index: int = 1,
    text: str = "ACE inhibitors reduce blood pressure in hypertension.",
) -> dict[str, Any]:
    return SourceContext(
        index=index,
        chunk_id=f"PMC1:passage:{index - 1}",
        text=text,
        pmcid="PMC1",
        section="Abstract",
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
    ).model_dump()


class SequentialFakeChatModel:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls = 0
        self.last_messages: list[Any] | None = None

    def invoke(self, messages: list[Any]) -> Any:
        self.calls += 1
        self.last_messages = messages
        answer = self.answers.pop(0) if self.answers else ""

        class Response:
            content = answer

        return Response()


class CitationValidationTests(unittest.TestCase):
    def test_accepts_aligned_cited_claim(self) -> None:
        contexts = [
            SourceContext.model_validate(
                _context(text="ACE inhibitors reduce blood pressure in hypertension.")
            )
        ]
        result = validate_citation_alignment(
            "ACE inhibitors reduce blood pressure in hypertension [1].",
            contexts,
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.cited_indices, [1])

    def test_rejects_unknown_citation_index(self) -> None:
        contexts = [SourceContext.model_validate(_context())]
        result = validate_citation_alignment(
            "This claim cites a missing source [9].",
            contexts,
        )
        self.assertFalse(result.is_valid)
        self.assertIn("unknown_citation", {issue.code for issue in result.issues})

    def test_rejects_misaligned_claim_content(self) -> None:
        contexts = [
            SourceContext.model_validate(
                _context(text="ACE inhibitors reduce blood pressure in hypertension.")
            )
        ]
        result = validate_citation_alignment(
            "Ankle fracture casting is first-line therapy [1].",
            contexts,
        )
        self.assertFalse(result.is_valid)
        self.assertIn("context_misalignment", {issue.code for issue in result.issues})

    def test_rejects_answer_without_citations(self) -> None:
        contexts = [SourceContext.model_validate(_context())]
        result = validate_citation_alignment(
            "ACE inhibitors are useful for hypertension.",
            contexts,
        )
        self.assertFalse(result.is_valid)
        self.assertIn("missing_citations", {issue.code for issue in result.issues})


class SelfCorrectionLoopTests(unittest.TestCase):
    def test_should_retry_when_validation_fails_and_attempts_remain(self) -> None:
        route = should_retry_generation(
            {
                "validation_passed": False,
                "generation_attempts": 1,
                "max_generation_attempts": 3,
            }
        )
        self.assertEqual(route, "retry")

    def test_should_end_when_validation_passes(self) -> None:
        route = should_retry_generation(
            {
                "validation_passed": True,
                "generation_attempts": 1,
                "max_generation_attempts": 3,
            }
        )
        self.assertEqual(route, "end")

    def test_workflow_retries_until_citations_align(self) -> None:
        llm = SequentialFakeChatModel(
            [
                "Ankle fracture casting is preferred [1].",
                "ACE inhibitors reduce blood pressure in hypertension [1].",
            ]
        )
        workflow = XRAGGenerationWorkflow(
            settings=Settings(
                gcp_project_id="test-project",
                langgraph_max_generation_attempts=3,
            ),
            node=GeminiGenerationNode(
                settings=Settings(gcp_project_id="test-project"),
                llm=llm,
            ),
        )

        result = workflow.run(
            "hypertension treatment",
            [_context()],
            max_generation_attempts=3,
        )

        self.assertEqual(llm.calls, 2)
        self.assertTrue(result["validation_passed"])
        self.assertIn("[1]", result["answer"])
        self.assertIsNone(result["error"])
        self.assertTrue(
            any(
                "validation failed" in str(message.content).lower()
                for message in (llm.last_messages or [])
                if hasattr(message, "content")
            )
        )

    def test_workflow_stops_after_max_attempts(self) -> None:
        llm = SequentialFakeChatModel(
            [
                "Ankle fracture casting is preferred [1].",
                "Dental prophylaxis checklist is required [1].",
            ]
        )
        workflow = XRAGGenerationWorkflow(
            settings=Settings(
                gcp_project_id="test-project",
                langgraph_max_generation_attempts=2,
            ),
            node=GeminiGenerationNode(
                settings=Settings(gcp_project_id="test-project"),
                llm=llm,
            ),
        )

        result = workflow.run(
            "hypertension treatment",
            [_context()],
            max_generation_attempts=2,
        )

        self.assertEqual(llm.calls, 2)
        self.assertFalse(result["validation_passed"])
        self.assertTrue(result["validation_issues"])
        self.assertIsNotNone(result["error"])


if __name__ == "__main__":
    unittest.main()
