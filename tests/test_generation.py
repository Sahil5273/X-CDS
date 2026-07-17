"""Tests for Gemini generation node and citation schemas."""

from __future__ import annotations

import unittest
from typing import Any

from backend.app.config.settings import Settings
from backend.app.llm.citations import (
    GenerationOutput,
    SourceContext,
    build_generation_output,
    extract_citation_indices,
    resolve_citations,
)
from backend.app.llm.generation import GeminiGenerationNode
from backend.app.llm.graph import XRAGGenerationWorkflow
from backend.app.llm.prompts import build_generation_messages


class FakeChatModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.last_messages: list[Any] | None = None

    def invoke(self, messages: list[Any]) -> Any:
        self.last_messages = messages

        class Response:
            content = self.answer

        return Response()


class CitationSchemaTests(unittest.TestCase):
    def test_extracts_unique_markdown_citation_indices(self) -> None:
        answer = "ACE inhibitors reduce blood pressure [1] and are first-line [1][2]."
        self.assertEqual(extract_citation_indices(answer), [1, 2])

    def test_resolves_citations_to_source_metadata(self) -> None:
        contexts = [
            SourceContext(
                index=1,
                chunk_id="PMC1:passage:0",
                text="ACE inhibitors reduce blood pressure.",
                pmcid="PMC1",
                section="Abstract",
                source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
            )
        ]
        answer = "ACE inhibitors are recommended for hypertension [1]."

        citations = resolve_citations(answer, contexts)

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].label, "[1]")
        self.assertEqual(citations[0].chunk_id, "PMC1:passage:0")
        self.assertEqual(citations[0].source_url, "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/")

    def test_build_generation_output_requires_markdown_citations(self) -> None:
        contexts = [
            SourceContext(
                index=1,
                chunk_id="PMC1:passage:0",
                text="ACE inhibitors reduce blood pressure.",
                pmcid="PMC1",
                section="Abstract",
                source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
            )
        ]

        output = build_generation_output(
            "hypertension treatment",
            "First-line therapy includes ACE inhibitors [1].",
            contexts,
        )

        self.assertIsInstance(output, GenerationOutput)
        self.assertEqual(output.cited_indices, [1])
        self.assertEqual(output.citations[0].pmcid, "PMC1")


class GeminiGenerationNodeTests(unittest.TestCase):
    def test_generation_node_returns_answer_and_resolved_citations(self) -> None:
        contexts = [
            SourceContext(
                index=1,
                chunk_id="PMC1:passage:0",
                text="ACE inhibitors reduce blood pressure.",
                pmcid="PMC1",
                section="Abstract",
                source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
            ).model_dump()
        ]
        node = GeminiGenerationNode(
            settings=Settings(google_api_key="test-key"),
            llm=FakeChatModel(
                "For hypertension, ACE inhibitors are a first-line option [1]."
            ),
        )

        result = node(
            {
                "query": "How is hypertension treated?",
                "contexts": contexts,
                "generation_attempts": 0,
            }
        )

        self.assertIn("[1]", result["answer"])
        self.assertEqual(result["cited_indices"], [1])
        self.assertEqual(result["citations"][0]["chunk_id"], "PMC1:passage:0")
        self.assertEqual(result["generation_attempts"], 1)
        self.assertIsNone(result["error"])

    def test_prompt_includes_numbered_source_passages(self) -> None:
        contexts = [
            SourceContext(
                index=1,
                chunk_id="PMC1:passage:0",
                text="ACE inhibitors reduce blood pressure.",
                pmcid="PMC1",
                section="Abstract",
                source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
            )
        ]
        messages = build_generation_messages("hypertension treatment", contexts)
        self.assertIn("[1]", messages[1][1])
        self.assertIn("ACE inhibitors reduce blood pressure.", messages[1][1])


class XRAGGenerationWorkflowTests(unittest.TestCase):
    def test_langgraph_workflow_runs_generation_node(self) -> None:
        contexts = [
            SourceContext(
                index=1,
                chunk_id="PMC1:passage:0",
                text="ACE inhibitors reduce blood pressure.",
                pmcid="PMC1",
                section="Abstract",
                source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
            ).model_dump()
        ]
        workflow = XRAGGenerationWorkflow(
            settings=Settings(google_api_key="test-key"),
            node=GeminiGenerationNode(
                settings=Settings(google_api_key="test-key"),
                llm=FakeChatModel("ACE inhibitors reduce blood pressure [1]."),
            ),
        )

        result = workflow.run("hypertension treatment", contexts)

        self.assertIn("[1]", result["answer"])
        self.assertEqual(result["citations"][0]["pmcid"], "PMC1")
        self.assertTrue(result["validation_passed"])


if __name__ == "__main__":
    unittest.main()
