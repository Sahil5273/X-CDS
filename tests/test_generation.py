"""Tests for Gemini generation node and citation schemas."""

from __future__ import annotations

import unittest
import unittest.mock
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
            settings=Settings(gcp_project_id="test-project"),
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
            settings=Settings(gcp_project_id="test-project"),
            node=GeminiGenerationNode(
                settings=Settings(gcp_project_id="test-project"),
                llm=FakeChatModel("ACE inhibitors reduce blood pressure [1]."),
            ),
        )

        result = workflow.run("hypertension treatment", contexts)

        self.assertIn("[1]", result["answer"])
        self.assertEqual(result["citations"][0]["pmcid"], "PMC1")
        self.assertTrue(result["validation_passed"])


class RobustChatVertexAITests(unittest.TestCase):
    @unittest.mock.patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_successful_invocation_on_first_try(self, mock_chat_class: unittest.mock.MagicMock) -> None:
        mock_instance = unittest.mock.MagicMock()
        mock_instance.invoke.return_value = "Response content"
        mock_chat_class.return_value = mock_instance

        from backend.app.llm.generation import RobustChatVertexAI
        from backend.app.config.settings import Settings

        wrapper = RobustChatVertexAI("gemini-3.5-flash", Settings(gcp_project_id="test-project", gcp_region="us-central1"))
        res = wrapper.invoke([])

        self.assertEqual(res, "Response content")
        mock_chat_class.assert_called_once_with(
            model="gemini-3.5-flash",
            vertexai=True,
            project="test-project",
            location="us-central1",
            temperature=0.1,
            max_retries=1,
        )

    @unittest.mock.patch("time.sleep", return_value=None)  # disable sleeping
    @unittest.mock.patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_retries_on_transient_error_and_succeeds(self, mock_chat_class: unittest.mock.MagicMock, mock_sleep: unittest.mock.MagicMock) -> None:
        # First 2 attempts fail with a transient error (503), 3rd attempt succeeds
        mock_fail = unittest.mock.MagicMock()
        mock_fail.invoke.side_effect = Exception("503 Service Unavailable")

        mock_success = unittest.mock.MagicMock()
        mock_success.invoke.return_value = "Success content"

        mock_chat_class.side_effect = [mock_fail, mock_fail, mock_success]

        from backend.app.llm.generation import RobustChatVertexAI
        from backend.app.config.settings import Settings

        wrapper = RobustChatVertexAI("gemini-3.5-flash", Settings(gcp_project_id="test-project", gcp_region="us-central1"))
        res = wrapper.invoke([])

        self.assertEqual(res, "Success content")
        self.assertEqual(mock_chat_class.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @unittest.mock.patch("time.sleep", return_value=None)
    @unittest.mock.patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_falls_back_to_flash_lite_when_primary_exhausted(self, mock_chat_class: unittest.mock.MagicMock, mock_sleep: unittest.mock.MagicMock) -> None:
        # Primary model (gemini-3.5-flash) fails 4 times with transient error
        # Fallback model (gemini-3.1-flash-lite) succeeds on its first attempt
        mock_fail = unittest.mock.MagicMock()
        mock_fail.invoke.side_effect = Exception("503 Service Unavailable")

        mock_success = unittest.mock.MagicMock()
        mock_success.invoke.return_value = "Fallback content"

        # 4 fails for primary, 1 success for fallback
        mock_chat_class.side_effect = [mock_fail] * 4 + [mock_success]

        from backend.app.llm.generation import RobustChatVertexAI
        from backend.app.config.settings import Settings

        wrapper = RobustChatVertexAI("gemini-3.5-flash", Settings(gcp_project_id="test-project", gcp_region="us-central1"))
        res = wrapper.invoke([])

        self.assertEqual(res, "Fallback content")
        # 4 attempts on primary + 1 attempt on fallback = 5 total
        self.assertEqual(mock_chat_class.call_count, 5)
        # Verify the 5th call was instantiated with fallback model
        last_call_args = mock_chat_class.call_args_list[-1]
        self.assertEqual(last_call_args.kwargs["model"], "gemini-3.1-flash-lite")

    @unittest.mock.patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_raises_non_transient_error_immediately(self, mock_chat_class: unittest.mock.MagicMock) -> None:
        mock_instance = unittest.mock.MagicMock()
        mock_instance.invoke.side_effect = Exception("401 Unauthorized")
        mock_chat_class.return_value = mock_instance

        from backend.app.llm.generation import RobustChatVertexAI
        from backend.app.config.settings import Settings

        wrapper = RobustChatVertexAI("gemini-3.5-flash", Settings(gcp_project_id="test-project", gcp_region="us-central1"))

        with self.assertRaises(Exception) as context:
            wrapper.invoke([])

        self.assertIn("401", str(context.exception))
        # Should not retry or fallback
        self.assertEqual(mock_chat_class.call_count, 1)


if __name__ == "__main__":
    unittest.main()
