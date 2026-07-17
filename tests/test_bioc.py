"""Tests for BioC parsing and request-rate constraints."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from backend.app.ingestion.bioc import BioCError, RequestRateLimiter, parse_bioc_json

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bioc_sample.json"


class ParseBioCJsonTests(unittest.TestCase):
    def test_parses_passages_into_traceable_chunks(self) -> None:
        with FIXTURE_PATH.open("r", encoding="utf-8") as fixture:
            chunks = parse_bioc_json(json.load(fixture))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].chunk_id, "PMC1234567:passage:0")
        self.assertEqual(chunks[0].section, "Title")
        self.assertEqual(chunks[1].passage_type, "abstract")
        self.assertEqual(
            chunks[1].source_url,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
        )

    def test_accepts_collection_lists_returned_by_bioc(self) -> None:
        payload = [
            {
                "documents": [
                    {
                        "id": "42",
                        "passages": [
                            {
                                "offset": "0",
                                "infons": {"type": "abstract"},
                                "text": "Clinical evidence.",
                            }
                        ],
                    }
                ]
            }
        ]

        chunks = parse_bioc_json(payload)

        self.assertEqual(chunks[0].pmcid, "PMC42")

    def test_rejects_payload_without_documents(self) -> None:
        with self.assertRaisesRegex(BioCError, "contains no documents"):
            parse_bioc_json({"documents": []})


class RequestRateLimiterTests(unittest.TestCase):
    def test_rejects_rates_above_ncbi_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 3"):
            RequestRateLimiter(requests_per_second=3.01)


if __name__ == "__main__":
    unittest.main()
