"""Build a BM25 sparse index from BioC JSONL passages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.ingestion.bioc import BiomedicalChunk
from backend.app.search.bm25 import BM25Index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/bioc_chunks.jsonl"),
        help="JSONL file produced by scripts/ingest_bioc.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bm25_corpus.jsonl"),
        help="Destination JSONL path for the BM25 corpus.",
    )
    return parser


def load_chunks(path: Path) -> list[BiomedicalChunk]:
    if not path.exists():
        raise SystemExit(f"Input JSONL does not exist: {path}")

    chunks: list[BiomedicalChunk] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
                chunks.append(
                    BiomedicalChunk(
                        chunk_id=str(record["chunk_id"]),
                        pmcid=str(record["pmcid"]),
                        text=str(record["text"]),
                        section=str(record["section"]),
                        passage_type=str(record["passage_type"]),
                        offset=int(record["offset"]),
                        source_url=str(record["source_url"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SystemExit(
                    f"Invalid BioC chunk on line {line_number} of {path}: {exc}"
                ) from exc
    return chunks


def main() -> None:
    args = build_parser().parse_args()
    index = BM25Index()
    indexed = index.index_chunks(load_chunks(args.input))
    index.save(args.output)
    print(f"Indexed {indexed} passages into BM25 corpus at {args.output}")


if __name__ == "__main__":
    main()
