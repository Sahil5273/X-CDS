"""Bootstrap local BioC JSONL and retrieval indexes from bundled fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.ingestion.bioc import BiomedicalChunk, parse_bioc_json
from backend.app.search.bm25 import BM25Index
from backend.app.vector.chroma_store import ChromaVectorStore


DEFAULT_FIXTURE = Path("tests/fixtures/bioc_sample.json")
DEFAULT_JSONL = Path("data/bioc_chunks.jsonl")
DEFAULT_BM25 = Path("data/bm25_corpus.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="BioC JSON fixture used for offline bootstrap.",
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=DEFAULT_JSONL,
        help="Destination JSONL for normalized BioC passages.",
    )
    parser.add_argument(
        "--bm25-output",
        type=Path,
        default=DEFAULT_BM25,
        help="Destination JSONL for BM25 corpus persistence.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the Chroma collection before indexing.",
    )
    parser.add_argument(
        "--skip-if-present",
        action="store_true",
        help="Skip bootstrap when JSONL and BM25 corpus already exist.",
    )
    return parser


def ingest_fixture(fixture: Path, output: Path) -> list[BiomedicalChunk]:
    if not fixture.exists():
        raise SystemExit(f"Fixture does not exist: {fixture}")

    with fixture.open("r", encoding="utf-8") as handle:
        chunks = parse_bioc_json(json.load(handle))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return chunks


def index_retrievers(
    chunks: list[BiomedicalChunk],
    *,
    reset: bool,
    bm25_output: Path,
) -> tuple[int, int]:
    store = ChromaVectorStore()
    if reset:
        store.reset()
    dense_count = store.upsert_chunks(chunks)

    sparse = BM25Index()
    sparse_count = sparse.index_chunks(chunks)
    sparse.save(bm25_output)
    return dense_count, sparse_count


def main() -> None:
    args = build_parser().parse_args()
    if args.skip_if_present and args.jsonl_output.exists() and args.bm25_output.exists():
        print("Indexes already present; skipping bootstrap.")
        return

    chunks = ingest_fixture(args.fixture, args.jsonl_output)
    dense_count, sparse_count = index_retrievers(
        chunks,
        reset=args.reset,
        bm25_output=args.bm25_output,
    )
    print(
        f"Bootstrapped {len(chunks)} passages "
        f"(dense={dense_count}, sparse={sparse_count})"
    )


if __name__ == "__main__":
    main()
