"""Index BioC JSONL passages into the local Chroma dense vector store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.ingestion.bioc import BiomedicalChunk
from backend.app.vector.chroma_store import ChromaVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/bioc_chunks.jsonl"),
        help="JSONL file produced by scripts/ingest_bioc.py.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=None,
        help="Optional Chroma persistence directory override.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Optional Chroma collection name override.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the collection before indexing.",
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
    store = ChromaVectorStore(
        persist_directory=args.persist_dir,
        collection_name=args.collection,
    )
    if args.reset:
        store.reset()

    chunks = load_chunks(args.input)
    upserted = store.upsert_chunks(chunks)
    print(
        f"Indexed {upserted} passages into collection "
        f"'{store.collection_name}' at {store.persist_directory}"
    )


if __name__ == "__main__":
    main()
