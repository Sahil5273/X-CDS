"""Ingest BioC JSON from a local fixture or the PMC BioC API into JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.app.ingestion.bioc import BioCClient, BiomedicalChunk, parse_bioc_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--mock-file",
        type=Path,
        help="Path to a local BioC JSON payload for offline ingestion.",
    )
    source.add_argument(
        "--pmcid",
        action="append",
        help="PMC identifier to fetch; repeat this flag for multiple articles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bioc_chunks.jsonl"),
        help="Destination JSONL path (default: data/bioc_chunks.jsonl).",
    )
    return parser


def load_mock_payload(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except FileNotFoundError as exc:
        raise SystemExit(f"Mock BioC file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Mock BioC file is not valid JSON: {path}") from exc


def collect_chunks(args: argparse.Namespace) -> list[BiomedicalChunk]:
    if args.mock_file:
        return parse_bioc_json(load_mock_payload(args.mock_file))

    client = BioCClient(requests_per_second=3)
    chunks: list[BiomedicalChunk] = []
    for pmcid in args.pmcid:
        chunks.extend(client.fetch_chunks(pmcid))
    return chunks


def write_jsonl(chunks: list[BiomedicalChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        for chunk in chunks:
            destination.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    chunks = collect_chunks(args)
    write_jsonl(chunks, args.output)
    print(f"Wrote {len(chunks)} BioC passages to {args.output}")


if __name__ == "__main__":
    main()
