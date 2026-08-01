"""Merge curated dengue-treatment / NSAID-guideline PMC articles into the local corpus and re-index.

Usage:
  python -m scripts.ingest_pilot_guidelines
  python -m scripts.ingest_pilot_guidelines --dry-run

Back up data/chroma/ before running if you may need to roll back the production index.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from backend.app.ingestion.bioc import BioCClient, BiomedicalChunk, parse_bioc_json
from backend.app.search.bm25 import BM25Index
from backend.app.vector.chroma_store import ChromaVectorStore

ROOT = Path(__file__).resolve().parents[1]
PMC_CATALOG = ROOT / "scripts" / "pilot_guideline_pmc_ids.json"
OUTPUT_JSONL = ROOT / "data" / "bioc_chunks.jsonl"
BM25_OUTPUT = ROOT / "data" / "bm25_corpus.jsonl"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "bioc_sample.json"


def load_pilot_pmc_ids() -> list[str]:
    payload = json.loads(PMC_CATALOG.read_text(encoding="utf-8"))
    return [entry["id"] for entry in payload["pmc_ids"]]


def load_existing_chunks(jsonl_path: Path) -> dict[str, BiomedicalChunk]:
    unique: dict[str, BiomedicalChunk] = {}
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                chunk = BiomedicalChunk(
                    chunk_id=str(record["chunk_id"]),
                    pmcid=str(record["pmcid"]),
                    text=str(record["text"]),
                    section=str(record.get("section", "unknown")),
                    passage_type=str(record.get("passage_type", "unknown")),
                    offset=int(record.get("offset", 0)),
                    source_url=str(record.get("source_url", "")),
                )
                unique[chunk.chunk_id] = chunk
    elif FIXTURE_PATH.exists():
        with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
            for chunk in parse_bioc_json(json.load(handle)):
                unique[chunk.chunk_id] = chunk
    return unique


def count_nsaid_hits(chunks: list[BiomedicalChunk]) -> int:
    keywords = ("ibuprofen", "nsaid", "aspirin", "contraindicat", "paracetamol", "acetaminophen")
    return sum(
        1
        for chunk in chunks
        if any(word in chunk.text.lower() for word in keywords)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report counts without writing indexes.",
    )
    parser.add_argument(
        "--backup-chroma",
        action="store_true",
        help="Copy data/chroma to data/chroma_backup_<timestamp> before re-indexing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pilot_ids = load_pilot_pmc_ids()
    unique_chunks = load_existing_chunks(OUTPUT_JSONL)
    before_count = len(unique_chunks)
    before_nsaid = count_nsaid_hits(list(unique_chunks.values()))

    print(f"Loaded {before_count} existing passages.")
    print(f"Passages mentioning NSAID/analgesic keywords (before): {before_nsaid}")
    print(f"Pilot catalog: {len(pilot_ids)} PMC articles from {PMC_CATALOG.name}")

    client = BioCClient(requests_per_second=3)
    added_by_pmc: dict[str, int] = {}

    for idx, pmcid in enumerate(pilot_ids, start=1):
        try:
            print(f"[{idx}/{len(pilot_ids)}] Fetching {pmcid}...")
            live_chunks = client.fetch_chunks(pmcid)
            added = 0
            for chunk in live_chunks:
                if chunk.chunk_id not in unique_chunks:
                    unique_chunks[chunk.chunk_id] = chunk
                    added += 1
            added_by_pmc[pmcid] = added
            print(f"  fetched={len(live_chunks)} added_new={added}")
        except Exception as exc:
            print(f"  WARNING: failed {pmcid}: {exc}")
            added_by_pmc[pmcid] = 0

    merged_chunks = list(unique_chunks.values())
    after_count = len(merged_chunks)
    after_nsaid = count_nsaid_hits(merged_chunks)

    print("\n--- Pilot ingest summary ---")
    print(f"Passages before: {before_count}")
    print(f"Passages after:  {after_count} (+{after_count - before_count})")
    print(f"NSAID-keyword passages before: {before_nsaid}")
    print(f"NSAID-keyword passages after:  {after_nsaid} (+{after_nsaid - before_nsaid})")
    for pmcid, added in added_by_pmc.items():
        print(f"  {pmcid}: +{added}")

    if args.dry_run:
        print("\nDry run complete — no files written.")
        return

    if args.backup_chroma:
        chroma_dir = ROOT / "data" / "chroma"
        if chroma_dir.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = ROOT / "data" / f"chroma_backup_{stamp}"
            shutil.copytree(chroma_dir, backup)
            print(f"\nChroma backup saved to {backup}")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as destination:
        for chunk in merged_chunks:
            destination.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    print(f"\nWrote merged corpus to {OUTPUT_JSONL}")
    print("Rebuilding Chroma + BM25 indexes...")

    store = ChromaVectorStore()
    store.reset()
    dense_count = store.upsert_chunks(merged_chunks)

    sparse = BM25Index()
    sparse_count = sparse.index_chunks(merged_chunks)
    sparse.save(BM25_OUTPUT)

    manifest = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "pilot_pmc_ids": pilot_ids,
        "added_by_pmc": added_by_pmc,
        "passage_count_before": before_count,
        "passage_count_after": after_count,
        "nsaid_keyword_passages_before": before_nsaid,
        "nsaid_keyword_passages_after": after_nsaid,
    }
    manifest_path = ROOT / "data" / "pilot_ingest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Indexed dense={dense_count}, sparse={sparse_count}")
    print(f"Manifest: {manifest_path}")
    print("\nNext: python -m scripts.build_pilot_eval_set")
    print("Then: python -m scripts.run_pilot_eval --use-pipeline")


if __name__ == "__main__":
    main()
