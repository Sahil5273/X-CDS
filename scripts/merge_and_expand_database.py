"""Merge the pre-bundled clinical sample fixture with expanded live PMC articles."""

from __future__ import annotations

import json
from pathlib import Path
from backend.app.ingestion.bioc import BioCClient, parse_bioc_json, BiomedicalChunk
from backend.app.search.bm25 import BM25Index
from backend.app.vector.chroma_store import ChromaVectorStore

# Paths
FIXTURE_PATH = Path("tests/fixtures/bioc_sample.json")
OUTPUT_JSONL = Path("data/bioc_chunks.jsonl")
BM25_OUTPUT = Path("data/bm25_corpus.jsonl")

# List of PMC articles to ingest live
PMC_IDS = [
    "PMC7403212",  # Zika and neurosensory system
    "PMC4567228",  # Dengue and Chikungunya diagnostic options
    "PMC8318625",  # Pediatric features of Zika, Dengue, Chikungunya
    "PMC4563989",  # West Nile Virus review
    "PMC5316377",  # Epidemiological & clinical WNV
    "PMC5759716",  # Zika, Chikungunya, Dengue causes & threats review
    "PMC10756841", # Neurological manifestations of Dengue, Zika, Chikungunya
]

def main() -> None:
    print("Step 1: Reading original pre-bundled sample fixture...")
    if not FIXTURE_PATH.exists():
        print(f"Error: Fixture file not found at {FIXTURE_PATH}")
        return

    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        fixture_payload = json.load(f)
    fixture_chunks = parse_bioc_json(fixture_payload)
    print(f"Loaded {len(fixture_chunks)} passages from fixture.")

    # Unique chunks dictionary to prevent duplicate chunk_ids
    unique_chunks = {chunk.chunk_id: chunk for chunk in fixture_chunks}

    print("\nStep 2: Fetching live PMC clinical reviews and guidelines...")
    client = BioCClient(requests_per_second=3)
    
    for pmcid in PMC_IDS:
        try:
            print(f"Fetching {pmcid}...")
            live_chunks = client.fetch_chunks(pmcid)
            added_count = 0
            for chunk in live_chunks:
                if chunk.chunk_id not in unique_chunks:
                    unique_chunks[chunk.chunk_id] = chunk
                    added_count += 1
            print(f"Fetched {len(live_chunks)} passages from {pmcid} (Added {added_count} new unique passages).")
        except Exception as exc:
            print(f"Warning: Failed to fetch {pmcid}: {exc}. Skipping this paper.")

    merged_chunks = list(unique_chunks.values())
    total_passages = len(merged_chunks)
    print(f"\nMerged Database Summary:")
    print(f"Total Unique Passages: {total_passages}")

    print(f"\nStep 3: Writing merged passages to {OUTPUT_JSONL}...")
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as destination:
        for chunk in merged_chunks:
            destination.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    print("\nStep 4: Rebuilding search indexes (ChromaDB and BM25)...")
    store = ChromaVectorStore()
    print("Resetting Chroma collection...")
    store.reset()
    print("Index chunks in Chroma (Dense)...")
    dense_count = store.upsert_chunks(merged_chunks)

    print("Index chunks in BM25 (Sparse)...")
    sparse = BM25Index()
    sparse_count = sparse.index_chunks(merged_chunks)
    sparse.save(BM25_OUTPUT)

    print(f"\nSuccess! Merged database bootstrapped locally.")
    print(f"Total passages indexed: dense={dense_count}, sparse={sparse_count}")

if __name__ == "__main__":
    main()
