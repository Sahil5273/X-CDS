"""Build the Gold-Standard clinical RAG database containing 50+ papers and WHO guidelines."""

from __future__ import annotations

import json
from pathlib import Path
from backend.app.ingestion.bioc import BioCClient, parse_bioc_json
from backend.app.search.bm25 import BM25Index
from backend.app.vector.chroma_store import ChromaVectorStore

# Paths
FIXTURE_PATH = Path("tests/fixtures/bioc_sample.json")
OUTPUT_JSONL = Path("data/bioc_chunks.jsonl")
BM25_OUTPUT = Path("data/bm25_corpus.jsonl")

# Extended list of 50 PMC articles covering Zika, Dengue, Chikungunya, WNV, Yellow Fever, Mayaro, and Oropouche
PMC_IDS = [
    # --- Official WHO / PAHO Guidelines ---
    "PMC7114207",   # WHO Dengue guidelines for diagnosis and treatment
    "PMC8439978",   # WHO guidelines on clinical management of arboviral diseases
    "PMC6486105",   # Comparative analysis of national dengue guidelines
    "PMC7122295",   # Clinical Management of Dengue (PAHO/WHO)
    
    # --- Tropical Fevers / AUFI Consensus ---
    "PMC3943129",   # ISCCM Tropical Fevers Management Guidelines (consensus)
    "PMC10309833",  # Epidemiology and Clinical Management of Severe Malaria
    "PMC10156942",  # Diagnosis and Management of Enteric Fever in Children
    "PMC6217421",   # Clinical Guidelines for Severe Leptospirosis
    
    # --- Zika Virus ---
    "PMC7403212",   # Zika neurosensory response
    "PMC5759716",   # Zika, Chikungunya, Dengue causes & threats review
    "PMC5061618",   # Zika virus transmission and pathogenesis review
    "PMC4936165",   # Congenital Zika Syndrome overview
    "PMC5139421",   # Fetal complications of Zika virus
    "PMC5088487",   # Zika clinical presentation and guidelines
    "PMC5384661",   # Zika diagnostics review
    "PMC5034633",   # Zika neuropathology
    "PMC4967399",   # Zika vaccine development review
    
    # --- Dengue Fever ---
    "PMC4567228",   # Dengue and Chikungunya comparative challenges
    "PMC8318625",   # Pediatric arbovirus infections (Dengue/Zika/Chiky)
    "PMC5409854",   # Severe Dengue classification and warning signs
    "PMC3317603",   # Indian adaptation of WHO Dengue Guidelines
    "PMC5982736",   # DENV/CHIKV serodiagnosis options
    "PMC6058244",   # Differentiating Dengue joint pain from Chikungunya
    "PMC7471908",   # Dengue Shock Syndrome management
    "PMC6346765",   # Guidelines for fluid therapy in severe Dengue
    "PMC3186191",   # Dengue vaccine candidates review
    "PMC7015421",   # Dengue treatment outcomes
    
    # --- Chikungunya ---
    "PMC3500242",   # Chikungunya clinical features and treatment review
    "PMC4712191",   # Chikungunya rheumatological manifestations
    "PMC4909384",   # Chronic Chikungunya arthritis management
    "PMC5400262",   # Chikungunya epidemiology and diagnostics
    "PMC5573673",   # Chikungunya acute joint pain criteria
    "PMC6503833",   # Chikungunya and Dengue Encephalitis review
    "PMC5513812",   # Chikungunya pathogenesis
    "PMC4982633",   # Chikungunya neurological complications
    
    # --- West Nile Virus ---
    "PMC4563989",   # West Nile Virus: Review of the literature
    "PMC5316377",   # Clinical aspects of West Nile virus
    "PMC7152026",   # West Nile neuroinvasive disease pathogenesis
    "PMC3111840",   # West Nile encephalitis clinical presentation
    "PMC4985025",   # Diagnostics for West Nile Virus
    "PMC5935391",   # Treatment guidelines for WNV-induced neuropathies
    "PMC7184421",   # West Nile surveillance and control
    
    # --- Yellow Fever ---
    "PMC11627485",  # Yellow Fever: Global impact, epidemiology, pathogenesis
    "PMC4089791",   # Yellow Fever vaccine safety review
    "PMC5381650",   # Yellow Fever clinical diagnostics and management
    "PMC6121404",   # Yellow Fever outbreaks in the Americas
    
    # --- Mayaro & Oropouche Virus ---
    "PMC10515904",  # Oropouche virus: An emerging threat in the Americas
    "PMC11288289",  # Oropouche virus clinical presentation
    "PMC11364506",  # Oropouche epidemiology and vector control
    "PMC10842880",  # Mayaro virus clinical features and diagnosis
    "PMC7021650",   # Mayaro virus prevalence and transmission
    
    # --- Comparative Reviews & Encephalitis ---
    "PMC10756841",  # Neurological manifestations of Dengue, Zika, and Chiky
    "PMC7151877",   # Arbovirus differential diagnostic guide
    "PMC7121404",   # Clinical approach to acute febrile arboviruses
    "PMC7120366"    # Comparative review of vector-borne encephalitis
]

def main() -> None:
    print("Step 1: Reading original baseline fixture...")
    if not FIXTURE_PATH.exists():
        print(f"Error: Fixture file not found at {FIXTURE_PATH}")
        return

    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        fixture_payload = json.load(f)
    fixture_chunks = parse_bioc_json(fixture_payload)
    print(f"Loaded {len(fixture_chunks)} passages from fixture.")

    unique_chunks = {chunk.chunk_id: chunk for chunk in fixture_chunks}

    print(f"\nStep 2: Fetching {len(PMC_IDS)} live PMC clinical articles and WHO guidelines...")
    client = BioCClient(requests_per_second=3)
    
    for idx, pmcid in enumerate(PMC_IDS, start=1):
        try:
            print(f"[{idx}/{len(PMC_IDS)}] Fetching {pmcid}...")
            live_chunks = client.fetch_chunks(pmcid)
            added_count = 0
            for chunk in live_chunks:
                if chunk.chunk_id not in unique_chunks:
                    unique_chunks[chunk.chunk_id] = chunk
                    added_count += 1
            print(f"  Fetched {len(live_chunks)} passages (Added {added_count} new unique passages).")
        except Exception as exc:
            print(f"  Warning: Failed to fetch {pmcid}: {exc}. Skipping.")

    merged_chunks = list(unique_chunks.values())
    total_passages = len(merged_chunks)
    print(f"\nGold-Standard Database Summary:")
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
    print("Indexing chunks in Chroma (Dense)...")
    dense_count = store.upsert_chunks(merged_chunks)

    print("Indexing chunks in BM25 (Sparse)...")
    sparse = BM25Index()
    sparse_count = sparse.index_chunks(merged_chunks)
    sparse.save(BM25_OUTPUT)

    print(f"\nSuccess! Gold-Standard database bootstrapped locally.")
    print(f"Total passages indexed: dense={dense_count}, sparse={sparse_count}")

if __name__ == "__main__":
    main()
