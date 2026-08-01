# PowerShell script to expand the CDSS knowledge base and update local indexes.
# Run this locally when you are ready to expand the database.

# Exit on error
$ErrorActionPreference = "Stop"

# 1. Ingest expanded literature (Zika, Dengue, Chikungunya, and West Nile Virus)
# PMC IDs:
# - PMC7403212: Zika Infection and neurosensory system
# - PMC4567228: Diagnostic options and challenges for Dengue and Chikungunya
# - PMC8318625: Dengue, Chikungunya, and Zika in children (pediatric clinical features)
# - PMC4563989: West Nile Virus literature review (clinical features and diagnosis)
# - PMC5316377: Epidemiological and clinical aspects of West Nile virus
Write-Host "Step 1: Ingesting expanded clinical literature from PMC..." -ForegroundColor Cyan
python -m scripts.ingest_bioc `
  --pmcid PMC7403212 `
  --pmcid PMC4567228 `
  --pmcid PMC8318625 `
  --pmcid PMC4563989 `
  --pmcid PMC5316377 `
  --output data/bioc_chunks.jsonl

# 2. Rebuild local Chroma (Dense) and BM25 (Sparse) indexes
Write-Host "`nStep 2: Rebuilding local vector database and BM25 index..." -ForegroundColor Cyan
python -m scripts.bootstrap_indexes --reset

Write-Host "`nLocal indexing completed successfully!" -ForegroundColor Green
Write-Host "A total of Zika, Dengue, Chikungunya, and West Nile Virus papers are now indexed locally." -ForegroundColor Green

# 3. Future evaluation steps (Commented out to prevent GCP API charges/credit usage)
Write-Host "`n=====================================================================" -ForegroundColor Yellow
Write-Host "FUTURE WORK STEPS (Requires Google Cloud API Key / Credits):" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Yellow
Write-Host "To generate the new 100-case clinical dataset:" -ForegroundColor Yellow
Write-Host "  python -m scripts.generate_clinical_dataset --count 100 --output data/my_eval_set_large.jsonl" -ForegroundColor Yellow
Write-Host ""
Write-Host "To run the Ragas evaluation on the expanded dataset:" -ForegroundColor Yellow
Write-Host "  python -m scripts.evaluate_ragas --dataset data/my_eval_set_large.jsonl --use-pipeline" -ForegroundColor Yellow
Write-Host ""
Write-Host "To run the baseline evaluation comparison:" -ForegroundColor Yellow
Write-Host "  python -m scripts.evaluate_baseline --dataset data/my_eval_set_large.jsonl" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Yellow
