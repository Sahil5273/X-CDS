# Wait for the active Naive RAG evaluation task to write its output file
Write-Host "Waiting for Naive RAG evaluation to complete..."
while (-not (Test-Path "data/naive_ragas_report.json")) {
    Start-Sleep 15
}

Write-Host "Naive RAG evaluation complete! Launching threshold parameter sweep..."
python -m scripts.run_threshold_sweep

# Restore default Windows sleep settings
Write-Host "Sweep complete! Restoring Windows sleep timeout to 15 minutes..."
powercfg /change standby-timeout-ac 15
