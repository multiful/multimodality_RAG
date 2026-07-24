$ErrorActionPreference='Continue'
$PY="C:\Users\wodlf\OneDrive\Desktop\pdfex\demo_venv\Scripts\python.exe"
Set-Location "C:\Users\wodlf\OneDrive\Desktop\파이프라인 고도화"
Write-Host "===== [1/2] MinerU parse (20 docs, skip done) $(Get-Date -Format HH:mm:ss) ====="
& $PY pipeline\s1_parse.py
Write-Host "===== [2/2] s2 image pipeline (industry) $(Get-Date -Format HH:mm:ss) ====="
& $PY pipeline\s2_image_pipeline.py --category industry
Write-Host "===== DONE $(Get-Date -Format HH:mm:ss) ====="
