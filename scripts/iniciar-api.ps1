$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
& "D:\00-Sistemas_Dev\.venv\Scripts\Activate.ps1"
python -m uvicorn rag_ingest.api:app --host 0.0.0.0 --port 8088
