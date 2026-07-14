$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
& "D:\00-Sistemas_Dev\.venv\Scripts\Activate.ps1"
rag-status
