$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path "D:\00-Sistemas_Dev\.venv\Scripts\Activate.ps1")) {
    throw "Ambiente virtual não encontrado em D:\00-Sistemas_Dev\.venv"
}

& "D:\00-Sistemas_Dev\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -e .

if (-not (Test-Path ".env.local")) {
    Copy-Item ".env.example" ".env.local"
}

Write-Host ""
Write-Host "Instalação concluída."
Write-Host "Edite .env.local e execute: .\scripts\status.ps1"
