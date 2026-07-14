param(
    [Parameter(Mandatory = $true)]
    [string]$Collection,

    [Parameter(Mandatory = $true)]
    [string]$Question,

    [ValidateSet("normal", "executiva", "tecnica", "academica", "resumo")]
    [string]$Style = "normal"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
& "D:\00-Sistemas_Dev\.venv\Scripts\Activate.ps1"

rag-ask `
  --collection $Collection `
  --question $Question `
  --style $Style
