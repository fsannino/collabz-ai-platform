param(
    [Parameter(Mandatory = $true)]
    [string]$Collection,

    [Parameter(Mandatory = $true)]
    [string]$Query
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
& "D:\00-Sistemas_Dev\.venv\Scripts\Activate.ps1"
rag-query --collection $Collection --query $Query
