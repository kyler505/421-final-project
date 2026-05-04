# Emit pinned dependency list for reproducibility / Canvas zip documentation.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Test-Path .venv)) {
    Write-Error "Create and activate .venv first, then run: pip install -r requirements.txt"
}
.\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 requirements-frozen.txt
Write-Host "Wrote requirements-frozen.txt"
