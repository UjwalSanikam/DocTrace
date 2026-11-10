param([string]$Checkpoint = "outputs/release_v1/model.pt", [int]$Port = 8765)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Checkpoint missing: $Checkpoint. Train/export a checkpoint as described in README.md, or pass -Checkpoint."
}
$runtime = Join-Path $PSScriptRoot ".runtime\python313\python.exe"
$venv = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $runtime) { $python = $runtime }
elseif (Test-Path -LiteralPath $venv) { $python = $venv }
else { $python = "python" }
Write-Host "Open http://127.0.0.1:$Port after the model is ready. Press Ctrl+C to stop."
& $python scripts/serve.py --checkpoint $Checkpoint --port $Port
exit $LASTEXITCODE
