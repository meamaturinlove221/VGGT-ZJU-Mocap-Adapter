[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$ReportsDir = "logs/modal_phase5/reports",
    [string]$MultiframeJson = "logs/modal_phase5/reports/support_generation_multiframe_latest.json",
    [string]$ContractJson = "logs/modal_phase5/probe_contract_g0_latest.json"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

$scriptPath = Join-Path $RepoDir "scripts/render_support_generation_g0_readiness.py"
if (-not (Test-Path $scriptPath)) {
    throw "render script not found: $scriptPath"
}

$outMdEn = Join-Path $ReportsDir "support_generation_g0_readiness_latest.md"
$outPngEn = Join-Path $ReportsDir "support_generation_g0_readiness_latest.png"
$outMdZh = Join-Path $ReportsDir "support_generation_g0_readiness_latest_zh.md"
$outPngZh = Join-Path $ReportsDir "support_generation_g0_readiness_latest_zh.png"

& python $scriptPath `
    --multiframe-json (Join-Path $RepoDir $MultiframeJson) `
    --contract-json (Join-Path $RepoDir $ContractJson) `
    --out-md-en (Join-Path $RepoDir $outMdEn) `
    --out-png-en (Join-Path $RepoDir $outPngEn) `
    --out-md-zh (Join-Path $RepoDir $outMdZh) `
    --out-png-zh (Join-Path $RepoDir $outPngZh)

if ($LASTEXITCODE -ne 0) {
    throw "render_support_generation_g0_readiness.py failed with exit code $LASTEXITCODE"
}

Write-Host ("[refresh-g0-readiness] wrote " + (Join-Path $RepoDir $outMdZh))
