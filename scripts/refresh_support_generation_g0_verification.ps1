[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$StatusDir = "logs/modal_phase5",
    [string]$OutMdEn = "logs/modal_phase5/reports/support_generation_g0_verification_latest.md",
    [string]$OutPngEn = "logs/modal_phase5/reports/support_generation_g0_verification_latest.png",
    [string]$OutMdZh = "logs/modal_phase5/reports/support_generation_g0_verification_latest_zh.md",
    [string]$OutPngZh = "logs/modal_phase5/reports/support_generation_g0_verification_latest_zh.png"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

$scriptPath = Join-Path $RepoDir "scripts/render_support_generation_g0_verification.py"
if (-not (Test-Path $scriptPath)) {
    throw "verification render script not found: $scriptPath"
}

$t0Json = Join-Path $StatusDir "candidate_result_ghost_mv_mv_0.001_mvmask_0_default_cand01_20260310_163501.json"
$s0Json = Join-Path $StatusDir "candidate_result_ghost_mv_mv_0.001_mvmask_0_default_cand01_20260310_171503.json"
$g0Json = Join-Path $StatusDir "candidate_result_ghost_mv_mv_0.001_mvmask_0_default_cand01_20260311_014102.json"

& python $scriptPath `
    --t0-json $t0Json `
    --s0-json $s0Json `
    --g0-json $g0Json `
    --out-md-en $OutMdEn `
    --out-png-en $OutPngEn `
    --out-md-zh $OutMdZh `
    --out-png-zh $OutPngZh

if ($LASTEXITCODE -ne 0) {
    throw "render_support_generation_g0_verification.py failed with exit code $LASTEXITCODE"
}
