[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$StatusDir = "logs/modal_phase5",
    [string]$OutMd = "logs/modal_phase5/human_probe_summary_latest.md",
    [string]$OutPng = "logs/modal_phase5/human_probe_summary_latest.png",
    [string]$OutGrid = "logs/modal_phase5/human_probe_visual_grid_latest.png"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

$scriptPath = Join-Path $RepoDir "scripts/render_support_probe_summary.py"
if (-not (Test-Path $scriptPath)) {
    throw "render script not found: $scriptPath"
}

& python $scriptPath `
    --repo-dir $RepoDir `
    --status-dir $StatusDir `
    --out-md $OutMd `
    --out-png $OutPng `
    --out-grid $OutGrid

if ($LASTEXITCODE -ne 0) {
    throw "render_support_probe_summary.py failed with exit code $LASTEXITCODE"
}

