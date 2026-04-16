[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$OutRoot = "logs/modal_phase5/reports",
    [string]$DatasetRoot = "F:\datasets\ZJU_MoCap\data\zju_mocap",
    [string]$NpzGlob = "logs/modal_phase5/geom_samples/*.npz",
    [int]$FgPreservePx = 3
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $OutRoot ("support_generation_multiframe_" + $ts)
python scripts\aggregate_support_generation_diagnosis.py --npz-glob $NpzGlob --dataset-root $DatasetRoot --out-dir $outDir --fg-preserve-px $FgPreservePx

$latestMd = Join-Path $OutRoot "support_generation_multiframe_latest.md"
$latestPng = Join-Path $OutRoot "support_generation_multiframe_latest.png"
$latestJson = Join-Path $OutRoot "support_generation_multiframe_latest.json"
Copy-Item (Join-Path $outDir "support_generation_multiframe_summary.md") $latestMd -Force
Copy-Item (Join-Path $outDir "support_generation_multiframe_summary.png") $latestPng -Force
Copy-Item (Join-Path $outDir "support_generation_multiframe_summary.json") $latestJson -Force

Write-Host ("[support-gen-multiframe] wrote " + $outDir)
