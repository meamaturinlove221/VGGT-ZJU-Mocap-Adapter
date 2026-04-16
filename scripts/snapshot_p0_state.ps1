[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$StatusDir = "logs/modal_phase5",
    [string]$OutRoot = "logs/modal_phase5/snapshots"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $OutRoot ("p0_state_" + $stamp)
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$files = @(
    (Join-Path $StatusDir "modal_run_progress_latest.json"),
    (Join-Path $StatusDir "overnight_ghost_autoloop_latest.json"),
    (Join-Path $StatusDir "overnight_ghost_autoloop_latest.md"),
    (Join-Path $StatusDir "watch_ghost_outputs_latest.json"),
    (Join-Path $StatusDir "watch_ghost_outputs_latest.md"),
    (Join-Path $StatusDir "vggt_ft_sweep_latest.csv"),
    (Join-Path $StatusDir "ghost_mvdepth_sweep_latest.csv"),
    (Join-Path $StatusDir "ghost_autoloop_latest.csv"),
    "docs/p0_resume_single_run_20260307.md"
)

$manifest = New-Object System.Collections.Generic.List[object]
foreach ($f in $files) {
    if (-not (Test-Path $f)) { continue }
    $name = Split-Path $f -Leaf
    Copy-Item $f (Join-Path $outDir $name) -Force
    $it = Get-Item $f
    $manifest.Add([pscustomobject]@{
        source = $f
        copied_name = $name
        last_write = $it.LastWriteTime.ToString("yyyy-MM-ddTHH:mm:ss")
        length = [int64]$it.Length
    }) | Out-Null
}

$manifestPath = Join-Path $outDir "manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "[snapshot] out_dir=$outDir"
Write-Host "[snapshot] files=$($manifest.Count)"
