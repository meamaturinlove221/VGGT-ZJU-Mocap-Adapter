[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [switch]$SkipSnapshot,
    [switch]$SkipManifest
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

function Invoke-LocalScript([string]$RelativePath) {
    $fullPath = Join-Path $RepoDir $RelativePath
    Write-Host "[local-maint] running $RelativePath"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $fullPath
    if ($LASTEXITCODE -ne 0) {
        throw "local script failed: $RelativePath exit=$LASTEXITCODE"
    }
}

$steps = @(
    "scripts/repair_latest_csv_schema.ps1",
    "scripts/repair_candidate_result_latest.ps1",
    "scripts/repair_paused_local_status.ps1",
    "scripts/check_paused_state.ps1",
    "scripts/check_p0_local_readiness.ps1",
    "scripts/check_p0_source_contract.ps1",
    "scripts/check_candidate_result_consistency.ps1",
    "scripts/preflight_p0_resume_local.ps1"
)

foreach ($step in $steps) {
    Invoke-LocalScript -RelativePath $step
}

if (-not $SkipSnapshot) {
    Invoke-LocalScript -RelativePath "scripts/snapshot_p0_state.ps1"
}

if (-not $SkipManifest) {
    Invoke-LocalScript -RelativePath "scripts/emit_p0_resume_manifest.ps1"
}

Write-Host "[local-maint] result=ready_local_only"
