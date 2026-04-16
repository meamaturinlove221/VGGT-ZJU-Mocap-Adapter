[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$StatusDir = "logs/modal_phase5"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

function Read-JsonMaybe([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try { return (Get-Content $Path -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { return $null }
}

function Get-CsvColumns([string]$Path) {
    if (-not (Test-Path $Path)) { return @() }
    $first = Get-Content $Path -TotalCount 1 -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($first)) { return @() }
    $header = $first.Trim()
    if ($header.StartsWith('"') -and $header.EndsWith('"')) {
        $header = $header.Substring(1, $header.Length - 2)
    }
    return @($header -split '","')
}

$requiredFiles = @(
    (Join-Path $StatusDir "modal_run_progress_latest.json"),
    (Join-Path $StatusDir "overnight_ghost_autoloop_latest.json"),
    (Join-Path $StatusDir "watch_ghost_outputs_latest.json"),
    (Join-Path $StatusDir "candidate_result_latest.json"),
    (Join-Path $StatusDir "vggt_ft_sweep_latest.csv"),
    (Join-Path $StatusDir "ghost_mvdepth_sweep_latest.csv"),
    (Join-Path $StatusDir "ghost_autoloop_latest.csv"),
    "docs/p0_resume_single_run_20260307.md"
)

$missing = @($requiredFiles | Where-Object { -not (Test-Path $_) })
$autoloopStatus = Read-JsonMaybe -Path (Join-Path $StatusDir "overnight_ghost_autoloop_latest.json")
$watchStatus = Read-JsonMaybe -Path (Join-Path $StatusDir "watch_ghost_outputs_latest.json")
$candidateResult = Read-JsonMaybe -Path (Join-Path $StatusDir "candidate_result_latest.json")
$ftCols = Get-CsvColumns -Path (Join-Path $StatusDir "vggt_ft_sweep_latest.csv")
$ghostCols = Get-CsvColumns -Path (Join-Path $StatusDir "ghost_mvdepth_sweep_latest.csv")
$autoCols = Get-CsvColumns -Path (Join-Path $StatusDir "ghost_autoloop_latest.csv")

$requiredGhostCols = @(
    "precompute_mv_support_on",
    "point_target_blend_by_mv_support",
    "candidate_invalid_reason",
    "precompute_source_requested",
    "precompute_source_resolved",
    "precompute_fallback_used",
    "precompute_timeout_hit"
)
$requiredAutoCols = @(
    "precompute_mv_support_on",
    "point_target_blend_by_mv_support",
    "precompute_source_requested",
    "precompute_source_resolved",
    "precompute_fallback_used",
    "precompute_timeout_hit"
)

$missingGhostCols = @($requiredGhostCols | Where-Object { $_ -notin $ghostCols })
$missingAutoCols = @($requiredAutoCols | Where-Object { $_ -notin $autoCols })

$report = [ordered]@{
    updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    missing_files = ($missing -join "; ")
    autoloop_current_stage = $(if ($autoloopStatus) { [string]$autoloopStatus.current_stage } else { "" })
    autoloop_p0_gate_pass = $(if ($autoloopStatus) { [string]$autoloopStatus.p0_gate_pass } else { "" })
    autoloop_precompute_mv_support_on = $(if ($autoloopStatus) { [string]$autoloopStatus.active_precompute_mv_support_on } else { "" })
    autoloop_point_target_blend_by_mv_support = $(if ($autoloopStatus) { [string]$autoloopStatus.active_point_target_blend_by_mv_support } else { "" })
    autoloop_candidate_result_json = $(if ($autoloopStatus) { [string]$autoloopStatus.active_candidate_result_json } else { "" })
    watch_current_stage = $(if ($watchStatus) { [string]$watchStatus.current_stage } else { "" })
    candidate_result_source = $(if ($candidateResult) { [string]$candidateResult.source } else { "" })
    candidate_result_json = $(if ($candidateResult) { [string]$candidateResult.candidate_result_json } else { "" })
    missing_ghost_csv_columns = ($missingGhostCols -join "; ")
    missing_autoloop_csv_columns = ($missingAutoCols -join "; ")
    ft_csv_has_declared_views = [string]("eval_num_src_views_declared" -in $ftCols)
    ghost_csv_has_candidate_invalid_reason = [string]("candidate_invalid_reason" -in $ghostCols)
}

$report.GetEnumerator() | ForEach-Object { Write-Host ("{0}: {1}" -f $_.Key, $_.Value) }

$ok = $true
if ($missing.Count -gt 0) { $ok = $false }
if ($missingGhostCols.Count -gt 0) { $ok = $false }
if ($missingAutoCols.Count -gt 0) { $ok = $false }
if (-not ("eval_num_src_views_declared" -in $ftCols)) { $ok = $false }
if (-not ("candidate_invalid_reason" -in $ghostCols)) { $ok = $false }

if (-not $ok) { exit 2 }
