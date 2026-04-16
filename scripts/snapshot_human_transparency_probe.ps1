[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$StatusDir = "logs/modal_phase5",
    [string]$OutRoot = "logs/modal_phase5/snapshots",
    [string]$ProbeId,
    [string]$Label = "",
    [string]$ContractPath = "logs/modal_phase5/probe_contract_latest.json"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

function Read-JsonMaybe([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try {
        return (Get-Content -Raw -Path $Path -Encoding UTF8 | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Read-LastCsvRow([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try {
        $rows = @(Import-Csv -Path $Path)
        if ($rows.Count -le 0) { return $null }
        return $rows[$rows.Count - 1]
    } catch {
        return $null
    }
}

function Copy-TrackedFile(
    [string]$Path,
    [string]$OutDir,
    [System.Collections.Generic.List[object]]$Manifest,
    [string]$Tag = ""
) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return }
    $resolved = $Path
    if (-not [System.IO.Path]::IsPathRooted($resolved)) {
        $resolved = Join-Path $RepoDir $resolved
    }
    if (-not (Test-Path $resolved)) { return }

    $name = Split-Path $resolved -Leaf
    $target = Join-Path $OutDir $name
    Copy-Item $resolved $target -Force
    $it = Get-Item $resolved
    $Manifest.Add([pscustomobject]@{
        tag = $Tag
        source = $resolved
        copied_name = $name
        last_write = $it.LastWriteTime.ToString("yyyy-MM-ddTHH:mm:ss")
        length = [int64]$it.Length
    }) | Out-Null
}

function Add-GhostEvalTripletPaths(
    [string]$Path,
    [System.Collections.Generic.List[object]]$Collector
) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return }
    $resolved = $Path
    if (-not [System.IO.Path]::IsPathRooted($resolved)) {
        $resolved = Join-Path $RepoDir $resolved
    }
    if (-not (Test-Path $resolved)) { return }
    $leaf = Split-Path $resolved -Leaf
    if ($leaf -notmatch '^(?<prefix>.+_step)\d{6}(?<suffix>\.png)$') { return }
    $prefix = $matches.prefix
    $suffix = $matches.suffix
    $dir = Split-Path $resolved -Parent
    foreach ($idx in 0..2) {
        $cand = Join-Path $dir ("{0}{1:D6}{2}" -f $prefix, $idx, $suffix)
        if (Test-Path $cand) {
            $Collector.Add($cand) | Out-Null
        }
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$safeProbe = if ([string]::IsNullOrWhiteSpace($ProbeId)) { "unknown" } else { ($ProbeId -replace '[^A-Za-z0-9_\-]+','_') }
$safeLabel = if ([string]::IsNullOrWhiteSpace($Label)) { "" } else { "_" + ($Label -replace '[^A-Za-z0-9_\-]+','_') }
$outDir = Join-Path $OutRoot ("human_probe_" + $safeProbe + $safeLabel + "_" + $stamp)
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$manifest = New-Object System.Collections.Generic.List[object]

$baseFiles = @(
    (Join-Path $StatusDir "modal_run_progress_latest.json"),
    (Join-Path $StatusDir "overnight_ghost_autoloop_latest.json"),
    (Join-Path $StatusDir "overnight_ghost_autoloop_latest.md"),
    (Join-Path $StatusDir "watch_ghost_outputs_latest.json"),
    (Join-Path $StatusDir "watch_ghost_outputs_latest.md"),
    (Join-Path $StatusDir "candidate_result_latest.json"),
    (Join-Path $StatusDir "human_probe_summary_latest.md"),
    (Join-Path $StatusDir "human_probe_summary_latest.png"),
    (Join-Path $StatusDir "human_probe_visual_grid_latest.png"),
    (Join-Path $StatusDir "vggt_ft_sweep_latest.csv"),
    (Join-Path $StatusDir "ghost_mvdepth_sweep_latest.csv"),
    (Join-Path $StatusDir "ghost_mvdepth_sweep_latest.md"),
    (Join-Path $StatusDir "ghost_autoloop_latest.csv"),
    $ContractPath
)

foreach ($f in $baseFiles) {
    Copy-TrackedFile -Path $f -OutDir $outDir -Manifest $manifest -Tag "base"
}

$watch = Read-JsonMaybe -Path (Join-Path $StatusDir "watch_ghost_outputs_latest.json")
$auto = Read-JsonMaybe -Path (Join-Path $StatusDir "overnight_ghost_autoloop_latest.json")
$candidate = Read-JsonMaybe -Path (Join-Path $StatusDir "candidate_result_latest.json")
$ghostLast = Read-LastCsvRow -Path (Join-Path $StatusDir "ghost_mvdepth_sweep_latest.csv")

$extraPaths = New-Object System.Collections.Generic.List[object]
foreach ($p in @(
        $(if ($watch) { [string]$watch.active_candidate_result_json } else { "" }),
        $(if ($watch) { [string]$watch.best_visual_png } else { "" }),
        $(if ($watch) { [string]$watch.stage_best_strip_png } else { "" }),
        $(if ($auto) { [string]$auto.active_candidate_result_json } else { "" }),
        $(if ($auto) { [string]$auto.global_best_visual_png } else { "" }),
        $(if ($candidate) { [string]$candidate.candidate_result_json } else { "" }),
        $(if ($ghostLast) { [string]$ghostLast.best_visual_png } else { "" }),
        $(if ($ghostLast) { [string]$ghostLast.stage_best_strip_png } else { "" }),
        $(if ($ghostLast) { [string]$ghostLast.ghost_summary_csv } else { "" }),
        $(if ($ghostLast) { [string]$ghostLast.ghost_rows_csv } else { "" }),
        $(if ($ghostLast) { [string]$ghostLast.sweep_csv } else { "" }),
        $(if ($ghostLast) { [string]$ghostLast.baseline_compare_csv } else { "" })
    )) {
    if (-not [string]::IsNullOrWhiteSpace([string]$p)) {
        $extraPaths.Add([string]$p) | Out-Null
        Add-GhostEvalTripletPaths -Path ([string]$p) -Collector $extraPaths
    }
}

foreach ($p in $extraPaths) {
    Copy-TrackedFile -Path $p -OutDir $outDir -Manifest $manifest -Tag "referenced"
}

$summary = [ordered]@{
    probe_id = $ProbeId
    label = $Label
    snapshot_dir = $outDir
    generated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    copied_files = [int]$manifest.Count
    active_candidate_result_json = $(if ($auto -and $auto.PSObject.Properties["active_candidate_result_json"]) { [string]$auto.active_candidate_result_json } else { "" })
    candidate_invalid_reason = $(if ($candidate -and $candidate.PSObject.Properties["candidate_invalid_reason"]) { [string]$candidate.candidate_invalid_reason } else { "" })
}

$summaryPath = Join-Path $outDir "snapshot_summary.json"
$manifestPath = Join-Path $outDir "manifest.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -Path $summaryPath -Encoding UTF8
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "[probe-snapshot] probe=$ProbeId label=$Label out_dir=$outDir files=$($manifest.Count)"
