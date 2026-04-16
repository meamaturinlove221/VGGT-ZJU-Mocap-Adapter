[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [Parameter(Mandatory = $true)]
    [string]$CandidateJsonPath,
    [Parameter(Mandatory = $true)]
    [string]$MetricsJsonlPath
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

function Get-MetricsSnapshot([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try {
        $lines = Get-Content $Path | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        $events = New-Object System.Collections.Generic.List[object]
        foreach ($ln in $lines) {
            try { $events.Add(($ln | ConvertFrom-Json)) } catch {}
        }
        for ($i = $events.Count - 1; $i -ge 0; $i--) {
            $ev = $events[$i]
            try {
                $evName = [string]$ev.event
                if ($evName -eq "epoch_end" -or $evName -eq "step_eval") {
                    return $ev
                }
            } catch {}
        }
        if ($events.Count -gt 0) { return $events[$events.Count - 1] }
        return $null
    } catch {
        return $null
    }
}

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $abs = Join-Path (Resolve-Path ".").Path $Path
    $dir = Split-Path -Parent $abs
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $json = $Obj | ConvertTo-Json -Depth 30
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($abs, $json, $enc)
}

$candidate = Read-JsonMaybe -Path $CandidateJsonPath
if ($null -eq $candidate) { throw "candidate json not found or invalid: $CandidateJsonPath" }
$snapshot = Get-MetricsSnapshot -Path $MetricsJsonlPath
if ($null -eq $snapshot) { throw "metrics snapshot not found: $MetricsJsonlPath" }

$names = @(
    "point_support_path_active",
    "point_mv_depth_support_path_active",
    "point_mv_mask_support_path_active",
    "point_target_blend_mv_support_active",
    "point_mv_mode",
    "point_mv_support_mean","point_mv_support_p10","point_mv_support_p90",
    "point_mv_support_fg_mean","point_mv_support_fg_p10","point_mv_support_fg_p90",
    "point_mv_support_bg_mean","point_mv_support_bg_p10","point_mv_support_bg_p90",
    "point_mv_pseudo_support_mean","point_mv_pseudo_support_p10","point_mv_pseudo_support_p90",
    "point_mv_pseudo_support_fg_mean","point_mv_pseudo_support_fg_p10","point_mv_pseudo_support_fg_p90",
    "point_mv_pseudo_support_bg_mean","point_mv_pseudo_support_bg_p10","point_mv_pseudo_support_bg_p90",
    "point_support_eff_mean","point_support_eff_p10","point_support_eff_p90",
    "point_support_eff_fg_mean","point_support_eff_fg_p10","point_support_eff_fg_p90",
    "point_support_eff_bg_mean","point_support_eff_bg_p10","point_support_eff_bg_p90",
    "point_mv_depth_support_eff_mean","point_mv_depth_support_eff_p10","point_mv_depth_support_eff_p90",
    "point_mv_depth_support_eff_fg_mean","point_mv_depth_support_eff_fg_p10","point_mv_depth_support_eff_fg_p90",
    "point_mv_depth_support_eff_bg_mean","point_mv_depth_support_eff_bg_p10","point_mv_depth_support_eff_bg_p90",
    "point_mv_mask_support_eff_mean","point_mv_mask_support_eff_p10","point_mv_mask_support_eff_p90",
    "point_mv_mask_support_eff_fg_mean","point_mv_mask_support_eff_fg_p10","point_mv_mask_support_eff_fg_p90",
    "point_mv_mask_support_eff_bg_mean","point_mv_mask_support_eff_bg_p10","point_mv_mask_support_eff_bg_p90"
)

foreach ($name in $names) {
    try {
        if ($snapshot.PSObject.Properties[$name]) {
            $candidate | Add-Member -NotePropertyName $name -NotePropertyValue $snapshot.$name -Force
        }
    } catch {}
}
$candidate | Add-Member -NotePropertyName support_metrics_backfilled_from -NotePropertyValue $MetricsJsonlPath -Force
$candidate | Add-Member -NotePropertyName support_metrics_backfilled_at -NotePropertyValue ((Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")) -Force

Write-JsonNoBom -Path $CandidateJsonPath -Obj $candidate
Write-Host "[candidate-support-backfill] updated $CandidateJsonPath from $MetricsJsonlPath"
