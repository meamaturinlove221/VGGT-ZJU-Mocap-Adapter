[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$StatusDir = "logs/modal_phase5",
    [string]$ProbeId = "",
    [ValidateSet("running","done","error")]
    [string]$State = "done",
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

function Read-CsvRows([string]$Path) {
    if (-not (Test-Path $Path)) { return @() }
    try {
        return @(Import-Csv -Path $Path)
    } catch {
        return @()
    }
}

function Normalize-JsonFiniteNumbers($Value) {
    if ($null -eq $Value) { return $null }
    if (($Value -is [double]) -or ($Value -is [float])) {
        $d = [double]$Value
        if ([double]::IsNaN($d) -or [double]::IsInfinity($d)) { return $null }
        return $Value
    }
    if ($Value -is [System.Collections.IDictionary]) {
        $o = [ordered]@{}
        foreach ($k in $Value.Keys) {
            $o[[string]$k] = Normalize-JsonFiniteNumbers $Value[$k]
        }
        return [pscustomobject]$o
    }
    if (($Value -is [System.Collections.IEnumerable]) -and (-not ($Value -is [string]))) {
        $arr = New-Object System.Collections.ArrayList
        foreach ($item in $Value) {
            [void]$arr.Add((Normalize-JsonFiniteNumbers $item))
        }
        return @($arr)
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $o = [ordered]@{}
        foreach ($p in $Value.PSObject.Properties) {
            $o[$p.Name] = Normalize-JsonFiniteNumbers $p.Value
        }
        return [pscustomobject]$o
    }
    return $Value
}

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    $safe = Normalize-JsonFiniteNumbers $Obj
    $json = $safe | ConvertTo-Json -Depth 20
    $fullPath = Join-Path (Resolve-Path ".").Path $Path
    $tmpPath = "$fullPath.tmp.$PID.$([DateTime]::UtcNow.Ticks)"
    [System.IO.File]::WriteAllText($tmpPath, $json, $enc)
    try {
        if ([System.IO.File]::Exists($fullPath)) {
            [System.IO.File]::Replace($tmpPath, $fullPath, $null, $true)
        } else {
            [System.IO.File]::Move($tmpPath, $fullPath)
        }
    } catch {
        try {
            [System.IO.File]::Copy($tmpPath, $fullPath, $true)
        } finally {
            if ([System.IO.File]::Exists($tmpPath)) {
                [System.IO.File]::Delete($tmpPath)
            }
        }
    }
}

function Set-Prop([object]$Obj, [string]$Name, $Value) {
    if ($null -eq $Obj) { return }
    if ($Obj.PSObject.Properties[$Name]) {
        $Obj.$Name = $Value
    } else {
        Add-Member -InputObject $Obj -MemberType NoteProperty -Name $Name -Value $Value -Force
    }
}

function To-BoolLoose($Value, [bool]$Default = $false) {
    if ($null -eq $Value) { return $Default }
    if ($Value -is [bool]) { return [bool]$Value }
    $raw = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }
    switch -Regex ($raw.ToLowerInvariant()) {
        '^(1|true|yes|y|on)$' { return $true }
        '^(0|false|no|n|off)$' { return $false }
        default { return $Default }
    }
}

function Pick-String([string[]]$Values) {
    foreach ($v in $Values) {
        if (-not [string]::IsNullOrWhiteSpace([string]$v)) { return [string]$v }
    }
    return ""
}

function Resolve-PrecisionString(
    [string]$Field,
    [object]$RunFtRow,
    [object]$Contract,
    [object]$Candidate,
    [object]$GhostLast,
    [object]$FtLast,
    [string]$Fallback = ""
) {
    $contractFallback = switch ($Field) {
        "runner_tf32" { $(if ($Contract) { [string]$Contract.tf32 } else { "" }) }
        "runner_amp" { $(if ($Contract) { [string]$Contract.amp } else { "" }) }
        "runner_strict_deterministic" { $(if ($Contract) { [string]$Contract.strict_deterministic } else { "" }) }
        "precompute_tf32" { $(if ($Contract) { [string]$Contract.tf32 } else { "" }) }
        "precompute_amp" { $(if ($Contract) { [string]$Contract.amp } else { "" }) }
        "precompute_strict_deterministic" { $(if ($Contract) { [string]$Contract.strict_deterministic } else { "" }) }
        "teacher_tf32" { $(if ($Contract) { [string]$Contract.tf32 } else { "" }) }
        "teacher_amp" { $(if ($Contract) { [string]$Contract.amp } else { "" }) }
        "teacher_deterministic" { $(if ($Contract) { [string]$Contract.strict_deterministic } else { "" }) }
        default { $(if ($Contract) { [string]$Contract.$Field } else { "" }) }
    }
    return Pick-String @(
        $(if ($RunFtRow) { [string]$RunFtRow.$Field } else { "" }),
        $contractFallback,
        $(if ($Candidate) { [string]$Candidate.$Field } else { "" }),
        $(if ($GhostLast) { [string]$GhostLast.$Field } else { "" }),
        $(if ($FtLast) { [string]$FtLast.$Field } else { "" }),
        $Fallback
    )
}

function Pick-Bool($Primary, $Fallback = $false) {
    if ($null -eq $Primary) { return (To-BoolLoose $Fallback $false) }
    return (To-BoolLoose $Primary $false)
}

function Is-BlankOrNaN($Value) {
    if ($null -eq $Value) { return $true }
    $raw = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($raw)) { return $true }
    return $raw.ToLowerInvariant() -eq "nan"
}

function To-DoubleLoose($Value, [double]$Default = [double]::NaN) {
    if ($null -eq $Value) { return $Default }
    $raw = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }
    $out = 0.0
    if ([double]::TryParse($raw, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$out)) {
        return $out
    }
    return $Default
}

function Select-BestGhostRow([object[]]$Rows) {
    if ($null -eq $Rows -or $Rows.Count -le 0) { return $null }
    $scored = foreach ($row in $Rows) {
        [pscustomobject]@{
            row = $row
            ghost_visual_score = To-DoubleLoose $row.ghost_visual_score ([double]::PositiveInfinity)
            ghost_score = To-DoubleLoose $row.ghost_score ([double]::PositiveInfinity)
        }
    }
    return ($scored | Sort-Object ghost_visual_score, ghost_score | Select-Object -First 1).row
}

$candidatePath = Join-Path $StatusDir "candidate_result_latest.json"
$autoloopPath = Join-Path $StatusDir "overnight_ghost_autoloop_latest.json"
$watchPath = Join-Path $StatusDir "watch_ghost_outputs_latest.json"
$modalProgressPath = Join-Path $StatusDir "modal_run_progress_latest.json"
$ghostCsvPath = Join-Path $StatusDir "ghost_mvdepth_sweep_latest.csv"
$ftCsvPath = Join-Path $StatusDir "vggt_ft_sweep_latest.csv"

$candidate = Read-JsonMaybe -Path $candidatePath
$autoloop = Read-JsonMaybe -Path $autoloopPath
$watch = Read-JsonMaybe -Path $watchPath
$contract = Read-JsonMaybe -Path $ContractPath
$modalProgress = Read-JsonMaybe -Path $modalProgressPath
$ghostLast = Read-LastCsvRow -Path $ghostCsvPath
$ftLast = Read-LastCsvRow -Path $ftCsvPath
$updatedAt = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")

$runTimestamp = Pick-String @(
    $(if ($candidate) { [string]$candidate.run_timestamp } else { "" }),
    $(if ($contract) { ([string]$contract.generated_at -replace '[-:T+]', '' -replace '[^0-9]', '').Substring(0,14) } else { "" })
)
$runTag = Pick-String @(
    $(if ($candidate) { [string]$candidate.run_tag } else { "" }),
    "mv_0.001_mvmask_0_default"
)
$runFtCsvPath = $(if ([string]::IsNullOrWhiteSpace($runTimestamp) -or [string]::IsNullOrWhiteSpace($runTag)) { "" } else { Join-Path $StatusDir ("vggt_ft_sweep_ghost_mv_${runTag}_$runTimestamp.csv") })
$runGhostJsonPath = $(if ([string]::IsNullOrWhiteSpace($runTimestamp) -or [string]::IsNullOrWhiteSpace($runTag)) { "" } else { Join-Path $StatusDir ("ghost_score_${runTag}_$runTimestamp.json") })
$runGhostRowsCsvPath = $(if ([string]::IsNullOrWhiteSpace($runTimestamp) -or [string]::IsNullOrWhiteSpace($runTag)) { "" } else { Join-Path $StatusDir ("ghost_score_rows_${runTag}_$runTimestamp.csv") })
$runGhostSummaryCsvPath = $(if ([string]::IsNullOrWhiteSpace($runTimestamp) -or [string]::IsNullOrWhiteSpace($runTag)) { "" } else { Join-Path $StatusDir ("ghost_score_summary_${runTag}_$runTimestamp.csv") })
$runBaselineCompareCsvPath = $(if ([string]::IsNullOrWhiteSpace($runTimestamp) -or [string]::IsNullOrWhiteSpace($runTag)) { "" } else { Join-Path $StatusDir ("baseline_compare_ghost_mv_${runTag}_$runTimestamp.csv") })
$runFtRow = $(if ([string]::IsNullOrWhiteSpace($runFtCsvPath)) { $null } else { Read-LastCsvRow -Path $runFtCsvPath })
$runGhostJson = $(if ([string]::IsNullOrWhiteSpace($runGhostJsonPath)) { $null } else { Read-JsonMaybe -Path $runGhostJsonPath })
$runGhostRows = $(if ([string]::IsNullOrWhiteSpace($runGhostRowsCsvPath)) { @() } else { Read-CsvRows -Path $runGhostRowsCsvPath })
$runGhostBestRow = Select-BestGhostRow -Rows $runGhostRows
$runGhostSummary = $null
if ($runGhostJson -and $runGhostJson.summary -and @($runGhostJson.summary).Count -gt 0) {
    $runGhostSummary = @($runGhostJson.summary)[0]
}

if ($candidate -and ($runFtRow -or $runGhostSummary)) {
    $preserveCandidateWhenFtBlank = @(
        "mv_support_raw_mean",
        "mv_support_valid_ratio",
        "mv_support_fg_valid_ratio",
        "mv_support_bg_valid_ratio",
        "mv_support_pair_count_eff",
        "mv_support_conf_mean",
        "mv_support_nan_ratio",
        "depth_conf_delta_mean",
        "mv_support_fg_mean",
        "mv_support_bg_mean",
        "depth_conf_delta_fg_mean",
        "depth_conf_delta_bg_mean",
        "depth_conf_fg_preserved_active",
        "depth_conf_fg_preserve_px",
        "depth_conf_fg_exact_ratio",
        "depth_conf_fg_preserve_ratio",
        "depth_conf_fg_raw_mean",
        "depth_conf_fg_after_support_mean",
        "depth_conf_fg_final_mean",
        "mv_support_generation_region_mode",
        "mv_support_generation_fg_mask_source"
    )
    if ($runFtRow) {
        foreach ($prop in @($runFtRow.PSObject.Properties)) {
            if (
                ($preserveCandidateWhenFtBlank -contains $prop.Name) -and
                (Is-BlankOrNaN $prop.Value) -and
                (-not (Is-BlankOrNaN $candidate.$($prop.Name)))
            ) {
                continue
            }
            switch ($prop.Name) {
                "status" { Set-Prop -Obj $candidate -Name "ft_status" -Value $prop.Value }
                "reason" { Set-Prop -Obj $candidate -Name "ft_failure_reason" -Value $prop.Value }
                "pointmap_source_requested" { Set-Prop -Obj $candidate -Name "precompute_source_requested" -Value $prop.Value }
                "pointmap_source_resolved" { Set-Prop -Obj $candidate -Name "precompute_source_resolved" -Value $prop.Value }
                default { Set-Prop -Obj $candidate -Name $prop.Name -Value $prop.Value }
            }
        }
        Set-Prop -Obj $candidate -Name "stage_status" -Value "done"
        if ([string]::IsNullOrWhiteSpace([string]$candidate.precompute_source)) {
            Set-Prop -Obj $candidate -Name "precompute_source" -Value ([string]$runFtRow.precompute_source)
        }
    }
    if ($runGhostSummary) {
        Set-Prop -Obj $candidate -Name "ghost_score_mean" -Value $runGhostSummary.ghost_score_mean
        Set-Prop -Obj $candidate -Name "ghost_score_p95" -Value $runGhostSummary.ghost_score_p95
        Set-Prop -Obj $candidate -Name "ghost_soft_score_mean" -Value $runGhostSummary.ghost_soft_score_mean
        Set-Prop -Obj $candidate -Name "ghost_soft_score_p95" -Value $runGhostSummary.ghost_soft_score_p95
        Set-Prop -Obj $candidate -Name "ghost_visual_score" -Value $runGhostSummary.ghost_visual_score_mean
        Set-Prop -Obj $candidate -Name "pred_luma_mean" -Value $runGhostSummary.pred_luma_mean_mean
        Set-Prop -Obj $candidate -Name "pred_nonblack_ratio_thr008" -Value $runGhostSummary.pred_nonblack_ratio_thr008_mean
        Set-Prop -Obj $candidate -Name "pred_nonblack_ratio_thr015" -Value $runGhostSummary.pred_nonblack_ratio_thr015_mean
        Set-Prop -Obj $candidate -Name "width_ratio_mean" -Value $runGhostSummary.width_ratio_mean
        Set-Prop -Obj $candidate -Name "area_ratio_mean" -Value $runGhostSummary.area_ratio_mean
        Set-Prop -Obj $candidate -Name "fg_pred_luma_mean" -Value $runGhostSummary.fg_pred_luma_mean_mean
        Set-Prop -Obj $candidate -Name "fg_pred_nonblack_ratio" -Value $runGhostSummary.fg_pred_nonblack_ratio_mean
        Set-Prop -Obj $candidate -Name "fg_pred_contrast" -Value $runGhostSummary.fg_pred_contrast_mean
        Set-Prop -Obj $candidate -Name "fg_pred_tgt_l1" -Value $runGhostSummary.fg_pred_tgt_l1_mean
        Set-Prop -Obj $candidate -Name "ghost_rows_csv" -Value ("logs/modal_phase5/" + [System.IO.Path]::GetFileName($runGhostRowsCsvPath))
        Set-Prop -Obj $candidate -Name "ghost_summary_csv" -Value ("logs/modal_phase5/" + [System.IO.Path]::GetFileName($runGhostSummaryCsvPath))
        Set-Prop -Obj $candidate -Name "baseline_compare_csv" -Value ("logs/modal_phase5/" + [System.IO.Path]::GetFileName($runBaselineCompareCsvPath))
        Set-Prop -Obj $candidate -Name "exit_code" -Value 0
    }
    if ($runGhostBestRow) {
        Set-Prop -Obj $candidate -Name "best_visual_png" -Value ([string]$runGhostBestRow.path)
    }
    Set-Prop -Obj $candidate -Name "updated_at" -Value $updatedAt
    Write-JsonNoBom -Path $candidatePath -Obj $candidate
    if (-not [string]::IsNullOrWhiteSpace($activeCandidateResultJson) -and ($activeCandidateResultJson -ne $candidatePath)) {
        Write-JsonNoBom -Path $activeCandidateResultJson -Obj $candidate
    }

    $ghostLatestObj = [ordered]@{}
    foreach ($prop in @($candidate.PSObject.Properties)) {
        $ghostLatestObj[$prop.Name] = $prop.Value
    }
    Write-JsonNoBom -Path (Join-Path $StatusDir "ghost_mvdepth_sweep_latest.json") -Obj ([pscustomobject]$ghostLatestObj)
    ([pscustomobject]$ghostLatestObj) | Export-Csv -Path $ghostCsvPath -NoTypeInformation -Encoding UTF8
    $ghostMd = @()
    $ghostMd += "# Ghost Sweep Latest (single-probe)"
    $ghostMd += ""
    $ghostMd += "- run_timestamp: $runTimestamp"
    $ghostMd += "- probe_id: $ProbeId"
    $ghostMd += "- ghost_score_mean: $($candidate.ghost_score_mean)"
    $ghostMd += "- ghost_visual_score: $($candidate.ghost_visual_score)"
    $ghostMd += "- pred_luma_mean: $($candidate.pred_luma_mean)"
    $ghostMd += "- pred_nonblack_ratio_thr008: $($candidate.pred_nonblack_ratio_thr008)"
    $ghostMd += "- fg_pred_luma_mean: $($candidate.fg_pred_luma_mean)"
    $ghostMd += "- fg_pred_nonblack_ratio: $($candidate.fg_pred_nonblack_ratio)"
    $ghostMd += "- fg_pred_contrast: $($candidate.fg_pred_contrast)"
    $ghostMd += "- fg_pred_tgt_l1: $($candidate.fg_pred_tgt_l1)"
    $ghostMd += "- best_visual_png: $($candidate.best_visual_png)"
    Set-Content -Path (Join-Path $StatusDir "ghost_mvdepth_sweep_latest.md") -Value ($ghostMd -join "`n") -Encoding UTF8
    $ghostLast = Read-LastCsvRow -Path $ghostCsvPath
}

$activeCandidateResultJson = Pick-String @(
    $(if ($candidate) { [string]$candidate.candidate_result_json } else { "" }),
    $(if ($autoloop) { [string]$autoloop.active_candidate_result_json } else { "" }),
    $(if ($watch) { [string]$watch.active_candidate_result_json } else { "" })
)
$activeCandidateResultJson = "logs/modal_phase5/candidate_result_latest.json"

$currentStage = ""
if ($ProbeId -eq "T0_smoke" -and $autoloop -and $autoloop.PSObject.Properties["current_stage"]) {
    $currentStage = [string]$autoloop.current_stage
}
if ([string]::IsNullOrWhiteSpace($currentStage)) {
    $currentStage = $(if ([string]::IsNullOrWhiteSpace($ProbeId)) { "single_probe" } else { "single_probe_$ProbeId" })
}

$activeCandidateInvalidReason = Pick-String @(
    $(if ($candidate) { [string]$candidate.candidate_invalid_reason } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.candidate_invalid_reason } else { "" }),
    $(if ($ftLast) { [string]$ftLast.candidate_invalid_reason } else { "" })
)

$activePrecomputeSource = Pick-String @(
    $(if ($candidate) { [string]$candidate.precompute_source } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.precompute_source } else { "" }),
    $(if ($ftLast) { [string]$ftLast.precompute_source } else { "" }),
    $(if ($contract) { [string]$contract.pointmap_source } else { "" })
)
$activePrecomputeSourceRequested = Pick-String @(
    $(if ($candidate) { [string]$candidate.precompute_source_requested } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.precompute_source_requested } else { "" }),
    $(if ($ftLast) { [string]$ftLast.precompute_source_requested } else { "" }),
    $activePrecomputeSource
)
$activePrecomputeSourceResolved = Pick-String @(
    $(if ($candidate) { [string]$candidate.precompute_source_resolved } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.precompute_source_resolved } else { "" }),
    $(if ($ftLast) { [string]$ftLast.precompute_source_resolved } else { "" }),
    $activePrecomputeSource
)

$activePrecomputeFallbackUsed = Pick-Bool $(if ($candidate) { $candidate.precompute_fallback_used } else { $null }) $(if ($ghostLast) { $ghostLast.precompute_fallback_used } else { $false })
$activePrecomputeTimeoutHit = Pick-Bool $(if ($candidate) { $candidate.precompute_timeout_hit } else { $null }) $(if ($ghostLast) { $ghostLast.precompute_timeout_hit } else { $false })

$activePrecomputeMvSupportOn = Pick-String @(
    $(if ($candidate) { [string]$candidate.precompute_mv_support_on } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.precompute_mv_support_on } else { "" }),
    $(if ($ftLast) { [string]$ftLast.precompute_mv_support_on } else { "" }),
    $(if ($contract) { [string]$contract.precompute_mv_support_on } else { "" })
)
$activePrecomputeMvSupportRegionMode = Pick-String @(
    $(if ($candidate) { [string]$candidate.precompute_mv_support_region_mode } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.precompute_mv_support_region_mode } else { "" }),
    $(if ($ftLast) { [string]$ftLast.precompute_mv_support_region_mode } else { "" }),
    $(if ($contract) { [string]$contract.precompute_mv_support_region_mode } else { "" })
)
$activePrecomputeMvSupportFgMaskSource = Pick-String @(
    $(if ($candidate) { [string]$candidate.precompute_mv_support_fg_mask_source } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.precompute_mv_support_fg_mask_source } else { "" }),
    $(if ($ftLast) { [string]$ftLast.precompute_mv_support_fg_mask_source } else { "" }),
    $(if ($contract) { [string]$contract.precompute_mv_support_fg_mask_source } else { "" })
)
$activePrecomputeMvSupportFgErodePx = Pick-String @(
    $(if ($candidate) { [string]$candidate.precompute_mv_support_fg_erode_px } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.precompute_mv_support_fg_erode_px } else { "" }),
    $(if ($ftLast) { [string]$ftLast.precompute_mv_support_fg_erode_px } else { "" }),
    $(if ($contract) { [string]$contract.precompute_mv_support_fg_erode_px } else { "" })
)
$activePrecomputeMvSupportFgPreservePx = Pick-String @(
    $(if ($candidate) { [string]$candidate.precompute_mv_support_fg_preserve_px } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.precompute_mv_support_fg_preserve_px } else { "" }),
    $(if ($ftLast) { [string]$ftLast.precompute_mv_support_fg_preserve_px } else { "" }),
    $(if ($contract) { [string]$contract.precompute_mv_support_fg_preserve_px } else { "" })
)
$activePointTargetBlendByMvSupport = Pick-String @(
    $(if ($candidate) { [string]$candidate.point_target_blend_by_mv_support } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.point_target_blend_by_mv_support } else { "" }),
    $(if ($ftLast) { [string]$ftLast.point_target_blend_by_mv_support } else { "" }),
    $(if ($contract) { [string]$contract.point_target_blend_by_mv_support } else { "" })
)
$activePointTargetMode = Pick-String @(
    $(if ($candidate) { [string]$candidate.point_target_mode } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.point_target_mode } else { "" }),
    $(if ($ftLast) { [string]$ftLast.point_target_mode } else { "" }),
    $(if ($contract) { [string]$contract.point_target_mode } else { "" })
)
$activePointTargetBlendMvRegionMode = Pick-String @(
    $(if ($candidate) { [string]$candidate.point_target_blend_mv_region_mode } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.point_target_blend_mv_region_mode } else { "" }),
    $(if ($contract) { [string]$contract.point_target_blend_mv_region_mode } else { "" })
)
$activePointSupportMode = Pick-String @(
    $(if ($candidate) { [string]$candidate.point_support_mode } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.point_support_mode } else { "" }),
    $(if ($ftLast) { [string]$ftLast.point_support_mode } else { "" }),
    $(if ($contract) { [string]$contract.point_support_mode } else { "" })
)
$activePointMvDepthSupportMode = Pick-String @(
    $(if ($candidate) { [string]$candidate.point_mv_depth_support_mode } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.point_mv_depth_support_mode } else { "" }),
    $(if ($ftLast) { [string]$ftLast.point_mv_depth_support_mode } else { "" }),
    $(if ($contract) { [string]$contract.point_mv_depth_support_mode } else { "" })
)
$activePointMvMaskSupportMode = Pick-String @(
    $(if ($candidate) { [string]$candidate.point_mv_mask_support_mode } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.point_mv_mask_support_mode } else { "" }),
    $(if ($ftLast) { [string]$ftLast.point_mv_mask_support_mode } else { "" }),
    $(if ($contract) { [string]$contract.point_mv_mask_support_mode } else { "" })
)
$activePointMvDepthRegionMode = Pick-String @(
    $(if ($candidate) { [string]$candidate.point_mv_depth_region_mode } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.point_mv_depth_region_mode } else { "" }),
    $(if ($contract) { [string]$contract.point_mv_depth_region_mode } else { "" })
)
$activeUseFgMask = Pick-String @(
    $(if ($candidate) { [string]$candidate.use_fg_mask } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.use_fg_mask } else { "" }),
    $(if ($contract) { [string]$contract.use_fg_mask } else { "" })
)
$activeTf32 = Resolve-PrecisionString -Field "tf32" -RunFtRow $runFtRow -Contract $contract -Candidate $candidate -GhostLast $ghostLast -FtLast $ftLast
$activeAmp = Resolve-PrecisionString -Field "amp" -RunFtRow $runFtRow -Contract $contract -Candidate $candidate -GhostLast $ghostLast -FtLast $ftLast
$activeStrictDeterministic = Resolve-PrecisionString -Field "strict_deterministic" -RunFtRow $runFtRow -Contract $contract -Candidate $candidate -GhostLast $ghostLast -FtLast $ftLast
$activeRunnerTf32 = Resolve-PrecisionString -Field "runner_tf32" -RunFtRow $runFtRow -Contract $contract -Candidate $candidate -GhostLast $ghostLast -FtLast $ftLast -Fallback $activeTf32
$activeRunnerAmp = Resolve-PrecisionString -Field "runner_amp" -RunFtRow $runFtRow -Contract $contract -Candidate $candidate -GhostLast $ghostLast -FtLast $ftLast -Fallback $activeAmp
$activeRunnerStrictDeterministic = Resolve-PrecisionString -Field "runner_strict_deterministic" -RunFtRow $runFtRow -Contract $contract -Candidate $candidate -GhostLast $ghostLast -FtLast $ftLast -Fallback $activeStrictDeterministic
$activeTeacherTf32 = Resolve-PrecisionString -Field "teacher_tf32" -RunFtRow $runFtRow -Contract $contract -Candidate $candidate -GhostLast $ghostLast -FtLast $ftLast -Fallback $activeTf32
$activeTeacherAmp = Resolve-PrecisionString -Field "teacher_amp" -RunFtRow $runFtRow -Contract $contract -Candidate $candidate -GhostLast $ghostLast -FtLast $ftLast -Fallback $activeAmp
$activeTeacherDeterministic = Resolve-PrecisionString -Field "teacher_deterministic" -RunFtRow $runFtRow -Contract $contract -Candidate $candidate -GhostLast $ghostLast -FtLast $ftLast -Fallback $activeStrictDeterministic
$activeFgMaskSource = Pick-String @(
    $(if ($candidate) { [string]$candidate.fg_mask_source } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.fg_mask_source } else { "" }),
    $(if ($contract) { [string]$contract.fg_mask_source } else { "" })
)

$activeEvalNumSrcViewsDeclared = Pick-String @(
    $(if ($candidate) { [string]$candidate.eval_num_src_views_declared } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.eval_num_src_views_declared } else { "" }),
    $(if ($ftLast) { [string]$ftLast.eval_num_src_views_declared } else { "" }),
    $(if ($contract) { [string]$contract.eval_num_src_views } else { "" })
)
$activeEvalNumSrcViewsActual = Pick-String @(
    $(if ($candidate) { [string]$candidate.eval_num_src_views_actual } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.eval_num_src_views_actual } else { "" }),
    $(if ($ftLast) { [string]$ftLast.eval_num_src_views_actual } else { "" })
)
$activeEvalNumSrcViewsMismatch = Pick-Bool $(if ($candidate) { $candidate.eval_num_src_views_mismatch } else { $null }) $(if ($ghostLast) { $ghostLast.eval_num_src_views_mismatch } else { $false })

$activeQualityGuardBlocked = Pick-Bool $(if ($candidate) { $candidate.quality_guard_blocked } else { $null }) $(if ($ghostLast) { $ghostLast.quality_guard_blocked } else { $false })
$activeQualityGuardReason = Pick-String @(
    $(if ($candidate) { [string]$candidate.quality_guard_reason } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.quality_guard_reason } else { "" })
)
$activeVisualGuardBlocked = Pick-Bool $(if ($candidate) { $candidate.visual_guard_blocked } else { $null }) $(if ($ghostLast) { $ghostLast.visual_guard_blocked } else { $false })
$activeVisualGuardReason = Pick-String @(
    $(if ($candidate) { [string]$candidate.visual_guard_reason } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.visual_guard_reason } else { "" })
)

$bestVisualPng = Pick-String @(
    $(if ($ghostLast) { [string]$ghostLast.best_visual_png } else { "" }),
    $(if ($autoloop) { [string]$autoloop.best_visual_png } else { "" })
)
$compareStripPng = Pick-String @(
    $(if ($ghostLast) { [string]$ghostLast.stage_best_strip_png } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.compare_strip_png } else { "" })
)

$ghostScoreMean = $(if ($candidate) { $candidate.ghost_score_mean } else { $ghostLast.ghost_score_mean })
$ghostVisualScore = $(if ($candidate) { $candidate.ghost_visual_score } else { $ghostLast.ghost_visual_score })
$predLumaMean = $(if ($candidate) { $candidate.pred_luma_mean } else { $ghostLast.pred_luma_mean })
$predNonblack = $(if ($candidate) { $candidate.pred_nonblack_ratio_thr008 } else { $ghostLast.pred_nonblack_ratio_thr008 })
$fgPredLumaMean = $(if ($candidate) { $candidate.fg_pred_luma_mean } else { $ghostLast.fg_pred_luma_mean })
$fgPredNonblackRatio = $(if ($candidate) { $candidate.fg_pred_nonblack_ratio } else { $ghostLast.fg_pred_nonblack_ratio })
$fgPredContrast = $(if ($candidate) { $candidate.fg_pred_contrast } else { $ghostLast.fg_pred_contrast })
$fgPredTgtL1 = $(if ($candidate) { $candidate.fg_pred_tgt_l1 } else { $ghostLast.fg_pred_tgt_l1 })
$pointMvSupportMean = To-DoubleLoose $(if ($candidate) { $candidate.point_mv_support_mean } else { $ghostLast.point_mv_support_mean })
$pointMvSupportFgMean = To-DoubleLoose $(if ($candidate) { $candidate.point_mv_support_fg_mean } else { $ghostLast.point_mv_support_fg_mean })
$pointMvSupportBgMean = To-DoubleLoose $(if ($candidate) { $candidate.point_mv_support_bg_mean } else { $ghostLast.point_mv_support_bg_mean })
$mvSupportRawMean = To-DoubleLoose $(if ($candidate) { $candidate.mv_support_raw_mean } else { $ghostLast.mv_support_raw_mean })
$mvSupportValidRatio = To-DoubleLoose $(if ($candidate) { $candidate.mv_support_valid_ratio } else { $ghostLast.mv_support_valid_ratio })
$mvSupportFgValidRatio = To-DoubleLoose $(if ($candidate) { $candidate.mv_support_fg_valid_ratio } else { $ghostLast.mv_support_fg_valid_ratio })
$mvSupportBgValidRatio = To-DoubleLoose $(if ($candidate) { $candidate.mv_support_bg_valid_ratio } else { $ghostLast.mv_support_bg_valid_ratio })
$mvSupportPairCountEff = To-DoubleLoose $(if ($candidate) { $candidate.mv_support_pair_count_eff } else { $ghostLast.mv_support_pair_count_eff })
$mvSupportConfMean = To-DoubleLoose $(if ($candidate) { $candidate.mv_support_conf_mean } else { $ghostLast.mv_support_conf_mean })
$mvSupportNanRatio = To-DoubleLoose $(if ($candidate) { $candidate.mv_support_nan_ratio } else { $ghostLast.mv_support_nan_ratio })
$depthConfDeltaMean = To-DoubleLoose $(if ($candidate) { $candidate.depth_conf_delta_mean } else { $ghostLast.depth_conf_delta_mean })
$mvSupportFgMean = To-DoubleLoose $(if ($candidate) { $candidate.mv_support_fg_mean } else { $ghostLast.mv_support_fg_mean })
$mvSupportBgMean = To-DoubleLoose $(if ($candidate) { $candidate.mv_support_bg_mean } else { $ghostLast.mv_support_bg_mean })
$depthConfDeltaFgMean = To-DoubleLoose $(if ($candidate) { $candidate.depth_conf_delta_fg_mean } else { $ghostLast.depth_conf_delta_fg_mean })
$depthConfDeltaBgMean = To-DoubleLoose $(if ($candidate) { $candidate.depth_conf_delta_bg_mean } else { $ghostLast.depth_conf_delta_bg_mean })
$depthConfFgPreservedActive = To-DoubleLoose $(if ($candidate) { $candidate.depth_conf_fg_preserved_active } else { $ghostLast.depth_conf_fg_preserved_active })
$depthConfFgPreservePx = To-DoubleLoose $(if ($candidate) { $candidate.depth_conf_fg_preserve_px } else { $ghostLast.depth_conf_fg_preserve_px })
$depthConfFgExactRatio = To-DoubleLoose $(if ($candidate) { $candidate.depth_conf_fg_exact_ratio } else { $ghostLast.depth_conf_fg_exact_ratio })
$depthConfFgPreserveRatio = To-DoubleLoose $(if ($candidate) { $candidate.depth_conf_fg_preserve_ratio } else { $ghostLast.depth_conf_fg_preserve_ratio })
$depthConfFgRawMean = To-DoubleLoose $(if ($candidate) { $candidate.depth_conf_fg_raw_mean } else { $ghostLast.depth_conf_fg_raw_mean })
$depthConfFgAfterSupportMean = To-DoubleLoose $(if ($candidate) { $candidate.depth_conf_fg_after_support_mean } else { $ghostLast.depth_conf_fg_after_support_mean })
$depthConfFgFinalMean = To-DoubleLoose $(if ($candidate) { $candidate.depth_conf_fg_final_mean } else { $ghostLast.depth_conf_fg_final_mean })
$mvSupportGenerationRegionMode = Pick-String @(
    $(if ($candidate) { [string]$candidate.mv_support_generation_region_mode } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.mv_support_generation_region_mode } else { "" }),
    $activePrecomputeMvSupportRegionMode
)
$mvSupportGenerationFgMaskSource = Pick-String @(
    $(if ($candidate) { [string]$candidate.mv_support_generation_fg_mask_source } else { "" }),
    $(if ($ghostLast) { [string]$ghostLast.mv_support_generation_fg_mask_source } else { "" }),
    $activePrecomputeMvSupportFgMaskSource
)
$supportGenerationActive = $(if (($activePrecomputeMvSupportOn).Trim().ToLowerInvariant() -eq 'on') { 1.0 } else { 0.0 })
$pointSupportPathActive = $(if (($activePointSupportMode).Trim().ToLowerInvariant() -ne 'off' -and -not [string]::IsNullOrWhiteSpace($activePointSupportMode)) { 1.0 } else { 0.0 })
$pointMvDepthSupportPathActive = $(if (($activePointMvDepthSupportMode).Trim().ToLowerInvariant() -ne 'off' -and -not [string]::IsNullOrWhiteSpace($activePointMvDepthSupportMode)) { 1.0 } else { 0.0 })
$pointMvMaskSupportPathActive = $(if (($activePointMvMaskSupportMode).Trim().ToLowerInvariant() -ne 'off' -and -not [string]::IsNullOrWhiteSpace($activePointMvMaskSupportMode)) { 1.0 } else { 0.0 })
$pointTargetBlendMvSupportActive = $(if (($activePointTargetBlendByMvSupport).Trim().ToLowerInvariant() -eq 'on') { 1.0 } else { 0.0 })
$pointSupportEffMean = To-DoubleLoose $(if ($candidate) { $candidate.point_support_eff_mean } else { $ghostLast.point_support_eff_mean })
$pointSupportEffFgMean = To-DoubleLoose $(if ($candidate) { $candidate.point_support_eff_fg_mean } else { $ghostLast.point_support_eff_fg_mean })
$pointSupportEffBgMean = To-DoubleLoose $(if ($candidate) { $candidate.point_support_eff_bg_mean } else { $ghostLast.point_support_eff_bg_mean })
$pointMvDepthSupportEffMean = To-DoubleLoose $(if ($candidate) { $candidate.point_mv_depth_support_eff_mean } else { $ghostLast.point_mv_depth_support_eff_mean })
$pointMvDepthSupportEffFgMean = To-DoubleLoose $(if ($candidate) { $candidate.point_mv_depth_support_eff_fg_mean } else { $ghostLast.point_mv_depth_support_eff_fg_mean })
$pointMvDepthSupportEffBgMean = To-DoubleLoose $(if ($candidate) { $candidate.point_mv_depth_support_eff_bg_mean } else { $ghostLast.point_mv_depth_support_eff_bg_mean })
$pointMvMaskSupportEffMean = To-DoubleLoose $(if ($candidate) { $candidate.point_mv_mask_support_eff_mean } else { $ghostLast.point_mv_mask_support_eff_mean })
$pointMvMaskSupportEffFgMean = To-DoubleLoose $(if ($candidate) { $candidate.point_mv_mask_support_eff_fg_mean } else { $ghostLast.point_mv_mask_support_eff_fg_mean })
$pointMvMaskSupportEffBgMean = To-DoubleLoose $(if ($candidate) { $candidate.point_mv_mask_support_eff_bg_mean } else { $ghostLast.point_mv_mask_support_eff_bg_mean })

if ($candidate) {
    Set-Prop -Obj $candidate -Name "support_generation_active" -Value $supportGenerationActive
    Set-Prop -Obj $candidate -Name "point_support_path_active" -Value $pointSupportPathActive
    Set-Prop -Obj $candidate -Name "point_mv_depth_support_path_active" -Value $pointMvDepthSupportPathActive
    Set-Prop -Obj $candidate -Name "point_mv_mask_support_path_active" -Value $pointMvMaskSupportPathActive
    Set-Prop -Obj $candidate -Name "point_target_blend_mv_support_active" -Value $pointTargetBlendMvSupportActive
    Set-Prop -Obj $candidate -Name "mv_support_raw_mean" -Value $mvSupportRawMean
    Set-Prop -Obj $candidate -Name "mv_support_valid_ratio" -Value $mvSupportValidRatio
    Set-Prop -Obj $candidate -Name "mv_support_fg_valid_ratio" -Value $mvSupportFgValidRatio
    Set-Prop -Obj $candidate -Name "mv_support_bg_valid_ratio" -Value $mvSupportBgValidRatio
    Set-Prop -Obj $candidate -Name "mv_support_pair_count_eff" -Value $mvSupportPairCountEff
    Set-Prop -Obj $candidate -Name "mv_support_conf_mean" -Value $mvSupportConfMean
    Set-Prop -Obj $candidate -Name "mv_support_nan_ratio" -Value $mvSupportNanRatio
    Set-Prop -Obj $candidate -Name "depth_conf_delta_mean" -Value $depthConfDeltaMean
    Set-Prop -Obj $candidate -Name "mv_support_fg_mean" -Value $mvSupportFgMean
    Set-Prop -Obj $candidate -Name "mv_support_bg_mean" -Value $mvSupportBgMean
    Set-Prop -Obj $candidate -Name "depth_conf_delta_fg_mean" -Value $depthConfDeltaFgMean
    Set-Prop -Obj $candidate -Name "depth_conf_delta_bg_mean" -Value $depthConfDeltaBgMean
    Set-Prop -Obj $candidate -Name "depth_conf_fg_preserved_active" -Value $depthConfFgPreservedActive
    Set-Prop -Obj $candidate -Name "depth_conf_fg_preserve_px" -Value $depthConfFgPreservePx
    Set-Prop -Obj $candidate -Name "depth_conf_fg_exact_ratio" -Value $depthConfFgExactRatio
    Set-Prop -Obj $candidate -Name "depth_conf_fg_preserve_ratio" -Value $depthConfFgPreserveRatio
    Set-Prop -Obj $candidate -Name "depth_conf_fg_raw_mean" -Value $depthConfFgRawMean
    Set-Prop -Obj $candidate -Name "depth_conf_fg_after_support_mean" -Value $depthConfFgAfterSupportMean
    Set-Prop -Obj $candidate -Name "depth_conf_fg_final_mean" -Value $depthConfFgFinalMean
    Set-Prop -Obj $candidate -Name "mv_support_generation_region_mode" -Value $mvSupportGenerationRegionMode
    Set-Prop -Obj $candidate -Name "mv_support_generation_fg_mask_source" -Value $mvSupportGenerationFgMaskSource
    Write-JsonNoBom -Path $candidatePath -Obj $candidate
    if (-not [string]::IsNullOrWhiteSpace($activeCandidateResultJson) -and ($activeCandidateResultJson -ne $candidatePath)) {
        Write-JsonNoBom -Path $activeCandidateResultJson -Obj $candidate
    }
}

$modalRunState = $(if ($modalProgress) { [string]$modalProgress.state } else { "" })
$modalRunScriptPath = $(if ($modalProgress) { [string]$modalProgress.script_path } else { "" })
$modalRunExitCode = $(if ($modalProgress -and $modalProgress.PSObject.Properties["exit_code"]) { $modalProgress.exit_code } else { $null })
$isTerminalState = $State -in @("done", "error", "stale")
$pausedValue = $isTerminalState
$pauseReasonValue = $(if ($isTerminalState) { "single_probe_complete_no_live_chain" } else { "" })

if ($null -eq $autoloop) {
    $autoloop = [pscustomobject]@{}
}
Set-Prop -Obj $autoloop -Name "updated_at" -Value $updatedAt
Set-Prop -Obj $autoloop -Name "source" -Value "single_probe_sync"
Set-Prop -Obj $autoloop -Name "probe_id" -Value $ProbeId
Set-Prop -Obj $autoloop -Name "probe_state" -Value $State
if ([string]::IsNullOrWhiteSpace([string]$autoloop.current_stage) -or ($ProbeId -ne "T0_smoke")) {
    Set-Prop -Obj $autoloop -Name "current_stage" -Value $currentStage
}
Set-Prop -Obj $autoloop -Name "active_candidate_result_json" -Value $activeCandidateResultJson
Set-Prop -Obj $autoloop -Name "active_candidate_invalid_reason" -Value $activeCandidateInvalidReason
Set-Prop -Obj $autoloop -Name "active_precompute_source" -Value $activePrecomputeSource
Set-Prop -Obj $autoloop -Name "active_precompute_source_requested" -Value $activePrecomputeSourceRequested
Set-Prop -Obj $autoloop -Name "active_precompute_source_resolved" -Value $activePrecomputeSourceResolved
Set-Prop -Obj $autoloop -Name "active_precompute_fallback_used" -Value $activePrecomputeFallbackUsed
Set-Prop -Obj $autoloop -Name "active_precompute_timeout_hit" -Value $activePrecomputeTimeoutHit
Set-Prop -Obj $autoloop -Name "active_precompute_mv_support_on" -Value $activePrecomputeMvSupportOn
Set-Prop -Obj $autoloop -Name "active_precompute_mv_support_region_mode" -Value $activePrecomputeMvSupportRegionMode
Set-Prop -Obj $autoloop -Name "active_precompute_mv_support_fg_mask_source" -Value $activePrecomputeMvSupportFgMaskSource
Set-Prop -Obj $autoloop -Name "active_precompute_mv_support_fg_erode_px" -Value $activePrecomputeMvSupportFgErodePx
Set-Prop -Obj $autoloop -Name "active_precompute_mv_support_fg_preserve_px" -Value $activePrecomputeMvSupportFgPreservePx
Set-Prop -Obj $autoloop -Name "active_point_target_mode" -Value $activePointTargetMode
Set-Prop -Obj $autoloop -Name "active_point_target_blend_by_mv_support" -Value $activePointTargetBlendByMvSupport
Set-Prop -Obj $autoloop -Name "active_point_target_blend_mv_region_mode" -Value $activePointTargetBlendMvRegionMode
Set-Prop -Obj $autoloop -Name "active_point_support_mode" -Value $activePointSupportMode
Set-Prop -Obj $autoloop -Name "active_point_mv_depth_support_mode" -Value $activePointMvDepthSupportMode
Set-Prop -Obj $autoloop -Name "active_point_mv_mask_support_mode" -Value $activePointMvMaskSupportMode
Set-Prop -Obj $autoloop -Name "active_point_mv_depth_region_mode" -Value $activePointMvDepthRegionMode
Set-Prop -Obj $autoloop -Name "active_use_fg_mask" -Value $activeUseFgMask
Set-Prop -Obj $autoloop -Name "active_fg_mask_source" -Value $activeFgMaskSource
Set-Prop -Obj $autoloop -Name "active_tf32" -Value $activeTf32
Set-Prop -Obj $autoloop -Name "active_amp" -Value $activeAmp
Set-Prop -Obj $autoloop -Name "active_strict_deterministic" -Value $activeStrictDeterministic
Set-Prop -Obj $autoloop -Name "active_runner_tf32" -Value $activeRunnerTf32
Set-Prop -Obj $autoloop -Name "active_runner_amp" -Value $activeRunnerAmp
Set-Prop -Obj $autoloop -Name "active_runner_strict_deterministic" -Value $activeRunnerStrictDeterministic
Set-Prop -Obj $autoloop -Name "active_teacher_tf32" -Value $activeTeacherTf32
Set-Prop -Obj $autoloop -Name "active_teacher_amp" -Value $activeTeacherAmp
Set-Prop -Obj $autoloop -Name "active_teacher_deterministic" -Value $activeTeacherDeterministic
Set-Prop -Obj $autoloop -Name "active_eval_num_src_views_declared" -Value $activeEvalNumSrcViewsDeclared
Set-Prop -Obj $autoloop -Name "active_eval_num_src_views_actual" -Value $activeEvalNumSrcViewsActual
Set-Prop -Obj $autoloop -Name "active_eval_num_src_views_mismatch" -Value $activeEvalNumSrcViewsMismatch
Set-Prop -Obj $autoloop -Name "active_quality_guard_blocked" -Value $activeQualityGuardBlocked
Set-Prop -Obj $autoloop -Name "active_quality_guard_reason" -Value $activeQualityGuardReason
Set-Prop -Obj $autoloop -Name "active_visual_guard_blocked" -Value $activeVisualGuardBlocked
Set-Prop -Obj $autoloop -Name "active_visual_guard_reason" -Value $activeVisualGuardReason
Set-Prop -Obj $autoloop -Name "ghost_score_mean" -Value $ghostScoreMean
Set-Prop -Obj $autoloop -Name "ghost_visual_score" -Value $ghostVisualScore
Set-Prop -Obj $autoloop -Name "pred_luma_mean" -Value $predLumaMean
Set-Prop -Obj $autoloop -Name "pred_nonblack_ratio_thr008" -Value $predNonblack
Set-Prop -Obj $autoloop -Name "fg_pred_luma_mean" -Value $fgPredLumaMean
Set-Prop -Obj $autoloop -Name "fg_pred_nonblack_ratio" -Value $fgPredNonblackRatio
Set-Prop -Obj $autoloop -Name "fg_pred_contrast" -Value $fgPredContrast
Set-Prop -Obj $autoloop -Name "fg_pred_tgt_l1" -Value $fgPredTgtL1
Set-Prop -Obj $autoloop -Name "point_mv_support_mean" -Value $pointMvSupportMean
Set-Prop -Obj $autoloop -Name "point_mv_support_fg_mean" -Value $pointMvSupportFgMean
Set-Prop -Obj $autoloop -Name "point_mv_support_bg_mean" -Value $pointMvSupportBgMean
Set-Prop -Obj $autoloop -Name "mv_support_raw_mean" -Value $mvSupportRawMean
Set-Prop -Obj $autoloop -Name "mv_support_valid_ratio" -Value $mvSupportValidRatio
Set-Prop -Obj $autoloop -Name "mv_support_fg_valid_ratio" -Value $mvSupportFgValidRatio
Set-Prop -Obj $autoloop -Name "mv_support_bg_valid_ratio" -Value $mvSupportBgValidRatio
Set-Prop -Obj $autoloop -Name "mv_support_pair_count_eff" -Value $mvSupportPairCountEff
Set-Prop -Obj $autoloop -Name "mv_support_conf_mean" -Value $mvSupportConfMean
Set-Prop -Obj $autoloop -Name "mv_support_nan_ratio" -Value $mvSupportNanRatio
Set-Prop -Obj $autoloop -Name "depth_conf_delta_mean" -Value $depthConfDeltaMean
Set-Prop -Obj $autoloop -Name "mv_support_fg_mean" -Value $mvSupportFgMean
Set-Prop -Obj $autoloop -Name "mv_support_bg_mean" -Value $mvSupportBgMean
Set-Prop -Obj $autoloop -Name "depth_conf_delta_fg_mean" -Value $depthConfDeltaFgMean
Set-Prop -Obj $autoloop -Name "depth_conf_delta_bg_mean" -Value $depthConfDeltaBgMean
Set-Prop -Obj $autoloop -Name "depth_conf_fg_preserved_active" -Value $depthConfFgPreservedActive
Set-Prop -Obj $autoloop -Name "depth_conf_fg_preserve_px" -Value $depthConfFgPreservePx
Set-Prop -Obj $autoloop -Name "depth_conf_fg_exact_ratio" -Value $depthConfFgExactRatio
Set-Prop -Obj $autoloop -Name "depth_conf_fg_preserve_ratio" -Value $depthConfFgPreserveRatio
Set-Prop -Obj $autoloop -Name "depth_conf_fg_raw_mean" -Value $depthConfFgRawMean
Set-Prop -Obj $autoloop -Name "depth_conf_fg_after_support_mean" -Value $depthConfFgAfterSupportMean
Set-Prop -Obj $autoloop -Name "depth_conf_fg_final_mean" -Value $depthConfFgFinalMean
Set-Prop -Obj $autoloop -Name "mv_support_generation_region_mode" -Value $mvSupportGenerationRegionMode
Set-Prop -Obj $autoloop -Name "mv_support_generation_fg_mask_source" -Value $mvSupportGenerationFgMaskSource
Set-Prop -Obj $autoloop -Name "support_generation_active" -Value $supportGenerationActive
Set-Prop -Obj $autoloop -Name "point_support_path_active" -Value $pointSupportPathActive
Set-Prop -Obj $autoloop -Name "point_mv_depth_support_path_active" -Value $pointMvDepthSupportPathActive
Set-Prop -Obj $autoloop -Name "point_mv_mask_support_path_active" -Value $pointMvMaskSupportPathActive
Set-Prop -Obj $autoloop -Name "point_target_blend_mv_support_active" -Value $pointTargetBlendMvSupportActive
Set-Prop -Obj $autoloop -Name "point_support_eff_mean" -Value $pointSupportEffMean
Set-Prop -Obj $autoloop -Name "point_support_eff_fg_mean" -Value $pointSupportEffFgMean
Set-Prop -Obj $autoloop -Name "point_support_eff_bg_mean" -Value $pointSupportEffBgMean
Set-Prop -Obj $autoloop -Name "point_mv_depth_support_eff_mean" -Value $pointMvDepthSupportEffMean
Set-Prop -Obj $autoloop -Name "point_mv_depth_support_eff_fg_mean" -Value $pointMvDepthSupportEffFgMean
Set-Prop -Obj $autoloop -Name "point_mv_depth_support_eff_bg_mean" -Value $pointMvDepthSupportEffBgMean
Set-Prop -Obj $autoloop -Name "point_mv_mask_support_eff_mean" -Value $pointMvMaskSupportEffMean
Set-Prop -Obj $autoloop -Name "point_mv_mask_support_eff_fg_mean" -Value $pointMvMaskSupportEffFgMean
Set-Prop -Obj $autoloop -Name "point_mv_mask_support_eff_bg_mean" -Value $pointMvMaskSupportEffBgMean
Set-Prop -Obj $autoloop -Name "best_visual_png" -Value $bestVisualPng
Set-Prop -Obj $autoloop -Name "stage_best_strip_png" -Value $compareStripPng
Set-Prop -Obj $autoloop -Name "paused" -Value $pausedValue
Set-Prop -Obj $autoloop -Name "pause_reason" -Value $pauseReasonValue
Set-Prop -Obj $autoloop -Name "active_modal_app_count" -Value 0
Set-Prop -Obj $autoloop -Name "active_local_process_count" -Value 0
Set-Prop -Obj $autoloop -Name "note" -Value ("single_probe_sync_" + $State)
Write-JsonNoBom -Path $autoloopPath -Obj $autoloop

$autoloopMd = @()
$autoloopMd += "# Single Probe Status"
$autoloopMd += ""
$autoloopMd += "- updated: $updatedAt"
$autoloopMd += "- probe_id: $ProbeId"
$autoloopMd += "- state: $State"
$autoloopMd += "- current_stage: $($autoloop.current_stage)"
$autoloopMd += "- active_candidate_result_json: $activeCandidateResultJson"
$autoloopMd += "- active_candidate_invalid_reason: $activeCandidateInvalidReason"
$autoloopMd += "- active_precompute_source: $activePrecomputeSource"
$autoloopMd += "- active_precompute_mv_support_on: $activePrecomputeMvSupportOn"
$autoloopMd += "- active_precompute_mv_support_region_mode: $activePrecomputeMvSupportRegionMode"
$autoloopMd += "- active_precompute_mv_support_fg_mask_source: $activePrecomputeMvSupportFgMaskSource"
$autoloopMd += "- active_precompute_mv_support_fg_erode_px: $activePrecomputeMvSupportFgErodePx"
$autoloopMd += "- active_precompute_mv_support_fg_preserve_px: $activePrecomputeMvSupportFgPreservePx"
$autoloopMd += "- active_point_target_mode: $activePointTargetMode"
$autoloopMd += "- active_point_target_blend_by_mv_support: $activePointTargetBlendByMvSupport"
$autoloopMd += "- active_point_target_blend_mv_region_mode: $activePointTargetBlendMvRegionMode"
$autoloopMd += "- active_point_support_mode: $activePointSupportMode"
$autoloopMd += "- active_point_mv_depth_support_mode: $activePointMvDepthSupportMode"
$autoloopMd += "- active_point_mv_mask_support_mode: $activePointMvMaskSupportMode"
$autoloopMd += "- active_point_mv_depth_region_mode: $activePointMvDepthRegionMode"
$autoloopMd += "- active_tf32: $activeTf32"
$autoloopMd += "- active_amp: $activeAmp"
$autoloopMd += "- active_strict_deterministic: $activeStrictDeterministic"
$autoloopMd += "- active_runner_tf32: $activeRunnerTf32"
$autoloopMd += "- active_runner_amp: $activeRunnerAmp"
$autoloopMd += "- active_runner_strict_deterministic: $activeRunnerStrictDeterministic"
$autoloopMd += "- active_teacher_tf32: $activeTeacherTf32"
$autoloopMd += "- active_teacher_amp: $activeTeacherAmp"
$autoloopMd += "- active_teacher_deterministic: $activeTeacherDeterministic"
$autoloopMd += "- ghost_score_mean: $ghostScoreMean"
$autoloopMd += "- ghost_visual_score: $ghostVisualScore"
$autoloopMd += "- pred_luma_mean: $predLumaMean"
$autoloopMd += "- pred_nonblack_ratio_thr008: $predNonblack"
$autoloopMd += "- fg_pred_luma_mean: $fgPredLumaMean"
$autoloopMd += "- fg_pred_nonblack_ratio: $fgPredNonblackRatio"
$autoloopMd += "- fg_pred_contrast: $fgPredContrast"
$autoloopMd += "- fg_pred_tgt_l1: $fgPredTgtL1"
$autoloopMd += "- point_mv_support_mean: $pointMvSupportMean"
$autoloopMd += "- point_mv_support_fg_mean: $pointMvSupportFgMean"
$autoloopMd += "- point_mv_support_bg_mean: $pointMvSupportBgMean"
$autoloopMd += "- mv_support_raw_mean: $mvSupportRawMean"
$autoloopMd += "- mv_support_valid_ratio: $mvSupportValidRatio"
$autoloopMd += "- mv_support_fg_valid_ratio: $mvSupportFgValidRatio"
$autoloopMd += "- mv_support_bg_valid_ratio: $mvSupportBgValidRatio"
$autoloopMd += "- mv_support_pair_count_eff: $mvSupportPairCountEff"
$autoloopMd += "- mv_support_conf_mean: $mvSupportConfMean"
$autoloopMd += "- mv_support_nan_ratio: $mvSupportNanRatio"
$autoloopMd += "- depth_conf_delta_mean: $depthConfDeltaMean"
$autoloopMd += "- mv_support_fg_mean: $mvSupportFgMean"
$autoloopMd += "- mv_support_bg_mean: $mvSupportBgMean"
$autoloopMd += "- depth_conf_delta_fg_mean: $depthConfDeltaFgMean"
$autoloopMd += "- depth_conf_delta_bg_mean: $depthConfDeltaBgMean"
$autoloopMd += "- depth_conf_fg_preserved_active: $depthConfFgPreservedActive"
$autoloopMd += "- depth_conf_fg_preserve_px: $depthConfFgPreservePx"
$autoloopMd += "- depth_conf_fg_exact_ratio: $depthConfFgExactRatio"
$autoloopMd += "- depth_conf_fg_preserve_ratio: $depthConfFgPreserveRatio"
$autoloopMd += "- depth_conf_fg_raw_mean: $depthConfFgRawMean"
$autoloopMd += "- depth_conf_fg_after_support_mean: $depthConfFgAfterSupportMean"
$autoloopMd += "- depth_conf_fg_final_mean: $depthConfFgFinalMean"
$autoloopMd += "- mv_support_generation_region_mode: $mvSupportGenerationRegionMode"
$autoloopMd += "- mv_support_generation_fg_mask_source: $mvSupportGenerationFgMaskSource"
$autoloopMd += "- support_generation_active: $supportGenerationActive"
$autoloopMd += "- point_support_path_active: $pointSupportPathActive"
$autoloopMd += "- point_mv_depth_support_path_active: $pointMvDepthSupportPathActive"
$autoloopMd += "- point_mv_mask_support_path_active: $pointMvMaskSupportPathActive"
$autoloopMd += "- point_target_blend_mv_support_active: $pointTargetBlendMvSupportActive"
$autoloopMd += "- point_support_eff_mean: $pointSupportEffMean"
$autoloopMd += "- point_support_eff_fg_mean: $pointSupportEffFgMean"
$autoloopMd += "- point_support_eff_bg_mean: $pointSupportEffBgMean"
$autoloopMd += "- point_mv_depth_support_eff_mean: $pointMvDepthSupportEffMean"
$autoloopMd += "- point_mv_depth_support_eff_fg_mean: $pointMvDepthSupportEffFgMean"
$autoloopMd += "- point_mv_depth_support_eff_bg_mean: $pointMvDepthSupportEffBgMean"
$autoloopMd += "- point_mv_mask_support_eff_mean: $pointMvMaskSupportEffMean"
$autoloopMd += "- point_mv_mask_support_eff_fg_mean: $pointMvMaskSupportEffFgMean"
$autoloopMd += "- point_mv_mask_support_eff_bg_mean: $pointMvMaskSupportEffBgMean"
$autoloopMd += "- paused: $pausedValue"
$autoloopMd += "- pause_reason: $pauseReasonValue"
$autoloopMd += "- note: single_probe_sync_$State"
Set-Content -Path (Join-Path $StatusDir "overnight_ghost_autoloop_latest.md") -Value ($autoloopMd -join "`n") -Encoding UTF8

$watchObj = [ordered]@{
    updated_at = $updatedAt
    source = "single_probe_sync"
    probe_id = $ProbeId
    state = $State
    current_stage = $currentStage
    active_candidate_result_json = $activeCandidateResultJson
    active_candidate_invalid_reason = $activeCandidateInvalidReason
    active_precompute_source = $activePrecomputeSource
    active_precompute_source_requested = $activePrecomputeSourceRequested
    active_precompute_source_resolved = $activePrecomputeSourceResolved
    active_precompute_fallback_used = $activePrecomputeFallbackUsed
    active_precompute_timeout_hit = $activePrecomputeTimeoutHit
    active_precompute_mv_support_on = $activePrecomputeMvSupportOn
    active_precompute_mv_support_region_mode = $activePrecomputeMvSupportRegionMode
    active_precompute_mv_support_fg_mask_source = $activePrecomputeMvSupportFgMaskSource
    active_precompute_mv_support_fg_erode_px = $activePrecomputeMvSupportFgErodePx
    active_precompute_mv_support_fg_preserve_px = $activePrecomputeMvSupportFgPreservePx
    active_point_target_mode = $activePointTargetMode
    active_point_target_blend_by_mv_support = $activePointTargetBlendByMvSupport
    active_point_target_blend_mv_region_mode = $activePointTargetBlendMvRegionMode
    active_point_support_mode = $activePointSupportMode
    active_point_mv_depth_support_mode = $activePointMvDepthSupportMode
    active_point_mv_mask_support_mode = $activePointMvMaskSupportMode
    active_point_mv_depth_region_mode = $activePointMvDepthRegionMode
    active_use_fg_mask = $activeUseFgMask
    active_fg_mask_source = $activeFgMaskSource
    active_tf32 = $activeTf32
    active_amp = $activeAmp
    active_strict_deterministic = $activeStrictDeterministic
    active_runner_tf32 = $activeRunnerTf32
    active_runner_amp = $activeRunnerAmp
    active_runner_strict_deterministic = $activeRunnerStrictDeterministic
    active_teacher_tf32 = $activeTeacherTf32
    active_teacher_amp = $activeTeacherAmp
    active_teacher_deterministic = $activeTeacherDeterministic
    active_eval_num_src_views_declared = $activeEvalNumSrcViewsDeclared
    active_eval_num_src_views_actual = $activeEvalNumSrcViewsActual
    active_eval_num_src_views_mismatch = $activeEvalNumSrcViewsMismatch
    active_quality_guard_blocked = $activeQualityGuardBlocked
    active_quality_guard_reason = $activeQualityGuardReason
    active_visual_guard_blocked = $activeVisualGuardBlocked
    active_visual_guard_reason = $activeVisualGuardReason
    ghost_score_mean = $ghostScoreMean
    ghost_visual_score = $ghostVisualScore
    pred_luma_mean = $predLumaMean
    pred_nonblack_ratio_thr008 = $predNonblack
    fg_pred_luma_mean = $fgPredLumaMean
    fg_pred_nonblack_ratio = $fgPredNonblackRatio
    fg_pred_contrast = $fgPredContrast
    fg_pred_tgt_l1 = $fgPredTgtL1
    point_mv_support_mean = $pointMvSupportMean
    point_mv_support_fg_mean = $pointMvSupportFgMean
    point_mv_support_bg_mean = $pointMvSupportBgMean
    mv_support_raw_mean = $mvSupportRawMean
    mv_support_valid_ratio = $mvSupportValidRatio
    mv_support_fg_valid_ratio = $mvSupportFgValidRatio
    mv_support_bg_valid_ratio = $mvSupportBgValidRatio
    mv_support_pair_count_eff = $mvSupportPairCountEff
    mv_support_conf_mean = $mvSupportConfMean
    mv_support_nan_ratio = $mvSupportNanRatio
    depth_conf_delta_mean = $depthConfDeltaMean
    mv_support_fg_mean = $mvSupportFgMean
    mv_support_bg_mean = $mvSupportBgMean
    depth_conf_delta_fg_mean = $depthConfDeltaFgMean
    depth_conf_delta_bg_mean = $depthConfDeltaBgMean
    depth_conf_fg_preserved_active = $depthConfFgPreservedActive
    depth_conf_fg_preserve_px = $depthConfFgPreservePx
    depth_conf_fg_exact_ratio = $depthConfFgExactRatio
    depth_conf_fg_preserve_ratio = $depthConfFgPreserveRatio
    depth_conf_fg_raw_mean = $depthConfFgRawMean
    depth_conf_fg_after_support_mean = $depthConfFgAfterSupportMean
    depth_conf_fg_final_mean = $depthConfFgFinalMean
    mv_support_generation_region_mode = $mvSupportGenerationRegionMode
    mv_support_generation_fg_mask_source = $mvSupportGenerationFgMaskSource
    support_generation_active = $supportGenerationActive
    point_support_path_active = $pointSupportPathActive
    point_mv_depth_support_path_active = $pointMvDepthSupportPathActive
    point_mv_mask_support_path_active = $pointMvMaskSupportPathActive
    point_target_blend_mv_support_active = $pointTargetBlendMvSupportActive
    point_support_eff_mean = $pointSupportEffMean
    point_support_eff_fg_mean = $pointSupportEffFgMean
    point_support_eff_bg_mean = $pointSupportEffBgMean
    point_mv_depth_support_eff_mean = $pointMvDepthSupportEffMean
    point_mv_depth_support_eff_fg_mean = $pointMvDepthSupportEffFgMean
    point_mv_depth_support_eff_bg_mean = $pointMvDepthSupportEffBgMean
    point_mv_mask_support_eff_mean = $pointMvMaskSupportEffMean
    point_mv_mask_support_eff_fg_mean = $pointMvMaskSupportEffFgMean
    point_mv_mask_support_eff_bg_mean = $pointMvMaskSupportEffBgMean
    best_visual_png = $bestVisualPng
    stage_best_strip_png = $compareStripPng
    paused = $pausedValue
    pause_reason = $pauseReasonValue
    modal_run_state = $modalRunState
    modal_run_script_path = $modalRunScriptPath
    modal_run_exit_code = $modalRunExitCode
    note = ("single_probe_sync_" + $State)
}
Write-JsonNoBom -Path $watchPath -Obj $watchObj

$watchMd = @()
$watchMd += "# Watch Ghost Outputs (single-probe)"
$watchMd += ""
$watchMd += "- updated: $updatedAt"
$watchMd += "- probe_id: $ProbeId"
$watchMd += "- state: $State"
$watchMd += "- current_stage: $currentStage"
$watchMd += "- active_candidate_result_json: $activeCandidateResultJson"
$watchMd += "- active_candidate_invalid_reason: $activeCandidateInvalidReason"
$watchMd += "- active_precompute_source: $activePrecomputeSource"
$watchMd += "- active_precompute_mv_support_on: $activePrecomputeMvSupportOn"
$watchMd += "- active_precompute_mv_support_region_mode: $activePrecomputeMvSupportRegionMode"
$watchMd += "- active_precompute_mv_support_fg_mask_source: $activePrecomputeMvSupportFgMaskSource"
$watchMd += "- active_precompute_mv_support_fg_erode_px: $activePrecomputeMvSupportFgErodePx"
$watchMd += "- active_precompute_mv_support_fg_preserve_px: $activePrecomputeMvSupportFgPreservePx"
$watchMd += "- active_point_target_mode: $activePointTargetMode"
$watchMd += "- active_point_target_blend_by_mv_support: $activePointTargetBlendByMvSupport"
$watchMd += "- active_point_target_blend_mv_region_mode: $activePointTargetBlendMvRegionMode"
$watchMd += "- active_point_support_mode: $activePointSupportMode"
$watchMd += "- active_point_mv_depth_support_mode: $activePointMvDepthSupportMode"
$watchMd += "- active_point_mv_mask_support_mode: $activePointMvMaskSupportMode"
$watchMd += "- active_point_mv_depth_region_mode: $activePointMvDepthRegionMode"
$watchMd += "- point_mv_support_mean: $pointMvSupportMean"
$watchMd += "- point_mv_support_fg_mean: $pointMvSupportFgMean"
$watchMd += "- point_mv_support_bg_mean: $pointMvSupportBgMean"
$watchMd += "- mv_support_raw_mean: $mvSupportRawMean"
$watchMd += "- mv_support_valid_ratio: $mvSupportValidRatio"
$watchMd += "- mv_support_fg_valid_ratio: $mvSupportFgValidRatio"
$watchMd += "- mv_support_bg_valid_ratio: $mvSupportBgValidRatio"
$watchMd += "- mv_support_pair_count_eff: $mvSupportPairCountEff"
$watchMd += "- mv_support_conf_mean: $mvSupportConfMean"
$watchMd += "- mv_support_nan_ratio: $mvSupportNanRatio"
$watchMd += "- depth_conf_delta_mean: $depthConfDeltaMean"
$watchMd += "- mv_support_fg_mean: $mvSupportFgMean"
$watchMd += "- mv_support_bg_mean: $mvSupportBgMean"
$watchMd += "- depth_conf_delta_fg_mean: $depthConfDeltaFgMean"
$watchMd += "- depth_conf_delta_bg_mean: $depthConfDeltaBgMean"
$watchMd += "- depth_conf_fg_preserved_active: $depthConfFgPreservedActive"
$watchMd += "- depth_conf_fg_preserve_px: $depthConfFgPreservePx"
$watchMd += "- depth_conf_fg_exact_ratio: $depthConfFgExactRatio"
$watchMd += "- depth_conf_fg_preserve_ratio: $depthConfFgPreserveRatio"
$watchMd += "- depth_conf_fg_raw_mean: $depthConfFgRawMean"
$watchMd += "- depth_conf_fg_after_support_mean: $depthConfFgAfterSupportMean"
$watchMd += "- depth_conf_fg_final_mean: $depthConfFgFinalMean"
$watchMd += "- mv_support_generation_region_mode: $mvSupportGenerationRegionMode"
$watchMd += "- mv_support_generation_fg_mask_source: $mvSupportGenerationFgMaskSource"
$watchMd += "- support_generation_active: $supportGenerationActive"
$watchMd += "- point_support_path_active: $pointSupportPathActive"
$watchMd += "- point_mv_depth_support_path_active: $pointMvDepthSupportPathActive"
$watchMd += "- point_mv_mask_support_path_active: $pointMvMaskSupportPathActive"
$watchMd += "- point_target_blend_mv_support_active: $pointTargetBlendMvSupportActive"
$watchMd += "- point_support_eff_mean: $pointSupportEffMean"
$watchMd += "- point_support_eff_fg_mean: $pointSupportEffFgMean"
$watchMd += "- point_support_eff_bg_mean: $pointSupportEffBgMean"
$watchMd += "- point_mv_depth_support_eff_mean: $pointMvDepthSupportEffMean"
$watchMd += "- point_mv_depth_support_eff_fg_mean: $pointMvDepthSupportEffFgMean"
$watchMd += "- point_mv_depth_support_eff_bg_mean: $pointMvDepthSupportEffBgMean"
$watchMd += "- point_mv_mask_support_eff_mean: $pointMvMaskSupportEffMean"
$watchMd += "- point_mv_mask_support_eff_fg_mean: $pointMvMaskSupportEffFgMean"
$watchMd += "- point_mv_mask_support_eff_bg_mean: $pointMvMaskSupportEffBgMean"
$watchMd += "- paused: $pausedValue"
$watchMd += "- pause_reason: $pauseReasonValue"
$watchMd += "- modal_run_state: $modalRunState"
$watchMd += "- modal_run_exit_code: $modalRunExitCode"
$watchMd += "- note: single_probe_sync_$State"
Set-Content -Path (Join-Path $StatusDir "watch_ghost_outputs_latest.md") -Value ($watchMd -join "`n") -Encoding UTF8

Write-Host "[single-probe-sync] probe=$ProbeId state=$State candidate=$activeCandidateResultJson"
