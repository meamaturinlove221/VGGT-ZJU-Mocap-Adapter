[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$StatusDir = "logs/modal_phase5",
    [string]$OutPath = "logs/modal_phase5/candidate_result_latest.json"
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

function Read-LastCsvRow([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try {
        $rows = @(Import-Csv $Path)
        if ($rows.Count -le 0) { return $null }
        return $rows[$rows.Count - 1]
    } catch {
        return $null
    }
}

function Pick-String([string[]]$Values) {
    foreach ($v in $Values) {
        if (-not [string]::IsNullOrWhiteSpace([string]$v)) { return [string]$v }
    }
    return ""
}

function Resolve-PrecisionField([object]$FtRow, [object]$Contract, [object]$AutoloopStatus, [object]$Payload, [string]$Field) {
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
        $(if ($FtRow) { [string]$FtRow.$Field } else { "" }),
        $contractFallback,
        $(if ($Payload) { [string]$Payload.$Field } else { "" }),
        $(if ($AutoloopStatus) { [string]$AutoloopStatus."active_$Field" } else { "" }),
        $(if ($AutoloopStatus) { [string]$AutoloopStatus.$Field } else { "" })
    )
}

function Is-BlankOrNaN([object]$Value) {
    if ($null -eq $Value) { return $true }
    $s = [string]$Value
    if ([string]::IsNullOrWhiteSpace($s)) { return $true }
    $norm = $s.Trim().ToLowerInvariant()
    return $norm -eq "nan"
}

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $abs = Join-Path (Resolve-Path ".").Path $Path
    $dir = Split-Path -Parent $abs
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 20
    [System.IO.File]::WriteAllText($abs, $json, $enc)
}

$ghostRow = Read-LastCsvRow -Path (Join-Path $StatusDir "ghost_mvdepth_sweep_latest.csv")
$ftRow = Read-LastCsvRow -Path (Join-Path $StatusDir "vggt_ft_sweep_latest.csv")
$autoloopStatus = Read-JsonMaybe -Path (Join-Path $StatusDir "overnight_ghost_autoloop_latest.json")
$contract = Read-JsonMaybe -Path (Join-Path $StatusDir "probe_contract_latest.json")

if (($null -eq $ghostRow) -and ($null -eq $ftRow)) {
    Write-Host "[repair-candidate-result] skip: no latest csv rows"
    exit 2
}

$payload = [ordered]@{
    updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    source = "repair_candidate_result_latest"
    candidate_result_version = 1
    synthetic_from_frozen_state = $true
    candidate_result_json = $OutPath
}

if ($autoloopStatus -ne $null) {
    $payload.current_stage = [string]$autoloopStatus.current_stage
    $payload.current_cycle = [string]$autoloopStatus.current_cycle
}

foreach ($row in @($ftRow, $ghostRow)) {
    if ($null -eq $row) { continue }
    foreach ($prop in @($row.PSObject.Properties)) {
        $payload[$prop.Name] = $prop.Value
    }
}

$probeId = Pick-String @(
    $(if ($contract) { [string]$contract.probe_id } else { "" }),
    $(if ($payload.probe_id) { [string]$payload.probe_id } else { "" }),
    $(if ($payload.lane_id) { [string]$payload.lane_id -replace '^probe_', '' } else { "" })
)

foreach ($field in @(
    "precompute_mv_support_region_mode",
    "precompute_mv_support_fg_mask_source",
    "precompute_mv_support_fg_erode_px",
    "mv_support_generation_region_mode",
    "mv_support_generation_fg_mask_source",
    "point_support_mode",
    "point_mv_depth_support_mode",
    "point_mv_mask_support_mode",
    "point_target_mode",
    "point_target_blend_by_mv_support",
    "point_target_blend_mv_region_mode",
    "point_mv_depth_region_mode",
    "use_fg_mask",
    "fg_mask_source"
)) {
    if ([string]::IsNullOrWhiteSpace([string]$payload[$field])) {
        $payload[$field] = Pick-String @(
            $(if ($contract) { [string]$contract.$field } else { "" }),
            $(if ($autoloopStatus) { [string]$autoloopStatus."active_$field" } else { "" }),
            $(if ($autoloopStatus) { [string]$autoloopStatus.$field } else { "" })
        )
    }
}

if ([string]::IsNullOrWhiteSpace([string]$payload.precompute_mv_support_region_mode)) {
    switch -Regex ($probeId) {
        '^G0$' { $payload.precompute_mv_support_region_mode = 'bg_only' }
        '^S[0-3]$' { $payload.precompute_mv_support_region_mode = 'all' }
        '^T0_smoke$' { $payload.precompute_mv_support_region_mode = 'all' }
        default { }
    }
    if ([string]::IsNullOrWhiteSpace([string]$payload.precompute_mv_support_region_mode)) {
        if (([string]$payload.precompute_mv_support_on).Trim().ToLowerInvariant() -eq 'on') {
            $payload.precompute_mv_support_region_mode = 'all'
        }
    }
}
if ([string]::IsNullOrWhiteSpace([string]$payload.precompute_mv_support_fg_mask_source)) {
    switch -Regex ($probeId) {
        '^G0$' { $payload.precompute_mv_support_fg_mask_source = 'mask' }
        '^S[0-3]$' { $payload.precompute_mv_support_fg_mask_source = 'mask' }
        '^T0_smoke$' { $payload.precompute_mv_support_fg_mask_source = 'mask' }
        default { }
    }
    if ([string]::IsNullOrWhiteSpace([string]$payload.precompute_mv_support_fg_mask_source)) {
        if (([string]$payload.precompute_mv_support_on).Trim().ToLowerInvariant() -eq 'on') {
            $payload.precompute_mv_support_fg_mask_source = 'mask'
        }
    }
}
if ([string]::IsNullOrWhiteSpace([string]$payload.precompute_mv_support_fg_erode_px)) {
    switch -Regex ($probeId) {
        '^G0$' { $payload.precompute_mv_support_fg_erode_px = '5' }
        '^S[0-3]$' { $payload.precompute_mv_support_fg_erode_px = '5' }
        '^T0_smoke$' { $payload.precompute_mv_support_fg_erode_px = '5' }
        default { }
    }
    if ([string]::IsNullOrWhiteSpace([string]$payload.precompute_mv_support_fg_erode_px)) {
        if (([string]$payload.precompute_mv_support_on).Trim().ToLowerInvariant() -eq 'on') {
            $payload.precompute_mv_support_fg_erode_px = '5'
        }
    }
}
foreach ($field in @(
    "tf32",
    "amp",
    "strict_deterministic",
    "runner_tf32",
    "runner_amp",
    "runner_strict_deterministic",
    "precompute_tf32",
    "precompute_amp",
    "precompute_strict_deterministic",
    "teacher_tf32",
    "teacher_amp",
    "teacher_deterministic"
)) {
    $resolved = Resolve-PrecisionField -FtRow $ftRow -Contract $contract -AutoloopStatus $autoloopStatus -Payload $payload -Field $field
    if (-not [string]::IsNullOrWhiteSpace($resolved)) {
        $payload[$field] = $resolved
    }
}

foreach ($field in @(
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
    "support_generation_active",
    "point_support_path_active",
    "point_mv_depth_support_path_active",
    "point_mv_mask_support_path_active",
    "point_target_blend_mv_support_active"
)) {
    if ([string]::IsNullOrWhiteSpace([string]$payload[$field])) {
        $payload[$field] = Pick-String @(
            $(if ($autoloopStatus) { [string]$autoloopStatus.$field } else { "" }),
            $(if ($contract) { [string]$contract.$field } else { "" })
        )
    }
}

if ([string]::IsNullOrWhiteSpace([string]$payload.mv_support_generation_region_mode)) {
    $payload.mv_support_generation_region_mode = [string]$payload.precompute_mv_support_region_mode
}
if ([string]::IsNullOrWhiteSpace([string]$payload.mv_support_generation_fg_mask_source)) {
    $payload.mv_support_generation_fg_mask_source = [string]$payload.precompute_mv_support_fg_mask_source
}
$supportGenActive = (([string]$payload.precompute_mv_support_on).Trim().ToLowerInvariant()) -eq 'on'
$precomputeRegionMode = ([string]$payload.precompute_mv_support_region_mode).Trim().ToLowerInvariant()
$currentGenRegionMode = ([string]$payload.mv_support_generation_region_mode).Trim().ToLowerInvariant()
if (
    $supportGenActive -and
    ($precomputeRegionMode -in @('bg_only', 'fg_eroded_off')) -and
    (
        [string]::IsNullOrWhiteSpace([string]$payload.mv_support_generation_region_mode) -or
        ($currentGenRegionMode -eq 'all')
    )
) {
    $payload.mv_support_generation_region_mode = [string]$payload.precompute_mv_support_region_mode
}
if (
    $supportGenActive -and
    (-not [string]::IsNullOrWhiteSpace([string]$payload.precompute_mv_support_fg_mask_source)) -and
    [string]::IsNullOrWhiteSpace([string]$payload.mv_support_generation_fg_mask_source)
) {
    $payload.mv_support_generation_fg_mask_source = [string]$payload.precompute_mv_support_fg_mask_source
}
$supportStatsFields = @(
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
$helperScript = Join-Path $PSScriptRoot "fetch_precompute_support_stats.py"
$seqNamesRaw = Pick-String @(
    $(if ($contract) { [string]$contract.seq_names } else { "" }),
    $(if ($payload.seq_names) { [string]$payload.seq_names } else { "" })
)
$geomSubdir = Pick-String @(
    $(if ($payload.best_geom_subdir) { [string]$payload.best_geom_subdir } else { "" }),
    $(if ($payload.geom_subdir) { [string]$payload.geom_subdir } else { "" }),
    $(if ($contract) { [string]$contract.pseudo_geom_subdir } else { "" })
)
if ((Test-Path $helperScript) -and -not [string]::IsNullOrWhiteSpace($seqNamesRaw) -and -not [string]::IsNullOrWhiteSpace($geomSubdir)) {
    try {
        $jsonText = @(
            & python $helperScript `
                --seq-names $seqNamesRaw `
                --geom-subdir $geomSubdir `
                --volume-name "vggt-zju-data" `
                --remote-root "/zju_mocap" 2>$null
        ) -join ""
        if (-not [string]::IsNullOrWhiteSpace($jsonText)) {
            $obj = $jsonText | ConvertFrom-Json
            foreach ($prop in @($obj.PSObject.Properties)) {
                $payload[$prop.Name] = $prop.Value
            }
        }
    } catch {
        Write-Host "[repair-candidate-result] support stats helper failed geom_subdir=$geomSubdir error=$($_.Exception.Message)"
    }
}
$payload.support_generation_active = $(if ($supportGenActive) { '1' } else { '0' })
$payload.point_support_path_active = $(if ((([string]$payload.point_support_mode).Trim().ToLowerInvariant()) -ne 'off' -and -not [string]::IsNullOrWhiteSpace([string]$payload.point_support_mode)) { '1' } else { '0' })
$payload.point_mv_depth_support_path_active = $(if ((([string]$payload.point_mv_depth_support_mode).Trim().ToLowerInvariant()) -ne 'off' -and -not [string]::IsNullOrWhiteSpace([string]$payload.point_mv_depth_support_mode)) { '1' } else { '0' })
$payload.point_mv_mask_support_path_active = $(if ((([string]$payload.point_mv_mask_support_mode).Trim().ToLowerInvariant()) -ne 'off' -and -not [string]::IsNullOrWhiteSpace([string]$payload.point_mv_mask_support_mode)) { '1' } else { '0' })
$payload.point_target_blend_mv_support_active = $(if ((([string]$payload.point_target_blend_by_mv_support).Trim().ToLowerInvariant()) -eq 'on') { '1' } else { '0' })

Write-JsonNoBom -Path $OutPath -Obj $payload
$referencedCandidatePath = [string]$payload.candidate_result_json
if (-not [string]::IsNullOrWhiteSpace($referencedCandidatePath)) {
    $referencedCandidatePath = $referencedCandidatePath.Trim()
    if ($referencedCandidatePath -ne $OutPath) {
        Write-JsonNoBom -Path $referencedCandidatePath -Obj $payload
        Write-Host "[repair-candidate-result] wrote $referencedCandidatePath"
    }
}
Write-Host "[repair-candidate-result] wrote $OutPath"
