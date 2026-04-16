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

function Test-StringEq(
    [string]$Name,
    [string]$Expected,
    [string]$Actual,
    [switch]$AllowBlankActual
) {
    if ($AllowBlankActual -and [string]::IsNullOrWhiteSpace($Actual)) {
        Write-Host ("{0}: SKIP(blank_actual)" -f $Name)
        return $true
    }
    $ok = ([string]$Expected -eq [string]$Actual)
    Write-Host ("{0}: {1} expected={2} actual={3}" -f $Name, $(if ($ok) { "OK" } else { "MISMATCH" }), $Expected, $Actual)
    return $ok
}

function Test-BoolEq(
    [string]$Name,
    $Expected,
    $Actual
) {
    $exp = To-BoolLoose $Expected $false
    $act = To-BoolLoose $Actual $false
    $ok = ($exp -eq $act)
    Write-Host ("{0}: {1} expected={2} actual={3}" -f $Name, $(if ($ok) { "OK" } else { "MISMATCH" }), $exp, $act)
    return $ok
}

function Test-DoubleEq(
    [string]$Name,
    $Expected,
    $Actual
) {
    $exp = if ($null -eq $Expected -or [string]::IsNullOrWhiteSpace([string]$Expected)) { [double]::NaN } else { [double]$Expected }
    $act = if ($null -eq $Actual -or [string]::IsNullOrWhiteSpace([string]$Actual)) { [double]::NaN } else { [double]$Actual }
    if (([double]::IsNaN($exp)) -and ([double]::IsNaN($act))) {
        Write-Host ("{0}: OK expected=NaN actual=NaN" -f $Name)
        return $true
    }
    $ok = ([math]::Abs($exp - $act) -lt 1e-9)
    Write-Host ("{0}: {1} expected={2} actual={3}" -f $Name, $(if ($ok) { "OK" } else { "MISMATCH" }), $exp, $act)
    return $ok
}

$candidate = Read-JsonMaybe -Path (Join-Path $StatusDir "candidate_result_latest.json")
$autoloop = Read-JsonMaybe -Path (Join-Path $StatusDir "overnight_ghost_autoloop_latest.json")
$watch = Read-JsonMaybe -Path (Join-Path $StatusDir "watch_ghost_outputs_latest.json")
$candidateLatestPath = "logs/modal_phase5/candidate_result_latest.json"

if (($null -eq $candidate) -or ($null -eq $autoloop) -or ($null -eq $watch)) {
    Write-Host "[candidate-consistency] missing required json(s)"
    exit 2
}

$ok = $true
$ok = (Test-StringEq -Name "candidate_result_json->autoloop" -Expected $candidateLatestPath -Actual ([string]$autoloop.active_candidate_result_json)) -and $ok
$ok = (Test-StringEq -Name "candidate_result_json->watch" -Expected $candidateLatestPath -Actual ([string]$watch.active_candidate_result_json)) -and $ok
$ok = (Test-StringEq -Name "candidate_invalid_reason->autoloop" -Expected ([string]$candidate.candidate_invalid_reason) -Actual ([string]$autoloop.active_candidate_invalid_reason) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "precompute_source->autoloop" -Expected ([string]$candidate.precompute_source) -Actual ([string]$autoloop.active_precompute_source) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "precompute_source_requested->autoloop" -Expected ([string]$candidate.precompute_source_requested) -Actual ([string]$autoloop.active_precompute_source_requested) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "precompute_source_resolved->autoloop" -Expected ([string]$candidate.precompute_source_resolved) -Actual ([string]$autoloop.active_precompute_source_resolved) -AllowBlankActual) -and $ok
$ok = (Test-BoolEq -Name "precompute_fallback_used->autoloop" -Expected $candidate.precompute_fallback_used -Actual $autoloop.active_precompute_fallback_used) -and $ok
$ok = (Test-BoolEq -Name "precompute_timeout_hit->autoloop" -Expected $candidate.precompute_timeout_hit -Actual $autoloop.active_precompute_timeout_hit) -and $ok
$ok = (Test-StringEq -Name "precompute_mv_support_on->autoloop" -Expected ([string]$candidate.precompute_mv_support_on) -Actual ([string]$autoloop.active_precompute_mv_support_on) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "precompute_mv_support_region_mode->autoloop" -Expected ([string]$candidate.precompute_mv_support_region_mode) -Actual ([string]$autoloop.active_precompute_mv_support_region_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "precompute_mv_support_fg_mask_source->autoloop" -Expected ([string]$candidate.precompute_mv_support_fg_mask_source) -Actual ([string]$autoloop.active_precompute_mv_support_fg_mask_source) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "precompute_mv_support_fg_erode_px->autoloop" -Expected ([string]$candidate.precompute_mv_support_fg_erode_px) -Actual ([string]$autoloop.active_precompute_mv_support_fg_erode_px) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "precompute_mv_support_fg_preserve_px->autoloop" -Expected ([string]$candidate.precompute_mv_support_fg_preserve_px) -Actual ([string]$autoloop.active_precompute_mv_support_fg_preserve_px) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "point_target_mode->autoloop" -Expected ([string]$candidate.point_target_mode) -Actual ([string]$autoloop.active_point_target_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "point_target_blend_by_mv_support->autoloop" -Expected ([string]$candidate.point_target_blend_by_mv_support) -Actual ([string]$autoloop.active_point_target_blend_by_mv_support) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "point_support_mode->autoloop" -Expected ([string]$candidate.point_support_mode) -Actual ([string]$autoloop.active_point_support_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "point_mv_depth_support_mode->autoloop" -Expected ([string]$candidate.point_mv_depth_support_mode) -Actual ([string]$autoloop.active_point_mv_depth_support_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "point_mv_mask_support_mode->autoloop" -Expected ([string]$candidate.point_mv_mask_support_mode) -Actual ([string]$autoloop.active_point_mv_mask_support_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "point_mv_depth_region_mode->autoloop" -Expected ([string]$candidate.point_mv_depth_region_mode) -Actual ([string]$autoloop.active_point_mv_depth_region_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "use_fg_mask->autoloop" -Expected ([string]$candidate.use_fg_mask) -Actual ([string]$autoloop.active_use_fg_mask) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "fg_mask_source->autoloop" -Expected ([string]$candidate.fg_mask_source) -Actual ([string]$autoloop.active_fg_mask_source) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "tf32->autoloop" -Expected ([string]$candidate.tf32) -Actual ([string]$autoloop.active_tf32) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "amp->autoloop" -Expected ([string]$candidate.amp) -Actual ([string]$autoloop.active_amp) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "strict_deterministic->autoloop" -Expected ([string]$candidate.strict_deterministic) -Actual ([string]$autoloop.active_strict_deterministic) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "runner_tf32->autoloop" -Expected ([string]$candidate.runner_tf32) -Actual ([string]$autoloop.active_runner_tf32) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "runner_amp->autoloop" -Expected ([string]$candidate.runner_amp) -Actual ([string]$autoloop.active_runner_amp) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "runner_strict_deterministic->autoloop" -Expected ([string]$candidate.runner_strict_deterministic) -Actual ([string]$autoloop.active_runner_strict_deterministic) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "teacher_tf32->autoloop" -Expected ([string]$candidate.teacher_tf32) -Actual ([string]$autoloop.active_teacher_tf32) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "teacher_amp->autoloop" -Expected ([string]$candidate.teacher_amp) -Actual ([string]$autoloop.active_teacher_amp) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "teacher_deterministic->autoloop" -Expected ([string]$candidate.teacher_deterministic) -Actual ([string]$autoloop.active_teacher_deterministic) -AllowBlankActual) -and $ok
$ok = (Test-DoubleEq -Name "support_generation_active->autoloop" -Expected $candidate.support_generation_active -Actual $autoloop.support_generation_active) -and $ok
$ok = (Test-DoubleEq -Name "point_support_path_active->autoloop" -Expected $candidate.point_support_path_active -Actual $autoloop.point_support_path_active) -and $ok
$ok = (Test-DoubleEq -Name "point_mv_depth_support_path_active->autoloop" -Expected $candidate.point_mv_depth_support_path_active -Actual $autoloop.point_mv_depth_support_path_active) -and $ok
$ok = (Test-DoubleEq -Name "point_mv_mask_support_path_active->autoloop" -Expected $candidate.point_mv_mask_support_path_active -Actual $autoloop.point_mv_mask_support_path_active) -and $ok
$ok = (Test-DoubleEq -Name "point_target_blend_mv_support_active->autoloop" -Expected $candidate.point_target_blend_mv_support_active -Actual $autoloop.point_target_blend_mv_support_active) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_raw_mean->autoloop" -Expected $candidate.mv_support_raw_mean -Actual $autoloop.mv_support_raw_mean) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_valid_ratio->autoloop" -Expected $candidate.mv_support_valid_ratio -Actual $autoloop.mv_support_valid_ratio) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_fg_valid_ratio->autoloop" -Expected $candidate.mv_support_fg_valid_ratio -Actual $autoloop.mv_support_fg_valid_ratio) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_bg_valid_ratio->autoloop" -Expected $candidate.mv_support_bg_valid_ratio -Actual $autoloop.mv_support_bg_valid_ratio) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_pair_count_eff->autoloop" -Expected $candidate.mv_support_pair_count_eff -Actual $autoloop.mv_support_pair_count_eff) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_conf_mean->autoloop" -Expected $candidate.mv_support_conf_mean -Actual $autoloop.mv_support_conf_mean) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_nan_ratio->autoloop" -Expected $candidate.mv_support_nan_ratio -Actual $autoloop.mv_support_nan_ratio) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_delta_mean->autoloop" -Expected $candidate.depth_conf_delta_mean -Actual $autoloop.depth_conf_delta_mean) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_fg_mean->autoloop" -Expected $candidate.mv_support_fg_mean -Actual $autoloop.mv_support_fg_mean) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_bg_mean->autoloop" -Expected $candidate.mv_support_bg_mean -Actual $autoloop.mv_support_bg_mean) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_delta_fg_mean->autoloop" -Expected $candidate.depth_conf_delta_fg_mean -Actual $autoloop.depth_conf_delta_fg_mean) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_delta_bg_mean->autoloop" -Expected $candidate.depth_conf_delta_bg_mean -Actual $autoloop.depth_conf_delta_bg_mean) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_preserved_active->autoloop" -Expected $candidate.depth_conf_fg_preserved_active -Actual $autoloop.depth_conf_fg_preserved_active) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_preserve_px->autoloop" -Expected $candidate.depth_conf_fg_preserve_px -Actual $autoloop.depth_conf_fg_preserve_px) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_exact_ratio->autoloop" -Expected $candidate.depth_conf_fg_exact_ratio -Actual $autoloop.depth_conf_fg_exact_ratio) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_preserve_ratio->autoloop" -Expected $candidate.depth_conf_fg_preserve_ratio -Actual $autoloop.depth_conf_fg_preserve_ratio) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_raw_mean->autoloop" -Expected $candidate.depth_conf_fg_raw_mean -Actual $autoloop.depth_conf_fg_raw_mean) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_after_support_mean->autoloop" -Expected $candidate.depth_conf_fg_after_support_mean -Actual $autoloop.depth_conf_fg_after_support_mean) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_final_mean->autoloop" -Expected $candidate.depth_conf_fg_final_mean -Actual $autoloop.depth_conf_fg_final_mean) -and $ok
$ok = (Test-StringEq -Name "mv_support_generation_region_mode->autoloop" -Expected ([string]$candidate.mv_support_generation_region_mode) -Actual ([string]$autoloop.mv_support_generation_region_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "mv_support_generation_fg_mask_source->autoloop" -Expected ([string]$candidate.mv_support_generation_fg_mask_source) -Actual ([string]$autoloop.mv_support_generation_fg_mask_source) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "precompute_mv_support_on->watch" -Expected ([string]$candidate.precompute_mv_support_on) -Actual ([string]$watch.active_precompute_mv_support_on) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "precompute_mv_support_region_mode->watch" -Expected ([string]$candidate.precompute_mv_support_region_mode) -Actual ([string]$watch.active_precompute_mv_support_region_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "precompute_mv_support_fg_mask_source->watch" -Expected ([string]$candidate.precompute_mv_support_fg_mask_source) -Actual ([string]$watch.active_precompute_mv_support_fg_mask_source) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "precompute_mv_support_fg_erode_px->watch" -Expected ([string]$candidate.precompute_mv_support_fg_erode_px) -Actual ([string]$watch.active_precompute_mv_support_fg_erode_px) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "precompute_mv_support_fg_preserve_px->watch" -Expected ([string]$candidate.precompute_mv_support_fg_preserve_px) -Actual ([string]$watch.active_precompute_mv_support_fg_preserve_px) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "point_target_mode->watch" -Expected ([string]$candidate.point_target_mode) -Actual ([string]$watch.active_point_target_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "point_target_blend_by_mv_support->watch" -Expected ([string]$candidate.point_target_blend_by_mv_support) -Actual ([string]$watch.active_point_target_blend_by_mv_support) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "point_support_mode->watch" -Expected ([string]$candidate.point_support_mode) -Actual ([string]$watch.active_point_support_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "point_mv_depth_support_mode->watch" -Expected ([string]$candidate.point_mv_depth_support_mode) -Actual ([string]$watch.active_point_mv_depth_support_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "point_mv_mask_support_mode->watch" -Expected ([string]$candidate.point_mv_mask_support_mode) -Actual ([string]$watch.active_point_mv_mask_support_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "point_mv_depth_region_mode->watch" -Expected ([string]$candidate.point_mv_depth_region_mode) -Actual ([string]$watch.active_point_mv_depth_region_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "use_fg_mask->watch" -Expected ([string]$candidate.use_fg_mask) -Actual ([string]$watch.active_use_fg_mask) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "fg_mask_source->watch" -Expected ([string]$candidate.fg_mask_source) -Actual ([string]$watch.active_fg_mask_source) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "tf32->watch" -Expected ([string]$candidate.tf32) -Actual ([string]$watch.active_tf32) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "amp->watch" -Expected ([string]$candidate.amp) -Actual ([string]$watch.active_amp) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "strict_deterministic->watch" -Expected ([string]$candidate.strict_deterministic) -Actual ([string]$watch.active_strict_deterministic) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "runner_tf32->watch" -Expected ([string]$candidate.runner_tf32) -Actual ([string]$watch.active_runner_tf32) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "runner_amp->watch" -Expected ([string]$candidate.runner_amp) -Actual ([string]$watch.active_runner_amp) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "runner_strict_deterministic->watch" -Expected ([string]$candidate.runner_strict_deterministic) -Actual ([string]$watch.active_runner_strict_deterministic) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "teacher_tf32->watch" -Expected ([string]$candidate.teacher_tf32) -Actual ([string]$watch.active_teacher_tf32) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "teacher_amp->watch" -Expected ([string]$candidate.teacher_amp) -Actual ([string]$watch.active_teacher_amp) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "teacher_deterministic->watch" -Expected ([string]$candidate.teacher_deterministic) -Actual ([string]$watch.active_teacher_deterministic) -AllowBlankActual) -and $ok
$ok = (Test-DoubleEq -Name "support_generation_active->watch" -Expected $candidate.support_generation_active -Actual $watch.support_generation_active) -and $ok
$ok = (Test-DoubleEq -Name "point_support_path_active->watch" -Expected $candidate.point_support_path_active -Actual $watch.point_support_path_active) -and $ok
$ok = (Test-DoubleEq -Name "point_mv_depth_support_path_active->watch" -Expected $candidate.point_mv_depth_support_path_active -Actual $watch.point_mv_depth_support_path_active) -and $ok
$ok = (Test-DoubleEq -Name "point_mv_mask_support_path_active->watch" -Expected $candidate.point_mv_mask_support_path_active -Actual $watch.point_mv_mask_support_path_active) -and $ok
$ok = (Test-DoubleEq -Name "point_target_blend_mv_support_active->watch" -Expected $candidate.point_target_blend_mv_support_active -Actual $watch.point_target_blend_mv_support_active) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_raw_mean->watch" -Expected $candidate.mv_support_raw_mean -Actual $watch.mv_support_raw_mean) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_valid_ratio->watch" -Expected $candidate.mv_support_valid_ratio -Actual $watch.mv_support_valid_ratio) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_fg_valid_ratio->watch" -Expected $candidate.mv_support_fg_valid_ratio -Actual $watch.mv_support_fg_valid_ratio) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_bg_valid_ratio->watch" -Expected $candidate.mv_support_bg_valid_ratio -Actual $watch.mv_support_bg_valid_ratio) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_pair_count_eff->watch" -Expected $candidate.mv_support_pair_count_eff -Actual $watch.mv_support_pair_count_eff) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_conf_mean->watch" -Expected $candidate.mv_support_conf_mean -Actual $watch.mv_support_conf_mean) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_nan_ratio->watch" -Expected $candidate.mv_support_nan_ratio -Actual $watch.mv_support_nan_ratio) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_delta_mean->watch" -Expected $candidate.depth_conf_delta_mean -Actual $watch.depth_conf_delta_mean) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_fg_mean->watch" -Expected $candidate.mv_support_fg_mean -Actual $watch.mv_support_fg_mean) -and $ok
$ok = (Test-DoubleEq -Name "mv_support_bg_mean->watch" -Expected $candidate.mv_support_bg_mean -Actual $watch.mv_support_bg_mean) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_delta_fg_mean->watch" -Expected $candidate.depth_conf_delta_fg_mean -Actual $watch.depth_conf_delta_fg_mean) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_delta_bg_mean->watch" -Expected $candidate.depth_conf_delta_bg_mean -Actual $watch.depth_conf_delta_bg_mean) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_preserved_active->watch" -Expected $candidate.depth_conf_fg_preserved_active -Actual $watch.depth_conf_fg_preserved_active) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_preserve_px->watch" -Expected $candidate.depth_conf_fg_preserve_px -Actual $watch.depth_conf_fg_preserve_px) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_exact_ratio->watch" -Expected $candidate.depth_conf_fg_exact_ratio -Actual $watch.depth_conf_fg_exact_ratio) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_preserve_ratio->watch" -Expected $candidate.depth_conf_fg_preserve_ratio -Actual $watch.depth_conf_fg_preserve_ratio) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_raw_mean->watch" -Expected $candidate.depth_conf_fg_raw_mean -Actual $watch.depth_conf_fg_raw_mean) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_after_support_mean->watch" -Expected $candidate.depth_conf_fg_after_support_mean -Actual $watch.depth_conf_fg_after_support_mean) -and $ok
$ok = (Test-DoubleEq -Name "depth_conf_fg_final_mean->watch" -Expected $candidate.depth_conf_fg_final_mean -Actual $watch.depth_conf_fg_final_mean) -and $ok
$ok = (Test-StringEq -Name "mv_support_generation_region_mode->watch" -Expected ([string]$candidate.mv_support_generation_region_mode) -Actual ([string]$watch.mv_support_generation_region_mode) -AllowBlankActual) -and $ok
$ok = (Test-StringEq -Name "mv_support_generation_fg_mask_source->watch" -Expected ([string]$candidate.mv_support_generation_fg_mask_source) -Actual ([string]$watch.mv_support_generation_fg_mask_source) -AllowBlankActual) -and $ok

if (-not $ok) { exit 2 }
