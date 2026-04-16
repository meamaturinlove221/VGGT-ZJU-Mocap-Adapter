[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

function Get-Text([string]$Path) {
    if (-not (Test-Path $Path)) { return "" }
    return (Get-Content $Path -Raw -Encoding UTF8)
}

function Test-Contains(
    [string]$Text,
    [string]$Pattern
) {
    return ($Text -match [regex]::Escape($Pattern))
}

$checks = @(
    @{
        name = "stage2_override_log"
        path = "scripts/run_overnight_ghost_autoloop.ps1"
        pattern = "stage2 P0 stability override:"
    },
    @{
        name = "stage2_override_depth_unproject"
        path = "scripts/run_overnight_ghost_autoloop.ps1"
        pattern = '$stage2PointmapSourceEffective = "depth_unproject"'
    },
    @{
        name = "stage2_override_precompute_mv_support_off"
        path = "scripts/run_overnight_ghost_autoloop.ps1"
        pattern = '$stage2PrecomputeMvSupportOnEffective = "off"'
    },
    @{
        name = "stage2_override_point_target_blend_off"
        path = "scripts/run_overnight_ghost_autoloop.ps1"
        pattern = '$stage2PointTargetBlendByMvSupportEffective = "off"'
    },
    @{
        name = "stage2_override_pass_precompute_mv_support"
        path = "scripts/run_overnight_ghost_autoloop.ps1"
        pattern = 'PrecomputeMvSupportOn = $stage2PrecomputeMvSupportOnEffective'
    },
    @{
        name = "stage2_override_pass_point_target_blend"
        path = "scripts/run_overnight_ghost_autoloop.ps1"
        pattern = 'PointTargetBlendByMvSupport = $stage2PointTargetBlendByMvSupportEffective'
    },
    @{
        name = "ghost_sweep_exports_candidate_invalid_reason"
        path = "scripts/run_vggt_ghost_mvdepth_sweep.ps1"
        pattern = 'candidate_invalid_reason = $candidateInvalidReason'
    },
    @{
        name = "ghost_sweep_exports_precompute_mv_support_on"
        path = "scripts/run_vggt_ghost_mvdepth_sweep.ps1"
        pattern = 'precompute_mv_support_on = $PrecomputeMvSupportOn'
    },
    @{
        name = "ghost_sweep_exports_point_target_blend"
        path = "scripts/run_vggt_ghost_mvdepth_sweep.ps1"
        pattern = 'point_target_blend_by_mv_support = $PointTargetBlendByMvSupport'
    },
    @{
        name = "ghost_sweep_candidate_result_json"
        path = "scripts/run_vggt_ghost_mvdepth_sweep.ps1"
        pattern = 'candidate_result_latest.json'
    },
    @{
        name = "precompute_batch_begin_marker"
        path = "precompute_zju_vggt_geom.py"
        pattern = 'batch_begin'
    },
    @{
        name = "precompute_structured_heartbeat_marker"
        path = "precompute_zju_vggt_geom.py"
        pattern = '[precompute-heartbeat]'
    },
    @{
        name = "precompute_teacher_done_marker"
        path = "precompute_zju_vggt_geom.py"
        pattern = 'batch_teacher_done'
    },
    @{
        name = "precompute_mv_support_done_marker"
        path = "precompute_zju_vggt_geom.py"
        pattern = 'mv_support_done'
    },
    @{
        name = "precompute_save_done_marker"
        path = "precompute_zju_vggt_geom.py"
        pattern = 'save_done'
    },
    @{
        name = "teacher_batch_timing_marker"
        path = "vggt_geom.py"
        pattern = '[VGGTGeomTeacher] batch='
    },
    @{
        name = "lr_sweep_structured_heartbeat_consumer"
        path = "scripts/run_vggt_ft_lr_sweep.ps1"
        pattern = 'Get-LatestStructuredHeartbeat'
    },
    @{
        name = "lr_sweep_heartbeat_stall_timeout"
        path = "scripts/run_vggt_ft_lr_sweep.ps1"
        pattern = 'heartbeat_stall_timeout_'
    },
    @{
        name = "watch_exports_precompute_mv_support"
        path = "scripts/watch_ghost_outputs.ps1"
        pattern = 'active_precompute_mv_support_on'
    },
    @{
        name = "watch_exports_point_target_blend"
        path = "scripts/watch_ghost_outputs.ps1"
        pattern = 'active_point_target_blend_by_mv_support'
    },
    @{
        name = "watch_exports_candidate_result_json"
        path = "scripts/watch_ghost_outputs.ps1"
        pattern = 'active_candidate_result_json'
    },
    @{
        name = "autoloop_exports_candidate_result_json"
        path = "scripts/run_overnight_ghost_autoloop.ps1"
        pattern = 'active_candidate_result_json'
    }
)

$cache = @{}
$failed = $false
foreach ($check in $checks) {
    $path = [string]$check.path
    if (-not $cache.ContainsKey($path)) {
        $cache[$path] = Get-Text -Path $path
    }
    $ok = Test-Contains -Text ([string]$cache[$path]) -Pattern ([string]$check.pattern)
    Write-Host ("{0}: {1}" -f $check.name, $(if ($ok) { "OK" } else { "MISSING" }))
    if (-not $ok) { $failed = $true }
}

if ($failed) { exit 2 }
