[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$StatusDir = "logs/modal_phase5",
    [string]$OutPath = "logs/modal_phase5/p0_resume_manifest_latest.json"
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

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 12
    [System.IO.File]::WriteAllText((Join-Path (Resolve-Path ".").Path $Path), $json, $enc)
}

$autoloopStatus = Read-JsonMaybe -Path (Join-Path $StatusDir "overnight_ghost_autoloop_latest.json")
$watchStatus = Read-JsonMaybe -Path (Join-Path $StatusDir "watch_ghost_outputs_latest.json")
$modalProgress = Read-JsonMaybe -Path (Join-Path $StatusDir "modal_run_progress_latest.json")

$manifest = [ordered]@{
    generated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    mode = "local_only"
    cloud_resume_allowed_now = $false
    runbook = "docs/p0_resume_single_run_20260307.md"
    local_tools = @(
        "scripts/repair_paused_local_status.ps1",
        "scripts/repair_latest_csv_schema.ps1",
        "scripts/repair_candidate_result_latest.ps1",
        "scripts/check_paused_state.ps1",
        "scripts/check_p0_local_readiness.ps1",
        "scripts/check_p0_source_contract.ps1",
        "scripts/check_candidate_result_consistency.ps1",
        "scripts/preflight_p0_resume_local.ps1",
        "scripts/snapshot_p0_state.ps1",
        "scripts/emit_p0_resume_manifest.ps1",
        "scripts/run_p0_local_maintenance.ps1"
    )
    expected_single_run_contract = [ordered]@{
        stage1_no_output_timeout_sec = 240
        stage2_pointmap_source = "depth_unproject"
        stage2_precompute_mv_support_on = "off"
        stage2_point_target_blend_by_mv_support = "off"
        stage2_dual_lane = $false
        stage2_post_rescue = $false
    }
    expected_log_markers = @(
        "batch_begin",
        "[precompute-heartbeat]",
        "batch_teacher_done",
        "save_done",
        "batch_done",
        "[VGGTGeomTeacher] batch="
    )
    frozen_state = [ordered]@{
        autoloop_current_stage = $(if ($autoloopStatus) { [string]$autoloopStatus.current_stage } else { "" })
        autoloop_p0_gate_pass = $(if ($autoloopStatus) { [string]$autoloopStatus.p0_gate_pass } else { "" })
        autoloop_pause_reason = $(if ($autoloopStatus) { [string]$autoloopStatus.pause_reason } else { "" })
        watch_current_stage = $(if ($watchStatus) { [string]$watchStatus.current_stage } else { "" })
        modal_progress_state = $(if ($modalProgress) { [string]$modalProgress.state } else { "" })
        modal_progress_note = $(if ($modalProgress) { [string]$modalProgress.note } else { "" })
    }
    preflight_commands = @(
        "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_paused_state.ps1",
        "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p0_local_readiness.ps1",
        "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p0_source_contract.ps1",
        "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_candidate_result_consistency.ps1",
        "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/preflight_p0_resume_local.ps1"
    )
    maintenance_command = "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_p0_local_maintenance.ps1"
}

Write-JsonNoBom -Path $OutPath -Obj $manifest
Write-Host "[manifest] wrote $OutPath"
