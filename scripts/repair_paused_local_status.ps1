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
    try {
        return (Get-Content $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
    } catch {
        return $null
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
    $json = $safe | ConvertTo-Json -Depth 12
    [System.IO.File]::WriteAllText((Join-Path (Resolve-Path ".").Path $Path), $json, $enc)
}

function Set-ObjectProperty(
    [object]$Obj,
    [string]$Name,
    [object]$Value
) {
    if ($null -eq $Obj) { return }
    $prop = $Obj.PSObject.Properties[$Name]
    if ($null -ne $prop) {
        $prop.Value = $Value
    } else {
        Add-Member -InputObject $Obj -NotePropertyName $Name -NotePropertyValue $Value
    }
}

function Get-TrackedLocalProcesses() {
    $patterns = @(
        "run_overnight_ghost_autoloop",
        "ensure_hot_update_watcher",
        "hot_update_autoloop_after_stage",
        "watch_ghost_outputs",
        "modal run",
        "modal_run_train.py"
    )
    $repairPid = $PID
    return @(
        Get-CimInstance Win32_Process | Where-Object {
            if ($_.ProcessId -eq $repairPid) { return $false }
            $cmd = [string]$_.CommandLine
            if ([string]::IsNullOrWhiteSpace($cmd)) { return $false }
            foreach ($pat in $patterns) {
                if ($cmd -like "*$pat*") { return $true }
            }
            return $false
        }
    )
}

function Get-LiveModalApps() {
    try {
        $raw = & modal app list --json 2>$null
        if ([string]::IsNullOrWhiteSpace([string]$raw)) { return @() }
        $apps = $raw | ConvertFrom-Json
        return @($apps | Where-Object { $_.State -in @("running", "deployed") })
    } catch {
        return @()
    }
}

function Write-MinimalMd(
    [string]$Path,
    [string]$Title,
    [System.Collections.IDictionary]$Lines
) {
    $md = @()
    $md += "# $Title"
    $md += ""
    foreach ($k in $Lines.Keys) {
        $md += "- ${k}: $($Lines[$k])"
    }
    Set-Content -Path $Path -Value ($md -join "`n") -Encoding UTF8
}

$localProcs = Get-TrackedLocalProcesses
$liveApps = Get-LiveModalApps
$nowText = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
$pauseReason = "manual_pause_no_local_chain_no_live_app"

if ($localProcs.Count -gt 0 -or $liveApps.Count -gt 0) {
    Write-Host "[repair-paused] skip: local_processes=$($localProcs.Count) live_modal_apps=$($liveApps.Count)"
    exit 2
}

$modalProgressPath = Join-Path $StatusDir "modal_run_progress_latest.json"
$autoloopJsonPath = Join-Path $StatusDir "overnight_ghost_autoloop_latest.json"
$autoloopMdPath = Join-Path $StatusDir "overnight_ghost_autoloop_latest.md"
$watchJsonPath = Join-Path $StatusDir "watch_ghost_outputs_latest.json"
$watchMdPath = Join-Path $StatusDir "watch_ghost_outputs_latest.md"
$candidateResultPath = Join-Path $StatusDir "candidate_result_latest.json"
$candidateResult = Read-JsonMaybe -Path $candidateResultPath
$candidateResultLatestRel = ($candidateResultPath -replace '\\','/')

$modalProgress = Read-JsonMaybe -Path $modalProgressPath
if ($null -ne $modalProgress) {
    if ([string]$modalProgress.state -in @("running", "retrying", "")) {
        Set-ObjectProperty -Obj $modalProgress -Name "state" -Value "stale"
    }
    Set-ObjectProperty -Obj $modalProgress -Name "updated_at" -Value $nowText
    Set-ObjectProperty -Obj $modalProgress -Name "proc_exists" -Value $false
    Set-ObjectProperty -Obj $modalProgress -Name "note" -Value "paused_local_repair_no_proc"
    Set-ObjectProperty -Obj $modalProgress -Name "pause_reason" -Value $pauseReason
    Set-ObjectProperty -Obj $modalProgress -Name "repaired_at" -Value $nowText
    Write-JsonNoBom -Path $modalProgressPath -Obj $modalProgress
}

$autoloopStatus = Read-JsonMaybe -Path $autoloopJsonPath
if ($null -ne $autoloopStatus) {
    Set-ObjectProperty -Obj $autoloopStatus -Name "updated_at" -Value $nowText
    Set-ObjectProperty -Obj $autoloopStatus -Name "paused" -Value $true
    Set-ObjectProperty -Obj $autoloopStatus -Name "pause_reason" -Value $pauseReason
    Set-ObjectProperty -Obj $autoloopStatus -Name "active_local_process_count" -Value 0
    Set-ObjectProperty -Obj $autoloopStatus -Name "active_modal_app_count" -Value 0
    if ($candidateResult -ne $null) {
        Set-ObjectProperty -Obj $autoloopStatus -Name "active_candidate_result_json" -Value $candidateResultLatestRel
        if ([string]::IsNullOrWhiteSpace([string]$autoloopStatus.active_candidate_invalid_reason)) {
            Set-ObjectProperty -Obj $autoloopStatus -Name "active_candidate_invalid_reason" -Value ([string]$candidateResult.candidate_invalid_reason)
        }
        if ([string]::IsNullOrWhiteSpace([string]$autoloopStatus.active_precompute_source)) {
            Set-ObjectProperty -Obj $autoloopStatus -Name "active_precompute_source" -Value ([string]$candidateResult.precompute_source)
        }
        if ([string]::IsNullOrWhiteSpace([string]$autoloopStatus.active_precompute_source_requested)) {
            Set-ObjectProperty -Obj $autoloopStatus -Name "active_precompute_source_requested" -Value ([string]$candidateResult.precompute_source_requested)
        }
        if ([string]::IsNullOrWhiteSpace([string]$autoloopStatus.active_precompute_source_resolved)) {
            Set-ObjectProperty -Obj $autoloopStatus -Name "active_precompute_source_resolved" -Value ([string]$candidateResult.precompute_source_resolved)
        }
        Set-ObjectProperty -Obj $autoloopStatus -Name "active_precompute_fallback_used" -Value ([string]$candidateResult.precompute_fallback_used)
        Set-ObjectProperty -Obj $autoloopStatus -Name "active_precompute_timeout_hit" -Value ([string]$candidateResult.precompute_timeout_hit)
        if ([string]::IsNullOrWhiteSpace([string]$autoloopStatus.active_precompute_mv_support_on)) {
            Set-ObjectProperty -Obj $autoloopStatus -Name "active_precompute_mv_support_on" -Value ([string]$candidateResult.precompute_mv_support_on)
        }
        if ([string]::IsNullOrWhiteSpace([string]$autoloopStatus.active_precompute_mv_support_region_mode)) {
            Set-ObjectProperty -Obj $autoloopStatus -Name "active_precompute_mv_support_region_mode" -Value ([string]$candidateResult.precompute_mv_support_region_mode)
        }
        if ([string]::IsNullOrWhiteSpace([string]$autoloopStatus.active_precompute_mv_support_fg_mask_source)) {
            Set-ObjectProperty -Obj $autoloopStatus -Name "active_precompute_mv_support_fg_mask_source" -Value ([string]$candidateResult.precompute_mv_support_fg_mask_source)
        }
        if ([string]::IsNullOrWhiteSpace([string]$autoloopStatus.active_precompute_mv_support_fg_erode_px)) {
            Set-ObjectProperty -Obj $autoloopStatus -Name "active_precompute_mv_support_fg_erode_px" -Value ([string]$candidateResult.precompute_mv_support_fg_erode_px)
        }
        if ([string]::IsNullOrWhiteSpace([string]$autoloopStatus.active_point_target_blend_by_mv_support)) {
            Set-ObjectProperty -Obj $autoloopStatus -Name "active_point_target_blend_by_mv_support" -Value ([string]$candidateResult.point_target_blend_by_mv_support)
        }
    }
    Set-ObjectProperty -Obj $autoloopStatus -Name "note" -Value "paused_local_repair_no_live_chain"
    Write-JsonNoBom -Path $autoloopJsonPath -Obj $autoloopStatus
    $autoloopMd = @()
    $autoloopMd += "# Overnight Ghost AutoLoop (paused/local-repair)"
    $autoloopMd += ""
    $autoloopMd += "- updated: $($autoloopStatus.updated_at)"
    $autoloopMd += "- current_cycle: $([string]$autoloopStatus.current_cycle)"
    $autoloopMd += "- current_stage: $([string]$autoloopStatus.current_stage)"
    $autoloopMd += "- p0_gate_pass: $([string]$autoloopStatus.p0_gate_pass)"
    $autoloopMd += "- p0_stage2_valid_rows_90m: $([string]$autoloopStatus.p0_stage2_valid_rows_90m)"
    $autoloopMd += "- active_candidate_result_json: $([string]$autoloopStatus.active_candidate_result_json)"
    $autoloopMd += "- active_precompute_mv_support_on: $([string]$autoloopStatus.active_precompute_mv_support_on)"
    $autoloopMd += "- active_precompute_mv_support_region_mode: $([string]$autoloopStatus.active_precompute_mv_support_region_mode)"
    $autoloopMd += "- active_precompute_mv_support_fg_mask_source: $([string]$autoloopStatus.active_precompute_mv_support_fg_mask_source)"
    $autoloopMd += "- active_precompute_mv_support_fg_erode_px: $([string]$autoloopStatus.active_precompute_mv_support_fg_erode_px)"
    $autoloopMd += "- pause_reason: $([string]$autoloopStatus.pause_reason)"
    $autoloopMd += "- note: $([string]$autoloopStatus.note)"
    Set-Content -Path $autoloopMdPath -Value ($autoloopMd -join "`n") -Encoding UTF8
}

$watchStatus = Read-JsonMaybe -Path $watchJsonPath
if ($null -ne $watchStatus) {
    Set-ObjectProperty -Obj $watchStatus -Name "updated_at" -Value $nowText
    Set-ObjectProperty -Obj $watchStatus -Name "paused" -Value $true
    Set-ObjectProperty -Obj $watchStatus -Name "pause_reason" -Value $pauseReason
    Set-ObjectProperty -Obj $watchStatus -Name "active_local_process_count" -Value 0
    Set-ObjectProperty -Obj $watchStatus -Name "active_modal_app_count" -Value 0
    if ($candidateResult -ne $null) {
        Set-ObjectProperty -Obj $watchStatus -Name "active_candidate_result_json" -Value $candidateResultLatestRel
        if ([string]::IsNullOrWhiteSpace([string]$watchStatus.active_precompute_mv_support_on)) {
            Set-ObjectProperty -Obj $watchStatus -Name "active_precompute_mv_support_on" -Value ([string]$candidateResult.precompute_mv_support_on)
        }
        if ([string]::IsNullOrWhiteSpace([string]$watchStatus.active_precompute_mv_support_region_mode)) {
            Set-ObjectProperty -Obj $watchStatus -Name "active_precompute_mv_support_region_mode" -Value ([string]$candidateResult.precompute_mv_support_region_mode)
        }
        if ([string]::IsNullOrWhiteSpace([string]$watchStatus.active_precompute_mv_support_fg_mask_source)) {
            Set-ObjectProperty -Obj $watchStatus -Name "active_precompute_mv_support_fg_mask_source" -Value ([string]$candidateResult.precompute_mv_support_fg_mask_source)
        }
        if ([string]::IsNullOrWhiteSpace([string]$watchStatus.active_precompute_mv_support_fg_erode_px)) {
            Set-ObjectProperty -Obj $watchStatus -Name "active_precompute_mv_support_fg_erode_px" -Value ([string]$candidateResult.precompute_mv_support_fg_erode_px)
        }
        if ([string]::IsNullOrWhiteSpace([string]$watchStatus.active_point_target_blend_by_mv_support)) {
            Set-ObjectProperty -Obj $watchStatus -Name "active_point_target_blend_by_mv_support" -Value ([string]$candidateResult.point_target_blend_by_mv_support)
        }
        if ([string]::IsNullOrWhiteSpace([string]$watchStatus.active_candidate_invalid_reason)) {
            Set-ObjectProperty -Obj $watchStatus -Name "active_candidate_invalid_reason" -Value ([string]$candidateResult.candidate_invalid_reason)
        }
        if ([string]::IsNullOrWhiteSpace([string]$watchStatus.active_precompute_source)) {
            Set-ObjectProperty -Obj $watchStatus -Name "active_precompute_source" -Value ([string]$candidateResult.precompute_source)
        }
    }
    if (($null -ne $autoloopStatus) -and [string]::IsNullOrWhiteSpace([string]$watchStatus.current_stage)) {
        Set-ObjectProperty -Obj $watchStatus -Name "current_stage" -Value ([string]$autoloopStatus.current_stage)
    }
    if (($null -ne $autoloopStatus) -and [string]::IsNullOrWhiteSpace([string]$watchStatus.active_precompute_mv_support_on)) {
        Set-ObjectProperty -Obj $watchStatus -Name "active_precompute_mv_support_on" -Value ([string]$autoloopStatus.active_precompute_mv_support_on)
    }
    if (($null -ne $autoloopStatus) -and [string]::IsNullOrWhiteSpace([string]$watchStatus.active_point_target_blend_by_mv_support)) {
        Set-ObjectProperty -Obj $watchStatus -Name "active_point_target_blend_by_mv_support" -Value ([string]$autoloopStatus.active_point_target_blend_by_mv_support)
    }
    Set-ObjectProperty -Obj $watchStatus -Name "note" -Value "paused_local_repair_no_live_chain"
    Write-JsonNoBom -Path $watchJsonPath -Obj $watchStatus
    $watchMd = @()
    $watchMd += "# Watch Ghost Outputs (paused/local-repair)"
    $watchMd += ""
    $watchMd += "- updated: $($watchStatus.updated_at)"
    $watchMd += "- current_stage: $([string]$watchStatus.current_stage)"
    $watchMd += "- active_candidate_result_json: $([string]$watchStatus.active_candidate_result_json)"
    $watchMd += "- active_candidate_invalid_reason: $([string]$watchStatus.active_candidate_invalid_reason)"
    $watchMd += "- active_precompute_source: $([string]$watchStatus.active_precompute_source)"
    $watchMd += "- active_precompute_mv_support_on: $([string]$watchStatus.active_precompute_mv_support_on)"
    $watchMd += "- active_precompute_mv_support_region_mode: $([string]$watchStatus.active_precompute_mv_support_region_mode)"
    $watchMd += "- active_precompute_mv_support_fg_mask_source: $([string]$watchStatus.active_precompute_mv_support_fg_mask_source)"
    $watchMd += "- active_precompute_mv_support_fg_erode_px: $([string]$watchStatus.active_precompute_mv_support_fg_erode_px)"
    $watchMd += "- active_point_target_blend_by_mv_support: $([string]$watchStatus.active_point_target_blend_by_mv_support)"
    $watchMd += "- pause_reason: $([string]$watchStatus.pause_reason)"
    $watchMd += "- note: $([string]$watchStatus.note)"
    Set-Content -Path $watchMdPath -Value ($watchMd -join "`n") -Encoding UTF8
}

Write-Host "[repair-paused] updated latest status files at $nowText"
