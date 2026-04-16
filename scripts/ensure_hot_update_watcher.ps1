param(
    [string]$RepoDir = "F:\vggt",
    [string]$TargetStage = "cycle001_stage1_strong",
    [string]$ForcedResumeCkpt = "",
    [int]$PollSec = 30,
    [int]$WatchHours = 12,
    [int]$WatcherPollSec = 20,
    [int]$WatcherMaxWaitMinutes = 120,
    [int]$WatcherStage1MaxWaitMinutes = 120,
    [int]$WatcherOtherStageMaxWaitMinutes = 90,
    [int]$WatcherStallMinutes = 30,
    [int]$WatcherNoChildStallMinutes = 20,
    [int]$WatcherCpuAssistWindowMinutes = 10,
    [int]$WatcherStartupGraceMinutes = 12,
    [int]$WatcherModalDoneStallMinutes = 6,
    [int]$WatcherWorkerAttachWindowMinutes = 20,
    [int]$WatcherWorkerKeepAliveMaxMinutes = 30,
    [bool]$WatcherEnforceSingleMainChain = $true,
    [int]$WatcherRedundantGuardPollEvery = 3,
    [string]$WatcherModalAppDescriptionRegex = "^vggt-zju-runner$",
    [switch]$EnableStageBoundaryRestart = $false,
    [int]$ForcedStopAfterHours = 12,
    [string]$MainFinalDeadline = "",
    [switch]$AutoContinueNextWindow = $true,
    [int]$CurrentWindowIndex = 1,
    [int]$MaxAutoContinueWindows = 999
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

$watcherMetaPath = "logs/modal_phase5/hot_update_autoloop_after_stage_latest.json"
$statusJsonPath = "logs/modal_phase5/ensure_hot_update_watcher_latest.json"
$statusMdPath = "logs/modal_phase5/ensure_hot_update_watcher_latest.md"
$statePath = "logs/modal_phase5/ensure_hot_update_watcher_state.json"
$launcherPath = "logs/modal_phase5/overnight_ghost_autoloop_launcher_latest.json"
$heartbeatPath = "logs/modal_phase5/overnight_ghost_autoloop_heartbeat_latest.json"
$finalJsonPath = "logs/modal_phase5/overnight_ghost_autoloop_12h_final_latest.json"
$finalMdPath = "logs/modal_phase5/overnight_ghost_autoloop_12h_final_latest.md"
$mentorPath = "logs/modal_phase5/mentor_update_latest.md"

function Read-JsonMaybe([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try { return (Get-Content $Path -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { return $null }
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

function Invoke-SingleEnsureWatcherGuard([string]$Reason = "loop") {
    $currentPid = [int]$PID
    $result = [ordered]@{
        current_pid = $currentPid
        keep_pid = $currentPid
        total = 0
        killed = 0
        should_exit = $false
        reason = $Reason
    }
    try {
        $repoPattern = [regex]::Escape([string]$RepoDir)
        $stagePattern = [regex]::Escape([string]$TargetStage)
        $rows = @(
            Get-CimInstance Win32_Process | Where-Object {
                $_.Name -match "powershell" -and
                $_.CommandLine -match "scripts/ensure_hot_update_watcher\.ps1" -and
                $_.CommandLine -match $repoPattern -and
                $_.CommandLine -match $stagePattern
            } | Sort-Object CreationDate -Descending
        )
        $result.total = @($rows).Count
        if ($rows.Count -gt 0) {
            $keepPid = [int]$rows[0].ProcessId
            $result.keep_pid = $keepPid
            foreach ($row in @($rows)) {
                $procId = [int]$row.ProcessId
                if ($procId -eq $keepPid) { continue }
                try {
                    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                    $result.killed = [int]$result.killed + 1
                    Write-Host "[ensure] single-instance guard stopped pid=$procId keep=$keepPid reason=$Reason"
                } catch {
                }
            }
            if ($keepPid -ne $currentPid) {
                $result.should_exit = $true
                Write-Host "[ensure] single-instance guard current pid=$currentPid is stale; keep=$keepPid reason=$Reason"
            }
        }
    } catch {
    }
    return [pscustomobject]$result
}

function Sanitize-TextForUtf8Log([string]$Text) {
    if ($null -eq $Text) { return "" }
    # Keep line breaks and tabs; strip non-printable control characters.
    return [regex]::Replace([string]$Text, "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "")
}

function Set-OrAddLaunchArg([string]$ArgsText, [string]$Flag, [string]$Value) {
    $base = [string]$ArgsText
    if ([string]::IsNullOrWhiteSpace($base)) {
        return "$Flag $Value"
    }
    $escapedFlag = [regex]::Escape($Flag)
    $pattern = "(?i)(^|\s)$escapedFlag\s+([^\s]+)"
    if ([regex]::IsMatch($base, $pattern)) {
        $updated = [regex]::Replace(
            $base,
            $pattern,
            {
                param($m)
                return "$($m.Groups[1].Value)$Flag $Value"
            },
            1
        )
        return $updated
    }
    return ($base.TrimEnd() + " $Flag $Value")
}

function Convert-LaunchArgsToArray([string]$ArgsText) {
    $raw = [string]$ArgsText
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return @()
    }
    $parts = New-Object System.Collections.Generic.List[string]
    $matches = [regex]::Matches($raw, '"[^"]*"|\S+')
    foreach ($m in $matches) {
        $token = [string]$m.Value
        if ($token.Length -ge 2 -and $token.StartsWith('"') -and $token.EndsWith('"')) {
            $token = $token.Substring(1, $token.Length - 2)
        }
        $parts.Add($token)
    }
    return $parts.ToArray()
}

function Normalize-BoolLaunchArgs([string]$ArgsText) {
    $next = [string]$ArgsText
    if ([string]::IsNullOrWhiteSpace($next)) { return "" }
    $boolFlags = @(
        "-Stage2HistoryQualityAware",
        "-StageEnableAbsoluteQualityGuard",
        "-PostRescueEnabled",
        "-Stage2DualLaneEnabled",
        "-EmergencyGhostShockEnabled",
        "-EmergencyRecoveryEnableHistoryQualityAware",
        "-EnableHistoricalSweepBootstrap",
        "-EnableABRouteOnStagnation",
        "-EnablePersistentCycleState"
    )
    foreach ($flag in @($boolFlags)) {
        $esc = [regex]::Escape($flag)
        $patternTrue = '(?i)(^|\s)' + $esc + '\s+(1|true|\$true)\b'
        $patternFalse = '(?i)(^|\s)' + $esc + '\s+(0|false|\$false)\b'
        $next = [regex]::Replace(
            $next,
            $patternTrue,
            { param($m) return "$($m.Groups[1].Value)$flag `$true" },
            1
        )
        $next = [regex]::Replace(
            $next,
            $patternFalse,
            { param($m) return "$($m.Groups[1].Value)$flag `$false" },
            1
        )
    }
    return $next
}

function Apply-TuneActionToLaunchArgs(
    [string]$ArgsText,
    [string]$TuneAction,
    [ref]$AppliedSummary
) {
    $AppliedSummary.Value = "none"
    $nextArgs = [string]$ArgsText
    $ta = [string]$TuneAction
    if ([string]::IsNullOrWhiteSpace($ta)) { return $nextArgs }

    if ($ta -match "point_mv_mask_soft_mix:\s*[-0-9\.]+\s*->\s*([-0-9\.]+)") {
        $v = [string]$Matches[1]
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-BasePointMvMaskSoftMix" -Value $v
        $AppliedSummary.Value = "BasePointMvMaskSoftMix=$v"
        return $nextArgs
    }
    if ($ta -match "point_mv_mask_soft_hit_thr:\s*[-0-9\.]+\s*->\s*([-0-9\.]+)") {
        $v = [string]$Matches[1]
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-BasePointMvMaskSoftHitThr" -Value $v
        $AppliedSummary.Value = "BasePointMvMaskSoftHitThr=$v"
        return $nextArgs
    }
    if ($ta -match "point_mv_stride:\s*[-0-9\.]+\s*->\s*([-0-9\.]+)") {
        $v = [string]$Matches[1]
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-BasePointMvStride" -Value $v
        $AppliedSummary.Value = "BasePointMvStride=$v"
        return $nextArgs
    }
    if ($ta -match "point_mv_depth_max_pairs:\s*[-0-9\.]+\s*->\s*([-0-9\.]+)") {
        $v = [string]$Matches[1]
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-BasePointMvDepthMaxPairs" -Value $v
        $AppliedSummary.Value = "BasePointMvDepthMaxPairs=$v"
        return $nextArgs
    }
    if ($ta -match "point_mv_depth_support_mode/floor:\s*[^ ]+\s*->\s*([a-zA-Z_]+)\/([-0-9\.]+)") {
        $mode = [string]$Matches[1]
        $floor = [string]$Matches[2]
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-BasePointMvDepthSupportMode" -Value $mode
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-BasePointMvDepthSupportFloor" -Value $floor
        $AppliedSummary.Value = "BasePointMvDepthSupportMode=$mode,BasePointMvDepthSupportFloor=$floor"
        return $nextArgs
    }
    if ($ta -match "point_mv_mask_min_tgt_fg_ratio:\s*[-0-9\.]+\s*->\s*([-0-9\.]+)") {
        $v = [string]$Matches[1]
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-BasePointMvMaskMinTgtFgRatio" -Value $v
        $AppliedSummary.Value = "BasePointMvMaskMinTgtFgRatio=$v"
        return $nextArgs
    }

    return $nextArgs
}

function Parse-DateMaybe([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return [datetime]::MinValue }
    try { return [datetime]::Parse($Value) } catch { return [datetime]::MinValue }
}

function Resolve-PreferredResumeCkpt([string]$Fallback) {
    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($Fallback)) {
        [void]$candidates.Add([string]$Fallback)
    }

    $globalBestPath = "logs/modal_phase5/ghost_global_best_latest.json"
    $globalBest = Read-JsonMaybe -Path $globalBestPath
    if ($globalBest -ne $null) {
        $gb = [string]$globalBest.best_ckpt
        if (-not [string]::IsNullOrWhiteSpace($gb)) {
            $candidates.Insert(0, $gb)
        }
    }

    $autoloopPath = "logs/modal_phase5/overnight_ghost_autoloop_latest.json"
    $autoloop = Read-JsonMaybe -Path $autoloopPath
    if ($autoloop -ne $null) {
        $gb2 = [string]$autoloop.global_best_ckpt
        if (-not [string]::IsNullOrWhiteSpace($gb2)) {
            $candidates.Insert(0, $gb2)
        }
        $curr = [string]$autoloop.current_resume_ckpt
        if (-not [string]::IsNullOrWhiteSpace($curr)) {
            [void]$candidates.Add($curr)
        }
    }

    foreach ($c in @($candidates)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$c)) {
            return [string]$c
        }
    }
    return ""
}

function Resolve-StageMaxWaitMinutes(
    [string]$StageName,
    [int]$Stage1Minutes,
    [int]$OtherMinutes,
    [int]$FallbackMinutes
) {
    $fallback = [Math]::Max(30, [int]$FallbackMinutes)
    if ([string]::IsNullOrWhiteSpace($StageName)) { return $fallback }
    $s = $StageName.Trim().ToLowerInvariant()
    if ($s -match "stage1") { return [Math]::Max(30, [int]$Stage1Minutes) }
    if ($s -match "stage[2-9]") { return [Math]::Max(30, [int]$OtherMinutes) }
    return $fallback
}

function To-DoubleOrNaN($Value) {
    try {
        if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) { return [double]::NaN }
        return [double]$Value
    } catch {
        return [double]::NaN
    }
}

function Test-ProcessCommandLineMatch(
    [int]$Pid,
    [string]$ExpectedPattern
) {
    if ($Pid -le 0) { return $false }
    try {
        $row = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f [int]$Pid) -ErrorAction Stop
    } catch {
        return $false
    }
    if ($null -eq $row) { return $false }
    $cmd = [string]$row.CommandLine
    if ([string]::IsNullOrWhiteSpace($cmd)) { return $false }
    return ($cmd -match $ExpectedPattern)
}

function Fmt-Num([double]$Value, [int]$Digits = 6) {
    if ([double]::IsNaN($Value)) { return "NaN" }
    return ("{0:F$Digits}" -f $Value)
}

function Start-Watcher(
    [int]$StallMinutes,
    [int]$NoChildStallMinutes,
    [int]$MaxWaitMinutes,
    [string]$ObservedStage,
    [string]$AdaptiveReason
) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $out = "logs/modal_phase5/hot_update_autoloop_after_stage_${ts}.out.log"
    $err = "logs/modal_phase5/hot_update_autoloop_after_stage_${ts}.err.log"
    $args = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/hot_update_autoloop_after_stage.ps1",
        "-RepoDir", $RepoDir,
        "-TargetStage", $TargetStage,
        "-ForcedResumeCkpt", $ForcedResumeCkpt,
        "-ForcedStopAfterHours", [string]$ForcedStopAfterHours,
        "-PollSec", [string]$WatcherPollSec,
        "-MaxWaitMinutes", [string]$MaxWaitMinutes,
        "-StallMinutes", [string]$StallMinutes,
        "-NoChildStallMinutes", [string]$NoChildStallMinutes,
        "-CpuAssistWindowMinutes", [string]$WatcherCpuAssistWindowMinutes,
        "-StartupGraceMinutes", [string]$WatcherStartupGraceMinutes,
        "-ModalDoneStallMinutes", [string]$WatcherModalDoneStallMinutes,
        "-WorkerAttachWindowMinutes", [string]$WatcherWorkerAttachWindowMinutes,
        "-WorkerKeepAliveMaxMinutes", [string]$WatcherWorkerKeepAliveMaxMinutes
    )
    if (-not [string]::IsNullOrWhiteSpace($MainFinalDeadline)) {
        $args += @("-MainFinalDeadline", $MainFinalDeadline)
    }
    if ($EnableStageBoundaryRestart) {
        $args += "-EnableStageBoundaryRestart"
    }
    Write-Host ("[ensure] start watcher args: " + ($args -join " "))
    $p = Start-Process -FilePath "powershell.exe" -ArgumentList $args -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    $meta = [ordered]@{
        started_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        pid = $p.Id
        stdout = $out
        stderr = $err
        target_stage = $TargetStage
        forced_resume_ckpt = $ForcedResumeCkpt
        stall_minutes = $StallMinutes
        no_child_stall_minutes = $NoChildStallMinutes
        max_wait_minutes = $MaxWaitMinutes
        observed_stage_at_launch = $ObservedStage
        worker_attach_window_minutes = $WatcherWorkerAttachWindowMinutes
        worker_keepalive_max_minutes = $WatcherWorkerKeepAliveMaxMinutes
        enforce_single_main_chain = [bool]$WatcherEnforceSingleMainChain
        redundant_guard_poll_every = [int]$WatcherRedundantGuardPollEvery
        modal_app_description_regex = $WatcherModalAppDescriptionRegex
        enable_stage_boundary_restart = [bool]$EnableStageBoundaryRestart
        forced_stop_after_hours = $ForcedStopAfterHours
        note = "auto-managed by ensure_hot_update_watcher ($AdaptiveReason)"
    }
    Write-JsonNoBom -Path $watcherMetaPath -Obj $meta
    return $meta
}

function Resolve-AdaptiveThreshold(
    [int]$BaseStallMinutes,
    [int]$BaseNoChildStallMinutes,
    [int]$ConsecutiveStalledRestarts,
    [bool]$HasAutoloopArtifacts
) {
    $stall = [Math]::Max(10, [int]$BaseStallMinutes)
    $noChild = [Math]::Max(10, [int]$BaseNoChildStallMinutes)
    $reason = "base"
    if ($HasAutoloopArtifacts) {
        return [pscustomobject]@{
            stall = $stall
            no_child = $noChild
            reason = "base_artifacts_ready"
        }
    }
    if ($ConsecutiveStalledRestarts -ge 6) {
        return [pscustomobject]@{
            stall = [Math]::Min([Math]::Max($stall, 45), 60)
            no_child = [Math]::Min([Math]::Max($noChild, 15), 25)
            reason = "adaptive_recycle_high_stall_ge6"
        }
    }
    if ($ConsecutiveStalledRestarts -ge 3) {
        return [pscustomobject]@{
            stall = [Math]::Max($stall, 90)
            no_child = [Math]::Max($noChild, 30)
            reason = "adaptive_consecutive_stall_ge3"
        }
    }
    if ($ConsecutiveStalledRestarts -ge 2) {
        return [pscustomobject]@{
            stall = [Math]::Max($stall, 75)
            no_child = [Math]::Max($noChild, 25)
            reason = "adaptive_consecutive_stall_ge2"
        }
    }
    return [pscustomobject]@{
        stall = $stall
        no_child = $noChild
        reason = "base_consecutive_stall_lt2"
    }
}

function Bootstrap-ConsecutiveStallsFromMentor([string]$Path) {
    if (-not (Test-Path $Path)) { return 0 }
    try {
        $all = @(Get-Content $Path -ErrorAction Stop)
        if ($all.Count -le 0) { return 0 }
        $tailStart = [Math]::Max(0, $all.Count - 600)
        $tail = @($all[$tailStart..($all.Count - 1)])
        $cnt = 0
        for ($i = $tail.Count - 1; $i -ge 0; $i--) {
            $line = [string]$tail[$i]
            if ($line -match "trigger:\s*stalled_no_activity_") {
                $cnt += 1
                continue
            }
            if ($line -match "trigger:") {
                break
            }
        }
        return $cnt
    } catch {
        return 0
    }
}

$launcherForT0 = Read-JsonMaybe -Path $launcherPath
$statusSeed = Read-JsonMaybe -Path $statusJsonPath
$finalSeed = Read-JsonMaybe -Path $finalJsonPath
$initialForcedResume = [string]$ForcedResumeCkpt
$forcedResumeSource = "arg_or_empty"
$resolvedInitResume = Resolve-PreferredResumeCkpt -Fallback $initialForcedResume
if (-not [string]::IsNullOrWhiteSpace($resolvedInitResume)) {
    $ForcedResumeCkpt = $resolvedInitResume
    if ($resolvedInitResume -ne $initialForcedResume) {
        $forcedResumeSource = "global_best_bootstrap"
    }
}
$t0 = Get-Date
$t0Source = "ensure_start_time"
if ($launcherForT0 -ne $null) {
    $launcherT0 = Parse-DateMaybe -Value ([string]$launcherForT0.started_at)
    if ($launcherT0 -ne [datetime]::MinValue) {
        $t0 = $launcherT0
        $t0Source = "launcher.started_at"
    }
}
$deadline = $t0.AddHours([Math]::Max(1, [int]$WatchHours))
$windowSeeds = @()
foreach ($seed in @(
        [pscustomobject]@{ obj = $statusSeed; src = "status_json" },
        [pscustomobject]@{ obj = $finalSeed; src = "final_json" }
    )) {
    if ($null -eq $seed.obj) { continue }
    $seedT0 = Parse-DateMaybe -Value ([string]$seed.obj.t0)
    $seedDeadline = Parse-DateMaybe -Value ([string]$seed.obj.deadline)
    if (($seedT0 -eq [datetime]::MinValue) -or ($seedDeadline -eq [datetime]::MinValue)) { continue }
    if ($seedDeadline -le $seedT0) { continue }
    $windowSeeds += [pscustomobject]@{
        t0 = $seedT0
        deadline = $seedDeadline
        src = [string]$seed.src
    }
}
if ($windowSeeds.Count -gt 0) {
    $nowWindow = Get-Date
    $activeSeeds = @($windowSeeds | Where-Object { $_.deadline -gt $nowWindow.AddMinutes(-1) })
    if ($activeSeeds.Count -gt 0) {
        $pick = $activeSeeds | Sort-Object t0 | Select-Object -First 1
        $t0 = [datetime]$pick.t0
        $deadline = [datetime]$pick.deadline
        $t0Source = "$([string]$pick.src).persisted_t0"
    }
}
$mainFinalDeadlineDt = Parse-DateMaybe -Value ([string]$MainFinalDeadline)
if ($mainFinalDeadlineDt -ne [datetime]::MinValue) {
    $deadline = $mainFinalDeadlineDt
    if (($t0 -eq [datetime]::MinValue) -or ($t0 -gt $deadline)) {
        $t0 = $deadline.AddHours(-1.0 * [Math]::Max(1, [int]$WatchHours))
        $t0Source = "derived_from_main_final_deadline"
    } elseif ([string]::IsNullOrWhiteSpace($t0Source)) {
        $t0Source = "main_final_deadline"
    }
}
$seedNow = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
$needSeedRunning = $false
if ($null -eq $finalSeed) {
    $needSeedRunning = $true
} else {
    $seedUpdated = Parse-DateMaybe -Value ([string]$finalSeed.updated_at)
    if (($seedUpdated -eq [datetime]::MinValue) -or ($seedUpdated -lt $t0.AddMinutes(-1))) {
        $needSeedRunning = $true
    }
}
if ($needSeedRunning) {
    $seedObj = [ordered]@{
        updated_at = $seedNow
        state = "running_current_window"
        t0 = $t0.ToString("yyyy-MM-ddTHH:mm:ss")
        deadline = $deadline.ToString("yyyy-MM-ddTHH:mm:ss")
        watch_hours = $WatchHours
        main_final_deadline = $MainFinalDeadline
        current_window_index = $CurrentWindowIndex
        auto_continue_next_window = [bool]$AutoContinueNextWindow
        max_auto_continue_windows = $MaxAutoContinueWindows
    }
    Write-JsonNoBom -Path $finalJsonPath -Obj $seedObj
    $seedMd = @()
    $seedMd += "# 过夜 12h 最终报告（窗口进行中）"
    $seedMd += ""
    $seedMd += "- updated: $($seedObj.updated_at)"
    $seedMd += "- state: $($seedObj.state)"
    $seedMd += "- t0: $($seedObj.t0)"
    $seedMd += "- deadline: $($seedObj.deadline)"
    $seedMd += "- watch_hours: $($seedObj.watch_hours)"
    $seedMd += "- main_final_deadline: $($seedObj.main_final_deadline)"
    $seedMd += "- current_window_index: $($seedObj.current_window_index)"
    $seedMd += "- auto_continue_next_window: $($seedObj.auto_continue_next_window)"
    Set-Content -Path $finalMdPath -Value ($seedMd -join "`n") -Encoding UTF8
}
$restartCount = 0
$lastAction = "none"
$baseWatcherStallMinutes = [int]$WatcherStallMinutes
$baseWatcherNoChildStallMinutes = [int]$WatcherNoChildStallMinutes

$watchState = Read-JsonMaybe -Path $statePath
if ($null -eq $watchState) {
    $bootstrapStalls = Bootstrap-ConsecutiveStallsFromMentor -Path $mentorPath
    $watchState = [pscustomobject]@{
        consecutive_stalled_restarts = $bootstrapStalls
        last_main_pid = 0
        last_launcher_started_at = ""
        last_launcher_reason = ""
        last_updated_at = ""
        bootstrap_stalls = $bootstrapStalls
    }
}

$instanceGuard = Invoke-SingleEnsureWatcherGuard -Reason "startup"
if ([bool]$instanceGuard.should_exit) {
    exit 0
}

while ((Get-Date) -lt $deadline) {
    $instanceGuard = Invoke-SingleEnsureWatcherGuard -Reason "loop"
    if ([bool]$instanceGuard.should_exit) {
        break
    }

    $launcher = Read-JsonMaybe -Path $launcherPath
    $hb = Read-JsonMaybe -Path $heartbeatPath
    $mainPid = 0
    $launcherStartedAt = ""
    $launcherStartedAtDt = [datetime]::MinValue
    $launcherReason = ""
    if ($launcher -ne $null) {
        $mainPid = [int]$launcher.pid
        $launcherStartedAt = [string]$launcher.started_at
        $launcherStartedAtDt = Parse-DateMaybe -Value $launcherStartedAt
        $launcherReason = [string]$launcher.restart_reason
    }
    if ($launcherStartedAtDt -ne [datetime]::MinValue) {
        if ($launcherStartedAtDt -gt $t0.AddMinutes(1)) {
            # Keep a fixed 12h window from initial T0. Restarts are recorded,
            # but must not shift deadline and effectively extend runtime budget.
            Write-Host "[ensure] launcher started_at advanced but fixed-window mode keeps original t0/deadline (launcher_started_at=$($launcherStartedAtDt.ToString('yyyy-MM-ddTHH:mm:ss')) t0=$($t0.ToString('yyyy-MM-ddTHH:mm:ss')) deadline=$($deadline.ToString('yyyy-MM-ddTHH:mm:ss')))"
        }
    }
    $observedStage = $TargetStage
    if ($hb -ne $null) {
        $hbStage = [string]$hb.stage
        if (-not [string]::IsNullOrWhiteSpace($hbStage)) {
            $observedStage = $hbStage
        }
    }
    $effectiveMaxWaitMinutes = Resolve-StageMaxWaitMinutes `
        -StageName $observedStage `
        -Stage1Minutes $WatcherStage1MaxWaitMinutes `
        -OtherMinutes $WatcherOtherStageMaxWaitMinutes `
        -FallbackMinutes $WatcherMaxWaitMinutes

    $lastMainPid = 0
    $lastLauncherStartedAt = ""
    $lastConsecutiveStalls = 0
    if ($watchState -ne $null) {
        try { $lastMainPid = [int]$watchState.last_main_pid } catch {}
        $lastLauncherStartedAt = [string]$watchState.last_launcher_started_at
        try { $lastConsecutiveStalls = [int]$watchState.consecutive_stalled_restarts } catch {}
    }

    $mainChanged = $false
    if ($mainPid -gt 0) {
        if (($mainPid -ne $lastMainPid) -or ($launcherStartedAt -ne $lastLauncherStartedAt)) {
            $mainChanged = $true
        }
    }
    if ($mainChanged) {
        if ($launcherReason -like "hot_update_after_stage_boundary:stalled_no_activity_*") {
            $lastConsecutiveStalls += 1
        } else {
            $lastConsecutiveStalls = 0
        }
    }

    $hasAutoloopArtifacts = (Test-Path "logs/modal_phase5/ghost_autoloop_latest.csv") -or (Test-Path "logs/modal_phase5/overnight_ghost_autoloop_latest.json")
    if ($hasAutoloopArtifacts) {
        $lastConsecutiveStalls = 0
    }

    $adaptive = Resolve-AdaptiveThreshold `
        -BaseStallMinutes $baseWatcherStallMinutes `
        -BaseNoChildStallMinutes $baseWatcherNoChildStallMinutes `
        -ConsecutiveStalledRestarts $lastConsecutiveStalls `
        -HasAutoloopArtifacts $hasAutoloopArtifacts

    $resolvedLoopResume = Resolve-PreferredResumeCkpt -Fallback $ForcedResumeCkpt
    if ((-not [string]::IsNullOrWhiteSpace($resolvedLoopResume)) -and ($resolvedLoopResume -ne $ForcedResumeCkpt)) {
        $ForcedResumeCkpt = $resolvedLoopResume
        $forcedResumeSource = "global_best_refresh"
    }

    $watchState = [pscustomobject]@{
        consecutive_stalled_restarts = $lastConsecutiveStalls
        last_main_pid = $mainPid
        last_launcher_started_at = $launcherStartedAt
        last_launcher_reason = $launcherReason
        last_updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    }
    Write-JsonNoBom -Path $statePath -Obj $watchState

    $meta = Read-JsonMaybe -Path $watcherMetaPath
    $state = "missing"
    $watcherPid = 0
    $stdout = ""
    $stderr = ""

    $metaStall = -1
    $metaNoChild = -1
    $metaMaxWait = -1
    $metaForcedResume = ""
    $metaEnforceSingleMainChain = $null
    $metaRedundantGuardPollEvery = -1
    $metaModalAppDescriptionRegex = ""
    if ($meta -ne $null) {
        $watcherPid = [int]$meta.pid
        $stdout = [string]$meta.stdout
        $stderr = [string]$meta.stderr
        $metaForcedResume = [string]$meta.forced_resume_ckpt
        try { $metaStall = [int]$meta.stall_minutes } catch {}
        try { $metaNoChild = [int]$meta.no_child_stall_minutes } catch {}
        try { $metaMaxWait = [int]$meta.max_wait_minutes } catch {}
        try { $metaEnforceSingleMainChain = [bool]$meta.enforce_single_main_chain } catch { $metaEnforceSingleMainChain = $null }
        try { $metaRedundantGuardPollEvery = [int]$meta.redundant_guard_poll_every } catch { $metaRedundantGuardPollEvery = -1 }
        $metaModalAppDescriptionRegex = [string]$meta.modal_app_description_regex
        $watcherAlive = Test-ProcessCommandLineMatch -Pid $watcherPid -ExpectedPattern "scripts/hot_update_autoloop_after_stage\.ps1"
        if ($watcherAlive) {
            $state = "alive"
        } else {
            $state = "dead"
        }
    }

    $needParamRotate = $false
    if ($state -eq "alive") {
        if (
            ($metaStall -ne [int]$adaptive.stall) -or
            ($metaNoChild -ne [int]$adaptive.no_child) -or
            ($metaMaxWait -ne [int]$effectiveMaxWaitMinutes) -or
            ($metaForcedResume -ne [string]$ForcedResumeCkpt)
        ) {
            $needParamRotate = $true
            try {
                Stop-Process -Id $watcherPid -Force -ErrorAction SilentlyContinue
            } catch {
            }
            $state = "dead"
            $lastAction = "rotate_watcher_for_param_change_$watcherPid"
        }
    }

    if ($state -ne "alive") {
        $meta = Start-Watcher `
            -StallMinutes ([int]$adaptive.stall) `
            -NoChildStallMinutes ([int]$adaptive.no_child) `
            -MaxWaitMinutes ([int]$effectiveMaxWaitMinutes) `
            -ObservedStage $observedStage `
            -AdaptiveReason ([string]$adaptive.reason)
        $watcherPid = [int]$meta.pid
        $stdout = [string]$meta.stdout
        $stderr = [string]$meta.stderr
        $state = $(if ($needParamRotate) { "restarted_after_param_rotate" } else { "restarted" })
        $restartCount += 1
        $lastAction = "start_watcher_pid_$watcherPid"
    } else {
        $lastAction = "keep_alive_pid_$watcherPid"
    }

    $status = [ordered]@{
        updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        t0 = $t0.ToString("yyyy-MM-ddTHH:mm:ss")
        t0_source = $t0Source
        deadline = $deadline.ToString("yyyy-MM-ddTHH:mm:ss")
        watch_hours = $WatchHours
        main_final_deadline = $MainFinalDeadline
        state = $state
        watcher_pid = $watcherPid
        watcher_stdout = $stdout
        watcher_stderr = $stderr
        restart_count = $restartCount
        last_action = $lastAction
        target_stage = $TargetStage
        forced_resume_ckpt = $ForcedResumeCkpt
        forced_resume_ckpt_source = $forcedResumeSource
        base_watcher_stall_minutes = $baseWatcherStallMinutes
        base_watcher_no_child_stall_minutes = $baseWatcherNoChildStallMinutes
        watcher_stall_minutes = [int]$adaptive.stall
        watcher_no_child_stall_minutes = [int]$adaptive.no_child
        watcher_cpu_assist_window_minutes = $WatcherCpuAssistWindowMinutes
        watcher_startup_grace_minutes = $WatcherStartupGraceMinutes
        watcher_modal_done_stall_minutes = $WatcherModalDoneStallMinutes
        watcher_worker_attach_window_minutes = $WatcherWorkerAttachWindowMinutes
        watcher_worker_keepalive_max_minutes = $WatcherWorkerKeepAliveMaxMinutes
        watcher_enforce_single_main_chain = [bool]$WatcherEnforceSingleMainChain
        watcher_redundant_guard_poll_every = [int]$WatcherRedundantGuardPollEvery
        watcher_modal_app_description_regex = $WatcherModalAppDescriptionRegex
        enable_stage_boundary_restart = [bool]$EnableStageBoundaryRestart
        watcher_stage1_max_wait_minutes = $WatcherStage1MaxWaitMinutes
        watcher_other_stage_max_wait_minutes = $WatcherOtherStageMaxWaitMinutes
        watcher_default_max_wait_minutes = $WatcherMaxWaitMinutes
        watcher_effective_max_wait_minutes = $effectiveMaxWaitMinutes
        observed_stage = $observedStage
        forced_stop_after_hours = $ForcedStopAfterHours
        adaptive_reason = [string]$adaptive.reason
        observed_main_pid = $mainPid
        observed_launcher_restart_reason = $launcherReason
        consecutive_stalled_restarts = $lastConsecutiveStalls
        has_autoloop_artifacts = $hasAutoloopArtifacts
        ensure_single_instance_total = [int]$instanceGuard.total
        ensure_single_instance_killed = [int]$instanceGuard.killed
        ensure_single_instance_keep_pid = [int]$instanceGuard.keep_pid
        ensure_single_instance_should_exit = [bool]$instanceGuard.should_exit
    }
    Write-JsonNoBom -Path $statusJsonPath -Obj $status

    $md = @()
    $md += "# 热更新监督器状态"
    $md += ""
    $md += "- updated: $($status.updated_at)"
    $md += "- t0: $($status.t0) ($($status.t0_source))"
    $md += "- deadline: $($status.deadline)"
    $md += "- watch_hours: $($status.watch_hours)"
    $md += "- main_final_deadline: $($status.main_final_deadline)"
    $md += "- state: $($status.state)"
    $md += "- watcher_pid: $($status.watcher_pid)"
    $md += "- restart_count: $($status.restart_count)"
    $md += "- last_action: $($status.last_action)"
    $md += "- target_stage: $($status.target_stage)"
    $md += "- base_watcher_stall_minutes: $($status.base_watcher_stall_minutes)"
    $md += "- base_watcher_no_child_stall_minutes: $($status.base_watcher_no_child_stall_minutes)"
    $md += "- watcher_stall_minutes: $($status.watcher_stall_minutes)"
    $md += "- watcher_no_child_stall_minutes: $($status.watcher_no_child_stall_minutes)"
    $md += "- watcher_cpu_assist_window_minutes: $($status.watcher_cpu_assist_window_minutes)"
    $md += "- watcher_startup_grace_minutes: $($status.watcher_startup_grace_minutes)"
    $md += "- watcher_modal_done_stall_minutes: $($status.watcher_modal_done_stall_minutes)"
    $md += "- watcher_worker_attach_window_minutes: $($status.watcher_worker_attach_window_minutes)"
    $md += "- watcher_worker_keepalive_max_minutes: $($status.watcher_worker_keepalive_max_minutes)"
    $md += "- enable_stage_boundary_restart: $($status.enable_stage_boundary_restart)"
    $md += "- watcher_stage1_max_wait_minutes: $($status.watcher_stage1_max_wait_minutes)"
    $md += "- watcher_other_stage_max_wait_minutes: $($status.watcher_other_stage_max_wait_minutes)"
    $md += "- watcher_default_max_wait_minutes: $($status.watcher_default_max_wait_minutes)"
    $md += "- watcher_effective_max_wait_minutes: $($status.watcher_effective_max_wait_minutes)"
    $md += "- observed_stage: $($status.observed_stage)"
    $md += "- forced_stop_after_hours: $($status.forced_stop_after_hours)"
    $md += "- adaptive_reason: $($status.adaptive_reason)"
    $md += "- observed_main_pid: $($status.observed_main_pid)"
    $md += "- observed_launcher_restart_reason: $($status.observed_launcher_restart_reason)"
    $md += "- consecutive_stalled_restarts: $($status.consecutive_stalled_restarts)"
    $md += "- has_autoloop_artifacts: $($status.has_autoloop_artifacts)"
    $md += "- ensure_single_instance_total: $($status.ensure_single_instance_total)"
    $md += "- ensure_single_instance_killed: $($status.ensure_single_instance_killed)"
    $md += "- ensure_single_instance_keep_pid: $($status.ensure_single_instance_keep_pid)"
    $md += "- ensure_single_instance_should_exit: $($status.ensure_single_instance_should_exit)"
    $md += "- watcher_stdout: $($status.watcher_stdout)"
    $md += "- watcher_stderr: $($status.watcher_stderr)"
    Set-Content -Path $statusMdPath -Value ($md -join "`n") -Encoding UTF8

    Start-Sleep -Seconds ([Math]::Max(10, [int]$PollSec))
}

$finalLauncher = Read-JsonMaybe -Path $launcherPath
$finalWatcherMeta = Read-JsonMaybe -Path $watcherMetaPath
$finalStatus = Read-JsonMaybe -Path "logs/modal_phase5/overnight_ghost_autoloop_latest.json"
$finalMainPid = 0
$finalWatcherPid = 0
$finalMainStopped = $false
$finalWatcherStopped = $false
if ($finalLauncher -ne $null) {
    try { $finalMainPid = [int]$finalLauncher.pid } catch {}
}
if ($finalWatcherMeta -ne $null) {
    try { $finalWatcherPid = [int]$finalWatcherMeta.pid } catch {}
}
if ($finalWatcherPid -gt 0) {
    $p = Get-Process -Id $finalWatcherPid -ErrorAction SilentlyContinue
    if ($p -ne $null) {
        try {
            Stop-Process -Id $finalWatcherPid -Force -ErrorAction SilentlyContinue
            $finalWatcherStopped = $true
        } catch {
        }
    } else {
        $finalWatcherStopped = $true
    }
}
if ($finalMainPid -gt 0) {
    $p = Get-Process -Id $finalMainPid -ErrorAction SilentlyContinue
    if ($p -ne $null) {
        try {
            Stop-Process -Id $finalMainPid -Force -ErrorAction SilentlyContinue
            $finalMainStopped = $true
        } catch {
        }
    } else {
        $finalMainStopped = $true
    }
}

$bestGhost = [double]::NaN
$bestPsnr = [double]::NaN
$bestSsim = [double]::NaN
$bestWl1 = [double]::NaN
$bestCycle = 0
$bestStage = ""
$bestComparePng = ""
$bestVisualConclusion = ""
$stopByStagnation = $false
$stopByRegression = $false
$nextTuneAction = ""
$historyCount = 0
$globalBestCkpt = ""
$globalBestGeom = ""
$statusSource = "none"
$trajectory = @()

if ($finalStatus -ne $null) {
    $statusSource = "logs/modal_phase5/overnight_ghost_autoloop_latest.json"
    $bestGhost = To-DoubleOrNaN($finalStatus.global_best_ghost)
    $bestPsnr = To-DoubleOrNaN($finalStatus.global_best_psnr)
    $bestWl1 = To-DoubleOrNaN($finalStatus.global_best_wl1)
    $globalBestCkpt = [string]$finalStatus.global_best_ckpt
    $globalBestGeom = [string]$finalStatus.global_best_geom
    $nextTuneAction = [string]$finalStatus.next_cycle_tune_action
    try { $stopByStagnation = [bool]$finalStatus.should_stop_by_stagnation } catch {}
    try { $stopByRegression = [bool]$finalStatus.should_stop_by_regression } catch {}

    $hist = @()
    try { $hist = @($finalStatus.history) } catch {}
    $historyCount = $hist.Count
    $bestCycleObj = $null
    foreach ($h in @($hist)) {
        $cycleId = 0
        try { $cycleId = [int]$h.cycle } catch {}
        $cg = To-DoubleOrNaN($h.cycle_best_ghost)
        $cp = To-DoubleOrNaN($h.cycle_best_psnr)
        $trajectory += [pscustomobject]@{
            cycle = $cycleId
            best_stage = [string]$h.cycle_best_stage
            best_ghost = $cg
            best_psnr = $cp
            regressed = [bool]$h.cycle_regressed
            tune_action_next = [string]$h.tune_action_next
            compare_png = [string]$h.cycle_compare_png
            visual_conclusion = [string]$h.cycle_visual_conclusion
            regress_reason = [string]$h.cycle_regress_reason
        }
        if (-not [double]::IsNaN($cg)) {
            $pick = $false
            if ($null -eq $bestCycleObj) {
                $pick = $true
            } elseif ($cg -lt $bestGhost) {
                $pick = $true
            } elseif (([Math]::Abs($cg - $bestGhost) -lt 1e-9) -and (-not [double]::IsNaN($cp)) -and ($cp -gt $bestPsnr)) {
                $pick = $true
            }
            if ($pick) {
                $bestCycleObj = $h
                $bestGhost = $cg
                $bestPsnr = $cp
                $bestCycle = $cycleId
            }
        }
    }

    if ($bestCycleObj -ne $null) {
        $bestStage = [string]$bestCycleObj.cycle_best_stage
        $bestComparePng = [string]$bestCycleObj.cycle_compare_png
        $bestVisualConclusion = [string]$bestCycleObj.cycle_visual_conclusion
        $detail = $null
        foreach ($k in @("stage1", "stage2", "stage3", "stage4", "stage5")) {
            $s = $bestCycleObj.$k
            if ($null -eq $s) { continue }
            if ([string]$s.stage -eq $bestStage) {
                $detail = $s
                break
            }
        }
        if ($detail -ne $null) {
            $dg = To-DoubleOrNaN($detail.ghost)
            $dp = To-DoubleOrNaN($detail.psnr)
            $ds = To-DoubleOrNaN($detail.ssim)
            $dw = To-DoubleOrNaN($detail.wl1)
            if (-not [double]::IsNaN($dg)) { $bestGhost = $dg }
            if (-not [double]::IsNaN($dp)) { $bestPsnr = $dp }
            if (-not [double]::IsNaN($ds)) { $bestSsim = $ds }
            if (-not [double]::IsNaN($dw)) { $bestWl1 = $dw }
        }
    }
}

if (($historyCount -le 0 -or [double]::IsNaN($bestGhost)) -and (Test-Path "logs/modal_phase5/ghost_autoloop_latest.csv")) {
    $autoloopCsv = "logs/modal_phase5/ghost_autoloop_latest.csv"
    try {
        $autoInfo = Get-Item $autoloopCsv -ErrorAction SilentlyContinue
        if (($null -ne $autoInfo) -and ($autoInfo.LastWriteTime -lt $t0.AddSeconds(-2))) {
            throw "autoloop_latest_before_t0"
        }
        $rows = @(
            Import-Csv $autoloopCsv |
                Where-Object { [int]$_.rc -eq 0 } |
                Where-Object {
                    $rowTs = Parse-DateMaybe -Value ([string]$_.updated_at)
                    if ($rowTs -eq [datetime]::MinValue) { return $true }
                    return ($rowTs -ge $t0.AddMinutes(-1))
                }
        )
        if ($rows.Count -gt 0) {
            $statusSource = $autoloopCsv
            $pick = @(
                $rows |
                    Sort-Object {
                        $g = To-DoubleOrNaN($_.ghost)
                        if ([double]::IsNaN($g)) { return [double]::PositiveInfinity }
                        return $g
                    }, {
                        -1.0 * (To-DoubleOrNaN($_.psnr))
                    } |
                    Select-Object -First 1
            )[0]
            if ($pick -ne $null) {
                $bestGhost = To-DoubleOrNaN($pick.ghost)
                $bestPsnr = To-DoubleOrNaN($pick.psnr)
                $bestSsim = To-DoubleOrNaN($pick.ssim)
                $bestWl1 = To-DoubleOrNaN($pick.wl1)
                try { $bestCycle = [int]$pick.cycle } catch {}
                $bestStage = [string]$pick.stage
                if ([string]::IsNullOrWhiteSpace($nextTuneAction)) { $nextTuneAction = [string]$pick.tune_action_next }
                if ([string]::IsNullOrWhiteSpace($globalBestCkpt)) { $globalBestCkpt = [string]$pick.best_ckpt }
                if ([string]::IsNullOrWhiteSpace($globalBestGeom)) { $globalBestGeom = [string]$pick.best_geom }
                if ([string]::IsNullOrWhiteSpace($bestComparePng)) { $bestComparePng = [string]$pick.cycle_compare_png }
            }

            if ($trajectory.Count -le 0) {
                $cycleGroups = @(
                    $rows |
                        Group-Object cycle |
                        Sort-Object {
                            try { return [int]$_.Name } catch { return 0 }
                        }
                )
                foreach ($grp in @($cycleGroups)) {
                    $cycleRows = @($grp.Group)
                    if ($cycleRows.Count -le 0) { continue }
                    $bestRow = @(
                        $cycleRows |
                            Sort-Object {
                                $g = To-DoubleOrNaN($_.ghost)
                                if ([double]::IsNaN($g)) { return [double]::PositiveInfinity }
                                return $g
                            }, {
                                -1.0 * (To-DoubleOrNaN($_.psnr))
                            } |
                            Select-Object -First 1
                    )[0]
                    if ($bestRow -eq $null) { continue }
                    $cycleId = 0
                    $regressed = $false
                    try { $cycleId = [int]$bestRow.cycle } catch {}
                    try { $regressed = [bool]$bestRow.cycle_regressed } catch {}
                    $trajectory += [pscustomobject]@{
                        cycle = $cycleId
                        best_stage = [string]$bestRow.stage
                        best_ghost = To-DoubleOrNaN($bestRow.ghost)
                        best_psnr = To-DoubleOrNaN($bestRow.psnr)
                        regressed = $regressed
                        tune_action_next = [string]$bestRow.tune_action_next
                        compare_png = [string]$bestRow.cycle_compare_png
                        visual_conclusion = ""
                        regress_reason = [string]$bestRow.cycle_regress_reason
                    }
                }
                if ($cycleGroups.Count -gt 0) { $historyCount = $cycleGroups.Count }
            } elseif ($historyCount -le 0) {
                $uniqCycles = @($rows | Select-Object -ExpandProperty cycle | Sort-Object -Unique)
                if ($uniqCycles.Count -gt 0) { $historyCount = $uniqCycles.Count }
            }
        }
    } catch {
    }
}

if ([double]::IsNaN($bestGhost)) {
    $sweepCsv = "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv"
    if (Test-Path $sweepCsv) {
        try {
            $sweepInfo = Get-Item $sweepCsv -ErrorAction SilentlyContinue
            if (($null -ne $sweepInfo) -and ($sweepInfo.LastWriteTime -lt $t0.AddSeconds(-2))) {
                throw "sweep_latest_before_t0"
            }
            $rows = @(
                Import-Csv $sweepCsv |
                    Where-Object { [int]$_.exit_code -eq 0 }
            )
            if ($rows.Count -gt 0) {
                $pick = @(
                    $rows |
                        Sort-Object {
                            To-DoubleOrNaN($_.ghost_score_mean)
                        }, {
                            -1.0 * (To-DoubleOrNaN($_.mean_PSNR))
                        } |
                        Select-Object -First 1
                )[0]
                if ($pick -ne $null) {
                    $statusSource = $sweepCsv
                    $bestGhost = To-DoubleOrNaN($pick.ghost_score_mean)
                    $bestPsnr = To-DoubleOrNaN($pick.mean_PSNR)
                    $bestSsim = To-DoubleOrNaN($pick.mean_SSIM)
                    $bestWl1 = To-DoubleOrNaN($pick.mean_weighted_L1)
                    $bestStage = "sweep_fallback"
                }
            }
        } catch {
        }
    }
}

$recommendStopCompute = $stopByStagnation -or $stopByRegression
$decision = $(if ($recommendStopCompute) { "stop_burn_compute" } else { "continue_next_12h_window_if_budget_allows" })
$decisionReason = ""
$nextRoute = @()
$minimalValidationPlan = @()
if ($recommendStopCompute) {
    if ($stopByRegression -and $stopByStagnation) {
        $decisionReason = "stagnation_and_regression_guard_triggered"
    } elseif ($stopByRegression) {
        $decisionReason = "regression_guard_triggered"
    } else {
        $decisionReason = "no_substantial_ghost_improve_for_multiple_cycles"
    }
    $nextRoute = @(
        "提高前景一致性约束强度，并加强 point-mask 与 depth-support 耦合。",
        "在 step0/1/2 引入短窗口时序一致性，压制双峰拖尾伪影。",
        "提高相机-点云重投影约束权重，并强化离群区域加权。"
    )
    $minimalValidationPlan = @(
        "A/B-1：仅增强前景一致性；N=40，max_steps=80。",
        "A/B-2：仅增强时序一致性；N=40，max_steps=80。",
        "A/B-3：仅增强相机-点云重投影；N=40，max_steps=80。",
        "三组均输出 ghost/PSNR/SSIM/wL1 与对比图，用于决策。"
    )
} else {
    $decisionReason = "within_12h_window_and_no_stop_guard"
    $nextRoute = @(
        "继续单变量小步：soft_mix -> soft_hit_thr -> stride -> max_pairs -> support_mode/floor -> min_tgt_fg_ratio。"
    )
    $minimalValidationPlan = @(
        "下一窗口保持同数据与评估协议，验证 ghost 稳定下降 >= 0.02。"
    )
}

$trajectoryRecent = @()
if ($trajectory.Count -gt 0) {
    $trajectoryRecent = @($trajectory | Select-Object -Last 10)
}

$finalObj = [ordered]@{
    updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    t0 = $t0.ToString("yyyy-MM-ddTHH:mm:ss")
    deadline = $deadline.ToString("yyyy-MM-ddTHH:mm:ss")
    watch_hours = $WatchHours
    main_final_deadline = $MainFinalDeadline
    main_pid = $finalMainPid
    watcher_pid = $finalWatcherPid
    main_stopped_at_deadline = $finalMainStopped
    watcher_stopped_at_deadline = $finalWatcherStopped
    status_source = $statusSource
    history_count = $historyCount
    global_best_ghost = $bestGhost
    global_best_psnr = $bestPsnr
    global_best_ssim = $bestSsim
    global_best_wl1 = $bestWl1
    global_best_ckpt = $globalBestCkpt
    global_best_geom = $globalBestGeom
    overall_best_cycle = $bestCycle
    overall_best_stage = $bestStage
    overall_best_compare_png = $bestComparePng
    overall_best_visual_conclusion = $bestVisualConclusion
    next_cycle_tune_action = $nextTuneAction
    should_stop_by_stagnation = $stopByStagnation
    should_stop_by_regression = $stopByRegression
    decision = $decision
    decision_reason = $decisionReason
    next_route = $nextRoute
    minimal_validation_plan = $minimalValidationPlan
    trajectory_recent_cycles = $trajectoryRecent
}
Write-JsonNoBom -Path $finalJsonPath -Obj $finalObj
Write-JsonNoBom -Path $statusJsonPath -Obj ([ordered]@{
    updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    t0 = $t0.ToString("yyyy-MM-ddTHH:mm:ss")
    deadline = $deadline.ToString("yyyy-MM-ddTHH:mm:ss")
    main_final_deadline = $MainFinalDeadline
    state = "deadline_reached"
    watcher_pid = $finalWatcherPid
    observed_main_pid = $finalMainPid
    watcher_stopped_at_deadline = $finalWatcherStopped
    main_stopped_at_deadline = $finalMainStopped
    final_decision = $decision
    final_report_json = $finalJsonPath
    final_report_md = $finalMdPath
})

$finalLines = @()
$finalLines += "# 过夜 12h 最终报告"
$finalLines += ""
$finalLines += "- updated: $($finalObj.updated_at)"
$finalLines += "- t0: $($finalObj.t0)"
$finalLines += "- deadline: $($finalObj.deadline)"
$finalLines += "- watch_hours: $($finalObj.watch_hours)"
$finalLines += "- main_final_deadline: $($finalObj.main_final_deadline)"
$finalLines += "- main_pid: $($finalObj.main_pid), stopped_at_deadline=$($finalObj.main_stopped_at_deadline)"
$finalLines += "- watcher_pid: $($finalObj.watcher_pid), stopped_at_deadline=$($finalObj.watcher_stopped_at_deadline)"
$finalLines += "- status_source: $($finalObj.status_source)"
$finalLines += "- history_count: $($finalObj.history_count)"
$finalLines += ""
$finalLines += "## 全窗口最优"
$finalLines += "- best_cycle: $($finalObj.overall_best_cycle)"
$finalLines += "- best_stage: $($finalObj.overall_best_stage)"
$finalLines += "- ghost: $(Fmt-Num $bestGhost)"
$finalLines += "- psnr: $(Fmt-Num $bestPsnr)"
$finalLines += "- ssim: $(Fmt-Num $bestSsim)"
$finalLines += "- wl1: $(Fmt-Num $bestWl1)"
$finalLines += "- best_ckpt: $($finalObj.global_best_ckpt)"
$finalLines += "- best_geom: $($finalObj.global_best_geom)"
$finalLines += "- compare_png: $($finalObj.overall_best_compare_png)"
$finalLines += "- visual_conclusion: $($finalObj.overall_best_visual_conclusion)"
$finalLines += ""
$finalLines += "## 近期迭代轨迹"
if ($trajectoryRecent.Count -le 0) {
    $finalLines += "- 当前窗口暂无 cycle 历史可用"
} else {
    foreach ($tr in @($trajectoryRecent)) {
        $finalLines += ("- cycle={0}, stage={1}, ghost={2}, psnr={3}, regressed={4}, tune_next={5}, compare_png={6}" -f `
            $tr.cycle, $tr.best_stage, (Fmt-Num (To-DoubleOrNaN($tr.best_ghost))), (Fmt-Num (To-DoubleOrNaN($tr.best_psnr))), $tr.regressed, $tr.tune_action_next, $tr.compare_png)
        if (-not [string]::IsNullOrWhiteSpace([string]$tr.visual_conclusion)) {
            $finalLines += ("  visual={0}" -f [string]$tr.visual_conclusion)
        }
        if (-not [string]::IsNullOrWhiteSpace([string]$tr.regress_reason)) {
            $finalLines += ("  regress_reason={0}" -f [string]$tr.regress_reason)
        }
    }
}
$finalLines += ""
$finalLines += "## 结论"
$finalLines += "- should_stop_by_stagnation: $($finalObj.should_stop_by_stagnation)"
$finalLines += "- should_stop_by_regression: $($finalObj.should_stop_by_regression)"
$finalLines += "- decision: $($finalObj.decision)"
$finalLines += "- decision_reason: $($finalObj.decision_reason)"
$finalLines += ""
$finalLines += "## 下一技术路线"
foreach ($line in @($nextRoute)) {
    $finalLines += "- $line"
}
$finalLines += ""
$finalLines += "## 最小验证计划"
foreach ($line in @($minimalValidationPlan)) {
    $finalLines += "- $line"
}
Set-Content -Path $finalMdPath -Value ($finalLines -join "`n") -Encoding UTF8
Set-Content -Path $statusMdPath -Value (($finalLines + @("", "- state: deadline_reached")) -join "`n") -Encoding UTF8

$enc = New-Object System.Text.UTF8Encoding($false)
$append = @()
$append += ""
$append += "## $(Get-Date -Format 'yyyy-MM-dd') 12h 监督器最终收尾"
$append += "- updated_at: $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')"
$append += "- t0: $($finalObj.t0)"
$append += "- deadline: $($finalObj.deadline)"
$append += "- main_final_deadline: $($finalObj.main_final_deadline)"
$append += "- main_stopped_at_deadline: $($finalObj.main_stopped_at_deadline)"
$append += "- watcher_stopped_at_deadline: $($finalObj.watcher_stopped_at_deadline)"
$append += "- best_cycle: $($finalObj.overall_best_cycle), best_stage: $($finalObj.overall_best_stage)"
$append += "- best_ghost: $(Fmt-Num $bestGhost), best_psnr: $(Fmt-Num $bestPsnr), best_ssim: $(Fmt-Num $bestSsim), best_wl1: $(Fmt-Num $bestWl1)"
$append += "- decision: $($finalObj.decision) ($($finalObj.decision_reason))"
$append += "- final_report_json: $finalJsonPath"
$append += "- final_report_md: $finalMdPath"
$mentorPayload = Sanitize-TextForUtf8Log -Text (($append -join "`n") + "`n")
[System.IO.File]::AppendAllText((Join-Path (Resolve-Path ".").Path $mentorPath), $mentorPayload, $enc)

$autoContinueTriggered = $false
if ($AutoContinueNextWindow -and ($decision -eq "continue_next_12h_window_if_budget_allows") -and ($CurrentWindowIndex -lt [Math]::Max(1, [int]$MaxAutoContinueWindows))) {
    try {
        $prevArgs = ""
        if ($finalLauncher -ne $null) {
            $prevArgs = [string]$finalLauncher.args
        }
        if ([string]::IsNullOrWhiteSpace($prevArgs)) {
            $prevArgs = "-NoProfile -ExecutionPolicy Bypass -File scripts/run_overnight_ghost_autoloop.ps1 -CodeDir $RepoDir -StopAfterHours $WatchHours"
        }

        $nextResume = [string]$globalBestCkpt
        if ([string]::IsNullOrWhiteSpace($nextResume) -and $finalStatus -ne $null) {
            $nextResume = [string]$finalStatus.current_resume_ckpt
        }
        if ([string]::IsNullOrWhiteSpace($nextResume) -and $finalLauncher -ne $null) {
            $nextResume = [string]$finalLauncher.hot_update_forced_resume_ckpt
        }
        if ([string]::IsNullOrWhiteSpace($nextResume)) {
            $nextResume = $ForcedResumeCkpt
        }
        if ([string]::IsNullOrWhiteSpace($nextResume)) {
            $nextResume = Resolve-PreferredResumeCkpt -Fallback $ForcedResumeCkpt
        }

        $nextPseudo = [string]$globalBestGeom
        if ([string]::IsNullOrWhiteSpace($nextPseudo) -and $finalStatus -ne $null) {
            $nextPseudo = [string]$finalStatus.current_pseudo_geom
        }
        if ([string]::IsNullOrWhiteSpace($nextPseudo)) {
            $nextPseudo = "vggt_geom"
        }

        $nextArgs = [string]$prevArgs
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-StopAfterHours" -Value ([string]$WatchHours)
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-StartResumeCkpt" -Value $nextResume
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-StartPseudoGeomSubdir" -Value $nextPseudo
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2DualLaneEnabled" -Value "`$true"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-PostRescueEnabled" -Value "`$true"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-QualityGuardMode" -Value "layered"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-PromotionGhostDelta" -Value "0.02"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2EnableAnySplatAblationSixPack" -Value "`$true"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2EnableExtendedCkptWaitOnMissing" -Value "`$true"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2CkptExtendedWaitTimeoutSec" -Value "1200"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2EnableResumeCkptFallbackOnShortCkptMissing" -Value "`$false"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2DisallowResumeFallbackResult" -Value "`$true"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2EvalNumSrcViewsList" -Value "8,12,16,20,22"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2GramDynEnable" -Value "off"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2DynProxyEnable" -Value "on"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2DynProxyMode" -Value "fg_static_soft"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2DynProxyUseGram" -Value "on"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2DynProxyUseSupport" -Value "on"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2DynProxyFloor" -Value "0.35"
        $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-Stage2DynProxyWarmupSteps" -Value "40"
        if (-not [string]::IsNullOrWhiteSpace($MainFinalDeadline)) {
            $nextArgs = Set-OrAddLaunchArg -ArgsText $nextArgs -Flag "-FinalDeadline" -Value $MainFinalDeadline
        }

        $appliedTune = ""
        $nextArgs = Apply-TuneActionToLaunchArgs -ArgsText $nextArgs -TuneAction ([string]$nextTuneAction) -AppliedSummary ([ref]$appliedTune)
        $nextArgs = Normalize-BoolLaunchArgs -ArgsText $nextArgs

        $launchTs = Get-Date -Format "yyyyMMdd_HHmmss"
        $mainOut = "logs/modal_phase5/overnight_ghost_autoloop_${launchTs}.out.log"
        $mainErr = "logs/modal_phase5/overnight_ghost_autoloop_${launchTs}.err.log"
        $newMainArgList = Convert-LaunchArgsToArray -ArgsText $nextArgs
        $newMain = Start-Process -FilePath "powershell.exe" -ArgumentList $newMainArgList -WindowStyle Hidden -RedirectStandardOutput $mainOut -RedirectStandardError $mainErr -PassThru
        $nextWindowStartedAt = Get-Date

        Write-JsonNoBom -Path $launcherPath -Obj ([ordered]@{
            started_at = $nextWindowStartedAt.ToString("yyyy-MM-ddTHH:mm:ss")
            pid = $newMain.Id
            stdout = $mainOut
            stderr = $mainErr
            args = $nextArgs
            restart_reason = "auto_continue_next_12h_window"
            hot_update_target_stage = $TargetStage
            hot_update_resume_ckpt = $nextResume
            hot_update_forced_resume_ckpt = $nextResume
            forced_stop_after_hours = $ForcedStopAfterHours
            trigger_exit_classification = ""
            trigger_exit_detail = ""
        })

        $nextWindowDeadline = $nextWindowStartedAt.AddHours([Math]::Max(1, [int]$WatchHours))
        $bootstrapAutoloop = [ordered]@{
            updated_at = $nextWindowStartedAt.ToString("yyyy-MM-ddTHH:mm:ss")
            deadline = $nextWindowDeadline.ToString("yyyy-MM-ddTHH:mm:ss")
            current_cycle = 1
            current_stage = "cycle001_stage1_strong"
            interim = $true
            active_lane = "lane_a"
            lane_a_best = $null
            lane_b_best = $null
            guard_tier = ""
            decision_reason = "new_window_bootstrap"
            rollback_reason = ""
            current_resume_ckpt = $nextResume
            current_pseudo_geom = $nextPseudo
            global_best_ghost = $bestGhost
            global_best_psnr = $bestPsnr
            global_best_ssim = $bestSsim
            global_best_wl1 = $bestWl1
            global_best_stage = $bestStage
            global_best_geom = $globalBestGeom
            next_cycle_tune_action = [string]$nextTuneAction
            note = "new_window_bootstrap_pending_first_sweep"
        }
        Write-JsonNoBom -Path "logs/modal_phase5/overnight_ghost_autoloop_latest.json" -Obj $bootstrapAutoloop
        $bootstrapMd = @()
        $bootstrapMd += "# 过夜 Ghost AutoLoop（window bootstrap）"
        $bootstrapMd += ""
        $bootstrapMd += "- updated: $($bootstrapAutoloop.updated_at)"
        $bootstrapMd += "- deadline: $($bootstrapAutoloop.deadline)"
        $bootstrapMd += "- cycle: $($bootstrapAutoloop.current_cycle)"
        $bootstrapMd += "- stage: $($bootstrapAutoloop.current_stage)"
        $bootstrapMd += "- current_resume_ckpt: $($bootstrapAutoloop.current_resume_ckpt)"
        $bootstrapMd += "- current_pseudo_geom: $($bootstrapAutoloop.current_pseudo_geom)"
        $bootstrapMd += "- global_best_ghost: $($bootstrapAutoloop.global_best_ghost)"
        $bootstrapMd += "- global_best_psnr: $($bootstrapAutoloop.global_best_psnr)"
        $bootstrapMd += "- global_best_ssim: $($bootstrapAutoloop.global_best_ssim)"
        $bootstrapMd += "- global_best_wl1: $($bootstrapAutoloop.global_best_wl1)"
        $bootstrapMd += "- next_cycle_tune_action: $($bootstrapAutoloop.next_cycle_tune_action)"
        $bootstrapMd += "- note: $($bootstrapAutoloop.note)"
        Set-Content -Path "logs/modal_phase5/overnight_ghost_autoloop_latest.md" -Value ($bootstrapMd -join "`n") -Encoding UTF8

        $watchProcs = @(Get-CimInstance Win32_Process | Where-Object {
                $_.Name -match "powershell" -and $_.CommandLine -match "scripts/watch_ghost_outputs\.ps1"
            })
        foreach ($wp in @($watchProcs)) {
            try { Stop-Process -Id ([int]$wp.ProcessId) -Force -ErrorAction SilentlyContinue } catch {}
        }
        $watchOut = "logs/modal_phase5/watch_ghost_outputs_${launchTs}.out.log"
        $watchErr = "logs/modal_phase5/watch_ghost_outputs_${launchTs}.err.log"
        $newWatch = Start-Process -FilePath "powershell.exe" -ArgumentList @(
            "-NoProfile","-ExecutionPolicy","Bypass",
            "-File","scripts/watch_ghost_outputs.ps1",
            "-WatchHours",[string]$WatchHours,
            "-PollSec","45"
        ) -WindowStyle Hidden -RedirectStandardOutput $watchOut -RedirectStandardError $watchErr -PassThru

        $nextEnsureOut = "logs/modal_phase5/ensure_hot_update_watcher_${launchTs}.out.log"
        $nextEnsureErr = "logs/modal_phase5/ensure_hot_update_watcher_${launchTs}.err.log"
        $nextEnsureArgs = @(
            "-NoProfile","-ExecutionPolicy","Bypass",
            "-File","scripts/ensure_hot_update_watcher.ps1",
            "-RepoDir",$RepoDir,
            "-TargetStage",$TargetStage,
            "-ForcedResumeCkpt",$nextResume,
            "-PollSec",[string]$PollSec,
            "-WatchHours",[string]$WatchHours,
            "-WatcherPollSec",[string]$WatcherPollSec,
            "-WatcherMaxWaitMinutes",[string]$WatcherMaxWaitMinutes,
            "-WatcherStage1MaxWaitMinutes",[string]$WatcherStage1MaxWaitMinutes,
            "-WatcherOtherStageMaxWaitMinutes",[string]$WatcherOtherStageMaxWaitMinutes,
            "-WatcherStallMinutes",[string]$WatcherStallMinutes,
            "-WatcherNoChildStallMinutes",[string]$WatcherNoChildStallMinutes,
            "-WatcherCpuAssistWindowMinutes",[string]$WatcherCpuAssistWindowMinutes,
            "-WatcherStartupGraceMinutes",[string]$WatcherStartupGraceMinutes,
            "-WatcherModalDoneStallMinutes",[string]$WatcherModalDoneStallMinutes,
            "-WatcherWorkerAttachWindowMinutes",[string]$WatcherWorkerAttachWindowMinutes,
            "-WatcherWorkerKeepAliveMaxMinutes",[string]$WatcherWorkerKeepAliveMaxMinutes,
            "-ForcedStopAfterHours",[string]$ForcedStopAfterHours,
            "-CurrentWindowIndex",[string]($CurrentWindowIndex + 1),
            "-MaxAutoContinueWindows",[string]$MaxAutoContinueWindows
        )
        if (-not [string]::IsNullOrWhiteSpace($MainFinalDeadline)) {
            $nextEnsureArgs += @("-MainFinalDeadline", $MainFinalDeadline)
        }
        if ($AutoContinueNextWindow) { $nextEnsureArgs += "-AutoContinueNextWindow" }
        $newEnsure = Start-Process -FilePath "powershell.exe" -ArgumentList $nextEnsureArgs -WindowStyle Hidden -RedirectStandardOutput $nextEnsureOut -RedirectStandardError $nextEnsureErr -PassThru

        $backupTs = Get-Date -Format "yyyyMMdd_HHmmss"
        try {
            if (Test-Path $finalJsonPath) {
                Copy-Item $finalJsonPath "logs/modal_phase5/overnight_ghost_autoloop_12h_final_window${CurrentWindowIndex}_$backupTs.json" -Force
            }
            if (Test-Path $finalMdPath) {
                Copy-Item $finalMdPath "logs/modal_phase5/overnight_ghost_autoloop_12h_final_window${CurrentWindowIndex}_$backupTs.md" -Force
            }
        } catch {
        }

        $placeholder = [ordered]@{
            updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
            state = "next_window_running"
            previous_window_index = $CurrentWindowIndex
            next_window_index = ($CurrentWindowIndex + 1)
            previous_deadline = $deadline.ToString("yyyy-MM-ddTHH:mm:ss")
            next_main_pid = $newMain.Id
            next_watch_pid = $newWatch.Id
            next_ensure_pid = $newEnsure.Id
            next_resume_ckpt = $nextResume
            next_pseudo_geom = $nextPseudo
            tune_action_next = [string]$nextTuneAction
            applied_tune = [string]$appliedTune
        }
        Write-JsonNoBom -Path $finalJsonPath -Obj $placeholder
        $placeholderMd = @()
        $placeholderMd += "# 过夜 12h 最终报告（当前窗口进行中）"
        $placeholderMd += ""
        $placeholderMd += "- updated: $($placeholder.updated_at)"
        $placeholderMd += "- state: $($placeholder.state)"
        $placeholderMd += "- previous_window_index: $($placeholder.previous_window_index)"
        $placeholderMd += "- next_window_index: $($placeholder.next_window_index)"
        $placeholderMd += "- previous_deadline: $($placeholder.previous_deadline)"
        $placeholderMd += "- next_main_pid: $($placeholder.next_main_pid)"
        $placeholderMd += "- next_watch_pid: $($placeholder.next_watch_pid)"
        $placeholderMd += "- next_ensure_pid: $($placeholder.next_ensure_pid)"
        $placeholderMd += "- next_resume_ckpt: $($placeholder.next_resume_ckpt)"
        $placeholderMd += "- next_pseudo_geom: $($placeholder.next_pseudo_geom)"
        $placeholderMd += "- tune_action_next: $($placeholder.tune_action_next)"
        $placeholderMd += "- applied_tune: $($placeholder.applied_tune)"
        Set-Content -Path $finalMdPath -Value ($placeholderMd -join "`n") -Encoding UTF8

        $handoffStatus = [ordered]@{
            updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
            t0 = $t0.ToString("yyyy-MM-ddTHH:mm:ss")
            deadline = $deadline.ToString("yyyy-MM-ddTHH:mm:ss")
            state = "handoff_next_window"
            previous_window_index = $CurrentWindowIndex
            next_window_index = ($CurrentWindowIndex + 1)
            next_main_pid = $newMain.Id
            next_watch_pid = $newWatch.Id
            next_ensure_pid = $newEnsure.Id
            next_launcher_stdout = $mainOut
            next_launcher_stderr = $mainErr
            applied_tune = [string]$appliedTune
        }
        Write-JsonNoBom -Path $statusJsonPath -Obj $handoffStatus

        $handoffMd = @()
        $handoffMd += "# 热更新监督器状态"
        $handoffMd += ""
        $handoffMd += "- updated: $($handoffStatus.updated_at)"
        $handoffMd += "- state: $($handoffStatus.state)"
        $handoffMd += "- previous_window_index: $($handoffStatus.previous_window_index)"
        $handoffMd += "- next_window_index: $($handoffStatus.next_window_index)"
        $handoffMd += "- next_main_pid: $($handoffStatus.next_main_pid)"
        $handoffMd += "- next_watch_pid: $($handoffStatus.next_watch_pid)"
        $handoffMd += "- next_ensure_pid: $($handoffStatus.next_ensure_pid)"
        $handoffMd += "- applied_tune: $($handoffStatus.applied_tune)"
        Set-Content -Path $statusMdPath -Value ($handoffMd -join "`n") -Encoding UTF8

        $handoffMentor = @()
        $handoffMentor += ""
        $handoffMentor += "## $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') 自动续跑（12h 窗口切换）"
        $handoffMentor += "- from_window: $CurrentWindowIndex, to_window: $($CurrentWindowIndex + 1)"
        $handoffMentor += "- decision: $decision ($decisionReason)"
        $handoffMentor += "- next_resume_ckpt: $nextResume"
        $handoffMentor += "- next_pseudo_geom: $nextPseudo"
        $handoffMentor += "- tune_action_next: $nextTuneAction"
        $handoffMentor += "- applied_tune: $appliedTune"
        $handoffMentor += "- next_main_pid: $($newMain.Id)"
        $handoffMentor += "- next_watch_pid: $($newWatch.Id)"
        $handoffMentor += "- next_ensure_pid: $($newEnsure.Id)"
        $handoffMentor += "- note: 已自动进入下一 12h 窗口，无需人工确认。"
        $handoffPayload = Sanitize-TextForUtf8Log -Text (($handoffMentor -join "`n") + "`n")
        [System.IO.File]::AppendAllText((Join-Path (Resolve-Path ".").Path $mentorPath), $handoffPayload, $enc)

        $autoContinueTriggered = $true
    } catch {
        $failMentor = @()
        $failMentor += ""
        $failMentor += "## $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') 自动续跑失败"
        $failMentor += "- from_window: $CurrentWindowIndex"
        $failMentor += "- error: $($_.Exception.Message)"
        $failMentor += "- action: 保留当前收尾产物，等待下一次监督器启动恢复。"
        $failPayload = Sanitize-TextForUtf8Log -Text (($failMentor -join "`n") + "`n")
        [System.IO.File]::AppendAllText((Join-Path (Resolve-Path ".").Path $mentorPath), $failPayload, $enc)
    }
}

exit 0

