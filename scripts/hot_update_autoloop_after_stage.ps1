param(
    [string]$RepoDir = "F:\vggt",
    [string]$TargetStage = "cycle001_stage1_strong",
    [string]$ForcedResumeCkpt = "",
    [int]$ForcedStopAfterHours = 12,
    [string]$MainFinalDeadline = "",
    [int]$PollSec = 30,
    [int]$MaxWaitMinutes = 360,
    [int]$StallMinutes = 90,
    [int]$NoChildStallMinutes = 20,
    [double]$CpuActiveDeltaSec = 0.05,
    [int]$CpuAssistWindowMinutes = 15,
    [int]$ModalDoneStallMinutes = 6,
    [int]$StartupGraceMinutes = 25,
    [string]$WorkerProcessNames = "modal,python",
    [int]$WorkerAttachWindowMinutes = 30,
    [int]$WorkerKeepAliveMaxMinutes = 90,
    [int]$Stage1TimeoutMinutes = 120,
    [int]$OtherStageTimeoutMinutes = 90,
    [bool]$StageHardTimeoutEnabled = $true,
    [int]$StageHardTimeoutGraceMinutes = 5,
    [switch]$EnableStageBoundaryRestart = $false,
    [object]$EnforceSingleMainChain = $true,
    [int]$RedundantGuardPollEvery = 3,
    [string]$ModalAppDescriptionRegex = "^vggt-zju-runner$"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

$launcherPath = "logs/modal_phase5/overnight_ghost_autoloop_launcher_latest.json"
$heartbeatPath = "logs/modal_phase5/overnight_ghost_autoloop_heartbeat_latest.json"
$watchStatusPath = "logs/modal_phase5/overnight_ghost_autoloop_hotupdate_watch_latest.json"
$ensureWatchPath = "logs/modal_phase5/ensure_hot_update_watcher_latest.json"
$mentorPath = "logs/modal_phase5/mentor_update_latest.md"
$finalPath = "logs/modal_phase5/overnight_ghost_autoloop_12h_final_latest.json"

function Read-JsonMaybe([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try {
        return (Get-Content $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Parse-DateMaybe([string]$Text) {
    if ([string]::IsNullOrWhiteSpace($Text)) { return [datetime]::MinValue }
    try {
        return [datetime]::Parse($Text)
    } catch {
        return [datetime]::MinValue
    }
}

function To-BoolLoose(
    [object]$Value,
    [bool]$Default = $false
) {
    if ($null -eq $Value) { return $Default }
    if ($Value -is [bool]) { return [bool]$Value }
    if ($Value -is [int] -or $Value -is [long] -or $Value -is [double] -or $Value -is [decimal]) {
        return ([double]$Value -ne 0.0)
    }
    $s = [string]$Value
    if ([string]::IsNullOrWhiteSpace($s)) { return $Default }
    $s = $s.Trim().ToLowerInvariant()
    if ($s.StartsWith('$')) { $s = $s.TrimStart('$') }
    if ($s.StartsWith('"') -and $s.EndsWith('"') -and $s.Length -ge 2) {
        $s = $s.Substring(1, $s.Length - 2).Trim().ToLowerInvariant()
    }
    if ($s -eq "system.string") { return $Default }
    if ($s -match '^(1|true|yes|y|on)$') { return $true }
    if ($s -match '^(0|false|no|n|off)$') { return $false }
    try { return ([double]$s -ne 0.0) } catch { return $Default }
}

$EnforceSingleMainChain = To-BoolLoose -Value $EnforceSingleMainChain -Default $true

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
    $json = $safe | ConvertTo-Json -Depth 30
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

function Get-FallbackBestCkpt() {
    $ghostSweep = "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv"
    if (Test-Path $ghostSweep) {
        try {
            $rows = @(
                Import-Csv $ghostSweep |
                    Where-Object { [int]$_.exit_code -eq 0 }
            )
            if ($rows.Count -gt 0) {
                $rowsSorted = @(
                    $rows |
                        Sort-Object {
                            $g = [double]::NaN
                            try { $g = [double]$_.ghost_score_mean } catch {}
                            if ([double]::IsNaN($g)) { return [double]::PositiveInfinity }
                            return $g
                        }, {
                            $p = [double]::NaN
                            try { $p = [double]$_.mean_PSNR } catch {}
                            if ([double]::IsNaN($p)) { return [double]::NegativeInfinity }
                            return (-1.0 * $p)
                        }
                )
                if ($rowsSorted.Count -gt 0) {
                    $pick = $rowsSorted[0]
                    $sweepCsv = [string]$pick.sweep_csv
                    $bestLabel = [string]$pick.best_label
                    if (-not [string]::IsNullOrWhiteSpace($sweepCsv) -and (Test-Path $sweepCsv)) {
                        $sweepRows = @(
                            Import-Csv $sweepCsv |
                                Where-Object { $_.status -eq "ok" -and $_.stage -eq "short" }
                        )
                        if ($sweepRows.Count -gt 0) {
                            if (-not [string]::IsNullOrWhiteSpace($bestLabel)) {
                                $hit = @($sweepRows | Where-Object { [string]$_.label -eq $bestLabel } | Select-Object -First 1)
                                if ($hit.Count -gt 0) {
                                    return [string]$hit[0].ft_ckpt
                                }
                            }
                            return [string]$sweepRows[0].ft_ckpt
                        }
                    }
                }
            }
        } catch {
        }
    }

    $sweepCsv = "logs/modal_phase5/vggt_ft_sweep_latest.csv"
    if (-not (Test-Path $sweepCsv)) { return "" }
    try {
        $rows = @(
            Import-Csv $sweepCsv |
                Where-Object { $_.status -eq "ok" -and $_.stage -eq "short" }
        )
        if ($rows.Count -gt 0) {
            return [string]$rows[0].ft_ckpt
        }
    } catch {
    }
    return ""
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

function Get-LatestActivity([int]$ProcId, [string]$StdoutPath) {
    $latest = [datetime]::MinValue
    $src = ""
    $paths = @(
        $StdoutPath,
        "logs/modal_phase5/modal_run_progress_latest.json",
        "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv",
        "logs/modal_phase5/vggt_ft_sweep_latest.csv",
        "logs/modal_phase5/vggt_ft_gate_latest.json"
    )
    foreach ($p in @($paths)) {
        if ([string]::IsNullOrWhiteSpace($p)) { continue }
        if (-not (Test-Path $p)) { continue }
        try {
            $t = (Get-Item $p).LastWriteTime
            if ($t -gt $latest) {
                $latest = $t
                $src = $p
            }
        } catch {
        }
    }

    return [pscustomobject]@{
        last_activity_at = $(if ($latest -eq [datetime]::MinValue) { "" } else { $latest.ToString("yyyy-MM-ddTHH:mm:ss") })
        last_activity_source = $src
        last_activity_dt = $latest
    }
}

function Get-ChildCount([int]$ProcId) {
    return -1
}

function Get-WorkerCount([int]$ProcId) {
    return -1
}

function Resolve-StageTimeoutMinutes(
    [string]$StageName,
    [int]$FallbackMinutes,
    [int]$Stage1Minutes,
    [int]$OtherMinutes
) {
    $fallback = [Math]::Max(10, [int]$FallbackMinutes)
    if ([string]::IsNullOrWhiteSpace($StageName)) { return $fallback }
    $s = $StageName.Trim().ToLowerInvariant()
    if ($s -match "stage1") { return [Math]::Max(10, [int]$Stage1Minutes) }
    if ($s -match "stage[2-9]") { return [Math]::Max(10, [int]$OtherMinutes) }
    return $fallback
}

function Get-ExitClassification(
    [string]$StdoutPath,
    [string]$StderrPath,
    [string]$ModalProgressState
) {
    $stdoutTail = ""
    $stderrTail = ""
    try {
        if (-not [string]::IsNullOrWhiteSpace($StdoutPath) -and (Test-Path $StdoutPath)) {
            $stdoutTail = ((Get-Content $StdoutPath -Tail 100) -join "`n")
        }
    } catch {
    }
    try {
        if (-not [string]::IsNullOrWhiteSpace($StderrPath) -and (Test-Path $StderrPath)) {
            $stderrTail = ((Get-Content $StderrPath -Tail 100) -join "`n")
        }
    } catch {
    }
    $combined = ($stderrTail + "`n" + $stdoutTail).ToLowerInvariant()
    $class = "unknown"
    $detail = ""

    if ($ModalProgressState -in @("failed", "error", "stopped")) {
        $class = "modal_failed"
        $detail = "modal_run_progress=" + $ModalProgressState
    } elseif ($ModalProgressState -eq "timeout") {
        $class = "modal_timeout"
        $detail = "modal_run_progress=timeout"
    } elseif ($combined -match "traceback|parsererror|terminatingerror|exception|is not recognized|cannot find path|file not found|module not found") {
        $class = "script_error"
        $detail = "stderr_or_stdout_contains_exception"
    } elseif ($combined -match "timed out|timeout|deadline exceeded") {
        $class = "timeout"
        $detail = "stderr_or_stdout_contains_timeout"
    } elseif ($combined -match "unavailable|connection reset|connection refused|temporary failure|network|dns|socket|grpc") {
        $class = "network_or_modal_infra"
        $detail = "stderr_or_stdout_contains_network_signal"
    }

    return [pscustomobject]@{
        class = $class
        detail = $detail
    }
}

function Stop-ProcessTreeSafe(
    [int]$TargetPid,
    [string]$Reason
) {
    if ($TargetPid -le 0) { return $false }
    try { & taskkill /PID $TargetPid /T /F > $null 2>&1 } catch {}
    Start-Sleep -Milliseconds 120
    try { Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue } catch {}
    $alive = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if ($alive -eq $null) {
        Write-Host "[hot-update] single-chain stop local pid=$TargetPid reason=$Reason"
        return $true
    }
    return $false
}

function Get-ModalRunCmdProcesses() {
    try {
        return @(
            Get-CimInstance Win32_Process | Where-Object {
                ([string]$_.Name).ToLowerInvariant() -eq "cmd.exe" -and
                ([string]$_.CommandLine -match '(?i)modal\s+run(?:\s+-q)?\s+"?modal_run_train\.py"?')
            }
        )
    } catch {
        return @()
    }
}

function Sort-ProcessesByCreationDesc([object[]]$Rows) {
    if ($null -eq $Rows) { return @() }
    return @(
        @($Rows) | Sort-Object {
            try {
                [System.Management.ManagementDateTimeConverter]::ToDateTime([string]$_.CreationDate)
            } catch {
                [datetime]::MinValue
            }
        } -Descending
    )
}

function Invoke-SingleChainGuard(
    [int]$MainPid,
    [int]$ModalProgressPid,
    [int]$KeepNewestLocalCmd = 1,
    [int]$KeepNewestActiveApps = 1,
    [string]$Reason = "watch_loop"
) {
    $result = [ordered]@{
        local_cmd_count = 0
        local_killed = 0
        keep_cmd_pids = ""
        app_active_count = 0
        app_stopped = 0
        keep_app_ids = ""
        reason = $Reason
    }

    if (-not [bool]$EnforceSingleMainChain) {
        return [pscustomobject]$result
    }

    $cmdRows = Sort-ProcessesByCreationDesc -Rows (Get-ModalRunCmdProcesses)
    $result.local_cmd_count = @($cmdRows).Count
    $keepCmdIds = New-Object System.Collections.Generic.List[int]

    if ([int]$KeepNewestLocalCmd -gt 0) {
        if ($ModalProgressPid -gt 0) {
            $hit = @($cmdRows | Where-Object { [int]$_.ProcessId -eq [int]$ModalProgressPid } | Select-Object -First 1)
            if ($hit.Count -gt 0) {
                [void]$keepCmdIds.Add([int]$ModalProgressPid)
            }
        }
        if (($MainPid -gt 0) -and ($keepCmdIds.Count -lt [int]$KeepNewestLocalCmd)) {
            $children = @($cmdRows | Where-Object { [int]$_.ParentProcessId -eq [int]$MainPid })
            foreach ($c in @($children)) {
                if ($keepCmdIds.Count -ge [int]$KeepNewestLocalCmd) { break }
                $cid = [int]$c.ProcessId
                if (-not ($keepCmdIds -contains $cid)) {
                    [void]$keepCmdIds.Add($cid)
                }
            }
        }
        if ($keepCmdIds.Count -lt [int]$KeepNewestLocalCmd) {
            foreach ($c in @($cmdRows)) {
                if ($keepCmdIds.Count -ge [int]$KeepNewestLocalCmd) { break }
                $cid = [int]$c.ProcessId
                if (-not ($keepCmdIds -contains $cid)) {
                    [void]$keepCmdIds.Add($cid)
                }
            }
        }
    }
    $result.keep_cmd_pids = ($keepCmdIds -join ",")

    foreach ($c in @($cmdRows)) {
        $cid = [int]$c.ProcessId
        if ($keepCmdIds -contains $cid) { continue }
        if (Stop-ProcessTreeSafe -TargetPid $cid -Reason "$Reason:redundant_local_cmd") {
            $result.local_killed = [int]$result.local_killed + 1
        }
    }

    $apps = @()
    try {
        $appsRaw = (modal app list --json 2>$null | Out-String)
        if (-not [string]::IsNullOrWhiteSpace($appsRaw)) {
            $apps = @($appsRaw | ConvertFrom-Json)
        }
    } catch {
        $apps = @()
    }
    if (@($apps).Count -gt 0) {
        $activeApps = @(
            @($apps) | Where-Object {
                ([string]$_.State).ToLowerInvariant() -eq "ephemeral" -and
                (
                    [string]::IsNullOrWhiteSpace($ModalAppDescriptionRegex) -or
                    ([string]$_.Description -match $ModalAppDescriptionRegex)
                )
            }
        )
        $result.app_active_count = @($activeApps).Count
        $activeApps = @(
            @($activeApps) | Sort-Object {
                Parse-DateMaybe -Text ([string]$_.'Created at')
            } -Descending
        )
        $keepAppIds = New-Object System.Collections.Generic.List[string]
        if ([int]$KeepNewestActiveApps -gt 0) {
            foreach ($a in @($activeApps)) {
                if ($keepAppIds.Count -ge [int]$KeepNewestActiveApps) { break }
                $aid = [string]$a.'App ID'
                if (-not [string]::IsNullOrWhiteSpace($aid) -and (-not ($keepAppIds -contains $aid))) {
                    [void]$keepAppIds.Add($aid)
                }
            }
        }
        $result.keep_app_ids = ($keepAppIds -join ",")

        foreach ($a in @($activeApps)) {
            $aid = [string]$a.'App ID'
            if ([string]::IsNullOrWhiteSpace($aid)) { continue }
            if ($keepAppIds -contains $aid) { continue }
            try {
                modal app stop $aid 2>$null | Out-Null
                $result.app_stopped = [int]$result.app_stopped + 1
                Write-Host "[hot-update] single-chain stop app=$aid reason=$Reason"
            } catch {
            }
        }
    }

    return [pscustomobject]$result
}

$launcher = Read-JsonMaybe -Path $launcherPath
if ($launcher -eq $null) {
    throw "launcher meta not found: $launcherPath"
}

$oldPid = [int]$launcher.pid
$oldArgs = [string]$launcher.args
$oldStdout = [string]$launcher.stdout
$oldErr = [string]$launcher.stderr
$launcherT0 = Parse-DateMaybe -Text ([string]$launcher.started_at)
$launcherDeadline = [datetime]::MinValue
if (($launcherT0 -ne [datetime]::MinValue) -and ($ForcedStopAfterHours -gt 0)) {
    $launcherDeadline = $launcherT0.AddHours([double]$ForcedStopAfterHours)
}
$deadline = [datetime]::MinValue
$windowT0 = [datetime]::MinValue
$t0Source = ""
$deadlineSource = ""
$ensureMeta = Read-JsonMaybe -Path $ensureWatchPath
if ($ensureMeta -ne $null) {
    $ensureT0 = Parse-DateMaybe -Text ([string]$ensureMeta.t0)
    $ensureDeadline = Parse-DateMaybe -Text ([string]$ensureMeta.deadline)
    if ($ensureT0 -ne [datetime]::MinValue) {
        $windowT0 = $ensureT0
        $t0Source = "ensure_watch.t0"
    }
    if ($ensureDeadline -ne [datetime]::MinValue) {
        $deadline = $ensureDeadline
        $deadlineSource = "ensure_watch.deadline"
    }
}
if (($windowT0 -eq [datetime]::MinValue) -or ($deadline -eq [datetime]::MinValue)) {
    $priorWatch = Read-JsonMaybe -Path $watchStatusPath
    if ($priorWatch -ne $null) {
        if ($windowT0 -eq [datetime]::MinValue) {
            $priorT0 = Parse-DateMaybe -Text ([string]$priorWatch.t0)
            if ($priorT0 -ne [datetime]::MinValue) {
                $windowT0 = $priorT0
                $t0Source = "hot_watch.t0"
            }
        }
        if ($deadline -eq [datetime]::MinValue) {
            $priorDeadline = Parse-DateMaybe -Text ([string]$priorWatch.deadline)
            if ($priorDeadline -ne [datetime]::MinValue) {
                $deadline = $priorDeadline
                $deadlineSource = "hot_watch.deadline"
            }
        }
    }
}
if (($windowT0 -eq [datetime]::MinValue) -or ($deadline -eq [datetime]::MinValue)) {
    $finalMeta = Read-JsonMaybe -Path $finalPath
    if ($finalMeta -ne $null) {
        if ($windowT0 -eq [datetime]::MinValue) {
            $finalT0 = Parse-DateMaybe -Text ([string]$finalMeta.t0)
            if ($finalT0 -ne [datetime]::MinValue) {
                $windowT0 = $finalT0
                $t0Source = "final_json.t0"
            }
        }
        if ($deadline -eq [datetime]::MinValue) {
            $finalDeadline = Parse-DateMaybe -Text ([string]$finalMeta.deadline)
            if ($finalDeadline -ne [datetime]::MinValue) {
                $deadline = $finalDeadline
                $deadlineSource = "final_json.deadline"
            }
        }
    }
}
if ($windowT0 -eq [datetime]::MinValue) {
    if ($launcherT0 -ne [datetime]::MinValue) {
        $windowT0 = $launcherT0
        $t0Source = "launcher.started_at"
    } else {
        $windowT0 = Get-Date
        $t0Source = "now.fallback"
    }
}
if ($deadline -eq [datetime]::MinValue) {
    if (($windowT0 -ne [datetime]::MinValue) -and ($ForcedStopAfterHours -gt 0)) {
        $deadline = $windowT0.AddHours([double]$ForcedStopAfterHours)
        $deadlineSource = "derived_from_t0"
    } elseif (($launcherT0 -ne [datetime]::MinValue) -and ($ForcedStopAfterHours -gt 0)) {
        $deadline = $launcherT0.AddHours([double]$ForcedStopAfterHours)
        $deadlineSource = "launcher.started_at"
    } elseif ($ForcedStopAfterHours -gt 0) {
        $deadline = (Get-Date).AddHours([double]$ForcedStopAfterHours)
        $deadlineSource = "now.plus_forced_stop"
    } else {
        $deadline = (Get-Date).AddMinutes([Math]::Max(5, [int]$MaxWaitMinutes))
        $deadlineSource = "now.plus_max_wait"
    }
}
$mainFinalDeadlineDt = Parse-DateMaybe -Text ([string]$MainFinalDeadline)
if ($mainFinalDeadlineDt -ne [datetime]::MinValue) {
    $deadline = $mainFinalDeadlineDt
    $deadlineSource = "param.main_final_deadline"
    if (($windowT0 -eq [datetime]::MinValue) -or ($windowT0 -gt $deadline)) {
        if ($ForcedStopAfterHours -gt 0) {
            $windowT0 = $deadline.AddHours(-1.0 * [double]$ForcedStopAfterHours)
            $t0Source = "derived_from_main_final_deadline"
        } else {
            $windowT0 = Get-Date
            $t0Source = "now.fallback_main_final_deadline"
        }
    }
}
if (($windowT0 -eq [datetime]::MinValue) -and ($ForcedStopAfterHours -gt 0) -and ($deadline -ne [datetime]::MinValue)) {
    $windowT0 = $deadline.AddHours(-1.0 * [double]$ForcedStopAfterHours)
    $t0Source = "derived_from_deadline"
}
$triggered = $false
$triggerReason = ""
$bestCkpt = ""
$effectiveForcedResumeCkpt = Resolve-PreferredResumeCkpt -Fallback $ForcedResumeCkpt
$exitClass = ""
$exitDetail = ""
$prevCpuSampleAt = [datetime]::MinValue
$prevCpuTotal = [double]::NaN
$lastCpuActiveAt = [datetime]::MinValue
$lastCpuDeltaSec = [double]::NaN
$prevOutLen = [int64]-1
$prevOutTailSig = ""
$lastOutRealProgressAt = [datetime]::MinValue
$guardTick = 0
$lastGuardAt = ""
$lastGuardReason = ""
$lastGuardLocalCmdCount = -1
$lastGuardLocalKilled = 0
$lastGuardKeepCmdPids = ""
$lastGuardAppActiveCount = -1
$lastGuardAppStopped = 0
$lastGuardKeepAppIds = ""
$workerNameSet = @()
foreach ($n in @(([string]$WorkerProcessNames -split ","))) {
    $name = [string]$n
    if (-not [string]::IsNullOrWhiteSpace($name)) {
        $workerNameSet += $name.Trim().ToLowerInvariant()
    }
}
$workerNameSet = @($workerNameSet | Select-Object -Unique)

while ((Get-Date) -lt $deadline) {
    $guardTick += 1
    $proc = $null
    if ($oldPid -gt 0) { $proc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue }
    $now = Get-Date
    $cpuDeltaSec = [double]::NaN
    $startupAgeMinutes = [double]::NaN
    if ($proc -ne $null) {
        try {
            $cpuNow = [double]$proc.CPU
            if (($prevCpuSampleAt -ne [datetime]::MinValue) -and (-not [double]::IsNaN($prevCpuTotal))) {
                $cpuDeltaSec = [Math]::Max(0.0, $cpuNow - $prevCpuTotal)
                if ($cpuDeltaSec -ge [Math]::Max(0.01, [double]$CpuActiveDeltaSec)) {
                    $lastCpuActiveAt = $now
                    $lastCpuDeltaSec = $cpuDeltaSec
                }
            }
            $prevCpuTotal = $cpuNow
            $prevCpuSampleAt = $now
        } catch {
        }
        try {
            $startupAgeMinutes = ($now - $proc.StartTime).TotalMinutes
        } catch {
        }
    }
    $hb = Read-JsonMaybe -Path $heartbeatPath
    $modalProgress = Read-JsonMaybe -Path "logs/modal_phase5/modal_run_progress_latest.json"
    $modalProgressState = ""
    $modalProgressUpdatedAt = ""
    $modalProgressUpdatedDt = [datetime]::MinValue
    $modalProgressStartedAt = ""
    $modalProgressStartedDt = [datetime]::MinValue
    $modalProgressScriptPath = ""
    $modalProgressPid = 0
    $modalProgressElapsedSec = [double]::NaN
    $modalProgressTimeoutSec = [double]::NaN
    $modalProgressOverrunSec = [double]::NaN
    $modalDoneStallEffectiveMinutes = [double]::NaN
    $outIdleAfterModalDoneMinutes = [double]::NaN
    $modalDoneGuardReason = ""
    $outObservedWriteAt = ""
    $outObservedWriteDt = [datetime]::MinValue
    $outObservedLength = [int64]-1
    $outRealProgressAt = [datetime]::MinValue
    $outRealProgressSignal = $false
    $modalVolumeCleanupActive = $false
    if ($modalProgress -ne $null) {
        $modalProgressState = ([string]$modalProgress.state).Trim().ToLowerInvariant()
        $modalProgressScriptPath = ([string]$modalProgress.script_path).Trim().ToLowerInvariant()
        try { $modalProgressPid = [int]$modalProgress.pid } catch { $modalProgressPid = 0 }
        $modalProgressUpdatedAt = [string]$modalProgress.updated_at
        $modalProgressStartedAt = [string]$modalProgress.started_at
        try { $modalProgressElapsedSec = [double]$modalProgress.elapsed_sec } catch { $modalProgressElapsedSec = [double]::NaN }
        try { $modalProgressTimeoutSec = [double]$modalProgress.timeout_sec } catch { $modalProgressTimeoutSec = [double]::NaN }
        try {
            if (-not [string]::IsNullOrWhiteSpace($modalProgressUpdatedAt)) {
                $modalProgressUpdatedDt = [datetime]::Parse($modalProgressUpdatedAt)
                if ($modalProgressUpdatedDt -ne [datetime]::MinValue) {
                    $modalProgressUpdatedAt = $modalProgressUpdatedDt.ToString("yyyy-MM-ddTHH:mm:ss")
                }
            }
        } catch {
            $modalProgressUpdatedDt = [datetime]::MinValue
        }
        try {
            if (-not [string]::IsNullOrWhiteSpace($modalProgressStartedAt)) {
                $modalProgressStartedDt = [datetime]::Parse($modalProgressStartedAt)
                if ($modalProgressStartedDt -ne [datetime]::MinValue) {
                    $modalProgressStartedAt = $modalProgressStartedDt.ToString("yyyy-MM-ddTHH:mm:ss")
                }
            }
        } catch {
            $modalProgressStartedDt = [datetime]::MinValue
        }
        if ([double]::IsNaN($modalProgressElapsedSec) -and ($modalProgressStartedDt -ne [datetime]::MinValue)) {
            $modalProgressElapsedSec = ($now - $modalProgressStartedDt).TotalSeconds
        }
        if ($modalProgressUpdatedDt -eq [datetime]::MinValue) {
            try {
                $modalProgressUpdatedDt = (Get-Item "logs/modal_phase5/modal_run_progress_latest.json").LastWriteTime
                $modalProgressUpdatedAt = $modalProgressUpdatedDt.ToString("yyyy-MM-ddTHH:mm:ss")
            } catch {
            }
        }
        if ($modalProgressState -in @("running", "retrying")) {
            $modalPidAlive = $false
            if ($modalProgressPid -gt 0) {
                try {
                    $null = Get-Process -Id $modalProgressPid -ErrorAction Stop
                    $modalPidAlive = $true
                } catch {
                    $modalPidAlive = $false
                }
            }
            $staleAgeSec = 0.0
            if ($modalProgressUpdatedDt -ne [datetime]::MinValue) {
                $staleAgeSec = ($now - $modalProgressUpdatedDt).TotalSeconds
            }
            if (($modalProgressPid -gt 0) -and (-not $modalPidAlive) -and ($staleAgeSec -ge 15.0)) {
                $modalProgressState = "stale"
                try {
                    $modalProgress.state = "stale"
                    $modalProgress.proc_exists = $false
                    $modalProgress.note = "stale_progress_pid_missing"
                    $modalProgress.stale_detected_at = $now.ToString("yyyy-MM-ddTHH:mm:ss")
                    Write-JsonNoBom -Path "logs/modal_phase5/modal_run_progress_latest.json" -Obj $modalProgress
                    $modalProgressUpdatedDt = $now
                    $modalProgressUpdatedAt = $now.ToString("yyyy-MM-ddTHH:mm:ss")
                } catch {
                }
            }
        }
    }
    if ([bool]$EnforceSingleMainChain) {
        $guardEvery = [Math]::Max(1, [int]$RedundantGuardPollEvery)
        if ((($guardTick - 1) % $guardEvery) -eq 0) {
            $guardRes = Invoke-SingleChainGuard `
                -MainPid $oldPid `
                -ModalProgressPid $modalProgressPid `
                -KeepNewestLocalCmd 1 `
                -KeepNewestActiveApps 1 `
                -Reason "watch_loop"
            $lastGuardAt = $now.ToString("yyyy-MM-ddTHH:mm:ss")
            $lastGuardReason = [string]$guardRes.reason
            $lastGuardLocalCmdCount = [int]$guardRes.local_cmd_count
            $lastGuardLocalKilled = [int]$guardRes.local_killed
            $lastGuardKeepCmdPids = [string]$guardRes.keep_cmd_pids
            $lastGuardAppActiveCount = [int]$guardRes.app_active_count
            $lastGuardAppStopped = [int]$guardRes.app_stopped
            $lastGuardKeepAppIds = [string]$guardRes.keep_app_ids
        }
    }
    $hit = $false
    $hitStageBoundary = $false
    $idleMinutes = [double]::NaN
    $fileActivityAgeMinutes = [double]::NaN
    $effectiveLastActivityDt = [datetime]::MinValue
    $effectiveLastActivitySource = ""
    $cpuSignalUsed = $false
    $childCount = Get-ChildCount -ProcId $oldPid
    $workerCount = -1
    $workerNames = ""
    $workerSignalUsed = $false
    $workerCanAssist = $false
    $observedStage = ""
    if ($hb -ne $null) {
        $observedStage = [string]$hb.stage
    }
    if ([string]::IsNullOrWhiteSpace($observedStage)) {
        $observedStage = $TargetStage
    }
    $stageTimeoutMinutes = Resolve-StageTimeoutMinutes `
        -StageName $observedStage `
        -FallbackMinutes $StallMinutes `
        -Stage1Minutes $Stage1TimeoutMinutes `
        -OtherMinutes $OtherStageTimeoutMinutes
    if (($proc -ne $null) -and ($workerNameSet.Count -gt 0)) {
        try {
            $mainStart = $proc.StartTime
            $workerStartUpper = $mainStart.AddMinutes([Math]::Max(1, [int]$WorkerAttachWindowMinutes))
            $detected = @(
                Get-Process -ErrorAction SilentlyContinue |
                    Where-Object {
                        $pn = [string]$_.ProcessName
                        if ([string]::IsNullOrWhiteSpace($pn)) { return $false }
                        if (-not ($workerNameSet -contains $pn.ToLowerInvariant())) { return $false }
                        try {
                            return ($_.StartTime -ge $mainStart.AddMinutes(-1)) -and ($_.StartTime -le $workerStartUpper)
                        } catch {
                            return $false
                        }
                    }
            )
            $workerCount = $detected.Count
            if ($detected.Count -gt 0) {
                $workerNames = (
                    $detected |
                        Select-Object -ExpandProperty ProcessName -Unique
                ) -join ","
            }
        } catch {
        }
    }
    $stallThreshold = [Math]::Max(10, [int]$stageTimeoutMinutes)
    $activity = Get-LatestActivity -ProcId $oldPid -StdoutPath $oldStdout
    if (-not [string]::IsNullOrWhiteSpace($oldStdout) -and (Test-Path $oldStdout)) {
        try {
            $outItem = Get-Item $oldStdout
            $outObservedLength = [int64]$outItem.Length
            $outObservedWriteDt = $outItem.LastWriteTime
            if ($outObservedWriteDt -ne [datetime]::MinValue) {
                $outObservedWriteAt = $outObservedWriteDt.ToString("yyyy-MM-ddTHH:mm:ss")
            }
            $tailLines = @()
            try {
                $tailLines = @(Get-Content $oldStdout -Tail 8)
            } catch {
                $tailLines = @()
            }
            $outTailSig = ($tailLines -join "`n")
            $outTailLower = $outTailSig.ToLowerInvariant()
            $modalDoneStates = @("done", "failed", "timeout", "stopped", "error")
            $tailLooksLikeKeepAliveOnly = $false
            if ($modalDoneStates -contains $modalProgressState) {
                if (
                    ($outTailLower -match "waiting checkpoint on volume") -or
                    ($outTailLower -match "\[lr-sweep\]\s+modal run alive script=") -or
                    ($outTailLower -match "\[lr-sweep\]\s+eval alive elapsed=")
                ) {
                    $tailLooksLikeKeepAliveOnly = $true
                }
            }
            if ($prevOutLen -lt 0) {
                $prevOutLen = $outObservedLength
                $prevOutTailSig = $outTailSig
                if ($outObservedWriteDt -ne [datetime]::MinValue) {
                    $lastOutRealProgressAt = $outObservedWriteDt
                } else {
                    $lastOutRealProgressAt = $now
                }
                $outRealProgressSignal = $true
            } elseif (($outObservedLength -ne $prevOutLen) -or ($outTailSig -ne $prevOutTailSig)) {
                $prevOutLen = $outObservedLength
                $prevOutTailSig = $outTailSig
                if (-not $tailLooksLikeKeepAliveOnly) {
                    if ($outObservedWriteDt -ne [datetime]::MinValue) {
                        $lastOutRealProgressAt = $outObservedWriteDt
                    } else {
                        $lastOutRealProgressAt = $now
                    }
                    $outRealProgressSignal = $true
                } else {
                    $outRealProgressSignal = $false
                }
            }
            if ($lastOutRealProgressAt -ne [datetime]::MinValue) {
                $outRealProgressAt = $lastOutRealProgressAt
            }
        } catch {
        }
    }

    if ($EnableStageBoundaryRestart -and ($hb -ne $null)) {
        $hbStage = [string]$hb.stage
        $hbState = [string]$hb.state
        if (($hbStage -eq $TargetStage) -and ($hbState -eq "stage_done")) {
            $hit = $true
            $hitStageBoundary = $true
            $triggerReason = "stage_done_heartbeat"
            $bestCkpt = [string]$hb.best_ckpt
        }
    }
    if ($EnableStageBoundaryRestart -and (-not $hit) -and (-not [string]::IsNullOrWhiteSpace($oldStdout)) -and (Test-Path $oldStdout)) {
        $tail = @(Get-Content $oldStdout -Tail 120)
        $needle = "stage=$TargetStage done"
        if (($tail -join "`n").Contains($needle)) {
            $hit = $true
            $hitStageBoundary = $true
            $triggerReason = "stage_done_log_tail"
            if ($hb -ne $null) {
                $bestCkpt = [string]$hb.best_ckpt
            }
        }
    }

    if (-not $hit) {
        $effectiveLastActivityDt = $activity.last_activity_dt
        $effectiveLastActivitySource = [string]$activity.last_activity_source
        $cpuCanAssist = $false
        if ($activity.last_activity_dt -ne [datetime]::MinValue) {
            $fileActivityAgeMinutes = ($now - $activity.last_activity_dt).TotalMinutes
            if ($fileActivityAgeMinutes -le [Math]::Max(1, [int]$CpuAssistWindowMinutes)) {
                $cpuCanAssist = $true
            }
        } elseif (($proc -ne $null) -and (-not [double]::IsNaN($startupAgeMinutes)) -and ($startupAgeMinutes -lt [double]$StartupGraceMinutes)) {
            $cpuCanAssist = $true
        }
        if ($modalProgressState -in @("done", "failed", "timeout", "stopped", "error")) {
            $cpuCanAssist = $false
        }

        # CPU delta is only an auxiliary keep-alive signal near real file/log activity.
        if ($cpuCanAssist -and ($lastCpuActiveAt -ne [datetime]::MinValue) -and (($effectiveLastActivityDt -eq [datetime]::MinValue) -or ($lastCpuActiveAt -gt $effectiveLastActivityDt))) {
            $effectiveLastActivityDt = $lastCpuActiveAt
            $effectiveLastActivitySource = "process_cpu_delta"
            $cpuSignalUsed = $true
        }
        if (($proc -ne $null) -and (-not [double]::IsNaN($startupAgeMinutes)) -and ($startupAgeMinutes -lt [double]$StartupGraceMinutes)) {
            $effectiveLastActivityDt = $now
            $effectiveLastActivitySource = "startup_grace"
        }
        $workerCanAssist = $true
        if ($modalProgress -ne $null) {
            $workerCanAssist = ($modalProgressState -eq "running")
        }
        if ($workerCanAssist -and ($workerCount -gt 0) -and (-not [double]::IsNaN($startupAgeMinutes)) -and ($startupAgeMinutes -le [Math]::Max(1, [int]$WorkerKeepAliveMaxMinutes))) {
            $effectiveLastActivityDt = $now
            $effectiveLastActivitySource = "detected_workers:$workerNames"
            $workerSignalUsed = $true
        }

        # If modal run already finished but main out/log no longer progresses, treat as stalled quickly.
        if (-not $hit -and ($modalProgressState -in @("done", "failed", "timeout", "stopped", "error"))) {
            $modalDoneCanTrigger = $true
            if (($proc -ne $null) -and (-not [double]::IsNaN($startupAgeMinutes)) -and ($startupAgeMinutes -lt [double]$StartupGraceMinutes)) {
                $modalDoneCanTrigger = $false
                $modalDoneGuardReason = ("startup_grace_guard:{0}<{1}m" -f [int][Math]::Round($startupAgeMinutes), [int]$StartupGraceMinutes)
            }
            if ($modalDoneCanTrigger -and ($proc -ne $null)) {
                $procStartDt = [datetime]::MinValue
                try { $procStartDt = $proc.StartTime } catch {}
                if ($procStartDt -ne [datetime]::MinValue) {
                    $modalRefDt = [datetime]::MinValue
                    if ($modalProgressStartedDt -ne [datetime]::MinValue) {
                        $modalRefDt = $modalProgressStartedDt
                    } elseif ($modalProgressUpdatedDt -ne [datetime]::MinValue) {
                        $modalRefDt = $modalProgressUpdatedDt
                    }
                    if (($modalRefDt -ne [datetime]::MinValue) -and ($modalRefDt -lt $procStartDt.AddMinutes(-1))) {
                        $modalDoneCanTrigger = $false
                        $modalDoneGuardReason = ("stale_modal_progress_guard:modal_ref={0}<proc_start={1}" -f $modalRefDt.ToString("yyyy-MM-ddTHH:mm:ss"), $procStartDt.ToString("yyyy-MM-ddTHH:mm:ss"))
                    }
                }
            }
            if (-not $modalDoneCanTrigger) {
                # Guard against stale modal state or startup race; skip quick-recycle trigger in this loop.
                $modalDoneStallEffectiveMinutes = [double]::NaN
            } else {
            $modalDoneStallThreshold = [Math]::Max(2, [int]$ModalDoneStallMinutes)
            $modalDoneStallEffective = $modalDoneStallThreshold
            # Enforce fast recycle when modal progress is already done/failed/timeout.
            # Do not extend this threshold with startup grace, worker keepalive, or cleanup heuristics.
            $modalVolumeCleanupActive = $false
            $modalDoneStallEffectiveMinutes = [double]$modalDoneStallEffective
            $modalDoneAgeMinutes = [double]::NaN
            if ($modalProgressUpdatedDt -ne [datetime]::MinValue) {
                $modalDoneAgeMinutes = ($now - $modalProgressUpdatedDt).TotalMinutes
            }
            $outLastDt = [datetime]::MinValue
            if ($outRealProgressAt -ne [datetime]::MinValue) {
                $outLastDt = $outRealProgressAt
            }
            # Use observed file write time as a fallback/upper-bound to avoid stale outRealProgressAt
            # causing false modal_done recycle loops.
            if ($outObservedWriteDt -ne [datetime]::MinValue) {
                if (($outLastDt -eq [datetime]::MinValue) -or ($outObservedWriteDt -gt $outLastDt)) {
                    $outLastDt = $outObservedWriteDt
                }
            } elseif (($outLastDt -eq [datetime]::MinValue) -and (-not [string]::IsNullOrWhiteSpace($oldStdout)) -and (Test-Path $oldStdout)) {
                try { $outLastDt = (Get-Item $oldStdout).LastWriteTime } catch {}
            }
            if ($outLastDt -ne [datetime]::MinValue) {
                $outIdleAfterModalDoneMinutes = ($now - $outLastDt).TotalMinutes
            } else {
                $outIdleAfterModalDoneMinutes = $modalDoneAgeMinutes
            }
            $modalDoneQuickGraceMinutes = [Math]::Max(0.5, ([double][Math]::Max(10, [int]$PollSec) / 60.0))
            $modalDoneAgeReached = ((-not [double]::IsNaN($modalDoneAgeMinutes)) -and ($modalDoneAgeMinutes -ge $modalDoneStallEffective))
            $outIdleReached = ((-not [double]::IsNaN($outIdleAfterModalDoneMinutes)) -and ($outIdleAfterModalDoneMinutes -ge $modalDoneStallEffective))
            $outAlreadyStaleWhenDone = ((-not [double]::IsNaN($modalDoneAgeMinutes)) -and ($modalDoneAgeMinutes -ge $modalDoneQuickGraceMinutes) -and $outIdleReached)
            if (($modalDoneAgeReached -and $outIdleReached) -or $outAlreadyStaleWhenDone) {
                $hit = $true
                $triggerReason = ("modal_done_no_progress_{0}m" -f [int][Math]::Round($outIdleAfterModalDoneMinutes))
            }
            }
        }
        if (-not $hit -and ($modalProgressState -eq "running") -and (-not [double]::IsNaN($modalProgressElapsedSec)) -and (-not [double]::IsNaN($modalProgressTimeoutSec)) -and ($modalProgressTimeoutSec -gt 0)) {
            $overtimeGraceSec = [Math]::Max(120.0, [double]([Math]::Max(1, [int]$PollSec) * 3))
            $modalProgressOverrunSec = $modalProgressElapsedSec - $modalProgressTimeoutSec
            if ($modalProgressOverrunSec -lt 0) {
                $modalProgressOverrunSec = [double]::NaN
            }
            if ((-not [double]::IsNaN($modalProgressOverrunSec)) -and ($modalProgressOverrunSec -ge $overtimeGraceSec)) {
                $hit = $true
                $triggerReason = ("modal_running_overtime_{0}s" -f [int][Math]::Round($modalProgressOverrunSec))
            }
        }

        if ($effectiveLastActivityDt -ne [datetime]::MinValue) {
            $idleMinutes = ($now - $effectiveLastActivityDt).TotalMinutes
            if ($idleMinutes -ge $stallThreshold) {
                $hit = $true
                $triggerReason = ("stalled_no_activity_{0}m" -f [int][Math]::Round($idleMinutes))
            }
        }
        if (-not $hit -and ($proc -ne $null) -and (-not [double]::IsNaN($startupAgeMinutes)) -and ($startupAgeMinutes -ge [double]$stageTimeoutMinutes)) {
            $hardTimeoutGrace = [Math]::Max(0, [int]$StageHardTimeoutGraceMinutes)
            $hardTimeoutReached = (([bool]$StageHardTimeoutEnabled) -and ($startupAgeMinutes -ge ([double]$stageTimeoutMinutes + [double]$hardTimeoutGrace)))
            if ($hardTimeoutReached) {
                $hit = $true
                $safeStage = ([string]$observedStage -replace "[^a-zA-Z0-9_]", "_")
                $triggerReason = ("stage_hard_timeout_{0}_{1}m_thr{2}m_g{3}m" -f $safeStage, [int][Math]::Round($startupAgeMinutes), [int]$stageTimeoutMinutes, [int]$hardTimeoutGrace)
            }
        }
        if (-not $hit -and ($proc -ne $null) -and (-not [double]::IsNaN($startupAgeMinutes)) -and ($startupAgeMinutes -ge [double]$stageTimeoutMinutes)) {
            $stageTimeoutIdleGraceMinutes = [Math]::Max(5, [Math]::Min(20, [int][Math]::Ceiling([double]$stageTimeoutMinutes / 6.0)))
            $idleForTimeout = [double]::NaN
            if ($effectiveLastActivityDt -ne [datetime]::MinValue) {
                $idleForTimeout = ($now - $effectiveLastActivityDt).TotalMinutes
            }
            $hasRecentActivity = ((-not [double]::IsNaN($idleForTimeout)) -and ($idleForTimeout -lt $stageTimeoutIdleGraceMinutes))
            if ($hasRecentActivity -or $workerSignalUsed -or $cpuSignalUsed) {
                # Stage is long but still active; do not force restart on pure wall-clock timeout.
            } else {
                $hit = $true
                $safeStage = ([string]$observedStage -replace "[^a-zA-Z0-9_]", "_")
                if (-not [double]::IsNaN($idleForTimeout)) {
                    $triggerReason = ("stage_timeout_{0}_{1}m_idle{2}m" -f $safeStage, [int][Math]::Round($startupAgeMinutes), [int][Math]::Round($idleForTimeout))
                } else {
                    $triggerReason = ("stage_timeout_{0}_{1}m" -f $safeStage, [int][Math]::Round($startupAgeMinutes))
                }
            }
        }
    }

    $state = if ($proc -ne $null) { "running" } else { "exited" }
    Write-JsonNoBom -Path $watchStatusPath -Obj ([ordered]@{
        updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        target_stage = $TargetStage
        state = $state
        pid = $oldPid
        hit_stage_boundary = $hitStageBoundary
        trigger_reason = $triggerReason
        best_ckpt = $bestCkpt
        forced_resume_ckpt = $effectiveForcedResumeCkpt
        last_activity_at = $activity.last_activity_at
        last_activity_source = $activity.last_activity_source
        effective_last_activity_at = $(if ($effectiveLastActivityDt -eq [datetime]::MinValue) { "" } else { $effectiveLastActivityDt.ToString("yyyy-MM-ddTHH:mm:ss") })
        effective_last_activity_source = $effectiveLastActivitySource
        idle_minutes = $idleMinutes
        cpu_delta_sec = $cpuDeltaSec
        cpu_active_delta_threshold_sec = $CpuActiveDeltaSec
        cpu_assist_window_minutes = $CpuAssistWindowMinutes
        cpu_signal_used = $cpuSignalUsed
        cpu_last_active_at = $(if ($lastCpuActiveAt -eq [datetime]::MinValue) { "" } else { $lastCpuActiveAt.ToString("yyyy-MM-ddTHH:mm:ss") })
        cpu_last_active_delta_sec = $lastCpuDeltaSec
        file_activity_age_minutes = $fileActivityAgeMinutes
        modal_progress_state = $modalProgressState
        modal_progress_updated_at = $modalProgressUpdatedAt
        modal_progress_started_at = $modalProgressStartedAt
        modal_progress_elapsed_sec = $modalProgressElapsedSec
        modal_progress_timeout_sec = $modalProgressTimeoutSec
        modal_progress_overrun_sec = $modalProgressOverrunSec
        modal_done_stall_minutes = $ModalDoneStallMinutes
        modal_done_stall_effective_minutes = $modalDoneStallEffectiveMinutes
        modal_done_guard_reason = $modalDoneGuardReason
        out_idle_after_modal_done_minutes = $outIdleAfterModalDoneMinutes
        out_real_progress_at = $(if ($outRealProgressAt -eq [datetime]::MinValue) { "" } else { $outRealProgressAt.ToString("yyyy-MM-ddTHH:mm:ss") })
        out_real_progress_signal = $outRealProgressSignal
        modal_volume_cleanup_active = $modalVolumeCleanupActive
        out_observed_write_at = $outObservedWriteAt
        out_observed_length = $outObservedLength
        observed_stage = $observedStage
        stage_timeout_minutes = $stageTimeoutMinutes
        stage1_timeout_minutes = $Stage1TimeoutMinutes
        other_stage_timeout_minutes = $OtherStageTimeoutMinutes
        stage_hard_timeout_enabled = [bool]$StageHardTimeoutEnabled
        stage_hard_timeout_grace_minutes = [int]$StageHardTimeoutGraceMinutes
        startup_grace_minutes = $StartupGraceMinutes
        startup_age_minutes = $startupAgeMinutes
        child_count = $childCount
        worker_count = $workerCount
        worker_names = $workerNames
        worker_process_names_config = ($workerNameSet -join ",")
        worker_attach_window_minutes = $WorkerAttachWindowMinutes
        worker_keepalive_max_minutes = $WorkerKeepAliveMaxMinutes
        single_chain_guard_enabled = [bool]$EnforceSingleMainChain
        single_chain_guard_poll_every = [int]$RedundantGuardPollEvery
        single_chain_guard_tick = $guardTick
        single_chain_guard_last_at = $lastGuardAt
        single_chain_guard_last_reason = $lastGuardReason
        single_chain_guard_local_cmd_count = $lastGuardLocalCmdCount
        single_chain_guard_local_killed = $lastGuardLocalKilled
        single_chain_guard_keep_cmd_pids = $lastGuardKeepCmdPids
        single_chain_guard_app_active_count = $lastGuardAppActiveCount
        single_chain_guard_app_stopped = $lastGuardAppStopped
        single_chain_guard_keep_app_ids = $lastGuardKeepAppIds
        worker_can_assist = $workerCanAssist
        worker_signal_used = $workerSignalUsed
        active_stall_threshold_minutes = $stallThreshold
        stall_minutes = $StallMinutes
        no_child_stall_minutes = $NoChildStallMinutes
        forced_stop_after_hours = $ForcedStopAfterHours
        main_final_deadline = $MainFinalDeadline
        exit_classification = $exitClass
        exit_detail = $exitDetail
        t0 = $(if ($windowT0 -eq [datetime]::MinValue) { "" } else { $windowT0.ToString("yyyy-MM-ddTHH:mm:ss") })
        t0_source = $t0Source
        deadline = $deadline.ToString("yyyy-MM-ddTHH:mm:ss")
        deadline_source = $deadlineSource
    })

    if ($hit) {
        $triggered = $true
        break
    }
    if ($proc -eq $null) {
        $triggered = $true
        $exitInfo = Get-ExitClassification `
            -StdoutPath $oldStdout `
            -StderrPath $oldErr `
            -ModalProgressState $modalProgressState
        $exitClass = [string]$exitInfo.class
        $exitDetail = [string]$exitInfo.detail
        $triggerReason = "pid_exited:$exitClass"
        break
    }
    Start-Sleep -Seconds ([Math]::Max(10, [int]$PollSec))
}

if (-not $triggered) {
    Write-JsonNoBom -Path $watchStatusPath -Obj ([ordered]@{
        updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        target_stage = $TargetStage
        state = "timeout"
        pid = $oldPid
        trigger_reason = "timeout"
        t0 = $(if ($windowT0 -eq [datetime]::MinValue) { "" } else { $windowT0.ToString("yyyy-MM-ddTHH:mm:ss") })
        t0_source = $t0Source
        deadline = $deadline.ToString("yyyy-MM-ddTHH:mm:ss")
        deadline_source = $deadlineSource
    })
    exit 0
}

if ([string]::IsNullOrWhiteSpace($bestCkpt)) {
    $bestCkpt = Get-FallbackBestCkpt
}
if (-not [string]::IsNullOrWhiteSpace($effectiveForcedResumeCkpt)) {
    $bestCkpt = $effectiveForcedResumeCkpt
}

if ([bool]$EnforceSingleMainChain) {
    $preGuard = Invoke-SingleChainGuard `
        -MainPid $oldPid `
        -ModalProgressPid 0 `
        -KeepNewestLocalCmd 0 `
        -KeepNewestActiveApps 0 `
        -Reason "pre_restart_cleanup"
    Write-Host "[hot-update] pre-restart cleanup local_killed=$($preGuard.local_killed) local_cmd_count=$($preGuard.local_cmd_count) app_stopped=$($preGuard.app_stopped) app_active_count=$($preGuard.app_active_count)"
}

if ($oldPid -gt 0) {
    $still = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
    if ($still -ne $null) {
        try { Stop-Process -Id $oldPid -Force } catch {}
    }
}

$newArgs = $oldArgs
if (-not [string]::IsNullOrWhiteSpace($bestCkpt)) {
    $newArgs = [regex]::Replace(
        $newArgs,
        "(?i)-StartResumeCkpt\s+(""[^""]+""|\S+)",
        "-StartResumeCkpt $bestCkpt"
    )
}
$newArgs = [regex]::Replace(
    $newArgs,
    "(?i)-StopAfterHours\s+(""[^""]+""|\S+)",
    "-StopAfterHours $ForcedStopAfterHours"
)
if ($newArgs -notmatch "(?i)-StopAfterHours\s+") {
    $newArgs = ($newArgs.TrimEnd() + " -StopAfterHours $ForcedStopAfterHours")
}
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2DualLaneEnabled" -Value "`$true"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-PostRescueEnabled" -Value "`$true"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-QualityGuardMode" -Value "layered"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-PromotionGhostDelta" -Value "0.02"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2EnableAnySplatAblationSixPack" -Value "`$true"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2EnableExtendedCkptWaitOnMissing" -Value "`$true"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2CkptExtendedWaitTimeoutSec" -Value "1200"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2EnableResumeCkptFallbackOnShortCkptMissing" -Value "`$false"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2DisallowResumeFallbackResult" -Value "`$true"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2EvalNumSrcViewsList" -Value "8,12,16,20,22"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2GramDynEnable" -Value "off"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2DynProxyEnable" -Value "on"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2DynProxyMode" -Value "fg_static_soft"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2DynProxyUseGram" -Value "on"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2DynProxyUseSupport" -Value "on"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2DynProxyFloor" -Value "0.35"
$newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-Stage2DynProxyWarmupSteps" -Value "40"
if (-not [string]::IsNullOrWhiteSpace($MainFinalDeadline)) {
    $newArgs = Set-OrAddLaunchArg -ArgsText $newArgs -Flag "-FinalDeadline" -Value $MainFinalDeadline
}
$newArgs = Normalize-BoolLaunchArgs -ArgsText $newArgs

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$newOut = "logs/modal_phase5/overnight_ghost_autoloop_$ts.out.log"
$newErr = "logs/modal_phase5/overnight_ghost_autoloop_$ts.err.log"
$newArgList = Convert-LaunchArgsToArray -ArgsText $newArgs
$p2 = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $newArgList `
    -WindowStyle Hidden `
    -RedirectStandardOutput $newOut `
    -RedirectStandardError $newErr `
    -PassThru

$newMeta = [ordered]@{
    started_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    pid = $p2.Id
    stdout = $newOut
    stderr = $newErr
    args = $newArgs
    restarted_from_pid = $oldPid
    restart_reason = "hot_update_after_stage_boundary:$triggerReason"
    hot_update_target_stage = $TargetStage
    hot_update_resume_ckpt = $bestCkpt
    hot_update_forced_resume_ckpt = $effectiveForcedResumeCkpt
    forced_stop_after_hours = $ForcedStopAfterHours
    main_final_deadline = $MainFinalDeadline
    trigger_exit_classification = $exitClass
    trigger_exit_detail = $exitDetail
}
Write-JsonNoBom -Path $launcherPath -Obj $newMeta

Write-JsonNoBom -Path $heartbeatPath -Obj ([ordered]@{
    updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    state = "restarted_hot_update"
    stage = $TargetStage
    trigger_reason = $triggerReason
    old_pid = $oldPid
    new_pid = $p2.Id
    resume_ckpt = $bestCkpt
    forced_stop_after_hours = $ForcedStopAfterHours
    main_final_deadline = $MainFinalDeadline
    exit_classification = $exitClass
    exit_detail = $exitDetail
})

$enc = New-Object System.Text.UTF8Encoding($false)
$append = @()
$append += ""
$append += "## $(Get-Date -Format 'yyyy-MM-dd') 热更新重启记录"
$append += "- updated_at: $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')"
$append += "- trigger: $triggerReason (stage boundary: $TargetStage)"
$append += "- old_pid: $oldPid"
$append += "- new_pid: $($p2.Id)"
$append += "- new_log: $newOut"
$append += "- resume_ckpt: $bestCkpt"
$append += "- forced_stop_after_hours: $ForcedStopAfterHours"
$append += "- main_final_deadline: $MainFinalDeadline"
if (-not [string]::IsNullOrWhiteSpace($exitClass)) {
    $append += "- exit_classification: $exitClass"
}
if (-not [string]::IsNullOrWhiteSpace($exitDetail)) {
    $append += "- exit_detail: $exitDetail"
}
$mentorPayload = Sanitize-TextForUtf8Log -Text (($append -join "`n") + "`n")
[System.IO.File]::AppendAllText((Join-Path (Resolve-Path ".").Path $mentorPath), $mentorPayload, $enc)

exit 0

