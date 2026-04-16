param(
    [string]$RepoDir = "F:\vggt",
    [string]$LauncherMetaPath = "logs/modal_phase5/overnight_ghost_autoloop_launcher_latest.json",
    [string]$HotUpdateMetaPath = "logs/modal_phase5/hot_update_autoloop_after_stage_latest.json",
    [string]$HeartbeatPath = "logs/modal_phase5/overnight_ghost_autoloop_heartbeat_latest.json",
    [string]$HotWatchPath = "logs/modal_phase5/overnight_ghost_autoloop_hotupdate_watch_latest.json",
    [string]$OutJsonPath = "logs/modal_phase5/watch_ghost_outputs_latest.json",
    [string]$OutMdPath = "logs/modal_phase5/watch_ghost_outputs_latest.md",
    [string]$InterimStatePath = "logs/modal_phase5/watch_ghost_outputs_interim_state_latest.json",
    [int]$PollSec = 20,
    [int]$WatchHours = 12,
    [double]$InterimGlobalBestMinPSNR = 20.9,
    [double]$InterimGlobalBestMinSSIM = 0.70,
    [double]$InterimGlobalBestMaxWl1 = 0.08,
    [double]$InterimStageBestMinPSNR = 20.9,
    [double]$InterimStageBestMinSSIM = 0.70,
    [double]$InterimStageBestMaxWl1 = 0.08,
    [bool]$InterimGlobalBestRequireCkpt = $true,
    [bool]$InterimGlobalBestRejectWatchSyncRows = $true,
    [bool]$AllowPersistBestFromInterim = $false,
    [switch]$RunOnce = $false
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

function Tail-Maybe([string]$Path, [int]$Lines = 30) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return @() }
    if (-not (Test-Path $Path)) { return @() }
    try {
        return @(
            Get-Content $Path -Tail $Lines |
                ForEach-Object { [string]$_ }
        )
    } catch {
        return @()
    }
}

function Proc-State([int]$ProcId) {
    if ($ProcId -le 0) {
        return [pscustomobject]@{
            pid = $ProcId
            state = "invalid"
            name = ""
            start_time = ""
            cpu = [double]::NaN
            ws_mb = [double]::NaN
        }
    }
    $p = Get-Process -Id $ProcId -ErrorAction SilentlyContinue
    if ($null -eq $p) {
        return [pscustomobject]@{
            pid = $ProcId
            state = "dead"
            name = ""
            start_time = ""
            cpu = [double]::NaN
            ws_mb = [double]::NaN
        }
    }
    return [pscustomobject]@{
        pid = $ProcId
        state = "alive"
        name = $p.ProcessName
        start_time = $p.StartTime.ToString("yyyy-MM-ddTHH:mm:ss")
        cpu = [double]$p.CPU
        ws_mb = [double]($p.WorkingSet64 / 1MB)
    }
}

function File-InfoMaybe([string]$Path) {
    if (-not (Test-Path $Path)) {
        return [pscustomobject]@{
            path = $Path
            exists = $false
            last_write = ""
            length = 0
        }
    }
    $it = Get-Item $Path
    return [pscustomobject]@{
        path = $Path
        exists = $true
        last_write = $it.LastWriteTime.ToString("yyyy-MM-ddTHH:mm:ss")
        length = [int64]$it.Length
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

function Parse-DateMaybe([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return [datetime]::MinValue }
    try { return [datetime]::Parse($Value) } catch { return [datetime]::MinValue }
}

function Normalize-TimeText([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return "" }
    $dt = Parse-DateMaybe -Value $Value
    if ($dt -eq [datetime]::MinValue) { return [string]$Value }
    return $dt.ToString("yyyy-MM-ddTHH:mm:ss")
}

function Meta-Slim([object]$M) {
    if ($null -eq $M) {
        return [pscustomobject]@{}
    }
    return [pscustomobject]@{
        started_at = Normalize-TimeText -Value ([string]$M.started_at)
        pid = [string]$M.pid
        stdout = [string]$M.stdout
        stderr = [string]$M.stderr
        target_stage = [string]$M.target_stage
        restart_reason = [string]$M.restart_reason
        note = [string]$M.note
    }
}

function To-DoubleOrNaN($Value) {
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) { return [double]::NaN }
    try { return [double]$Value } catch { return [double]::NaN }
}

function To-BoolLoose($Value) {
    if ($null -eq $Value) { return $false }
    if ($Value -is [bool]) { return [bool]$Value }
    $s = [string]$Value
    if ([string]::IsNullOrWhiteSpace($s)) { return $false }
    $s = $s.Trim().ToLowerInvariant()
    if ($s -match '^(1|true|yes|y|on)$') { return $true }
    if ($s -match '^(0|false|no|n|off)$') { return $false }
    try { return ([double]$s -ne 0.0) } catch { return $false }
}

function Parse-CycleFromStage([string]$Stage) {
    if ([string]::IsNullOrWhiteSpace($Stage)) { return 0 }
    $m = [regex]::Match($Stage, "cycle(\d+)_")
    if (-not $m.Success) { return 0 }
    try { return [int]$m.Groups[1].Value } catch { return 0 }
}

function Get-QualityGuardReason(
    [object]$Row,
    [double]$MinPSNR,
    [double]$MinSSIM,
    [double]$MaxWl1
) {
    $reasons = New-Object System.Collections.Generic.List[string]
    $p = To-DoubleOrNaN($Row.mean_PSNR)
    $s = To-DoubleOrNaN($Row.mean_SSIM)
    $w = To-DoubleOrNaN($Row.mean_weighted_L1)
    if ([double]::IsNaN($p) -or ($p -lt [double]$MinPSNR)) {
        $reasons.Add("psnr<$([double]$MinPSNR) (now=$p)") | Out-Null
    }
    if (([double]$MinSSIM -gt 0) -and ([double]::IsNaN($s) -or ($s -lt [double]$MinSSIM))) {
        $reasons.Add("ssim<$([double]$MinSSIM) (now=$s)") | Out-Null
    }
    if (([double]$MaxWl1 -gt 0) -and ([double]::IsNaN($w) -or ($w -gt [double]$MaxWl1))) {
        $reasons.Add("wl1>$([double]$MaxWl1) (now=$w)") | Out-Null
    }
    if ($reasons.Count -le 0) { return "" }
    return ($reasons -join "; ")
}

$script:GhostAutoloopSchema = @(
    "cycle","stage","policy","pointmap_source","point_target_mode",
    "lambda_point_mv_depth_list","lambda_point_mv_mask_list","lambda_point",
    "point_mv_mask_hit_thr","point_mv_mask_min_tgt_fg_ratio","point_mv_mask_soft_mix","point_mv_mask_soft_hit_thr",
    "point_mv_stride","point_mv_depth_max_pairs","point_mv_depth_support_mode","point_mv_depth_support_floor",
    "point_cons_focus","point_residual_focus","rc","ghost","psnr","ssim","wl1","best_geom","best_ckpt",
    "best_lambda_point_mv_depth","best_lambda_point_mv_mask","best_ghost_rows_csv","best_visual_png","stage_best_strip_png","stage_skip_reason",
    "best_ghost_width_ratio","best_ghost_area_ratio","best_ghost_peak_count","best_ghost_center_offset_ratio","sweep_csv",
    "cycle_substantial_improved","cycle_regressed","cycle_regress_reason","cycle_quality_guard_blocked","cycle_quality_guard_reason","regress_cycles","rolled_back_last_tune",
    "rolled_back_tune_action","tune_action_next","cycle_compare_png","resume_update_reason","updated_at"
)

function New-GhostAutoloopRow([hashtable]$Data) {
    $o = [ordered]@{}
    foreach ($k in $script:GhostAutoloopSchema) {
        if ($Data.ContainsKey($k)) {
            $o[$k] = $Data[$k]
        } else {
            $o[$k] = ""
        }
    }
    return [pscustomobject]$o
}

function Export-CsvUtf8NoBom([string]$Path, [object[]]$Rows) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    if ($null -eq $Rows -or $Rows.Count -eq 0) {
        [System.IO.File]::WriteAllText((Join-Path (Resolve-Path ".").Path $Path), "", $enc)
        return
    }
    $tmp = [System.IO.Path]::GetTempFileName()
    try {
        $Rows | Export-Csv $tmp -NoTypeInformation -Encoding UTF8
        $content = Get-Content $tmp -Raw
        [System.IO.File]::WriteAllText((Join-Path (Resolve-Path ".").Path $Path), $content, $enc)
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
}

function Upsert-InterimAutoloopFiles(
    [object]$Heartbeat,
    [object]$Launcher,
    [datetime]$Deadline
) {
    if ($null -eq $Heartbeat) { return $null }
    $hbState = [string]$Heartbeat.state
    $stage = [string]$Heartbeat.stage
    if ([string]::IsNullOrWhiteSpace($stage)) { return $null }
    if ($hbState -ne "running_stage") { return $null }

    $sweepCsv = "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv"
    if (-not (Test-Path $sweepCsv)) { return $null }
    $launcherStartedAt = [datetime]::MinValue
    if ($Launcher -ne $null) {
        $launcherStartedAt = Parse-DateMaybe -Value ([string]$Launcher.started_at)
    }
    if ($launcherStartedAt -ne [datetime]::MinValue) {
        $sweepInfo = Get-Item $sweepCsv -ErrorAction SilentlyContinue
        if ($null -ne $sweepInfo -and $sweepInfo.LastWriteTime -lt $launcherStartedAt.AddSeconds(-2)) {
            # Prevent previous-window/latest-file residue from being mis-synced into the new run.
            return $null
        }
    }
    $rows = @(Import-Csv $sweepCsv)
    if ($rows.Count -le 0) { return $null }

    # Keep only rows that belong to the currently running stage.
    # `ghost_mvdepth_sweep_latest.csv` can transiently contain previous-stage rows.
    $stageRows = @($rows)
    $targetPolicy = [string]$Heartbeat.policy
    $targetSource = [string]$Heartbeat.pointmap_source
    $hasPolicyCol = ($rows[0].PSObject.Properties.Name -contains "point_target_blend_mv_policy")
    $hasSourceCol = ($rows[0].PSObject.Properties.Name -contains "pointmap_source")

    if ($hasPolicyCol -and (-not [string]::IsNullOrWhiteSpace($targetPolicy))) {
        $policyRows = @(
            $stageRows |
                Where-Object { [string]$_.point_target_blend_mv_policy -eq $targetPolicy }
        )
        if ($policyRows.Count -le 0) {
            # Stage just switched and no current-stage rows landed yet.
            return $null
        }
        $stageRows = $policyRows
    }

    if ($hasSourceCol -and (-not [string]::IsNullOrWhiteSpace($targetSource))) {
        $sourceRows = @(
            $stageRows |
                Where-Object { [string]$_.pointmap_source -eq $targetSource }
        )
        if ($sourceRows.Count -le 0) {
            return $null
        }
        $stageRows = $sourceRows
    }

    if ($stageRows.Count -le 0) { return $null }

    $stageRows = @(
        $stageRows | ForEach-Object {
            $guardReason = ""
            $guardBlocked = $false
            $hasExplicitGuard = $false
            if ($_.PSObject.Properties["quality_guard_blocked"]) {
                $guardBlocked = To-BoolLoose($_.quality_guard_blocked)
                $hasExplicitGuard = $true
            }
            if ($_.PSObject.Properties["quality_guard_reason"]) {
                $guardReason = [string]$_.quality_guard_reason
                if (-not [string]::IsNullOrWhiteSpace($guardReason)) {
                    $hasExplicitGuard = $true
                }
            }
            if (-not $hasExplicitGuard) {
                $guardReason = Get-QualityGuardReason `
                    -Row $_ `
                    -MinPSNR ([double]$InterimStageBestMinPSNR) `
                    -MinSSIM ([double]$InterimStageBestMinSSIM) `
                    -MaxWl1 ([double]$InterimStageBestMaxWl1)
                $guardBlocked = (-not [string]::IsNullOrWhiteSpace([string]$guardReason))
            }
            $_ | Add-Member -NotePropertyName interim_quality_guard_reason -NotePropertyValue ([string]$guardReason) -Force
            $_ | Add-Member -NotePropertyName interim_quality_guard_blocked -NotePropertyValue ([bool]$guardBlocked) -Force
            $_
        }
    )

    $cycle = Parse-CycleFromStage -Stage $stage
    $updatedAt = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    $validRows = @(
        $stageRows |
            Where-Object {
                (-not [double]::IsNaN((To-DoubleOrNaN($_.ghost_score_mean)))) -and
                (-not (To-BoolLoose($_.interim_quality_guard_blocked)))
            }
    )
    if ($validRows.Count -le 0) {
        $validRows = @(
            $stageRows |
                Where-Object {
                    -not [double]::IsNaN((To-DoubleOrNaN($_.ghost_score_mean)))
                }
        )
        if ($validRows.Count -gt 0) {
            Write-Host "[watch-ghost][warn] interim stage-best strict quality guard yielded 0 rows; fallback to ghost-only rows."
        }
    }
    $bestRow = $null
    if ($validRows.Count -gt 0) {
        $bestRow = $validRows |
            Sort-Object @{ Expression = { To-DoubleOrNaN($_.ghost_score_mean) }; Ascending = $true } |
            Select-Object -First 1
    } else {
        $bestRow = $stageRows | Select-Object -First 1
    }
    $latestRow = $stageRows | Select-Object -Last 1
    $bestGhost = To-DoubleOrNaN($bestRow.ghost_score_mean)
    $bestPsnr = To-DoubleOrNaN($bestRow.mean_PSNR)
    $bestWl1 = To-DoubleOrNaN($bestRow.mean_weighted_L1)
    $bestSsim = To-DoubleOrNaN($bestRow.mean_SSIM)

    $existingPath = "logs/modal_phase5/ghost_autoloop_latest.csv"
    $existingRows = @()
    if (Test-Path $existingPath) {
        try { $existingRows = @(Import-Csv $existingPath) } catch { $existingRows = @() }
    }

    $keptRows = @()
    foreach ($er in $existingRows) {
        if ($launcherStartedAt -ne [datetime]::MinValue) {
            $erTs = Parse-DateMaybe -Value ([string]$er.updated_at)
            if (($erTs -ne [datetime]::MinValue) -and ($erTs -lt $launcherStartedAt.AddMinutes(-1))) {
                continue
            }
        }
        $sameCycle = ([string]$er.cycle -eq [string]$cycle)
        $sameStage = ([string]$er.stage -eq $stage)
        if (-not ($sameCycle -and $sameStage)) {
            $map = @{}
            foreach ($k in $script:GhostAutoloopSchema) {
                $map[$k] = [string]$er.$k
            }
            $keptRows += New-GhostAutoloopRow -Data $map
        }
    }

    $newRows = @()
    foreach ($r in $stageRows) {
        $map = @{
            cycle = $cycle
            stage = $stage
            policy = [string]$Heartbeat.policy
            pointmap_source = [string]$Heartbeat.pointmap_source
            point_target_mode = [string]$Heartbeat.point_target_mode
            lambda_point_mv_depth_list = [string]$Heartbeat.lambda_point_mv_depth_list
            lambda_point_mv_mask_list = [string]$Heartbeat.lambda_point_mv_mask_list
            lambda_point = [string]$Heartbeat.lambda_point
            point_mv_mask_hit_thr = [string]$Heartbeat.point_mv_mask_hit_thr
            point_mv_mask_min_tgt_fg_ratio = [string]$Heartbeat.point_mv_mask_min_tgt_fg_ratio
            point_mv_mask_soft_mix = [string]$Heartbeat.point_mv_mask_soft_mix
            point_mv_mask_soft_hit_thr = [string]$Heartbeat.point_mv_mask_soft_hit_thr
            point_mv_stride = [string]$Heartbeat.point_mv_stride
            point_mv_depth_max_pairs = [string]$Heartbeat.point_mv_depth_max_pairs
            point_mv_depth_support_mode = [string]$Heartbeat.point_mv_depth_support_mode
            point_mv_depth_support_floor = [string]$Heartbeat.point_mv_depth_support_floor
            point_cons_focus = [string]$Heartbeat.point_cons_focus
            point_residual_focus = [string]$Heartbeat.point_residual_focus
            rc = [string]$r.exit_code
            ghost = [string]$r.ghost_score_mean
            psnr = [string]$r.mean_PSNR
            ssim = [string]$r.mean_SSIM
            wl1 = [string]$r.mean_weighted_L1
            best_geom = [string]$r.best_geom_subdir
            best_ckpt = ""
            best_lambda_point_mv_depth = [string]$r.lambda_point_mv_depth
            best_lambda_point_mv_mask = [string]$r.lambda_point_mv_mask
            best_ghost_rows_csv = [string]$r.ghost_rows_csv
            best_visual_png = ""
            stage_best_strip_png = ""
            stage_skip_reason = ""
            best_ghost_width_ratio = ""
            best_ghost_area_ratio = ""
            best_ghost_peak_count = ""
            best_ghost_center_offset_ratio = ""
            sweep_csv = [string]$r.sweep_csv
            cycle_substantial_improved = ""
            cycle_regressed = ""
            cycle_regress_reason = ""
            cycle_quality_guard_blocked = $(if (To-BoolLoose($r.interim_quality_guard_blocked)) { "true" } else { "false" })
            cycle_quality_guard_reason = [string]$r.interim_quality_guard_reason
            regress_cycles = ""
            rolled_back_last_tune = ""
            rolled_back_tune_action = ""
            tune_action_next = "pending_stage_complete"
            cycle_compare_png = ""
            resume_update_reason = "interim_watch_sync"
            updated_at = $updatedAt
        }
        $newRows += New-GhostAutoloopRow -Data $map
    }

    $mergedRows = @($keptRows + $newRows)
    Export-CsvUtf8NoBom -Path $existingPath -Rows $mergedRows

    # Compute true global-best from merged autoloop rows in current window,
    # so stage-local interim best does not overwrite historical best in this run.
    $globalBestRow = $null
    $globalBestGhost = [double]::NaN
    $globalBestPsnr = [double]::NaN
    $globalBestWl1 = [double]::NaN
    $globalBestSsim = [double]::NaN
    $globalBestStage = ""
    $globalBestGeom = ""
    $globalBestDepth = ""
    $globalBestMask = ""
    try {
        $globalCandidates = @(
            $mergedRows | Where-Object {
                $g = To-DoubleOrNaN($_.ghost)
                if ([double]::IsNaN($g)) { return $false }
                if ([bool]$InterimGlobalBestRequireCkpt -and [string]::IsNullOrWhiteSpace([string]$_.best_ckpt)) { return $false }
                if ([bool]$InterimGlobalBestRejectWatchSyncRows -and ([string]$_.resume_update_reason -eq "interim_watch_sync")) { return $false }
                $tier = ([string]$_.guard_tier).Trim().ToLowerInvariant()
                if ($tier -eq "blocked") { return $false }
                if (To-BoolLoose($_.cycle_quality_guard_blocked)) { return $false }
                $p = To-DoubleOrNaN($_.psnr)
                $s = To-DoubleOrNaN($_.ssim)
                $w = To-DoubleOrNaN($_.wl1)
                if ([double]::IsNaN($p) -or ($p -lt [double]$InterimGlobalBestMinPSNR)) { return $false }
                if ([double]::IsNaN($s) -or ($s -lt [double]$InterimGlobalBestMinSSIM)) { return $false }
                if ([double]::IsNaN($w) -or ($w -gt [double]$InterimGlobalBestMaxWl1)) { return $false }
                return $true
            }
        )
        if ($globalCandidates.Count -le 0) {
            $globalCandidates = @(
                $mergedRows | Where-Object {
                    (-not [double]::IsNaN((To-DoubleOrNaN($_.ghost)))) -and
                    (-not [string]::IsNullOrWhiteSpace([string]$_.best_ckpt))
                }
            )
            if ($globalCandidates.Count -gt 0) {
                Write-Host "[watch-ghost][warn] interim global-best strict filter yielded 0 rows; fallback to ckpt-only rows."
            }
        }
        if ($globalCandidates.Count -gt 0) {
            $globalBestRow = $globalCandidates |
                Sort-Object @{ Expression = { To-DoubleOrNaN($_.ghost) }; Ascending = $true } |
                Select-Object -First 1
            if ($null -ne $globalBestRow) {
                $globalBestGhost = To-DoubleOrNaN($globalBestRow.ghost)
                $globalBestPsnr = To-DoubleOrNaN($globalBestRow.psnr)
                $globalBestWl1 = To-DoubleOrNaN($globalBestRow.wl1)
                $globalBestSsim = To-DoubleOrNaN($globalBestRow.ssim)
                $globalBestStage = [string]$globalBestRow.stage
                $globalBestGeom = [string]$globalBestRow.best_geom
                $globalBestDepth = [string]$globalBestRow.best_lambda_point_mv_depth
                $globalBestMask = [string]$globalBestRow.best_lambda_point_mv_mask
            }
        }
    } catch {
    }

    $globalBestCkpt = ""
    $globalBestRowsCsv = ""
    $globalBestVisual = ""
    if ($globalBestRow -ne $null) {
        $globalBestCkpt = [string]$globalBestRow.best_ckpt
        $globalBestRowsCsv = [string]$globalBestRow.best_ghost_rows_csv
        $globalBestVisual = [string]$globalBestRow.best_visual_png
        if ([string]::IsNullOrWhiteSpace($globalBestCkpt) -or [string]::IsNullOrWhiteSpace($globalBestRowsCsv) -or [string]::IsNullOrWhiteSpace($globalBestVisual)) {
            try {
                $sameGhost = @(
                    $globalCandidates |
                        Where-Object {
                            (-not [double]::IsNaN((To-DoubleOrNaN($_.ghost)))) -and
                            ([Math]::Abs((To-DoubleOrNaN($_.ghost)) - $globalBestGhost) -le 1e-9)
                        }
                )
                if ($sameGhost.Count -gt 0) {
                    if ([string]::IsNullOrWhiteSpace($globalBestCkpt)) {
                        $hitCkpt = @($sameGhost | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_.best_ckpt) } | Select-Object -First 1)
                        if ($hitCkpt.Count -gt 0) { $globalBestCkpt = [string]$hitCkpt[0].best_ckpt }
                    }
                    if ([string]::IsNullOrWhiteSpace($globalBestRowsCsv)) {
                        $hitRows = @($sameGhost | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_.best_ghost_rows_csv) } | Select-Object -First 1)
                        if ($hitRows.Count -gt 0) { $globalBestRowsCsv = [string]$hitRows[0].best_ghost_rows_csv }
                    }
                    if ([string]::IsNullOrWhiteSpace($globalBestVisual)) {
                        $hitVisual = @($sameGhost | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_.best_visual_png) } | Select-Object -First 1)
                        if ($hitVisual.Count -gt 0) { $globalBestVisual = [string]$hitVisual[0].best_visual_png }
                    }
                }
            } catch {
            }
        }
    }

    $persistBestPath = "logs/modal_phase5/ghost_global_best_latest.json"
    $persistBest = Read-JsonMaybe -Path $persistBestPath
    $persistGhost = [double]::NaN
    if ($persistBest -ne $null) {
        $persistSource = [string]$persistBest.source
        if ($persistSource -eq "watch_interim_sync_anyrow") {
            $persistGhost = [double]::NaN
        } else {
            $persistGhost = To-DoubleOrNaN($persistBest.global_best_ghost)
        }
    }
    if (($persistBest -ne $null) -and (-not [double]::IsNaN($persistGhost)) -and ([double]::IsNaN($globalBestGhost) -or ($persistGhost -lt $globalBestGhost))) {
        $globalBestGhost = $persistGhost
        $globalBestPsnr = To-DoubleOrNaN($persistBest.global_best_psnr)
        $globalBestSsim = To-DoubleOrNaN($persistBest.global_best_ssim)
        $globalBestWl1 = To-DoubleOrNaN($persistBest.global_best_wl1)
        $globalBestStage = [string]$persistBest.global_best_stage
        $globalBestGeom = [string]$persistBest.global_best_geom
        $globalBestDepth = [string]$persistBest.global_best_lambda_point_mv_depth
        $globalBestMask = [string]$persistBest.global_best_lambda_point_mv_mask
        $globalBestCkpt = [string]$persistBest.best_ckpt
        $globalBestRowsCsv = [string]$persistBest.best_ghost_rows_csv
        $globalBestVisual = [string]$persistBest.best_visual_png
    } else {
        $canPersistCurrentBest = [bool]$AllowPersistBestFromInterim -and ($globalBestRow -ne $null) -and (-not [double]::IsNaN($globalBestGhost))
        if ($canPersistCurrentBest) {
            if ([string]::IsNullOrWhiteSpace($globalBestCkpt) -and ($persistBest -ne $null)) {
                $persistBestGhostNow = To-DoubleOrNaN($persistBest.global_best_ghost)
                if ((-not [double]::IsNaN($persistBestGhostNow)) -and ([Math]::Abs($persistBestGhostNow - $globalBestGhost) -le 1e-9)) {
                    $globalBestCkpt = [string]$persistBest.best_ckpt
                    if ([string]::IsNullOrWhiteSpace($globalBestRowsCsv)) { $globalBestRowsCsv = [string]$persistBest.best_ghost_rows_csv }
                    if ([string]::IsNullOrWhiteSpace($globalBestVisual)) { $globalBestVisual = [string]$persistBest.best_visual_png }
                }
            }
            $persistObj = [ordered]@{
                updated_at = $updatedAt
                global_best_ghost = $globalBestGhost
                global_best_psnr = $globalBestPsnr
                global_best_ssim = $globalBestSsim
                global_best_wl1 = $globalBestWl1
                global_best_stage = $globalBestStage
                global_best_geom = $globalBestGeom
                global_best_lambda_point_mv_depth = $globalBestDepth
                global_best_lambda_point_mv_mask = $globalBestMask
                best_ckpt = $globalBestCkpt
                best_ghost_rows_csv = $globalBestRowsCsv
                best_visual_png = $globalBestVisual
                source = "watch_interim_sync_anyrow"
            }
            Write-JsonNoBom -Path $persistBestPath -Obj $persistObj
        }
    }

    $resumeCkpt = ""
    if ($Launcher -ne $null) {
        $resumeCkpt = [string]$Launcher.hot_update_forced_resume_ckpt
        if ([string]::IsNullOrWhiteSpace($resumeCkpt)) {
            $resumeCkpt = [string]$Launcher.hot_update_resume_ckpt
        }
    }
    if ([string]::IsNullOrWhiteSpace($resumeCkpt)) {
        $resumeCkpt = [string]$Heartbeat.resume_ckpt_in
    }
    $pseudo = [string]$Heartbeat.pseudo_geom_in

    $prevStatusPath = "logs/modal_phase5/overnight_ghost_autoloop_latest.json"
    $prevStatus = Read-JsonMaybe -Path $prevStatusPath
    $candidateResultLatest = Read-JsonMaybe -Path "logs/modal_phase5/candidate_result_latest.json"
    $activeLane = [string]$Heartbeat.lane_id
    if ([string]::IsNullOrWhiteSpace($activeLane) -and $prevStatus -ne $null) {
        $activeLane = [string]$prevStatus.active_lane
    }
    if ([string]::IsNullOrWhiteSpace($activeLane)) {
        $activeLane = "lane_a"
    }
    $guardTier = [string]$Heartbeat.guard_tier
    if ([string]::IsNullOrWhiteSpace($guardTier) -and $prevStatus -ne $null) {
        $guardTier = [string]$prevStatus.guard_tier
    }
    $decisionReason = [string]$Heartbeat.decision_reason
    if ([string]::IsNullOrWhiteSpace($decisionReason) -and $prevStatus -ne $null) {
        $decisionReason = [string]$prevStatus.decision_reason
    }
    if ([string]::IsNullOrWhiteSpace($decisionReason)) {
        $decisionReason = "watch_sync_pending_stage_complete"
    }
    $rollbackReason = [string]$Heartbeat.rollback_reason
    if ([string]::IsNullOrWhiteSpace($rollbackReason) -and $prevStatus -ne $null) {
        $rollbackReason = [string]$prevStatus.rollback_reason
    }
    $activePrecomputeMvSupportOn = ""
    $activePointTargetBlendByMvSupport = ""
    $activeCandidateResultJson = ""
    if ($candidateResultLatest -ne $null) {
        if ($candidateResultLatest.PSObject.Properties["precompute_mv_support_on"]) {
            $activePrecomputeMvSupportOn = [string]$candidateResultLatest.precompute_mv_support_on
        }
        if ($candidateResultLatest.PSObject.Properties["point_target_blend_by_mv_support"]) {
            $activePointTargetBlendByMvSupport = [string]$candidateResultLatest.point_target_blend_by_mv_support
        }
        $activeCandidateResultJson = "logs/modal_phase5/candidate_result_latest.json"
    }
    if ($latestRow -ne $null) {
        if ([string]::IsNullOrWhiteSpace($activePrecomputeMvSupportOn)) {
            $activePrecomputeMvSupportOn = [string]$latestRow.precompute_mv_support_on
        }
        if ([string]::IsNullOrWhiteSpace($activePointTargetBlendByMvSupport)) {
            $activePointTargetBlendByMvSupport = [string]$latestRow.point_target_blend_by_mv_support
        }
    }
    if ([string]::IsNullOrWhiteSpace($activePrecomputeMvSupportOn) -and ($Heartbeat -ne $null)) {
        $activePrecomputeMvSupportOn = [string]$Heartbeat.precompute_mv_support_on
    }
    if ([string]::IsNullOrWhiteSpace($activePointTargetBlendByMvSupport) -and ($Heartbeat -ne $null)) {
        $activePointTargetBlendByMvSupport = [string]$Heartbeat.point_target_blend_by_mv_support
    }
    $laneABest = $null
    $laneBBest = $null
    if ($prevStatus -ne $null) {
        $laneABest = $prevStatus.lane_a_best
        $laneBBest = $prevStatus.lane_b_best
    }

    $status = [ordered]@{
        updated_at = $updatedAt
        deadline = $Deadline.ToString("yyyy-MM-ddTHH:mm:ss")
        current_cycle = $cycle
        current_stage = $stage
        interim = $true
        active_lane = $activeLane
        lane_a_best = $laneABest
        lane_b_best = $laneBBest
        guard_tier = $guardTier
        decision_reason = $decisionReason
        rollback_reason = $rollbackReason
        current_resume_ckpt = $resumeCkpt
        current_pseudo_geom = $pseudo
        global_best_ghost = $globalBestGhost
        global_best_psnr = $globalBestPsnr
        global_best_ssim = $globalBestSsim
        global_best_wl1 = $globalBestWl1
        global_best_stage = $globalBestStage
        global_best_geom = $globalBestGeom
        global_best_lambda_point_mv_depth = $globalBestDepth
        global_best_lambda_point_mv_mask = $globalBestMask
        stage_partial_rows = $stageRows.Count
        stage_best_ghost = $bestGhost
        stage_best_psnr = $bestPsnr
        stage_best_ssim = $bestSsim
        stage_best_wl1 = $bestWl1
        stage_best_quality_guard_blocked = $(if ($bestRow -ne $null) { To-BoolLoose($bestRow.interim_quality_guard_blocked) } else { $false })
        stage_best_quality_guard_reason = $(if ($bestRow -ne $null) { [string]$bestRow.interim_quality_guard_reason } else { "" })
        best_lambda_point_mv_depth = [string]$bestRow.lambda_point_mv_depth
        best_lambda_point_mv_mask = [string]$bestRow.lambda_point_mv_mask
        active_precompute_mv_support_on = $activePrecomputeMvSupportOn
        active_point_target_blend_by_mv_support = $activePointTargetBlendByMvSupport
        active_candidate_result_json = $activeCandidateResultJson
        next_cycle_tune_action = "pending_stage_complete"
        # Keep JSON-safe ASCII text here to avoid codepage corruption in Windows PowerShell.
        note = "watch_sync_interim_stage_rows"
    }
    Write-JsonNoBom -Path "logs/modal_phase5/overnight_ghost_autoloop_latest.json" -Obj $status

    $md = @()
    $md += "# 过夜 Ghost AutoLoop（interim/watch-sync）"
    $md += ""
    $md += "- updated: $($status.updated_at)"
    $md += "- deadline: $($status.deadline)"
    $md += "- cycle: $($status.current_cycle)"
    $md += "- stage: $($status.current_stage)"
    $md += "- active_lane: $($status.active_lane)"
    $md += "- guard_tier: $($status.guard_tier)"
    $md += "- decision_reason: $($status.decision_reason)"
    $md += "- stage_partial_rows: $($status.stage_partial_rows)"
    $md += "- global_best_ghost: $($status.global_best_ghost)"
    $md += "- global_best_psnr: $($status.global_best_psnr)"
    $md += "- global_best_ssim: $($status.global_best_ssim)"
    $md += "- global_best_wl1: $($status.global_best_wl1)"
    $md += "- global_best_stage: $($status.global_best_stage)"
    $md += "- global_best_lambda_point_mv_depth: $($status.global_best_lambda_point_mv_depth)"
    $md += "- global_best_lambda_point_mv_mask: $($status.global_best_lambda_point_mv_mask)"
    $md += "- stage_best_ghost: $($status.stage_best_ghost)"
    $md += "- stage_best_psnr: $($status.stage_best_psnr)"
    $md += "- stage_best_ssim: $($status.stage_best_ssim)"
    $md += "- stage_best_wl1: $($status.stage_best_wl1)"
    $md += "- stage_best_quality_guard_blocked: $($status.stage_best_quality_guard_blocked)"
    $md += "- stage_best_quality_guard_reason: $($status.stage_best_quality_guard_reason)"
    $md += "- best_lambda_point_mv_depth: $($status.best_lambda_point_mv_depth)"
    $md += "- best_lambda_point_mv_mask: $($status.best_lambda_point_mv_mask)"
    $md += "- active_precompute_mv_support_on: $($status.active_precompute_mv_support_on)"
    $md += "- active_point_target_blend_by_mv_support: $($status.active_point_target_blend_by_mv_support)"
    $md += "- active_candidate_result_json: $($status.active_candidate_result_json)"
    $md += "- global_best_geom: $($status.global_best_geom)"
    $md += "- note: stage 运行中，已基于当前产物自动同步。"
    Set-Content -Path "logs/modal_phase5/overnight_ghost_autoloop_latest.md" -Value ($md -join "`n") -Encoding UTF8

    return [pscustomobject]@{
        synced = $true
        signature = ("{0}|{1}|{2}|{3}|{4}" -f $stage, $stageRows.Count, [string]$bestRow.ghost_score_mean, [string]$latestRow.lambda_point_mv_depth, [string]$latestRow.lambda_point_mv_mask)
        stage = $stage
        cycle = $cycle
        rows = $stageRows.Count
        best_ghost = $bestGhost
        best_psnr = $bestPsnr
        best_ssim = $bestSsim
        best_wl1 = $bestWl1
        best_depth = [string]$bestRow.lambda_point_mv_depth
        best_mask = [string]$bestRow.lambda_point_mv_mask
        latest_depth = [string]$latestRow.lambda_point_mv_depth
        latest_mask = [string]$latestRow.lambda_point_mv_mask
        latest_ghost = To-DoubleOrNaN($latestRow.ghost_score_mean)
        latest_psnr = To-DoubleOrNaN($latestRow.mean_PSNR)
        latest_ssim = To-DoubleOrNaN($latestRow.mean_SSIM)
        latest_wl1 = To-DoubleOrNaN($latestRow.mean_weighted_L1)
        updated_at = $updatedAt
    }
}

function Ensure-AutoloopBootstrapStatus(
    [object]$Heartbeat,
    [object]$Launcher,
    [datetime]$Deadline
) {
    if ($null -eq $Launcher) { return $false }
    $launcherStartedAt = Parse-DateMaybe -Value ([string]$Launcher.started_at)
    if ($launcherStartedAt -eq [datetime]::MinValue) { return $false }

    $statusPath = "logs/modal_phase5/overnight_ghost_autoloop_latest.json"
    $statusMdPath = "logs/modal_phase5/overnight_ghost_autoloop_latest.md"
    $statusObj = Read-JsonMaybe -Path $statusPath
    $statusUpdatedAt = [datetime]::MinValue
    if ($statusObj -ne $null) {
        $statusUpdatedAt = Parse-DateMaybe -Value ([string]$statusObj.updated_at)
    }
    if (($statusUpdatedAt -ne [datetime]::MinValue) -and ($statusUpdatedAt -ge $launcherStartedAt.AddSeconds(-2))) {
        if (($Heartbeat -ne $null) -and ([string]$Heartbeat.state -eq "running_stage")) {
            $hbStage = [string]$Heartbeat.stage
            $currStage = [string]$statusObj.current_stage
            if ((-not [string]::IsNullOrWhiteSpace($hbStage)) -and ($hbStage -ne $currStage)) {
                $statusObj.updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
                $statusObj.current_stage = $hbStage
                $statusObj.current_cycle = [Math]::Max(1, (Parse-CycleFromStage -Stage $hbStage))
                $statusObj.current_resume_ckpt = [string]$Heartbeat.resume_ckpt_in
                $statusObj.current_pseudo_geom = [string]$Heartbeat.pseudo_geom_in
                if ([string]::IsNullOrWhiteSpace([string]$statusObj.active_lane)) {
                    $statusObj.active_lane = [string]$Heartbeat.lane_id
                }
                $statusObj.active_precompute_mv_support_on = [string]$Heartbeat.precompute_mv_support_on
                $statusObj.active_point_target_blend_by_mv_support = [string]$Heartbeat.point_target_blend_by_mv_support
                if ([string]::IsNullOrWhiteSpace([string]$statusObj.decision_reason)) {
                    $statusObj.decision_reason = "watch_sync_stage_from_heartbeat"
                }
                Write-JsonNoBom -Path $statusPath -Obj $statusObj
                $syncMd = @()
                $syncMd += "# 过夜 Ghost AutoLoop（watch-stage-sync）"
                $syncMd += ""
                $syncMd += "- updated: $($statusObj.updated_at)"
                $syncMd += "- deadline: $($statusObj.deadline)"
                $syncMd += "- cycle: $($statusObj.current_cycle)"
                $syncMd += "- stage: $($statusObj.current_stage)"
                $syncMd += "- active_lane: $($statusObj.active_lane)"
                $syncMd += "- decision_reason: $($statusObj.decision_reason)"
                $syncMd += "- active_precompute_mv_support_on: $($statusObj.active_precompute_mv_support_on)"
                $syncMd += "- active_point_target_blend_by_mv_support: $($statusObj.active_point_target_blend_by_mv_support)"
                $syncMd += "- note: watcher 从 heartbeat 同步 stage，等待当前 stage sweep 产物。"
                Set-Content -Path $statusMdPath -Value ($syncMd -join "`n") -Encoding UTF8
                return $true
            }
        }
        return $false
    }

    $stage = "cycle001_stage1_strong"
    if (($Heartbeat -ne $null) -and (-not [string]::IsNullOrWhiteSpace([string]$Heartbeat.stage))) {
        $stage = [string]$Heartbeat.stage
    }
    $cycle = Parse-CycleFromStage -Stage $stage
    if ($cycle -le 0) { $cycle = 1 }

    $resumeCkpt = [string]$Launcher.hot_update_forced_resume_ckpt
    if ([string]::IsNullOrWhiteSpace($resumeCkpt)) {
        $resumeCkpt = [string]$Launcher.hot_update_resume_ckpt
    }
    if ([string]::IsNullOrWhiteSpace($resumeCkpt) -and ($Heartbeat -ne $null)) {
        $resumeCkpt = [string]$Heartbeat.resume_ckpt_in
    }
    $pseudoGeom = ""
    if ($Heartbeat -ne $null) {
        $pseudoGeom = [string]$Heartbeat.pseudo_geom_in
    }
    $bootstrapPrecomputeMvSupportOn = ""
    $bootstrapPointTargetBlendByMvSupport = ""
    if ($Heartbeat -ne $null) {
        $bootstrapPrecomputeMvSupportOn = [string]$Heartbeat.precompute_mv_support_on
        $bootstrapPointTargetBlendByMvSupport = [string]$Heartbeat.point_target_blend_by_mv_support
    }

    $bestPath = "logs/modal_phase5/ghost_global_best_latest.json"
    $bestObj = Read-JsonMaybe -Path $bestPath
    $globalBestGhost = [double]::NaN
    $globalBestPsnr = [double]::NaN
    $globalBestSsim = [double]::NaN
    $globalBestWl1 = [double]::NaN
    $globalBestStage = ""
    $globalBestGeom = ""
    $globalBestDepth = ""
    $globalBestMask = ""
    if ($bestObj -ne $null) {
        $globalBestGhost = To-DoubleOrNaN($bestObj.global_best_ghost)
        $globalBestPsnr = To-DoubleOrNaN($bestObj.global_best_psnr)
        $globalBestSsim = To-DoubleOrNaN($bestObj.global_best_ssim)
        $globalBestWl1 = To-DoubleOrNaN($bestObj.global_best_wl1)
        $globalBestStage = [string]$bestObj.global_best_stage
        $globalBestGeom = [string]$bestObj.global_best_geom
        $globalBestDepth = [string]$bestObj.global_best_lambda_point_mv_depth
        $globalBestMask = [string]$bestObj.global_best_lambda_point_mv_mask
    }

    $updatedAt = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    $bootstrapStatus = [ordered]@{
        updated_at = $updatedAt
        deadline = $Deadline.ToString("yyyy-MM-ddTHH:mm:ss")
        current_cycle = $cycle
        current_stage = $stage
        interim = $true
        active_lane = "lane_a"
        lane_a_best = $null
        lane_b_best = $null
        guard_tier = ""
        decision_reason = "watch_bootstrap_pending_stage2"
        rollback_reason = ""
        current_resume_ckpt = $resumeCkpt
        current_pseudo_geom = $pseudoGeom
        global_best_ghost = $globalBestGhost
        global_best_psnr = $globalBestPsnr
        global_best_ssim = $globalBestSsim
        global_best_wl1 = $globalBestWl1
        global_best_stage = $globalBestStage
        global_best_geom = $globalBestGeom
        global_best_lambda_point_mv_depth = $globalBestDepth
        global_best_lambda_point_mv_mask = $globalBestMask
        stage_partial_rows = 0
        stage_best_ghost = [double]::NaN
        stage_best_psnr = [double]::NaN
        stage_best_ssim = [double]::NaN
        stage_best_wl1 = [double]::NaN
        best_lambda_point_mv_depth = ""
        best_lambda_point_mv_mask = ""
        active_precompute_mv_support_on = $bootstrapPrecomputeMvSupportOn
        active_point_target_blend_by_mv_support = $bootstrapPointTargetBlendByMvSupport
        next_cycle_tune_action = "pending_stage_complete"
        # Keep JSON-safe ASCII text here to avoid codepage corruption in Windows PowerShell.
        note = "watcher_bootstrap_wait_first_sweep_row"
    }
    Write-JsonNoBom -Path $statusPath -Obj $bootstrapStatus

    $md = @()
    $md += "# 过夜 Ghost AutoLoop（bootstrap/watch-sync）"
    $md += ""
    $md += "- updated: $($bootstrapStatus.updated_at)"
    $md += "- deadline: $($bootstrapStatus.deadline)"
    $md += "- cycle: $($bootstrapStatus.current_cycle)"
    $md += "- stage: $($bootstrapStatus.current_stage)"
    $md += "- active_lane: $($bootstrapStatus.active_lane)"
    $md += "- decision_reason: $($bootstrapStatus.decision_reason)"
    $md += "- current_resume_ckpt: $($bootstrapStatus.current_resume_ckpt)"
    $md += "- current_pseudo_geom: $($bootstrapStatus.current_pseudo_geom)"
    $md += "- global_best_ghost: $($bootstrapStatus.global_best_ghost)"
    $md += "- global_best_psnr: $($bootstrapStatus.global_best_psnr)"
    $md += "- global_best_ssim: $($bootstrapStatus.global_best_ssim)"
    $md += "- global_best_wl1: $($bootstrapStatus.global_best_wl1)"
    $md += "- active_precompute_mv_support_on: $($bootstrapStatus.active_precompute_mv_support_on)"
    $md += "- active_point_target_blend_by_mv_support: $($bootstrapStatus.active_point_target_blend_by_mv_support)"
    $md += "- next_cycle_tune_action: $($bootstrapStatus.next_cycle_tune_action)"
    $md += "- note: $($bootstrapStatus.note)"
    Set-Content -Path $statusMdPath -Value ($md -join "`n") -Encoding UTF8
    return $true
}

function Append-MentorInterim([object]$Sync, [string]$MentorPath) {
    if ($null -eq $Sync -or -not $Sync.synced) { return }
    $visual = "当前点位与阶段最优接近，重影变化有限。"
    if (($Sync.latest_ghost - $Sync.best_ghost) -ge 0.1) {
        $visual = "最新点位相对阶段最优明显恶化，重影拖尾回升。"
    } elseif ([Math]::Abs($Sync.latest_ghost - $Sync.best_ghost) -le 0.02) {
        $visual = "最新点位接近阶段最优，重影抑制保持稳定。"
    } elseif ($Sync.latest_ghost -lt $Sync.best_ghost) {
        $visual = "最新点位刷新阶段最优，重影有减轻趋势。"
    }

    $lines = @()
    $lines += ""
    $lines += "## $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') 自动中间落盘（watch-sync）"
    $lines += "- stage: $($Sync.stage)"
    $lines += "- partial_rows: $($Sync.rows)"
    $lines += "- best: depth=$($Sync.best_depth), mask=$($Sync.best_mask), ghost=$(('{0:F6}' -f $Sync.best_ghost)), PSNR=$(('{0:F6}' -f $Sync.best_psnr)), SSIM=$(('{0:F6}' -f $Sync.best_ssim)), wL1=$(('{0:F6}' -f $Sync.best_wl1))"
    $lines += "- latest: depth=$($Sync.latest_depth), mask=$($Sync.latest_mask), ghost=$(('{0:F6}' -f $Sync.latest_ghost)), PSNR=$(('{0:F6}' -f $Sync.latest_psnr)), SSIM=$(('{0:F6}' -f $Sync.latest_ssim)), wL1=$(('{0:F6}' -f $Sync.latest_wl1))"
    $lines += "- visual_reading: $visual"
    $lines += "- refreshed_files:"
    $lines += "  - logs/modal_phase5/ghost_autoloop_latest.csv"
    $lines += "  - logs/modal_phase5/overnight_ghost_autoloop_latest.json"
    $lines += "  - logs/modal_phase5/overnight_ghost_autoloop_latest.md"

    $abs = Join-Path (Resolve-Path ".").Path $MentorPath
    $dir = Split-Path -Parent $abs
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $enc = New-Object System.Text.UTF8Encoding($false)
    $payload = Sanitize-TextForUtf8Log -Text (($lines -join "`n") + "`n")
    [System.IO.File]::AppendAllText($abs, $payload, $enc)
}

$launcherForT0 = Read-JsonMaybe -Path $LauncherMetaPath
$watchSeed = Read-JsonMaybe -Path $OutJsonPath
$ensureSeed = Read-JsonMaybe -Path "logs/modal_phase5/ensure_hot_update_watcher_latest.json"
$t0 = Get-Date
$t0Source = "watch_start_time"
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
        [pscustomobject]@{ obj = $watchSeed; src = "watch_json" },
        [pscustomobject]@{ obj = $ensureSeed; src = "ensure_json" }
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
$mainFinalDeadline = ""
if ($ensureSeed -ne $null) {
    $mainFinalDeadline = [string]$ensureSeed.main_final_deadline
}
if ([string]::IsNullOrWhiteSpace($mainFinalDeadline) -and $watchSeed -ne $null) {
    $mainFinalDeadline = [string]$watchSeed.main_final_deadline
}
$mainFinalDeadlineDt = Parse-DateMaybe -Value $mainFinalDeadline
if ($mainFinalDeadlineDt -ne [datetime]::MinValue) {
    $deadline = $mainFinalDeadlineDt
    if (($t0 -eq [datetime]::MinValue) -or ($t0 -gt $deadline)) {
        $t0 = $deadline.AddHours(-1.0 * [Math]::Max(1, [int]$WatchHours))
        $t0Source = "derived_from_main_final_deadline"
    }
}
$lastInterimSignature = ""
$savedInterim = Read-JsonMaybe -Path $InterimStatePath
if ($savedInterim -ne $null) {
    $lastInterimSignature = [string]$savedInterim.last_signature
}
while ((Get-Date) -lt $deadline) {
    $launcher = Read-JsonMaybe -Path $LauncherMetaPath
    if ($launcher -ne $null) {
        $launcherT0Loop = Parse-DateMaybe -Value ([string]$launcher.started_at)
        if (($launcherT0Loop -ne [datetime]::MinValue) -and ($launcherT0Loop -gt $t0.AddSeconds(2))) {
            # Keep fixed watch window from initial T0; do not push deadline on launcher restart.
            $t0Source = "launcher.started_at_fixed_window"
        }
    }
    $hot = Read-JsonMaybe -Path $HotUpdateMetaPath
    if ($null -eq $hot) {
        $ensureLatest = Read-JsonMaybe -Path "logs/modal_phase5/ensure_hot_update_watcher_latest.json"
        if ($ensureLatest -ne $null) {
            $hot = [pscustomobject]@{
                started_at = [string]$ensureLatest.updated_at
                pid = [int]$ensureLatest.watcher_pid
                stdout = [string]$ensureLatest.watcher_stdout
                stderr = [string]$ensureLatest.watcher_stderr
                target_stage = [string]$ensureLatest.target_stage
                restart_reason = [string]$ensureLatest.last_action
                note = "fallback_from_ensure_hot_update_watcher_latest"
            }
        }
    }
    $hb = Read-JsonMaybe -Path $HeartbeatPath
    $hw = Read-JsonMaybe -Path $HotWatchPath

    $mainPid = 0
    $mainOut = ""
    $mainErr = ""
    if ($launcher -ne $null) {
        $mainPid = [int]$launcher.pid
        $mainOut = [string]$launcher.stdout
        $mainErr = [string]$launcher.stderr
    }
    $hotPid = 0
    $hotOut = ""
    $hotErr = ""
    if ($hot -ne $null) {
        $hotPid = [int]$hot.pid
        $hotOut = [string]$hot.stdout
        $hotErr = [string]$hot.stderr
    }

    $mainProc = Proc-State -ProcId $mainPid
    $hotProc = Proc-State -ProcId $hotPid

    $childSummary = [pscustomobject]@{
        lv1_count = $(if ($hw -ne $null) { [string]$hw.child_count } else { "" })
        worker_count = $(if ($hw -ne $null) { [string]$hw.worker_count } else { "" })
        names = ""
        sample_cmd = ""
    }

    $tracked = @(
        "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv",
        "logs/modal_phase5/ghost_autoloop_latest.csv",
        "logs/modal_phase5/overnight_ghost_autoloop_latest.json",
        "logs/modal_phase5/overnight_ghost_autoloop_latest.md",
        "logs/modal_phase5/vggt_ft_sweep_latest.csv",
        "logs/modal_phase5/vggt_ft_gate_latest.json",
        "logs/modal_phase5/mentor_update_latest.md"
    )
    $sync = $null
    $bootstrapSynced = $false
    try {
        $bootstrapSynced = Ensure-AutoloopBootstrapStatus -Heartbeat $hb -Launcher $launcher -Deadline $deadline
        $sync = Upsert-InterimAutoloopFiles -Heartbeat $hb -Launcher $launcher -Deadline $deadline
        if ($sync -ne $null -and $sync.synced -and ([string]$sync.signature -ne $lastInterimSignature)) {
            Append-MentorInterim -Sync $sync -MentorPath "logs/modal_phase5/mentor_update_latest.md"
            $lastInterimSignature = [string]$sync.signature
            Write-JsonNoBom -Path $InterimStatePath -Obj ([ordered]@{
                updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
                last_signature = $lastInterimSignature
                stage = [string]$sync.stage
                cycle = [int]$sync.cycle
                rows = [int]$sync.rows
                best_ghost = [double]$sync.best_ghost
                latest_ghost = [double]$sync.latest_ghost
            })
        }
    } catch {
        $sync = [pscustomobject]@{
            synced = $false
            error = $_.Exception.Message
        }
    }

    $files = @()
    foreach ($p in $tracked) {
        $files += File-InfoMaybe -Path $p
    }

    $obj = [ordered]@{
        updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        t0 = $t0.ToString("yyyy-MM-ddTHH:mm:ss")
        t0_source = $t0Source
        deadline = $deadline.ToString("yyyy-MM-ddTHH:mm:ss")
        watch_hours = $WatchHours
        main_final_deadline = $mainFinalDeadline
        launcher_meta = Meta-Slim -M $launcher
        hotupdate_meta = Meta-Slim -M $hot
        heartbeat = $(if ($hb) { [pscustomobject]@{
            updated_at = Normalize-TimeText -Value ([string]$hb.updated_at)
            state = [string]$hb.state
            stage = [string]$hb.stage
        } } else { $null })
        hot_watch = $(if ($hw) { [pscustomobject]@{
            updated_at = Normalize-TimeText -Value ([string]$hw.updated_at)
            state = [string]$hw.state
            pid = [string]$hw.pid
            hit_stage_boundary = [string]$hw.hit_stage_boundary
            trigger_reason = [string]$hw.trigger_reason
            idle_minutes = [string]$hw.idle_minutes
            worker_count = [string]$hw.worker_count
            child_count = [string]$hw.child_count
            active_stall_threshold_minutes = [string]$hw.active_stall_threshold_minutes
        } } else { $null })
        main_proc = $mainProc
        hot_proc = $hotProc
        child_summary = $childSummary
        interim_sync = $sync
        bootstrap_sync = $bootstrapSynced
        files = $files
        tails = [ordered]@{
            main_out = Tail-Maybe -Path $mainOut -Lines 40
            main_err = Tail-Maybe -Path $mainErr -Lines 40
            hot_out = Tail-Maybe -Path $hotOut -Lines 30
            hot_err = Tail-Maybe -Path $hotErr -Lines 30
        }
    }
    Write-JsonNoBom -Path $OutJsonPath -Obj $obj

    $md = @()
    $md += "# Ghost 产物观测"
    $md += ""
    $md += "- updated: $($obj.updated_at)"
    $md += "- t0: $($obj.t0) ($($obj.t0_source))"
    $md += "- deadline: $($obj.deadline)"
    $md += "- watch_hours: $($obj.watch_hours)"
    $md += "- main_final_deadline: $($obj.main_final_deadline)"
    $md += "- main_proc: state=$($mainProc.state), pid=$($mainProc.pid), name=$($mainProc.name), start=$($mainProc.start_time)"
    $md += "- hot_proc: state=$($hotProc.state), pid=$($hotProc.pid), name=$($hotProc.name), start=$($hotProc.start_time)"
    if ($hw -ne $null) {
        $md += "- hot_watch: state=$([string]$hw.state), hit=$([string]$hw.hit_stage_boundary), trigger=$([string]$hw.trigger_reason), idle=$([string]$hw.idle_minutes)"
    }
    $md += "- child_summary: lv1=$($childSummary.lv1_count), worker=$($childSummary.worker_count)"
    $md += ""
    $md += "## 文件落盘"
    foreach ($f in @($files)) {
        $md += "- $($f.path): exists=$($f.exists), last_write=$($f.last_write), length=$($f.length)"
    }
    $md += ""
    $md += "## 主链输出尾部"
    $md += '```text'
    $md += ($obj.tails.main_out -join [Environment]::NewLine)
    $md += '```'
    $md += ""
    $md += "## 热更新输出尾部"
    $md += '```text'
    $md += ($obj.tails.hot_out -join [Environment]::NewLine)
    $md += '```'
    Set-Content -Path $OutMdPath -Value ($md -join [Environment]::NewLine) -Encoding UTF8

    if ($RunOnce) { break }
    Start-Sleep -Seconds ([Math]::Max(10, [int]$PollSec))
}

exit 0

