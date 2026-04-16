param(
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$PseudoGeomSubdir = "vggt_geom_ft_20260208_044454",
    [string]$PretrainedCkpt = "model.pt",
    [int]$SleepSec = 180,
    [int]$TimeoutHours = 10,
    [bool]$EnableFreezeFix = $true,
    [bool]$EnableLongRun = $true,
    [string]$LrList = "4e-6,2e-6,1e-6",
    [int]$EpochsShort = 3,
    [int]$EpochsLong = 6,
    [int]$MaxFramesShort = 400,
    [int]$MaxFramesLong = 0,
    [int]$EvalNumSamples = 60,
    [int]$EarlyStopPatience = 1,
    [double]$MinImprove = 0.0001,
    [bool]$EnablePhase2 = $true
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$runLog = "logs/modal_phase5/overnight_autopilot_$ts.log"
$runJson = "logs/modal_phase5/overnight_autopilot_$ts.json"
$runJsonLatest = "logs/modal_phase5/overnight_autopilot_latest.json"
$runMdLatest = "logs/modal_phase5/overnight_autopilot_latest.md"

function Write-Log([string]$Msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Msg
    Write-Host $line
    Add-Content -Path $runLog -Value $line -Encoding UTF8
}

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 20
    $root = (Resolve-Path ".").Path
    $abs = Join-Path $root $Path
    [System.IO.File]::WriteAllText($abs, $json, $enc)
}

function Get-RunningSweepProcs() {
    return @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.CommandLine -match 'run_vggt_ft_lr_sweep\.ps1|run_vggt_ft_freeze_sweep\.ps1|modal run modal_run_train\.py'
            } |
            Select-Object ProcessId, ParentProcessId, Name, CreationDate, CommandLine
    )
}

function Read-CsvIfExists([string]$Path) {
    if (Test-Path $Path) {
        return @(Import-Csv $Path)
    }
    return @()
}

function Read-JsonIfExists([string]$Path) {
    if (Test-Path $Path) {
        try {
            return (Get-Content $Path -Raw | ConvertFrom-Json)
        } catch {
            return $null
        }
    }
    return $null
}

function Get-LatestFile([string]$Pattern) {
    $f = Get-ChildItem logs/modal_phase5 -Filter $Pattern -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    return $f
}

function Need-FreezeFix() {
    $rows = Read-CsvIfExists -Path "logs/modal_phase5/vggt_ft_sweep_latest.csv"
    if ($rows.Count -eq 0) {
        return $true
    }
    $freezeRows = @($rows | Where-Object { $_.stage -eq "freeze" })
    if ($freezeRows.Count -lt 3) {
        return $true
    }
    foreach ($r in $freezeRows) {
        $fm = [string]$r.freeze_mode
        if ($fm -match '\s') { return $true }
    }
    $uniq = @($freezeRows | Select-Object -ExpandProperty freeze_mode | Sort-Object -Unique)
    return ($uniq.Count -lt 3)
}

function Run-FreezeFix() {
    Write-Log "run freeze fix sweep with explicit mode array"
    & "$CodeDir\scripts\run_vggt_ft_freeze_sweep.ps1" `
        -CodeDir $CodeDir `
        -SeqNames $SeqNames `
        -PseudoGeomSubdir $PseudoGeomSubdir `
        -PretrainedCkpt $PretrainedCkpt `
        -Lr 2e-6 `
        -FreezeModes @("depth_point", "depth_only", "point_only") `
        -Epochs $EpochsShort `
        -MaxFrames $MaxFramesShort `
        -EvalNumSamples $EvalNumSamples `
        -EarlyStopPatience $EarlyStopPatience `
        -MinImprove $MinImprove `
        -RunSupSweep `
        -SupDepthScaleAlignList @("off", "median") `
        -SupLambdaConfList @("0.05", "0.02")
    $rc = $LASTEXITCODE
    Write-Log "freeze fix sweep exit_code=$rc"
    return $rc
}

function Get-BestLrFromCompare() {
    $rows = Read-CsvIfExists -Path "logs/modal_phase5/baseline_compare_latest.csv"
    if ($rows.Count -eq 0) { return "" }
    $ok = @(
        $rows |
            Where-Object {
                $_.status -eq "ok" -and $_.pass -eq "True" -and ([string]$_.label).StartsWith("lr_")
            }
    )
    if ($ok.Count -eq 0) { return "" }
    $best = $ok | Sort-Object { [double]$_.mean_PSNR } -Descending | Select-Object -First 1
    if ($null -eq $best) { return "" }
    return ([string]$best.label).Substring(3)
}

function Has-LongRow() {
    $rows = Read-CsvIfExists -Path "logs/modal_phase5/vggt_ft_sweep_latest.csv"
    if ($rows.Count -eq 0) { return $false }
    return (@($rows | Where-Object { $_.stage -eq "long" }).Count -gt 0)
}

function Run-LrLong([string]$BestLr) {
    if ([string]::IsNullOrWhiteSpace($BestLr)) {
        Write-Log "skip long run: no best lr found"
        return 0
    }
    Write-Log "run lr long follow-up best_lr=$BestLr"
    & "$CodeDir\scripts\run_vggt_ft_lr_sweep.ps1" `
        -CodeDir $CodeDir `
        -SeqNames $SeqNames `
        -PseudoGeomSubdir $PseudoGeomSubdir `
        -PretrainedCkpt $PretrainedCkpt `
        -LrList $BestLr `
        -FreezeMode depth_point `
        -EpochsShort $EpochsShort `
        -EpochsLong $EpochsLong `
        -MaxFramesShort $MaxFramesShort `
        -MaxFramesLong $MaxFramesLong `
        -EvalNumSamples $EvalNumSamples `
        -EarlyStopPatience $EarlyStopPatience `
        -MinImprove $MinImprove `
        -RunLongOnImprove
    $rc = $LASTEXITCODE
    Write-Log "lr long follow-up exit_code=$rc"
    return $rc
}

function Run-Phase2Prep() {
    Write-Log "run phase2 pixelsplat prep"
    & "$CodeDir\scripts\run_phase2_pixelsplat_prep.ps1" `
        -CodeDir $CodeDir `
        -SeqNames $SeqNames `
        -GeomSubdir auto `
        -MaxFrames 300 `
        -MaxViews 6
    $rc = $LASTEXITCODE
    Write-Log "phase2 prep exit_code=$rc"
    return $rc
}

function Write-Summary([string]$State, [string]$Note) {
    $sweepRows = Read-CsvIfExists -Path "logs/modal_phase5/vggt_ft_sweep_latest.csv"
    $cmpRows = Read-CsvIfExists -Path "logs/modal_phase5/baseline_compare_latest.csv"
    $gate = Read-JsonIfExists -Path "logs/modal_phase5/vggt_ft_gate_latest.json"
    $apps = @()
    try {
        $apps = @((modal app list --json | ConvertFrom-Json) | Where-Object { $_.State -eq "ephemeral" })
    } catch {
        $apps = @()
    }

    $obj = [ordered]@{
        timestamp = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        state = $State
        note = $Note
        active_modal_apps = $apps
        sweep_rows = $sweepRows
        compare_rows = $cmpRows
        gate = $gate
    }
    Write-JsonNoBom -Path $runJson -Obj $obj
    Write-JsonNoBom -Path $runJsonLatest -Obj $obj

    $lines = @()
    $lines += "# Overnight Autopilot Status"
    $lines += ""
    $lines += "- timestamp: $($obj.timestamp)"
    $lines += "- state: $State"
    $lines += "- note: $Note"
    $lines += "- active_modal_apps: $($apps.Count)"
    $lines += "- latest_sweep_rows: $(@($sweepRows).Count)"
    $lines += "- latest_compare_rows: $(@($cmpRows).Count)"
    if ($gate -ne $null) {
        $lines += "- latest_gate_pass: $($gate.pass)"
        $gateRows = @($gate.rows)
        if ($gateRows.Count -gt 0) {
            $lines += "- latest_gate_top_label: $($gateRows[0].label)"
            $lines += "- latest_gate_top_psnr: $($gateRows[0].mean_PSNR)"
        }
    }
    Set-Content -Path $runMdLatest -Value ($lines -join "`n") -Encoding UTF8
}

$deadline = (Get-Date).AddHours($TimeoutHours)
$freezeFixDone = $false
$longRunDone = $false
$phase2Done = $false

Write-Log "autopilot start timeout_hours=$TimeoutHours sleep_sec=$SleepSec"
Write-Summary -State "running" -Note "start"

while ((Get-Date) -lt $deadline) {
    $running = Get-RunningSweepProcs
    if ($running.Count -gt 0) {
        Write-Log ("running sweep processes={0}" -f $running.Count)
        Write-Summary -State "running" -Note ("running_procs={0}" -f $running.Count)
        Start-Sleep -Seconds $SleepSec
        continue
    }

    if ($EnableFreezeFix -and -not $freezeFixDone) {
        $need = Need-FreezeFix
        Write-Log "freeze_fix_check need=$need"
        if ($need) {
            $rc = Run-FreezeFix
            if ($rc -ne 0) {
                Write-Summary -State "failed" -Note "freeze_fix_failed"
                exit 2
            }
        }
        $freezeFixDone = $true
        Write-Summary -State "running" -Note "freeze_fix_done"
        Start-Sleep -Seconds 5
        continue
    }

    if ($EnableLongRun -and -not $longRunDone) {
        if (-not (Has-LongRow)) {
            $bestLr = Get-BestLrFromCompare
            $rc2 = Run-LrLong -BestLr $bestLr
            if ($rc2 -ne 0) {
                Write-Summary -State "failed" -Note "lr_long_failed"
                exit 3
            }
        } else {
            Write-Log "long row already exists, skip long run"
        }
        $longRunDone = $true
        Write-Summary -State "running" -Note "long_run_done"
        Start-Sleep -Seconds 5
        continue
    }

    if ($EnablePhase2 -and -not $phase2Done) {
        $rc3 = Run-Phase2Prep
        if ($rc3 -ne 0) {
            Write-Summary -State "failed" -Note "phase2_prep_failed"
            exit 5
        }
        $phase2Done = $true
        Write-Summary -State "running" -Note "phase2_prep_done"
        Start-Sleep -Seconds 5
        continue
    }

    Write-Summary -State "done" -Note "all tasks complete"
    Write-Log "autopilot done"
    exit 0
}

Write-Summary -State "timeout" -Note "deadline reached"
Write-Log "autopilot timeout"
exit 4
