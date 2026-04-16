param(
    [string]$LegacyMetaPath = "logs/modal_phase5/overnight_cde_legacy_latest.json",
    [string]$UpstreamMetaPath = "logs/modal_phase5/overnight_cde_upstream_latest.json",
    [string]$WatcherMetaPath = "logs/modal_phase5/overnight_cde_watcher_latest.json",
    [string]$OutMdPath = "logs/modal_phase5/overnight_cde_status_latest.md",
    [string]$OutJsonPath = "logs/modal_phase5/overnight_cde_status_latest.json",
    [int]$PollSec = 120,
    [int]$TimeoutHours = 12
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot "..")
Set-Location $repoRoot

function Read-JsonMaybe([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try {
        return (Get-Content $Path -Raw | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Get-ProcState([object]$Meta) {
    if ($null -eq $Meta) {
        return [pscustomobject]@{
            state = "not_started"
            pid = ""
            started = ""
            stdout = ""
            stderr = ""
        }
    }
    $procId = [int]$Meta.pid
    $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
    return [pscustomobject]@{
        state = $(if ($null -ne $p) { "running" } else { "stopped" })
        pid = $procId
        started = [string]$Meta.started
        stdout = [string]$Meta.stdout
        stderr = [string]$Meta.stderr
    }
}

function Tail-Maybe([string]$Path, [int]$Lines = 30) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path $Path)) { return @() }
    return @(Get-Content $Path -Tail $Lines)
}

function To-DoubleOrNaN($x) {
    try {
        if ($null -eq $x -or [string]::IsNullOrWhiteSpace([string]$x)) { return [double]::NaN }
        return [double]$x
    } catch {
        return [double]::NaN
    }
}

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 20
    $abs = Join-Path (Resolve-Path ".").Path $Path
    [System.IO.File]::WriteAllText($abs, $json, $enc)
}

$deadline = (Get-Date).AddHours([Math]::Max(1, $TimeoutHours))

while ((Get-Date) -lt $deadline) {
    $legacyMeta = Read-JsonMaybe $LegacyMetaPath
    $upstreamMeta = Read-JsonMaybe $UpstreamMetaPath
    $watcherMeta = Read-JsonMaybe $WatcherMetaPath

    $legacy = Get-ProcState $legacyMeta
    $upstream = Get-ProcState $upstreamMeta
    $watcher = Get-ProcState $watcherMeta

    $apps = @()

    $sweepLatestPath = "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv"
    $loopLatestPath = "logs/modal_phase5/ghost_cde_loop_latest.csv"
    $sweepRows = @()
    $loopRows = @()
    if (Test-Path $sweepLatestPath) { $sweepRows = @(Import-Csv $sweepLatestPath) }
    if (Test-Path $loopLatestPath) { $loopRows = @(Import-Csv $loopLatestPath) }

    $bestSweep = $null
    if ($sweepRows.Count -gt 0) {
        $bestSweep = @(
            $sweepRows |
                Sort-Object {
                    To-DoubleOrNaN($_.ghost_score_mean)
                }, {
                    -1.0 * (To-DoubleOrNaN($_.mean_PSNR))
                } |
                Select-Object -First 1
        )[0]
    }

    $obj = [ordered]@{
        timestamp = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        deadline = ($deadline.ToString("yyyy-MM-dd HH:mm:ss"))
        legacy = $legacy
        upstream = $upstream
        watcher = $watcher
        modal_apps = $apps
        sweep_latest = [ordered]@{
            path = $sweepLatestPath
            exists = (Test-Path $sweepLatestPath)
            last_write = $(if (Test-Path $sweepLatestPath) { (Get-Item $sweepLatestPath).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss") } else { "" })
            rows = $sweepRows.Count
            best = $bestSweep
        }
        loop_latest = [ordered]@{
            path = $loopLatestPath
            exists = (Test-Path $loopLatestPath)
            last_write = $(if (Test-Path $loopLatestPath) { (Get-Item $loopLatestPath).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss") } else { "" })
            rows = $loopRows.Count
            last = $(if ($loopRows.Count -gt 0) { $loopRows[-1] } else { $null })
        }
        tails = [ordered]@{
            legacy_stdout = (Tail-Maybe $legacy.stdout 20)
            legacy_stderr = (Tail-Maybe $legacy.stderr 20)
            upstream_stdout = (Tail-Maybe $upstream.stdout 20)
            upstream_stderr = (Tail-Maybe $upstream.stderr 20)
        }
    }

    Write-JsonNoBom -Path $OutJsonPath -Obj $obj

    $lines = @()
    $lines += "# Overnight CDE Status"
    $lines += ""
    $lines += "- timestamp: $($obj.timestamp)"
    $lines += "- legacy: state=$($legacy.state), pid=$($legacy.pid), started=$($legacy.started)"
    $lines += "- upstream: state=$($upstream.state), pid=$($upstream.pid), started=$($upstream.started)"
    $lines += "- watcher: state=$($watcher.state), pid=$($watcher.pid), started=$($watcher.started)"
    $lines += "- modal_apps: $(@($apps).Count)"
    $lines += "- sweep_latest: exists=$($obj.sweep_latest.exists), rows=$($obj.sweep_latest.rows), last_write=$($obj.sweep_latest.last_write)"
    if ($null -ne $bestSweep) {
        $lines += "- sweep_best: mv=$($bestSweep.lambda_point_mv_depth), ghost=$($bestSweep.ghost_score_mean), psnr=$($bestSweep.mean_PSNR), ssim=$($bestSweep.mean_SSIM), wL1=$($bestSweep.mean_weighted_L1)"
    }
    $lines += "- loop_latest: exists=$($obj.loop_latest.exists), rows=$($obj.loop_latest.rows), last_write=$($obj.loop_latest.last_write)"
    if ($null -ne $obj.loop_latest.last) {
        $r = $obj.loop_latest.last
        $lines += "- loop_last: round=$($r.round), reason=$($r.reason), ghost=$($r.ghost_score_mean), psnr=$($r.mean_PSNR), ssim=$($r.mean_SSIM), wL1=$($r.mean_weighted_L1)"
    }
    $lines += ""
    $lines += "## Legacy Tail"
    $lines += '```text'
    $lines += ($obj.tails.legacy_stdout -join "`n")
    $lines += '```'
    $lines += ""
    $lines += "## Upstream Tail"
    $lines += '```text'
    $lines += ($obj.tails.upstream_stdout -join "`n")
    $lines += '```'
    Set-Content -Path $OutMdPath -Value ($lines -join "`n") -Encoding UTF8

    $legacyDone = ($legacy.state -eq "stopped" -or $legacy.state -eq "not_started")
    $upstreamDone = ($upstream.state -eq "stopped" -or $upstream.state -eq "not_started")
    $watcherDone = ($watcher.state -eq "stopped" -or $watcher.state -eq "not_started")
    $upstreamWasStarted = ($null -ne $upstreamMeta)

    if ($legacyDone -and $watcherDone -and ($upstreamDone -and $upstreamWasStarted)) {
        break
    }

    Start-Sleep -Seconds ([Math]::Max(15, $PollSec))
}

exit 0
