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

function Get-TrackedLocalProcesses() {
    $patterns = @(
        "run_overnight_ghost_autoloop",
        "ensure_hot_update_watcher",
        "hot_update_autoloop_after_stage",
        "watch_ghost_outputs",
        "modal run",
        "modal_run_train.py"
    )
    return @(
        Get-CimInstance Win32_Process | Where-Object {
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

$localProcs = Get-TrackedLocalProcesses
$liveApps = Get-LiveModalApps
$modalProgress = Read-JsonMaybe -Path (Join-Path $StatusDir "modal_run_progress_latest.json")
$autoloopStatus = Read-JsonMaybe -Path (Join-Path $StatusDir "overnight_ghost_autoloop_latest.json")
$watchStatus = Read-JsonMaybe -Path (Join-Path $StatusDir "watch_ghost_outputs_latest.json")

$report = [ordered]@{
    updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    local_process_count = $localProcs.Count
    live_modal_app_count = $liveApps.Count
    modal_progress_state = $(if ($modalProgress) { [string]$modalProgress.state } else { "" })
    modal_progress_note = $(if ($modalProgress) { [string]$modalProgress.note } else { "" })
    autoloop_paused = $(if ($autoloopStatus) { [string]$autoloopStatus.paused } else { "" })
    autoloop_pause_reason = $(if ($autoloopStatus) { [string]$autoloopStatus.pause_reason } else { "" })
    watch_paused = $(if ($watchStatus) { [string]$watchStatus.paused } else { "" })
    watch_pause_reason = $(if ($watchStatus) { [string]$watchStatus.pause_reason } else { "" })
}

$report.GetEnumerator() | ForEach-Object { Write-Host ("{0}: {1}" -f $_.Key, $_.Value) }

$ok = $true
if ($localProcs.Count -gt 0) { $ok = $false }
if ($liveApps.Count -gt 0) { $ok = $false }
if ($modalProgress -and ([string]$modalProgress.state -notin @("stale", "done", "error", "stopped"))) { $ok = $false }
if ($autoloopStatus -and (-not [bool]$autoloopStatus.paused)) { $ok = $false }
if ($watchStatus -and (-not [bool]$watchStatus.paused)) { $ok = $false }

if (-not $ok) { exit 2 }
