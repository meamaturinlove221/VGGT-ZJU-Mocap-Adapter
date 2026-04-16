param(
    [string]$LauncherMetaPath = "logs/modal_phase5/overnight_ghost_autoloop_launcher_latest.json",
    [string]$StatusJsonPath = "logs/modal_phase5/overnight_ghost_autoloop_watch_latest.json",
    [string]$StatusMdPath = "logs/modal_phase5/overnight_ghost_autoloop_watch_latest.md",
    [int]$PollSec = 60,
    [int]$WatchHours = 12
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

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 20
    [System.IO.File]::WriteAllText((Resolve-Path ".").Path + "\" + $Path, $json, $enc)
}

function Tail-Maybe([string]$Path, [int]$Lines = 20) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return @() }
    if (-not (Test-Path $Path)) { return @() }
    return @(Get-Content $Path -Tail $Lines)
}

$deadline = (Get-Date).AddHours([Math]::Max(1, [int]$WatchHours))
$restartCount = 0

while ((Get-Date) -lt $deadline) {
    $meta = Read-JsonMaybe -Path $LauncherMetaPath
    $state = "meta_missing"
    $procPid = 0
    $stdout = ""
    $stderr = ""
    $args = ""
    $startedAt = ""

    if ($meta -ne $null) {
        $procPid = [int]$meta.pid
        $stdout = [string]$meta.stdout
        $stderr = [string]$meta.stderr
        $args = [string]$meta.args
        $startedAt = [string]$meta.started_at
        $p = $null
        if ($procPid -gt 0) { $p = Get-Process -Id $procPid -ErrorAction SilentlyContinue }

        if ($p -ne $null) {
            $state = "running"
        } else {
            $state = "restarting"
            if (-not [string]::IsNullOrWhiteSpace($args)) {
                $restartCount += 1
                $ts = Get-Date -Format "yyyyMMdd_HHmmss"
                $outLog = "logs/modal_phase5/overnight_ghost_autoloop_restart${restartCount}_$ts.out.log"
                $errLog = "logs/modal_phase5/overnight_ghost_autoloop_restart${restartCount}_$ts.err.log"
                $p2 = Start-Process `
                    -FilePath "powershell.exe" `
                    -ArgumentList $args `
                    -RedirectStandardOutput $outLog `
                    -RedirectStandardError $errLog `
                    -PassThru
                $meta = [pscustomobject]@{
                    started_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
                    pid = $p2.Id
                    stdout = $outLog
                    stderr = $errLog
                    args = $args
                    restart_count = $restartCount
                    restarted_from_pid = $procPid
                }
                $meta | ConvertTo-Json -Depth 10 | Set-Content -Path $LauncherMetaPath -Encoding UTF8
                $procPid = [int]$meta.pid
                $stdout = [string]$meta.stdout
                $stderr = [string]$meta.stderr
                $startedAt = [string]$meta.started_at
                $state = "running_after_restart"
            } else {
                $state = "stopped_no_args"
            }
        }
    }

    $status = [ordered]@{
        updated_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        deadline = $deadline.ToString("yyyy-MM-dd HH:mm:ss")
        state = $state
        pid = $procPid
        started_at = $startedAt
        launcher_meta = $LauncherMetaPath
        stdout = $stdout
        stderr = $stderr
        restart_count = $restartCount
        stdout_tail = (Tail-Maybe -Path $stdout -Lines 20)
        stderr_tail = (Tail-Maybe -Path $stderr -Lines 20)
    }
    Write-JsonNoBom -Path $StatusJsonPath -Obj $status

    $md = @()
    $md += "# AutoLoop Watch"
    $md += ""
    $md += "- updated: $($status.updated_at)"
    $md += "- deadline: $($status.deadline)"
    $md += "- state: $($status.state)"
    $md += "- pid: $($status.pid)"
    $md += "- restart_count: $($status.restart_count)"
    $md += "- stdout: $($status.stdout)"
    $md += "- stderr: $($status.stderr)"
    $md += ""
    $md += '## stdout tail'
    $md += '```text'
    $md += ($status.stdout_tail -join [Environment]::NewLine)
    $md += '```'
    $md += ''
    $md += '## stderr tail'
    $md += '```text'
    $md += ($status.stderr_tail -join [Environment]::NewLine)
    $md += '```'
    Set-Content -Path $StatusMdPath -Value ($md -join [Environment]::NewLine) -Encoding UTF8

    Start-Sleep -Seconds ([Math]::Max(15, [int]$PollSec))
}

exit 0
