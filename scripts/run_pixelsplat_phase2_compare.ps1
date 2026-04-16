param(
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$PretrainedCkpt = "model.pt",
    [string]$DatasetRoot = "/mnt/out/pixelsplat_datasets/zju_phase2_300_6v",
    [string]$EvalIndexUniform = "/mnt/out/pixelsplat_assets/evaluation_index_zju_uniform_6v.json",
    [string]$EvalIndexRandom = "/mnt/out/pixelsplat_assets/evaluation_index_zju_random_6v.json",
    [string]$OutputRoot = "/mnt/out/pixelsplat_runs",
    [int]$MaxSteps = 300,
    [int]$ValCheckInterval = 100,
    [int]$CheckpointEvery = 100,
    [int]$BatchSize = 1,
    [int]$NumWorkersTrain = 4,
    [int]$NumWorkersVal = 1,
    [int]$NumWorkersTest = 2,
    [int]$NumContextViews = 2,
    [int]$SummaryWaitTimeoutSec = 7200
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 20
    $root = (Resolve-Path ".").Path
    $abs = Join-Path $root $Path
    [System.IO.File]::WriteAllText($abs, $json, $enc)
}

function To-VolumePath([string]$OutPath) {
    $s = [string]$OutPath
    if ($s.StartsWith("/mnt/out/")) {
        return "/" + $s.Substring("/mnt/out/".Length)
    }
    if ($s.StartsWith("mnt/out/")) {
        return "/" + $s.Substring("mnt/out/".Length)
    }
    if ($s.StartsWith("/")) {
        return $s
    }
    return "/" + $s
}

function Get-ModalRunUrl([string[]]$Lines) {
    $url = ""
    foreach ($line in $Lines) {
        if ($line -match "https://modal\.com/apps/\S+") {
            $url = $Matches[0]
        }
    }
    return $url
}

function Invoke-ModalRun(
    [string]$ScriptPath = "modal_run_train.py",
    [int]$MaxRetries = 3,
    [int]$RetrySleepSec = 10
) {
    $attempt = 0
    $last = $null
    while ($attempt -lt [Math]::Max(1, $MaxRetries)) {
        $attempt += 1
        $stdoutFile = [System.IO.Path]::GetTempFileName()
        $stderrFile = [System.IO.Path]::GetTempFileName()
        try {
            $proc = Start-Process `
                -FilePath "cmd.exe" `
                -ArgumentList @("/c", "modal run -q -d $ScriptPath") `
                -NoNewWindow `
                -Wait `
                -PassThru `
                -RedirectStandardOutput $stdoutFile `
                -RedirectStandardError $stderrFile
            $output = @()
            if (Test-Path $stdoutFile) { $output += @(Get-Content $stdoutFile) }
            if (Test-Path $stderrFile) { $output += @(Get-Content $stderrFile) }
            $rc = [int]$proc.ExitCode
            $last = [pscustomobject]@{
                Output = $output
                ExitCode = $rc
                Attempt = $attempt
            }
            if ($rc -eq 0) {
                return $last
            }
            $blob = ($output -join "`n")
            $isTransient = (
                ($blob -match "Connection lost") -or
                ($blob -match "WinError 10053") -or
                ($blob -match "WinError 10054") -or
                ($blob -match "SSL shutdown timed out") -or
                ($blob -match "timed out waiting for final app logs")
            )
            if ($isTransient -and $attempt -lt $MaxRetries) {
                Write-Host "[modal-run] transient failure attempt=$attempt/$MaxRetries, retry in ${RetrySleepSec}s"
                Start-Sleep -Seconds $RetrySleepSec
                continue
            }
            return $last
        } finally {
            Remove-Item $stdoutFile -ErrorAction SilentlyContinue
            Remove-Item $stderrFile -ErrorAction SilentlyContinue
        }
    }
    return $last
}

function Wait-FetchVolumeFile(
    [string]$RemotePath,
    [string]$LocalPath,
    [int]$TimeoutSec = 7200,
    [int]$PollSec = 20
) {
    $deadline = (Get-Date).AddSeconds([Math]::Max(30, [int]$TimeoutSec))
    while ((Get-Date) -lt $deadline) {
        try {
            & modal volume get vggt-out $RemotePath $LocalPath 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0 -and (Test-Path $LocalPath)) {
                return $true
            }
        } catch {
        }
        Start-Sleep -Seconds ([Math]::Max(3, [int]$PollSec))
    }
    return $false
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$runTag = "zju_phase2_cmp_$ts"
$resultNow = "logs/modal_phase5/pixelsplat_phase2_compare_$ts.json"
$resultLatest = "logs/modal_phase5/pixelsplat_phase2_compare_latest.json"
$resultMdLatest = "logs/modal_phase5/pixelsplat_phase2_compare_latest.md"
$runLog = "logs/modal_phase5/pixelsplat_phase2_compare_$ts.log"

$argsExtra = @(
    "--repo_root /mnt/code/external/pixelsplat",
    "--dataset_root $DatasetRoot",
    "--eval_index_uniform $EvalIndexUniform",
    "--eval_index_random $EvalIndexRandom",
    "--output_root $OutputRoot",
    "--run_tag $runTag",
    "--max_steps $MaxSteps",
    "--val_check_interval $ValCheckInterval",
    "--checkpoint_every $CheckpointEvery",
    "--batch_size $BatchSize",
    "--num_workers_train $NumWorkersTrain",
    "--num_workers_val $NumWorkersVal",
    "--num_workers_test $NumWorkersTest",
    "--num_context_views $NumContextViews",
    "--save_json /mnt/out/pixelsplat_runs/$runTag/compare_summary.json"
) -join " "

$env:VGGT_CODE_DIR = $CodeDir
$env:VGGT_MODE = "precompute"
$env:VGGT_SEQ_NAMES = $SeqNames
$env:VGGT_GEOM_SUBDIR = "vggt_geom"
$env:VGGT_PRECOMPUTE_SCRIPT = "tools/run_pixelsplat_experiment.py"
$env:VGGT_PRECOMPUTE_CKPT = $PretrainedCkpt
$env:VGGT_PRECOMPUTE_ARGS_EXTRA = $argsExtra
Remove-Item Env:VGGT_PROFILE -ErrorAction SilentlyContinue

$run = Invoke-ModalRun -ScriptPath "modal_run_train.py"
$output = @($run.Output)
$rc = [int]$run.ExitCode
$output | Tee-Object -FilePath $runLog | Out-Null
$runUrl = Get-ModalRunUrl -Lines $output

$remoteSummary = To-VolumePath -OutPath "$OutputRoot/$runTag/compare_summary.json"
$localSummary = "logs/modal_phase5/pixelsplat_phase2_compare_summary_$ts.json"
$fetchOk = $false
$waitSec = [Math]::Max(30, [int]$SummaryWaitTimeoutSec)
if (($rc -ne 0) -and [string]::IsNullOrWhiteSpace($runUrl)) {
    $waitSec = [Math]::Min($waitSec, 300)
}
$fetchOk = Wait-FetchVolumeFile -RemotePath $remoteSummary -LocalPath $localSummary -TimeoutSec $waitSec -PollSec 20

$summaryObj = $null
if ($fetchOk) {
    $summaryObj = Get-Content $localSummary -Raw | ConvertFrom-Json
}

$result = [ordered]@{
    timestamp = $ts
    run_tag = $runTag
    modal_exit_code = $rc
    modal_run_url = $runUrl
    modal_log = $runLog
    remote_summary = $remoteSummary
    local_summary = $localSummary
    fetch_ok = $fetchOk
    summary = $summaryObj
}
Write-JsonNoBom -Path $resultNow -Obj $result
Write-JsonNoBom -Path $resultLatest -Obj $result

$md = @()
$md += "# PixelSplat Phase2 Compare"
$md += ""
$md += "- timestamp: $ts"
$md += "- run_tag: $runTag"
$md += "- modal_exit_code: $rc"
$md += "- modal_run_url: $runUrl"
$md += "- remote_summary: $remoteSummary"
$md += "- local_summary: $localSummary"
$md += "- fetch_ok: $fetchOk"
if ($summaryObj -ne $null) {
    $md += "- checkpoint: $($summaryObj.checkpoint)"
    $md += "- uniform.psnr: $($summaryObj.uniform.mean_psnr)"
    $md += "- uniform.ssim: $($summaryObj.uniform.mean_ssim)"
    $md += "- uniform.l1: $($summaryObj.uniform.mean_l1)"
    $md += "- random.psnr: $($summaryObj.random.mean_psnr)"
    $md += "- random.ssim: $($summaryObj.random.mean_ssim)"
    $md += "- random.l1: $($summaryObj.random.mean_l1)"
    $md += "- delta.psnr: $($summaryObj.delta_uniform_minus_random.psnr)"
    $md += "- delta.ssim: $($summaryObj.delta_uniform_minus_random.ssim)"
    $md += "- delta.l1: $($summaryObj.delta_uniform_minus_random.l1)"
}
Set-Content -Path $resultMdLatest -Value ($md -join "`n") -Encoding UTF8

Write-Host "[pixelsplat-phase2] wrote: $resultLatest"
Write-Host "[pixelsplat-phase2] wrote: $resultMdLatest"
if ($rc -ne 0 -or -not $fetchOk) {
    exit 2
}
exit 0
