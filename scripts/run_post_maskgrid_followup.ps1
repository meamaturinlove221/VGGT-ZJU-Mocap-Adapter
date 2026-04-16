param(
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [int]$WaitJsonTimeoutSec = 14400,
    [int]$PollSec = 30,
    [double]$GhostTarget = 4.45,
    [double]$PsnrFloor = 21.7,
    [string]$PseudoGeomSubdir = "vggt_geom",
    [string]$Lr = "1e-6",
    [string]$LambdaPointMvDepthList = "0.0005,0.001",
    [string]$LambdaPointMvMaskList = "0,0.0005,0.001,0.002",
    [int]$EpochsShort = 1,
    [int]$MaxFramesShort = 400,
    [int]$MaxStepsPerEpoch = 80,
    [int]$EvalNumSamples = 40,
    [string]$UnprojectImpl = "upstream433",
    [string]$PointTargetMode = "depth_consensus_unproject",
    [double]$PointTargetConsensusAlphaFloor = 0.35,
    [string]$PointTargetBlendMvPolicy = "strong_to_depth",
    [string]$PointmapSource = "point_head",
    [int]$ModalRunTimeoutSec = 3600,
    [int]$ModalRunNoOutputTimeoutSec = 600,
    [int]$ModalRunNoOutputTimeoutSecPointHead = 600
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
Set-Location $CodeDir

function To-DoubleOrNaN($x) {
    try {
        if ($null -eq $x -or [string]::IsNullOrWhiteSpace([string]$x)) { return [double]::NaN }
        return [double]$x
    } catch {
        return [double]::NaN
    }
}

$start = Get-Date
$latestJson = "logs/modal_phase5/post_upstream_maskgrid_latest.json"
$procPid = $null
while (((Get-Date) - $start).TotalSeconds -lt [Math]::Max(60, $WaitJsonTimeoutSec)) {
    if (Test-Path $latestJson) {
        try {
            $obj = Get-Content $latestJson -Raw | ConvertFrom-Json
            if ($null -ne $obj.proc_pid -and [int]$obj.proc_pid -gt 0) {
                $procPid = [int]$obj.proc_pid
                Write-Host "[post-followup] found upstream pid=$procPid from $latestJson"
                break
            }
        } catch {
        }
    }
    Write-Host "[post-followup] waiting upstream pid from $latestJson ..."
    Start-Sleep -Seconds ([Math]::Max(5, $PollSec))
}

if ($null -eq $procPid) {
    $status = [ordered]@{
        timestamp = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        status = "skip"
        reason = "maskgrid_proc_pid_not_found"
    }
    $status | ConvertTo-Json -Depth 10 | Set-Content -Path "logs/modal_phase5/post_maskgrid_followup_latest.json" -Encoding UTF8
    exit 0
}

try {
    Write-Host "[post-followup] waiting upstream pid=$procPid to finish ..."
    Wait-Process -Id $procPid -ErrorAction SilentlyContinue
} catch {
}

$best = $null
$resumeCkpt = ""
$latestSweep = "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv"
if (Test-Path $latestSweep) {
    $best = @(
        Import-Csv $latestSweep |
            Where-Object { [int]$_.exit_code -eq 0 -and -not [string]::IsNullOrWhiteSpace([string]$_.ghost_score_mean) } |
            Sort-Object {
                To-DoubleOrNaN($_.ghost_score_mean)
            }, {
                -1.0 * (To-DoubleOrNaN($_.mean_PSNR))
            } |
            Select-Object -First 1
    )
    if ($best.Count -gt 0) {
        $sw = [string]$best[0].sweep_csv
        if (Test-Path $sw) {
            $row = @(
                Import-Csv $sw |
                    Where-Object { $_.status -eq "ok" -and $_.stage -eq "short" } |
                    Select-Object -First 1
            )
            if ($row.Count -gt 0) {
                $resumeCkpt = [string]$row[0].ft_ckpt
            }
        }
    }
}

$bestGhost = [double]::NaN
$bestPsnr = [double]::NaN
if ($best -and $best.Count -gt 0) {
    $bestGhost = To-DoubleOrNaN($best[0].ghost_score_mean)
    $bestPsnr = To-DoubleOrNaN($best[0].mean_PSNR)
}

$needFollow = $true
if ((-not [double]::IsNaN($bestGhost)) -and (-not [double]::IsNaN($bestPsnr))) {
    if (($bestGhost -le $GhostTarget) -and ($bestPsnr -ge $PsnrFloor)) {
        $needFollow = $false
    }
}

if (-not $needFollow) {
    $status = [ordered]@{
        timestamp = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        status = "skip"
        reason = "maskgrid_already_good"
        best_ghost = $bestGhost
        best_psnr = $bestPsnr
    }
    $status | ConvertTo-Json -Depth 10 | Set-Content -Path "logs/modal_phase5/post_maskgrid_followup_latest.json" -Encoding UTF8
    exit 0
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = "logs/modal_phase5/post_maskgrid_followup_$ts.out.log"
$errLog = "logs/modal_phase5/post_maskgrid_followup_$ts.err.log"

$psArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", "scripts/run_vggt_ghost_mvdepth_sweep.ps1",
    "-CodeDir", $CodeDir,
    "-SeqNames", $SeqNames,
    "-PseudoGeomSubdir", $PseudoGeomSubdir,
    "-Lr", $Lr,
    "-LambdaPointMvDepthList", $LambdaPointMvDepthList,
    "-LambdaPointMvMaskList", $LambdaPointMvMaskList,
    "-EpochsShort", [string]$EpochsShort,
    "-MaxFramesShort", [string]$MaxFramesShort,
    "-MaxStepsPerEpoch", [string]$MaxStepsPerEpoch,
    "-EvalNumSamples", [string]$EvalNumSamples,
    "-UnprojectImpl", $UnprojectImpl,
    "-PointTargetMode", $PointTargetMode,
    "-PointTargetConsensusAlphaFloor", [string]$PointTargetConsensusAlphaFloor,
    "-PointTargetBlendMvPolicy", $PointTargetBlendMvPolicy,
    "-PointmapSource", $PointmapSource,
    "-PrecomputeMvSupportOn", "off",
    "-NoImprovePatience", "99",
    "-MinGhostImprove", "0.0",
    "-ModalRunTimeoutSec", [string]$ModalRunTimeoutSec,
    "-ModalRunNoOutputTimeoutSec", [string]$ModalRunNoOutputTimeoutSec,
    "-ModalRunNoOutputTimeoutSecPointHead", [string]$ModalRunNoOutputTimeoutSecPointHead,
    "-NoOutputProbeTimeoutSec", [string]([Math]::Max(300, [int]$ModalRunNoOutputTimeoutSec + 60)),
    "-NoOutputProbeTimeoutSecPointHeadWeak", [string]([Math]::Max(300, [int]$ModalRunNoOutputTimeoutSec + 60))
)
if (-not [string]::IsNullOrWhiteSpace($resumeCkpt)) {
    $psArgs += @("-ResumeCkpt", $resumeCkpt)
}

$meta = [ordered]@{
    timestamp = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    status = "started"
    based_on_best_ghost = $bestGhost
    based_on_best_psnr = $bestPsnr
    resume_ckpt = $resumeCkpt
    command = "powershell.exe " + ($psArgs -join " ")
    arg_count = @($psArgs).Count
    stdout = $outLog
    stderr = $errLog
}
$meta | ConvertTo-Json -Depth 10 | Set-Content -Path "logs/modal_phase5/post_maskgrid_followup_latest.json" -Encoding UTF8

if (@($psArgs).Count -le 0) {
    throw "post-followup ArgumentList is empty"
}
$p = Start-Process -FilePath "powershell.exe" -ArgumentList @($psArgs) -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
$meta.proc_pid = $p.Id
$meta.started_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
$meta | ConvertTo-Json -Depth 10 | Set-Content -Path "logs/modal_phase5/post_maskgrid_followup_latest.json" -Encoding UTF8
Write-Host "[post-followup] launched pid=$($p.Id) modal_timeout=$ModalRunTimeoutSec no_output_timeout=$ModalRunNoOutputTimeoutSec point_head_no_output_timeout=$ModalRunNoOutputTimeoutSecPointHead out=$outLog"

exit 0
