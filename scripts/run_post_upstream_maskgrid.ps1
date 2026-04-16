param(
    [int]$WaitPid,
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
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
    [string]$PrecomputeMvSupportOn = "on",
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

try {
    Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue
} catch {
}

function To-DoubleOrNaN($x) {
    try {
        if ($null -eq $x -or [string]::IsNullOrWhiteSpace([string]$x)) { return [double]::NaN }
        return [double]$x
    } catch {
        return [double]::NaN
    }
}

$resumeCkpt = ""
try {
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
            $sweepCsv = [string]$best[0].sweep_csv
            if (Test-Path $sweepCsv) {
                $row = @(
                    Import-Csv $sweepCsv |
                        Where-Object { $_.status -eq "ok" -and $_.stage -eq "short" } |
                        Select-Object -First 1
                )
                if ($row.Count -gt 0) {
                    $resumeCkpt = [string]$row[0].ft_ckpt
                }
            }
        }
    }
} catch {
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = "logs/modal_phase5/post_upstream_maskgrid_$ts.out.log"
$errLog = "logs/modal_phase5/post_upstream_maskgrid_$ts.err.log"

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
    "-PrecomputeMvSupportOn", $PrecomputeMvSupportOn,
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
    triggered_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    wait_pid = $WaitPid
    resume_ckpt = $resumeCkpt
    command = "powershell.exe " + ($psArgs -join " ")
    arg_count = @($psArgs).Count
    stdout = $outLog
    stderr = $errLog
}

$meta | ConvertTo-Json -Depth 10 | Set-Content -Path "logs/modal_phase5/post_upstream_maskgrid_latest.json" -Encoding UTF8

if (@($psArgs).Count -le 0) {
    throw "post-upstream ArgumentList is empty"
}
$p = Start-Process -FilePath "powershell.exe" -ArgumentList @($psArgs) -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
$meta.proc_pid = $p.Id
$meta.started_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
$meta | ConvertTo-Json -Depth 10 | Set-Content -Path "logs/modal_phase5/post_upstream_maskgrid_latest.json" -Encoding UTF8
Write-Host "[post-upstream] launched pid=$($p.Id) modal_timeout=$ModalRunTimeoutSec no_output_timeout=$ModalRunNoOutputTimeoutSec point_head_no_output_timeout=$ModalRunNoOutputTimeoutSecPointHead out=$outLog"

exit 0
