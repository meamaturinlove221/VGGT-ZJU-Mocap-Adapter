param(
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$PseudoGeomSubdir = "vggt_geom_ft_lr_1e-7_20260212_140645",
    [string]$PretrainedCkpt = "model.pt",
    [string]$ResumeCkpt = "",
    [string]$Lr = "1e-7",
    [string]$Modes = "pointmap,depth_unproject,blend,depth_consensus_unproject",
    [int]$EpochsShort = 1,
    [int]$MaxFramesShort = 400,
    [int]$MaxStepsPerEpoch = 80,
    [int]$EvalNumSamples = 40,
    [string]$EvalInferArgsExtra = "--num_src_views=6",
    [string]$DecoderCkpt = "",
    [double]$PointTargetBlendAlpha = 0.7,
    [double]$PointTargetConsensusAlphaFloor = 0.35,
    [double]$LambdaPointReproj = 0.0,
    [string]$UseFgMask = "off",
    [string]$FgMaskSource = "auto",
    [int]$PointReprojWarmupSteps = 40,
    [double]$PointReprojClampPx = 64.0
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Sanitize([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) { return "item" }
    return ([regex]::Replace($Raw, "[^A-Za-z0-9_.-]+", "_")).Trim("_")
}

function Parse-Modes([string]$Raw) {
    return @(
        $Raw -split "[,\s;|]+" |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_.Trim().ToLower() } |
            Select-Object -Unique
    )
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$rows = @()
$modeList = Parse-Modes -Raw $Modes
if ($modeList.Count -eq 0) {
    throw "Modes is empty: $Modes"
}

foreach ($mode in $modeList) {
    if (@("pointmap", "depth_unproject", "blend", "depth_consensus_unproject") -notcontains $mode) {
        throw "unsupported mode: $mode"
    }

    Write-Host "[point-target-sweep] mode=$mode start"
    $argv = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts/run_vggt_ft_lr_sweep.ps1",
        "-CodeDir", $CodeDir,
        "-SeqNames", $SeqNames,
        "-PseudoGeomSubdir", $PseudoGeomSubdir,
        "-PretrainedCkpt", $PretrainedCkpt,
        "-LrList", $Lr,
        "-FreezeMode", "all_trainable",
        "-DepthScaleAlign", "median",
        "-EpochsShort", [string]$EpochsShort,
        "-MaxFramesShort", [string]$MaxFramesShort,
        "-EvalNumSamples", [string]$EvalNumSamples,
        "-EvalInferArgsExtra", $EvalInferArgsExtra,
        "-LambdaDepth", "1.0",
        "-LambdaPoint", "0.5",
        "-LambdaPointReproj", [string]$LambdaPointReproj,
        "-LambdaPointMvDepth", "0.0",
        "-LambdaConf", "0.02",
        "-LambdaGeomCons", "0.05",
        "-LambdaCam", "0.03",
        "-Jitter", "0.02",
        "-NoiseStd", "0.0",
        "-PointTargetMode", $mode,
        "-PointTargetBlendAlpha", [string]$PointTargetBlendAlpha,
        "-PointTargetConsensusAlphaFloor", [string]$PointTargetConsensusAlphaFloor,
        "-PointTargetBlendByReliability", "on",
        "-UseFgMask", $UseFgMask,
        "-FgMaskSource", $FgMaskSource,
        "-PointReprojWarmupSteps", [string]$PointReprojWarmupSteps,
        "-PointReprojClampPx", [string]$PointReprojClampPx,
        "-PointMvConsistency", "off",
        "-EarlyStopPatience", "0",
        "-MinImprove", "0.0",
        "-MaxStepsPerEpoch", [string]$MaxStepsPerEpoch
    )
    if (-not [string]::IsNullOrWhiteSpace($DecoderCkpt)) {
        $argv += @("-DecoderCkpt", $DecoderCkpt)
    }
    if (-not [string]::IsNullOrWhiteSpace($ResumeCkpt)) {
        $argv += @("-ResumeCkpt", $ResumeCkpt)
    }
    & powershell @argv
    $rc = $LASTEXITCODE

    $modeSafe = Sanitize($mode)
    $cmpLatest = "logs/modal_phase5/baseline_compare_latest.csv"
    $sweepLatest = "logs/modal_phase5/vggt_ft_sweep_latest.csv"
    $cmpOut = "logs/modal_phase5/baseline_compare_point_target_${modeSafe}_$ts.csv"
    $sweepOut = "logs/modal_phase5/vggt_ft_sweep_point_target_${modeSafe}_$ts.csv"
    if (Test-Path $cmpLatest) { Copy-Item $cmpLatest $cmpOut -Force }
    if (Test-Path $sweepLatest) { Copy-Item $sweepLatest $sweepOut -Force }

    $best = $null
    if (Test-Path $cmpLatest) {
        $cand = @(
            Import-Csv $cmpLatest |
                Where-Object { $_.status -eq "ok" } |
                Sort-Object { [double]$_.mean_PSNR } -Descending
        )
        if ($cand.Count -gt 0) {
            $best = $cand[0]
        }
    }

    $rows += [pscustomobject]@{
        mode = $mode
        exit_code = [int]$rc
        best_label = $(if ($best) { [string]$best.label } else { "" })
        best_geom_subdir = $(if ($best) { [string]$best.geom_subdir } else { "" })
        mean_PSNR = $(if ($best) { [double]$best.mean_PSNR } else { "" })
        mean_SSIM = $(if ($best) { [double]$best.mean_SSIM } else { "" })
        mean_weighted_L1 = $(if ($best) { [double]$best.mean_weighted_L1 } else { "" })
        baseline_compare_csv = $cmpOut
        sweep_csv = $sweepOut
    }
    Write-Host "[point-target-sweep] mode=$mode exit_code=$rc"
}

$outCsv = "logs/modal_phase5/point_target_sweep_$ts.csv"
$outLatest = "logs/modal_phase5/point_target_sweep_latest.csv"
$rows | Export-Csv $outCsv -NoTypeInformation -Encoding UTF8
$rows | Export-Csv $outLatest -NoTypeInformation -Encoding UTF8
Write-Host "[point-target-sweep] wrote: $outLatest"

$outMd = "logs/modal_phase5/point_target_sweep_latest.md"
$lines = @()
$lines += "# Point Target Sweep (Latest)"
$lines += ""
foreach ($r in $rows) {
    $lines += "- mode=$($r.mode), rc=$($r.exit_code), PSNR=$($r.mean_PSNR), SSIM=$($r.mean_SSIM), wL1=$($r.mean_weighted_L1), label=$($r.best_label)"
}
Set-Content -Path $outMd -Value ($lines -join "`n") -Encoding UTF8

if ((@($rows | Where-Object { [int]$_.exit_code -ne 0 }).Count) -gt 0) {
    exit 2
}
exit 0
