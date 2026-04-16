param(
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$PseudoGeomSubdir = "vggt_geom_ft_lr_1e-7_20260212_140645",
    [string]$PretrainedCkpt = "model.pt",
    [string]$ResumeCkpt = "/mnt/out/vggt/finetune/lr_1e-7_20260212_140645/ckpt/model_ft_zju.pt",
    [string]$Lr = "5e-8",
    [string]$LambdaPointReprojList = "0.0,0.02,0.05",
    [int]$EpochsShort = 1,
    [int]$MaxFramesShort = 400,
    [int]$MaxStepsPerEpoch = 80,
    [int]$EvalNumSamples = 40,
    [string]$EvalInferArgsExtra = "--num_src_views=6"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Parse-Nums([string]$Raw) {
    return @(
        $Raw -split "[,\s;|]+" |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_.Trim() }
    )
}

$vals = Parse-Nums -Raw $LambdaPointReprojList
if ($vals.Count -eq 0) {
    throw "LambdaPointReprojList is empty"
}
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$rows = @()
foreach ($lpr in $vals) {
    Write-Host "[ghost-fix] lambda_point_reproj=$lpr start"
    & "$CodeDir\scripts\run_vggt_ft_lr_sweep.ps1" `
        -CodeDir $CodeDir `
        -SeqNames $SeqNames `
        -PseudoGeomSubdir $PseudoGeomSubdir `
        -PretrainedCkpt $PretrainedCkpt `
        -ResumeCkpt $ResumeCkpt `
        -LrList $Lr `
        -FreezeMode all_trainable `
        -DepthScaleAlign median `
        -EpochsShort $EpochsShort `
        -MaxFramesShort $MaxFramesShort `
        -MaxStepsPerEpoch $MaxStepsPerEpoch `
        -EvalNumSamples $EvalNumSamples `
        -EvalInferArgsExtra $EvalInferArgsExtra `
        -PointTargetMode depth_unproject `
        -UseFgMask on `
        -FgMaskSource mask `
        -LambdaPointReproj ([double]$lpr) `
        -PointReprojWarmupSteps 40 `
        -PointReprojClampPx 64 `
        -PointMvConsistency off `
        -LambdaPointMvDepth 0.0 `
        -EarlyStopPatience 0 `
        -MinImprove 0.0
    $rc = $LASTEXITCODE

    $cmpLatest = "logs/modal_phase5/baseline_compare_latest.csv"
    $sweepLatest = "logs/modal_phase5/vggt_ft_sweep_latest.csv"
    $tag = ("lpr_" + ($lpr -replace "[^0-9eE\\.-]", "_"))
    $cmpOut = "logs/modal_phase5/baseline_compare_ghost_fix_${tag}_$ts.csv"
    $sweepOut = "logs/modal_phase5/vggt_ft_sweep_ghost_fix_${tag}_$ts.csv"
    if (Test-Path $cmpLatest) { Copy-Item $cmpLatest $cmpOut -Force }
    if (Test-Path $sweepLatest) { Copy-Item $sweepLatest $sweepOut -Force }

    $best = $null
    if (Test-Path $cmpLatest) {
        $cand = @(Import-Csv $cmpLatest | Where-Object { $_.status -eq "ok" } | Sort-Object { [double]$_.mean_PSNR } -Descending)
        if ($cand.Count -gt 0) { $best = $cand[0] }
    }

    $rows += [pscustomobject]@{
        lambda_point_reproj = $lpr
        exit_code = [int]$rc
        best_label = $(if ($best) { [string]$best.label } else { "" })
        best_geom_subdir = $(if ($best) { [string]$best.geom_subdir } else { "" })
        mean_PSNR = $(if ($best) { [double]$best.mean_PSNR } else { "" })
        mean_SSIM = $(if ($best) { [double]$best.mean_SSIM } else { "" })
        mean_weighted_L1 = $(if ($best) { [double]$best.mean_weighted_L1 } else { "" })
        baseline_compare_csv = $cmpOut
        sweep_csv = $sweepOut
    }
    Write-Host "[ghost-fix] lambda_point_reproj=$lpr exit_code=$rc"
}

$outCsv = "logs/modal_phase5/ghost_fix_sweep_$ts.csv"
$outLatest = "logs/modal_phase5/ghost_fix_sweep_latest.csv"
$rows | Export-Csv $outCsv -NoTypeInformation -Encoding UTF8
$rows | Export-Csv $outLatest -NoTypeInformation -Encoding UTF8
Write-Host "[ghost-fix] wrote: $outLatest"

$outMd = "logs/modal_phase5/ghost_fix_sweep_latest.md"
$lines = @("# Ghost Fix Sweep (Latest)", "")
foreach ($r in $rows) {
    $lines += "- lambda_point_reproj=$($r.lambda_point_reproj), rc=$($r.exit_code), PSNR=$($r.mean_PSNR), SSIM=$($r.mean_SSIM), wL1=$($r.mean_weighted_L1), label=$($r.best_label)"
}
Set-Content -Path $outMd -Value ($lines -join "`n") -Encoding UTF8

if ((@($rows | Where-Object { [int]$_.exit_code -ne 0 }).Count) -gt 0) {
    exit 2
}
exit 0
