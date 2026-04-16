param(
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$PseudoGeomSubdir = "vggt_geom_ft_lr_1e-7_20260212_140645",
    [string]$PretrainedCkpt = "model.pt",
    [string]$ResumeCkpt = "/mnt/out/vggt/finetune/lr_1e-7_20260212_140645/ckpt/model_ft_zju.pt",
    [string]$Lr = "5e-8",
    [string]$WeightModeList = "conf,uniform,mix",
    [string]$FgMaskErodeList = "0,1",
    [double]$WeightMixAlpha = 0.5,
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

function Parse-Tokens([string]$Raw) {
    return @(
        $Raw -split "[,\s;|]+" |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_.Trim() }
    )
}

function San([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) { return "item" }
    return ([regex]::Replace($Raw, "[^A-Za-z0-9_.-]+", "_")).Trim("_")
}

$modes = Parse-Tokens -Raw $WeightModeList
$erodes = Parse-Tokens -Raw $FgMaskErodeList
if ($modes.Count -eq 0) { throw "WeightModeList is empty" }
if ($erodes.Count -eq 0) { throw "FgMaskErodeList is empty" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$rows = @()

foreach ($mode in $modes) {
    foreach ($er in $erodes) {
        $tag = "$(San($mode))_er$(San($er))"
        Write-Host "[ghost-weight] run mode=$mode erode=$er"
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
            -FgMaskErodePx ([int]$er) `
            -SupervisionWeightMode $mode `
            -SupervisionWeightMixAlpha $WeightMixAlpha `
            -LambdaPointReproj 0.05 `
            -PointReprojWarmupSteps 40 `
            -PointReprojClampPx 64 `
            -PointMvConsistency off `
            -LambdaPointMvDepth 0.0 `
            -EarlyStopPatience 0 `
            -MinImprove 0.0
        $rc = [int]$LASTEXITCODE

        $cmpLatest = "logs/modal_phase5/baseline_compare_latest.csv"
        $sweepLatest = "logs/modal_phase5/vggt_ft_sweep_latest.csv"
        $cmpOut = "logs/modal_phase5/baseline_compare_ghost_weight_${tag}_$ts.csv"
        $sweepOut = "logs/modal_phase5/vggt_ft_sweep_ghost_weight_${tag}_$ts.csv"
        if (Test-Path $cmpLatest) { Copy-Item $cmpLatest $cmpOut -Force }
        if (Test-Path $sweepLatest) { Copy-Item $sweepLatest $sweepOut -Force }

        $best = $null
        if (Test-Path $cmpLatest) {
            $cand = @(
                Import-Csv $cmpLatest |
                    Where-Object { $_.status -eq "ok" } |
                    Sort-Object { [double]$_.mean_PSNR } -Descending
            )
            if ($cand.Count -gt 0) { $best = $cand[0] }
        }

        $ghostMean = ""
        $ghostP95 = ""
        $ghostSummaryCsv = ""
        if ($best -ne $null) {
            $label = [string]$best.label
            $outDir = [string]$best.infer_out_dir
            $mts = ""
            if ($outDir -match "(\d{8}_\d{6})$") { $mts = $Matches[1] }
            if (-not [string]::IsNullOrWhiteSpace($mts)) {
                $pattern = "logs/modal_phase5/baseline_${label}_${mts}_infer_*_cat_fg_mask_pred_tgt_step*.png"
                $ghostCsv = "logs/modal_phase5/ghost_score_rows_${tag}_$ts.csv"
                $ghostSummaryCsv = "logs/modal_phase5/ghost_score_summary_${tag}_$ts.csv"
                $ghostJson = "logs/modal_phase5/ghost_score_${tag}_$ts.json"
                python "$CodeDir\tools\score_ghosting_from_cat_pred.py" `
                    --input "$tag=$pattern" `
                    --out_csv "$ghostCsv" `
                    --out_summary_csv "$ghostSummaryCsv" `
                    --out_json "$ghostJson" | Out-Null
                if (Test-Path $ghostSummaryCsv) {
                    $grows = @(Import-Csv $ghostSummaryCsv)
                    if ($grows.Count -gt 0) {
                        $ghostMean = [double]$grows[0].ghost_score_mean
                        $ghostP95 = [double]$grows[0].ghost_score_p95
                    }
                }
            }
        }

        $rows += [pscustomobject]@{
            mode = $mode
            fg_mask_erode_px = [int]$er
            exit_code = $rc
            best_label = $(if ($best) { [string]$best.label } else { "" })
            best_geom_subdir = $(if ($best) { [string]$best.geom_subdir } else { "" })
            mean_PSNR = $(if ($best) { [double]$best.mean_PSNR } else { "" })
            mean_SSIM = $(if ($best) { [double]$best.mean_SSIM } else { "" })
            mean_weighted_L1 = $(if ($best) { [double]$best.mean_weighted_L1 } else { "" })
            ghost_score_mean = $ghostMean
            ghost_score_p95 = $ghostP95
            baseline_compare_csv = $cmpOut
            sweep_csv = $sweepOut
            ghost_summary_csv = $ghostSummaryCsv
        }
        Write-Host "[ghost-weight] mode=$mode erode=$er rc=$rc"
    }
}

$outCsv = "logs/modal_phase5/ghost_weight_sweep_$ts.csv"
$outLatest = "logs/modal_phase5/ghost_weight_sweep_latest.csv"
$rows | Export-Csv $outCsv -NoTypeInformation -Encoding UTF8
$rows | Export-Csv $outLatest -NoTypeInformation -Encoding UTF8

$outMd = "logs/modal_phase5/ghost_weight_sweep_latest.md"
$lines = @("# Ghost Weight Sweep (Latest)", "")
foreach ($r in $rows) {
    $lines += "- mode=$($r.mode), erode=$($r.fg_mask_erode_px), rc=$($r.exit_code), PSNR=$($r.mean_PSNR), SSIM=$($r.mean_SSIM), wL1=$($r.mean_weighted_L1), ghost=$($r.ghost_score_mean)"
}
Set-Content -Path $outMd -Value ($lines -join "`n") -Encoding UTF8

if ((@($rows | Where-Object { [int]$_.exit_code -ne 0 }).Count) -gt 0) {
    exit 2
}
exit 0
