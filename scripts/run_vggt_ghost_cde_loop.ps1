param(
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$StartPseudoGeomSubdir = "vggt_geom",
    [string]$StartResumeCkpt = "",
    [string]$PretrainedCkpt = "model.pt",
    [string]$Lr = "1e-6",
    [double]$LrBackboneScale = 0.02,
    [double]$LrHeadScale = 8.0,
    [double]$LrCameraScale = 0.0,
    [double]$GradClip = 2.0,
    [double]$MinMeanStepUpdateRatio = 1e-10,
    [string]$LambdaPointMvDepthList = "0.001",
    [double]$LambdaPointMvMask = 0.0,
    [string]$LambdaPointMvMaskList = "",
    [double]$LambdaPointReproj = 0.05,
    [double]$LambdaPoint = 0.5,
    [double]$LambdaConf = 0.002,
    [int]$LambdaConfWarmupSteps = 80,
    [double]$LambdaCam = 0.0,
    [int]$LambdaCamWarmupSteps = 0,
    [int]$EpochsShort = 1,
    [int]$MaxFramesShort = 400,
    [int]$MaxStepsPerEpoch = 80,
    [int]$EvalNumSamples = 40,
    [string]$EvalInferArgsExtra = "--num_src_views=6",
    [string]$PointTargetMode = "depth_unproject",
    [double]$PointTargetBlendAlpha = 0.85,
    [double]$PointTargetBlendAlphaMin = 0.0,
    [double]$PointTargetBlendAlphaMax = 1.0,
    [double]$PointTargetConsensusAlphaFloor = 0.35,
    [double]$PointTargetBlendRelGain = 1.0,
    [double]$PointTargetBlendMvGain = 1.0,
    [string]$PointTargetBlendByReliability = "on",
    [string]$PointTargetBlendByMvSupport = "on",
    [string]$PointTargetBlendMvPolicy = "strong_to_depth",
    [string]$PointmapSource = "auto",
    [string]$TargetPointFrame = "auto",
    [string]$PredPointFrame = "auto",
    [string]$PrecomputeMvSupportOn = "off",
    [string]$UnprojectImpl = "legacy",
    [double]$PrecomputeMvSupportTolAbs = 0.06,
    [double]$PrecomputeMvSupportTolRel = 0.10,
    [int]$PrecomputeMvSupportStride = 2,
    [string]$PrecomputeMvSupportMode = "clip",
    [double]$PrecomputeMvSupportFloor = 0.05,
    [double]$PrecomputeMvSupportGamma = 1.0,
    [double]$PrecomputeMvSupportClipThr = 0.20,
    [double]$PrecomputeMvSupportClipFloor = 0.30,
    [double]$PrecomputeMvSupportHardThr = -1.0,
    [double]$PrecomputeMvConfValidFloor = 0.02,
    [string]$PrecomputeMvSupportSave = "off",
    [string]$PrecomputeMvSupportSaveRawConf = "off",
    [string]$SupervisionWeightMode = "mix",
    [double]$SupervisionWeightMixAlpha = 0.35,
    [double]$ConfWeightThr = 0.0,
    [double]$ConfWeightGamma = 1.0,
    [int]$FgMaskErodePx = 0,
    [int]$PointLossFgErodePx = 1,
    [int]$PointWarmupSteps = 10,
    [int]$PointReprojWarmupSteps = 10,
    [int]$PointMvDepthWarmupSteps = 10,
    [int]$PointMvMaskWarmupSteps = 10,
    [double]$PointLossScaleDepthUnproject = 1.0,
    [double]$PointConsClipMinQv = 1e-6,
    [double]$PointConsQuantile = 0.5,
    [string]$PointConsFocus = "inlier",
    [double]$PointResidualQuantile = 1.0,
    [string]$PointResidualFocus = "inlier",
    [double]$PointResidualBoost = 1.5,
    [double]$PointResidualBoostCap = 4.0,
    [string]$PointMvDepthInlierOnly = "off",
    [double]$PointMvDepthOutlierBoost = 1.5,
    [double]$PointMvDepthOutlierCap = 4.0,
    [string]$PointMvDepthTgtValidMode = "soft",
    [double]$PointMvDepthTgtValidFloor = 0.1,
    [double]$PointMvDepthMinTgtValidRatio = 0.0,
    [double]$PointMvMaskMinTgtFgRatio = 0.0,
    [double]$PointMvMaskHitThr = 0.5,
    [string]$PointMvDepthTgtValidScaleMode = "off",
    [double]$PointMvDepthTgtValidScaleThr = 0.01,
    [string]$PointMvDepthAdaptMode = "valid_ratio",
    [double]$PointMvDepthAdaptTargetValid = 0.01,
    [double]$PointMvDepthAdaptMinScale = 1.0,
    [double]$PointMvDepthAdaptMaxScale = 24.0,
    [string]$PointSupportMode = "off",
    [double]$PointSupportFloor = 0.0,
    [string]$PointMvDepthSupportMode = "off",
    [double]$PointMvDepthSupportFloor = 0.0,
    [int]$PointMvDepthFgErodePx = 0,
    [double]$PointMvTolAbs = 0.06,
    [double]$PointMvTolRel = 0.10,
    [double]$PointMvWeightFloor = 0.5,
    [int]$PointMvDepthMaxPairs = 2,
    [string]$PointMvDepthPairMode = "adjacent",
    [double]$PointMvDepthErrQuantile = 1.0,
    [int]$EvalEverySteps = 1,
    [int]$DebugMetricsEverySteps = 1,
    [int]$DebugVisEverySteps = 1,
    [int]$DebugVisMaxSteps = 60,
    [int]$DebugVisViews = 1,
    [string]$DebugVisDir = "",
    [int]$ModalRunTimeoutSec = 3600,
    [int]$RoundMax = 8,
    [int]$NoImprovePatience = 2,
    [double]$MinGhostImprove = 0.03,
    [double]$MinPSNRGuard = 20.2,
    [bool]$LockPseudoTarget = $true
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function San([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) { return "item" }
    return ([regex]::Replace($Raw, "[^A-Za-z0-9_.-]+", "_")).Trim("_")
}

function To-DoubleOrNaN($x) {
    try {
        if ($null -eq $x -or [string]::IsNullOrWhiteSpace([string]$x)) { return [double]::NaN }
        return [double]$x
    } catch {
        return [double]::NaN
    }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$rows = @()
$currPseudo = $StartPseudoGeomSubdir
$currResume = $StartResumeCkpt
$bestGhost = [double]::PositiveInfinity
$bestPsnr = [double]::NegativeInfinity
$noImproveRounds = 0

for ($round = 1; $round -le [Math]::Max(1, [int]$RoundMax); $round++) {
    Write-Host "[ghost-loop] round=$round pseudo=$currPseudo"
    & "$CodeDir\scripts\run_vggt_ghost_mvdepth_sweep.ps1" `
        -CodeDir $CodeDir `
        -SeqNames $SeqNames `
        -PseudoGeomSubdir $currPseudo `
        -PretrainedCkpt $PretrainedCkpt `
        -ResumeCkpt $currResume `
        -Lr $Lr `
        -LrBackboneScale $LrBackboneScale `
        -LrHeadScale $LrHeadScale `
        -LrCameraScale $LrCameraScale `
        -GradClip $GradClip `
        -MinMeanStepUpdateRatio $MinMeanStepUpdateRatio `
        -LambdaPointMvDepthList $LambdaPointMvDepthList `
        -LambdaPointMvMaskList $LambdaPointMvMaskList `
        -LambdaPointReproj $LambdaPointReproj `
        -LambdaPoint $LambdaPoint `
        -LambdaConf $LambdaConf `
        -LambdaConfWarmupSteps $LambdaConfWarmupSteps `
        -LambdaCam $LambdaCam `
        -LambdaCamWarmupSteps $LambdaCamWarmupSteps `
        -EpochsShort $EpochsShort `
        -MaxFramesShort $MaxFramesShort `
        -MaxStepsPerEpoch $MaxStepsPerEpoch `
        -EvalNumSamples $EvalNumSamples `
        -EvalInferArgsExtra $EvalInferArgsExtra `
        -PointTargetMode $PointTargetMode `
        -PointTargetBlendAlpha $PointTargetBlendAlpha `
        -PointTargetBlendAlphaMin $PointTargetBlendAlphaMin `
        -PointTargetBlendAlphaMax $PointTargetBlendAlphaMax `
        -PointTargetConsensusAlphaFloor $PointTargetConsensusAlphaFloor `
        -PointTargetBlendRelGain $PointTargetBlendRelGain `
        -PointTargetBlendMvGain $PointTargetBlendMvGain `
        -PointTargetBlendByReliability $PointTargetBlendByReliability `
        -PointTargetBlendByMvSupport $PointTargetBlendByMvSupport `
        -PointTargetBlendMvPolicy $PointTargetBlendMvPolicy `
        -PointmapSource $PointmapSource `
        -TargetPointFrame $TargetPointFrame `
        -PredPointFrame $PredPointFrame `
        -PrecomputeMvSupportOn $PrecomputeMvSupportOn `
        -UnprojectImpl $UnprojectImpl `
        -PrecomputeMvSupportTolAbs $PrecomputeMvSupportTolAbs `
        -PrecomputeMvSupportTolRel $PrecomputeMvSupportTolRel `
        -PrecomputeMvSupportStride $PrecomputeMvSupportStride `
        -PrecomputeMvSupportMode $PrecomputeMvSupportMode `
        -PrecomputeMvSupportFloor $PrecomputeMvSupportFloor `
        -PrecomputeMvSupportGamma $PrecomputeMvSupportGamma `
        -PrecomputeMvSupportClipThr $PrecomputeMvSupportClipThr `
        -PrecomputeMvSupportClipFloor $PrecomputeMvSupportClipFloor `
        -PrecomputeMvSupportHardThr $PrecomputeMvSupportHardThr `
        -PrecomputeMvConfValidFloor $PrecomputeMvConfValidFloor `
        -PrecomputeMvSupportSave $PrecomputeMvSupportSave `
        -PrecomputeMvSupportSaveRawConf $PrecomputeMvSupportSaveRawConf `
        -SupervisionWeightMode $SupervisionWeightMode `
        -SupervisionWeightMixAlpha $SupervisionWeightMixAlpha `
        -ConfWeightThr $ConfWeightThr `
        -ConfWeightGamma $ConfWeightGamma `
        -FgMaskErodePx $FgMaskErodePx `
        -PointLossFgErodePx $PointLossFgErodePx `
        -PointWarmupSteps $PointWarmupSteps `
        -PointReprojWarmupSteps $PointReprojWarmupSteps `
        -PointMvDepthWarmupSteps $PointMvDepthWarmupSteps `
        -PointMvMaskWarmupSteps $PointMvMaskWarmupSteps `
        -PointLossScaleDepthUnproject $PointLossScaleDepthUnproject `
        -PointConsClipMinQv $PointConsClipMinQv `
        -PointConsQuantile $PointConsQuantile `
        -PointConsFocus $PointConsFocus `
        -PointResidualQuantile $PointResidualQuantile `
        -PointResidualFocus $PointResidualFocus `
        -PointResidualBoost $PointResidualBoost `
        -PointResidualBoostCap $PointResidualBoostCap `
        -PointMvDepthInlierOnly $PointMvDepthInlierOnly `
        -PointMvDepthOutlierBoost $PointMvDepthOutlierBoost `
        -PointMvDepthOutlierCap $PointMvDepthOutlierCap `
        -PointMvDepthTgtValidMode $PointMvDepthTgtValidMode `
        -PointMvDepthTgtValidFloor $PointMvDepthTgtValidFloor `
        -PointMvDepthMinTgtValidRatio $PointMvDepthMinTgtValidRatio `
        -PointMvMaskMinTgtFgRatio $PointMvMaskMinTgtFgRatio `
        -PointMvMaskHitThr $PointMvMaskHitThr `
        -PointMvDepthTgtValidScaleMode $PointMvDepthTgtValidScaleMode `
        -PointMvDepthTgtValidScaleThr $PointMvDepthTgtValidScaleThr `
        -PointMvDepthAdaptMode $PointMvDepthAdaptMode `
        -PointMvDepthAdaptTargetValid $PointMvDepthAdaptTargetValid `
        -PointMvDepthAdaptMinScale $PointMvDepthAdaptMinScale `
        -PointMvDepthAdaptMaxScale $PointMvDepthAdaptMaxScale `
        -PointSupportMode $PointSupportMode `
        -PointSupportFloor $PointSupportFloor `
        -PointMvDepthSupportMode $PointMvDepthSupportMode `
        -PointMvDepthSupportFloor $PointMvDepthSupportFloor `
        -PointMvDepthFgErodePx $PointMvDepthFgErodePx `
        -PointMvTolAbs $PointMvTolAbs `
        -PointMvTolRel $PointMvTolRel `
        -PointMvWeightFloor $PointMvWeightFloor `
        -PointMvDepthMaxPairs $PointMvDepthMaxPairs `
        -PointMvDepthPairMode $PointMvDepthPairMode `
        -PointMvDepthErrQuantile $PointMvDepthErrQuantile `
        -LambdaPointMvMask $LambdaPointMvMask `
        -EvalEverySteps $EvalEverySteps `
        -DebugMetricsEverySteps $DebugMetricsEverySteps `
        -DebugVisEverySteps $DebugVisEverySteps `
        -DebugVisMaxSteps $DebugVisMaxSteps `
        -DebugVisViews $DebugVisViews `
        -DebugVisDir $DebugVisDir `
        -ModalRunTimeoutSec $ModalRunTimeoutSec `
        -NoImprovePatience 2 `
        -MinGhostImprove $MinGhostImprove `
        -MinPSNRGuard $MinPSNRGuard
    $rc = [int]$LASTEXITCODE

    $sweepLatest = "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv"
    if (-not (Test-Path $sweepLatest)) {
        $rows += [pscustomobject]@{
            round = $round
            status = "error"
            reason = "missing ghost_mvdepth_sweep_latest.csv"
            exit_code = $rc
            pseudo_geom_in = $currPseudo
            resume_ckpt_in = $currResume
            lambda_point_mv_depth = ""
            lambda_point_mv_mask = ""
            ghost_score_mean = ""
            mean_PSNR = ""
            mean_SSIM = ""
            mean_weighted_L1 = ""
            next_geom = ""
            next_ckpt = ""
        }
        break
    }

    $candRows = @(
        Import-Csv $sweepLatest |
            Where-Object { [int]$_.exit_code -eq 0 } |
            Sort-Object {
                To-DoubleOrNaN($_.ghost_score_mean)
            }, {
                -1.0 * (To-DoubleOrNaN($_.mean_PSNR))
            }
    )
    if ($candRows.Count -le 0) {
        $rows += [pscustomobject]@{
            round = $round
            status = "error"
            reason = "no valid candidate in ghost sweep"
            exit_code = $rc
            pseudo_geom_in = $currPseudo
            resume_ckpt_in = $currResume
            lambda_point_mv_depth = ""
            lambda_point_mv_mask = ""
            ghost_score_mean = ""
            mean_PSNR = ""
            mean_SSIM = ""
            mean_weighted_L1 = ""
            next_geom = ""
            next_ckpt = ""
        }
        break
    }

    $best = $candRows[0]
    $ghostNow = To-DoubleOrNaN($best.ghost_score_mean)
    $psnrNow = To-DoubleOrNaN($best.mean_PSNR)
    $ssimNow = To-DoubleOrNaN($best.mean_SSIM)
    $wl1Now = To-DoubleOrNaN($best.mean_weighted_L1)
    $mvBest = [string]$best.lambda_point_mv_depth
    $mvMaskBest = [string]$best.lambda_point_mv_mask
    $sweepCsv = [string]$best.sweep_csv

    $nextGeom = ""
    $nextCkpt = ""
    if (Test-Path $sweepCsv) {
        $sw = @(
            Import-Csv $sweepCsv |
                Where-Object { $_.status -eq "ok" -and $_.stage -eq "short" } |
                Select-Object -First 1
        )
        if ($sw.Count -gt 0) {
            $nextGeom = [string]$sw[0].geom_subdir
            $nextCkpt = [string]$sw[0].ft_ckpt
        }
    }

    $improved = $false
    $validRound = (
        (-not [double]::IsNaN($ghostNow)) -and
        (-not [double]::IsNaN($psnrNow)) -and
        ($psnrNow -ge $MinPSNRGuard)
    )
    if ($validRound) {
        $prevBestGhost = $bestGhost
        $strictBetter = ($ghostNow -lt $bestGhost)
        if ($strictBetter) {
            $bestGhost = $ghostNow
            $bestPsnr = $psnrNow
        }
        if ($ghostNow -le ($prevBestGhost - [Math]::Max(0.0, $MinGhostImprove))) {
            $improved = $true
            $noImproveRounds = 0
        } else {
            $noImproveRounds += 1
        }
    } else {
        $noImproveRounds += 1
    }

    $rows += [pscustomobject]@{
        round = $round
        status = $(if ($rc -eq 0) { "ok" } else { "error" })
        reason = $(if ($improved) { "improved" } else { "no_improve" })
        exit_code = $rc
        pseudo_geom_in = $currPseudo
        resume_ckpt_in = $currResume
        lambda_point_mv_depth = $mvBest
        lambda_point_mv_mask = $mvMaskBest
        point_target_consensus_alpha_floor = $PointTargetConsensusAlphaFloor
        ghost_score_mean = $ghostNow
        mean_PSNR = $psnrNow
        mean_SSIM = $ssimNow
        mean_weighted_L1 = $wl1Now
        next_geom = $nextGeom
        next_ckpt = $nextCkpt
    }

    if ($LockPseudoTarget) {
        if (-not [string]::IsNullOrWhiteSpace($nextCkpt)) {
            $currResume = $nextCkpt
        }
    } else {
        if (-not [string]::IsNullOrWhiteSpace($nextGeom) -and -not [string]::IsNullOrWhiteSpace($nextCkpt)) {
            $currPseudo = $nextGeom
            $currResume = $nextCkpt
        }
    }

    if ($noImproveRounds -ge [Math]::Max(1, [int]$NoImprovePatience)) {
        Write-Host "[ghost-loop] early stop: no-improve rounds=$noImproveRounds"
        break
    }
}

$outCsv = "logs/modal_phase5/ghost_cde_loop_$ts.csv"
$outLatest = "logs/modal_phase5/ghost_cde_loop_latest.csv"
$rows | Export-Csv $outCsv -NoTypeInformation -Encoding UTF8
$rows | Export-Csv $outLatest -NoTypeInformation -Encoding UTF8

$outMd = "logs/modal_phase5/ghost_cde_loop_latest.md"
$lines = @("# Ghost CDE Loop (Latest)", "")
foreach ($r in $rows) {
    $lines += "- round=$($r.round), status=$($r.status), reason=$($r.reason), mv=$($r.lambda_point_mv_depth), ghost=$($r.ghost_score_mean), psnr=$($r.mean_PSNR), ssim=$($r.mean_SSIM), wL1=$($r.mean_weighted_L1)"
}
$lines += ""
$lines += "- best_ghost=$bestGhost, best_psnr=$bestPsnr, no_improve_patience=$NoImprovePatience"
Set-Content -Path $outMd -Value ($lines -join "`n") -Encoding UTF8

exit 0
