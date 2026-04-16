param(
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$StartPseudoGeomSubdir = "vggt_geom",
    [string]$StartResumeCkpt = "/mnt/out/vggt/finetune/lr_1e-6_20260219_151024/ckpt/model_ft_zju.pt",
    [string]$PretrainedCkpt = "model.pt",
    [string]$Lr = "1e-6",
    [string]$LambdaPointMvDepthList = "0.0005,0.001,0.0015",
    [string]$LambdaPointMvMaskList = "0,0.0005,0.001,0.002",
    [int]$EvalNumSamples = 40,
    [int]$MaxFramesShort = 400,
    [int]$MaxStepsPerEpoch = 80,
    [int]$ModalRunTimeoutSec = 3600,
    [int]$ModalRunNoOutputTimeoutSec = 600,
    [int]$ModalRunNoOutputMaxRetries = 1,
    [bool]$ModalRunQuiet = $true,
    [int]$InfraNoOutputStopConsecutive = 2,
    [int]$Stage1ModalRunNoOutputTimeoutSec = 480,
    [int]$Stage1ModalRunNoOutputMaxRetries = 1,
    [int]$Stage1InfraNoOutputStopConsecutive = 1,
    [int]$Stage2ModalRunNoOutputTimeoutSec = 600,
    [int]$Stage2ModalRunNoOutputMaxRetries = 1,
    [int]$Stage2InfraNoOutputStopConsecutive = 2,
    [bool]$Stage2EnableInfraRecoveryNoOutputRelax = $true,
    [int]$Stage2InfraRecoveryNoOutputTimeoutSec = 1200,
    [int]$Stage2InfraRecoveryNoOutputMaxRetries = 1,
    [int]$Stage2InfraRecoveryMinStopConsecutive = 2,
    [bool]$EnablePreemptiveProbeForPointHead = $true,
    [int]$PreemptiveProbeMaxCandidates = 2,
    [int]$NoOutputProbeTimeoutSecDepthUnproject = 600,
    [int]$PrecomputeNoOutputTimeoutSecDepthUnproject = 900,
    [int]$StageNoImprovePatience = 6,
    [int]$Stage3NoImprovePatience = 4,
    [int]$Stage1NoImprovePatience = 5,
    [string]$Stage1PointmapSource = "depth_unproject",
    [string]$Stage1LambdaPointMvDepthList = "0.0015,0.001,0.0005",
    [string]$Stage1LambdaPointMvMaskList = "0.0005,0.001,0,0.0015",
    [int]$Stage1HistoryTopRows = 12,
    [int]$Stage1FocusMaxDepthValues = 2,
    [int]$Stage1FocusMaxMaskValues = 1,
    [bool]$Stage1PrioritizeHistoryTopAfterCrossHint = $true,
    [bool]$Stage1EnableCrossStageHintFromGlobal = $true,
    [double]$Stage1CrossStageHintGhostLag = 0.12,
    [double]$Stage1GhostPressureLagThreshold = 0.15,
    [double]$Stage1GhostPressureMaskSoftMix = 0.45,
    [double]$Stage1GhostPressureMaskSoftHitThr = 0.5,
    [double]$Stage1GhostPressureMaskMinTgtFgRatio = 0.01,
    [double]$Stage1GhostPressureMaskMinValue = 0.0005,
    [double]$Stage1GhostPressureMaskMaxValue = 0.0010,
    [bool]$Stage1GhostPressurePreserveMaskHints = $true,
    [int]$Stage1GhostPressureMaxMaskValues = 2,
    [int]$Stage1GhostPressureStride = 1,
    [int]$Stage1GhostPressureDepthMaxPairs = 3,
    [string]$Stage1GhostPressureDepthSupportMode = "direct",
    [double]$Stage1GhostPressureDepthSupportFloor = 0.05,
    [int]$Stage1AggressiveFocusNoImproveCycles = 2,
    [int]$Stage1AggressiveFocusMaxDepthValues = 1,
    [int]$Stage1AggressiveFocusMaxMaskValues = 1,
    [int]$Stage2NoImprovePatience = 5,
    [string]$Stage2LambdaPointMvDepthList = "0.001,0.0015",
    [string]$Stage2LambdaPointMvMaskList = "0,0.0005,0.001",
    [int]$Stage2AggressiveFocusMaxDepthValues = 1,
    [double]$Stage2MaskHardMaxValue = 0.001,
    [double]$Stage2DepthHardMinValue = 0.001,
    [int]$Stage2HistoryTopRows = 10,
    [int]$Stage2HistoryMinDepthValues = 2,
    [int]$Stage2HistoryMinMaskValues = 2,
    [int]$Stage2FocusMaxDepthValues = 2,
    [int]$Stage2FocusMaxMaskValues = 2,
    [int]$Stage2AggressiveFocusMaxMaskValues = 3,
    [double]$Stage2MaskLagExpandThreshold = 0.25,
    [int]$Stage2MaskHardMinCountWhenLagged = 2,
    [bool]$Stage2MaskHardPreferZeroPreserve = $true,
    [bool]$Stage2MaskDisableZeroPreserveWhenLagged = $true,
    [double]$Stage2MaskZeroPreserveLagThreshold = 0.30,
    [object]$Stage2HistoryQualityAware = $true,
    [double]$Stage2HistoryMaxPSNRDrop = 0.12,
    [double]$Stage2HistoryMaxSSIMDrop = 0.006,
    [double]$Stage2HistoryMaxWl1Rise = 0.0015,
    [object]$Stage2DualLaneEnabled = $true,
    [object]$PostRescueEnabled = $true,
    [string]$QualityGuardMode = "layered",
    [double]$PromotionGhostDelta = 0.02,
    [object]$EmergencyGhostShockEnabled = $false,
    [int]$EmergencyShockWindowMinutes = 90,
    [int]$EmergencyShockExtraCycles = 1,
    [double]$EmergencyShockTargetGhost = 3.3,
    [string]$EmergencyShockPrimaryDepthList = "0.001",
    [string]$EmergencyShockPrimaryMaskList = "0",
    [string]$EmergencyShockFallbackDepthList = "0.0015",
    [string]$EmergencyShockFallbackMaskList = "0.0005",
    [double]$EmergencyShockFailGhostThreshold = 4.8,
    [int]$EmergencyShockFailConsecutiveLimit = 2,
    [int]$EmergencyStage1NoOutputTimeoutSec = 240,
    [double]$EmergencyShockMinPSNRGuard = 20.2,
    [double]$EmergencyShockMinSSIMGuard = 0.40,
    [double]$EmergencyShockMaxWl1Guard = 0.08,
    [double]$EmergencyRecoveryMinPSNRGuard = 20.9,
    [double]$EmergencyRecoveryMinSSIMGuard = 0.70,
    [double]$EmergencyRecoveryMaxWl1Guard = 0.08,
    [object]$EmergencyRecoveryEnableHistoryQualityAware = $true,
    [string]$EmergencyRollbackDepthList = "",
    [string]$EmergencyRollbackMaskList = "",
    [double]$LayeredExploreMaxPSNRDrop = 0.25,
    [double]$LayeredExploreMaxSSIMDrop = 0.010,
    [double]$LayeredExploreMaxWl1Rise = 0.0030,
    [double]$LayeredPromotionMaxPSNRDrop = 0.12,
    [double]$LayeredPromotionMaxSSIMDrop = 0.006,
    [double]$LayeredPromotionMaxWl1Rise = 0.0015,
    [int]$AggressiveRouteStartNoSubstantialCycles = 3,
    [string]$AggressiveStage2LambdaPointMvDepthList = "0.0015,0.001",
    [string]$AggressiveStage2LambdaPointMvMaskList = "0.001,0.0005,0",
    [double]$AggressiveStage2PointTargetConsensusAlphaFloor = 0.40,
    [double]$AggressiveStage2PointMvMaskSoftMix = 0.35,
    [double]$AggressiveStage2PointMvMaskSoftHitThr = 0.55,
    [double]$AggressiveStage2PointMvMaskMinTgtFgRatio = 0.03,
    [int]$AggressiveStage2PointMvStride = 1,
    [int]$AggressiveStage2PointMvDepthMaxPairs = 3,
    [string]$AggressiveStage2PointMvDepthSupportMode = "direct",
    [double]$AggressiveStage2PointMvDepthSupportFloor = 0.05,
    [double]$Stage2PotentialGhostLagThreshold = 0.12,
    [double]$Stage2PotentialVsStage1Improve = 0.03,
    [double]$Stage2PotentialMaxPSNRDrop = 0.25,
    [double]$Stage2PotentialMaxSSIMDrop = 0.01,
    [double]$Stage2PotentialMaxWl1Rise = 0.0025,
    [double]$Stage2GhostPressureLagThreshold = 0.12,
    [double]$Stage2GhostPressureMaskSoftMix = 0.35,
    [double]$Stage2GhostPressureMaskSoftHitThr = 0.52,
    [double]$Stage2GhostPressureMaskMinTgtFgRatio = 0.02,
    [double]$Stage2GhostPressureMaskMinValue = 0.0005,
    [double]$Stage2GhostPressureMaskMaxValue = 0.001,
    [int]$Stage2GhostPressureStride = 1,
    [int]$Stage2GhostPressureDepthMaxPairs = 3,
    [string]$Stage2GhostPressureDepthSupportMode = "direct",
    [double]$Stage2GhostPressureDepthSupportFloor = 0.05,
    [double]$Stage2GhostPressureConsensusAlphaFloor = 0.45,
    [double]$StageMinGhostImprove = 0.0,
    [double]$StageMinPSNRGuard = 20.9,
    [double]$StageMinSSIMGuard = 0.70,
    [double]$StageMaxWl1Guard = 0.08,
    [object]$StageEnableAbsoluteQualityGuard = $true,
    [string]$StageCamNames = "Camera_B1,Camera_B2,Camera_B3,Camera_B4,Camera_B5,Camera_B6,Camera_B7,Camera_B8,Camera_B9,Camera_B10,Camera_B11,Camera_B12,Camera_B13,Camera_B14,Camera_B15,Camera_B16,Camera_B17,Camera_B18,Camera_B19,Camera_B20,Camera_B21,Camera_B22,Camera_B23",
    [string]$Stage2EvalNumSrcViewsList = "12,16,20,22,8",
    [string]$Stage2PointMvDepthPairModeList = "adjacent,farthest",
    [bool]$EnableVisualAntiBlackGuard = $true,
    [double]$MinPredLumaMean = 0.045,
    [double]$MinPredNonBlackRatio = 0.10,
    [double]$MinAreaRatio = 0.55,
    [double]$MinWidthRatio = 0.65,
    [string]$PointTargetMode = "depth_consensus_unproject",
    [string]$BaseUseFgMask = "on",
    [string]$BaseFgMaskSource = "mask",
    [string]$BasePointTargetBlendMvRegionMode = "all",
    [double]$PointTargetConsensusAlphaFloor = 0.35,
    [string]$UnprojectImpl = "upstream433",
    [double]$LambdaPointReproj = 0.05,
    [double]$LambdaPoint = 0.5,
    [double]$LambdaConf = 0.002,
    [int]$LambdaConfWarmupSteps = 80,
    [double]$BasePointMvMaskHitThr = 0.5,
    [double]$BasePointMvMaskMinTgtFgRatio = 0.0,
    [int]$BasePointMvMaskSoftBlurPx = 1,
    [int]$BasePointMvMaskSoftBlurIters = 1,
    [double]$BasePointMvMaskSoftMix = 0.4,
    [double]$BasePointMvMaskSoftHitThr = 0.45,
    [int]$BasePointMvStride = 2,
    [int]$BasePointMvDepthMaxPairs = 2,
    [string]$BasePointMvDepthPairMode = "adjacent",
    [string]$BasePointMvDepthRegionMode = "all",
    [string]$BasePointMvDepthSupportMode = "off",
    [double]$BasePointMvDepthSupportFloor = 0.0,
    [string]$BasePointMvMaskSupportMode = "inverse",
    [double]$BasePointMvMaskSupportFloor = 0.0,
    [double]$BaseConfWeightPerViewQuantile = 0.65,
    [int]$BaseConfWeightPerViewMinValid = 16,
    [double]$BaseLambdaPointNormalConsis = 0.05,
    [string]$Stage2GramDynEnable = "off",
    [int]$Stage2GramDynLayerIdx = -1,
    [double]$Stage2GramDynQuantile = 0.30,
    [double]$Stage2GramDynWeightFloor = 0.25,
    [int]$Stage2GramDynWarmupSteps = 40,
    [string]$Stage2DynProxyEnable = "off",
    [string]$Stage2DynProxyMode = "fg_static_soft",
    [string]$Stage2DynProxyUseGram = "on",
    [string]$Stage2DynProxyUseSupport = "on",
    [double]$Stage2DynProxyFloor = 0.35,
    [int]$Stage2DynProxyWarmupSteps = 40,
    [object]$Stage2EnableAnySplatAblationSixPack = $false,
    [object]$Stage2EnableExtendedCkptWaitOnMissing = $true,
    [int]$Stage2CkptExtendedWaitTimeoutSec = 1200,
    [object]$Stage2EnableResumeCkptFallbackOnShortCkptMissing = $false,
    [object]$Stage2DisallowResumeFallbackResult = $true,
    [int]$BasePointNormalConsisWarmupSteps = 40,
    [int]$BasePointLossFgErodePx = 1,
    [int]$BasePointMvDepthFgErodePx = 0,
    [double]$BasePointConsQuantile = 0.5,
    [string]$BasePointConsFocus = "inlier",
    [double]$BasePointResidualQuantile = 1.0,
    [string]$BasePointResidualFocus = "inlier",
    [double]$BasePointResidualBoost = 1.5,
    [double]$BasePointMvDepthOutlierBoost = 1.5,
    [string]$BaseSupervisionWeightMode = "mix",
    [string]$DepthAnchorPointTargetMode = "depth_unproject",
    [string]$DepthAnchorLambdaPointMvDepthList = "0.001,0.0015",
    [string]$DepthAnchorLambdaPointMvMaskList = "0.002,0.004,0.008",
    [double]$DepthAnchorLambdaPoint = 0.35,
    [double]$DepthAnchorLambdaPointReproj = 0.05,
    [double]$DepthAnchorPointTargetConsensusAlphaFloor = 0.75,
    [double]$DepthAnchorPointMvMaskHitThr = 0.65,
    [double]$DepthAnchorPointMvMaskMinTgtFgRatio = 0.03,
    [int]$DepthAnchorPointMvMaskSoftBlurPx = 2,
    [int]$DepthAnchorPointMvMaskSoftBlurIters = 2,
    [double]$DepthAnchorPointMvMaskSoftMix = 0.7,
    [double]$DepthAnchorPointMvMaskSoftHitThr = 0.6,
    [int]$DepthAnchorPointMvStride = 1,
    [int]$DepthAnchorPointMvDepthMaxPairs = 4,
    [string]$DepthAnchorPointMvDepthSupportMode = "direct",
    [double]$DepthAnchorPointMvDepthSupportFloor = 0.1,
    [int]$DepthAnchorPointLossFgErodePx = 2,
    [int]$DepthAnchorPointMvDepthFgErodePx = 1,
    [double]$DepthAnchorPointConsQuantile = 0.7,
    [string]$DepthAnchorPointConsFocus = "outlier",
    [double]$DepthAnchorPointResidualQuantile = 0.7,
    [string]$DepthAnchorPointResidualFocus = "outlier",
    [double]$DepthAnchorPointResidualBoost = 2.5,
    [double]$DepthAnchorPointMvDepthOutlierBoost = 2.0,
    [string]$DepthAnchorSupervisionWeightMode = "uniform",
    [int]$StopAfterHours = 12,
    [string]$FinalDeadline = "",
    [int]$MaxCycles = 999,
    [switch]$ForceStage2Only,
    [int]$NoImproveCyclesPatience = 999,
    [double]$CycleMinGhostImprove = 0.0,
    [double]$StageResumePromoteGhostMargin = 0.15,
    [string]$MentorUpdatePath = "logs/modal_phase5/mentor_update_latest.md",
    [double]$SubstantialGhostImprove = 0.02,
    [int]$StagnationStopCycles = 6,
    [double]$CycleRegressGhostThreshold = 0.03,
    [double]$CycleRegressPSNRDropThreshold = 0.08,
    [double]$CycleRegressWl1RiseThreshold = 0.003,
    [double]$CyclePromoteMaxPSNRDrop = 0.12,
    [double]$CyclePromoteMaxSSIMDrop = 0.006,
    [double]$CyclePromoteMaxWl1Rise = 0.0015,
    [double]$CyclePromoteRelaxedMinGhostGain = 0.10,
    [double]$CyclePromoteRelaxedMaxPSNRDrop = 0.20,
    [double]$CyclePromoteRelaxedMaxSSIMDrop = 0.012,
    [double]$CyclePromoteRelaxedMaxWl1Rise = 0.0025,
    [int]$InfraNoOutputStageAbortThreshold = 2,
    [bool]$EnableHistoricalSweepBootstrap = $false,
    [double]$HistoryBootstrapRelaxedMinGhostGain = 0.10,
    [double]$HistoryBootstrapRelaxedMaxPSNRDrop = 0.12,
    [double]$HistoryBootstrapRelaxedMaxSSIMDrop = 0.006,
    [double]$HistoryBootstrapRelaxedMaxWl1Rise = 0.0015,
    [double]$SkipDeepStagesGhostLag = 0.2,
    [int]$SkipDeepStagesAfterNoSubstantialCycles = 2,
    [double]$SkipStage5GhostLagVsGlobal = 0.12,
    [int]$RegressionStopCycles = 4,
    [bool]$EnableABRouteOnStagnation = $true,
    [int]$NoProgressCyclesForABRoute = 4,
    [double]$StableGhostTarget = 4.80,
    [string]$ABBalanceDepthList = "0.0005",
    [string]$ABBalanceMaskList = "0",
    [string]$ABAggressiveDepthList = "0.001",
    [string]$ABAggressiveMaskList = "0.001",
    [bool]$EnablePersistentCycleState = $true,
    [string]$PersistentCycleStatePath = "logs/modal_phase5/ghost_autoloop_runtime_state_latest.json",
    [int]$PersistentCycleStateMaxAgeHours = 36
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot "..")
Set-Location $repoRoot

function To-DoubleOrNaN($x) {
    try {
        if ($null -eq $x -or [string]::IsNullOrWhiteSpace([string]$x)) { return [double]::NaN }
        return [double]$x
    } catch {
        return [double]::NaN
    }
}

function To-BoolLoose(
    [object]$Value,
    [bool]$Default = $false
) {
    if ($null -eq $Value) { return $Default }
    if ($Value -is [bool]) { return [bool]$Value }
    if ($Value -is [int] -or $Value -is [long] -or $Value -is [double] -or $Value -is [decimal]) {
        return ([double]$Value -ne 0.0)
    }
    $s = [string]$Value
    if ([string]::IsNullOrWhiteSpace($s)) { return $Default }
    $s = $s.Trim().ToLowerInvariant()
    if ($s.StartsWith('$')) {
        # Accept PowerShell-style bool literals coming from launcher/hot-update args.
        $s = $s.TrimStart('$')
    }
    if ($s.StartsWith('"') -and $s.EndsWith('"') -and $s.Length -ge 2) {
        $s = $s.Substring(1, $s.Length - 2).Trim().ToLowerInvariant()
    }
    if ($s -match '^(1|true|yes|y|on)$') { return $true }
    if ($s -match '^(0|false|no|n|off)$') { return $false }
    try { return ([double]$s -ne 0.0) } catch { return $Default }
}

$Stage2DualLaneEnabled = To-BoolLoose -Value $Stage2DualLaneEnabled -Default $true
$PostRescueEnabled = To-BoolLoose -Value $PostRescueEnabled -Default $true
$Stage2HistoryQualityAware = To-BoolLoose -Value $Stage2HistoryQualityAware -Default $true
$StageEnableAbsoluteQualityGuard = To-BoolLoose -Value $StageEnableAbsoluteQualityGuard -Default $true
$EnableVisualAntiBlackGuard = To-BoolLoose -Value $EnableVisualAntiBlackGuard -Default $true
$Stage2EnableAnySplatAblationSixPack = To-BoolLoose -Value $Stage2EnableAnySplatAblationSixPack -Default $false
$Stage2EnableExtendedCkptWaitOnMissing = To-BoolLoose -Value $Stage2EnableExtendedCkptWaitOnMissing -Default $true
$Stage2EnableResumeCkptFallbackOnShortCkptMissing = To-BoolLoose -Value $Stage2EnableResumeCkptFallbackOnShortCkptMissing -Default $false
$Stage2DisallowResumeFallbackResult = To-BoolLoose -Value $Stage2DisallowResumeFallbackResult -Default $true
$EmergencyGhostShockEnabled = To-BoolLoose -Value $EmergencyGhostShockEnabled -Default $false
$EmergencyRecoveryEnableHistoryQualityAware = To-BoolLoose -Value $EmergencyRecoveryEnableHistoryQualityAware -Default $true

function Test-SweepRowEligibleForBest([object]$Row) {
    if ($null -eq $Row) { return $false }
    $exitCode = -1
    try { $exitCode = [int]$Row.exit_code } catch { $exitCode = -1 }
    if ($exitCode -ne 0) { return $false }
    if (To-BoolLoose -Value $Row.visual_guard_blocked -Default $false) { return $false }
    if ($Row.PSObject.Properties["quality_guard_blocked"]) {
        if (To-BoolLoose -Value $Row.quality_guard_blocked -Default $false) { return $false }
    }
    if (To-BoolLoose -Value $Row.eval_num_src_views_mismatch -Default $false) { return $false }
    return $true
}

function Test-AutoloopRowEligibleForBootstrap([object]$Row) {
    if ($null -eq $Row) { return $false }
    $g = To-DoubleOrNaN($Row.ghost)
    if ([double]::IsNaN($g)) { return $false }
    $resumeReason = [string]$Row.resume_update_reason
    if (($resumeReason -eq "interim_watch_sync") -or ($resumeReason -eq "watch_sync")) { return $false }
    $guardTier = ([string]$Row.guard_tier).Trim().ToLowerInvariant()
    if ($guardTier -eq "blocked") { return $false }
    if (To-BoolLoose -Value $Row.cycle_quality_guard_blocked -Default $false) { return $false }
    if ($Row.PSObject.Properties["visual_guard_blocked"]) {
        if (To-BoolLoose -Value $Row.visual_guard_blocked -Default $false) { return $false }
    }
    if ($Row.PSObject.Properties["best_visual_guard_blocked"]) {
        if (To-BoolLoose -Value $Row.best_visual_guard_blocked -Default $false) { return $false }
    }
    if ($Row.PSObject.Properties["quality_guard_blocked"]) {
        if (To-BoolLoose -Value $Row.quality_guard_blocked -Default $false) { return $false }
    }
    if ($Row.PSObject.Properties["best_quality_guard_blocked"]) {
        if (To-BoolLoose -Value $Row.best_quality_guard_blocked -Default $false) { return $false }
    }
    $ckpt = [string]$Row.best_ckpt
    if ([string]::IsNullOrWhiteSpace($ckpt)) { return $false }
    if ($Row.PSObject.Properties["best_eval_num_src_views_mismatch"]) {
        if (To-BoolLoose -Value $Row.best_eval_num_src_views_mismatch -Default $false) { return $false }
    }
    if ($Row.PSObject.Properties["eval_num_src_views_mismatch"]) {
        if (To-BoolLoose -Value $Row.eval_num_src_views_mismatch -Default $false) { return $false }
    }
    return $true
}

function Get-P0Stage2Stats([int]$WindowMinutes = 90) {
    $cutoff = (Get-Date).AddMinutes(-1 * [Math]::Max(1, $WindowMinutes))
    $files = @(
        Get-ChildItem "logs/modal_phase5" -Filter "ghost_mvdepth_sweep_cycle*_stage2_*.csv" -File -ErrorAction SilentlyContinue |
            Where-Object { $_.LastWriteTime -ge $cutoff }
    )
    $validRows = 0
    $latestUpdatedAt = ""
    $latestUpdatedAtDt = [datetime]::MinValue
    foreach ($f in $files) {
        if (($latestUpdatedAtDt -eq [datetime]::MinValue) -or ($f.LastWriteTime -gt $latestUpdatedAtDt)) {
            $latestUpdatedAtDt = $f.LastWriteTime
            $latestUpdatedAt = $f.LastWriteTime.ToString("yyyy-MM-ddTHH:mm:ss")
        }
        try {
            $rows = @(Import-Csv $f.FullName)
        } catch {
            continue
        }
        foreach ($row in $rows) {
            if (-not (Test-SweepRowEligibleForBest $row)) { continue }
            $ghost = To-DoubleOrNaN($row.ghost_score_mean)
            $ghostVisual = To-DoubleOrNaN($row.ghost_visual_score)
            $predLuma = To-DoubleOrNaN($row.pred_luma_mean)
            $predNonBlack = To-DoubleOrNaN($row.pred_nonblack_ratio_thr008)
            if ([double]::IsNaN($ghost) -or [double]::IsNaN($ghostVisual) -or [double]::IsNaN($predLuma) -or [double]::IsNaN($predNonBlack)) {
                continue
            }
            $validRows += 1
        }
    }
    return [pscustomobject]@{
        window_minutes = [int]$WindowMinutes
        valid_rows = [int]$validRows
        pass = ([int]$validRows -ge 3)
        latest_updated_at = $latestUpdatedAt
        reason = $(if ([int]$validRows -ge 3) { "pass" } else { "need_valid_stage2_rows>=3" })
    }
}

function Parse-BestSweepRow([string]$CsvPath) {
    if (-not (Test-Path $CsvPath)) { return $null }
    $rows = @(
        Import-Csv $CsvPath |
            Where-Object { Test-SweepRowEligibleForBest $_ }
    )
    if ($rows.Count -le 0) { return $null }

    $withVisualGhost = @(
        $rows |
            Where-Object { -not [double]::IsNaN((To-DoubleOrNaN($_.ghost_visual_score))) } |
            Sort-Object {
                To-DoubleOrNaN($_.ghost_visual_score)
            }, {
                To-DoubleOrNaN($_.ghost_score_mean)
            }, {
                -1.0 * (To-DoubleOrNaN($_.mean_PSNR))
            }
    )
    if ($withVisualGhost.Count -gt 0) { return $withVisualGhost[0] }

    $withGhost = @(
        $rows |
            Where-Object { -not [double]::IsNaN((To-DoubleOrNaN($_.ghost_score_mean))) } |
            Sort-Object {
                To-DoubleOrNaN($_.ghost_score_mean)
            }, {
                -1.0 * (To-DoubleOrNaN($_.mean_PSNR))
            }
    )
    if ($withGhost.Count -gt 0) { return $withGhost[0] }

    $fallback = @(
        $rows |
            Sort-Object {
                -1.0 * (To-DoubleOrNaN($_.mean_PSNR))
            }, {
                To-DoubleOrNaN($_.mean_weighted_L1)
            }
    )
    if ($fallback.Count -gt 0) { return $fallback[0] }
    return $null
}

function Get-LatestCsvRow([string]$CsvPath) {
    if ([string]::IsNullOrWhiteSpace($CsvPath) -or (-not (Test-Path $CsvPath))) { return $null }
    try {
        $rows = @(Import-Csv $CsvPath)
        if ($rows.Count -le 0) { return $null }
        return $rows[$rows.Count - 1]
    } catch {
        return $null
    }
}

function Resolve-BestCkpt([string]$SweepCsvPath, [string]$BestLabel) {
    if (-not (Test-Path $SweepCsvPath)) { return $null }
    $rows = @(
        Import-Csv $SweepCsvPath |
            Where-Object { $_.status -eq "ok" -and $_.stage -eq "short" }
    )
    if ($rows.Count -le 0) { return $null }
    $hit = $null
    if (-not [string]::IsNullOrWhiteSpace($BestLabel)) {
        $hit = @($rows | Where-Object { [string]$_.label -eq $BestLabel } | Select-Object -First 1)
        if ($hit.Count -gt 0) { return $hit[0] }
    }
    return $rows[0]
}

function Resolve-ExistingGlobalBestBootstrap([string]$CsvPath) {
    $candidate = $null
    if (Test-Path $CsvPath) {
        try {
            $rows = @(
                Import-Csv $CsvPath |
                    Where-Object { Test-AutoloopRowEligibleForBootstrap $_ }
            )
            if ($rows.Count -gt 0) {
                $best = @(
                    $rows |
                        Sort-Object {
                            To-DoubleOrNaN($_.ghost_visual_score)
                        }, {
                            To-DoubleOrNaN($_.ghost)
                        }, {
                            -1.0 * (To-DoubleOrNaN($_.psnr))
                        }
                ) | Select-Object -First 1
                if ($null -ne $best) {
                    $bestVisual = [string]$best.best_visual_png
                    if ([string]::IsNullOrWhiteSpace($bestVisual)) {
                        $bestVisual = [string]$best.stage_best_strip_png
                    }
                    $candidate = [pscustomobject]@{
                        ghost = To-DoubleOrNaN($best.ghost)
                        psnr = To-DoubleOrNaN($best.psnr)
                        ssim = To-DoubleOrNaN($best.ssim)
                        wl1 = To-DoubleOrNaN($best.wl1)
                        best_ckpt = [string]$best.best_ckpt
                        best_geom = [string]$best.best_geom
                        best_lambda_point_mv_depth = [string]$best.best_lambda_point_mv_depth
                        best_lambda_point_mv_mask = [string]$best.best_lambda_point_mv_mask
                        best_ghost_rows_csv = [string]$best.best_ghost_rows_csv
                        best_visual_png = $bestVisual
                        best_stage = [string]$best.stage
                        updated_at = [string]$best.updated_at
                    }
                }
            }
        } catch {
        }
    }

    $persistPath = "logs/modal_phase5/ghost_global_best_latest.json"
    if (Test-Path $persistPath) {
        try {
            $persist = Get-Content $persistPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $persistSource = [string]$persist.source
            $persistGhost = To-DoubleOrNaN($persist.global_best_ghost)
            $persistEvalMismatch = $false
            if ($persist.PSObject.Properties["best_eval_num_src_views_mismatch"]) {
                $persistEvalMismatch = To-BoolLoose -Value $persist.best_eval_num_src_views_mismatch -Default $false
            }
            if ($persistSource -eq "watch_interim_sync_anyrow") {
                Write-Host "[autoloop][warn] ignore persisted global best from interim watcher source=$persistSource"
                $persistGhost = [double]::NaN
            }
            if ($persistEvalMismatch) {
                Write-Host "[autoloop][warn] ignore persisted global best due to eval_num_src_views mismatch."
                $persistGhost = [double]::NaN
            }
            if (-not [double]::IsNaN($persistGhost)) {
                $preferPersist = $false
                if ($null -eq $candidate) {
                    $preferPersist = $true
                } else {
                    $candGhost = To-DoubleOrNaN($candidate.ghost)
                    if ([double]::IsNaN($candGhost) -or ($persistGhost -lt $candGhost)) {
                        $preferPersist = $true
                    }
                }
                if ($preferPersist) {
                    $candidate = [pscustomobject]@{
                        ghost = $persistGhost
                        psnr = To-DoubleOrNaN($persist.global_best_psnr)
                        ssim = To-DoubleOrNaN($persist.global_best_ssim)
                        wl1 = To-DoubleOrNaN($persist.global_best_wl1)
                        best_ckpt = [string]$persist.best_ckpt
                        best_geom = [string]$persist.global_best_geom
                        best_lambda_point_mv_depth = [string]$persist.global_best_lambda_point_mv_depth
                        best_lambda_point_mv_mask = [string]$persist.global_best_lambda_point_mv_mask
                        best_ghost_rows_csv = [string]$persist.best_ghost_rows_csv
                        best_visual_png = [string]$persist.best_visual_png
                        best_stage = [string]$persist.global_best_stage
                        updated_at = [string]$persist.updated_at
                    }
                }
            }
        } catch {
        }
    }
    return $candidate
}

function Resolve-HistoricalSweepBootstrapCandidate(
    [object]$BaselineCandidate,
    [double]$StrictMaxPSNRDrop,
    [double]$StrictMaxSSIMDrop,
    [double]$StrictMaxWl1Rise,
    [double]$RelaxedMinGhostGain,
    [double]$RelaxedMaxPSNRDrop,
    [double]$RelaxedMaxSSIMDrop,
    [double]$RelaxedMaxWl1Rise
) {
    $files = @(Get-ChildItem "logs/modal_phase5" -Filter "ghost_mvdepth_sweep_cycle*_stage*_*.csv" -File -ErrorAction SilentlyContinue)
    if ($files.Count -le 0) { return $null }

    $baseGhost = [double]::NaN
    $basePsnr = [double]::NaN
    $baseSsim = [double]::NaN
    $baseWl1 = [double]::NaN
    if ($null -ne $BaselineCandidate) {
        $baseGhost = To-DoubleOrNaN($BaselineCandidate.ghost)
        $basePsnr = To-DoubleOrNaN($BaselineCandidate.psnr)
        $baseSsim = To-DoubleOrNaN($BaselineCandidate.ssim)
        $baseWl1 = To-DoubleOrNaN($BaselineCandidate.wl1)
    }

    $best = $null
    foreach ($f in $files) {
        $stageName = ""
        $m = [regex]::Match([string]$f.Name, "ghost_mvdepth_sweep_cycle\d+_(stage\d+_[^_]+(?:_[^_]+)?)_")
        if ($m.Success) {
            $stageName = [string]$m.Groups[1].Value
        }
        try {
            $rows = @(
                Import-Csv $f.FullName |
                    Where-Object {
                        (Test-SweepRowEligibleForBest $_) -and
                        (-not [double]::IsNaN((To-DoubleOrNaN($_.ghost_score_mean))))
                    }
            )
            foreach ($r in $rows) {
                $g = To-DoubleOrNaN($r.ghost_score_mean)
                $gv = To-DoubleOrNaN($r.ghost_visual_score)
                $p = To-DoubleOrNaN($r.mean_PSNR)
                $s = To-DoubleOrNaN($r.mean_SSIM)
                $w = To-DoubleOrNaN($r.mean_weighted_L1)
                if ([double]::IsNaN($g)) { continue }

                $sweepCsv = [string]$r.sweep_csv
                $bestLabel = [string]$r.best_label
                if ([string]::IsNullOrWhiteSpace($sweepCsv)) { continue }
                $ckptInfo = Resolve-BestCkpt -SweepCsvPath $sweepCsv -BestLabel $bestLabel
                if ($null -eq $ckptInfo) { continue }
                $candCkpt = [string]$ckptInfo.ft_ckpt
                if ([string]::IsNullOrWhiteSpace($candCkpt)) { continue }

                $guardMode = "no_guard"
                if ((-not [double]::IsNaN($baseGhost)) -and ($g -ge $baseGhost)) {
                    continue
                }
                if ((-not [double]::IsNaN($baseGhost)) -and ($g -lt $baseGhost)) {
                    $ghostGain = $baseGhost - $g
                    $maxP = [Math]::Max(0.0, $StrictMaxPSNRDrop)
                    $maxS = [Math]::Max(0.0, $StrictMaxSSIMDrop)
                    $maxW = [Math]::Max(0.0, $StrictMaxWl1Rise)
                    $guardMode = "strict"
                    if ($ghostGain -ge [Math]::Max(0.0, $RelaxedMinGhostGain)) {
                        $guardMode = "relaxed"
                        $maxP = [Math]::Max($maxP, [Math]::Max(0.0, $RelaxedMaxPSNRDrop))
                        $maxS = [Math]::Max($maxS, [Math]::Max(0.0, $RelaxedMaxSSIMDrop))
                        $maxW = [Math]::Max($maxW, [Math]::Max(0.0, $RelaxedMaxWl1Rise))
                    }
                    if ((-not [double]::IsNaN($basePsnr)) -and (-not [double]::IsNaN($p))) {
                        $psnrDrop = $basePsnr - $p
                        if ($psnrDrop -gt $maxP) { continue }
                    }
                    if ((-not [double]::IsNaN($baseSsim)) -and (-not [double]::IsNaN($s))) {
                        $ssimDrop = $baseSsim - $s
                        if ($ssimDrop -gt $maxS) { continue }
                    }
                    if ((-not [double]::IsNaN($baseWl1)) -and (-not [double]::IsNaN($w))) {
                        $wl1Rise = $w - $baseWl1
                        if ($wl1Rise -gt $maxW) { continue }
                    }
                }

                $candGeom = [string]$r.best_geom_subdir
                if ([string]::IsNullOrWhiteSpace($candGeom)) {
                    $candGeom = [string]$ckptInfo.geom_subdir
                }
                $rowsCsv = Resolve-GhostRowsCsv -BestRow $r
                $bestVisual = ""
                if (-not [string]::IsNullOrWhiteSpace($rowsCsv)) {
                    $stats = Get-GhostRowsStats -GhostRowsCsv $rowsCsv
                    if ($stats -ne $null) {
                        $bestVisual = [string]$stats.first_path
                    }
                }

                $cand = [pscustomobject]@{
                    ghost_visual_score = $gv
                    ghost = $g
                    psnr = $p
                    ssim = $s
                    wl1 = $w
                    best_ckpt = $candCkpt
                    best_geom = $candGeom
                    best_lambda_point_mv_depth = [string]$r.lambda_point_mv_depth
                    best_lambda_point_mv_mask = [string]$r.lambda_point_mv_mask
                    best_ghost_rows_csv = $rowsCsv
                    best_visual_png = $bestVisual
                    best_stage = $stageName
                    updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
                    bootstrap_source = [string]$f.Name
                    bootstrap_guard_mode = $guardMode
                }

                $preferCand = $false
                if ($best -eq $null) {
                    $preferCand = $true
                } else {
                    $bestVisual = To-DoubleOrNaN($best.ghost_visual_score)
                    if ((-not [double]::IsNaN($gv)) -and (([double]::IsNaN($bestVisual)) -or ($gv -lt $bestVisual))) {
                        $preferCand = $true
                    } elseif ((([double]::IsNaN($gv) -and [double]::IsNaN($bestVisual)) -or ((-not [double]::IsNaN($gv)) -and ($gv -eq $bestVisual)))) {
                        if (($cand.ghost -lt $best.ghost) -or (($cand.ghost -eq $best.ghost) -and ($cand.psnr -gt $best.psnr))) {
                            $preferCand = $true
                        }
                    }
                }
                if ($preferCand) {
                    $best = $cand
                }
            }
        } catch {
        }
    }
    return $best
}

function Normalize-JsonFiniteNumbers($Value) {
    if ($null -eq $Value) { return $null }
    if (($Value -is [double]) -or ($Value -is [float])) {
        $d = [double]$Value
        if ([double]::IsNaN($d) -or [double]::IsInfinity($d)) { return $null }
        return $Value
    }
    if ($Value -is [System.Collections.IDictionary]) {
        $o = [ordered]@{}
        foreach ($k in $Value.Keys) {
            $o[[string]$k] = Normalize-JsonFiniteNumbers $Value[$k]
        }
        return [pscustomobject]$o
    }
    if (($Value -is [System.Collections.IEnumerable]) -and (-not ($Value -is [string]))) {
        $arr = New-Object System.Collections.ArrayList
        foreach ($item in $Value) {
            [void]$arr.Add((Normalize-JsonFiniteNumbers $item))
        }
        return @($arr)
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $o = [ordered]@{}
        foreach ($p in $Value.PSObject.Properties) {
            $o[$p.Name] = Normalize-JsonFiniteNumbers $p.Value
        }
        return [pscustomobject]$o
    }
    return $Value
}

function Build-FocusedLambdaList(
    [string]$PrimaryList,
    [string]$FallbackList,
    [string]$Preferred
) {
    $ordered = New-Object System.Collections.ArrayList
    foreach ($src in @($PrimaryList, $FallbackList)) {
        foreach ($tok in ([string]$src -split ",")) {
            $t = [string]$tok.Trim()
            if ([string]::IsNullOrWhiteSpace($t)) { continue }
            if (-not ($ordered -contains $t)) {
                [void]$ordered.Add($t)
            }
        }
    }
    $pref = [string]$Preferred
    if (-not [string]::IsNullOrWhiteSpace($pref)) {
        $pref = $pref.Trim()
        [void]$ordered.Remove($pref)
        $ordered.Insert(0, $pref)
    }
    return ((@($ordered)) -join ",")
}

function Parse-LambdaList([string]$List) {
    $ordered = New-Object System.Collections.ArrayList
    foreach ($tok in ([string]$List -split ",")) {
        $t = [string]$tok.Trim()
        if ([string]::IsNullOrWhiteSpace($t)) { continue }
        if (-not ($ordered -contains $t)) {
            [void]$ordered.Add($t)
        }
    }
    return @($ordered)
}

function Limit-LambdaList(
    [string]$List,
    [int]$MaxCount
) {
    $vals = @(Parse-LambdaList -List $List)
    if (($MaxCount -le 0) -or ($vals.Count -le $MaxCount)) {
        return ($vals -join ",")
    }
    return ((@($vals | Select-Object -First $MaxCount)) -join ",")
}

function Parse-GenericTokens([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) { return @() }
    return @(
        $Raw -split "[,\s;|]+" |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_.Trim() }
    )
}

function Filter-LambdaListNumericRange(
    [string]$List,
    [double]$MinValue = [double]::NegativeInfinity,
    [double]$MaxValue = [double]::PositiveInfinity,
    [string]$FallbackList = ""
) {
    $pick = New-Object System.Collections.ArrayList
    foreach ($v in @(Parse-LambdaList -List $List)) {
        $d = [double]::NaN
        if (-not [double]::TryParse([string]$v, [ref]$d)) { continue }
        if (($d -lt $MinValue) -or ($d -gt $MaxValue)) { continue }
        if (-not ($pick -contains $v)) { [void]$pick.Add($v) }
    }
    if ($pick.Count -eq 0 -and (-not [string]::IsNullOrWhiteSpace($FallbackList))) {
        foreach ($v in @(Parse-LambdaList -List $FallbackList)) {
            $d = [double]::NaN
            if (-not [double]::TryParse([string]$v, [ref]$d)) { continue }
            if (($d -lt $MinValue) -or ($d -gt $MaxValue)) { continue }
            if (-not ($pick -contains $v)) { [void]$pick.Add($v) }
        }
    }
    if ($pick.Count -gt 0) { return ((@($pick)) -join ",") }
    return $List
}

function Resolve-StageHistoryFocus(
    [string]$StageToken,
    [int]$TopRows = 12,
    [double]$ReferencePSNR = [double]::NaN,
    [double]$ReferenceSSIM = [double]::NaN,
    [double]$ReferenceWl1 = [double]::NaN,
    [double]$MaxPSNRDrop = [double]::PositiveInfinity,
    [double]$MaxSSIMDrop = [double]::PositiveInfinity,
    [double]$MaxWl1Rise = [double]::PositiveInfinity,
    [bool]$EnableQualityGuard = $false
) {
    if ([string]::IsNullOrWhiteSpace($StageToken)) { return $null }
    $pattern = "ghost_mvdepth_sweep_cycle*_${StageToken}_*.csv"
    $files = @(Get-ChildItem "logs/modal_phase5" -Filter $pattern -File -ErrorAction SilentlyContinue)
    if ($files.Count -le 0) { return $null }

    $cands = New-Object System.Collections.Generic.List[object]
    foreach ($f in $files) {
        try {
            $rows = @(
                Import-Csv $f.FullName |
                    Where-Object {
                        (Test-SweepRowEligibleForBest $_) -and
                        (-not [string]::IsNullOrWhiteSpace([string]$_.lambda_point_mv_depth)) -and
                        (-not [string]::IsNullOrWhiteSpace([string]$_.lambda_point_mv_mask)) -and
                        (-not [double]::IsNaN((To-DoubleOrNaN($_.ghost_score_mean))))
                    }
            )
            foreach ($r in $rows) {
                $cands.Add([pscustomobject]@{
                    depth = [string]$r.lambda_point_mv_depth
                    mask = [string]$r.lambda_point_mv_mask
                    ghost_visual = To-DoubleOrNaN($r.ghost_visual_score)
                    ghost = To-DoubleOrNaN($r.ghost_score_mean)
                    psnr = To-DoubleOrNaN($r.mean_PSNR)
                    ssim = To-DoubleOrNaN($r.mean_SSIM)
                    wl1 = To-DoubleOrNaN($r.mean_weighted_L1)
                    file = $f.Name
                }) | Out-Null
            }
        } catch {
        }
    }
    if ($cands.Count -le 0) { return $null }

    $orderedPool = @($cands.ToArray())
    if ($EnableQualityGuard) {
        $guarded = @(
            $orderedPool |
                Where-Object {
                    $ok = $true
                    $candPsnr = To-DoubleOrNaN($_.psnr)
                    $candSsim = To-DoubleOrNaN($_.ssim)
                    $candWl1 = To-DoubleOrNaN($_.wl1)
                    if ((-not [double]::IsNaN($ReferencePSNR)) -and (-not [double]::IsNaN($candPsnr))) {
                        $ok = $ok -and (($ReferencePSNR - $candPsnr) -le [Math]::Max(0.0, $MaxPSNRDrop))
                    }
                    if ((-not [double]::IsNaN($ReferenceSSIM)) -and (-not [double]::IsNaN($candSsim))) {
                        $ok = $ok -and (($ReferenceSSIM - $candSsim) -le [Math]::Max(0.0, $MaxSSIMDrop))
                    }
                    if ((-not [double]::IsNaN($ReferenceWl1)) -and (-not [double]::IsNaN($candWl1))) {
                        $ok = $ok -and (($candWl1 - $ReferenceWl1) -le [Math]::Max(0.0, $MaxWl1Rise))
                    }
                    return $ok
                }
        )
        if ($guarded.Count -gt 0) {
            $orderedPool = $guarded
            Write-Host "[autoloop] stage history focus quality guard accepted stage=$StageToken rows=$($guarded.Count)/$($cands.Count) (psnr_drop<=$MaxPSNRDrop ssim_drop<=$MaxSSIMDrop wl1_rise<=$MaxWl1Rise)"
        } else {
            Write-Host "[autoloop] stage history focus quality guard fallback stage=$StageToken rows=0/$($cands.Count); keep visual-safe pool ordering."
        }
    }

    $ordered = @(
        $orderedPool |
            Sort-Object {
                To-DoubleOrNaN($_.ghost_visual)
            }, {
                To-DoubleOrNaN($_.ghost)
            }, {
                -1.0 * (To-DoubleOrNaN($_.psnr))
            } |
            Select-Object -First ([Math]::Max(1, $TopRows))
    )
    if ($ordered.Count -le 0) { return $null }

    $depths = New-Object System.Collections.ArrayList
    $masks = New-Object System.Collections.ArrayList
    foreach ($r in $ordered) {
        $d = [string]$r.depth
        $m = [string]$r.mask
        if ((-not [string]::IsNullOrWhiteSpace($d)) -and (-not ($depths -contains $d))) {
            [void]$depths.Add($d)
        }
        if ((-not [string]::IsNullOrWhiteSpace($m)) -and (-not ($masks -contains $m))) {
            [void]$masks.Add($m)
        }
    }
    if (($depths.Count -le 0) -or ($masks.Count -le 0)) { return $null }

    $top = $ordered[0]
    return [pscustomobject]@{
        depth_list = ((@($depths)) -join ",")
        mask_list = ((@($masks)) -join ",")
        top_depth = [string]$top.depth
        top_mask = [string]$top.mask
        top_ghost = To-DoubleOrNaN($top.ghost)
        sample_count = $cands.Count
        used_rows = $ordered.Count
    }
}

function Resolve-Stage1HistoryFocus([int]$TopRows = 12) {
    return Resolve-StageHistoryFocus -StageToken "stage1_strong" -TopRows $TopRows
}

function Resolve-StageFamily([string]$StageName) {
    if ([string]::IsNullOrWhiteSpace($StageName)) { return "unknown" }
    $s = ([string]$StageName).ToLowerInvariant()
    if ($s -match "stage1") { return "stage1" }
    if ($s -match "stage2") { return "stage2" }
    if ($s -match "stage3") { return "stage3" }
    if ($s -match "stage4") { return "stage4" }
    if ($s -match "stage5") { return "stage5" }
    return "other"
}

function Resolve-HintScope(
    [string]$BestStage,
    [bool]$IsHistoricalBootstrap
) {
    $family = Resolve-StageFamily -StageName $BestStage
    $applyStage1 = $true
    $applyStage2 = $true

    if ($IsHistoricalBootstrap) {
        if ($family -eq "stage1") {
            $applyStage1 = $true
            $applyStage2 = $false
        } elseif ($family -eq "stage2") {
            $applyStage1 = $false
            # Historical stage2 hints can drift toward stale/local minima; keep stage2 on its own history focus.
            $applyStage2 = $false
        }
    } else {
        if ($family -eq "stage1") {
            $applyStage1 = $true
            $applyStage2 = $false
        } elseif ($family -eq "stage2") {
            $applyStage1 = $false
            $applyStage2 = $true
        }
    }

    return [pscustomobject]@{
        stage_family = $family
        apply_stage1 = $applyStage1
        apply_stage2 = $applyStage2
    }
}

function Test-Stage2HasPotential(
    [object]$Stage2,
    [object]$Stage1,
    [double]$GlobalBestGhostRef,
    [double]$GhostLagThreshold,
    [double]$VsStage1ImproveMin
) {
    if ($null -eq $Stage2) { return $false }
    if (-not (Test-StageHasUsableGhost -Stage $Stage2)) { return $false }
    $s2Ghost = To-DoubleOrNaN($Stage2.ghost)
    if ([double]::IsNaN($s2Ghost)) { return $false }

    $lagThr = [Math]::Max(0.0, [double]$GhostLagThreshold)
    if ((-not [double]::IsInfinity($GlobalBestGhostRef)) -and (-not [double]::IsNaN($GlobalBestGhostRef))) {
        if ($s2Ghost -le ($GlobalBestGhostRef + $lagThr)) { return $true }
    } else {
        return $true
    }

    if ($null -ne $Stage1) {
        $s1Ghost = To-DoubleOrNaN($Stage1.ghost)
        if (-not [double]::IsNaN($s1Ghost)) {
            $improveMin = [Math]::Max(0.0, [double]$VsStage1ImproveMin)
            if ($s2Ghost -le ($s1Ghost - $improveMin)) { return $true }
        }
    }
    return $false
}

function Resolve-Stage2PotentialQualityCheck(
    [object]$Stage2,
    [double]$GlobalBestPsnrRef,
    [double]$GlobalBestSsimRef,
    [double]$GlobalBestWl1Ref,
    [double]$MaxPSNRDrop,
    [double]$MaxSSIMDrop,
    [double]$MaxWl1Rise
) {
    if ($null -eq $Stage2) {
        return [pscustomobject]@{ pass = $false; reason = "stage2_null" }
    }
    $reasons = New-Object System.Collections.Generic.List[string]
    $maxP = [Math]::Max(0.0, [double]$MaxPSNRDrop)
    $maxS = [Math]::Max(0.0, [double]$MaxSSIMDrop)
    $maxW = [Math]::Max(0.0, [double]$MaxWl1Rise)

    $s2P = To-DoubleOrNaN($Stage2.psnr)
    $s2S = To-DoubleOrNaN($Stage2.ssim)
    $s2W = To-DoubleOrNaN($Stage2.wl1)

    if ((-not [double]::IsInfinity($GlobalBestPsnrRef)) -and (-not [double]::IsNaN($GlobalBestPsnrRef)) -and (-not [double]::IsNaN($s2P))) {
        $psnrDrop = $GlobalBestPsnrRef - $s2P
        if ($psnrDrop -gt $maxP) {
            $reasons.Add("psnr_drop=$([string](Fmt-Num $psnrDrop 4))>$(Fmt-Num $maxP 4)") | Out-Null
        }
    }
    if ((-not [double]::IsInfinity($GlobalBestSsimRef)) -and (-not [double]::IsNaN($GlobalBestSsimRef)) -and (-not [double]::IsNaN($s2S))) {
        $ssimDrop = $GlobalBestSsimRef - $s2S
        if ($ssimDrop -gt $maxS) {
            $reasons.Add("ssim_drop=$([string](Fmt-Num $ssimDrop 4))>$(Fmt-Num $maxS 4)") | Out-Null
        }
    }
    if ((-not [double]::IsInfinity($GlobalBestWl1Ref)) -and (-not [double]::IsNaN($GlobalBestWl1Ref)) -and (-not [double]::IsNaN($s2W))) {
        $wl1Rise = $s2W - $GlobalBestWl1Ref
        if ($wl1Rise -gt $maxW) {
            $reasons.Add("wl1_rise=$([string](Fmt-Num $wl1Rise 4))>$(Fmt-Num $maxW 4)") | Out-Null
        }
    }

    if ($reasons.Count -le 0) {
        return [pscustomobject]@{ pass = $true; reason = "pass" }
    }
    return [pscustomobject]@{ pass = $false; reason = ($reasons -join "; ") }
}

function Test-QualityGuardForCandidate(
    [double]$CandidatePsnr,
    [double]$CandidateSsim,
    [double]$CandidateWl1,
    [double]$ReferencePsnr,
    [double]$ReferenceSsim,
    [double]$ReferenceWl1,
    [double]$MaxPSNRDrop,
    [double]$MaxSSIMDrop,
    [double]$MaxWl1Rise
) {
    $reasons = New-Object System.Collections.Generic.List[string]
    $maxP = [Math]::Max(0.0, [double]$MaxPSNRDrop)
    $maxS = [Math]::Max(0.0, [double]$MaxSSIMDrop)
    $maxW = [Math]::Max(0.0, [double]$MaxWl1Rise)
    if ((-not [double]::IsNaN($ReferencePsnr)) -and (-not [double]::IsNaN($CandidatePsnr))) {
        $psnrDrop = $ReferencePsnr - $CandidatePsnr
        if ($psnrDrop -gt $maxP) {
            $reasons.Add("psnr_drop=$([string](Fmt-Num $psnrDrop 4))>$(Fmt-Num $maxP 4)") | Out-Null
        }
    }
    if ((-not [double]::IsNaN($ReferenceSsim)) -and (-not [double]::IsNaN($CandidateSsim))) {
        $ssimDrop = $ReferenceSsim - $CandidateSsim
        if ($ssimDrop -gt $maxS) {
            $reasons.Add("ssim_drop=$([string](Fmt-Num $ssimDrop 4))>$(Fmt-Num $maxS 4)") | Out-Null
        }
    }
    if ((-not [double]::IsNaN($ReferenceWl1)) -and (-not [double]::IsNaN($CandidateWl1))) {
        $wl1Rise = $CandidateWl1 - $ReferenceWl1
        if ($wl1Rise -gt $maxW) {
            $reasons.Add("wl1_rise=$([string](Fmt-Num $wl1Rise 4))>$(Fmt-Num $maxW 4)") | Out-Null
        }
    }
    if ($reasons.Count -le 0) {
        return [pscustomobject]@{ pass = $true; reason = "pass" }
    }
    return [pscustomobject]@{ pass = $false; reason = ($reasons -join "; ") }
}

function Resolve-LayeredGuardResult(
    [object]$Candidate,
    [double]$ReferencePsnr,
    [double]$ReferenceSsim,
    [double]$ReferenceWl1,
    [string]$Mode
) {
    if ($null -eq $Candidate) {
        return [pscustomobject]@{
            pass_explore = $false
            pass_promotion = $false
            guard_tier = "blocked"
            reason = "candidate_missing"
        }
    }
    $candP = To-DoubleOrNaN($Candidate.psnr)
    $candS = To-DoubleOrNaN($Candidate.ssim)
    $candW = To-DoubleOrNaN($Candidate.wl1)
    $modeNorm = ([string]$Mode).Trim().ToLowerInvariant()
    if ([string]::IsNullOrWhiteSpace($modeNorm)) { $modeNorm = "layered" }

    if ($modeNorm -eq "off") {
        return [pscustomobject]@{
            pass_explore = $true
            pass_promotion = $true
            guard_tier = "promotion"
            reason = "guard_off"
        }
    }

    $promotion = Test-QualityGuardForCandidate `
        -CandidatePsnr $candP `
        -CandidateSsim $candS `
        -CandidateWl1 $candW `
        -ReferencePsnr $ReferencePsnr `
        -ReferenceSsim $ReferenceSsim `
        -ReferenceWl1 $ReferenceWl1 `
        -MaxPSNRDrop $LayeredPromotionMaxPSNRDrop `
        -MaxSSIMDrop $LayeredPromotionMaxSSIMDrop `
        -MaxWl1Rise $LayeredPromotionMaxWl1Rise
    if ($modeNorm -eq "strict") {
        return [pscustomobject]@{
            pass_explore = [bool]$promotion.pass
            pass_promotion = [bool]$promotion.pass
            guard_tier = $(if ($promotion.pass) { "promotion" } else { "blocked" })
            reason = [string]$promotion.reason
        }
    }

    $explore = Test-QualityGuardForCandidate `
        -CandidatePsnr $candP `
        -CandidateSsim $candS `
        -CandidateWl1 $candW `
        -ReferencePsnr $ReferencePsnr `
        -ReferenceSsim $ReferenceSsim `
        -ReferenceWl1 $ReferenceWl1 `
        -MaxPSNRDrop $LayeredExploreMaxPSNRDrop `
        -MaxSSIMDrop $LayeredExploreMaxSSIMDrop `
        -MaxWl1Rise $LayeredExploreMaxWl1Rise
    $tier = "blocked"
    if ($promotion.pass) {
        $tier = "promotion"
    } elseif ($explore.pass) {
        $tier = "exploration"
    }
    $reason = "explore=$([string]$explore.reason); promotion=$([string]$promotion.reason)"
    return [pscustomobject]@{
        pass_explore = [bool]$explore.pass
        pass_promotion = [bool]$promotion.pass
        guard_tier = $tier
        reason = $reason
    }
}

function Build-LaneSnapshot(
    [string]$Lane,
    [object]$Stage,
    [object]$Guard
) {
    if ($null -eq $Stage) { return $null }
    $guardTier = ""
    $passExplore = $false
    $passPromotion = $false
    $guardReason = ""
    if ($null -ne $Guard) {
        $guardTier = [string]$Guard.guard_tier
        $passExplore = [bool]$Guard.pass_explore
        $passPromotion = [bool]$Guard.pass_promotion
        $guardReason = [string]$Guard.reason
    }
    return [pscustomobject]@{
        lane = $Lane
        stage = [string]$Stage.stage
        ghost = To-DoubleOrNaN($Stage.ghost)
        ghost_soft_score = To-DoubleOrNaN($Stage.ghost_soft_score)
        psnr = To-DoubleOrNaN($Stage.psnr)
        ssim = To-DoubleOrNaN($Stage.ssim)
        wl1 = To-DoubleOrNaN($Stage.wl1)
        best_ckpt = [string]$Stage.best_ckpt
        best_geom = [string]$Stage.best_geom
        best_visual_png = [string]$Stage.best_visual_png
        best_ghost_rows_csv = [string]$Stage.best_ghost_rows_csv
        guard_tier = $guardTier
        pass_explore = $passExplore
        pass_promotion = $passPromotion
        guard_reason = $guardReason
    }
}

function Invoke-Stage2PostRescue(
    [int]$Cycle,
    [object]$LaneAStage
) {
    if (-not $PostRescueEnabled) {
        return $null
    }
    if ($null -eq $LaneAStage) {
        return New-SkippedStageResult `
            -StageName ("cycle{0:D3}_stage2_post_rescue" -f $Cycle) `
            -PointTargetBlendMvPolicy "weak_to_depth" `
            -PointmapSource "point_head" `
            -PseudoGeomSubdir $currPseudo `
            -ResumeCkpt $currResume `
            -Reason "lane_a_missing" `
            -Overrides @{
                LaneId = "lane_b"
                CandidateFamily = "stage2_post_rescue"
            }
    }

    Write-Host "[autoloop] stage2 laneB start: run post-rescue chain."
    $upRc = 0
    & "$CodeDir\scripts\run_post_upstream_maskgrid.ps1" `
        -WaitPid 0 `
        -CodeDir $CodeDir `
        -SeqNames $SeqNames `
        -PseudoGeomSubdir ([string]$LaneAStage.best_geom) `
        -Lr $Lr `
        -LambdaPointMvDepthList ([string]$LaneAStage.stage_lambda_point_mv_depth_list) `
        -LambdaPointMvMaskList ([string]$LaneAStage.stage_lambda_point_mv_mask_list) `
        -MaxFramesShort $MaxFramesShort `
        -MaxStepsPerEpoch $MaxStepsPerEpoch `
        -EvalNumSamples $EvalNumSamples `
        -UnprojectImpl $UnprojectImpl `
        -PointTargetMode $PointTargetMode `
        -PointTargetConsensusAlphaFloor $PointTargetConsensusAlphaFloor `
        -PointTargetBlendMvPolicy "weak_to_depth" `
        -PointmapSource "point_head" `
        -ModalRunTimeoutSec $ModalRunTimeoutSec
    $upRc = [int]$LASTEXITCODE

    $followRc = 0
    & "$CodeDir\scripts\run_post_maskgrid_followup.ps1" `
        -CodeDir $CodeDir `
        -SeqNames $SeqNames `
        -PseudoGeomSubdir ([string]$LaneAStage.best_geom) `
        -Lr $Lr `
        -LambdaPointMvDepthList ([string]$LaneAStage.stage_lambda_point_mv_depth_list) `
        -LambdaPointMvMaskList ([string]$LaneAStage.stage_lambda_point_mv_mask_list) `
        -MaxFramesShort $MaxFramesShort `
        -MaxStepsPerEpoch $MaxStepsPerEpoch `
        -EvalNumSamples $EvalNumSamples `
        -UnprojectImpl $UnprojectImpl `
        -PointTargetMode $PointTargetMode `
        -PointTargetConsensusAlphaFloor $PointTargetConsensusAlphaFloor `
        -PointTargetBlendMvPolicy "weak_to_depth" `
        -PointmapSource "point_head" `
        -ModalRunTimeoutSec $ModalRunTimeoutSec
    $followRc = [int]$LASTEXITCODE

    $blendRc = 0
    & "$CodeDir\scripts\run_post_maskgrid_blend_rescue.ps1" `
        -CodeDir $CodeDir `
        -MaskgridMetaPath "logs/modal_phase5/post_upstream_maskgrid_latest.json" `
        -SeqNames $SeqNames `
        -PseudoGeomSubdir ([string]$LaneAStage.best_geom) `
        -Lr $Lr `
        -MaxFramesShort $MaxFramesShort `
        -MaxStepsPerEpoch $MaxStepsPerEpoch `
        -EvalNumSamples $EvalNumSamples `
        -UnprojectImpl $UnprojectImpl `
        -PointmapSource "point_head" `
        -ModalRunTimeoutSec $ModalRunTimeoutSec
    $blendRc = [int]$LASTEXITCODE

    $blendGate = Read-JsonMaybe -Path "logs/modal_phase5/blend_rescue_gate_latest.json"
    if ($blendGate -eq $null -or $blendGate.best -eq $null) {
        return New-SkippedStageResult `
            -StageName ("cycle{0:D3}_stage2_post_rescue" -f $Cycle) `
            -PointTargetBlendMvPolicy "weak_to_depth" `
            -PointmapSource "point_head" `
            -PseudoGeomSubdir ([string]$LaneAStage.best_geom) `
            -ResumeCkpt ([string]$LaneAStage.best_ckpt) `
            -Reason ("post_rescue_missing_best(up=$upRc follow=$followRc blend=$blendRc)") `
            -Overrides @{
                LaneId = "lane_b"
                CandidateFamily = "stage2_post_rescue"
            }
    }

    $bestSweepCsv = [string]$blendGate.best.sweep_csv
    if ([string]::IsNullOrWhiteSpace($bestSweepCsv) -or (-not (Test-Path $bestSweepCsv))) {
        $bestSweepCsv = "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv"
    }
    $best = Parse-BestSweepRow -CsvPath $bestSweepCsv
    if ($best -eq $null) {
        return New-SkippedStageResult `
            -StageName ("cycle{0:D3}_stage2_post_rescue" -f $Cycle) `
            -PointTargetBlendMvPolicy "weak_to_depth" `
            -PointmapSource "point_head" `
            -PseudoGeomSubdir ([string]$LaneAStage.best_geom) `
            -ResumeCkpt ([string]$LaneAStage.best_ckpt) `
            -Reason "post_rescue_best_parse_failed" `
            -Overrides @{
                LaneId = "lane_b"
                CandidateFamily = "stage2_post_rescue"
            }
    }

    $bestLabel = [string]$best.best_label
    $bestSweepFromRow = [string]$best.sweep_csv
    if ([string]::IsNullOrWhiteSpace($bestSweepFromRow)) {
        $bestSweepFromRow = $bestSweepCsv
    }
    $ckptInfo = Resolve-BestCkpt -SweepCsvPath $bestSweepFromRow -BestLabel $bestLabel
    $bestCkpt = ""
    $bestGeom = ""
    if ($ckptInfo -ne $null) {
        $bestCkpt = [string]$ckptInfo.ft_ckpt
        $bestGeom = [string]$ckptInfo.geom_subdir
    }
    if ([string]::IsNullOrWhiteSpace($bestCkpt)) { $bestCkpt = [string]$LaneAStage.best_ckpt }
    if ([string]::IsNullOrWhiteSpace($bestGeom)) { $bestGeom = [string]$LaneAStage.best_geom }

    $bestGhostRowsCsv = Resolve-GhostRowsCsv -BestRow $best
    $stats = Get-GhostRowsStats -GhostRowsCsv $bestGhostRowsCsv
    $bestVisualPng = ""
    $stageBestStripPng = ""
    $bestGhostWidthRatio = [double]::NaN
    $bestGhostAreaRatio = [double]::NaN
    $bestGhostPeakCount = [double]::NaN
    $bestGhostCenterOffset = [double]::NaN
    if ($stats -ne $null) {
        $bestGhostWidthRatio = To-DoubleOrNaN($stats.mean_width_ratio)
        $bestGhostAreaRatio = To-DoubleOrNaN($stats.mean_area_ratio)
        $bestGhostPeakCount = To-DoubleOrNaN($stats.mean_peak_count)
        $bestGhostCenterOffset = To-DoubleOrNaN($stats.mean_center_offset_ratio)
        $bestVisualPng = [string]$stats.first_path
        $stripOut = "logs/modal_phase5/ghost_stage_best_cycle{0:D3}_stage2_post_rescue.png" -f $Cycle
        $stripMade = Make-ContactSheetSafe -ImagePaths @($stats.image_paths) -OutPng $stripOut
        if (-not [string]::IsNullOrWhiteSpace($stripMade)) {
            $stageBestStripPng = $stripMade
        }
    }

    $ghostSoft = To-DoubleOrNaN($best.ghost_soft_score)
    $ghostVisual = To-DoubleOrNaN($best.ghost_visual_score)
    $predLuma = To-DoubleOrNaN($best.pred_luma_mean)
    $predNonBlack = To-DoubleOrNaN($best.pred_nonblack_ratio_thr008)
    $visualGuardBlocked = To-BoolLoose -Value $best.visual_guard_blocked -Default $false
    $visualGuardReason = [string]$best.visual_guard_reason
    $evalNumSrcViews = [string]$best.eval_num_src_views
    $evalNumSrcViewsActual = [string]$best.eval_num_src_views_actual
    $evalNumSrcViewsMismatch = To-BoolLoose -Value $best.eval_num_src_views_mismatch -Default $false
    $camCountUsed = [string]$best.cam_count_used
    $precomputeSourceRequested = [string]$best.precompute_source_requested
    $precomputeSourceResolved = [string]$best.precompute_source_resolved
    $precomputeFallbackUsed = To-BoolLoose -Value $best.precompute_fallback_used -Default $false
    $precomputeTimeoutHit = To-BoolLoose -Value $best.precompute_timeout_hit -Default $false
    return [pscustomobject]@{
        stage = ("cycle{0:D3}_stage2_post_rescue" -f $Cycle)
        policy = "weak_to_depth"
        pointmap_source = "point_head"
        lane_id = "lane_b"
        candidate_family = "stage2_post_rescue"
        guard_tier = ""
        rollback_triggered = $false
        stage_lambda_point_mv_depth_list = [string]$LaneAStage.stage_lambda_point_mv_depth_list
        stage_lambda_point_mv_mask_list = [string]$LaneAStage.stage_lambda_point_mv_mask_list
        stage_point_target_mode = "blend"
        stage_point_target_consensus_alpha_floor = 0.0
        stage_lambda_point = To-DoubleOrNaN($LaneAStage.stage_lambda_point)
        stage_lambda_point_reproj = To-DoubleOrNaN($LaneAStage.stage_lambda_point_reproj)
        stage_point_mv_mask_hit_thr = To-DoubleOrNaN($LaneAStage.stage_point_mv_mask_hit_thr)
        stage_point_mv_mask_min_tgt_fg_ratio = To-DoubleOrNaN($LaneAStage.stage_point_mv_mask_min_tgt_fg_ratio)
        stage_point_mv_mask_soft_blur_px = $LaneAStage.stage_point_mv_mask_soft_blur_px
        stage_point_mv_mask_soft_blur_iters = $LaneAStage.stage_point_mv_mask_soft_blur_iters
        stage_point_mv_mask_soft_mix = To-DoubleOrNaN($LaneAStage.stage_point_mv_mask_soft_mix)
        stage_point_mv_mask_soft_hit_thr = To-DoubleOrNaN($LaneAStage.stage_point_mv_mask_soft_hit_thr)
        stage_point_mv_stride = $LaneAStage.stage_point_mv_stride
        stage_point_mv_depth_max_pairs = $LaneAStage.stage_point_mv_depth_max_pairs
        stage_point_mv_depth_support_mode = [string]$LaneAStage.stage_point_mv_depth_support_mode
        stage_point_mv_depth_support_floor = To-DoubleOrNaN($LaneAStage.stage_point_mv_depth_support_floor)
        stage_point_loss_fg_erode_px = $LaneAStage.stage_point_loss_fg_erode_px
        stage_point_mv_depth_fg_erode_px = $LaneAStage.stage_point_mv_depth_fg_erode_px
        stage_point_cons_quantile = To-DoubleOrNaN($LaneAStage.stage_point_cons_quantile)
        stage_point_cons_focus = [string]$LaneAStage.stage_point_cons_focus
        stage_point_residual_quantile = To-DoubleOrNaN($LaneAStage.stage_point_residual_quantile)
        stage_point_residual_focus = [string]$LaneAStage.stage_point_residual_focus
        stage_point_residual_boost = To-DoubleOrNaN($LaneAStage.stage_point_residual_boost)
        stage_point_mv_depth_outlier_boost = To-DoubleOrNaN($LaneAStage.stage_point_mv_depth_outlier_boost)
        stage_supervision_weight_mode = [string]$LaneAStage.stage_supervision_weight_mode
        rc = $blendRc
        ghost = To-DoubleOrNaN($best.ghost_score_mean)
        psnr = To-DoubleOrNaN($best.mean_PSNR)
        ssim = To-DoubleOrNaN($best.mean_SSIM)
        wl1 = To-DoubleOrNaN($best.mean_weighted_L1)
        best_label = [string]$best.best_label
        best_geom = $bestGeom
        best_ckpt = $bestCkpt
        best_lambda_point_mv_depth = [string]$best.lambda_point_mv_depth
        best_lambda_point_mv_mask = [string]$best.lambda_point_mv_mask
        best_ghost_rows_csv = $bestGhostRowsCsv
        best_visual_png = $bestVisualPng
        stage_best_strip_png = $stageBestStripPng
        best_ghost_width_ratio = $bestGhostWidthRatio
        best_ghost_area_ratio = $bestGhostAreaRatio
        best_ghost_peak_count = $bestGhostPeakCount
        best_ghost_center_offset_ratio = $bestGhostCenterOffset
        ghost_soft_score = $ghostSoft
        ghost_visual_score = $ghostVisual
        pred_luma_mean = $predLuma
        pred_nonblack_ratio_thr008 = $predNonBlack
        visual_guard_blocked = $visualGuardBlocked
        visual_guard_reason = $visualGuardReason
        eval_num_src_views = $evalNumSrcViews
        eval_num_src_views_actual = $evalNumSrcViewsActual
        eval_num_src_views_mismatch = $evalNumSrcViewsMismatch
        cam_count_used = $camCountUsed
        precompute_source_requested = $precomputeSourceRequested
        precompute_source_resolved = $precomputeSourceResolved
        precompute_fallback_used = $precomputeFallbackUsed
        precompute_timeout_hit = $precomputeTimeoutHit
        sweep_csv = $bestSweepFromRow
        sweep_md = "logs/modal_phase5/blend_rescue_sweep_latest.md"
        raw_sweep_csv = $bestSweepFromRow
        pseudo_geom_in = [string]$LaneAStage.best_geom
        resume_ckpt_in = [string]$LaneAStage.best_ckpt
        stage_skip_reason = ""
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    }
}

function Resolve-Stage2DualLaneDecision(
    [object]$LaneAStage,
    [object]$LaneBStage,
    [double]$ReferencePsnr,
    [double]$ReferenceSsim,
    [double]$ReferenceWl1
) {
    $laneAGuard = Resolve-LayeredGuardResult `
        -Candidate $LaneAStage `
        -ReferencePsnr $ReferencePsnr `
        -ReferenceSsim $ReferenceSsim `
        -ReferenceWl1 $ReferenceWl1 `
        -Mode $QualityGuardMode
    $laneBGuard = $null
    if ($LaneBStage -ne $null) {
        $laneBGuard = Resolve-LayeredGuardResult `
            -Candidate $LaneBStage `
            -ReferencePsnr $ReferencePsnr `
            -ReferenceSsim $ReferenceSsim `
            -ReferenceWl1 $ReferenceWl1 `
            -Mode $QualityGuardMode
    }

    $selected = $LaneAStage
    $activeLane = "lane_a"
    $decisionReason = "lane_a_default"
    $rollbackReason = ""

    $laneAGhost = if ($LaneAStage -ne $null) { To-DoubleOrNaN($LaneAStage.ghost) } else { [double]::NaN }
    $laneBGhost = if ($LaneBStage -ne $null) { To-DoubleOrNaN($LaneBStage.ghost) } else { [double]::NaN }

    if (($LaneBStage -ne $null) -and (-not [double]::IsNaN($laneBGhost))) {
        $ghostGainVsA = [double]::NaN
        if (-not [double]::IsNaN($laneAGhost)) {
            $ghostGainVsA = $laneAGhost - $laneBGhost
        }
        if (([double]::IsNaN($laneAGhost)) -and [bool]$laneBGuard.pass_explore) {
            $selected = $LaneBStage
            $activeLane = "lane_b"
            $decisionReason = "lane_a_no_usable_ghost_use_lane_b"
        } elseif ((-not [double]::IsNaN($ghostGainVsA)) -and ($ghostGainVsA -ge [Math]::Max(0.0, [double]$PromotionGhostDelta)) -and [bool]$laneBGuard.pass_explore) {
            $selected = $LaneBStage
            $activeLane = "lane_b"
            $decisionReason = "lane_b_ghost_gain_pass_delta"
        } elseif ((-not [double]::IsNaN($ghostGainVsA)) -and ($ghostGainVsA -gt 0.0) -and [bool]$laneBGuard.pass_explore -and (-not [bool]$laneAGuard.pass_explore)) {
            $selected = $LaneBStage
            $activeLane = "lane_b"
            $decisionReason = "lane_b_pass_explore_while_lane_a_blocked"
        } elseif ((-not [double]::IsNaN($ghostGainVsA)) -and ($ghostGainVsA -gt 0.0) -and (-not [bool]$laneBGuard.pass_explore)) {
            $rollbackReason = "lane_b_better_ghost_but_quality_blocked: $([string]$laneBGuard.reason)"
            $decisionReason = "keep_lane_a_due_to_lane_b_quality_block"
        } elseif ((-not [double]::IsNaN($ghostGainVsA)) -and ($ghostGainVsA -le 0.0) -and (-not [bool]$laneBGuard.pass_explore)) {
            $rollbackReason = "lane_b_regress_and_quality_blocked: $([string]$laneBGuard.reason)"
            $decisionReason = "keep_lane_a_lane_b_regressed"
        } else {
            $decisionReason = "keep_lane_a_micro_gain_or_no_gain"
        }
    } else {
        $decisionReason = "lane_b_missing_or_invalid"
    }

    if ($selected -ne $null) {
        $selTier = if ($activeLane -eq "lane_b") { [string]$laneBGuard.guard_tier } else { [string]$laneAGuard.guard_tier }
        $selRollback = $false
        if (-not [string]::IsNullOrWhiteSpace($rollbackReason)) { $selRollback = $true }
        $selected | Add-Member -NotePropertyName lane_id -NotePropertyValue $activeLane -Force
        $selected | Add-Member -NotePropertyName guard_tier -NotePropertyValue $selTier -Force
        $selected | Add-Member -NotePropertyName rollback_triggered -NotePropertyValue $selRollback -Force
        if ([string]::IsNullOrWhiteSpace([string]$selected.candidate_family)) {
            $selected | Add-Member -NotePropertyName candidate_family -NotePropertyValue $(if ($activeLane -eq "lane_b") { "stage2_post_rescue" } else { "stage2_training" }) -Force
        }
    }
    if ($LaneAStage -ne $null) {
        $LaneAStage | Add-Member -NotePropertyName lane_id -NotePropertyValue "lane_a" -Force
        $LaneAStage | Add-Member -NotePropertyName guard_tier -NotePropertyValue ([string]$laneAGuard.guard_tier) -Force
        $LaneAStage | Add-Member -NotePropertyName rollback_triggered -NotePropertyValue $false -Force
        if ([string]::IsNullOrWhiteSpace([string]$LaneAStage.candidate_family)) {
            $LaneAStage | Add-Member -NotePropertyName candidate_family -NotePropertyValue "stage2_training" -Force
        }
    }
    if ($LaneBStage -ne $null) {
        $laneBRollback = (-not [string]::IsNullOrWhiteSpace($rollbackReason)) -and ($activeLane -ne "lane_b")
        $LaneBStage | Add-Member -NotePropertyName lane_id -NotePropertyValue "lane_b" -Force
        $LaneBStage | Add-Member -NotePropertyName guard_tier -NotePropertyValue ([string]$laneBGuard.guard_tier) -Force
        $LaneBStage | Add-Member -NotePropertyName rollback_triggered -NotePropertyValue $laneBRollback -Force
        if ([string]::IsNullOrWhiteSpace([string]$LaneBStage.candidate_family)) {
            $LaneBStage | Add-Member -NotePropertyName candidate_family -NotePropertyValue "stage2_post_rescue" -Force
        }
    }

    return [pscustomobject]@{
        selected_stage = $selected
        active_lane = $activeLane
        lane_a_guard = $laneAGuard
        lane_b_guard = $laneBGuard
        lane_a_best = Build-LaneSnapshot -Lane "lane_a" -Stage $LaneAStage -Guard $laneAGuard
        lane_b_best = Build-LaneSnapshot -Lane "lane_b" -Stage $LaneBStage -Guard $laneBGuard
        guard_tier = $(if ($activeLane -eq "lane_b") { [string]$laneBGuard.guard_tier } else { [string]$laneAGuard.guard_tier })
        decision_reason = $decisionReason
        rollback_reason = $rollbackReason
    }
}

function Test-StageHasUsableGhost([object]$Stage) {
    if ($null -eq $Stage) { return $false }
    $g = To-DoubleOrNaN($Stage.ghost)
    if (-not [double]::IsNaN($g)) { return $true }
    try {
        if ([int]$Stage.rc -eq 0) { return $true }
    } catch {
    }
    return $false
}

function Test-StageHasUsableResume([object]$Stage) {
    if ($null -eq $Stage) { return $false }
    $ckpt = [string]$Stage.best_ckpt
    if ([string]::IsNullOrWhiteSpace($ckpt)) { return $false }
    if (Test-StageHasUsableGhost -Stage $Stage) { return $true }
    try {
        if ([int]$Stage.rc -eq 0) { return $true }
    } catch {
    }
    return $false
}

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    $safe = Normalize-JsonFiniteNumbers $Obj
    $json = $safe | ConvertTo-Json -Depth 20
    $fullPath = Join-Path (Resolve-Path ".").Path $Path
    $tmpPath = "$fullPath.tmp.$PID.$([DateTime]::UtcNow.Ticks)"
    [System.IO.File]::WriteAllText($tmpPath, $json, $enc)
    try {
        if ([System.IO.File]::Exists($fullPath)) {
            [System.IO.File]::Replace($tmpPath, $fullPath, $null, $true)
        } else {
            [System.IO.File]::Move($tmpPath, $fullPath)
        }
    } catch {
        try {
            [System.IO.File]::Copy($tmpPath, $fullPath, $true)
        } finally {
            if ([System.IO.File]::Exists($tmpPath)) {
                [System.IO.File]::Delete($tmpPath)
            }
        }
    }
}

function Sanitize-TextForUtf8Log([string]$Text) {
    if ($null -eq $Text) { return "" }
    # Keep line breaks and tabs; strip other control characters that corrupt markdown logs.
    return [regex]::Replace([string]$Text, "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "")
}

function Read-JsonMaybe([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try {
        return (Get-Content $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Parse-DateMaybe([string]$Text) {
    if ([string]::IsNullOrWhiteSpace($Text)) { return [datetime]::MinValue }
    try {
        return [datetime]::Parse($Text)
    } catch {
        return [datetime]::MinValue
    }
}

function Fmt-Num([double]$x, [int]$Digits = 6) {
    if ([double]::IsNaN($x)) { return "NaN" }
    return ("{0:F$Digits}" -f $x)
}

function Mean-FromRows([object[]]$Rows, [string]$Key) {
    $vals = @()
    foreach ($r in @($Rows)) {
        $v = To-DoubleOrNaN($r.$Key)
        if (-not [double]::IsNaN($v)) { $vals += $v }
    }
    if ($vals.Count -le 0) { return [double]::NaN }
    return [double](($vals | Measure-Object -Average).Average)
}

function Resolve-GhostRowsCsv([object]$BestRow) {
    if ($null -eq $BestRow) { return "" }
    $direct = [string]$BestRow.ghost_rows_csv
    if (-not [string]::IsNullOrWhiteSpace($direct) -and (Test-Path $direct)) {
        return $direct
    }
    $summary = [string]$BestRow.ghost_summary_csv
    if (-not [string]::IsNullOrWhiteSpace($summary)) {
        $guess = [string]$summary -replace "ghost_score_summary_", "ghost_score_rows_"
        if (Test-Path $guess) { return $guess }
    }
    return ""
}

function Get-GhostRowsStats([string]$GhostRowsCsv) {
    if ([string]::IsNullOrWhiteSpace($GhostRowsCsv) -or (-not (Test-Path $GhostRowsCsv))) {
        return $null
    }
    $rows = @(Import-Csv $GhostRowsCsv)
    if ($rows.Count -le 0) { return $null }

    $okRows = @(
        $rows | Where-Object { [string]::IsNullOrWhiteSpace([string]$_.error) }
    )
    if ($okRows.Count -le 0) { $okRows = @($rows) }

    $imgRows = @(
        $okRows |
            Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_.path) } |
            Sort-Object { [int]$_.step }
    )
    $firstPath = ""
    if ($imgRows.Count -gt 0) {
        $step0 = @($imgRows | Where-Object { [int]$_.step -eq 0 } | Select-Object -First 1)
        if ($step0.Count -gt 0) {
            $firstPath = [string]$step0[0].path
        } else {
            $firstPath = [string]$imgRows[0].path
        }
    }

    return [pscustomobject]@{
        mean_ghost = Mean-FromRows -Rows $okRows -Key "ghost_score"
        mean_width_ratio = Mean-FromRows -Rows $okRows -Key "width_ratio"
        mean_area_ratio = Mean-FromRows -Rows $okRows -Key "area_ratio"
        mean_peak_count = Mean-FromRows -Rows $okRows -Key "peak_count"
        mean_center_offset_ratio = Mean-FromRows -Rows $okRows -Key "center_offset_ratio"
        first_path = $firstPath
        image_paths = @($imgRows | ForEach-Object { [string]$_.path })
    }
}

function Make-ContactSheetSafe([string[]]$ImagePaths, [string]$OutPng) {
    $valid = @(
        @($ImagePaths) |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path $_) } |
            Select-Object -Unique
    )
    if ($valid.Count -le 0) { return "" }
    $tool = Join-Path $repoRoot "tools/make_contact_sheet.py"
    if (-not (Test-Path $tool)) { return "" }
    $outDir = Split-Path -Parent $OutPng
    if (-not [string]::IsNullOrWhiteSpace($outDir)) {
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    }
    try {
        $args = @($tool, "--out", $OutPng, "--pad", "8", "--bg", "18,18,18", "--images") + $valid
        python @args | Out-Null
    } catch {
    }
    if (Test-Path $OutPng) { return $OutPng }
    return ""
}

function Build-VisualJudgement(
    [object]$PrevStats,
    [object]$CurrStats,
    [double]$GhostDelta
) {
    if ($null -eq $CurrStats) {
        return "可视化统计解析失败，需要人工复核。"
    }
    if ($null -eq $PrevStats) {
        return "暂无上一轮参考，当前可视化作为新基线。"
    }
    $peakDelta = To-DoubleOrNaN($CurrStats.mean_peak_count) - To-DoubleOrNaN($PrevStats.mean_peak_count)
    $widthDelta = To-DoubleOrNaN($CurrStats.mean_width_ratio) - To-DoubleOrNaN($PrevStats.mean_width_ratio)
    $areaDelta = To-DoubleOrNaN($CurrStats.mean_area_ratio) - To-DoubleOrNaN($PrevStats.mean_area_ratio)
    if ((-not [double]::IsNaN($GhostDelta)) -and ($GhostDelta -le -0.03) -and ($peakDelta -le -0.2) -and ($widthDelta -le -0.02)) {
        return "双峰与横向拖影收敛，重影有明显减轻。"
    }
    if ((-not [double]::IsNaN($GhostDelta)) -and ($GhostDelta -ge 0.03) -and ($peakDelta -ge 0.2)) {
        return "双峰/拖尾增强，重影恶化。"
    }
    if ((-not [double]::IsNaN($areaDelta)) -and ($areaDelta -le -0.05) -and ($widthDelta -le 0.02)) {
        return "前景膨胀略有收敛，但多峰重影仍然明显。"
    }
    return "可视变化有限，重影形态与参考接近。"
}

function Append-MentorCycleUpdate(
    [string]$MentorPath,
    [int]$Cycle,
    [object[]]$Stages,
    [object]$CycleBest,
    [double]$GlobalBestGhost,
    [double]$GlobalBestSsim,
    [int]$NoSubstantialImproveCycles,
    [string]$TuneAction,
    [string]$CycleComparePng,
    [string]$VisualConclusion,
    [bool]$CycleRegressed,
    [string]$CycleRegressReason,
    [bool]$CycleQualityGuardBlocked,
    [string]$CycleQualityGuardReason,
    [bool]$RolledBackLastTune,
    [string]$RollbackAction,
    [bool]$ShouldStop,
    [string]$RouteMode = "",
    [string]$NextRouteMode = "",
    [string]$ActiveLane = "",
    [string]$GuardTier = "",
    [string]$DecisionReason = "",
    [string]$RollbackReason = "",
    $LaneABest = $null,
    $LaneBBest = $null,
    [bool]$Stage2HasPotential = $true,
    [string]$Stage2PotentialReason = "",
    [bool]$SkipDeepStages = $false,
    [string]$SkipDeepStagesReason = "",
    [bool]$ShouldStopByABValidation = $false,
    [string]$StopByABValidationReason = ""
) {
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("") | Out-Null
    $lines.Add("## $(Get-Date -Format 'yyyy-MM-dd') autoloop 更新（cycle $Cycle）") | Out-Null
    $lines.Add("- updated_at: $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')") | Out-Null
    $lines.Add("- cycle_best: stage=$($CycleBest.stage), ghost=$(Fmt-Num (To-DoubleOrNaN($CycleBest.ghost))), PSNR=$(Fmt-Num (To-DoubleOrNaN($CycleBest.psnr))), SSIM=$(Fmt-Num (To-DoubleOrNaN($CycleBest.ssim))), wL1=$(Fmt-Num (To-DoubleOrNaN($CycleBest.wl1)))") | Out-Null
    $lines.Add("- global_best_ghost: $(Fmt-Num $GlobalBestGhost)") | Out-Null
    $lines.Add("- global_best_ssim: $(Fmt-Num $GlobalBestSsim)") | Out-Null
    $lines.Add("- no_substantial_improve_cycles: $NoSubstantialImproveCycles") | Out-Null
    if (-not [string]::IsNullOrWhiteSpace($RouteMode)) {
        $lines.Add("- route_mode: $RouteMode -> $NextRouteMode") | Out-Null
    }
    if (-not [string]::IsNullOrWhiteSpace($ActiveLane)) {
        $lines.Add("- stage2_lane: $ActiveLane (tier=$GuardTier, decision=$DecisionReason)") | Out-Null
    }
    if (-not [string]::IsNullOrWhiteSpace($RollbackReason)) {
        $lines.Add("- stage2_lane_rollback_reason: $RollbackReason") | Out-Null
    }
    if ($LaneABest -ne $null) {
        $lines.Add("- lane_a_best: ghost=$(Fmt-Num (To-DoubleOrNaN($LaneABest.ghost))), PSNR=$(Fmt-Num (To-DoubleOrNaN($LaneABest.psnr))), SSIM=$(Fmt-Num (To-DoubleOrNaN($LaneABest.ssim))), wL1=$(Fmt-Num (To-DoubleOrNaN($LaneABest.wl1))), guard=$($LaneABest.guard_tier)") | Out-Null
    }
    if ($LaneBBest -ne $null) {
        $lines.Add("- lane_b_best: ghost=$(Fmt-Num (To-DoubleOrNaN($LaneBBest.ghost))), PSNR=$(Fmt-Num (To-DoubleOrNaN($LaneBBest.psnr))), SSIM=$(Fmt-Num (To-DoubleOrNaN($LaneBBest.ssim))), wL1=$(Fmt-Num (To-DoubleOrNaN($LaneBBest.wl1))), guard=$($LaneBBest.guard_tier)") | Out-Null
    }
    $lines.Add("- stage2_has_potential: $Stage2HasPotential ($Stage2PotentialReason)") | Out-Null
    if ($SkipDeepStages) {
        $lines.Add("- skip_deep_stages: true ($SkipDeepStagesReason)") | Out-Null
    }
    $lines.Add("- next_single_var_tune: $TuneAction") | Out-Null
    $lines.Add("- cycle_regressed: $CycleRegressed") | Out-Null
    if (-not [string]::IsNullOrWhiteSpace($CycleRegressReason)) {
        $lines.Add("- cycle_regress_reason: $CycleRegressReason") | Out-Null
    }
    if ($RolledBackLastTune) {
        $lines.Add("- rollback_last_tune: true ($RollbackAction)") | Out-Null
    }
    if ($CycleQualityGuardBlocked) {
        $lines.Add("- quality_guard_blocked: true ($CycleQualityGuardReason)") | Out-Null
    }
    if ($ShouldStopByABValidation) {
        $lines.Add("- ab_validation_stop: true ($StopByABValidationReason)") | Out-Null
    }
    if (-not [string]::IsNullOrWhiteSpace($CycleComparePng)) {
        $lines.Add("- compare_png: $CycleComparePng") | Out-Null
    }
    $lines.Add("- visual_reading: $VisualConclusion") | Out-Null
    $lines.Add("- stage_best_metrics:") | Out-Null
    foreach ($s in @($Stages)) {
        $skipNote = ""
        if (-not [string]::IsNullOrWhiteSpace([string]$s.stage_skip_reason)) {
            $skipNote = ", skip_reason=$([string]$s.stage_skip_reason)"
        }
        $lines.Add("  - $($s.stage): ghost=$(Fmt-Num (To-DoubleOrNaN($s.ghost))), PSNR=$(Fmt-Num (To-DoubleOrNaN($s.psnr))), SSIM=$(Fmt-Num (To-DoubleOrNaN($s.ssim))), wL1=$(Fmt-Num (To-DoubleOrNaN($s.wl1))), strip=$($s.stage_best_strip_png)$skipNote") | Out-Null
    }
    if ($ShouldStop) {
        $lines.Add("- stop_decision: 建议停止继续烧算力；多轮未出现实质性 ghost 下降。") | Out-Null
        $lines.Add("- next_route: 提升前景一致性约束、加入短窗口时序一致性、提高相机-点云重投影约束权重。") | Out-Null
        $lines.Add("- minimal_validation: 运行两组 A/B 短跑（N=40, max_steps=80），单变量改动并对比指标与 step0/1/2 视觉。") | Out-Null
    }

    $abs = Join-Path (Resolve-Path ".").Path $MentorPath
    $dir = Split-Path -Parent $abs
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $enc = New-Object System.Text.UTF8Encoding($false)
    $txt = Sanitize-TextForUtf8Log -Text (($lines -join "`n") + "`n")
    [System.IO.File]::AppendAllText($abs, $txt, $enc)
}

function Write-InterimAutoloopArtifacts(
    [object[]]$History,
    [int]$CurrentCycle,
    [object[]]$CurrentStages,
    [datetime]$Deadline,
    [double]$GlobalBestGhost,
    [double]$GlobalBestPsnr,
    [double]$GlobalBestSsim,
    [double]$GlobalBestWl1,
    [string]$CurrResume,
    [string]$CurrPseudo,
    [string]$PendingTuneAction,
    [string]$ActiveLane = "lane_a",
    $LaneABest = $null,
    $LaneBBest = $null,
    [string]$GuardTier = "",
    [string]$DecisionReason = "",
    [string]$RollbackReason = ""
) {
    $flatRows = @()
    $nowText = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"

    $addRow = {
        param(
            [int]$CycleId,
            [object]$S,
            [string]$UpdatedAt,
            $CycleSubstantialImproved,
            $CycleRegressed,
            [string]$CycleRegressReason,
            $CycleQualityGuardBlocked,
            [string]$CycleQualityGuardReason,
            $RegressCycles,
            $RolledBackLastTune,
            [string]$RolledBackTuneAction,
            [string]$TuneActionNext,
            [string]$CycleComparePng,
            [string]$ResumeUpdateReason
        )
        if ($null -eq $S) { return }
        $flatRows += [pscustomobject]@{
            cycle = $CycleId
            stage = $S.stage
            policy = $S.policy
            pointmap_source = $S.pointmap_source
            lane_id = $S.lane_id
            candidate_family = $S.candidate_family
            guard_tier = $S.guard_tier
            rollback_triggered = $S.rollback_triggered
            point_target_mode = $S.stage_point_target_mode
            lambda_point_mv_depth_list = $S.stage_lambda_point_mv_depth_list
            lambda_point_mv_mask_list = $S.stage_lambda_point_mv_mask_list
            lambda_point = $S.stage_lambda_point
            point_mv_mask_hit_thr = $S.stage_point_mv_mask_hit_thr
            point_mv_mask_min_tgt_fg_ratio = $S.stage_point_mv_mask_min_tgt_fg_ratio
            point_mv_mask_soft_mix = $S.stage_point_mv_mask_soft_mix
            point_mv_mask_soft_hit_thr = $S.stage_point_mv_mask_soft_hit_thr
            point_mv_stride = $S.stage_point_mv_stride
            point_mv_depth_max_pairs = $S.stage_point_mv_depth_max_pairs
            point_mv_depth_pair_mode = $S.stage_point_mv_depth_pair_mode
            point_mv_depth_support_mode = $S.stage_point_mv_depth_support_mode
            point_mv_depth_support_floor = $S.stage_point_mv_depth_support_floor
            point_cons_focus = $S.stage_point_cons_focus
            point_residual_focus = $S.stage_point_residual_focus
            rc = $S.rc
            ghost = $S.ghost
            psnr = $S.psnr
            ssim = $S.ssim
            wl1 = $S.wl1
            best_geom = $S.best_geom
            best_ckpt = $S.best_ckpt
            best_lambda_point_mv_depth = $S.best_lambda_point_mv_depth
            best_lambda_point_mv_mask = $S.best_lambda_point_mv_mask
            best_ghost_rows_csv = $S.best_ghost_rows_csv
            best_visual_png = $S.best_visual_png
            stage_best_strip_png = $S.stage_best_strip_png
            stage_skip_reason = $S.stage_skip_reason
            best_ghost_width_ratio = $S.best_ghost_width_ratio
            best_ghost_area_ratio = $S.best_ghost_area_ratio
            best_ghost_peak_count = $S.best_ghost_peak_count
            best_ghost_center_offset_ratio = $S.best_ghost_center_offset_ratio
            ghost_soft_score = $S.ghost_soft_score
            ghost_visual_score = $S.ghost_visual_score
            pred_luma_mean = $S.pred_luma_mean
            pred_nonblack_ratio_thr008 = $S.pred_nonblack_ratio_thr008
            visual_guard_blocked = $S.visual_guard_blocked
            visual_guard_reason = $S.visual_guard_reason
            quality_guard_blocked = $S.quality_guard_blocked
            quality_guard_reason = $S.quality_guard_reason
            candidate_invalid_reason = $S.candidate_invalid_reason
            eval_num_src_views = $S.eval_num_src_views
            eval_num_src_views_declared = $S.eval_num_src_views_declared
            eval_num_src_views_actual = $S.eval_num_src_views_actual
            eval_num_src_views_mismatch = $S.eval_num_src_views_mismatch
            cam_count_used = $S.cam_count_used
            precompute_source = $S.precompute_source
            precompute_source_requested = $S.precompute_source_requested
            precompute_source_resolved = $S.precompute_source_resolved
            precompute_fallback_used = $S.precompute_fallback_used
            precompute_timeout_hit = $S.precompute_timeout_hit
            sweep_csv = $S.sweep_csv
            cycle_substantial_improved = $CycleSubstantialImproved
            cycle_regressed = $CycleRegressed
            cycle_regress_reason = $CycleRegressReason
            cycle_quality_guard_blocked = $CycleQualityGuardBlocked
            cycle_quality_guard_reason = $CycleQualityGuardReason
            regress_cycles = $RegressCycles
            rolled_back_last_tune = $RolledBackLastTune
            rolled_back_tune_action = $RolledBackTuneAction
            tune_action_next = $TuneActionNext
            cycle_compare_png = $CycleComparePng
            resume_update_reason = $ResumeUpdateReason
            updated_at = $UpdatedAt
        }
    }

    foreach ($h in @($History)) {
        foreach ($s in @($h.stage1, $h.stage2_lane_a, $h.stage2, $h.stage2_lane_b, $h.stage3, $h.stage4, $h.stage5)) {
            & $addRow `
                -CycleId ([int]$h.cycle) `
                -S $s `
                -UpdatedAt ([string]$h.updated_at) `
                -CycleSubstantialImproved $h.cycle_substantial_improved `
                -CycleRegressed $h.cycle_regressed `
                -CycleRegressReason ([string]$h.cycle_regress_reason) `
                -CycleQualityGuardBlocked $h.cycle_quality_guard_blocked `
                -CycleQualityGuardReason ([string]$h.cycle_quality_guard_reason) `
                -RegressCycles $h.regress_cycles `
                -RolledBackLastTune $h.rolled_back_last_tune `
                -RolledBackTuneAction ([string]$h.rolled_back_tune_action) `
                -TuneActionNext ([string]$h.tune_action_next) `
                -CycleComparePng ([string]$h.cycle_compare_png) `
                -ResumeUpdateReason ([string]$h.resume_update_reason)
        }
    }

    foreach ($s in @($CurrentStages)) {
        & $addRow `
            -CycleId $CurrentCycle `
            -S $s `
            -UpdatedAt $nowText `
            -CycleSubstantialImproved "" `
            -CycleRegressed "" `
            -CycleRegressReason "" `
            -CycleQualityGuardBlocked "" `
            -CycleQualityGuardReason "" `
            -RegressCycles "" `
            -RolledBackLastTune "" `
            -RolledBackTuneAction "" `
            -TuneActionNext $PendingTuneAction `
            -CycleComparePng "" `
            -ResumeUpdateReason "interim"
    }

    if ($flatRows.Count -gt 0) {
        $flatRows | Export-Csv "logs/modal_phase5/ghost_autoloop_latest.csv" -NoTypeInformation -Encoding UTF8
    }

    $candidateResultLatest = Read-JsonMaybe -Path "logs/modal_phase5/candidate_result_latest.json"

    $currStageName = ""
    $currEvalNumSrcViews = ""
    $currEvalNumSrcViewsDeclared = ""
    $currEvalNumSrcViewsActual = ""
    $currEvalNumSrcViewsMismatch = $false
    $currCamCount = ""
    $currVisualGuardBlocked = ""
    $currVisualGuardReason = ""
    $currQualityGuardBlocked = $false
    $currQualityGuardReason = ""
    $currCandidateInvalidReason = ""
    $currPrecomputeSource = ""
    $currPrecomputeSourceRequested = ""
    $currPrecomputeSourceResolved = ""
    $currPrecomputeFallbackUsed = $false
    $currPrecomputeTimeoutHit = $false
    $currPrecomputeMvSupportOn = ""
    $currPointTargetBlendByMvSupport = ""
    $currPointTargetBlendMvRegionMode = ""
    $currPointMvDepthRegionMode = ""
    $currUseFgMask = ""
    $currFgMaskSource = ""
    if ($CurrentStages.Count -gt 0) {
        $currStageName = [string]$CurrentStages[$CurrentStages.Count - 1].stage
        $currEvalNumSrcViews = [string]$CurrentStages[$CurrentStages.Count - 1].eval_num_src_views
        $currEvalNumSrcViewsDeclared = if ($CurrentStages[$CurrentStages.Count - 1].PSObject.Properties["eval_num_src_views_declared"]) { [string]$CurrentStages[$CurrentStages.Count - 1].eval_num_src_views_declared } else { [string]$CurrentStages[$CurrentStages.Count - 1].eval_num_src_views }
        $currEvalNumSrcViewsActual = [string]$CurrentStages[$CurrentStages.Count - 1].eval_num_src_views_actual
        $currEvalNumSrcViewsMismatch = To-BoolLoose -Value $CurrentStages[$CurrentStages.Count - 1].eval_num_src_views_mismatch -Default $false
        $currCamCount = [string]$CurrentStages[$CurrentStages.Count - 1].cam_count_used
        $currVisualGuardBlocked = [string]$CurrentStages[$CurrentStages.Count - 1].visual_guard_blocked
        $currVisualGuardReason = [string]$CurrentStages[$CurrentStages.Count - 1].visual_guard_reason
        $currQualityGuardBlocked = To-BoolLoose -Value $CurrentStages[$CurrentStages.Count - 1].quality_guard_blocked -Default $false
        $currQualityGuardReason = [string]$CurrentStages[$CurrentStages.Count - 1].quality_guard_reason
        $currCandidateInvalidReason = [string]$CurrentStages[$CurrentStages.Count - 1].candidate_invalid_reason
        $currPrecomputeSource = [string]$CurrentStages[$CurrentStages.Count - 1].precompute_source
        $currPrecomputeSourceRequested = [string]$CurrentStages[$CurrentStages.Count - 1].precompute_source_requested
        $currPrecomputeSourceResolved = [string]$CurrentStages[$CurrentStages.Count - 1].precompute_source_resolved
        $currPrecomputeFallbackUsed = To-BoolLoose -Value $CurrentStages[$CurrentStages.Count - 1].precompute_fallback_used -Default $false
        $currPrecomputeTimeoutHit = To-BoolLoose -Value $CurrentStages[$CurrentStages.Count - 1].precompute_timeout_hit -Default $false
        $currPrecomputeMvSupportOn = [string]$CurrentStages[$CurrentStages.Count - 1].precompute_mv_support_on
        $currPointTargetBlendByMvSupport = [string]$CurrentStages[$CurrentStages.Count - 1].point_target_blend_by_mv_support
        $currPointTargetBlendMvRegionMode = [string]$CurrentStages[$CurrentStages.Count - 1].point_target_blend_mv_region_mode
        $currPointMvDepthRegionMode = [string]$CurrentStages[$CurrentStages.Count - 1].point_mv_depth_region_mode
        $currUseFgMask = [string]$CurrentStages[$CurrentStages.Count - 1].use_fg_mask
        $currFgMaskSource = [string]$CurrentStages[$CurrentStages.Count - 1].fg_mask_source
    }
    if ($candidateResultLatest -ne $null) {
        if ($candidateResultLatest.PSObject.Properties["eval_num_src_views"] -and [string]::IsNullOrWhiteSpace($currEvalNumSrcViews)) {
            $currEvalNumSrcViews = [string]$candidateResultLatest.eval_num_src_views
        }
        if ($candidateResultLatest.PSObject.Properties["eval_num_src_views_declared"] -and [string]::IsNullOrWhiteSpace($currEvalNumSrcViewsDeclared)) {
            $currEvalNumSrcViewsDeclared = [string]$candidateResultLatest.eval_num_src_views_declared
        }
        if ($candidateResultLatest.PSObject.Properties["eval_num_src_views_actual"] -and [string]::IsNullOrWhiteSpace($currEvalNumSrcViewsActual)) {
            $currEvalNumSrcViewsActual = [string]$candidateResultLatest.eval_num_src_views_actual
        }
        if ($candidateResultLatest.PSObject.Properties["eval_num_src_views_mismatch"]) {
            $currEvalNumSrcViewsMismatch = To-BoolLoose -Value $candidateResultLatest.eval_num_src_views_mismatch -Default $currEvalNumSrcViewsMismatch
        }
        if ($candidateResultLatest.PSObject.Properties["visual_guard_blocked"]) {
            $currVisualGuardBlocked = [string]$candidateResultLatest.visual_guard_blocked
        }
        if ($candidateResultLatest.PSObject.Properties["visual_guard_reason"] -and [string]::IsNullOrWhiteSpace($currVisualGuardReason)) {
            $currVisualGuardReason = [string]$candidateResultLatest.visual_guard_reason
        }
        if ($candidateResultLatest.PSObject.Properties["quality_guard_blocked"]) {
            $currQualityGuardBlocked = To-BoolLoose -Value $candidateResultLatest.quality_guard_blocked -Default $currQualityGuardBlocked
        }
        if ($candidateResultLatest.PSObject.Properties["quality_guard_reason"] -and [string]::IsNullOrWhiteSpace($currQualityGuardReason)) {
            $currQualityGuardReason = [string]$candidateResultLatest.quality_guard_reason
        }
        if ($candidateResultLatest.PSObject.Properties["candidate_invalid_reason"] -and [string]::IsNullOrWhiteSpace($currCandidateInvalidReason)) {
            $currCandidateInvalidReason = [string]$candidateResultLatest.candidate_invalid_reason
        }
        if ($candidateResultLatest.PSObject.Properties["precompute_source"] -and [string]::IsNullOrWhiteSpace($currPrecomputeSource)) {
            $currPrecomputeSource = [string]$candidateResultLatest.precompute_source
        }
        if ($candidateResultLatest.PSObject.Properties["precompute_source_requested"] -and [string]::IsNullOrWhiteSpace($currPrecomputeSourceRequested)) {
            $currPrecomputeSourceRequested = [string]$candidateResultLatest.precompute_source_requested
        }
        if ($candidateResultLatest.PSObject.Properties["precompute_source_resolved"] -and [string]::IsNullOrWhiteSpace($currPrecomputeSourceResolved)) {
            $currPrecomputeSourceResolved = [string]$candidateResultLatest.precompute_source_resolved
        }
        if ($candidateResultLatest.PSObject.Properties["precompute_fallback_used"]) {
            $currPrecomputeFallbackUsed = To-BoolLoose -Value $candidateResultLatest.precompute_fallback_used -Default $currPrecomputeFallbackUsed
        }
        if ($candidateResultLatest.PSObject.Properties["precompute_timeout_hit"]) {
            $currPrecomputeTimeoutHit = To-BoolLoose -Value $candidateResultLatest.precompute_timeout_hit -Default $currPrecomputeTimeoutHit
        }
        if ($candidateResultLatest.PSObject.Properties["precompute_mv_support_on"] -and [string]::IsNullOrWhiteSpace($currPrecomputeMvSupportOn)) {
            $currPrecomputeMvSupportOn = [string]$candidateResultLatest.precompute_mv_support_on
        }
        if ($candidateResultLatest.PSObject.Properties["point_target_blend_by_mv_support"] -and [string]::IsNullOrWhiteSpace($currPointTargetBlendByMvSupport)) {
            $currPointTargetBlendByMvSupport = [string]$candidateResultLatest.point_target_blend_by_mv_support
        }
        if ($candidateResultLatest.PSObject.Properties["point_target_blend_mv_region_mode"] -and [string]::IsNullOrWhiteSpace($currPointTargetBlendMvRegionMode)) {
            $currPointTargetBlendMvRegionMode = [string]$candidateResultLatest.point_target_blend_mv_region_mode
        }
        if ($candidateResultLatest.PSObject.Properties["point_mv_depth_region_mode"] -and [string]::IsNullOrWhiteSpace($currPointMvDepthRegionMode)) {
            $currPointMvDepthRegionMode = [string]$candidateResultLatest.point_mv_depth_region_mode
        }
        if ($candidateResultLatest.PSObject.Properties["use_fg_mask"] -and [string]::IsNullOrWhiteSpace($currUseFgMask)) {
            $currUseFgMask = [string]$candidateResultLatest.use_fg_mask
        }
        if ($candidateResultLatest.PSObject.Properties["fg_mask_source"] -and [string]::IsNullOrWhiteSpace($currFgMaskSource)) {
            $currFgMaskSource = [string]$candidateResultLatest.fg_mask_source
        }
    }
    $status = [ordered]@{
        updated_at = $nowText
        deadline = $Deadline.ToString("yyyy-MM-ddTHH:mm:ss")
        current_cycle = $CurrentCycle
        current_stage = $currStageName
        interim = $true
        active_lane = $ActiveLane
        lane_a_best = $LaneABest
        lane_b_best = $LaneBBest
        guard_tier = $GuardTier
        visual_guard_tier = $(if (To-BoolLoose -Value $currVisualGuardBlocked -Default $false) { "blocked" } else { "pass_or_na" })
        visual_guard_reason = $currVisualGuardReason
        decision_reason = $DecisionReason
        rollback_reason = $RollbackReason
        active_eval_num_src_views = $currEvalNumSrcViews
        active_eval_num_src_views_declared = $currEvalNumSrcViewsDeclared
        active_eval_num_src_views_actual = $currEvalNumSrcViewsActual
        active_eval_num_src_views_mismatch = $currEvalNumSrcViewsMismatch
        active_cam_count = $currCamCount
        active_quality_guard_blocked = $currQualityGuardBlocked
        active_quality_guard_reason = $currQualityGuardReason
        active_candidate_invalid_reason = $currCandidateInvalidReason
        active_precompute_source = $currPrecomputeSource
        active_precompute_source_requested = $currPrecomputeSourceRequested
        active_precompute_source_resolved = $currPrecomputeSourceResolved
        active_precompute_fallback_used = $currPrecomputeFallbackUsed
        active_precompute_timeout_hit = $currPrecomputeTimeoutHit
        active_precompute_mv_support_on = $currPrecomputeMvSupportOn
        active_point_target_blend_by_mv_support = $currPointTargetBlendByMvSupport
        active_point_target_blend_mv_region_mode = $currPointTargetBlendMvRegionMode
        active_point_mv_depth_region_mode = $currPointMvDepthRegionMode
        active_use_fg_mask = $currUseFgMask
        active_fg_mask_source = $currFgMaskSource
        active_candidate_result_json = $(if ($candidateResultLatest -ne $null) { "logs/modal_phase5/candidate_result_latest.json" } else { "" })
        current_resume_ckpt = $CurrResume
        current_pseudo_geom = $CurrPseudo
        global_best_ghost = $GlobalBestGhost
        global_best_psnr = $GlobalBestPsnr
        global_best_ssim = $GlobalBestSsim
        global_best_wl1 = $GlobalBestWl1
        next_cycle_tune_action = $PendingTuneAction
        recent_history = @($History | Select-Object -Last 3)
    }
    Write-JsonNoBom -Path "logs/modal_phase5/overnight_ghost_autoloop_latest.json" -Obj $status

    $md = @()
    $md += "# 过夜 Ghost AutoLoop（interim）"
    $md += ""
    $md += "- updated: $($status.updated_at)"
    $md += "- deadline: $($status.deadline)"
    $md += "- cycle: $($status.current_cycle)"
    $md += "- stage: $($status.current_stage)"
    $md += "- active_eval_num_src_views: $($status.active_eval_num_src_views)"
    $md += "- active_eval_num_src_views_declared: $($status.active_eval_num_src_views_declared)"
    $md += "- active_eval_num_src_views_actual: $($status.active_eval_num_src_views_actual)"
    $md += "- active_eval_num_src_views_mismatch: $($status.active_eval_num_src_views_mismatch)"
    $md += "- active_cam_count: $($status.active_cam_count)"
    $md += "- active_quality_guard_blocked: $($status.active_quality_guard_blocked)"
    if (-not [string]::IsNullOrWhiteSpace([string]$status.active_quality_guard_reason)) {
        $md += "- active_quality_guard_reason: $($status.active_quality_guard_reason)"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$status.active_candidate_invalid_reason)) {
        $md += "- active_candidate_invalid_reason: $($status.active_candidate_invalid_reason)"
    }
    $md += "- active_precompute_source: $($status.active_precompute_source)"
    $md += "- active_precompute_source_requested: $($status.active_precompute_source_requested)"
    $md += "- active_precompute_source_resolved: $($status.active_precompute_source_resolved)"
    $md += "- active_precompute_fallback_used: $($status.active_precompute_fallback_used)"
    $md += "- active_precompute_timeout_hit: $($status.active_precompute_timeout_hit)"
    $md += "- active_precompute_mv_support_on: $($status.active_precompute_mv_support_on)"
    $md += "- active_point_target_blend_by_mv_support: $($status.active_point_target_blend_by_mv_support)"
    $md += "- active_point_target_blend_mv_region_mode: $($status.active_point_target_blend_mv_region_mode)"
    $md += "- active_point_mv_depth_region_mode: $($status.active_point_mv_depth_region_mode)"
    $md += "- active_use_fg_mask: $($status.active_use_fg_mask)"
    $md += "- active_fg_mask_source: $($status.active_fg_mask_source)"
    $md += "- visual_guard_tier: $($status.visual_guard_tier)"
    if (-not [string]::IsNullOrWhiteSpace([string]$status.visual_guard_reason)) {
        $md += "- visual_guard_reason: $($status.visual_guard_reason)"
    }
    $md += "- global_best_ghost: $($status.global_best_ghost)"
    $md += "- global_best_psnr: $($status.global_best_psnr)"
    $md += "- global_best_ssim: $($status.global_best_ssim)"
    $md += "- global_best_wl1: $($status.global_best_wl1)"
    $md += "- next_cycle_tune_action: $($status.next_cycle_tune_action)"
    $md += "- note: cycle 未完成，当前为阶段级中间落盘。"
    Set-Content -Path "logs/modal_phase5/overnight_ghost_autoloop_latest.md" -Value ($md -join "`n") -Encoding UTF8
}

function Safe-WriteInterimAutoloopArtifacts(
    [object[]]$History,
    [int]$CurrentCycle,
    [object[]]$CurrentStages,
    [datetime]$Deadline,
    [double]$GlobalBestGhost,
    [double]$GlobalBestPsnr,
    [double]$GlobalBestSsim,
    [double]$GlobalBestWl1,
    [string]$CurrResume,
    [string]$CurrPseudo,
    [string]$PendingTuneAction,
    [string]$ActiveLane = "lane_a",
    $LaneABest = $null,
    $LaneBBest = $null,
    [string]$GuardTier = "",
    [string]$DecisionReason = "",
    [string]$RollbackReason = ""
) {
    try {
        Write-InterimAutoloopArtifacts `
            -History $History `
            -CurrentCycle $CurrentCycle `
            -CurrentStages $CurrentStages `
            -Deadline $Deadline `
            -GlobalBestGhost $GlobalBestGhost `
            -GlobalBestPsnr $GlobalBestPsnr `
            -GlobalBestSsim $GlobalBestSsim `
            -GlobalBestWl1 $GlobalBestWl1 `
            -CurrResume $CurrResume `
            -CurrPseudo $CurrPseudo `
            -PendingTuneAction $PendingTuneAction `
            -ActiveLane $ActiveLane `
            -LaneABest $LaneABest `
            -LaneBBest $LaneBBest `
            -GuardTier $GuardTier `
            -DecisionReason $DecisionReason `
            -RollbackReason $RollbackReason
    } catch {
        $stageName = ""
        if ($null -ne $CurrentStages -and $CurrentStages.Count -gt 0) {
            $stageName = [string]$CurrentStages[$CurrentStages.Count - 1].stage
        }
        Write-Host "[autoloop][warn] interim artifact write failed cycle=$CurrentCycle stage=$stageName msg=$($_.Exception.Message)"
    }
}

function Apply-NoImproveSingleStep(
    [int]$StepIndex,
    [int]$NoSubstantialImproveCycles = 1,
    [double]$GhostLag = [double]::NaN
) {
    $slot = [Math]::Abs([int]$StepIndex) % 6
    $pressure = 0
    if ($NoSubstantialImproveCycles -ge 4) {
        $pressure = 2
    } elseif ($NoSubstantialImproveCycles -ge 2) {
        $pressure = 1
    }
    if ((-not [double]::IsNaN($GhostLag)) -and ($GhostLag -ge 0.2)) {
        $pressure = [Math]::Min(2, $pressure + 1)
    }

    switch ($slot) {
        0 {
            $old = [double]$script:BasePointMvMaskSoftMix
            # 优先压低 soft_mix，停滞期允许探索到 0（近似关闭软混合）。
            $delta = if ($pressure -ge 2) { 0.10 } elseif ($pressure -ge 1) { 0.07 } else { 0.05 }
            $minMix = if ($pressure -ge 2) { 0.00 } elseif ($pressure -ge 1) { 0.10 } else { 0.20 }
            $next = [Math]::Round([Math]::Max($minMix, $old - $delta), 2)
            if ($next -eq $old) {
                $next = [Math]::Round([Math]::Min(0.45, $old + 0.03), 2)
            }
            $script:BasePointMvMaskSoftMix = $next
            return "point_mv_mask_soft_mix: $old -> $($script:BasePointMvMaskSoftMix) (pressure=$pressure)"
        }
        1 {
            $old = [double]$script:BasePointMvMaskSoftHitThr
            $maxThr = if ($pressure -ge 2) { 0.72 } elseif ($pressure -ge 1) { 0.70 } else { 0.68 }
            $minThr = if ($pressure -ge 2) { 0.35 } else { 0.42 }
            $upStep = if ($pressure -ge 2) { 0.05 } elseif ($pressure -ge 1) { 0.04 } else { 0.03 }
            $downStep = if ($pressure -ge 2) { 0.06 } else { 0.04 }
            if ($old -lt 0.62) {
                $script:BasePointMvMaskSoftHitThr = [Math]::Round([Math]::Min($maxThr, $old + $upStep), 2)
            } elseif ($old -lt $maxThr) {
                $script:BasePointMvMaskSoftHitThr = [Math]::Round([Math]::Min($maxThr, $old + 0.02), 2)
            } else {
                $script:BasePointMvMaskSoftHitThr = [Math]::Round([Math]::Max($minThr, $old - $downStep), 2)
            }
            return "point_mv_mask_soft_hit_thr: $old -> $($script:BasePointMvMaskSoftHitThr) (pressure=$pressure)"
        }
        2 {
            $old = [int]$script:BasePointMvStride
            if ($pressure -ge 1) {
                # Avoid no-op loops (1->1). Under pressure, keep exploring in {1,2}.
                if ($old -le 1) { $script:BasePointMvStride = 2 } else { $script:BasePointMvStride = 1 }
            } else {
                if ($old -le 1) { $script:BasePointMvStride = 2 } else { $script:BasePointMvStride = 1 }
            }
            return "point_mv_stride: $old -> $($script:BasePointMvStride) (pressure=$pressure)"
        }
        3 {
            $old = [int]$script:BasePointMvDepthMaxPairs
            if ($pressure -ge 2) {
                if ($old -ge 5) { $script:BasePointMvDepthMaxPairs = 2 } else { $script:BasePointMvDepthMaxPairs = $old + 1 }
            } else {
                if ($old -ge 4) { $script:BasePointMvDepthMaxPairs = 2 } else { $script:BasePointMvDepthMaxPairs = $old + 1 }
            }
            return "point_mv_depth_max_pairs: $old -> $($script:BasePointMvDepthMaxPairs) (pressure=$pressure)"
        }
        4 {
            $oldMode = [string]$script:BasePointMvDepthSupportMode
            $oldFloor = [double]$script:BasePointMvDepthSupportFloor
            $targetFloor = if ($pressure -ge 2) { 0.15 } else { 0.1 }
            if ($oldMode -eq "off") {
                $script:BasePointMvDepthSupportMode = "direct"
                $script:BasePointMvDepthSupportFloor = $targetFloor
            } elseif ($oldMode -eq "direct" -and $oldFloor -lt $targetFloor) {
                $script:BasePointMvDepthSupportFloor = $targetFloor
            } elseif ($oldMode -eq "direct") {
                $script:BasePointMvDepthSupportMode = "inverse"
                $script:BasePointMvDepthSupportFloor = $targetFloor
            } elseif (($oldMode -eq "inverse") -and ($pressure -ge 2) -and ($oldFloor -lt 0.2)) {
                $script:BasePointMvDepthSupportFloor = 0.2
            } else {
                $script:BasePointMvDepthSupportMode = "off"
                $script:BasePointMvDepthSupportFloor = 0.0
            }
            return "point_mv_depth_support_mode/floor: $oldMode/$oldFloor -> $($script:BasePointMvDepthSupportMode)/$($script:BasePointMvDepthSupportFloor) (pressure=$pressure)"
        }
        default {
            $old = [double]$script:BasePointMvMaskMinTgtFgRatio
            $step = if ($pressure -ge 2) { 0.02 } else { 0.01 }
            $maxVal = if ($pressure -ge 2) { 0.08 } else { 0.05 }
            if ($old -lt $maxVal) {
                $script:BasePointMvMaskMinTgtFgRatio = [Math]::Round([Math]::Min($maxVal, $old + $step), 2)
            } else {
                $script:BasePointMvMaskMinTgtFgRatio = 0.0
            }
            return "point_mv_mask_min_tgt_fg_ratio: $old -> $($script:BasePointMvMaskMinTgtFgRatio) (pressure=$pressure)"
        }
    }
}

function Get-TuneState() {
    return [pscustomobject]@{
        BasePointMvMaskSoftMix = [double]$script:BasePointMvMaskSoftMix
        BasePointMvMaskSoftHitThr = [double]$script:BasePointMvMaskSoftHitThr
        BasePointMvStride = [int]$script:BasePointMvStride
        BasePointMvDepthMaxPairs = [int]$script:BasePointMvDepthMaxPairs
        BasePointMvDepthSupportMode = [string]$script:BasePointMvDepthSupportMode
        BasePointMvDepthSupportFloor = [double]$script:BasePointMvDepthSupportFloor
        BasePointMvMaskMinTgtFgRatio = [double]$script:BasePointMvMaskMinTgtFgRatio
    }
}

function Set-TuneState([object]$State) {
    if ($null -eq $State) { return }
    $script:BasePointMvMaskSoftMix = [double]$State.BasePointMvMaskSoftMix
    $script:BasePointMvMaskSoftHitThr = [double]$State.BasePointMvMaskSoftHitThr
    $script:BasePointMvStride = [int]$State.BasePointMvStride
    $script:BasePointMvDepthMaxPairs = [int]$State.BasePointMvDepthMaxPairs
    $script:BasePointMvDepthSupportMode = [string]$State.BasePointMvDepthSupportMode
    $script:BasePointMvDepthSupportFloor = [double]$State.BasePointMvDepthSupportFloor
    $script:BasePointMvMaskMinTgtFgRatio = [double]$State.BasePointMvMaskMinTgtFgRatio
}

function Parse-DateTimeOrNull([string]$Text) {
    if ([string]::IsNullOrWhiteSpace($Text)) { return $null }
    try {
        return [datetime]::Parse($Text)
    } catch {
        return $null
    }
}

function Clamp-DoubleValue([object]$Value, [double]$Min, [double]$Max, [double]$Fallback) {
    $v = To-DoubleOrNaN($Value)
    if ([double]::IsNaN($v)) { return $Fallback }
    if ($v -lt $Min) { return $Min }
    if ($v -gt $Max) { return $Max }
    return $v
}

function Clamp-IntValue([object]$Value, [int]$Min, [int]$Max, [int]$Fallback) {
    $v = $Fallback
    try { $v = [int]$Value } catch { $v = $Fallback }
    if ($v -lt $Min) { return $Min }
    if ($v -gt $Max) { return $Max }
    return $v
}

function Resolve-PersistentCycleState(
    [string]$Path,
    [int]$MaxAgeHours = 36
) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
    if (-not (Test-Path $Path)) { return $null }
    try {
        $raw = Get-Content $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return $null
    }
    if ($null -eq $raw) { return $null }

    $updatedAt = Parse-DateTimeOrNull ([string]$raw.updated_at)
    if (($MaxAgeHours -gt 0) -and ($null -ne $updatedAt)) {
        $ageHours = ((Get-Date) - $updatedAt).TotalHours
        if ($ageHours -gt [double]$MaxAgeHours) {
            return $null
        }
    }

    $routeMode = [string]$raw.route_mode
    if ($routeMode -ne "ab_validation") { $routeMode = "main" }
    $abVariant = [string]$raw.ab_route_variant
    if (($abVariant -ne "balance") -and ($abVariant -ne "aggressive")) { $abVariant = "balance" }
    $pendingTune = [string]$raw.pending_tune_action
    if ([string]::IsNullOrWhiteSpace($pendingTune)) { $pendingTune = "none" }

    return [pscustomobject]@{
        updated_at = [string]$raw.updated_at
        no_improve_cycles = Clamp-IntValue $raw.no_improve_cycles 0 9999 0
        no_substantial_improve_cycles = Clamp-IntValue $raw.no_substantial_improve_cycles 0 9999 0
        tune_step = Clamp-IntValue $raw.tune_step 0 100000 0
        regress_cycles = Clamp-IntValue $raw.regress_cycles 0 9999 0
        pending_tune_action = $pendingTune
        route_mode = $routeMode
        ab_route_executed = [bool]$raw.ab_route_executed
        ab_route_variant = $abVariant
        aggressive_route_cooldown_cycles = Clamp-IntValue $raw.aggressive_route_cooldown_cycles 0 9999 0
        base_point_mv_mask_soft_mix = Clamp-DoubleValue $raw.base_point_mv_mask_soft_mix 0.0 1.0 ([double]$BasePointMvMaskSoftMix)
        base_point_mv_mask_soft_hit_thr = Clamp-DoubleValue $raw.base_point_mv_mask_soft_hit_thr 0.0 1.0 ([double]$BasePointMvMaskSoftHitThr)
        base_point_mv_stride = Clamp-IntValue $raw.base_point_mv_stride 1 8 ([int]$BasePointMvStride)
        base_point_mv_depth_max_pairs = Clamp-IntValue $raw.base_point_mv_depth_max_pairs 1 8 ([int]$BasePointMvDepthMaxPairs)
        base_point_mv_depth_support_mode = if ([string]::IsNullOrWhiteSpace([string]$raw.base_point_mv_depth_support_mode)) { [string]$BasePointMvDepthSupportMode } else { [string]$raw.base_point_mv_depth_support_mode }
        base_point_mv_depth_support_floor = Clamp-DoubleValue $raw.base_point_mv_depth_support_floor 0.0 1.0 ([double]$BasePointMvDepthSupportFloor)
        base_point_mv_mask_min_tgt_fg_ratio = Clamp-DoubleValue $raw.base_point_mv_mask_min_tgt_fg_ratio 0.0 1.0 ([double]$BasePointMvMaskMinTgtFgRatio)
    }
}

function Write-PersistentCycleState(
    [string]$Path,
    [object]$State
) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return }
    if ($null -eq $State) { return }
    try {
        Write-JsonNoBom -Path $Path -Obj $State
    } catch {
        Write-Host "[autoloop][warn] write persistent cycle state failed path=$Path msg=$($_.Exception.Message)"
    }
}

function Resolve-NextStageResumeCkpt(
    [object]$PrevStage,
    [string]$FallbackResume,
    [double]$GlobalBestGhostRef,
    [double]$PromoteGhostMargin
) {
    if ($null -eq $PrevStage) { return $FallbackResume }
    if (-not (Test-StageHasUsableResume -Stage $PrevStage)) { return $FallbackResume }
    $cand = [string]$PrevStage.best_ckpt
    if ([string]::IsNullOrWhiteSpace($cand)) { return $FallbackResume }

    $gPrev = To-DoubleOrNaN($PrevStage.ghost)
    $gRef = To-DoubleOrNaN($GlobalBestGhostRef)
    if ((-not [double]::IsNaN($gPrev)) -and (-not [double]::IsNaN($gRef))) {
        if ($gPrev -gt ($gRef + [Math]::Max(0.0, $PromoteGhostMargin))) {
            return $FallbackResume
        }
    }
    return $cand
}

function Get-BestStageByGhost([object[]]$Stages) {
    return @(
        $Stages |
            Where-Object { Test-StageHasUsableGhost -Stage $_ } |
            Sort-Object {
                $g = To-DoubleOrNaN($_.ghost)
                if ([double]::IsNaN($g)) { return [double]::PositiveInfinity }
                return $g
            }, {
                -1.0 * (To-DoubleOrNaN($_.psnr))
            }
    ) | Select-Object -First 1
}

function Test-StageInfraNoOutputFailure(
    [string]$RawSweepCsvPath,
    [int]$MinConsecutive = 2
) {
    if ([string]::IsNullOrWhiteSpace($RawSweepCsvPath)) { return $false }
    if (-not (Test-Path $RawSweepCsvPath)) { return $false }
    $thr = [Math]::Max(1, [int]$MinConsecutive)
    try {
        $rows = @(Import-Csv $RawSweepCsvPath)
        if ($rows.Count -le 0) { return $false }
        foreach ($r in $rows) {
            $streak = 0
            try { $streak = [int]$r.infra_no_output_consecutive } catch { $streak = 0 }
            $reason = [string]$r.ft_failure_reason
            $rc = 0
            try { $rc = [int]$r.exit_code } catch { $rc = 0 }
            if ($streak -ge $thr) { return $true }
            if (($rc -ne 0) -and (-not [string]::IsNullOrWhiteSpace($reason)) -and ($reason -match "(?i)(?:heartbeat_stall_timeout|no_output_timeout)_\d+s")) {
                return $true
            }
        }
    } catch {
        return $false
    }
    return $false
}

function Resolve-PointMvDepthPairMode(
    [string]$Raw,
    [string]$Default = "adjacent"
) {
    $valid = @("sequential", "adjacent", "farthest", "random")
    $fallback = if ([string]::IsNullOrWhiteSpace($Default)) { "adjacent" } else { ([string]$Default).Trim().ToLowerInvariant() }
    if (-not ($valid -contains $fallback)) { $fallback = "adjacent" }
    if ([string]::IsNullOrWhiteSpace($Raw)) { return $fallback }
    $mode = ([string]$Raw).Trim().ToLowerInvariant()
    if ($valid -contains $mode) { return $mode }
    return $fallback
}

function New-SkippedStageResult(
    [string]$StageName,
    [string]$PointTargetBlendMvPolicy,
    [string]$PointmapSource,
    [string]$PseudoGeomSubdir,
    [string]$ResumeCkpt,
    [string]$Reason,
    [hashtable]$Overrides = @{}
) {
    $stageLambdaPointMvDepthList = [string](Resolve-StageOverride $Overrides "LambdaPointMvDepthList" $LambdaPointMvDepthList)
    $stageLambdaPointMvMaskList = [string](Resolve-StageOverride $Overrides "LambdaPointMvMaskList" $LambdaPointMvMaskList)
    $stagePointTargetMode = [string](Resolve-StageOverride $Overrides "PointTargetMode" $PointTargetMode)
    $stagePointTargetConsensusAlphaFloor = [double](Resolve-StageOverride $Overrides "PointTargetConsensusAlphaFloor" $PointTargetConsensusAlphaFloor)
    $stageLambdaPoint = [double](Resolve-StageOverride $Overrides "LambdaPoint" $LambdaPoint)
    $stagePointMvMaskHitThr = [double](Resolve-StageOverride $Overrides "PointMvMaskHitThr" $BasePointMvMaskHitThr)
    $stagePointMvMaskMinTgtFgRatio = [double](Resolve-StageOverride $Overrides "PointMvMaskMinTgtFgRatio" $BasePointMvMaskMinTgtFgRatio)
    $stagePointMvMaskSoftMix = [double](Resolve-StageOverride $Overrides "PointMvMaskSoftMix" $BasePointMvMaskSoftMix)
    $stagePointMvMaskSoftHitThr = [double](Resolve-StageOverride $Overrides "PointMvMaskSoftHitThr" $BasePointMvMaskSoftHitThr)
    $stagePointMvStride = [int](Resolve-StageOverride $Overrides "PointMvStride" $BasePointMvStride)
    $stagePointMvDepthMaxPairs = [int](Resolve-StageOverride $Overrides "PointMvDepthMaxPairs" $BasePointMvDepthMaxPairs)
    $stagePointMvDepthPairMode = Resolve-PointMvDepthPairMode `
        -Raw ([string](Resolve-StageOverride $Overrides "PointMvDepthPairMode" $BasePointMvDepthPairMode)) `
        -Default ([string]$BasePointMvDepthPairMode)
    $stagePointMvDepthRegionMode = [string](Resolve-StageOverride $Overrides "PointMvDepthRegionMode" $BasePointMvDepthRegionMode)
    $stagePointMvDepthSupportMode = [string](Resolve-StageOverride $Overrides "PointMvDepthSupportMode" $BasePointMvDepthSupportMode)
    $stagePointMvDepthSupportFloor = [double](Resolve-StageOverride $Overrides "PointMvDepthSupportFloor" $BasePointMvDepthSupportFloor)
    $stagePointMvMaskSupportMode = [string](Resolve-StageOverride $Overrides "PointMvMaskSupportMode" $BasePointMvMaskSupportMode)
    $stagePointMvMaskSupportFloor = [double](Resolve-StageOverride $Overrides "PointMvMaskSupportFloor" $BasePointMvMaskSupportFloor)
    $stageConfWeightPerViewQuantile = [double](Resolve-StageOverride $Overrides "ConfWeightPerViewQuantile" $BaseConfWeightPerViewQuantile)
    $stageConfWeightPerViewMinValid = [int](Resolve-StageOverride $Overrides "ConfWeightPerViewMinValid" $BaseConfWeightPerViewMinValid)
    $stageLambdaPointNormalConsis = [double](Resolve-StageOverride $Overrides "LambdaPointNormalConsis" $BaseLambdaPointNormalConsis)
    $stagePointNormalConsisWarmupSteps = [int](Resolve-StageOverride $Overrides "PointNormalConsisWarmupSteps" $BasePointNormalConsisWarmupSteps)
    $stagePointConsFocus = [string](Resolve-StageOverride $Overrides "PointConsFocus" $BasePointConsFocus)
    $stagePointResidualFocus = [string](Resolve-StageOverride $Overrides "PointResidualFocus" $BasePointResidualFocus)
    $stageLaneId = [string](Resolve-StageOverride $Overrides "LaneId" "lane_a")
    $stageCandidateFamily = [string](Resolve-StageOverride $Overrides "CandidateFamily" "")
    $stageGuardTier = [string](Resolve-StageOverride $Overrides "GuardTier" "")
    $stageRollbackTriggered = [bool](Resolve-StageOverride $Overrides "RollbackTriggered" $false)
    $stageUseFgMask = [string](Resolve-StageOverride $Overrides "UseFgMask" $BaseUseFgMask)
    $stageFgMaskSource = [string](Resolve-StageOverride $Overrides "FgMaskSource" $BaseFgMaskSource)
    $stagePointTargetBlendMvRegionMode = [string](Resolve-StageOverride $Overrides "PointTargetBlendMvRegionMode" $BasePointTargetBlendMvRegionMode)
    $stageModalRunQuiet = To-BoolLoose -Value (Resolve-StageOverride $Overrides "ModalRunQuiet" $ModalRunQuiet) -Default $ModalRunQuiet
    Write-Host "[autoloop] stage=$StageName skipped reason=$Reason"
    return [pscustomobject]@{
        stage = $StageName
        policy = $PointTargetBlendMvPolicy
        pointmap_source = $PointmapSource
        lane_id = $stageLaneId
        candidate_family = $stageCandidateFamily
        guard_tier = $stageGuardTier
        rollback_triggered = $stageRollbackTriggered
        stage_lambda_point_mv_depth_list = $stageLambdaPointMvDepthList
        stage_lambda_point_mv_mask_list = $stageLambdaPointMvMaskList
        stage_point_target_mode = $stagePointTargetMode
        stage_point_target_consensus_alpha_floor = $stagePointTargetConsensusAlphaFloor
        stage_lambda_point = $stageLambdaPoint
        stage_point_mv_mask_hit_thr = $stagePointMvMaskHitThr
        stage_point_mv_mask_min_tgt_fg_ratio = $stagePointMvMaskMinTgtFgRatio
        stage_point_mv_mask_soft_mix = $stagePointMvMaskSoftMix
        stage_point_mv_mask_soft_hit_thr = $stagePointMvMaskSoftHitThr
        stage_point_mv_stride = $stagePointMvStride
        stage_point_mv_depth_max_pairs = $stagePointMvDepthMaxPairs
        stage_point_mv_depth_pair_mode = $stagePointMvDepthPairMode
        stage_point_mv_depth_region_mode = $stagePointMvDepthRegionMode
        stage_point_mv_depth_support_mode = $stagePointMvDepthSupportMode
        stage_point_mv_depth_support_floor = $stagePointMvDepthSupportFloor
        stage_point_mv_mask_support_mode = $stagePointMvMaskSupportMode
        stage_point_mv_mask_support_floor = $stagePointMvMaskSupportFloor
        stage_use_fg_mask = $stageUseFgMask
        stage_fg_mask_source = $stageFgMaskSource
        stage_point_target_blend_mv_region_mode = $stagePointTargetBlendMvRegionMode
        stage_conf_weight_per_view_quantile = $stageConfWeightPerViewQuantile
        stage_conf_weight_per_view_min_valid = $stageConfWeightPerViewMinValid
        stage_lambda_point_normal_consis = $stageLambdaPointNormalConsis
        stage_point_normal_consis_warmup_steps = $stagePointNormalConsisWarmupSteps
        stage_point_cons_focus = $stagePointConsFocus
        stage_point_residual_focus = $stagePointResidualFocus
        rc = 0
        ghost = [double]::NaN
        psnr = [double]::NaN
        ssim = [double]::NaN
        wl1 = [double]::NaN
        best_label = ""
        best_geom = $PseudoGeomSubdir
        best_ckpt = $ResumeCkpt
        best_lambda_point_mv_depth = ""
        best_lambda_point_mv_mask = ""
        best_ghost_rows_csv = ""
        best_visual_png = ""
        stage_best_strip_png = ""
        best_ghost_width_ratio = [double]::NaN
        best_ghost_area_ratio = [double]::NaN
        best_ghost_peak_count = [double]::NaN
        best_ghost_center_offset_ratio = [double]::NaN
        ghost_soft_score = [double]::NaN
        sweep_csv = ""
        sweep_md = ""
        raw_sweep_csv = ""
        pseudo_geom_in = $PseudoGeomSubdir
        resume_ckpt_in = $ResumeCkpt
        stage_skip_reason = $Reason
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    }
}

function Resolve-StageOverride(
    [hashtable]$Overrides,
    [string]$Key,
    $DefaultValue
) {
    if ($null -eq $Overrides) { return $DefaultValue }
    if (-not $Overrides.ContainsKey($Key)) { return $DefaultValue }
    $raw = $Overrides[$Key]
    if ($null -eq $raw) { return $DefaultValue }
    if (($raw -is [string]) -and [string]::IsNullOrWhiteSpace($raw)) { return $DefaultValue }
    return $raw
}

function Invoke-GhostStage(
    [string]$StageName,
    [string]$PointTargetBlendMvPolicy,
    [string]$PointmapSource,
    [string]$PseudoGeomSubdir,
    [string]$ResumeCkpt,
    [hashtable]$Overrides = @{}
) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stageLambdaPointMvDepthList = [string](Resolve-StageOverride $Overrides "LambdaPointMvDepthList" $LambdaPointMvDepthList)
    $stageLambdaPointMvMaskList = [string](Resolve-StageOverride $Overrides "LambdaPointMvMaskList" $LambdaPointMvMaskList)
    $stagePointTargetMode = [string](Resolve-StageOverride $Overrides "PointTargetMode" $PointTargetMode)
    $stagePointTargetConsensusAlphaFloor = [double](Resolve-StageOverride $Overrides "PointTargetConsensusAlphaFloor" $PointTargetConsensusAlphaFloor)
    $stageLambdaPoint = [double](Resolve-StageOverride $Overrides "LambdaPoint" $LambdaPoint)
    $stageLambdaPointReproj = [double](Resolve-StageOverride $Overrides "LambdaPointReproj" $LambdaPointReproj)
    $stagePointMvMaskHitThr = [double](Resolve-StageOverride $Overrides "PointMvMaskHitThr" $BasePointMvMaskHitThr)
    $stagePointMvMaskMinTgtFgRatio = [double](Resolve-StageOverride $Overrides "PointMvMaskMinTgtFgRatio" $BasePointMvMaskMinTgtFgRatio)
    $stagePointMvMaskSoftBlurPx = [int](Resolve-StageOverride $Overrides "PointMvMaskSoftBlurPx" $BasePointMvMaskSoftBlurPx)
    $stagePointMvMaskSoftBlurIters = [int](Resolve-StageOverride $Overrides "PointMvMaskSoftBlurIters" $BasePointMvMaskSoftBlurIters)
    $stagePointMvMaskSoftMix = [double](Resolve-StageOverride $Overrides "PointMvMaskSoftMix" $BasePointMvMaskSoftMix)
    $stagePointMvMaskSoftHitThr = [double](Resolve-StageOverride $Overrides "PointMvMaskSoftHitThr" $BasePointMvMaskSoftHitThr)
    $stagePointMvStride = [int](Resolve-StageOverride $Overrides "PointMvStride" $BasePointMvStride)
    $stagePointMvDepthMaxPairs = [int](Resolve-StageOverride $Overrides "PointMvDepthMaxPairs" $BasePointMvDepthMaxPairs)
    $stagePointMvDepthRegionMode = [string](Resolve-StageOverride $Overrides "PointMvDepthRegionMode" $BasePointMvDepthRegionMode)
    $stagePointMvDepthSupportMode = [string](Resolve-StageOverride $Overrides "PointMvDepthSupportMode" $BasePointMvDepthSupportMode)
    $stagePointMvDepthSupportFloor = [double](Resolve-StageOverride $Overrides "PointMvDepthSupportFloor" $BasePointMvDepthSupportFloor)
    $stagePointMvDepthPairMode = Resolve-PointMvDepthPairMode `
        -Raw ([string](Resolve-StageOverride $Overrides "PointMvDepthPairMode" $BasePointMvDepthPairMode)) `
        -Default ([string]$BasePointMvDepthPairMode)
    $stagePointMvMaskSupportMode = [string](Resolve-StageOverride $Overrides "PointMvMaskSupportMode" $BasePointMvMaskSupportMode)
    $stagePointMvMaskSupportFloor = [double](Resolve-StageOverride $Overrides "PointMvMaskSupportFloor" $BasePointMvMaskSupportFloor)
    $stageConfWeightPerViewQuantile = [double](Resolve-StageOverride $Overrides "ConfWeightPerViewQuantile" $BaseConfWeightPerViewQuantile)
    $stageConfWeightPerViewMinValid = [int](Resolve-StageOverride $Overrides "ConfWeightPerViewMinValid" $BaseConfWeightPerViewMinValid)
    $stageLambdaPointNormalConsis = [double](Resolve-StageOverride $Overrides "LambdaPointNormalConsis" $BaseLambdaPointNormalConsis)
    $stagePointNormalConsisWarmupSteps = [int](Resolve-StageOverride $Overrides "PointNormalConsisWarmupSteps" $BasePointNormalConsisWarmupSteps)
    $stagePointLossFgErodePx = [int](Resolve-StageOverride $Overrides "PointLossFgErodePx" $BasePointLossFgErodePx)
    $stagePointMvDepthFgErodePx = [int](Resolve-StageOverride $Overrides "PointMvDepthFgErodePx" $BasePointMvDepthFgErodePx)
    $stagePointConsQuantile = [double](Resolve-StageOverride $Overrides "PointConsQuantile" $BasePointConsQuantile)
    $stagePointConsFocus = [string](Resolve-StageOverride $Overrides "PointConsFocus" $BasePointConsFocus)
    $stagePointResidualQuantile = [double](Resolve-StageOverride $Overrides "PointResidualQuantile" $BasePointResidualQuantile)
    $stagePointResidualFocus = [string](Resolve-StageOverride $Overrides "PointResidualFocus" $BasePointResidualFocus)
    $stagePointResidualBoost = [double](Resolve-StageOverride $Overrides "PointResidualBoost" $BasePointResidualBoost)
    $stagePointMvDepthOutlierBoost = [double](Resolve-StageOverride $Overrides "PointMvDepthOutlierBoost" $BasePointMvDepthOutlierBoost)
    $stageSupervisionWeightMode = [string](Resolve-StageOverride $Overrides "SupervisionWeightMode" $BaseSupervisionWeightMode)
    $stagePointTargetBlendByMvSupport = [string](Resolve-StageOverride $Overrides "PointTargetBlendByMvSupport" "on")
    $stageUseFgMask = [string](Resolve-StageOverride $Overrides "UseFgMask" $BaseUseFgMask)
    $stageFgMaskSource = [string](Resolve-StageOverride $Overrides "FgMaskSource" $BaseFgMaskSource)
    $stagePointTargetBlendMvRegionMode = [string](Resolve-StageOverride $Overrides "PointTargetBlendMvRegionMode" $BasePointTargetBlendMvRegionMode)
    $stageLaneId = [string](Resolve-StageOverride $Overrides "LaneId" "lane_a")
    $stageCandidateFamily = [string](Resolve-StageOverride $Overrides "CandidateFamily" "")
    $stageGuardTier = [string](Resolve-StageOverride $Overrides "GuardTier" "")
    $stageRollbackTriggered = [bool](Resolve-StageOverride $Overrides "RollbackTriggered" $false)
    $stageNoImprovePatience = [int](Resolve-StageOverride $Overrides "NoImprovePatience" $StageNoImprovePatience)
    $stageModalRunNoOutputTimeoutSec = [int](Resolve-StageOverride $Overrides "ModalRunNoOutputTimeoutSec" $ModalRunNoOutputTimeoutSec)
    $stageModalRunNoOutputMaxRetries = [int](Resolve-StageOverride $Overrides "ModalRunNoOutputMaxRetries" $ModalRunNoOutputMaxRetries)
    $stageModalRunQuiet = To-BoolLoose -Value (Resolve-StageOverride $Overrides "ModalRunQuiet" $ModalRunQuiet) -Default $ModalRunQuiet
    $stageInfraNoOutputStopConsecutive = [int](Resolve-StageOverride $Overrides "InfraNoOutputStopConsecutive" $InfraNoOutputStopConsecutive)
    $stageCamNames = [string](Resolve-StageOverride $Overrides "CamNames" $StageCamNames)
    $stageEvalNumSrcViewsList = [string](Resolve-StageOverride $Overrides "EvalNumSrcViewsList" "")
    $stageEnableVisualAntiBlackGuard = To-BoolLoose -Value (Resolve-StageOverride $Overrides "EnableVisualAntiBlackGuard" $EnableVisualAntiBlackGuard) -Default $EnableVisualAntiBlackGuard
    $stageMinPredLumaMean = [double](Resolve-StageOverride $Overrides "MinPredLumaMean" $MinPredLumaMean)
    $stageMinPredNonBlackRatio = [double](Resolve-StageOverride $Overrides "MinPredNonBlackRatio" $MinPredNonBlackRatio)
    $stageMinAreaRatio = [double](Resolve-StageOverride $Overrides "MinAreaRatio" $MinAreaRatio)
    $stageMinWidthRatio = [double](Resolve-StageOverride $Overrides "MinWidthRatio" $MinWidthRatio)
    $stageGramDynEnable = [string](Resolve-StageOverride $Overrides "GramDynEnable" $Stage2GramDynEnable)
    $stageGramDynLayerIdx = [int](Resolve-StageOverride $Overrides "GramDynLayerIdx" $Stage2GramDynLayerIdx)
    $stageGramDynQuantile = [double](Resolve-StageOverride $Overrides "GramDynQuantile" $Stage2GramDynQuantile)
    $stageGramDynWeightFloor = [double](Resolve-StageOverride $Overrides "GramDynWeightFloor" $Stage2GramDynWeightFloor)
    $stageGramDynWarmupSteps = [int](Resolve-StageOverride $Overrides "GramDynWarmupSteps" $Stage2GramDynWarmupSteps)
    $stageDynProxyEnable = [string](Resolve-StageOverride $Overrides "DynProxyEnable" $Stage2DynProxyEnable)
    $stageDynProxyMode = [string](Resolve-StageOverride $Overrides "DynProxyMode" $Stage2DynProxyMode)
    $stageDynProxyUseGram = [string](Resolve-StageOverride $Overrides "DynProxyUseGram" $Stage2DynProxyUseGram)
    $stageDynProxyUseSupport = [string](Resolve-StageOverride $Overrides "DynProxyUseSupport" $Stage2DynProxyUseSupport)
    $stageDynProxyFloor = [double](Resolve-StageOverride $Overrides "DynProxyFloor" $Stage2DynProxyFloor)
    $stageDynProxyWarmupSteps = [int](Resolve-StageOverride $Overrides "DynProxyWarmupSteps" $Stage2DynProxyWarmupSteps)
    # Stage2-specific ablation switch should not implicitly spill over to stage1/3/4/5.
    $stageEnableAnySplatAblationSixPack = To-BoolLoose -Value (Resolve-StageOverride $Overrides "EnableAnySplatAblationSixPack" $false) -Default $false
    $stageEnableExtendedCkptWaitOnMissing = To-BoolLoose -Value (Resolve-StageOverride $Overrides "EnableExtendedCkptWaitOnMissing" $Stage2EnableExtendedCkptWaitOnMissing) -Default $Stage2EnableExtendedCkptWaitOnMissing
    $stageCkptExtendedWaitTimeoutSec = [int](Resolve-StageOverride $Overrides "CkptExtendedWaitTimeoutSec" $Stage2CkptExtendedWaitTimeoutSec)
    $stageEnableResumeCkptFallbackOnShortCkptMissing = To-BoolLoose -Value (Resolve-StageOverride $Overrides "EnableResumeCkptFallbackOnShortCkptMissing" $Stage2EnableResumeCkptFallbackOnShortCkptMissing) -Default $Stage2EnableResumeCkptFallbackOnShortCkptMissing
    $stageDisallowResumeFallbackResult = To-BoolLoose -Value (Resolve-StageOverride $Overrides "DisallowResumeFallbackResult" $Stage2DisallowResumeFallbackResult) -Default $Stage2DisallowResumeFallbackResult
    $stagePrecomputeMvSupportOn = [string](Resolve-StageOverride $Overrides "PrecomputeMvSupportOn" "on")
    $stagePrecomputeMvSupportRegionMode = [string](Resolve-StageOverride $Overrides "PrecomputeMvSupportRegionMode" "auto")
    $stagePrecomputeMvSupportFgMaskSource = [string](Resolve-StageOverride $Overrides "PrecomputeMvSupportFgMaskSource" $stageFgMaskSource)
    if ([string]::IsNullOrWhiteSpace($stagePrecomputeMvSupportFgMaskSource)) { $stagePrecomputeMvSupportFgMaskSource = "mask" }
    $stagePrecomputeMvSupportFgErodePx = [int](Resolve-StageOverride $Overrides "PrecomputeMvSupportFgErodePx" 5)

    Write-Host "[autoloop] stage=$StageName start policy=$PointTargetBlendMvPolicy pointmap_source=$PointmapSource pseudo=$PseudoGeomSubdir"
    Write-Host "[autoloop] stage=$StageName no_output_policy timeout_sec=$stageModalRunNoOutputTimeoutSec max_retries=$stageModalRunNoOutputMaxRetries infra_stop_consecutive=$stageInfraNoOutputStopConsecutive"
    Write-Host "[autoloop] stage=$StageName quality_guard abs=$StageEnableAbsoluteQualityGuard psnr>=$StageMinPSNRGuard ssim>=$StageMinSSIMGuard wl1<=$StageMaxWl1Guard"
    Write-Host "[autoloop] stage=$StageName precompute_mv_support_on=$stagePrecomputeMvSupportOn region_mode=$stagePrecomputeMvSupportRegionMode fg_mask_source=$stagePrecomputeMvSupportFgMaskSource fg_erode_px=$stagePrecomputeMvSupportFgErodePx"
    Write-Host "[autoloop] stage=$StageName point_target_blend_by_mv_support=$stagePointTargetBlendByMvSupport"
    Write-Host "[autoloop] stage=$StageName use_fg_mask=$stageUseFgMask fg_mask_source=$stageFgMaskSource point_target_blend_mv_region_mode=$stagePointTargetBlendMvRegionMode point_mv_depth_region_mode=$stagePointMvDepthRegionMode"
    Write-JsonNoBom -Path "logs/modal_phase5/overnight_ghost_autoloop_heartbeat_latest.json" -Obj ([ordered]@{
        updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        state = "running_stage"
        stage = $StageName
        policy = $PointTargetBlendMvPolicy
        pointmap_source = $PointmapSource
        lane_id = $stageLaneId
        candidate_family = $stageCandidateFamily
        guard_tier = $stageGuardTier
        rollback_triggered = $stageRollbackTriggered
        pseudo_geom_in = $PseudoGeomSubdir
        resume_ckpt_in = $ResumeCkpt
        lambda_point_mv_depth_list = $stageLambdaPointMvDepthList
        lambda_point_mv_mask_list = $stageLambdaPointMvMaskList
        point_target_mode = $stagePointTargetMode
        point_target_consensus_alpha_floor = $stagePointTargetConsensusAlphaFloor
        lambda_point = $stageLambdaPoint
        lambda_point_reproj = $stageLambdaPointReproj
        point_mv_mask_hit_thr = $stagePointMvMaskHitThr
        point_mv_mask_min_tgt_fg_ratio = $stagePointMvMaskMinTgtFgRatio
        point_mv_mask_soft_blur_px = $stagePointMvMaskSoftBlurPx
        point_mv_mask_soft_blur_iters = $stagePointMvMaskSoftBlurIters
        point_mv_mask_soft_mix = $stagePointMvMaskSoftMix
        point_mv_mask_soft_hit_thr = $stagePointMvMaskSoftHitThr
        point_mv_stride = $stagePointMvStride
        point_mv_depth_max_pairs = $stagePointMvDepthMaxPairs
        point_mv_depth_pair_mode = $stagePointMvDepthPairMode
        point_mv_depth_region_mode = $stagePointMvDepthRegionMode
        point_mv_depth_support_mode = $stagePointMvDepthSupportMode
        point_mv_depth_support_floor = $stagePointMvDepthSupportFloor
        point_mv_mask_support_mode = $stagePointMvMaskSupportMode
        point_mv_mask_support_floor = $stagePointMvMaskSupportFloor
        use_fg_mask = $stageUseFgMask
        fg_mask_source = $stageFgMaskSource
        conf_weight_per_view_quantile = $stageConfWeightPerViewQuantile
        conf_weight_per_view_min_valid = $stageConfWeightPerViewMinValid
        lambda_point_normal_consis = $stageLambdaPointNormalConsis
        point_normal_consis_warmup_steps = $stagePointNormalConsisWarmupSteps
        point_loss_fg_erode_px = $stagePointLossFgErodePx
        point_mv_depth_fg_erode_px = $stagePointMvDepthFgErodePx
        point_cons_quantile = $stagePointConsQuantile
        point_cons_focus = $stagePointConsFocus
        point_residual_quantile = $stagePointResidualQuantile
        point_residual_focus = $stagePointResidualFocus
        point_residual_boost = $stagePointResidualBoost
        point_mv_depth_outlier_boost = $stagePointMvDepthOutlierBoost
        supervision_weight_mode = $stageSupervisionWeightMode
        point_target_blend_by_mv_support = $stagePointTargetBlendByMvSupport
        point_target_blend_mv_region_mode = $stagePointTargetBlendMvRegionMode
        min_psnr_guard = $StageMinPSNRGuard
        min_ssim_guard = $StageMinSSIMGuard
        max_wl1_guard = $StageMaxWl1Guard
        enable_absolute_quality_guard = $StageEnableAbsoluteQualityGuard
        cam_names = $stageCamNames
        eval_num_src_views_list = $stageEvalNumSrcViewsList
        enable_visual_anti_black_guard = $stageEnableVisualAntiBlackGuard
        min_pred_luma_mean = $stageMinPredLumaMean
        min_pred_nonblack_ratio = $stageMinPredNonBlackRatio
        min_area_ratio = $stageMinAreaRatio
        min_width_ratio = $stageMinWidthRatio
        gram_dyn_enable = $stageGramDynEnable
        gram_dyn_layer_idx = $stageGramDynLayerIdx
        gram_dyn_quantile = $stageGramDynQuantile
        gram_dyn_weight_floor = $stageGramDynWeightFloor
        gram_dyn_warmup_steps = $stageGramDynWarmupSteps
        dyn_proxy_enable = $stageDynProxyEnable
        dyn_proxy_mode = $stageDynProxyMode
        dyn_proxy_use_gram = $stageDynProxyUseGram
        dyn_proxy_use_support = $stageDynProxyUseSupport
        dyn_proxy_floor = $stageDynProxyFloor
        dyn_proxy_warmup_steps = $stageDynProxyWarmupSteps
        enable_anysplat_ablation_six_pack = $stageEnableAnySplatAblationSixPack
        enable_extended_ckpt_wait_on_missing = $stageEnableExtendedCkptWaitOnMissing
        ckpt_extended_wait_timeout_sec = $stageCkptExtendedWaitTimeoutSec
        enable_resume_ckpt_fallback_on_short_ckpt_missing = $stageEnableResumeCkptFallbackOnShortCkptMissing
        disallow_resume_fallback_result = $stageDisallowResumeFallbackResult
        precompute_mv_support_on = $stagePrecomputeMvSupportOn
        precompute_mv_support_region_mode = $stagePrecomputeMvSupportRegionMode
        precompute_mv_support_fg_mask_source = $stagePrecomputeMvSupportFgMaskSource
        precompute_mv_support_fg_erode_px = $stagePrecomputeMvSupportFgErodePx
        modal_run_no_output_timeout_sec = $stageModalRunNoOutputTimeoutSec
        modal_run_no_output_max_retries = $stageModalRunNoOutputMaxRetries
        infra_no_output_stop_consecutive = $stageInfraNoOutputStopConsecutive
        modal_run_quiet = $stageModalRunQuiet
    })

    $pointmapSourceNorm = ([string]$PointmapSource).Trim().ToLowerInvariant()
    $blendPolicyNorm = ([string]$PointTargetBlendMvPolicy).Trim().ToLowerInvariant()
    $stagePrecomputeNoOutputTimeoutSecPointHead = [Math]::Max(900, [int]$stageModalRunNoOutputTimeoutSec)
    $stagePrecomputeNoOutputTimeoutSecPointHeadWeak = $stagePrecomputeNoOutputTimeoutSecPointHead
    $stage1DepthUnprojectPrecomputeFloorSec = 1500
    if (($pointmapSourceNorm -eq "depth_unproject") -and [bool]$EmergencyGhostShockEnabled -and ($cycleEmergencyProfile -in @("shock_primary", "shock_fallback"))) {
        $stage1DepthUnprojectPrecomputeFloorSec = [Math]::Max(300, [int]$PrecomputeNoOutputTimeoutSecDepthUnproject)
    }
    if (($pointmapSourceNorm -eq "point_head") -and ($blendPolicyNorm -eq "weak_to_depth")) {
        $stagePrecomputeNoOutputTimeoutSecPointHeadWeak = [Math]::Max(
            [int]$stagePrecomputeNoOutputTimeoutSecPointHeadWeak,
            900
        )
    }
    $stagePrecomputeNoOutputRetryTimeoutSecPointHead = [Math]::Max(
        [int]$stagePrecomputeNoOutputTimeoutSecPointHeadWeak,
        [Math]::Max(900, [int]$stageModalRunNoOutputTimeoutSec + 300)
    )
    $stagePrecomputeNoOutputRetryTimeoutSecPointHead = [Math]::Min(
        [int]$stagePrecomputeNoOutputRetryTimeoutSecPointHead,
        1500
    )
    Write-Host "[autoloop] stage=$StageName no-output profile modal_timeout=$stageModalRunNoOutputTimeoutSec precompute_depth_unproject_floor=$stage1DepthUnprojectPrecomputeFloorSec precompute_point_head=$stagePrecomputeNoOutputTimeoutSecPointHead precompute_point_head_weak=$stagePrecomputeNoOutputTimeoutSecPointHeadWeak precompute_retry_point_head=$stagePrecomputeNoOutputRetryTimeoutSecPointHead"

    & "$CodeDir\scripts\run_vggt_ghost_mvdepth_sweep.ps1" `
        -CodeDir $CodeDir `
        -SeqNames $SeqNames `
        -PseudoGeomSubdir $PseudoGeomSubdir `
        -PretrainedCkpt $PretrainedCkpt `
        -ResumeCkpt $ResumeCkpt `
        -Lr $Lr `
        -LambdaPointMvDepthList $stageLambdaPointMvDepthList `
        -LambdaPointMvMaskList $stageLambdaPointMvMaskList `
        -EvalNumSamples $EvalNumSamples `
        -MaxFramesShort $MaxFramesShort `
        -MaxStepsPerEpoch $MaxStepsPerEpoch `
        -ModalRunTimeoutSec $ModalRunTimeoutSec `
        -ModalRunNoOutputTimeoutSec $stageModalRunNoOutputTimeoutSec `
        -ModalRunNoOutputTimeoutSecPointHead $stageModalRunNoOutputTimeoutSec `
        -ModalRunNoOutputMaxRetries $stageModalRunNoOutputMaxRetries `
        -ModalRunQuiet:$stageModalRunQuiet `
        -InfraNoOutputStopConsecutive $stageInfraNoOutputStopConsecutive `
        -EnablePreemptiveProbeForPointHead $EnablePreemptiveProbeForPointHead `
        -PreemptiveProbeMaxCandidates $PreemptiveProbeMaxCandidates `
        -NoOutputProbeMaxRetries $stageModalRunNoOutputMaxRetries `
        -NoOutputProbeTimeoutSec ([Math]::Max(0, [int]$stageModalRunNoOutputTimeoutSec)) `
        -NoOutputProbeTimeoutSecPointHeadWeak ([Math]::Max(0, [int]$stageModalRunNoOutputTimeoutSec)) `
        -NoOutputProbeTimeoutSecDepthUnproject ([Math]::Max(0, [int]$stageModalRunNoOutputTimeoutSec)) `
        -PrecomputeNoOutputTimeoutSecPointHead $stagePrecomputeNoOutputTimeoutSecPointHead `
        -PrecomputeNoOutputTimeoutSecPointHeadWeak $stagePrecomputeNoOutputTimeoutSecPointHeadWeak `
        -PrecomputeNoOutputRetryTimeoutSecPointHead $stagePrecomputeNoOutputRetryTimeoutSecPointHead `
        -PrecomputeNoOutputTimeoutSecDepthUnproject $PrecomputeNoOutputTimeoutSecDepthUnproject `
        -Stage1DepthUnprojectPrecomputeFloorSec $stage1DepthUnprojectPrecomputeFloorSec `
        -NoImprovePatience $stageNoImprovePatience `
        -MinGhostImprove $StageMinGhostImprove `
        -MinPSNRGuard $StageMinPSNRGuard `
        -MinSSIMGuard $StageMinSSIMGuard `
        -MaxWl1Guard $StageMaxWl1Guard `
        -EnableAbsoluteQualityGuard $StageEnableAbsoluteQualityGuard `
        -PointTargetMode $stagePointTargetMode `
        -PointTargetConsensusAlphaFloor $stagePointTargetConsensusAlphaFloor `
        -PointTargetBlendMvPolicy $PointTargetBlendMvPolicy `
        -PointmapSource $PointmapSource `
        -UnprojectImpl $UnprojectImpl `
        -LambdaPointReproj $stageLambdaPointReproj `
        -LambdaPoint $stageLambdaPoint `
        -LambdaPointNormalConsis $stageLambdaPointNormalConsis `
        -LambdaConf $LambdaConf `
        -LambdaConfWarmupSteps $LambdaConfWarmupSteps `
        -ConfWeightPerViewQuantile $stageConfWeightPerViewQuantile `
        -ConfWeightPerViewMinValid $stageConfWeightPerViewMinValid `
        -GramDynEnable $stageGramDynEnable `
        -GramDynLayerIdx $stageGramDynLayerIdx `
        -GramDynQuantile $stageGramDynQuantile `
        -GramDynWeightFloor $stageGramDynWeightFloor `
        -GramDynWarmupSteps $stageGramDynWarmupSteps `
        -DynProxyEnable $stageDynProxyEnable `
        -DynProxyMode $stageDynProxyMode `
        -DynProxyUseGram $stageDynProxyUseGram `
        -DynProxyUseSupport $stageDynProxyUseSupport `
        -DynProxyFloor $stageDynProxyFloor `
        -DynProxyWarmupSteps $stageDynProxyWarmupSteps `
        -PointMvMaskHitThr $stagePointMvMaskHitThr `
        -PointMvMaskMinTgtFgRatio $stagePointMvMaskMinTgtFgRatio `
        -PointMvMaskSoftBlurPx $stagePointMvMaskSoftBlurPx `
        -PointMvMaskSoftBlurIters $stagePointMvMaskSoftBlurIters `
        -PointMvMaskSoftMix $stagePointMvMaskSoftMix `
        -PointMvMaskSoftHitThr $stagePointMvMaskSoftHitThr `
        -PointMvStride $stagePointMvStride `
        -PointMvDepthMaxPairs $stagePointMvDepthMaxPairs `
        -PointMvDepthPairMode $stagePointMvDepthPairMode `
        -PointMvDepthRegionMode $stagePointMvDepthRegionMode `
        -PointMvDepthSupportMode $stagePointMvDepthSupportMode `
        -PointMvDepthSupportFloor $stagePointMvDepthSupportFloor `
        -PointMvMaskSupportMode $stagePointMvMaskSupportMode `
        -PointMvMaskSupportFloor $stagePointMvMaskSupportFloor `
        -PointLossFgErodePx $stagePointLossFgErodePx `
        -PointMvDepthFgErodePx $stagePointMvDepthFgErodePx `
        -PointNormalConsisWarmupSteps $stagePointNormalConsisWarmupSteps `
        -PointConsQuantile $stagePointConsQuantile `
        -PointConsFocus $stagePointConsFocus `
        -PointResidualQuantile $stagePointResidualQuantile `
        -PointResidualFocus $stagePointResidualFocus `
        -PointResidualBoost $stagePointResidualBoost `
        -PointMvDepthOutlierBoost $stagePointMvDepthOutlierBoost `
        -SupervisionWeightMode $stageSupervisionWeightMode `
        -PointTargetBlendByMvSupport $stagePointTargetBlendByMvSupport `
        -PointTargetBlendMvRegionMode $stagePointTargetBlendMvRegionMode `
        -UseFgMask $stageUseFgMask `
        -FgMaskSource $stageFgMaskSource `
        -CamNames $stageCamNames `
        -EvalNumSrcViewsList $stageEvalNumSrcViewsList `
        -EnableVisualAntiBlackGuard $stageEnableVisualAntiBlackGuard `
        -MinPredLumaMean $stageMinPredLumaMean `
        -MinPredNonBlackRatio $stageMinPredNonBlackRatio `
        -MinAreaRatio $stageMinAreaRatio `
        -MinWidthRatio $stageMinWidthRatio `
        -EnableAnySplatAblationSixPack $stageEnableAnySplatAblationSixPack `
        -EnableExtendedCkptWaitOnMissing $stageEnableExtendedCkptWaitOnMissing `
        -CkptExtendedWaitTimeoutSec $stageCkptExtendedWaitTimeoutSec `
        -EnableResumeCkptFallbackOnShortCkptMissing $stageEnableResumeCkptFallbackOnShortCkptMissing `
        -DisallowResumeFallbackResult $stageDisallowResumeFallbackResult `
        -PrecomputeMvSupportOn $stagePrecomputeMvSupportOn `
        -PrecomputeMvSupportRegionMode $stagePrecomputeMvSupportRegionMode `
        -PrecomputeMvSupportFgMaskSource $stagePrecomputeMvSupportFgMaskSource `
        -PrecomputeMvSupportFgErodePx $stagePrecomputeMvSupportFgErodePx `
        -LaneId $stageLaneId `
        -CandidateFamily $stageCandidateFamily `
        -GuardTier $stageGuardTier `
        -RollbackTriggered:$stageRollbackTriggered
    $rc = [int]$LASTEXITCODE

    $latestCsv = "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv"
    $latestMd = "logs/modal_phase5/ghost_mvdepth_sweep_latest.md"
    $snapCsv = "logs/modal_phase5/ghost_mvdepth_sweep_${StageName}_$stamp.csv"
    $snapMd = "logs/modal_phase5/ghost_mvdepth_sweep_${StageName}_$stamp.md"
    if (Test-Path $latestCsv) { Copy-Item $latestCsv $snapCsv -Force }
    if (Test-Path $latestMd) { Copy-Item $latestMd $snapMd -Force }

    $best = Parse-BestSweepRow -CsvPath $latestCsv
    $latestRow = Get-LatestCsvRow -CsvPath $latestCsv
    $bestCkpt = ""
    $bestGeom = ""
    $bestLabel = ""
    $bestSweepCsv = ""
    $ghost = [double]::NaN
    $psnr = [double]::NaN
    $ssim = [double]::NaN
    $wl1 = [double]::NaN
    $bestGhostRowsCsv = ""
    $bestVisualPng = ""
    $stageBestStripPng = ""
    $bestLambdaMvDepth = ""
    $bestLambdaMvMask = ""
    $bestGhostWidthRatio = [double]::NaN
    $bestGhostAreaRatio = [double]::NaN
    $bestGhostPeakCount = [double]::NaN
    $bestGhostCenterOffset = [double]::NaN
    $bestGhostSoftScore = [double]::NaN
    $bestGhostVisualScore = [double]::NaN
    $bestPredLumaMean = [double]::NaN
    $bestPredNonBlackRatio = [double]::NaN
    $bestVisualGuardBlocked = $false
    $bestVisualGuardReason = ""
    $bestQualityGuardBlocked = $false
    $bestQualityGuardReason = ""
    $bestCandidateInvalidReason = ""
    $bestEvalNumSrcViews = ""
    $bestEvalNumSrcViewsDeclared = ""
    $bestEvalNumSrcViewsActual = ""
    $bestEvalNumSrcViewsMismatch = $false
    $bestCamCountUsed = ""
    $bestPrecomputeSource = ""
    $bestPrecomputeSourceRequested = ""
    $bestPrecomputeSourceResolved = ""
    $bestPrecomputeFallbackUsed = $false
    $bestPrecomputeTimeoutHit = $false
    $metaRow = if ($best -ne $null) { $best } else { $latestRow }
    if ($metaRow -ne $null) {
        $bestVisualGuardBlocked = To-BoolLoose -Value $metaRow.visual_guard_blocked -Default $false
        $bestVisualGuardReason = [string]$metaRow.visual_guard_reason
        $bestQualityGuardBlocked = To-BoolLoose -Value $metaRow.quality_guard_blocked -Default $false
        $bestQualityGuardReason = [string]$metaRow.quality_guard_reason
        $bestCandidateInvalidReason = [string]$metaRow.candidate_invalid_reason
        $bestEvalNumSrcViews = [string]$metaRow.eval_num_src_views
        $bestEvalNumSrcViewsDeclared = if ($metaRow.PSObject.Properties["eval_num_src_views_declared"]) { [string]$metaRow.eval_num_src_views_declared } else { [string]$metaRow.eval_num_src_views }
        $bestEvalNumSrcViewsActual = [string]$metaRow.eval_num_src_views_actual
        $bestEvalNumSrcViewsMismatch = To-BoolLoose -Value $metaRow.eval_num_src_views_mismatch -Default $false
        $bestCamCountUsed = [string]$metaRow.cam_count_used
        $bestPrecomputeSourceRequested = [string]$metaRow.precompute_source_requested
        $bestPrecomputeSourceResolved = [string]$metaRow.precompute_source_resolved
        $bestPrecomputeFallbackUsed = To-BoolLoose -Value $metaRow.precompute_fallback_used -Default $false
        $bestPrecomputeTimeoutHit = To-BoolLoose -Value $metaRow.precompute_timeout_hit -Default $false
        if ($metaRow.PSObject.Properties["precompute_source"]) {
            $bestPrecomputeSource = [string]$metaRow.precompute_source
        }
        if ([string]::IsNullOrWhiteSpace($bestPrecomputeSource)) {
            $bestPrecomputeSource = if (-not [string]::IsNullOrWhiteSpace($bestPrecomputeSourceResolved)) { $bestPrecomputeSourceResolved } else { $bestPrecomputeSourceRequested }
        }
    }
    if ($best -ne $null) {
        $bestLabel = [string]$best.best_label
        $bestSweepCsv = [string]$best.sweep_csv
        $ghost = To-DoubleOrNaN($best.ghost_score_mean)
        $psnr = To-DoubleOrNaN($best.mean_PSNR)
        $ssim = To-DoubleOrNaN($best.mean_SSIM)
        $wl1 = To-DoubleOrNaN($best.mean_weighted_L1)
        $bestLambdaMvDepth = [string]$best.lambda_point_mv_depth
        $bestLambdaMvMask = [string]$best.lambda_point_mv_mask
        $bestGhostSoftScore = To-DoubleOrNaN($best.ghost_soft_score)
        $bestGhostVisualScore = To-DoubleOrNaN($best.ghost_visual_score)
        $bestPredLumaMean = To-DoubleOrNaN($best.pred_luma_mean)
        $bestPredNonBlackRatio = To-DoubleOrNaN($best.pred_nonblack_ratio_thr008)
        $ckptInfo = Resolve-BestCkpt -SweepCsvPath $bestSweepCsv -BestLabel $bestLabel
        if ($ckptInfo -ne $null) {
            $bestCkpt = [string]$ckptInfo.ft_ckpt
            $bestGeom = [string]$ckptInfo.geom_subdir
        }

        $bestGhostRowsCsv = Resolve-GhostRowsCsv -BestRow $best
        $stats = Get-GhostRowsStats -GhostRowsCsv $bestGhostRowsCsv
        if ($stats -ne $null) {
            $bestGhostWidthRatio = To-DoubleOrNaN($stats.mean_width_ratio)
            $bestGhostAreaRatio = To-DoubleOrNaN($stats.mean_area_ratio)
            $bestGhostPeakCount = To-DoubleOrNaN($stats.mean_peak_count)
            $bestGhostCenterOffset = To-DoubleOrNaN($stats.mean_center_offset_ratio)
            $bestVisualPng = [string]$stats.first_path
            $stripOut = "logs/modal_phase5/ghost_stage_best_${StageName}_$stamp.png"
            $stripMade = Make-ContactSheetSafe -ImagePaths @($stats.image_paths) -OutPng $stripOut
            if (-not [string]::IsNullOrWhiteSpace($stripMade)) {
                $stageBestStripPng = $stripMade
            }
        }
    }

    Write-Host "[autoloop] stage=$StageName done rc=$rc ghost=$ghost psnr=$psnr geom=$bestGeom"
    Write-JsonNoBom -Path "logs/modal_phase5/overnight_ghost_autoloop_heartbeat_latest.json" -Obj ([ordered]@{
        updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        state = "stage_done"
        stage = $StageName
        policy = $PointTargetBlendMvPolicy
        pointmap_source = $PointmapSource
        lane_id = $stageLaneId
        candidate_family = $stageCandidateFamily
        guard_tier = $stageGuardTier
        rollback_triggered = $stageRollbackTriggered
        rc = $rc
        ghost = $ghost
        psnr = $psnr
        ssim = $ssim
        wl1 = $wl1
        best_geom = $bestGeom
        best_ckpt = $bestCkpt
        best_lambda_point_mv_depth = $bestLambdaMvDepth
        best_lambda_point_mv_mask = $bestLambdaMvMask
        best_ghost_rows_csv = $bestGhostRowsCsv
        best_visual_png = $bestVisualPng
        stage_best_strip_png = $stageBestStripPng
        best_ghost_width_ratio = $bestGhostWidthRatio
        best_ghost_area_ratio = $bestGhostAreaRatio
        best_ghost_peak_count = $bestGhostPeakCount
        best_ghost_center_offset_ratio = $bestGhostCenterOffset
        best_ghost_visual_score = $bestGhostVisualScore
        best_pred_luma_mean = $bestPredLumaMean
        best_pred_nonblack_ratio = $bestPredNonBlackRatio
        best_visual_guard_blocked = $bestVisualGuardBlocked
        best_visual_guard_reason = $bestVisualGuardReason
        best_quality_guard_blocked = $bestQualityGuardBlocked
        best_quality_guard_reason = $bestQualityGuardReason
        best_candidate_invalid_reason = $bestCandidateInvalidReason
        best_eval_num_src_views = $bestEvalNumSrcViews
        best_eval_num_src_views_declared = $bestEvalNumSrcViewsDeclared
        best_eval_num_src_views_actual = $bestEvalNumSrcViewsActual
        best_eval_num_src_views_mismatch = $bestEvalNumSrcViewsMismatch
        best_cam_count_used = $bestCamCountUsed
        best_precompute_source = $bestPrecomputeSource
        best_precompute_source_requested = $bestPrecomputeSourceRequested
        best_precompute_source_resolved = $bestPrecomputeSourceResolved
        best_precompute_fallback_used = $bestPrecomputeFallbackUsed
        best_precompute_timeout_hit = $bestPrecomputeTimeoutHit
        use_fg_mask = $stageUseFgMask
        fg_mask_source = $stageFgMaskSource
        point_target_blend_mv_region_mode = $stagePointTargetBlendMvRegionMode
        point_mv_depth_region_mode = $stagePointMvDepthRegionMode
    })
    return [pscustomobject]@{
        stage = $StageName
        policy = $PointTargetBlendMvPolicy
        pointmap_source = $PointmapSource
        lane_id = $stageLaneId
        candidate_family = $stageCandidateFamily
        guard_tier = $stageGuardTier
        rollback_triggered = $stageRollbackTriggered
        stage_lambda_point_mv_depth_list = $stageLambdaPointMvDepthList
        stage_lambda_point_mv_mask_list = $stageLambdaPointMvMaskList
        stage_point_target_mode = $stagePointTargetMode
        stage_point_target_consensus_alpha_floor = $stagePointTargetConsensusAlphaFloor
        stage_lambda_point = $stageLambdaPoint
        stage_lambda_point_reproj = $stageLambdaPointReproj
        stage_point_mv_mask_hit_thr = $stagePointMvMaskHitThr
        stage_point_mv_mask_min_tgt_fg_ratio = $stagePointMvMaskMinTgtFgRatio
        stage_point_mv_mask_soft_blur_px = $stagePointMvMaskSoftBlurPx
        stage_point_mv_mask_soft_blur_iters = $stagePointMvMaskSoftBlurIters
        stage_point_mv_mask_soft_mix = $stagePointMvMaskSoftMix
        stage_point_mv_mask_soft_hit_thr = $stagePointMvMaskSoftHitThr
        stage_point_mv_stride = $stagePointMvStride
        stage_point_mv_depth_max_pairs = $stagePointMvDepthMaxPairs
        stage_point_mv_depth_pair_mode = $stagePointMvDepthPairMode
        stage_point_mv_depth_region_mode = $stagePointMvDepthRegionMode
        stage_point_mv_depth_support_mode = $stagePointMvDepthSupportMode
        stage_point_mv_depth_support_floor = $stagePointMvDepthSupportFloor
        stage_point_mv_mask_support_mode = $stagePointMvMaskSupportMode
        stage_point_mv_mask_support_floor = $stagePointMvMaskSupportFloor
        stage_use_fg_mask = $stageUseFgMask
        stage_fg_mask_source = $stageFgMaskSource
        stage_point_target_blend_mv_region_mode = $stagePointTargetBlendMvRegionMode
        stage_conf_weight_per_view_quantile = $stageConfWeightPerViewQuantile
        stage_conf_weight_per_view_min_valid = $stageConfWeightPerViewMinValid
        stage_lambda_point_normal_consis = $stageLambdaPointNormalConsis
        stage_point_normal_consis_warmup_steps = $stagePointNormalConsisWarmupSteps
        stage_point_loss_fg_erode_px = $stagePointLossFgErodePx
        stage_point_mv_depth_fg_erode_px = $stagePointMvDepthFgErodePx
        stage_point_cons_quantile = $stagePointConsQuantile
        stage_point_cons_focus = $stagePointConsFocus
        stage_point_residual_quantile = $stagePointResidualQuantile
        stage_point_residual_focus = $stagePointResidualFocus
        stage_point_residual_boost = $stagePointResidualBoost
        stage_point_mv_depth_outlier_boost = $stagePointMvDepthOutlierBoost
        stage_supervision_weight_mode = $stageSupervisionWeightMode
        rc = $rc
        ghost = $ghost
        psnr = $psnr
        ssim = $ssim
        wl1 = $wl1
        best_label = $bestLabel
        best_geom = $bestGeom
        best_ckpt = $bestCkpt
        best_lambda_point_mv_depth = $bestLambdaMvDepth
        best_lambda_point_mv_mask = $bestLambdaMvMask
        best_ghost_rows_csv = $bestGhostRowsCsv
        best_visual_png = $bestVisualPng
        stage_best_strip_png = $stageBestStripPng
        best_ghost_width_ratio = $bestGhostWidthRatio
        best_ghost_area_ratio = $bestGhostAreaRatio
        best_ghost_peak_count = $bestGhostPeakCount
        best_ghost_center_offset_ratio = $bestGhostCenterOffset
        ghost_soft_score = $bestGhostSoftScore
        ghost_visual_score = $bestGhostVisualScore
        pred_luma_mean = $bestPredLumaMean
        pred_nonblack_ratio_thr008 = $bestPredNonBlackRatio
        visual_guard_blocked = $bestVisualGuardBlocked
        visual_guard_reason = $bestVisualGuardReason
        quality_guard_blocked = $bestQualityGuardBlocked
        quality_guard_reason = $bestQualityGuardReason
        candidate_invalid_reason = $bestCandidateInvalidReason
        eval_num_src_views = $bestEvalNumSrcViews
        eval_num_src_views_declared = $bestEvalNumSrcViewsDeclared
        eval_num_src_views_actual = $bestEvalNumSrcViewsActual
        eval_num_src_views_mismatch = $bestEvalNumSrcViewsMismatch
        cam_count_used = $bestCamCountUsed
        precompute_source = $bestPrecomputeSource
        precompute_source_requested = $bestPrecomputeSourceRequested
        precompute_source_resolved = $bestPrecomputeSourceResolved
        precompute_fallback_used = $bestPrecomputeFallbackUsed
        precompute_timeout_hit = $bestPrecomputeTimeoutHit
        use_fg_mask = $stageUseFgMask
        fg_mask_source = $stageFgMaskSource
        point_target_blend_mv_region_mode = $stagePointTargetBlendMvRegionMode
        point_mv_depth_region_mode = $stagePointMvDepthRegionMode
        sweep_csv = $snapCsv
        sweep_md = $snapMd
        raw_sweep_csv = $(if ([string]::IsNullOrWhiteSpace($bestSweepCsv)) { $snapCsv } else { $bestSweepCsv })
        pseudo_geom_in = $PseudoGeomSubdir
        resume_ckpt_in = $ResumeCkpt
        stage_skip_reason = ""
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    }
}

$windowT0 = [datetime]::MinValue
$windowT0Source = ""
$deadline = [datetime]::MinValue
$deadlineSource = ""

$ensureMeta = Read-JsonMaybe -Path "logs/modal_phase5/ensure_hot_update_watcher_latest.json"
if ($ensureMeta -ne $null) {
    $ensureT0 = Parse-DateMaybe -Text ([string]$ensureMeta.t0)
    $ensureDeadline = Parse-DateMaybe -Text ([string]$ensureMeta.deadline)
    if ($ensureT0 -ne [datetime]::MinValue) {
        $windowT0 = $ensureT0
        $windowT0Source = "ensure_watch.t0"
    }
    if ($ensureDeadline -ne [datetime]::MinValue) {
        $deadline = $ensureDeadline
        $deadlineSource = "ensure_watch.deadline"
    }
}

if (($windowT0 -eq [datetime]::MinValue) -or ($deadline -eq [datetime]::MinValue)) {
    $hotWatch = Read-JsonMaybe -Path "logs/modal_phase5/overnight_ghost_autoloop_hotupdate_watch_latest.json"
    if ($hotWatch -ne $null) {
        if ($windowT0 -eq [datetime]::MinValue) {
            $hotT0 = Parse-DateMaybe -Text ([string]$hotWatch.t0)
            if ($hotT0 -ne [datetime]::MinValue) {
                $windowT0 = $hotT0
                $windowT0Source = "hot_watch.t0"
            }
        }
        if ($deadline -eq [datetime]::MinValue) {
            $hotDeadline = Parse-DateMaybe -Text ([string]$hotWatch.deadline)
            if ($hotDeadline -ne [datetime]::MinValue) {
                $deadline = $hotDeadline
                $deadlineSource = "hot_watch.deadline"
            }
        }
    }
}

if (($windowT0 -eq [datetime]::MinValue) -or ($deadline -eq [datetime]::MinValue)) {
    $finalMeta = Read-JsonMaybe -Path "logs/modal_phase5/overnight_ghost_autoloop_12h_final_latest.json"
    if ($finalMeta -ne $null) {
        if ($windowT0 -eq [datetime]::MinValue) {
            $finalT0 = Parse-DateMaybe -Text ([string]$finalMeta.t0)
            if ($finalT0 -ne [datetime]::MinValue) {
                $windowT0 = $finalT0
                $windowT0Source = "final_json.t0"
            }
        }
        if ($deadline -eq [datetime]::MinValue) {
            $finalDeadline = Parse-DateMaybe -Text ([string]$finalMeta.deadline)
            if ($finalDeadline -ne [datetime]::MinValue) {
                $deadline = $finalDeadline
                $deadlineSource = "final_json.deadline"
            }
        }
    }
}

if (($windowT0 -eq [datetime]::MinValue) -or ($deadline -eq [datetime]::MinValue)) {
    $latestMeta = Read-JsonMaybe -Path "logs/modal_phase5/overnight_ghost_autoloop_latest.json"
    if ($latestMeta -ne $null) {
        if ($windowT0 -eq [datetime]::MinValue) {
            $latestT0 = Parse-DateMaybe -Text ([string]$latestMeta.t0)
            if ($latestT0 -ne [datetime]::MinValue) {
                $windowT0 = $latestT0
                $windowT0Source = "latest_json.t0"
            }
        }
        if ($deadline -eq [datetime]::MinValue) {
            $latestDeadline = Parse-DateMaybe -Text ([string]$latestMeta.deadline)
            if ($latestDeadline -ne [datetime]::MinValue) {
                $deadline = $latestDeadline
                $deadlineSource = "latest_json.deadline"
            }
        }
    }
}

$forcedDeadline = Parse-DateMaybe -Text $FinalDeadline
if ($forcedDeadline -ne [datetime]::MinValue) {
    $deadline = $forcedDeadline
    $deadlineSource = "param.final_deadline"
    if ($windowT0 -eq [datetime]::MinValue) {
        $windowT0 = $deadline.AddHours(-1.0 * [Math]::Max(1, [int]$StopAfterHours))
        $windowT0Source = "derived_from_forced_deadline"
    }
}

if ($windowT0 -eq [datetime]::MinValue) {
    $windowT0 = Get-Date
    $windowT0Source = "now.fallback"
}
if ($deadline -eq [datetime]::MinValue) {
    $deadline = $windowT0.AddHours([Math]::Max(1, [int]$StopAfterHours))
    $deadlineSource = "derived_from_t0"
}
$nowForWindow = Get-Date
if (($deadline -ne [datetime]::MinValue) -and ($deadline -le $nowForWindow)) {
    $oldDeadlineText = $deadline.ToString("yyyy-MM-ddTHH:mm:ss")
    $windowT0 = $nowForWindow
    $windowT0Source = "refresh.now_expired_deadline"
    $deadline = $windowT0.AddHours([Math]::Max(1, [int]$StopAfterHours))
    $deadlineSource = "refresh.expired_deadline_plus_hours"
    Write-Host "[autoloop] stale deadline detected old=$oldDeadlineText -> refreshed deadline=$($deadline.ToString('yyyy-MM-ddTHH:mm:ss')) stop_after_hours=$StopAfterHours"
}
$windowT0Text = $windowT0.ToString("yyyy-MM-ddTHH:mm:ss")
$deadlineText = $deadline.ToString("yyyy-MM-ddTHH:mm:ss")
Write-Host "[autoloop] window t0=$windowT0Text deadline=$deadlineText source=$windowT0Source/$deadlineSource"
$globalBestGhost = [double]::PositiveInfinity
$globalBestPsnr = [double]::NegativeInfinity
$globalBestSsim = [double]::NegativeInfinity
$globalBestWl1 = [double]::PositiveInfinity
$globalBestCkpt = $StartResumeCkpt
$globalBestGeom = $StartPseudoGeomSubdir
$currPseudo = $StartPseudoGeomSubdir
$currResume = $StartResumeCkpt
$noImproveCycles = 0
$noSubstantialImproveCycles = 0
$tuneStep = 0
    $pendingTuneAction = "none"
$globalBestVisualPng = ""
$globalBestStats = $null
$globalBestLambdaDepthHint = ""
$globalBestLambdaMaskHint = ""
$globalBestHintStageFamily = "unknown"
$globalBestHintApplyStage1 = $true
$globalBestHintApplyStage2 = $true
$resumeUpdateReason = "init"
$lastTuneStateBeforeApply = $null
$lastTuneActionApplied = "none"
$lastTuneAppliedCycle = 0
$regressCycles = 0
$routeMode = "main"
$abRouteExecuted = $false
$abRouteVariant = "balance"
$abBalanceResult = $null
$abAggressiveResult = $null
$aggressiveRouteCooldownCycles = 0
$history = New-Object System.Collections.Generic.List[object]

if ($EnablePersistentCycleState) {
    $persistState = Resolve-PersistentCycleState `
        -Path $PersistentCycleStatePath `
        -MaxAgeHours ([Math]::Max(1, [int]$PersistentCycleStateMaxAgeHours))
    if ($persistState -ne $null) {
        $noImproveCycles = [int]$persistState.no_improve_cycles
        $noSubstantialImproveCycles = [int]$persistState.no_substantial_improve_cycles
        $tuneStep = [int]$persistState.tune_step
        $regressCycles = [int]$persistState.regress_cycles
        $pendingTuneAction = [string]$persistState.pending_tune_action
        $routeMode = [string]$persistState.route_mode
        $abRouteExecuted = [bool]$persistState.ab_route_executed
        $abRouteVariant = [string]$persistState.ab_route_variant
        $aggressiveRouteCooldownCycles = [int]$persistState.aggressive_route_cooldown_cycles
        Set-TuneState ([pscustomobject]@{
            BasePointMvMaskSoftMix = [double]$persistState.base_point_mv_mask_soft_mix
            BasePointMvMaskSoftHitThr = [double]$persistState.base_point_mv_mask_soft_hit_thr
            BasePointMvStride = [int]$persistState.base_point_mv_stride
            BasePointMvDepthMaxPairs = [int]$persistState.base_point_mv_depth_max_pairs
            BasePointMvDepthSupportMode = [string]$persistState.base_point_mv_depth_support_mode
            BasePointMvDepthSupportFloor = [double]$persistState.base_point_mv_depth_support_floor
            BasePointMvMaskMinTgtFgRatio = [double]$persistState.base_point_mv_mask_min_tgt_fg_ratio
        })
        Write-Host "[autoloop] restored persistent cycle state updated_at=$($persistState.updated_at) no_substantial=$noSubstantialImproveCycles tune_step=$tuneStep tune=$pendingTuneAction"
    }
}

$stage1DepthList = if ([string]::IsNullOrWhiteSpace($Stage1LambdaPointMvDepthList)) { $LambdaPointMvDepthList } else { $Stage1LambdaPointMvDepthList }
$stage1MaskList = if ([string]::IsNullOrWhiteSpace($Stage1LambdaPointMvMaskList)) { $LambdaPointMvMaskList } else { $Stage1LambdaPointMvMaskList }
$stage1NoImprovePatienceEffective = [Math]::Max(3, [Math]::Min($StageNoImprovePatience, [Math]::Max(1, $Stage1NoImprovePatience)))
$stage2DepthList = if ([string]::IsNullOrWhiteSpace($Stage2LambdaPointMvDepthList)) { $LambdaPointMvDepthList } else { $Stage2LambdaPointMvDepthList }
$stage2MaskList = if ([string]::IsNullOrWhiteSpace($Stage2LambdaPointMvMaskList)) { $LambdaPointMvMaskList } else { $Stage2LambdaPointMvMaskList }
$stage2NoImprovePatienceEffective = [Math]::Max(2, [Math]::Min($StageNoImprovePatience, [Math]::Max(1, $Stage2NoImprovePatience)))
$historyQualityRef = Resolve-ExistingGlobalBestBootstrap -CsvPath "logs/modal_phase5/ghost_autoloop_latest.csv"
$stage2HistoryRefPsnr = if ($historyQualityRef -ne $null) { To-DoubleOrNaN($historyQualityRef.psnr) } else { [double]::NaN }
$stage2HistoryRefSsim = if ($historyQualityRef -ne $null) { To-DoubleOrNaN($historyQualityRef.ssim) } else { [double]::NaN }
$stage2HistoryRefWl1 = if ($historyQualityRef -ne $null) { To-DoubleOrNaN($historyQualityRef.wl1) } else { [double]::NaN }
$stage1HistoryFocus = Resolve-Stage1HistoryFocus -TopRows ([Math]::Max(1, [int]$Stage1HistoryTopRows))
if ($stage1HistoryFocus -ne $null) {
    $stage1DepthList = Build-FocusedLambdaList -PrimaryList ([string]$stage1HistoryFocus.depth_list) -FallbackList $stage1DepthList -Preferred ([string]$stage1HistoryFocus.top_depth)
    $stage1MaskList = Build-FocusedLambdaList -PrimaryList ([string]$stage1HistoryFocus.mask_list) -FallbackList $stage1MaskList -Preferred ([string]$stage1HistoryFocus.top_mask)
}
$stage2HistoryFocus = Resolve-StageHistoryFocus `
    -StageToken "stage2_weak" `
    -TopRows ([Math]::Max(1, [int]$Stage2HistoryTopRows)) `
    -ReferencePSNR $stage2HistoryRefPsnr `
    -ReferenceSSIM $stage2HistoryRefSsim `
    -ReferenceWl1 $stage2HistoryRefWl1 `
    -MaxPSNRDrop ([Math]::Max(0.0, [double]$Stage2HistoryMaxPSNRDrop)) `
    -MaxSSIMDrop ([Math]::Max(0.0, [double]$Stage2HistoryMaxSSIMDrop)) `
    -MaxWl1Rise ([Math]::Max(0.0, [double]$Stage2HistoryMaxWl1Rise)) `
    -EnableQualityGuard:$Stage2HistoryQualityAware
if ($stage2HistoryFocus -ne $null) {
    $stage2DepthList = Build-FocusedLambdaList -PrimaryList ([string]$stage2HistoryFocus.depth_list) -FallbackList $stage2DepthList -Preferred ([string]$stage2HistoryFocus.top_depth)
    $stage2MaskList = Build-FocusedLambdaList -PrimaryList ([string]$stage2HistoryFocus.mask_list) -FallbackList $stage2MaskList -Preferred ([string]$stage2HistoryFocus.top_mask)

    $stage2DepthNeed = [Math]::Max(1, [int]$Stage2HistoryMinDepthValues)
    $stage2MaskNeed = [Math]::Max(1, [int]$Stage2HistoryMinMaskValues)
    $stage2DepthCount = @(Parse-LambdaList -List $stage2DepthList).Count
    $stage2MaskCount = @(Parse-LambdaList -List $stage2MaskList).Count

    if ($stage2DepthCount -lt $stage2DepthNeed) {
        $stage2DepthList = Build-FocusedLambdaList `
            -PrimaryList $Stage2LambdaPointMvDepthList `
            -FallbackList $stage2DepthList `
            -Preferred ([string]$stage2HistoryFocus.top_depth)
        $stage2DepthCount = @(Parse-LambdaList -List $stage2DepthList).Count
        Write-Host "[autoloop] stage2 history focus diversity expand depth_count=$stage2DepthCount required=$stage2DepthNeed depth_list=$stage2DepthList"
    }
    if ($stage2MaskCount -lt $stage2MaskNeed) {
        $stage2MaskList = Build-FocusedLambdaList `
            -PrimaryList $Stage2LambdaPointMvMaskList `
            -FallbackList $stage2MaskList `
            -Preferred ([string]$stage2HistoryFocus.top_mask)
        $stage2MaskCount = @(Parse-LambdaList -List $stage2MaskList).Count
        Write-Host "[autoloop] stage2 history focus diversity expand mask_count=$stage2MaskCount required=$stage2MaskNeed mask_list=$stage2MaskList"
    }
}

$bootstrapBest = Resolve-ExistingGlobalBestBootstrap -CsvPath "logs/modal_phase5/ghost_autoloop_latest.csv"
if ($EnableHistoricalSweepBootstrap) {
    $histBootstrap = Resolve-HistoricalSweepBootstrapCandidate `
        -BaselineCandidate $bootstrapBest `
        -StrictMaxPSNRDrop $CyclePromoteMaxPSNRDrop `
        -StrictMaxSSIMDrop $CyclePromoteMaxSSIMDrop `
        -StrictMaxWl1Rise $CyclePromoteMaxWl1Rise `
        -RelaxedMinGhostGain $HistoryBootstrapRelaxedMinGhostGain `
        -RelaxedMaxPSNRDrop $HistoryBootstrapRelaxedMaxPSNRDrop `
        -RelaxedMaxSSIMDrop $HistoryBootstrapRelaxedMaxSSIMDrop `
        -RelaxedMaxWl1Rise $HistoryBootstrapRelaxedMaxWl1Rise
    if (($histBootstrap -ne $null) -and (-not [double]::IsNaN((To-DoubleOrNaN($histBootstrap.ghost))))) {
        $histGhost = To-DoubleOrNaN($histBootstrap.ghost)
        $baseGhost = if ($bootstrapBest -ne $null) { To-DoubleOrNaN($bootstrapBest.ghost) } else { [double]::PositiveInfinity }
        if (($bootstrapBest -eq $null) -or [double]::IsNaN($baseGhost) -or ($histGhost -lt $baseGhost)) {
            $bootstrapBest = $histBootstrap
            Write-Host "[autoloop] bootstrap upgraded from historical sweep source=$([string]$histBootstrap.bootstrap_source) guard=$([string]$histBootstrap.bootstrap_guard_mode) ghost=$([string](Fmt-Num $histGhost 6))"
        }
    }
}
if (($bootstrapBest -ne $null) -and (-not [double]::IsNaN((To-DoubleOrNaN($bootstrapBest.ghost))))) {
    $globalBestGhost = To-DoubleOrNaN($bootstrapBest.ghost)
    if (-not [double]::IsNaN((To-DoubleOrNaN($bootstrapBest.psnr)))) {
        $globalBestPsnr = To-DoubleOrNaN($bootstrapBest.psnr)
    }
    if (-not [double]::IsNaN((To-DoubleOrNaN($bootstrapBest.ssim)))) {
        $globalBestSsim = To-DoubleOrNaN($bootstrapBest.ssim)
    }
    if (-not [double]::IsNaN((To-DoubleOrNaN($bootstrapBest.wl1)))) {
        $globalBestWl1 = To-DoubleOrNaN($bootstrapBest.wl1)
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$bootstrapBest.best_ckpt)) {
        $globalBestCkpt = [string]$bootstrapBest.best_ckpt
        $currResume = $globalBestCkpt
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$bootstrapBest.best_geom)) {
        $globalBestGeom = [string]$bootstrapBest.best_geom
        $currPseudo = $globalBestGeom
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$bootstrapBest.best_visual_png) -and (Test-Path ([string]$bootstrapBest.best_visual_png))) {
        $globalBestVisualPng = [string]$bootstrapBest.best_visual_png
    }
    $bootstrapStats = Get-GhostRowsStats -GhostRowsCsv ([string]$bootstrapBest.best_ghost_rows_csv)
    if ($bootstrapStats -ne $null) {
        $globalBestStats = $bootstrapStats
    }
    $bootstrapHistorical = -not [string]::IsNullOrWhiteSpace([string]$bootstrapBest.bootstrap_source)
    $bootstrapSourceType = if ($bootstrapHistorical) { "historical" } else { "existing" }
    $bootstrapHintScope = Resolve-HintScope `
        -BestStage ([string]$bootstrapBest.best_stage) `
        -IsHistoricalBootstrap:$bootstrapHistorical
    $globalBestHintStageFamily = [string]$bootstrapHintScope.stage_family
    $globalBestHintApplyStage1 = [bool]$bootstrapHintScope.apply_stage1
    $globalBestHintApplyStage2 = [bool]$bootstrapHintScope.apply_stage2
    $depthHint = [string]$bootstrapBest.best_lambda_point_mv_depth
    $maskHint = [string]$bootstrapBest.best_lambda_point_mv_mask
    if (-not [string]::IsNullOrWhiteSpace($depthHint)) {
        if ($globalBestHintApplyStage1) {
            $stage1DepthList = Build-FocusedLambdaList -PrimaryList $stage1DepthList -FallbackList $LambdaPointMvDepthList -Preferred $depthHint
        }
        if ($globalBestHintApplyStage2) {
            $stage2DepthList = Build-FocusedLambdaList -PrimaryList $stage2DepthList -FallbackList $Stage2LambdaPointMvDepthList -Preferred $depthHint
        }
        $globalBestLambdaDepthHint = $depthHint
    }
    if (-not [string]::IsNullOrWhiteSpace($maskHint)) {
        if ($globalBestHintApplyStage1) {
            $stage1MaskList = Build-FocusedLambdaList -PrimaryList $stage1MaskList -FallbackList $LambdaPointMvMaskList -Preferred $maskHint
        }
        if ($globalBestHintApplyStage2) {
            $stage2MaskList = Build-FocusedLambdaList -PrimaryList $stage2MaskList -FallbackList $Stage2LambdaPointMvMaskList -Preferred $maskHint
        }
        $globalBestLambdaMaskHint = $maskHint
    }
    if (-not $globalBestHintApplyStage1) {
        Write-Host "[autoloop] hint scope: stage1 keep history focus (bootstrap_stage=$([string]$bootstrapBest.best_stage), source=$bootstrapSourceType)"
    }
    if (-not $globalBestHintApplyStage2) {
        Write-Host "[autoloop] hint scope: stage2 keep history focus (bootstrap_stage=$([string]$bootstrapBest.best_stage), source=$bootstrapSourceType)"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$bootstrapBest.bootstrap_source)) {
        $resumeUpdateReason = "bootstrap_historical_sweep"
        Write-Host "[autoloop] bootstrap historical best source=$([string]$bootstrapBest.bootstrap_source) guard=$([string]$bootstrapBest.bootstrap_guard_mode) stage=$($bootstrapBest.best_stage) ghost=$([string](Fmt-Num $globalBestGhost 6)) psnr=$([string](Fmt-Num $globalBestPsnr 6)) ssim=$([string](Fmt-Num $globalBestSsim 6)) wl1=$([string](Fmt-Num $globalBestWl1 6))"
    } else {
        $resumeUpdateReason = "bootstrap_existing_global_best"
        Write-Host "[autoloop] bootstrap existing best stage=$($bootstrapBest.best_stage) ghost=$([string](Fmt-Num $globalBestGhost 6)) psnr=$([string](Fmt-Num $globalBestPsnr 6)) ssim=$([string](Fmt-Num $globalBestSsim 6)) wl1=$([string](Fmt-Num $globalBestWl1 6))"
    }
}

# Cross-stage hint override: if stage1 historical focus lags global best ghost by a clear margin,
# allow injecting global best lambda hints into stage1 even when bootstrap stage family is stage2.
if ($Stage1EnableCrossStageHintFromGlobal -and (-not $globalBestHintApplyStage1)) {
    $stage1TopGhost = [double]::NaN
    if ($stage1HistoryFocus -ne $null) {
        $stage1TopGhost = To-DoubleOrNaN($stage1HistoryFocus.top_ghost)
    }
    $ghostLag = [double]::NaN
    if ((-not [double]::IsNaN($stage1TopGhost)) -and (-not [double]::IsNaN($globalBestGhost)) -and (-not [double]::IsInfinity($globalBestGhost))) {
        $ghostLag = $stage1TopGhost - $globalBestGhost
    }
    if ([double]::IsNaN($ghostLag) -or ($ghostLag -ge [Math]::Max(0.0, [double]$Stage1CrossStageHintGhostLag))) {
        $crossApplied = $false
        if (-not [string]::IsNullOrWhiteSpace($globalBestLambdaDepthHint)) {
            $stage1DepthList = Build-FocusedLambdaList -PrimaryList $stage1DepthList -FallbackList $LambdaPointMvDepthList -Preferred $globalBestLambdaDepthHint
            $crossApplied = $true
        }
        if (-not [string]::IsNullOrWhiteSpace($globalBestLambdaMaskHint)) {
            $stage1MaskList = Build-FocusedLambdaList -PrimaryList $stage1MaskList -FallbackList $LambdaPointMvMaskList -Preferred $globalBestLambdaMaskHint
            $crossApplied = $true
        }
        if ($crossApplied) {
            $globalBestHintApplyStage1 = $true
            $lagText = if ([double]::IsNaN($ghostLag)) { "NaN" } else { [string](Fmt-Num $ghostLag 6) }
            Write-Host "[autoloop] stage1 cross-stage hint enabled: lag=$lagText >= $([string](Fmt-Num ([double]$Stage1CrossStageHintGhostLag) 6)), depth_hint=$globalBestLambdaDepthHint, mask_hint=$globalBestLambdaMaskHint"
        }
    }
}

if ($Stage1PrioritizeHistoryTopAfterCrossHint -and ($stage1HistoryFocus -ne $null)) {
    $stage1TopDepthPreferred = [string]$stage1HistoryFocus.top_depth
    $stage1TopMaskPreferred = [string]$stage1HistoryFocus.top_mask
    if (-not [string]::IsNullOrWhiteSpace($stage1TopDepthPreferred)) {
        $stage1DepthList = Build-FocusedLambdaList -PrimaryList $stage1DepthList -FallbackList $LambdaPointMvDepthList -Preferred $stage1TopDepthPreferred
    }
    if (-not [string]::IsNullOrWhiteSpace($stage1TopMaskPreferred)) {
        $stage1MaskList = Build-FocusedLambdaList -PrimaryList $stage1MaskList -FallbackList $LambdaPointMvMaskList -Preferred $stage1TopMaskPreferred
    }
    Write-Host "[autoloop] stage1 history-top priority applied depth_first=$stage1TopDepthPreferred mask_first=$stage1TopMaskPreferred"
}

$stage1DepthList = Limit-LambdaList -List $stage1DepthList -MaxCount ([Math]::Max(0, [int]$Stage1FocusMaxDepthValues))
$stage1MaskList = Limit-LambdaList -List $stage1MaskList -MaxCount ([Math]::Max(0, [int]$Stage1FocusMaxMaskValues))
$stage2DepthList = Limit-LambdaList -List $stage2DepthList -MaxCount ([Math]::Max(0, [int]$Stage2FocusMaxDepthValues))
$stage2MaskList = Limit-LambdaList -List $stage2MaskList -MaxCount ([Math]::Max(0, [int]$Stage2FocusMaxMaskValues))
Write-Host "[autoloop] stage1 focused sweep depth_list=$stage1DepthList mask_list=$stage1MaskList"
if ($stage1HistoryFocus -ne $null) {
    Write-Host "[autoloop] stage1 history focus top_ghost=$([string](Fmt-Num (To-DoubleOrNaN($stage1HistoryFocus.top_ghost)) 6)) top_depth=$($stage1HistoryFocus.top_depth) top_mask=$($stage1HistoryFocus.top_mask) used_rows=$($stage1HistoryFocus.used_rows)"
}
Write-Host "[autoloop] stage2 focused sweep depth_list=$stage2DepthList mask_list=$stage2MaskList"
if ($stage2HistoryFocus -ne $null) {
    Write-Host "[autoloop] stage2 history focus top_ghost=$([string](Fmt-Num (To-DoubleOrNaN($stage2HistoryFocus.top_ghost)) 6)) top_depth=$($stage2HistoryFocus.top_depth) top_mask=$($stage2HistoryFocus.top_mask) used_rows=$($stage2HistoryFocus.used_rows)"
}

$baseStageMinPSNRGuard = [double]$StageMinPSNRGuard
$baseStageMinSSIMGuard = [double]$StageMinSSIMGuard
$baseStageMaxWl1Guard = [double]$StageMaxWl1Guard
$baseStage2HistoryQualityAware = [bool]$Stage2HistoryQualityAware
$baseStage2DepthList = [string]$stage2DepthList
$baseStage2MaskList = [string]$stage2MaskList
$baseCyclePromoteMaxPSNRDrop = [double]$CyclePromoteMaxPSNRDrop
$baseCyclePromoteMaxSSIMDrop = [double]$CyclePromoteMaxSSIMDrop
$baseCyclePromoteMaxWl1Rise = [double]$CyclePromoteMaxWl1Rise
$baseCyclePromoteRelaxedMinGhostGain = [double]$CyclePromoteRelaxedMinGhostGain
$baseCyclePromoteRelaxedMaxPSNRDrop = [double]$CyclePromoteRelaxedMaxPSNRDrop
$baseCyclePromoteRelaxedMaxSSIMDrop = [double]$CyclePromoteRelaxedMaxSSIMDrop
$baseCyclePromoteRelaxedMaxWl1Rise = [double]$CyclePromoteRelaxedMaxWl1Rise

$emergencyState = "disabled"
$emergencyShockAnchor = [datetime]::MinValue
$emergencyShockDeadline = [datetime]::MinValue
$emergencyPrimaryExtraCyclesLeft = 0
$emergencyHitTarget = $false
$emergencyFallbackActivated = $false
$emergencyRecoveryActivated = $false
$emergencyRollbackActivated = $false
$emergencyConsecutiveHighGhost = 0
$emergencyLastCycleBestGhost = [double]::NaN
$emergencyLastReason = ""
$emergencyRollbackDepthListEffective = if ([string]::IsNullOrWhiteSpace($EmergencyRollbackDepthList)) { $baseStage2DepthList } else { [string]$EmergencyRollbackDepthList }
$emergencyRollbackMaskListEffective = if ([string]::IsNullOrWhiteSpace($EmergencyRollbackMaskList)) { $baseStage2MaskList } else { [string]$EmergencyRollbackMaskList }
if ([bool]$EmergencyGhostShockEnabled) {
    $emergencyState = "shock_primary"
    $shockWindowMinutes = [Math]::Max(1, [int]$EmergencyShockWindowMinutes)
    $emergencyShockAnchor = $windowT0
    $nowDt = Get-Date
    if (($emergencyShockAnchor -eq [datetime]::MinValue) -or ($emergencyShockAnchor.AddMinutes($shockWindowMinutes) -le $nowDt)) {
        # If persisted window is already expired, start a fresh emergency window from this launch.
        $emergencyShockAnchor = $nowDt
        $emergencyLastReason = "shock_bootstrap_reset_anchor_now"
    } else {
        $emergencyLastReason = "shock_bootstrap"
    }
    $emergencyShockDeadline = $emergencyShockAnchor.AddMinutes($shockWindowMinutes)
    $emergencyPrimaryExtraCyclesLeft = [Math]::Max(0, [int]$EmergencyShockExtraCycles)
    Write-Host "[autoloop] emergency shock enabled window_minutes=$EmergencyShockWindowMinutes anchor=$($emergencyShockAnchor.ToString('yyyy-MM-ddTHH:mm:ss')) deadline=$($emergencyShockDeadline.ToString('yyyy-MM-ddTHH:mm:ss')) extra_cycles=$EmergencyShockExtraCycles target_ghost=$EmergencyShockTargetGhost primary=($EmergencyShockPrimaryDepthList|$EmergencyShockPrimaryMaskList) fallback=($EmergencyShockFallbackDepthList|$EmergencyShockFallbackMaskList)"
}

for ($cycle = 1; $cycle -le [Math]::Max(1, [int]$MaxCycles); $cycle++) {
    if ((Get-Date) -ge $deadline) {
        Write-Host "[autoloop] reach deadline, stop at cycle=$cycle"
        break
    }

    Write-Host "[autoloop] ===== cycle $cycle start ====="
    if ($aggressiveRouteCooldownCycles -gt 0) { $aggressiveRouteCooldownCycles -= 1 }
    $isABRouteMode = ($routeMode -eq "ab_validation")
    $isABBalanceCycle = $isABRouteMode -and ($abRouteVariant -eq "balance")
    $isABAggressiveCycle = $isABRouteMode -and ($abRouteVariant -eq "aggressive")
    $aggressiveRouteActive = (-not $isABRouteMode) -and ($noSubstantialImproveCycles -ge [Math]::Max(1, [int]$AggressiveRouteStartNoSubstantialCycles)) -and ($aggressiveRouteCooldownCycles -le 0)
    if ($isABRouteMode) {
        Write-Host "[autoloop] route_mode=ab_validation (A/B short-run verification, variant=$abRouteVariant)."
    } elseif ($aggressiveRouteActive) {
        Write-Host "[autoloop] route_mode=aggressive (no_substantial=$noSubstantialImproveCycles >= $AggressiveRouteStartNoSubstantialCycles)."
    } else {
        Write-Host "[autoloop] route_mode=main"
    }
    $cycleRouteMode = if ($isABRouteMode) { "ab_validation/$abRouteVariant" } elseif ($aggressiveRouteActive) { "aggressive" } else { "main" }
    $p0Stage2Stats = Get-P0Stage2Stats -WindowMinutes 90
    $p0GatePass = To-BoolLoose -Value $p0Stage2Stats.pass -Default $false
    $p0GateReason = [string]$p0Stage2Stats.reason
    $stage2HistoryFocusEffective = $stage2HistoryFocus
    $cycleForceStage2DepthList = ""
    $cycleForceStage2MaskList = ""
    $cycleEmergencyProfile = "default"
    $cycleEmergencyReason = ""
    $cycleApplyShockParamSet = $false
    if ([bool]$EmergencyGhostShockEnabled) {
        $targetGhost = [double]$EmergencyShockTargetGhost
        if ((-not [double]::IsNaN($emergencyLastCycleBestGhost)) -and ($emergencyLastCycleBestGhost -le $targetGhost)) {
            $emergencyHitTarget = $true
        } elseif ((-not [double]::IsInfinity($globalBestGhost)) -and (-not [double]::IsNaN($globalBestGhost)) -and ($globalBestGhost -le $targetGhost)) {
            $emergencyHitTarget = $true
        }
        if ($emergencyHitTarget -and ($emergencyState -ne "recovery")) {
            $emergencyState = "recovery"
            $emergencyRecoveryActivated = $true
            $emergencyLastReason = "hit_target_ghost<=$targetGhost"
        }
        if (($emergencyState -eq "shock_primary") -and ($emergencyShockDeadline -ne [datetime]::MinValue) -and ((Get-Date) -gt $emergencyShockDeadline)) {
            if ($emergencyPrimaryExtraCyclesLeft -gt 0) {
                $emergencyPrimaryExtraCyclesLeft -= 1
                $emergencyLastReason = "window_expired_use_extra_cycle"
            } else {
                $emergencyState = "shock_fallback"
                $emergencyFallbackActivated = $true
                $emergencyLastReason = "window_expired_switch_fallback"
            }
        }
        if (($emergencyConsecutiveHighGhost -ge [Math]::Max(1, [int]$EmergencyShockFailConsecutiveLimit)) -and ($emergencyState -like "shock*")) {
            if ($emergencyState -eq "shock_primary") {
                $emergencyState = "shock_fallback"
                $emergencyFallbackActivated = $true
                $emergencyLastReason = "high_ghost_streak=$emergencyConsecutiveHighGhost -> switch_fallback"
                # Give fallback one full chance before rolling back to steady.
                $emergencyConsecutiveHighGhost = 0
            } else {
                $emergencyState = "rollback_steady"
                $emergencyRollbackActivated = $true
                $emergencyLastReason = "high_ghost_streak=$emergencyConsecutiveHighGhost"
            }
        }

        # Reset to base each cycle first, then apply emergency profile.
        $StageMinPSNRGuard = $baseStageMinPSNRGuard
        $StageMinSSIMGuard = $baseStageMinSSIMGuard
        $StageMaxWl1Guard = $baseStageMaxWl1Guard
        $Stage2HistoryQualityAware = $baseStage2HistoryQualityAware
        $CyclePromoteMaxPSNRDrop = $baseCyclePromoteMaxPSNRDrop
        $CyclePromoteMaxSSIMDrop = $baseCyclePromoteMaxSSIMDrop
        $CyclePromoteMaxWl1Rise = $baseCyclePromoteMaxWl1Rise
        $CyclePromoteRelaxedMinGhostGain = $baseCyclePromoteRelaxedMinGhostGain
        $CyclePromoteRelaxedMaxPSNRDrop = $baseCyclePromoteRelaxedMaxPSNRDrop
        $CyclePromoteRelaxedMaxSSIMDrop = $baseCyclePromoteRelaxedMaxSSIMDrop
        $CyclePromoteRelaxedMaxWl1Rise = $baseCyclePromoteRelaxedMaxWl1Rise

        switch ($emergencyState) {
            "shock_primary" {
                $cycleEmergencyProfile = "shock_primary"
                $cycleEmergencyReason = $emergencyLastReason
                $cycleForceStage2DepthList = [string]$EmergencyShockPrimaryDepthList
                $cycleForceStage2MaskList = [string]$EmergencyShockPrimaryMaskList
                $cycleApplyShockParamSet = $true
                $Stage2HistoryQualityAware = $false
                $stage2HistoryFocusEffective = $null
                $StageMinPSNRGuard = [double]$EmergencyShockMinPSNRGuard
                $StageMinSSIMGuard = [double]$EmergencyShockMinSSIMGuard
                $StageMaxWl1Guard = [double]$EmergencyShockMaxWl1Guard
                $CyclePromoteMaxPSNRDrop = 10.0
                $CyclePromoteMaxSSIMDrop = 1.0
                $CyclePromoteMaxWl1Rise = 1.0
                $CyclePromoteRelaxedMinGhostGain = 0.0
                $CyclePromoteRelaxedMaxPSNRDrop = 10.0
                $CyclePromoteRelaxedMaxSSIMDrop = 1.0
                $CyclePromoteRelaxedMaxWl1Rise = 1.0
            }
            "shock_fallback" {
                $cycleEmergencyProfile = "shock_fallback"
                $cycleEmergencyReason = $emergencyLastReason
                $cycleForceStage2DepthList = [string]$EmergencyShockFallbackDepthList
                $cycleForceStage2MaskList = [string]$EmergencyShockFallbackMaskList
                $cycleApplyShockParamSet = $true
                $Stage2HistoryQualityAware = $false
                $stage2HistoryFocusEffective = $null
                $StageMinPSNRGuard = [double]$EmergencyShockMinPSNRGuard
                $StageMinSSIMGuard = [double]$EmergencyShockMinSSIMGuard
                $StageMaxWl1Guard = [double]$EmergencyShockMaxWl1Guard
                $CyclePromoteMaxPSNRDrop = 10.0
                $CyclePromoteMaxSSIMDrop = 1.0
                $CyclePromoteMaxWl1Rise = 1.0
                $CyclePromoteRelaxedMinGhostGain = 0.0
                $CyclePromoteRelaxedMaxPSNRDrop = 10.0
                $CyclePromoteRelaxedMaxSSIMDrop = 1.0
                $CyclePromoteRelaxedMaxWl1Rise = 1.0
            }
            "recovery" {
                $cycleEmergencyProfile = "recovery"
                $cycleEmergencyReason = $emergencyLastReason
                $cycleForceStage2DepthList = "0.001,0.0015"
                $cycleForceStage2MaskList = "0,0.0005"
                $Stage2HistoryQualityAware = [bool]$EmergencyRecoveryEnableHistoryQualityAware
                $StageMinPSNRGuard = [double]$EmergencyRecoveryMinPSNRGuard
                $StageMinSSIMGuard = [double]$EmergencyRecoveryMinSSIMGuard
                $StageMaxWl1Guard = [double]$EmergencyRecoveryMaxWl1Guard
                $QualityGuardMode = "layered"
                $PromotionGhostDelta = 0.02
            }
            "rollback_steady" {
                $cycleEmergencyProfile = "rollback_steady"
                $cycleEmergencyReason = $emergencyLastReason
                $cycleForceStage2DepthList = [string]$emergencyRollbackDepthListEffective
                $cycleForceStage2MaskList = [string]$emergencyRollbackMaskListEffective
            }
            default {
                $cycleEmergencyProfile = "default"
            }
        }

        Write-Host "[autoloop] emergency profile=$cycleEmergencyProfile reason=$cycleEmergencyReason guard(psnr>=$StageMinPSNRGuard ssim>=$StageMinSSIMGuard wl1<=$StageMaxWl1Guard) stage2_grid=($cycleForceStage2DepthList|$cycleForceStage2MaskList)"
    }
    $preCycleBestGhost = $globalBestGhost
    $preCycleBestPsnr = $globalBestPsnr
    $preCycleBestSsim = $globalBestSsim
    $preCycleBestWl1 = $globalBestWl1
    $cycleStage1DepthList = $stage1DepthList
    $cycleStage1MaskList = $stage1MaskList
    $cycleStage1GhostLag = [double]::NaN
    $stage1GhostPressureOn = $false
    $stage1InfraNoOutputFailure = $false
    $stage1InfraNoOutputForStage2FastStop = $false
    $skipDeepStagesDueInfra = $false
    $skipDeepStagesDueInfraReason = ""
    $stage2HasPotential = $true
    $stage2PotentialReason = "not_checked"
    $skipDeepStagesByPotential = $false
    $skipDeepStagesByPotentialReason = ""
    $skipDeepStagesCombined = $false
    $skipDeepStagesCombinedReason = ""
    $shouldStopByABValidation = $false
    $stopByABValidationReason = ""
    $activeLane = "lane_a"
    $laneABestSnapshot = $null
    $laneBBestSnapshot = $null
    $cycleGuardTier = ""
    $cycleDecisionReason = "stage2_not_run"
    $cycleRollbackReason = ""
    $cycleStage1PointmapSource = [string]$Stage1PointmapSource
    $cycleStage1NoOutputTimeoutSec = [int]$Stage1ModalRunNoOutputTimeoutSec
    $cycleStage1NoOutputMaxRetries = [int]$Stage1ModalRunNoOutputMaxRetries
    if ([bool]$EmergencyGhostShockEnabled -and ($cycleEmergencyProfile -in @("shock_primary", "shock_fallback"))) {
        if ($cycleStage1PointmapSource.Trim().ToLowerInvariant() -eq "depth_unproject") {
            # Emergency throughput: keep depth_unproject to avoid point_head dead loops,
            # and only tighten no-output budget to move to the next candidate faster.
            $emergencyStage1NoOutputCap = [Math]::Max(180, [int]$EmergencyStage1NoOutputTimeoutSec)
            $cycleStage1NoOutputTimeoutSec = [Math]::Min($cycleStage1NoOutputTimeoutSec, $emergencyStage1NoOutputCap)
            $cycleStage1NoOutputMaxRetries = [Math]::Min($cycleStage1NoOutputMaxRetries, 0)
            Write-Host "[autoloop] emergency stage1 throughput override: pointmap_source=depth_unproject no_output_timeout_sec=$cycleStage1NoOutputTimeoutSec no_output_max_retries=$cycleStage1NoOutputMaxRetries"
        }
    }

    if ($isABRouteMode -or [bool]$ForceStage2Only) {
        $stage1SkipReason = if ([bool]$ForceStage2Only) { "force_stage2_only_single_run" } else { "ab_route_validation_mode" }
        $s1 = New-SkippedStageResult `
            -StageName ("cycle{0:D3}_stage1_strong" -f $cycle) `
            -PointTargetBlendMvPolicy "strong_to_depth" `
            -PointmapSource $cycleStage1PointmapSource `
            -PseudoGeomSubdir $currPseudo `
            -ResumeCkpt $currResume `
            -Reason $stage1SkipReason
        if ([bool]$ForceStage2Only) {
            Write-Host "[autoloop] stage1 skipped in force_stage2_only mode."
        } else {
            Write-Host "[autoloop] stage1 skipped in A/B route mode."
        }
    } else {
        if (($stage1HistoryFocus -ne $null) -and (-not [double]::IsNaN((To-DoubleOrNaN($stage1HistoryFocus.top_ghost)))) -and (-not [double]::IsNaN($globalBestGhost)) -and (-not [double]::IsInfinity($globalBestGhost))) {
            $cycleStage1GhostLag = (To-DoubleOrNaN($stage1HistoryFocus.top_ghost)) - $globalBestGhost
        }
        if ($globalBestHintApplyStage1 -and (-not [string]::IsNullOrWhiteSpace($globalBestLambdaDepthHint))) {
            $cycleStage1DepthList = Build-FocusedLambdaList -PrimaryList $cycleStage1DepthList -FallbackList $LambdaPointMvDepthList -Preferred $globalBestLambdaDepthHint
        }
        if ($globalBestHintApplyStage1 -and (-not [string]::IsNullOrWhiteSpace($globalBestLambdaMaskHint))) {
            $cycleStage1MaskList = Build-FocusedLambdaList -PrimaryList $cycleStage1MaskList -FallbackList $LambdaPointMvMaskList -Preferred $globalBestLambdaMaskHint
        }
        if ($Stage1PrioritizeHistoryTopAfterCrossHint -and ($stage1HistoryFocus -ne $null)) {
            $cycleStage1TopDepth = [string]$stage1HistoryFocus.top_depth
            $cycleStage1TopMask = [string]$stage1HistoryFocus.top_mask
            if (-not [string]::IsNullOrWhiteSpace($cycleStage1TopDepth)) {
                $cycleStage1DepthList = Build-FocusedLambdaList -PrimaryList $cycleStage1DepthList -FallbackList $LambdaPointMvDepthList -Preferred $cycleStage1TopDepth
            }
            if (-not [string]::IsNullOrWhiteSpace($cycleStage1TopMask)) {
                $cycleStage1MaskList = Build-FocusedLambdaList -PrimaryList $cycleStage1MaskList -FallbackList $LambdaPointMvMaskList -Preferred $cycleStage1TopMask
            }
        }
        # Re-apply stage1 focus caps after in-cycle hint merges to prevent list re-expansion.
        $cycleStage1DepthList = Limit-LambdaList -List $cycleStage1DepthList -MaxCount ([Math]::Max(0, [int]$Stage1FocusMaxDepthValues))
        $cycleStage1MaskList = Limit-LambdaList -List $cycleStage1MaskList -MaxCount ([Math]::Max(0, [int]$Stage1FocusMaxMaskValues))
        Write-Host "[autoloop] stage1 cycle focus enforced depth_list=$cycleStage1DepthList mask_list=$cycleStage1MaskList"
        $stage1GhostPressureOn = ((-not [double]::IsNaN($cycleStage1GhostLag)) -and ($cycleStage1GhostLag -ge [Math]::Max(0.0, [double]$Stage1GhostPressureLagThreshold)))
        if ($stage1GhostPressureOn) {
            $stage1GhostPressurePreserveMasks = @()
            if ([bool]$Stage1GhostPressurePreserveMaskHints) {
                if ($globalBestHintApplyStage1 -and (-not [string]::IsNullOrWhiteSpace($globalBestLambdaMaskHint))) {
                    $stage1GhostPressurePreserveMasks += [string]$globalBestLambdaMaskHint
                }
                if ($stage1HistoryFocus -ne $null) {
                    $histTopMask = [string]$stage1HistoryFocus.top_mask
                    if (-not [string]::IsNullOrWhiteSpace($histTopMask)) {
                        $stage1GhostPressurePreserveMasks += $histTopMask
                    }
                }
                $stage1GhostPressurePreserveMasks = @(
                    $stage1GhostPressurePreserveMasks |
                        Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
                        Select-Object -Unique
                )
            }
            $cycleStage1MaskList = Filter-LambdaListNumericRange `
                -List $cycleStage1MaskList `
                -MinValue ([double]$Stage1GhostPressureMaskMinValue) `
                -MaxValue ([double]$Stage1GhostPressureMaskMaxValue) `
                -FallbackList $LambdaPointMvMaskList
            foreach ($keepMask in @($stage1GhostPressurePreserveMasks)) {
                $cycleStage1MaskList = Build-FocusedLambdaList `
                    -PrimaryList $cycleStage1MaskList `
                    -FallbackList $LambdaPointMvMaskList `
                    -Preferred ([string]$keepMask)
            }
            $cycleStage1MaskList = Limit-LambdaList `
                -List $cycleStage1MaskList `
                -MaxCount ([Math]::Max(1, [int]$Stage1GhostPressureMaxMaskValues))
            Write-Host "[autoloop] stage1 ghost-pressure enabled lag=$([string](Fmt-Num $cycleStage1GhostLag 6)) threshold=$([string](Fmt-Num ([double]$Stage1GhostPressureLagThreshold) 6)) mix=$Stage1GhostPressureMaskSoftMix soft_hit=$Stage1GhostPressureMaskSoftHitThr stride=$Stage1GhostPressureStride pairs=$Stage1GhostPressureDepthMaxPairs support=$Stage1GhostPressureDepthSupportMode/$Stage1GhostPressureDepthSupportFloor"
            Write-Host "[autoloop] stage1 ghost-pressure mask range applied: min=$Stage1GhostPressureMaskMinValue max=$Stage1GhostPressureMaskMaxValue max_count=$Stage1GhostPressureMaxMaskValues mask_list=$cycleStage1MaskList"
            if ($stage1GhostPressurePreserveMasks.Count -gt 0) {
                Write-Host "[autoloop] stage1 ghost-pressure preserve mask hints: keep=$($stage1GhostPressurePreserveMasks -join ',')"
            }
        }
        if ($noSubstantialImproveCycles -ge [Math]::Max(1, [int]$Stage1AggressiveFocusNoImproveCycles)) {
            $cycleStage1DepthList = Limit-LambdaList -List $cycleStage1DepthList -MaxCount ([Math]::Max(1, [int]$Stage1AggressiveFocusMaxDepthValues))
            $cycleStage1MaskList = Limit-LambdaList -List $cycleStage1MaskList -MaxCount ([Math]::Max(1, [int]$Stage1AggressiveFocusMaxMaskValues))
            Write-Host "[autoloop] stage1 aggressive focus enabled no_substantial=$noSubstantialImproveCycles depth_list=$cycleStage1DepthList mask_list=$cycleStage1MaskList"
        }
        $s1 = Invoke-GhostStage `
            -StageName ("cycle{0:D3}_stage1_strong" -f $cycle) `
            -PointTargetBlendMvPolicy "strong_to_depth" `
            -PointmapSource $cycleStage1PointmapSource `
            -PseudoGeomSubdir $currPseudo `
            -ResumeCkpt $currResume `
            -Overrides @{
                LaneId = "lane_a"
                CandidateFamily = "stage1_training"
                LambdaPointMvDepthList = $cycleStage1DepthList
                LambdaPointMvMaskList = $cycleStage1MaskList
                NoImprovePatience = $stage1NoImprovePatienceEffective
                PointMvMaskSoftMix = $(if ($stage1GhostPressureOn) { $Stage1GhostPressureMaskSoftMix } else { $BasePointMvMaskSoftMix })
                PointMvMaskSoftHitThr = $(if ($stage1GhostPressureOn) { $Stage1GhostPressureMaskSoftHitThr } else { $BasePointMvMaskSoftHitThr })
                PointMvMaskMinTgtFgRatio = $(if ($stage1GhostPressureOn) { $Stage1GhostPressureMaskMinTgtFgRatio } else { $BasePointMvMaskMinTgtFgRatio })
                PointMvStride = $(if ($stage1GhostPressureOn) { $Stage1GhostPressureStride } else { $BasePointMvStride })
                PointMvDepthMaxPairs = $(if ($stage1GhostPressureOn) { $Stage1GhostPressureDepthMaxPairs } else { $BasePointMvDepthMaxPairs })
                PointMvDepthSupportMode = $(if ($stage1GhostPressureOn) { $Stage1GhostPressureDepthSupportMode } else { $BasePointMvDepthSupportMode })
                PointMvDepthSupportFloor = $(if ($stage1GhostPressureOn) { $Stage1GhostPressureDepthSupportFloor } else { $BasePointMvDepthSupportFloor })
                ModalRunNoOutputTimeoutSec = [int]$cycleStage1NoOutputTimeoutSec
                ModalRunNoOutputMaxRetries = [int]$cycleStage1NoOutputMaxRetries
                InfraNoOutputStopConsecutive = [int]$Stage1InfraNoOutputStopConsecutive
            }
        Safe-WriteInterimAutoloopArtifacts `
            -History @($history.ToArray()) `
            -CurrentCycle $cycle `
            -CurrentStages @($s1) `
            -Deadline $deadline `
            -GlobalBestGhost $globalBestGhost `
            -GlobalBestPsnr $globalBestPsnr `
            -GlobalBestSsim $globalBestSsim `
            -GlobalBestWl1 $globalBestWl1 `
            -CurrResume $currResume `
            -CurrPseudo $currPseudo `
            -PendingTuneAction $pendingTuneAction

        $s1GhostForLag = To-DoubleOrNaN($s1.ghost)
        if ((-not [double]::IsNaN($s1GhostForLag)) -and (-not [double]::IsInfinity($preCycleBestGhost)) -and (-not [double]::IsNaN($preCycleBestGhost))) {
            $cycleStage1GhostLag = $s1GhostForLag - $preCycleBestGhost
            Write-Host "[autoloop] stage1 actual ghost lag refreshed: s1=$(Fmt-Num $s1GhostForLag 6) pre_cycle_best=$(Fmt-Num $preCycleBestGhost 6) lag=$(Fmt-Num $cycleStage1GhostLag 6)"
        }

        $stage1InfraNoOutputFailure = Test-StageInfraNoOutputFailure `
            -RawSweepCsvPath ([string]$s1.raw_sweep_csv) `
            -MinConsecutive $InfraNoOutputStageAbortThreshold
        $stage1InfraNoOutputForStage2FastStop = Test-StageInfraNoOutputFailure `
            -RawSweepCsvPath ([string]$s1.raw_sweep_csv) `
            -MinConsecutive ([Math]::Max(1, [int]$Stage1InfraNoOutputStopConsecutive))
        $skipDeepStagesDueInfra = $stage1InfraNoOutputFailure
        if ($skipDeepStagesDueInfra) {
            $skipDeepStagesDueInfraReason = "stage1_infra_no_output_threshold_reached(thr=$InfraNoOutputStageAbortThreshold)"
            Write-Host "[autoloop] stage1 infra no-output detected; keep stage2, skip stage3-5 (reason=$skipDeepStagesDueInfraReason)"
        } elseif ($stage1InfraNoOutputForStage2FastStop) {
            Write-Host "[autoloop] stage1 infra no-output fast-stop signal detected (no deep-stage skip); stage2 policy may be relaxed by config."
        }
    }

    $resumeForS2 = Resolve-NextStageResumeCkpt `
        -PrevStage $s1 `
        -FallbackResume $currResume `
        -GlobalBestGhostRef $globalBestGhost `
        -PromoteGhostMargin $StageResumePromoteGhostMargin
    $cycleStage2DepthList = $stage2DepthList
    $cycleStage2MaskList = $stage2MaskList
    $stage2MaskPreserveHints = @()
    if (-not [string]::IsNullOrWhiteSpace($globalBestLambdaMaskHint)) {
        $stage2MaskPreserveHints += [string]$globalBestLambdaMaskHint
    }
    if ($stage2HistoryFocusEffective -ne $null) {
        $stage2TopMaskHint = [string]$stage2HistoryFocusEffective.top_mask
        if (-not [string]::IsNullOrWhiteSpace($stage2TopMaskHint)) {
            $stage2MaskPreserveHints += $stage2TopMaskHint
        }
    }
    $stage2DisableZeroPreserveByLag = $false
    if ((-not [double]::IsNaN($cycleStage1GhostLag)) -and
        (([double]$cycleStage1GhostLag) -ge [Math]::Max(0.0, [double]$Stage2MaskZeroPreserveLagThreshold)) -and
        [bool]$Stage2MaskDisableZeroPreserveWhenLagged) {
        $stage2DisableZeroPreserveByLag = $true
    }
    if ([bool]$Stage2MaskHardPreferZeroPreserve -and
        (-not [double]::IsNaN([double]$Stage2MaskHardMaxValue)) -and
        ([double]$Stage2MaskHardMaxValue -ge 0.0) -and
        ([Math]::Max([int]$Stage2FocusMaxMaskValues, [int]$Stage2AggressiveFocusMaxMaskValues) -gt 1) -and
        (-not $stage2DisableZeroPreserveByLag)) {
        $stage2MaskPreserveHints += "0"
    }
    if ($stage2DisableZeroPreserveByLag) {
        Write-Host "[autoloop] stage2 zero-mask preserve disabled by lag: lag=$([string](Fmt-Num $cycleStage1GhostLag 6)) threshold=$([string](Fmt-Num ([double]$Stage2MaskZeroPreserveLagThreshold) 6))"
    }
    $stage2MaskPreserveHints = @(
        $stage2MaskPreserveHints |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_.Trim() } |
            Select-Object -Unique
    )
    $stage2PointTargetConsensusAlphaFloor = $PointTargetConsensusAlphaFloor
    $stage2PointMvMaskSoftMix = $BasePointMvMaskSoftMix
    $stage2PointMvMaskSoftHitThr = $BasePointMvMaskSoftHitThr
    $stage2PointMvMaskMinTgtFgRatio = $BasePointMvMaskMinTgtFgRatio
    $stage2PointMvStride = $BasePointMvStride
    $stage2PointMvDepthMaxPairs = $BasePointMvDepthMaxPairs
    $stage2PointMvDepthSupportMode = $BasePointMvDepthSupportMode
    $stage2PointMvDepthSupportFloor = $BasePointMvDepthSupportFloor
    $stage2InfraNoOutputStopConsecutive = [int]$Stage2InfraNoOutputStopConsecutive
    $stage2ModalRunNoOutputTimeoutSecEffective = [int]$Stage2ModalRunNoOutputTimeoutSec
    $stage2ModalRunNoOutputMaxRetriesEffective = [int]$Stage2ModalRunNoOutputMaxRetries
    $stage2InfraNoOutputStopConsecutiveEffective = [int]$stage2InfraNoOutputStopConsecutive
    if ($stage1InfraNoOutputForStage2FastStop -and [bool]$Stage2EnableInfraRecoveryNoOutputRelax) {
        $isEmergencyShockProfile = [bool]$EmergencyGhostShockEnabled -and ($cycleEmergencyProfile -in @("shock_primary", "shock_fallback"))
        if ($isEmergencyShockProfile) {
            Write-Host "[autoloop] stage2 infra recovery relax skipped in emergency shock profile=$cycleEmergencyProfile"
        } else {
            $stage2ModalRunNoOutputTimeoutSecEffective = [Math]::Max($stage2ModalRunNoOutputTimeoutSecEffective, [int]$Stage2InfraRecoveryNoOutputTimeoutSec)
            $stage2ModalRunNoOutputMaxRetriesEffective = [Math]::Max($stage2ModalRunNoOutputMaxRetriesEffective, [int]$Stage2InfraRecoveryNoOutputMaxRetries)
            $stage2InfraNoOutputStopConsecutiveEffective = [Math]::Max($stage2InfraNoOutputStopConsecutiveEffective, [int]$Stage2InfraRecoveryMinStopConsecutive)
            Write-Host "[autoloop] stage2 infra recovery relax applied timeout_sec=$stage2ModalRunNoOutputTimeoutSecEffective max_retries=$stage2ModalRunNoOutputMaxRetriesEffective infra_stop_consecutive=$stage2InfraNoOutputStopConsecutiveEffective"
        }
    }

    if ($isABRouteMode) {
        if ($isABBalanceCycle) {
            $cycleStage2DepthList = [string]$ABBalanceDepthList
            $cycleStage2MaskList = [string]$ABBalanceMaskList
            Write-Host "[autoloop] stage2 A/B balance candidate depth_list=$cycleStage2DepthList mask_list=$cycleStage2MaskList"
        } else {
            $cycleStage2DepthList = [string]$ABAggressiveDepthList
            $cycleStage2MaskList = [string]$ABAggressiveMaskList
            $stage2PointTargetConsensusAlphaFloor = [Math]::Max([double]$AggressiveStage2PointTargetConsensusAlphaFloor, 0.40)
            $stage2PointMvMaskSoftMix = [double]$AggressiveStage2PointMvMaskSoftMix
            $stage2PointMvMaskSoftHitThr = [Math]::Max([double]$AggressiveStage2PointMvMaskSoftHitThr, 0.55)
            $stage2PointMvMaskMinTgtFgRatio = [Math]::Max([double]$AggressiveStage2PointMvMaskMinTgtFgRatio, 0.03)
            $stage2PointMvStride = [Math]::Min([int]$AggressiveStage2PointMvStride, 1)
            $stage2PointMvDepthMaxPairs = [Math]::Max([int]$AggressiveStage2PointMvDepthMaxPairs, 3)
            $stage2PointMvDepthSupportMode = [string]$AggressiveStage2PointMvDepthSupportMode
            $stage2PointMvDepthSupportFloor = [Math]::Max([double]$AggressiveStage2PointMvDepthSupportFloor, 0.1)
            Write-Host "[autoloop] stage2 A/B aggressive candidate depth_list=$cycleStage2DepthList mask_list=$cycleStage2MaskList mix=$stage2PointMvMaskSoftMix soft_hit=$stage2PointMvMaskSoftHitThr min_fg=$stage2PointMvMaskMinTgtFgRatio stride=$stage2PointMvStride pairs=$stage2PointMvDepthMaxPairs support=$stage2PointMvDepthSupportMode/$stage2PointMvDepthSupportFloor alpha_floor=$stage2PointTargetConsensusAlphaFloor"
        }
    } else {
        if ($globalBestHintApplyStage2 -and (-not [string]::IsNullOrWhiteSpace($globalBestLambdaDepthHint))) {
            $cycleStage2DepthList = Build-FocusedLambdaList -PrimaryList $cycleStage2DepthList -FallbackList $Stage2LambdaPointMvDepthList -Preferred $globalBestLambdaDepthHint
        }
        if ($globalBestHintApplyStage2 -and (-not [string]::IsNullOrWhiteSpace($globalBestLambdaMaskHint))) {
            $cycleStage2MaskList = Build-FocusedLambdaList -PrimaryList $cycleStage2MaskList -FallbackList $Stage2LambdaPointMvMaskList -Preferred $globalBestLambdaMaskHint
        }
        $stage2GhostPressureOn = $stage1InfraNoOutputFailure -or ((-not [double]::IsNaN($cycleStage1GhostLag)) -and ($cycleStage1GhostLag -ge [Math]::Max(0.0, [double]$Stage2GhostPressureLagThreshold)))
        if ($stage2GhostPressureOn) {
            $cycleStage2MaskList = Filter-LambdaListNumericRange `
                -List $cycleStage2MaskList `
                -MinValue ([double]$Stage2GhostPressureMaskMinValue) `
                -MaxValue ([double]$Stage2GhostPressureMaskMaxValue) `
                -FallbackList $Stage2LambdaPointMvMaskList
            $stage2PointTargetConsensusAlphaFloor = $Stage2GhostPressureConsensusAlphaFloor
            # Keep ghost-pressure as a lower bound so cycle-level soft_mix tuning
            # (BasePointMvMaskSoftMix) can still take effect under pressure.
            $stage2PointMvMaskSoftMix = [Math]::Max([double]$stage2PointMvMaskSoftMix, [double]$Stage2GhostPressureMaskSoftMix)
            $stage2PointMvMaskSoftHitThr = $Stage2GhostPressureMaskSoftHitThr
            $stage2PointMvMaskMinTgtFgRatio = $Stage2GhostPressureMaskMinTgtFgRatio
            $stage2PointMvStride = $Stage2GhostPressureStride
            $stage2PointMvDepthMaxPairs = $Stage2GhostPressureDepthMaxPairs
            $stage2PointMvDepthSupportMode = $Stage2GhostPressureDepthSupportMode
            $stage2PointMvDepthSupportFloor = $Stage2GhostPressureDepthSupportFloor
            Write-Host "[autoloop] stage2 ghost-pressure enabled lag=$([string](Fmt-Num $cycleStage1GhostLag 6)) threshold=$([string](Fmt-Num ([double]$Stage2GhostPressureLagThreshold) 6)) mix_floor=$Stage2GhostPressureMaskSoftMix mix_effective=$stage2PointMvMaskSoftMix soft_hit=$Stage2GhostPressureMaskSoftHitThr stride=$Stage2GhostPressureStride pairs=$Stage2GhostPressureDepthMaxPairs support=$Stage2GhostPressureDepthSupportMode/$Stage2GhostPressureDepthSupportFloor alpha_floor=$Stage2GhostPressureConsensusAlphaFloor"
            Write-Host "[autoloop] stage2 ghost-pressure mask range applied: min=$Stage2GhostPressureMaskMinValue max=$Stage2GhostPressureMaskMaxValue mask_list=$cycleStage2MaskList"
        }
        if ($aggressiveRouteActive) {
            if (-not [string]::IsNullOrWhiteSpace($AggressiveStage2LambdaPointMvDepthList)) { $cycleStage2DepthList = $AggressiveStage2LambdaPointMvDepthList }
            if (-not [string]::IsNullOrWhiteSpace($AggressiveStage2LambdaPointMvMaskList)) { $cycleStage2MaskList = $AggressiveStage2LambdaPointMvMaskList }
            $cycleStage2DepthList = Limit-LambdaList -List $cycleStage2DepthList -MaxCount ([Math]::Max(1, [int]$Stage2AggressiveFocusMaxDepthValues))
            $stage2PointTargetConsensusAlphaFloor = $AggressiveStage2PointTargetConsensusAlphaFloor
            $stage2PointMvMaskSoftMix = $AggressiveStage2PointMvMaskSoftMix
            $stage2PointMvMaskSoftHitThr = $AggressiveStage2PointMvMaskSoftHitThr
            $stage2PointMvMaskMinTgtFgRatio = $AggressiveStage2PointMvMaskMinTgtFgRatio
            $stage2PointMvStride = $AggressiveStage2PointMvStride
            $stage2PointMvDepthMaxPairs = $AggressiveStage2PointMvDepthMaxPairs
            $stage2PointMvDepthSupportMode = $AggressiveStage2PointMvDepthSupportMode
            $stage2PointMvDepthSupportFloor = $AggressiveStage2PointMvDepthSupportFloor
            Write-Host "[autoloop] stage2 aggressive route applied depth_list=$cycleStage2DepthList mask_list=$cycleStage2MaskList mix=$stage2PointMvMaskSoftMix soft_hit=$stage2PointMvMaskSoftHitThr stride=$stage2PointMvStride pairs=$stage2PointMvDepthMaxPairs support=$stage2PointMvDepthSupportMode/$stage2PointMvDepthSupportFloor alpha_floor=$stage2PointTargetConsensusAlphaFloor max_depth_count=$Stage2AggressiveFocusMaxDepthValues"
        }
    }
    if ((-not [double]::IsNaN([double]$Stage2DepthHardMinValue)) -and ([double]$Stage2DepthHardMinValue -gt 0.0)) {
        $cycleStage2DepthList = Filter-LambdaListNumericRange `
            -List $cycleStage2DepthList `
            -MinValue ([double]$Stage2DepthHardMinValue) `
            -MaxValue ([double]::PositiveInfinity) `
            -FallbackList $Stage2LambdaPointMvDepthList
        Write-Host "[autoloop] stage2 depth hard-floor applied: min=$Stage2DepthHardMinValue depth_list=$cycleStage2DepthList"
    }
    if ((-not [double]::IsNaN([double]$Stage2MaskHardMaxValue)) -and ([double]$Stage2MaskHardMaxValue -ge 0.0)) {
        $stage2MaskFocusMaxCountEffective = [Math]::Max(1, [int]$Stage2FocusMaxMaskValues)
        if ($aggressiveRouteActive -or $isABAggressiveCycle) {
            $stage2MaskFocusMaxCountEffective = [Math]::Max(
                $stage2MaskFocusMaxCountEffective,
                [Math]::Max(1, [int]$Stage2AggressiveFocusMaxMaskValues)
            )
        }
        $stage2MaskLagExpandOn = $false
        $stage2MaskLagExpandThresholdEffective = [Math]::Max(0.0, [double]$Stage2MaskLagExpandThreshold)
        if ((-not [double]::IsNaN($cycleStage1GhostLag)) -and ($cycleStage1GhostLag -ge $stage2MaskLagExpandThresholdEffective)) {
            $stage2MaskLagExpandOn = $true
            $stage2MaskLagMinCount = [Math]::Max(1, [int]$Stage2MaskHardMinCountWhenLagged)
            if ($stage2MaskFocusMaxCountEffective -lt $stage2MaskLagMinCount) {
                $stage2MaskFocusMaxCountEffective = $stage2MaskLagMinCount
            }
        }
        $cycleStage2MaskList = Filter-LambdaListNumericRange `
            -List $cycleStage2MaskList `
            -MinValue 0.0 `
            -MaxValue ([double]$Stage2MaskHardMaxValue) `
            -FallbackList $Stage2LambdaPointMvMaskList
        foreach ($keepMask in @($stage2MaskPreserveHints)) {
            $keepVal = To-DoubleOrNaN ([string]$keepMask)
            if ((-not [double]::IsNaN($keepVal)) -and ($keepVal -ge 0.0) -and ($keepVal -le [double]$Stage2MaskHardMaxValue)) {
                $cycleStage2MaskList = Build-FocusedLambdaList `
                    -PrimaryList $cycleStage2MaskList `
                    -FallbackList $Stage2LambdaPointMvMaskList `
                    -Preferred ([string]$keepMask)
            }
        }
        $cycleStage2MaskList = Limit-LambdaList -List $cycleStage2MaskList -MaxCount $stage2MaskFocusMaxCountEffective
        if ($stage2MaskPreserveHints.Count -gt 0) {
            Write-Host "[autoloop] stage2 mask hard-cap preserve hints: keep=$($stage2MaskPreserveHints -join ',')"
        }
        if ($stage2MaskLagExpandOn) {
            Write-Host "[autoloop] stage2 mask lag-expand enabled: lag=$([string](Fmt-Num $cycleStage1GhostLag 6)) threshold=$([string](Fmt-Num $stage2MaskLagExpandThresholdEffective 6)) min_count=$Stage2MaskHardMinCountWhenLagged"
        }
        Write-Host "[autoloop] stage2 mask hard-cap applied: max=$Stage2MaskHardMaxValue max_count=$stage2MaskFocusMaxCountEffective mask_list=$cycleStage2MaskList"
    }
    if (-not [string]::IsNullOrWhiteSpace($cycleForceStage2DepthList)) {
        $cycleStage2DepthList = [string]$cycleForceStage2DepthList
    }
    if (-not [string]::IsNullOrWhiteSpace($cycleForceStage2MaskList)) {
        $cycleStage2MaskList = [string]$cycleForceStage2MaskList
    }
    if ($cycleApplyShockParamSet) {
        # Shock profile: prioritize immediate ghost suppression over quality stability.
        $stage2PointTargetConsensusAlphaFloor = [double]$PointTargetConsensusAlphaFloor
        $stage2PointMvMaskSoftMix = 0.28
        $stage2PointMvMaskSoftHitThr = 0.50
        $stage2PointMvMaskMinTgtFgRatio = 0.0
        $stage2PointMvStride = 2
        $stage2PointMvDepthMaxPairs = 2
        $stage2PointMvDepthSupportMode = "off"
        $stage2PointMvDepthSupportFloor = 0.0
    }
    if ([bool]$EmergencyGhostShockEnabled -and ($cycleEmergencyProfile -ne "default")) {
        Write-Host "[autoloop] stage2 emergency override applied profile=$cycleEmergencyProfile depth_list=$cycleStage2DepthList mask_list=$cycleStage2MaskList mix=$stage2PointMvMaskSoftMix soft_hit=$stage2PointMvMaskSoftHitThr stride=$stage2PointMvStride support=$stage2PointMvDepthSupportMode/$stage2PointMvDepthSupportFloor"
    }
    $stage2PairModeTokens = Parse-GenericTokens -Raw $Stage2PointMvDepthPairModeList
    $stage2PointMvDepthPairMode = Resolve-PointMvDepthPairMode -Raw ([string]$BasePointMvDepthPairMode) -Default "adjacent"
    if ($stage2PairModeTokens.Count -gt 0) {
        $pairPick = [Math]::Max(0, [int]$cycle - 1) % $stage2PairModeTokens.Count
        $stage2PointMvDepthPairMode = Resolve-PointMvDepthPairMode `
            -Raw ([string]$stage2PairModeTokens[$pairPick]) `
            -Default ([string]$stage2PointMvDepthPairMode)
    }
    $stage2EvalNumSrcViewsList = [string]$Stage2EvalNumSrcViewsList
    $stage2EnableAnySplatAblationSixPackEffective = ([bool]$Stage2EnableAnySplatAblationSixPack) -and [bool]$p0GatePass
    $stage2DynProxyEnableEffective = if ([bool]$p0GatePass) { [string]$Stage2DynProxyEnable } else { "off" }
    $stage2PointmapSourceEffective = "point_head"
    $stage2DualLaneEnabledEffective = [bool]$Stage2DualLaneEnabled
    $stage2PostRescueEnabledEffective = [bool]$PostRescueEnabled
    $stage2PrecomputeMvSupportOnEffective = "on"
    $stage2PointTargetBlendByMvSupportEffective = "on"
    if (-not [bool]$p0GatePass) {
        Write-Host "[autoloop] P0 gate pending: stage2_valid_rows_90m=$($p0Stage2Stats.valid_rows) reason=$p0GateReason; keep AnySplat six-pack off and dyn_proxy off."
        $isEmergencyShockProfile = [bool]$EmergencyGhostShockEnabled -and ($cycleEmergencyProfile -in @("shock_primary", "shock_fallback"))
        if ($isEmergencyShockProfile) {
            # P0-first stabilization: avoid the repeatedly failing point_head
            # primary precompute path and skip lane_b/post-rescue until stage2
            # can produce valid rows reliably.
            $stage2PointmapSourceEffective = "depth_unproject"
            $stage2DualLaneEnabledEffective = $false
            $stage2PostRescueEnabledEffective = $false
            $stage2PrecomputeMvSupportOnEffective = "off"
            $stage2PointTargetBlendByMvSupportEffective = "off"
            Write-Host "[autoloop] stage2 P0 stability override: pointmap_source=$stage2PointmapSourceEffective disable_dual_lane=$stage2DualLaneEnabledEffective disable_post_rescue=$stage2PostRescueEnabledEffective precompute_mv_support_on=$stage2PrecomputeMvSupportOnEffective point_target_blend_by_mv_support=$stage2PointTargetBlendByMvSupportEffective"
        }
    }
    Write-Host "[autoloop] stage2 view/pair policy eval_num_src_views_list=$stage2EvalNumSrcViewsList pair_mode=$stage2PointMvDepthPairMode cam_count=$(([string]$StageCamNames -split '[,\\s;|]+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count) pointmap_source=$stage2PointmapSourceEffective"
    $s2 = Invoke-GhostStage `
        -StageName ("cycle{0:D3}_stage2_weak" -f $cycle) `
        -PointTargetBlendMvPolicy "weak_to_depth" `
        -PointmapSource $stage2PointmapSourceEffective `
        -PseudoGeomSubdir $currPseudo `
        -ResumeCkpt $resumeForS2 `
        -Overrides @{
            LaneId = "lane_a"
            CandidateFamily = "stage2_training"
            LambdaPointMvDepthList = $cycleStage2DepthList
            LambdaPointMvMaskList = $cycleStage2MaskList
            NoImprovePatience = $stage2NoImprovePatienceEffective
            PointTargetConsensusAlphaFloor = $stage2PointTargetConsensusAlphaFloor
            PointMvMaskSoftMix = $stage2PointMvMaskSoftMix
            PointMvMaskSoftHitThr = $stage2PointMvMaskSoftHitThr
            PointMvMaskMinTgtFgRatio = $stage2PointMvMaskMinTgtFgRatio
            PointMvStride = $stage2PointMvStride
            PointMvDepthMaxPairs = $stage2PointMvDepthMaxPairs
            PointMvDepthPairMode = $stage2PointMvDepthPairMode
            PointMvDepthSupportMode = $stage2PointMvDepthSupportMode
            PointMvDepthSupportFloor = $stage2PointMvDepthSupportFloor
            CamNames = $StageCamNames
            EvalNumSrcViewsList = $stage2EvalNumSrcViewsList
            GramDynEnable = $Stage2GramDynEnable
            GramDynLayerIdx = $Stage2GramDynLayerIdx
            GramDynQuantile = $Stage2GramDynQuantile
            GramDynWeightFloor = $Stage2GramDynWeightFloor
            GramDynWarmupSteps = $Stage2GramDynWarmupSteps
            DynProxyEnable = $stage2DynProxyEnableEffective
            DynProxyMode = $Stage2DynProxyMode
            DynProxyUseGram = $Stage2DynProxyUseGram
            DynProxyUseSupport = $Stage2DynProxyUseSupport
            DynProxyFloor = $Stage2DynProxyFloor
            DynProxyWarmupSteps = $Stage2DynProxyWarmupSteps
            EnableAnySplatAblationSixPack = $stage2EnableAnySplatAblationSixPackEffective
            PrecomputeMvSupportOn = $stage2PrecomputeMvSupportOnEffective
            PointTargetBlendByMvSupport = $stage2PointTargetBlendByMvSupportEffective
            EnableExtendedCkptWaitOnMissing = $Stage2EnableExtendedCkptWaitOnMissing
            CkptExtendedWaitTimeoutSec = $Stage2CkptExtendedWaitTimeoutSec
            EnableResumeCkptFallbackOnShortCkptMissing = $Stage2EnableResumeCkptFallbackOnShortCkptMissing
            DisallowResumeFallbackResult = $Stage2DisallowResumeFallbackResult
            EnableVisualAntiBlackGuard = $EnableVisualAntiBlackGuard
            MinPredLumaMean = $MinPredLumaMean
            MinPredNonBlackRatio = $MinPredNonBlackRatio
            MinAreaRatio = $MinAreaRatio
            MinWidthRatio = $MinWidthRatio
            ModalRunNoOutputTimeoutSec = [int]$stage2ModalRunNoOutputTimeoutSecEffective
            ModalRunNoOutputMaxRetries = [int]$stage2ModalRunNoOutputMaxRetriesEffective
            InfraNoOutputStopConsecutive = [int]$stage2InfraNoOutputStopConsecutiveEffective
        }

    $s2LaneA = $s2
    $s2LaneB = $null
    $laneDecision = $null
    if ((-not $isABRouteMode) -and $stage2DualLaneEnabledEffective -and $stage2PostRescueEnabledEffective) {
        $s2LaneB = Invoke-Stage2PostRescue -Cycle $cycle -LaneAStage $s2LaneA
        $laneDecision = Resolve-Stage2DualLaneDecision `
            -LaneAStage $s2LaneA `
            -LaneBStage $s2LaneB `
            -ReferencePsnr $preCycleBestPsnr `
            -ReferenceSsim $preCycleBestSsim `
            -ReferenceWl1 $preCycleBestWl1
        if ($laneDecision -ne $null -and $laneDecision.selected_stage -ne $null) {
            $s2 = $laneDecision.selected_stage
            $activeLane = [string]$laneDecision.active_lane
            $laneABestSnapshot = $laneDecision.lane_a_best
            $laneBBestSnapshot = $laneDecision.lane_b_best
            $cycleGuardTier = [string]$laneDecision.guard_tier
            $cycleDecisionReason = [string]$laneDecision.decision_reason
            $cycleRollbackReason = [string]$laneDecision.rollback_reason
            Write-Host "[autoloop] stage2 lane select active=$activeLane tier=$cycleGuardTier reason=$cycleDecisionReason"
            if (-not [string]::IsNullOrWhiteSpace($cycleRollbackReason)) {
                Write-Host "[autoloop] stage2 lane rollback reason: $cycleRollbackReason"
            }
        }
    } else {
        $laneAGuard = Resolve-LayeredGuardResult `
            -Candidate $s2LaneA `
            -ReferencePsnr $preCycleBestPsnr `
            -ReferenceSsim $preCycleBestSsim `
            -ReferenceWl1 $preCycleBestWl1 `
            -Mode $QualityGuardMode
        $activeLane = "lane_a"
        $laneABestSnapshot = Build-LaneSnapshot -Lane "lane_a" -Stage $s2LaneA -Guard $laneAGuard
        $laneBBestSnapshot = $null
        $cycleGuardTier = [string]$laneAGuard.guard_tier
        $cycleDecisionReason = "single_lane_stage2"
        $cycleRollbackReason = ""
    }

    Safe-WriteInterimAutoloopArtifacts `
        -History @($history.ToArray()) `
        -CurrentCycle $cycle `
        -CurrentStages @($s1, $s2) `
        -Deadline $deadline `
        -GlobalBestGhost $globalBestGhost `
        -GlobalBestPsnr $globalBestPsnr `
        -GlobalBestSsim $globalBestSsim `
        -GlobalBestWl1 $globalBestWl1 `
        -CurrResume $currResume `
        -CurrPseudo $currPseudo `
        -PendingTuneAction $pendingTuneAction `
        -ActiveLane $activeLane `
        -LaneABest $laneABestSnapshot `
        -LaneBBest $laneBBestSnapshot `
        -GuardTier $cycleGuardTier `
        -DecisionReason $cycleDecisionReason `
        -RollbackReason $cycleRollbackReason

    if ($isABRouteMode) {
        $stage2HasPotential = $true
        $stage2PotentialReason = "ab_route_stage2_only"
    } else {
        $stage2GhostPotential = Test-Stage2HasPotential `
            -Stage2 $s2 `
            -Stage1 $s1 `
            -GlobalBestGhostRef $preCycleBestGhost `
            -GhostLagThreshold $Stage2PotentialGhostLagThreshold `
            -VsStage1ImproveMin $Stage2PotentialVsStage1Improve
        $stage2QualityPotential = Resolve-Stage2PotentialQualityCheck `
            -Stage2 $s2 `
            -GlobalBestPsnrRef $preCycleBestPsnr `
            -GlobalBestSsimRef $preCycleBestSsim `
            -GlobalBestWl1Ref $preCycleBestWl1 `
            -MaxPSNRDrop ([Math]::Max(0.0, [double]$Stage2PotentialMaxPSNRDrop)) `
            -MaxSSIMDrop ([Math]::Max(0.0, [double]$Stage2PotentialMaxSSIMDrop)) `
            -MaxWl1Rise ([Math]::Max(0.0, [double]$Stage2PotentialMaxWl1Rise))
        $stage2HasPotential = $stage2GhostPotential -and [bool]$stage2QualityPotential.pass
        if ($stage2HasPotential) {
            $stage2PotentialReason = "pass"
        } else {
            $s2GhostForReason = To-DoubleOrNaN($s2.ghost)
            $s1GhostForReason = To-DoubleOrNaN($s1.ghost)
            $ghostReason = "ghost_gate(s2=$(Fmt-Num $s2GhostForReason 4), s1=$(Fmt-Num $s1GhostForReason 4), ref=$(Fmt-Num $preCycleBestGhost 4), lag_thr=$(Fmt-Num ([double]$Stage2PotentialGhostLagThreshold) 4), s1_improve_min=$(Fmt-Num ([double]$Stage2PotentialVsStage1Improve) 4), pass=$stage2GhostPotential)"
            $qualityReason = "quality_gate(psnr_drop_max=$(Fmt-Num ([double]$Stage2PotentialMaxPSNRDrop) 4), ssim_drop_max=$(Fmt-Num ([double]$Stage2PotentialMaxSSIMDrop) 4), wl1_rise_max=$(Fmt-Num ([double]$Stage2PotentialMaxWl1Rise) 4), detail=$([string]$stage2QualityPotential.reason))"
            $stage2PotentialReason = "fail($ghostReason; $qualityReason)"
            $skipDeepStagesByPotential = $true
            $skipDeepStagesByPotentialReason = "stage2_not_potential; $stage2PotentialReason"
            Write-Host "[autoloop] stage2 potential check failed -> skip stage3-5 ($stage2PotentialReason)"
        }
    }

    if ($skipDeepStagesDueInfra) {
        $skipDeepStagesCombined = $true
        $skipDeepStagesCombinedReason = $skipDeepStagesDueInfraReason
    } elseif ($isABRouteMode) {
        $skipDeepStagesCombined = $true
        $skipDeepStagesCombinedReason = "ab_route_short_run_stage2_only"
    } elseif ($skipDeepStagesByPotential) {
        $skipDeepStagesCombined = $true
        $skipDeepStagesCombinedReason = $skipDeepStagesByPotentialReason
    }
    if ([bool]$ForceStage2Only) {
        $skipDeepStagesCombined = $true
        $skipDeepStagesCombinedReason = "force_stage2_only_single_run"
        Write-Host "[autoloop] force_stage2_only enabled -> skip stage3-5"
    }

    $resumeForS3 = Resolve-NextStageResumeCkpt `
        -PrevStage $s2 `
        -FallbackResume $resumeForS2 `
        -GlobalBestGhostRef $globalBestGhost `
        -PromoteGhostMargin $StageResumePromoteGhostMargin
    if ($skipDeepStagesCombined) {
        $s3 = New-SkippedStageResult `
            -StageName ("cycle{0:D3}_stage3_depth_unproject" -f $cycle) `
            -PointTargetBlendMvPolicy "weak_to_depth" `
            -PointmapSource "depth_unproject" `
            -PseudoGeomSubdir $currPseudo `
            -ResumeCkpt $resumeForS3 `
            -Reason $skipDeepStagesCombinedReason `
            -Overrides @{
                NoImprovePatience = [Math]::Max(3, [Math]::Min($StageNoImprovePatience, $Stage3NoImprovePatience))
            }
    } else {
        $s3 = Invoke-GhostStage `
            -StageName ("cycle{0:D3}_stage3_depth_unproject" -f $cycle) `
            -PointTargetBlendMvPolicy "weak_to_depth" `
            -PointmapSource "depth_unproject" `
            -PseudoGeomSubdir $currPseudo `
            -ResumeCkpt $resumeForS3 `
            -Overrides @{
                # depth_unproject stage has repeatedly shown slow no-improve tails;
                # use tighter patience to increase cycle throughput for auto-tuning.
                NoImprovePatience = [Math]::Max(3, [Math]::Min($StageNoImprovePatience, $Stage3NoImprovePatience))
            }
    }
    Safe-WriteInterimAutoloopArtifacts `
        -History @($history.ToArray()) `
        -CurrentCycle $cycle `
        -CurrentStages @($s1, $s2, $s3) `
        -Deadline $deadline `
        -GlobalBestGhost $globalBestGhost `
        -GlobalBestPsnr $globalBestPsnr `
        -GlobalBestSsim $globalBestSsim `
        -GlobalBestWl1 $globalBestWl1 `
        -CurrResume $currResume `
        -CurrPseudo $currPseudo `
        -PendingTuneAction $pendingTuneAction
    
    $depthAnchorOverrides = @{
        LambdaPointMvDepthList = $DepthAnchorLambdaPointMvDepthList
        LambdaPointMvMaskList = $DepthAnchorLambdaPointMvMaskList
        PointTargetMode = $DepthAnchorPointTargetMode
        PointTargetConsensusAlphaFloor = $DepthAnchorPointTargetConsensusAlphaFloor
        LambdaPoint = $DepthAnchorLambdaPoint
        LambdaPointReproj = $DepthAnchorLambdaPointReproj
        PointMvMaskHitThr = $DepthAnchorPointMvMaskHitThr
        PointMvMaskMinTgtFgRatio = $DepthAnchorPointMvMaskMinTgtFgRatio
        PointMvMaskSoftBlurPx = $DepthAnchorPointMvMaskSoftBlurPx
        PointMvMaskSoftBlurIters = $DepthAnchorPointMvMaskSoftBlurIters
        PointMvMaskSoftMix = $DepthAnchorPointMvMaskSoftMix
        PointMvMaskSoftHitThr = $DepthAnchorPointMvMaskSoftHitThr
        PointMvStride = $DepthAnchorPointMvStride
        PointMvDepthMaxPairs = $DepthAnchorPointMvDepthMaxPairs
        PointMvDepthSupportMode = $DepthAnchorPointMvDepthSupportMode
        PointMvDepthSupportFloor = $DepthAnchorPointMvDepthSupportFloor
        PointLossFgErodePx = $DepthAnchorPointLossFgErodePx
        PointMvDepthFgErodePx = $DepthAnchorPointMvDepthFgErodePx
        PointConsQuantile = $DepthAnchorPointConsQuantile
        PointConsFocus = $DepthAnchorPointConsFocus
        PointResidualQuantile = $DepthAnchorPointResidualQuantile
        PointResidualFocus = $DepthAnchorPointResidualFocus
        PointResidualBoost = $DepthAnchorPointResidualBoost
        PointMvDepthOutlierBoost = $DepthAnchorPointMvDepthOutlierBoost
        SupervisionWeightMode = $DepthAnchorSupervisionWeightMode
        NoImprovePatience = [Math]::Max($StageNoImprovePatience, 8)
    }
    $consensusHighfloorOverrides = @{
        LambdaPointMvDepthList = $DepthAnchorLambdaPointMvDepthList
        LambdaPointMvMaskList = $DepthAnchorLambdaPointMvMaskList
        PointTargetMode = "depth_consensus_unproject"
        PointTargetConsensusAlphaFloor = [Math]::Max($DepthAnchorPointTargetConsensusAlphaFloor, 0.7)
        LambdaPoint = [Math]::Max($DepthAnchorLambdaPoint, 0.4)
        LambdaPointReproj = $DepthAnchorLambdaPointReproj
        PointMvMaskHitThr = $DepthAnchorPointMvMaskHitThr
        PointMvMaskMinTgtFgRatio = $DepthAnchorPointMvMaskMinTgtFgRatio
        PointMvMaskSoftBlurPx = $DepthAnchorPointMvMaskSoftBlurPx
        PointMvMaskSoftBlurIters = $DepthAnchorPointMvMaskSoftBlurIters
        PointMvMaskSoftMix = $DepthAnchorPointMvMaskSoftMix
        PointMvMaskSoftHitThr = $DepthAnchorPointMvMaskSoftHitThr
        PointMvStride = $DepthAnchorPointMvStride
        PointMvDepthMaxPairs = $DepthAnchorPointMvDepthMaxPairs
        PointMvDepthSupportMode = $DepthAnchorPointMvDepthSupportMode
        PointMvDepthSupportFloor = $DepthAnchorPointMvDepthSupportFloor
        PointLossFgErodePx = $DepthAnchorPointLossFgErodePx
        PointMvDepthFgErodePx = $DepthAnchorPointMvDepthFgErodePx
        PointConsQuantile = $DepthAnchorPointConsQuantile
        PointConsFocus = $DepthAnchorPointConsFocus
        PointResidualQuantile = $DepthAnchorPointResidualQuantile
        PointResidualFocus = $DepthAnchorPointResidualFocus
        PointResidualBoost = $DepthAnchorPointResidualBoost
        PointMvDepthOutlierBoost = $DepthAnchorPointMvDepthOutlierBoost
        SupervisionWeightMode = $DepthAnchorSupervisionWeightMode
        NoImprovePatience = [Math]::Max($StageNoImprovePatience, 8)
    }

    $s4 = $null
    $s5 = $null
    if ($skipDeepStagesCombined) {
        $s4 = New-SkippedStageResult `
            -StageName ("cycle{0:D3}_stage4_depth_anchor_hard" -f $cycle) `
            -PointTargetBlendMvPolicy "strong_to_depth" `
            -PointmapSource "depth_unproject" `
            -PseudoGeomSubdir $currPseudo `
            -ResumeCkpt $resumeForS3 `
            -Reason $skipDeepStagesCombinedReason `
            -Overrides $depthAnchorOverrides
        $s5 = New-SkippedStageResult `
            -StageName ("cycle{0:D3}_stage5_consensus_highfloor" -f $cycle) `
            -PointTargetBlendMvPolicy "strong_to_depth" `
            -PointmapSource "point_head" `
            -PseudoGeomSubdir $currPseudo `
            -ResumeCkpt $resumeForS3 `
            -Reason $skipDeepStagesCombinedReason `
            -Overrides $consensusHighfloorOverrides
    } else {
        $s123Best = Get-BestStageByGhost -Stages @($s1, $s2, $s3)
        $s123Ghost = if ($s123Best -ne $null) { To-DoubleOrNaN($s123Best.ghost) } else { [double]::NaN }
        $skipDeepStagesByLag = $false
        $skipDeepReason = ""
        if (($noSubstantialImproveCycles -ge [Math]::Max(0, [int]$SkipDeepStagesAfterNoSubstantialCycles)) -and
            (-not [double]::IsInfinity($globalBestGhost)) -and (-not [double]::IsNaN($globalBestGhost)) -and
            (-not [double]::IsNaN($s123Ghost)) -and
            ($s123Ghost -gt ($globalBestGhost + [Math]::Max(0.0, $SkipDeepStagesGhostLag)))) {
            $skipDeepStagesByLag = $true
            $skipDeepReason = "s123_ghost_lag=$([string](Fmt-Num ($s123Ghost - $globalBestGhost) 4)) > $([string](Fmt-Num $SkipDeepStagesGhostLag 4))"
        }

        if ($skipDeepStagesByLag) {
            $s4 = New-SkippedStageResult `
                -StageName ("cycle{0:D3}_stage4_depth_anchor_hard" -f $cycle) `
                -PointTargetBlendMvPolicy "strong_to_depth" `
                -PointmapSource "depth_unproject" `
                -PseudoGeomSubdir $currPseudo `
                -ResumeCkpt $resumeForS3 `
                -Reason ("skip_deep_stages_due_to_lag; " + $skipDeepReason) `
                -Overrides $depthAnchorOverrides
            $s5 = New-SkippedStageResult `
                -StageName ("cycle{0:D3}_stage5_consensus_highfloor" -f $cycle) `
                -PointTargetBlendMvPolicy "strong_to_depth" `
                -PointmapSource "point_head" `
                -PseudoGeomSubdir $currPseudo `
                -ResumeCkpt $resumeForS3 `
                -Reason ("skip_deep_stages_due_to_lag; " + $skipDeepReason) `
                -Overrides $consensusHighfloorOverrides
        } else {
            $resumeForS4 = Resolve-NextStageResumeCkpt `
                -PrevStage $s3 `
                -FallbackResume $resumeForS3 `
                -GlobalBestGhostRef $globalBestGhost `
                -PromoteGhostMargin $StageResumePromoteGhostMargin
            $s4 = Invoke-GhostStage `
                -StageName ("cycle{0:D3}_stage4_depth_anchor_hard" -f $cycle) `
                -PointTargetBlendMvPolicy "strong_to_depth" `
                -PointmapSource "depth_unproject" `
                -PseudoGeomSubdir $currPseudo `
                -ResumeCkpt $resumeForS4 `
                -Overrides $depthAnchorOverrides
            Safe-WriteInterimAutoloopArtifacts `
                -History @($history.ToArray()) `
                -CurrentCycle $cycle `
                -CurrentStages @($s1, $s2, $s3, $s4) `
                -Deadline $deadline `
                -GlobalBestGhost $globalBestGhost `
                -GlobalBestPsnr $globalBestPsnr `
                -GlobalBestSsim $globalBestSsim `
                -GlobalBestWl1 $globalBestWl1 `
                -CurrResume $currResume `
                -CurrPseudo $currPseudo `
                -PendingTuneAction $pendingTuneAction

            $skipStage5 = $false
            $skipStage5Reason = ""
            $s4Ghost = To-DoubleOrNaN($s4.ghost)
            if (($noSubstantialImproveCycles -ge [Math]::Max(0, [int]$SkipDeepStagesAfterNoSubstantialCycles)) -and
                (-not [double]::IsInfinity($globalBestGhost)) -and (-not [double]::IsNaN($globalBestGhost)) -and
                (-not [double]::IsNaN($s4Ghost)) -and
                ($s4Ghost -gt ($globalBestGhost + [Math]::Max(0.0, $SkipStage5GhostLagVsGlobal)))) {
                $skipStage5 = $true
                $skipStage5Reason = "s4_ghost_lag=$([string](Fmt-Num ($s4Ghost - $globalBestGhost) 4)) > $([string](Fmt-Num $SkipStage5GhostLagVsGlobal 4))"
            }

            if ($skipStage5) {
                $s5 = New-SkippedStageResult `
                    -StageName ("cycle{0:D3}_stage5_consensus_highfloor" -f $cycle) `
                    -PointTargetBlendMvPolicy "strong_to_depth" `
                    -PointmapSource "point_head" `
                    -PseudoGeomSubdir $currPseudo `
                    -ResumeCkpt $resumeForS4 `
                    -Reason ("skip_stage5_due_to_stage4_lag; " + $skipStage5Reason) `
                    -Overrides $consensusHighfloorOverrides
            } else {
                $resumeForS5 = Resolve-NextStageResumeCkpt `
                    -PrevStage $s4 `
                    -FallbackResume $resumeForS4 `
                    -GlobalBestGhostRef $globalBestGhost `
                    -PromoteGhostMargin $StageResumePromoteGhostMargin
                $s5 = Invoke-GhostStage `
                    -StageName ("cycle{0:D3}_stage5_consensus_highfloor" -f $cycle) `
                    -PointTargetBlendMvPolicy "strong_to_depth" `
                    -PointmapSource "point_head" `
                    -PseudoGeomSubdir $currPseudo `
                    -ResumeCkpt $resumeForS5 `
                    -Overrides $consensusHighfloorOverrides
            }
        }
    }
    Safe-WriteInterimAutoloopArtifacts `
        -History @($history.ToArray()) `
        -CurrentCycle $cycle `
        -CurrentStages @($s1, $s2, $s3, $s4, $s5) `
        -Deadline $deadline `
        -GlobalBestGhost $globalBestGhost `
        -GlobalBestPsnr $globalBestPsnr `
        -GlobalBestSsim $globalBestSsim `
        -GlobalBestWl1 $globalBestWl1 `
        -CurrResume $currResume `
        -CurrPseudo $currPseudo `
        -PendingTuneAction $pendingTuneAction

    $stages = @($s1, $s2, $s3, $s4, $s5)
    $bestCycle = Get-BestStageByGhost -Stages $stages

    $cycleImproved = $false
    $cycleSubstantialImproved = $false
    $cycleRegressed = $false
    $cycleRegressReason = ""
    $rolledBackLastTune = $false
    $rolledBackTuneAction = ""
    $cycleComparePng = ""
    $visualConclusion = "本轮无有效结果，需人工复核。"
    $refStatsForVisual = $globalBestStats
    $refVisualForCompare = $globalBestVisualPng

    $gNow = [double]::NaN
    $pNow = [double]::NaN
    $sNow = [double]::NaN
    $wNow = [double]::NaN
    $cycleQualityGuardBlocked = $false
    $cycleQualityGuardReason = ""
    if ($bestCycle -ne $null) {
        $gNow = To-DoubleOrNaN($bestCycle.ghost)
        $pNow = To-DoubleOrNaN($bestCycle.psnr)
        $sNow = To-DoubleOrNaN($bestCycle.ssim)
        $wNow = To-DoubleOrNaN($bestCycle.wl1)

        if (-not [double]::IsNaN($gNow)) {
            $qualityGuardReasons = New-Object System.Collections.Generic.List[string]
            $promoteGuardMode = "strict"
            $maxPromotePSNRDrop = [Math]::Max(0.0, $CyclePromoteMaxPSNRDrop)
            $maxPromoteSSIMDrop = [Math]::Max(0.0, $CyclePromoteMaxSSIMDrop)
            $maxPromoteWl1Rise = [Math]::Max(0.0, $CyclePromoteMaxWl1Rise)
            if ((-not [double]::IsInfinity($preCycleBestGhost)) -and (-not [double]::IsNaN($preCycleBestGhost)) -and ($gNow -lt $preCycleBestGhost)) {
                $ghostGainForPromote = $preCycleBestGhost - $gNow
                if ($ghostGainForPromote -ge [Math]::Max(0.0, $CyclePromoteRelaxedMinGhostGain)) {
                    $promoteGuardMode = "relaxed"
                    $maxPromotePSNRDrop = [Math]::Max($maxPromotePSNRDrop, [Math]::Max(0.0, $CyclePromoteRelaxedMaxPSNRDrop))
                    $maxPromoteSSIMDrop = [Math]::Max($maxPromoteSSIMDrop, [Math]::Max(0.0, $CyclePromoteRelaxedMaxSSIMDrop))
                    $maxPromoteWl1Rise = [Math]::Max($maxPromoteWl1Rise, [Math]::Max(0.0, $CyclePromoteRelaxedMaxWl1Rise))
                }
                if ((-not [double]::IsNaN($pNow)) -and (-not [double]::IsInfinity($preCycleBestPsnr)) -and (-not [double]::IsNaN($preCycleBestPsnr))) {
                    $psnrDropForPromote = $preCycleBestPsnr - $pNow
                    if ($psnrDropForPromote -gt $maxPromotePSNRDrop) {
                        $qualityGuardReasons.Add("psnr_drop=$([string](Fmt-Num $psnrDropForPromote 4))>$(Fmt-Num $maxPromotePSNRDrop 4)") | Out-Null
                    }
                }
                if ((-not [double]::IsNaN($sNow)) -and (-not [double]::IsInfinity($preCycleBestSsim)) -and (-not [double]::IsNaN($preCycleBestSsim))) {
                    $ssimDropForPromote = $preCycleBestSsim - $sNow
                    if ($ssimDropForPromote -gt $maxPromoteSSIMDrop) {
                        $qualityGuardReasons.Add("ssim_drop=$([string](Fmt-Num $ssimDropForPromote 4))>$(Fmt-Num $maxPromoteSSIMDrop 4)") | Out-Null
                    }
                }
                if ((-not [double]::IsNaN($wNow)) -and (-not [double]::IsInfinity($preCycleBestWl1)) -and (-not [double]::IsNaN($preCycleBestWl1))) {
                    $wl1RiseForPromote = $wNow - $preCycleBestWl1
                    if ($wl1RiseForPromote -gt $maxPromoteWl1Rise) {
                        $qualityGuardReasons.Add("wl1_rise=$([string](Fmt-Num $wl1RiseForPromote 4))>$(Fmt-Num $maxPromoteWl1Rise 4)") | Out-Null
                    }
                }
            }
            if ($qualityGuardReasons.Count -gt 0) {
                $cycleQualityGuardBlocked = $true
                $cycleQualityGuardReason = "mode=$promoteGuardMode; " + ($qualityGuardReasons -join "; ")
            }

            if ($gNow -le ($globalBestGhost - [Math]::Max(0.0, $CycleMinGhostImprove))) {
                if (-not $cycleQualityGuardBlocked) { $cycleImproved = $true }
            }
            if ($gNow -le ($globalBestGhost - [Math]::Max(0.0, $SubstantialGhostImprove))) {
                if (-not $cycleQualityGuardBlocked) { $cycleSubstantialImproved = $true }
            }
            if (($gNow -lt $globalBestGhost) -and (-not $cycleQualityGuardBlocked)) {
                $globalBestGhost = $gNow
                if (-not [double]::IsNaN($pNow)) { $globalBestPsnr = $pNow }
                if (-not [double]::IsNaN($sNow)) { $globalBestSsim = $sNow }
                if (-not [double]::IsNaN($wNow)) { $globalBestWl1 = $wNow }
                $bestCycleHintScope = Resolve-HintScope `
                    -BestStage ([string]$bestCycle.stage) `
                    -IsHistoricalBootstrap:$false
                $globalBestHintStageFamily = [string]$bestCycleHintScope.stage_family
                $globalBestHintApplyStage1 = [bool]$bestCycleHintScope.apply_stage1
                $globalBestHintApplyStage2 = [bool]$bestCycleHintScope.apply_stage2
                if (-not [string]::IsNullOrWhiteSpace([string]$bestCycle.best_lambda_point_mv_depth)) {
                    $globalBestLambdaDepthHint = [string]$bestCycle.best_lambda_point_mv_depth
                }
                if (-not [string]::IsNullOrWhiteSpace([string]$bestCycle.best_lambda_point_mv_mask)) {
                    $globalBestLambdaMaskHint = [string]$bestCycle.best_lambda_point_mv_mask
                }
                if (-not [string]::IsNullOrWhiteSpace([string]$bestCycle.best_visual_png)) {
                    $globalBestVisualPng = [string]$bestCycle.best_visual_png
                }
                $currBestStats = Get-GhostRowsStats -GhostRowsCsv ([string]$bestCycle.best_ghost_rows_csv)
                if ($currBestStats -ne $null) {
                    $globalBestStats = $currBestStats
                }
            }

            $ghostRise = $gNow - $preCycleBestGhost
            $ghostRegress = (-not [double]::IsInfinity($preCycleBestGhost)) -and (-not [double]::IsNaN($preCycleBestGhost)) -and ($ghostRise -ge [Math]::Max(0.0, $CycleRegressGhostThreshold))
            $psnrRegress = $false
            $wl1Regress = $false
            $hasGuardMetric = $false
            if ((-not [double]::IsNaN($pNow)) -and (-not [double]::IsInfinity($preCycleBestPsnr)) -and (-not [double]::IsNaN($preCycleBestPsnr))) {
                $hasGuardMetric = $true
                $psnrDrop = $preCycleBestPsnr - $pNow
                if ($psnrDrop -ge [Math]::Max(0.0, $CycleRegressPSNRDropThreshold)) { $psnrRegress = $true }
            }
            if ((-not [double]::IsNaN($wNow)) -and (-not [double]::IsInfinity($preCycleBestWl1)) -and (-not [double]::IsNaN($preCycleBestWl1))) {
                $hasGuardMetric = $true
                $wl1Rise = $wNow - $preCycleBestWl1
                if ($wl1Rise -ge [Math]::Max(0.0, $CycleRegressWl1RiseThreshold)) { $wl1Regress = $true }
            }
            if ($ghostRegress -and ((-not $hasGuardMetric) -or $psnrRegress -or $wl1Regress)) {
                $cycleRegressed = $true
                $cycleRegressReason = "ghost_rise=$([string](Fmt-Num $ghostRise 4)), psnr_drop_guard=$psnrRegress, wl1_rise_guard=$wl1Regress"
            }
            if ($cycleQualityGuardBlocked) {
                $blockMsg = "quality_guard_blocked_promotion: $cycleQualityGuardReason"
                if ([string]::IsNullOrWhiteSpace($cycleRegressReason)) {
                    $cycleRegressReason = $blockMsg
                } else {
                    $cycleRegressReason = "$cycleRegressReason; $blockMsg"
                }
            }
        } else {
            if ((-not [double]::IsNaN($pNow) -and $pNow -gt $globalBestPsnr) -or
                (-not [double]::IsNaN($wNow) -and $wNow -lt $globalBestWl1)) {
                $cycleImproved = $true
                if (-not [double]::IsNaN($pNow)) { $globalBestPsnr = $pNow }
                if (-not [double]::IsNaN($sNow)) { $globalBestSsim = $sNow }
                if (-not [double]::IsNaN($wNow)) { $globalBestWl1 = $wNow }
            }
        }

        $currStatsForVisual = Get-GhostRowsStats -GhostRowsCsv ([string]$bestCycle.best_ghost_rows_csv)
        $ghostDelta = [double]::NaN
        if (($refStatsForVisual -ne $null) -and ($currStatsForVisual -ne $null)) {
            $ghostDelta = To-DoubleOrNaN($currStatsForVisual.mean_ghost) - To-DoubleOrNaN($refStatsForVisual.mean_ghost)
        }
        $visualConclusion = Build-VisualJudgement -PrevStats $refStatsForVisual -CurrStats $currStatsForVisual -GhostDelta $ghostDelta

        $compareImgs = @()
        if (-not [string]::IsNullOrWhiteSpace($refVisualForCompare) -and (Test-Path $refVisualForCompare)) {
            $compareImgs += $refVisualForCompare
        }
        if (-not [string]::IsNullOrWhiteSpace([string]$bestCycle.best_visual_png) -and (Test-Path ([string]$bestCycle.best_visual_png))) {
            $compareImgs += [string]$bestCycle.best_visual_png
        }
        if ($compareImgs.Count -gt 0) {
            $compareOut = "logs/modal_phase5/ghost_autoloop_cycle{0:D3}_compare.png" -f $cycle
            $made = Make-ContactSheetSafe -ImagePaths $compareImgs -OutPng $compareOut
            if (-not [string]::IsNullOrWhiteSpace($made)) { $cycleComparePng = $made }
        }
    }

    if ([bool]$EmergencyGhostShockEnabled) {
        if (-not [double]::IsNaN($gNow)) {
            $emergencyLastCycleBestGhost = $gNow
            if ($gNow -le [double]$EmergencyShockTargetGhost) {
                if (-not $emergencyHitTarget) {
                    Write-Host "[autoloop] emergency hit target ghost: g_now=$(Fmt-Num $gNow 6) <= target=$(Fmt-Num ([double]$EmergencyShockTargetGhost) 6)"
                }
                $emergencyHitTarget = $true
                if ($emergencyState -ne "recovery") {
                    $emergencyState = "recovery"
                    $emergencyRecoveryActivated = $true
                    $emergencyLastReason = "hit_target_ghost<=$([double]$EmergencyShockTargetGhost)"
                }
            }

            if ($gNow -gt [double]$EmergencyShockFailGhostThreshold) {
                $emergencyConsecutiveHighGhost += 1
            } else {
                $emergencyConsecutiveHighGhost = 0
            }
        }

        if (($emergencyConsecutiveHighGhost -ge [Math]::Max(1, [int]$EmergencyShockFailConsecutiveLimit)) -and ($emergencyState -like "shock*")) {
            if ($emergencyState -eq "shock_primary") {
                $emergencyState = "shock_fallback"
                $emergencyFallbackActivated = $true
                $emergencyLastReason = "high_ghost_streak=$emergencyConsecutiveHighGhost -> switch_fallback"
                $emergencyConsecutiveHighGhost = 0
                Write-Host "[autoloop] emergency fallback triggered: $emergencyLastReason (threshold=$EmergencyShockFailGhostThreshold)"
            } else {
                $emergencyState = "rollback_steady"
                $emergencyRollbackActivated = $true
                $emergencyLastReason = "high_ghost_streak=$emergencyConsecutiveHighGhost"
                Write-Host "[autoloop] emergency rollback triggered: $emergencyLastReason (threshold=$EmergencyShockFailGhostThreshold)"
            }
        }
    }

    if ($cycleRegressed) {
        $regressCycles += 1
        if (($lastTuneStateBeforeApply -ne $null) -and ($lastTuneAppliedCycle -eq ($cycle - 1))) {
            Set-TuneState -State $lastTuneStateBeforeApply
            $rolledBackLastTune = $true
            $rolledBackTuneAction = [string]$lastTuneActionApplied
            $pendingTuneAction = "回滚上一步调参（因回归触发）: $rolledBackTuneAction"
            $lastTuneStateBeforeApply = $null
            $lastTuneActionApplied = "none"
            $lastTuneAppliedCycle = 0
        }
    } else {
        $regressCycles = 0
    }

    $promoteResume = $cycleImproved
    if ($promoteResume -and ($bestCycle -ne $null)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$bestCycle.best_ckpt)) {
            $currResume = [string]$bestCycle.best_ckpt
            $globalBestCkpt = [string]$bestCycle.best_ckpt
            $resumeUpdateReason = "cycle_improved"
        } else {
            $resumeUpdateReason = "cycle_improved_but_ckpt_missing"
        }
        if (-not [string]::IsNullOrWhiteSpace([string]$bestCycle.best_geom)) {
            $globalBestGeom = [string]$bestCycle.best_geom
        }
    } else {
        if (-not [string]::IsNullOrWhiteSpace($globalBestCkpt)) {
            $currResume = $globalBestCkpt
        }
        $resumeUpdateReason = "hold_global_best"
    }

    if ($cycleImproved) {
        $noImproveCycles = 0
    } else {
        $noImproveCycles += 1
    }
    if ($isABRouteMode) {
        $abCycleResult = [pscustomobject]@{
            cycle = $cycle
            variant = $abRouteVariant
            ghost = $gNow
            psnr = $pNow
            ssim = $sNow
            wl1 = $wNow
            quality_guard_blocked = $cycleQualityGuardBlocked
            quality_guard_reason = $cycleQualityGuardReason
        }
        $abDirectPass = (-not [double]::IsNaN($gNow)) -and (-not $cycleQualityGuardBlocked) -and ($gNow -le [double]$StableGhostTarget)
        if ($abDirectPass) {
            $pendingTuneAction = "A/B通过(提前): variant=$abRouteVariant, ghost=$(Fmt-Num $gNow 4) <= target=$(Fmt-Num ([double]$StableGhostTarget) 4)"
            $routeMode = "main"
            $abRouteExecuted = $true
            $abRouteVariant = "balance"
            $noSubstantialImproveCycles = 0
            $aggressiveRouteCooldownCycles = [Math]::Max($aggressiveRouteCooldownCycles, 1)
        } elseif ($isABBalanceCycle) {
            $abBalanceResult = $abCycleResult
            $abRouteVariant = "aggressive"
            $pendingTuneAction = "A/B验证: 下一轮运行激进候选(B)"
            $routeMode = "ab_validation"
        } else {
            $abAggressiveResult = $abCycleResult
            $abBalancePass = ($abBalanceResult -ne $null) -and (-not [bool]$abBalanceResult.quality_guard_blocked) -and (-not [double]::IsNaN((To-DoubleOrNaN($abBalanceResult.ghost))))
            $abAggressivePass = ($abAggressiveResult -ne $null) -and (-not [bool]$abAggressiveResult.quality_guard_blocked) -and (-not [double]::IsNaN((To-DoubleOrNaN($abAggressiveResult.ghost))))
            $abWinner = $null
            $abWinnerName = "none"
            if ($abBalancePass -and $abAggressivePass) {
                $abBalanceGhost = To-DoubleOrNaN($abBalanceResult.ghost)
                $abAggressiveGhost = To-DoubleOrNaN($abAggressiveResult.ghost)
                if ($abAggressiveGhost -lt $abBalanceGhost) {
                    $abWinner = $abAggressiveResult
                    $abWinnerName = "B(aggressive)"
                } else {
                    $abWinner = $abBalanceResult
                    $abWinnerName = "A(balance)"
                }
            } elseif ($abAggressivePass) {
                $abWinner = $abAggressiveResult
                $abWinnerName = "B(aggressive)"
            } elseif ($abBalancePass) {
                $abWinner = $abBalanceResult
                $abWinnerName = "A(balance)"
            }
            $abWinnerGhost = if ($abWinner -ne $null) { To-DoubleOrNaN($abWinner.ghost) } else { [double]::NaN }
            if (($abWinner -ne $null) -and (-not [double]::IsNaN($abWinnerGhost)) -and ($abWinnerGhost -le [double]$StableGhostTarget)) {
                $pendingTuneAction = "A/B通过: winner=$abWinnerName, ghost=$(Fmt-Num $abWinnerGhost 4) <= target=$(Fmt-Num ([double]$StableGhostTarget) 4)"
                $routeMode = "main"
                $abRouteExecuted = $true
                $noSubstantialImproveCycles = 0
                $aggressiveRouteCooldownCycles = [Math]::Max($aggressiveRouteCooldownCycles, 1)
            } else {
                $shouldStopByABValidation = $true
                $stopByABValidationReason = "A/B未通过稳定目标 target=$(Fmt-Num ([double]$StableGhostTarget) 4); A=$(Fmt-Num (To-DoubleOrNaN($abBalanceResult.ghost)) 4), B=$(Fmt-Num (To-DoubleOrNaN($abAggressiveResult.ghost)) 4), A_guard_blocked=$($abBalanceResult.quality_guard_blocked), B_guard_blocked=$($abAggressiveResult.quality_guard_blocked)"
                $pendingTuneAction = "A/B未通过: 建议停止继续烧算力并切换下一技术路线"
                $routeMode = "main"
                $abRouteExecuted = $true
            }
            $abRouteVariant = "balance"
        }
        $lastTuneStateBeforeApply = $null
        $lastTuneActionApplied = "none"
        $lastTuneAppliedCycle = 0
    } elseif ($cycleSubstantialImproved) {
        $noSubstantialImproveCycles = 0
        $pendingTuneAction = "none (保持当前参数)"
        $lastTuneStateBeforeApply = $null
        $lastTuneActionApplied = "none"
        $lastTuneAppliedCycle = 0
        $aggressiveRouteCooldownCycles = [Math]::Max($aggressiveRouteCooldownCycles, 1)
    } else {
        $noSubstantialImproveCycles += 1
        if (-not $rolledBackLastTune) {
            $stateBeforeTune = Get-TuneState
            $tuneGhostLag = [double]::NaN
            if ((-not [double]::IsNaN($gNow)) -and (-not [double]::IsInfinity($preCycleBestGhost)) -and (-not [double]::IsNaN($preCycleBestGhost))) {
                $tuneGhostLag = $gNow - $preCycleBestGhost
            }
            $nextTuneAction = Apply-NoImproveSingleStep `
                -StepIndex $tuneStep `
                -NoSubstantialImproveCycles $noSubstantialImproveCycles `
                -GhostLag $tuneGhostLag
            $lastTuneStateBeforeApply = $stateBeforeTune
            $lastTuneActionApplied = $nextTuneAction
            $lastTuneAppliedCycle = $cycle
            $pendingTuneAction = $nextTuneAction
            $tuneStep += 1
        }
    }

    if ((-not $isABRouteMode) -and $EnableABRouteOnStagnation -and (-not $abRouteExecuted) -and
        ($noSubstantialImproveCycles -ge [Math]::Max(1, [int]$NoProgressCyclesForABRoute)) -and
        (-not [double]::IsInfinity($globalBestGhost)) -and (-not [double]::IsNaN($globalBestGhost)) -and
        ($globalBestGhost -gt [double]$StableGhostTarget)) {
        $routeMode = "ab_validation"
        $abRouteVariant = "balance"
        $abBalanceResult = $null
        $abAggressiveResult = $null
        $pendingTuneAction = "触发A/B短跑验证: no_substantial=$noSubstantialImproveCycles, global_best_ghost=$(Fmt-Num $globalBestGhost 4) > target=$(Fmt-Num ([double]$StableGhostTarget) 4)"
        Write-Host "[autoloop] trigger A/B route due to stagnation: no_substantial=$noSubstantialImproveCycles global_best_ghost=$(Fmt-Num $globalBestGhost 4) target=$(Fmt-Num ([double]$StableGhostTarget) 4)"
    }

    if ($EnablePersistentCycleState) {
        $persistStateOut = [pscustomobject]@{
            updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
            no_improve_cycles = $noImproveCycles
            no_substantial_improve_cycles = $noSubstantialImproveCycles
            tune_step = $tuneStep
            regress_cycles = $regressCycles
            pending_tune_action = $pendingTuneAction
            route_mode = $routeMode
            ab_route_executed = $abRouteExecuted
            ab_route_variant = $abRouteVariant
            aggressive_route_cooldown_cycles = $aggressiveRouteCooldownCycles
            base_point_mv_mask_soft_mix = [double]$BasePointMvMaskSoftMix
            base_point_mv_mask_soft_hit_thr = [double]$BasePointMvMaskSoftHitThr
            base_point_mv_stride = [int]$BasePointMvStride
            base_point_mv_depth_max_pairs = [int]$BasePointMvDepthMaxPairs
            base_point_mv_depth_support_mode = [string]$BasePointMvDepthSupportMode
            base_point_mv_depth_support_floor = [double]$BasePointMvDepthSupportFloor
            base_point_mv_mask_min_tgt_fg_ratio = [double]$BasePointMvMaskMinTgtFgRatio
            current_resume_ckpt = $currResume
            global_best_ghost = $globalBestGhost
            global_best_psnr = $globalBestPsnr
            global_best_ssim = $globalBestSsim
            global_best_wl1 = $globalBestWl1
            process_pid = $PID
        }
        Write-PersistentCycleState -Path $PersistentCycleStatePath -State $persistStateOut
    }

    $history.Add([pscustomobject]@{
        cycle = $cycle
        stage1 = $s1
        stage2 = $s2
        stage2_lane_a = $s2LaneA
        stage2_lane_b = $s2LaneB
        active_lane = $activeLane
        lane_a_best = $laneABestSnapshot
        lane_b_best = $laneBBestSnapshot
        guard_tier = $cycleGuardTier
        stage2_lane_decision_reason = $cycleDecisionReason
        stage2_lane_rollback_reason = $cycleRollbackReason
        stage3 = $s3
        stage4 = $s4
        stage5 = $s5
        cycle_best_stage = $(if ($bestCycle -ne $null) { $bestCycle.stage } else { "" })
        cycle_best_ghost = $(if ($bestCycle -ne $null) { $bestCycle.ghost } else { "" })
        cycle_best_psnr = $(if ($bestCycle -ne $null) { $bestCycle.psnr } else { "" })
        cycle_best_ssim = $(if ($bestCycle -ne $null) { $bestCycle.ssim } else { "" })
        cycle_improved = $cycleImproved
        cycle_substantial_improved = $cycleSubstantialImproved
        cycle_route_mode = $cycleRouteMode
        next_route_mode = $routeMode
        ab_route_pending_next_cycle = $abRoutePendingNextCycle
        ab_route_variant = $abRouteVariant
        ab_route_executed = $abRouteExecuted
        stage2_has_potential = $stage2HasPotential
        stage2_potential_reason = $stage2PotentialReason
        skip_deep_stages = $skipDeepStagesCombined
        skip_deep_stages_reason = $skipDeepStagesCombinedReason
        cycle_regressed = $cycleRegressed
        cycle_regress_reason = $cycleRegressReason
        cycle_quality_guard_blocked = $cycleQualityGuardBlocked
        cycle_quality_guard_reason = $cycleQualityGuardReason
        emergency_profile = $cycleEmergencyProfile
        emergency_reason = $cycleEmergencyReason
        emergency_state = $emergencyState
        emergency_hit_target = $emergencyHitTarget
        emergency_consecutive_high_ghost = $emergencyConsecutiveHighGhost
        should_stop_by_ab_validation = $shouldStopByABValidation
        stop_by_ab_validation_reason = $stopByABValidationReason
        regress_cycles = $regressCycles
        rolled_back_last_tune = $rolledBackLastTune
        rolled_back_tune_action = $rolledBackTuneAction
        no_improve_cycles = $noImproveCycles
        no_substantial_improve_cycles = $noSubstantialImproveCycles
        tune_action_next = $pendingTuneAction
        cycle_compare_png = $cycleComparePng
        cycle_visual_conclusion = $visualConclusion
        resume_update_reason = $resumeUpdateReason
        current_resume_ckpt = $currResume
        current_pseudo_geom = $currPseudo
        global_best_ghost = $globalBestGhost
        global_best_psnr = $globalBestPsnr
        global_best_ssim = $globalBestSsim
        global_best_wl1 = $globalBestWl1
        global_best_ckpt = $globalBestCkpt
        global_best_geom = $globalBestGeom
        updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    }) | Out-Null

    $flatRows = @()
    foreach ($h in @($history.ToArray())) {
        foreach ($s in @($h.stage1, $h.stage2_lane_a, $h.stage2, $h.stage2_lane_b, $h.stage3, $h.stage4, $h.stage5)) {
            if ($null -eq $s) { continue }
            $flatRows += [pscustomobject]@{
                cycle = $h.cycle
                stage = $s.stage
                policy = $s.policy
                pointmap_source = $s.pointmap_source
                lane_id = $s.lane_id
                candidate_family = $s.candidate_family
                guard_tier = $s.guard_tier
                rollback_triggered = $s.rollback_triggered
                point_target_mode = $s.stage_point_target_mode
                lambda_point_mv_depth_list = $s.stage_lambda_point_mv_depth_list
                lambda_point_mv_mask_list = $s.stage_lambda_point_mv_mask_list
                lambda_point = $s.stage_lambda_point
                point_mv_mask_hit_thr = $s.stage_point_mv_mask_hit_thr
                point_mv_mask_min_tgt_fg_ratio = $s.stage_point_mv_mask_min_tgt_fg_ratio
                point_mv_mask_soft_mix = $s.stage_point_mv_mask_soft_mix
                point_mv_mask_soft_hit_thr = $s.stage_point_mv_mask_soft_hit_thr
                point_mv_stride = $s.stage_point_mv_stride
                point_mv_depth_max_pairs = $s.stage_point_mv_depth_max_pairs
                point_mv_depth_pair_mode = $s.stage_point_mv_depth_pair_mode
                point_mv_depth_support_mode = $s.stage_point_mv_depth_support_mode
                point_mv_depth_support_floor = $s.stage_point_mv_depth_support_floor
                point_cons_focus = $s.stage_point_cons_focus
                point_residual_focus = $s.stage_point_residual_focus
                rc = $s.rc
                ghost = $s.ghost
                psnr = $s.psnr
                ssim = $s.ssim
                wl1 = $s.wl1
                best_geom = $s.best_geom
                best_ckpt = $s.best_ckpt
                best_lambda_point_mv_depth = $s.best_lambda_point_mv_depth
                best_lambda_point_mv_mask = $s.best_lambda_point_mv_mask
                best_ghost_rows_csv = $s.best_ghost_rows_csv
                best_visual_png = $s.best_visual_png
                stage_best_strip_png = $s.stage_best_strip_png
                stage_skip_reason = $s.stage_skip_reason
                best_ghost_width_ratio = $s.best_ghost_width_ratio
                best_ghost_area_ratio = $s.best_ghost_area_ratio
                best_ghost_peak_count = $s.best_ghost_peak_count
                best_ghost_center_offset_ratio = $s.best_ghost_center_offset_ratio
                ghost_soft_score = $s.ghost_soft_score
                ghost_visual_score = $s.ghost_visual_score
                pred_luma_mean = $s.pred_luma_mean
                pred_nonblack_ratio_thr008 = $s.pred_nonblack_ratio_thr008
                visual_guard_blocked = $s.visual_guard_blocked
                visual_guard_reason = $s.visual_guard_reason
                eval_num_src_views = $s.eval_num_src_views
                eval_num_src_views_actual = $s.eval_num_src_views_actual
                eval_num_src_views_mismatch = $s.eval_num_src_views_mismatch
                cam_count_used = $s.cam_count_used
                precompute_source_requested = $s.precompute_source_requested
                precompute_source_resolved = $s.precompute_source_resolved
                precompute_fallback_used = $s.precompute_fallback_used
                precompute_timeout_hit = $s.precompute_timeout_hit
                precompute_mv_support_on = $s.precompute_mv_support_on
                point_target_blend_by_mv_support = $s.point_target_blend_by_mv_support
                sweep_csv = $s.sweep_csv
                cycle_substantial_improved = $h.cycle_substantial_improved
                cycle_route_mode = $h.cycle_route_mode
                next_route_mode = $h.next_route_mode
                ab_route_pending_next_cycle = $h.ab_route_pending_next_cycle
                ab_route_variant = $h.ab_route_variant
                ab_route_executed = $h.ab_route_executed
                stage2_has_potential = $h.stage2_has_potential
                stage2_potential_reason = $h.stage2_potential_reason
                skip_deep_stages = $h.skip_deep_stages
                skip_deep_stages_reason = $h.skip_deep_stages_reason
                cycle_regressed = $h.cycle_regressed
                cycle_regress_reason = $h.cycle_regress_reason
                cycle_quality_guard_blocked = $h.cycle_quality_guard_blocked
                cycle_quality_guard_reason = $h.cycle_quality_guard_reason
                emergency_profile = $h.emergency_profile
                emergency_reason = $h.emergency_reason
                emergency_state = $h.emergency_state
                emergency_hit_target = $h.emergency_hit_target
                emergency_consecutive_high_ghost = $h.emergency_consecutive_high_ghost
                should_stop_by_ab_validation = $h.should_stop_by_ab_validation
                stop_by_ab_validation_reason = $h.stop_by_ab_validation_reason
                regress_cycles = $h.regress_cycles
                rolled_back_last_tune = $h.rolled_back_last_tune
                rolled_back_tune_action = $h.rolled_back_tune_action
                tune_action_next = $h.tune_action_next
                cycle_compare_png = $h.cycle_compare_png
                resume_update_reason = $h.resume_update_reason
                updated_at = $h.updated_at
            }
        }
    }
    $flatRows | Export-Csv "logs/modal_phase5/ghost_autoloop_latest.csv" -NoTypeInformation -Encoding UTF8
    $flatRows | Export-Csv ("logs/modal_phase5/ghost_autoloop_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".csv") -NoTypeInformation -Encoding UTF8

    $abRoutePendingNextCycle = ((-not $isABRouteMode) -and ($routeMode -eq "ab_validation"))
    $shouldStopByStagnation = ($noSubstantialImproveCycles -ge [Math]::Max(1, [int]$StagnationStopCycles)) -and (-not $isABRouteMode) -and (-not $abRoutePendingNextCycle)
    $shouldStopByRegression = ($regressCycles -ge [Math]::Max(1, [int]$RegressionStopCycles))
    $shouldStopByAB = $shouldStopByABValidation

    $currentStageObj = $null
    foreach ($stageCandidate in @($s1, $s2LaneA, $s2, $s2LaneB, $s3, $s4, $s5)) {
        if ($null -ne $stageCandidate) {
            $currentStageObj = $stageCandidate
        }
    }
    $currentStageName = ""
    $currentEvalNumSrcViews = ""
    $currentEvalNumSrcViewsDeclared = ""
    $currentEvalNumSrcViewsActual = ""
    $currentEvalNumSrcViewsMismatch = $false
    $currentCamCountUsed = ""
    $currentVisualGuardBlocked = $false
    $currentVisualGuardReason = ""
    $currentQualityGuardBlocked = $false
    $currentQualityGuardReason = ""
    $currentCandidateInvalidReason = ""
    $currentPointMvDepthPairMode = ""
    $currentPrecomputeSource = ""
    $currentPrecomputeSourceRequested = ""
    $currentPrecomputeSourceResolved = ""
    $currentPrecomputeFallbackUsed = $false
    $currentPrecomputeTimeoutHit = $false
    $currentPrecomputeMvSupportOn = ""
    $currentPointTargetBlendByMvSupport = ""
    if ($null -ne $currentStageObj) {
        $currentStageName = [string]$currentStageObj.stage
        $currentEvalNumSrcViews = [string]$currentStageObj.eval_num_src_views
        $currentEvalNumSrcViewsDeclared = if ($currentStageObj.PSObject.Properties["eval_num_src_views_declared"]) { [string]$currentStageObj.eval_num_src_views_declared } else { [string]$currentStageObj.eval_num_src_views }
        $currentEvalNumSrcViewsActual = [string]$currentStageObj.eval_num_src_views_actual
        $currentEvalNumSrcViewsMismatch = To-BoolLoose -Value $currentStageObj.eval_num_src_views_mismatch -Default $false
        $currentCamCountUsed = [string]$currentStageObj.cam_count_used
        $currentVisualGuardBlocked = To-BoolLoose -Value $currentStageObj.visual_guard_blocked -Default $false
        $currentVisualGuardReason = [string]$currentStageObj.visual_guard_reason
        $currentQualityGuardBlocked = To-BoolLoose -Value $currentStageObj.quality_guard_blocked -Default $false
        $currentQualityGuardReason = [string]$currentStageObj.quality_guard_reason
        $currentCandidateInvalidReason = [string]$currentStageObj.candidate_invalid_reason
        $currentPointMvDepthPairMode = [string]$currentStageObj.stage_point_mv_depth_pair_mode
        $currentPrecomputeSource = [string]$currentStageObj.precompute_source
        $currentPrecomputeSourceRequested = [string]$currentStageObj.precompute_source_requested
        $currentPrecomputeSourceResolved = [string]$currentStageObj.precompute_source_resolved
        $currentPrecomputeFallbackUsed = To-BoolLoose -Value $currentStageObj.precompute_fallback_used -Default $false
        $currentPrecomputeTimeoutHit = To-BoolLoose -Value $currentStageObj.precompute_timeout_hit -Default $false
        $currentPrecomputeMvSupportOn = [string]$currentStageObj.precompute_mv_support_on
        $currentPointTargetBlendByMvSupport = [string]$currentStageObj.point_target_blend_by_mv_support
    }

    $status = [ordered]@{
        updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        deadline = $deadline.ToString("yyyy-MM-ddTHH:mm:ss")
        current_cycle = $cycle
        current_stage = $currentStageName
        cycle_route_mode = $cycleRouteMode
        next_route_mode = $routeMode
        active_lane = $activeLane
        lane_a_best = $laneABestSnapshot
        lane_b_best = $laneBBestSnapshot
        guard_tier = $cycleGuardTier
        visual_guard_tier = $(if ($currentVisualGuardBlocked) { "blocked" } else { "pass_or_na" })
        visual_guard_reason = $currentVisualGuardReason
        decision_reason = $cycleDecisionReason
        rollback_reason = $cycleRollbackReason
        active_eval_num_src_views = $currentEvalNumSrcViews
        active_eval_num_src_views_declared = $currentEvalNumSrcViewsDeclared
        active_eval_num_src_views_actual = $currentEvalNumSrcViewsActual
        active_eval_num_src_views_mismatch = $currentEvalNumSrcViewsMismatch
        active_cam_count = $currentCamCountUsed
        active_quality_guard_blocked = $currentQualityGuardBlocked
        active_quality_guard_reason = $currentQualityGuardReason
        active_candidate_invalid_reason = $currentCandidateInvalidReason
        active_precompute_source = $currentPrecomputeSource
        active_precompute_source_requested = $currentPrecomputeSourceRequested
        active_precompute_source_resolved = $currentPrecomputeSourceResolved
        active_precompute_fallback_used = $currentPrecomputeFallbackUsed
        active_precompute_timeout_hit = $currentPrecomputeTimeoutHit
        active_precompute_mv_support_on = $currentPrecomputeMvSupportOn
        active_point_target_blend_by_mv_support = $currentPointTargetBlendByMvSupport
        active_point_mv_depth_pair_mode = $currentPointMvDepthPairMode
        p0_stage2_valid_rows_90m = $p0Stage2Stats.valid_rows
        p0_gate_pass = $p0GatePass
        p0_gate_reason = $p0GateReason
        ab_route_pending_next_cycle = $abRoutePendingNextCycle
        ab_route_variant = $abRouteVariant
        ab_route_executed = $abRouteExecuted
        no_improve_cycles = $noImproveCycles
        no_substantial_improve_cycles = $noSubstantialImproveCycles
        current_resume_ckpt = $currResume
        current_pseudo_geom = $currPseudo
        global_best_ghost = $globalBestGhost
        global_best_psnr = $globalBestPsnr
        global_best_ssim = $globalBestSsim
        global_best_wl1 = $globalBestWl1
        global_best_ckpt = $globalBestCkpt
        global_best_geom = $globalBestGeom
        global_best_visual_png = $globalBestVisualPng
        global_best_hint_stage_family = $globalBestHintStageFamily
        global_best_hint_apply_stage1 = $globalBestHintApplyStage1
        global_best_hint_apply_stage2 = $globalBestHintApplyStage2
        next_cycle_tune_action = $pendingTuneAction
        emergency_enabled = [bool]$EmergencyGhostShockEnabled
        emergency_state = $emergencyState
        emergency_profile = $cycleEmergencyProfile
        emergency_reason = $cycleEmergencyReason
        emergency_hit_target = $emergencyHitTarget
        emergency_target_ghost = $EmergencyShockTargetGhost
        emergency_last_cycle_best_ghost = $emergencyLastCycleBestGhost
        emergency_consecutive_high_ghost = $emergencyConsecutiveHighGhost
        emergency_fallback_activated = $emergencyFallbackActivated
        emergency_recovery_activated = $emergencyRecoveryActivated
        emergency_rollback_activated = $emergencyRollbackActivated
        resume_update_reason = $resumeUpdateReason
        substantial_ghost_improve = $SubstantialGhostImprove
        max_cycles = $MaxCycles
        stop_after_hours = $StopAfterHours
        force_stage2_only = [bool]$ForceStage2Only
        no_improve_cycles_patience = $NoImproveCyclesPatience
        stagnation_stop_cycles = $StagnationStopCycles
        regression_stop_cycles = $RegressionStopCycles
        stable_ghost_target = $StableGhostTarget
        no_progress_cycles_for_ab_route = $NoProgressCyclesForABRoute
        stage2_has_potential = $stage2HasPotential
        stage2_potential_reason = $stage2PotentialReason
        skip_deep_stages = $skipDeepStagesCombined
        skip_deep_stages_reason = $skipDeepStagesCombinedReason
        should_stop_by_stagnation = $shouldStopByStagnation
        should_stop_by_regression = $shouldStopByRegression
        should_stop_by_ab_validation = $shouldStopByAB
        stop_by_ab_validation_reason = $stopByABValidationReason
        cycle_regressed = $cycleRegressed
        cycle_regress_reason = $cycleRegressReason
        cycle_quality_guard_blocked = $cycleQualityGuardBlocked
        cycle_quality_guard_reason = $cycleQualityGuardReason
        rolled_back_last_tune = $rolledBackLastTune
        rolled_back_tune_action = $rolledBackTuneAction
        regress_cycles = $regressCycles
        history = @($history.ToArray())
    }
    Write-JsonNoBom -Path "logs/modal_phase5/overnight_ghost_autoloop_latest.json" -Obj $status

    $md = @()
    $md += "# 过夜 Ghost AutoLoop"
    $md += ""
    $md += "- updated: $($status.updated_at)"
    $md += "- deadline: $($status.deadline)"
    $md += "- cycle: $($status.current_cycle)"
    $md += "- current_stage: $($status.current_stage)"
    $md += "- cycle_route_mode: $($status.cycle_route_mode)"
    $md += "- next_route_mode: $($status.next_route_mode)"
    $md += "- active_lane: $($status.active_lane)"
    $md += "- guard_tier: $($status.guard_tier)"
    $md += "- visual_guard_tier: $($status.visual_guard_tier)"
    if (-not [string]::IsNullOrWhiteSpace([string]$status.visual_guard_reason)) {
        $md += "- visual_guard_reason: $($status.visual_guard_reason)"
    }
    $md += "- active_eval_num_src_views: $($status.active_eval_num_src_views)"
    $md += "- active_eval_num_src_views_declared: $($status.active_eval_num_src_views_declared)"
    $md += "- active_eval_num_src_views_actual: $($status.active_eval_num_src_views_actual)"
    $md += "- active_eval_num_src_views_mismatch: $($status.active_eval_num_src_views_mismatch)"
    $md += "- active_cam_count: $($status.active_cam_count)"
    $md += "- active_quality_guard_blocked: $($status.active_quality_guard_blocked)"
    if (-not [string]::IsNullOrWhiteSpace([string]$status.active_quality_guard_reason)) {
        $md += "- active_quality_guard_reason: $($status.active_quality_guard_reason)"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$status.active_candidate_invalid_reason)) {
        $md += "- active_candidate_invalid_reason: $($status.active_candidate_invalid_reason)"
    }
    $md += "- active_precompute_source: $($status.active_precompute_source)"
    $md += "- active_precompute_source_requested: $($status.active_precompute_source_requested)"
    $md += "- active_precompute_source_resolved: $($status.active_precompute_source_resolved)"
    $md += "- active_precompute_fallback_used: $($status.active_precompute_fallback_used)"
    $md += "- active_precompute_timeout_hit: $($status.active_precompute_timeout_hit)"
    $md += "- active_precompute_mv_support_on: $($status.active_precompute_mv_support_on)"
    $md += "- active_point_target_blend_by_mv_support: $($status.active_point_target_blend_by_mv_support)"
    $md += "- active_point_mv_depth_pair_mode: $($status.active_point_mv_depth_pair_mode)"
    $md += "- p0_stage2_valid_rows_90m: $($status.p0_stage2_valid_rows_90m)"
    $md += "- p0_gate_pass: $($status.p0_gate_pass)"
    $md += "- p0_gate_reason: $($status.p0_gate_reason)"
    $md += "- decision_reason: $($status.decision_reason)"
    if (-not [string]::IsNullOrWhiteSpace([string]$status.rollback_reason)) {
        $md += "- rollback_reason: $($status.rollback_reason)"
    }
    $md += "- ab_route_pending_next_cycle: $($status.ab_route_pending_next_cycle)"
    $md += "- ab_route_variant: $($status.ab_route_variant)"
    $md += "- ab_route_executed: $($status.ab_route_executed)"
    $md += "- no_improve_cycles: $($status.no_improve_cycles) / $($status.no_improve_cycles_patience)"
    $md += "- no_substantial_improve_cycles: $($status.no_substantial_improve_cycles) / $($status.stagnation_stop_cycles)"
    $md += "- regress_cycles: $($status.regress_cycles) / $($status.regression_stop_cycles)"
    $md += "- global_best_ghost: $($status.global_best_ghost)"
    $md += "- global_best_psnr: $($status.global_best_psnr)"
    $md += "- global_best_ssim: $($status.global_best_ssim)"
    $md += "- global_best_wl1: $($status.global_best_wl1)"
    $md += "- global_best_geom: $($status.global_best_geom)"
    $md += "- global_best_ckpt: $($status.global_best_ckpt)"
    $md += "- global_best_visual_png: $($status.global_best_visual_png)"
    $md += "- global_best_hint_scope: family=$($status.global_best_hint_stage_family), stage1=$($status.global_best_hint_apply_stage1), stage2=$($status.global_best_hint_apply_stage2)"
    $md += "- next_cycle_tune_action: $($status.next_cycle_tune_action)"
    $md += "- emergency: enabled=$($status.emergency_enabled), state=$($status.emergency_state), profile=$($status.emergency_profile), reason=$($status.emergency_reason)"
    $md += "- emergency_target_ghost: $($status.emergency_target_ghost), emergency_last_cycle_best_ghost: $($status.emergency_last_cycle_best_ghost)"
    $md += "- emergency_hit_target: $($status.emergency_hit_target), emergency_consecutive_high_ghost: $($status.emergency_consecutive_high_ghost)"
    $md += "- force_stage2_only: $($status.force_stage2_only)"
    $md += "- stage2_has_potential: $($status.stage2_has_potential) ($($status.stage2_potential_reason))"
    $md += "- skip_deep_stages: $($status.skip_deep_stages) ($($status.skip_deep_stages_reason))"
    $md += "- should_stop_by_stagnation: $($status.should_stop_by_stagnation)"
    $md += "- should_stop_by_regression: $($status.should_stop_by_regression)"
    $md += "- should_stop_by_ab_validation: $($status.should_stop_by_ab_validation)"
    if (-not [string]::IsNullOrWhiteSpace([string]$status.stop_by_ab_validation_reason)) {
        $md += "- stop_by_ab_validation_reason: $($status.stop_by_ab_validation_reason)"
    }
    $md += "- regression_policy: 单变量回归时先回滚上一步参数并继续；仅在回归停机阈值触发时才考虑代码回滚"
    $md += ""
    $md += "## 最近轮次"
    foreach ($h in @($history | Select-Object -Last 3)) {
        $md += "- cycle=$($h.cycle), route=$($h.cycle_route_mode)->$($h.next_route_mode), best_stage=$($h.cycle_best_stage), best_ghost=$($h.cycle_best_ghost), best_psnr=$($h.cycle_best_psnr), best_ssim=$($h.cycle_best_ssim), improved=$($h.cycle_improved), substantial=$($h.cycle_substantial_improved), regressed=$($h.cycle_regressed), tune_next=$($h.tune_action_next)"
        $md += "  - lane=$($h.active_lane), tier=$($h.guard_tier), decision=$($h.stage2_lane_decision_reason)"
        if (-not [string]::IsNullOrWhiteSpace([string]$h.stage2_lane_rollback_reason)) {
            $md += "  - lane_rollback_reason=$($h.stage2_lane_rollback_reason)"
        }
        $md += "  - stage2_potential=$($h.stage2_has_potential), reason=$($h.stage2_potential_reason)"
        if ($h.skip_deep_stages) {
            $md += "  - skip_deep_stages_reason=$($h.skip_deep_stages_reason)"
        }
        if (-not [string]::IsNullOrWhiteSpace([string]$h.cycle_compare_png)) {
            $md += "  - compare_png=$($h.cycle_compare_png)"
        }
        if ($h.rolled_back_last_tune) {
            $md += "  - rollback=$($h.rolled_back_tune_action)"
        }
        if (-not [string]::IsNullOrWhiteSpace([string]$h.cycle_regress_reason)) {
            $md += "  - regress_reason=$($h.cycle_regress_reason)"
        }
        if ($h.cycle_quality_guard_blocked) {
            $md += "  - quality_guard_blocked=true reason=$($h.cycle_quality_guard_reason)"
        }
        if ($h.should_stop_by_ab_validation) {
            $md += "  - ab_validation_stop=true reason=$($h.stop_by_ab_validation_reason)"
        }
        $md += "  - visual=$($h.cycle_visual_conclusion)"
    }
    Set-Content -Path "logs/modal_phase5/overnight_ghost_autoloop_latest.md" -Value ($md -join "`n") -Encoding UTF8

    $reportBest = $bestCycle
    if ($reportBest -eq $null) {
        $reportBest = [pscustomobject]@{
            stage = "none"
            ghost = [double]::NaN
            psnr = [double]::NaN
            ssim = [double]::NaN
            wl1 = [double]::NaN
        }
    }
    Append-MentorCycleUpdate `
        -MentorPath $MentorUpdatePath `
        -Cycle $cycle `
        -Stages @($s1, $s2, $s3, $s4, $s5) `
        -CycleBest $reportBest `
        -GlobalBestGhost $globalBestGhost `
        -GlobalBestSsim $globalBestSsim `
        -NoSubstantialImproveCycles $noSubstantialImproveCycles `
        -TuneAction $pendingTuneAction `
        -CycleComparePng $cycleComparePng `
        -VisualConclusion $visualConclusion `
        -CycleRegressed $cycleRegressed `
        -CycleRegressReason $cycleRegressReason `
        -CycleQualityGuardBlocked $cycleQualityGuardBlocked `
        -CycleQualityGuardReason $cycleQualityGuardReason `
        -RolledBackLastTune $rolledBackLastTune `
        -RollbackAction $rolledBackTuneAction `
        -ShouldStop ($shouldStopByStagnation -or $shouldStopByRegression -or $shouldStopByAB) `
        -RouteMode $cycleRouteMode `
        -NextRouteMode $routeMode `
        -ActiveLane $activeLane `
        -GuardTier $cycleGuardTier `
        -DecisionReason $cycleDecisionReason `
        -RollbackReason $cycleRollbackReason `
        -LaneABest $laneABestSnapshot `
        -LaneBBest $laneBBestSnapshot `
        -Stage2HasPotential $stage2HasPotential `
        -Stage2PotentialReason $stage2PotentialReason `
        -SkipDeepStages $skipDeepStagesCombined `
        -SkipDeepStagesReason $skipDeepStagesCombinedReason `
        -ShouldStopByABValidation $shouldStopByAB `
        -StopByABValidationReason $stopByABValidationReason

    Write-Host "[autoloop] ===== cycle $cycle done, best ghost=$($status.global_best_ghost) ====="
    if ($noImproveCycles -ge [Math]::Max(1, [int]$NoImproveCyclesPatience)) {
        Write-Host "[autoloop] early stop by cycle-level no-improve patience"
        break
    }
    if ($shouldStopByStagnation) {
        Write-Host "[autoloop] stop by stagnation rule: no substantial ghost improvement"
        break
    }
    if ($shouldStopByRegression) {
        Write-Host "[autoloop] stop by regression rule: repeated regression, freeze code and move to focused A/B"
        break
    }
    if ($shouldStopByAB) {
        Write-Host "[autoloop] stop by A/B validation rule: $stopByABValidationReason"
        break
    }
}

Write-Host "[autoloop] finished"
exit 0

