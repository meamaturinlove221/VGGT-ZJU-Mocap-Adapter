param(
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$PseudoGeomSubdir = "vggt_geom",
    [string]$PretrainedCkpt = "model.pt",
    [string]$ResumeCkpt = "",
    [string]$ReuseShortFtCkpt = "",
    [string]$LrList = "4e-6,2e-6,1e-6",
    [string]$FreezeMode = "all_trainable",
    [string]$DepthScaleAlign = "median",
    [int]$EpochsShort = 3,
    [int]$EpochsLong = 6,
    [int]$MaxFramesShort = 400,
    [int]$MaxFramesLong = 0,
    [int]$EvalNumSamples = 60,
    [string]$EvalInferArgsExtra = "--num_src_views=6",
    [string]$DecoderCkpt = "",
    [double]$MinPSNR = 20.9,
    [double]$MinSSIM = 0.84,
    [double]$MaxWL1 = 0.08,
    [string]$CamNames = "Camera_B1,Camera_B2,Camera_B3,Camera_B4,Camera_B5,Camera_B6,Camera_B7,Camera_B8,Camera_B9,Camera_B10,Camera_B11,Camera_B12,Camera_B13,Camera_B14,Camera_B15,Camera_B16,Camera_B17,Camera_B18,Camera_B19,Camera_B20,Camera_B21,Camera_B22,Camera_B23",
    [double]$LambdaDepth = 1.0,
    [double]$LambdaPoint = 0.5,
    [double]$LambdaPointReproj = 0.0,
    [double]$LambdaPointNormalConsis = 0.0,
    [double]$LambdaPointMvDepth = 0.0,
    [double]$LambdaPointMvMask = 0.0,
    [double]$LambdaConf = 0.02,
    [int]$LambdaConfWarmupSteps = 0,
    [double]$LambdaGeomCons = 0.05,
    [double]$LambdaCam = 0.03,
    [int]$LambdaCamWarmupSteps = 0,
    [double]$CamRotWeight = 1.0,
    [double]$CamFovWeight = 0.2,
    [int]$CamWarmupSteps = 40,
    [double]$Jitter = 0.02,
    [double]$NoiseStd = 0.0,
    [double]$RobustL1Eps = 0.01,
    [double]$ConfWeightThr = 0.05,
    [double]$ConfWeightGamma = 1.0,
    [double]$ConfWeightPerViewQuantile = 0.0,
    [int]$ConfWeightPerViewMinValid = 16,
    [string]$GramDynEnable = "off",
    [int]$GramDynLayerIdx = -1,
    [double]$GramDynQuantile = 0.30,
    [double]$GramDynWeightFloor = 0.25,
    [int]$GramDynWarmupSteps = 40,
    [string]$DynProxyEnable = "off",
    [string]$DynProxyMode = "fg_static_soft",
    [string]$DynProxyUseGram = "on",
    [string]$DynProxyUseSupport = "on",
    [double]$DynProxyFloor = 0.35,
    [int]$DynProxyWarmupSteps = 40,
    [double]$PointConsTau = 0.03,
    [double]$PointConsWeightFloor = 0.2,
    [double]$PointConsClipMinQv = 1e-6,
    [double]$PointConsQuantile = 0.5,
    [string]$PointConsFocus = "inlier",
    [double]$PointResidualQuantile = 1.0,
    [string]$PointResidualFocus = "inlier",
    [double]$PointResidualBoost = 0.0,
    [double]$PointResidualBoostCap = 4.0,
    [string]$PointTargetMode = "pointmap",
    [double]$PointTargetBlendAlpha = 0.7,
    [double]$PointTargetBlendAlphaMin = 0.0,
    [double]$PointTargetBlendAlphaMax = 1.0,
    [double]$PointTargetBlendRelGain = 1.0,
    [double]$PointTargetBlendMvGain = 1.0,
    [string]$PointTargetBlendByReliability = "on",
    [string]$PointTargetBlendByMvSupport = "off",
    [string]$PointTargetBlendMvRegionMode = "all",
    [string]$PointTargetBlendMvPolicy = "weak_to_depth",
    [double]$PointTargetConsensusAlphaFloor = 0.0,
    [string]$TargetPointFrame = "auto",
    [string]$PredPointFrame = "auto",
    [string]$UseFgMask = "off",
    [string]$FgMaskSource = "auto",
    [int]$FgMaskErodePx = 0,
    [int]$PointLossFgErodePx = 0,
    [double]$FgSupervisionBoost = 1.0,
    [double]$FgSupervisionBgFloor = 0.0,
    [ValidateSet("all","interior_only")]
    [string]$FgSupervisionRegionMode = "all",
    [int]$FgSupervisionRegionErodePx = 0,
    [double]$LambdaFgConfPresence = 0.0,
    [double]$FgConfPresenceTargetRatio = 0.9,
    [double]$LambdaFgStructureDepthEdge = 0.0,
    [int]$FgStructureBboxMarginPx = 12,
    [int]$FgStructureBboxMinSidePx = 24,
    [ValidateSet("bbox","bbox_fg_interior")]
    [string]$FgStructureRegionMode = "bbox",
    [int]$FgStructureRegionErodePx = 0,
    [int]$FgStructureDepthEdgeWarmupSteps = 0,
    [int]$FgStructureBoundaryProbePx = 2,
    [ValidateSet("off","target_edge_quantile")]
    [string]$FgStructureEdgeSupportMode = "off",
    [double]$FgStructureEdgeSupportQuantile = 0.0,
    [int]$FgStructureEdgeSupportMinPx = 32,
    [ValidateSet("uniform","target_edge_sqrt")]
    [string]$FgStructureEdgeWeightMode = "uniform",
    [int]$FgStructureBoundaryFalloffPx = 0,
    [ValidateSet("off","largest_soft")]
    [string]$FgStructureComponentBiasMode = "off",
    [double]$FgStructureComponentBiasThresholdRatio = 0.25,
    [double]$FgStructureComponentBiasOtherScale = 1.0,
    [ValidateSet("off","front_soft")]
    [string]$FgStructureFrontDepthBiasMode = "off",
    [double]$FgStructureFrontDepthBiasTau = 0.75,
    [double]$FgStructureFrontDepthBiasCenterQuantile = 0.55,
    [double]$LambdaPointMvOutsideRing = 0.0,
    [int]$PointMvOutsideRingPx = 3,
    [string]$SupervisionWeightMode = "conf",
    [double]$SupervisionWeightMixAlpha = 0.5,
    [int]$PointReprojWarmupSteps = 40,
    [double]$PointReprojClampPx = 64.0,
    [string]$PointMvConsistency = "off",
    [double]$PointMvTolAbs = 0.03,
    [double]$PointMvTolRel = 0.05,
    [double]$PointMvWeightFloor = 0.2,
    [int]$PointMvStride = 2,
    [int]$PointMvDepthMaxPairs = 3,
    [string]$PointMvDepthPairMode = "adjacent",
    [int]$PointMvDepthWarmupSteps = 40,
    [string]$PointMvDepthRegionMode = "all",
    [int]$PointMvMaskWarmupSteps = 40,
    [string]$PointMvDepthInlierOnly = "off",
    [double]$PointMvDepthErrQuantile = 1.0,
    [double]$PointMvDepthOutlierBoost = 0.0,
    [double]$PointMvDepthOutlierCap = 3.0,
    [string]$PointMvDepthTgtValidMode = "hard",
    [double]$PointMvDepthTgtValidFloor = 0.2,
    [double]$PointMvDepthMinTgtValidRatio = 0.0,
    [double]$PointMvMaskMinTgtFgRatio = 0.0,
    [double]$PointMvMaskHitThr = 0.5,
    [int]$PointMvMaskSoftBlurPx = 0,
    [int]$PointMvMaskSoftBlurIters = 1,
    [double]$PointMvMaskSoftMix = 0.0,
    [double]$PointMvMaskSoftHitThr = 0.35,
    [string]$PointMvDepthTgtValidScaleMode = "off",
    [double]$PointMvDepthTgtValidScaleThr = 0.01,
    [string]$PointMvDepthAdaptMode = "off",
    [double]$PointMvDepthAdaptTargetValid = 0.01,
    [double]$PointMvDepthAdaptMinScale = 1.0,
    [double]$PointMvDepthAdaptMaxScale = 32.0,
    [string]$PointSupportMode = "direct",
    [double]$PointSupportFloor = 0.0,
    [string]$PointMvDepthSupportMode = "direct",
    [double]$PointMvDepthSupportFloor = 0.0,
    [string]$PointMvMaskSupportMode = "inverse",
    [double]$PointMvMaskSupportFloor = 0.0,
    [int]$PointMvDepthFgErodePx = 0,
    [double]$PointLossScaleDepthUnproject = 0.5,
    [int]$PointWarmupSteps = 40,
    [int]$PointNormalConsisWarmupSteps = 40,
    [double]$LrBackboneScale = 0.05,
    [double]$LrHeadScale = 1.0,
    [double]$LrCameraScale = 0.1,
    [double]$MinMeanStepUpdateRatio = 0.0,
    [double]$GradClip = 0.5,
    [bool]$Tf32 = $true,
    [bool]$Amp = $true,
    [bool]$StrictDeterministic = $false,
    [string]$PointmapSource = "auto",
    [string]$PointHeadFrame = "auto",
    [string]$UnprojectImpl = "legacy",
    [string]$PrecomputeMvSupportOn = "off",
    [double]$PrecomputeMvSupportTolAbs = 0.06,
    [double]$PrecomputeMvSupportTolRel = 0.10,
    [int]$PrecomputeMvSupportStride = 2,
    [string]$PrecomputeMvSupportMode = "linear",
    [double]$PrecomputeMvSupportFloor = 0.05,
    [double]$PrecomputeMvSupportGamma = 1.0,
    [double]$PrecomputeMvSupportClipThr = 0.20,
    [double]$PrecomputeMvSupportClipFloor = 0.30,
    [double]$PrecomputeMvSupportHardThr = -1.0,
    [double]$PrecomputeMvConfValidFloor = 0.02,
    [string]$PrecomputeMvSupportSave = "off",
    [string]$PrecomputeMvSupportSaveRawConf = "off",
    [string]$PrecomputeMvSupportRegionMode = "auto",
    [string]$PrecomputeMvSupportFgMaskSource = "auto",
    [int]$PrecomputeMvSupportFgErodePx = 5,
    [int]$PrecomputeMvSupportFgPreservePx = 5,
    [int]$EvalEverySteps = 20,
    [int]$DebugMetricsEverySteps = 0,
    [int]$DebugVisEverySteps = 0,
    [int]$DebugVisMaxSteps = 0,
    [int]$DebugVisViews = 1,
    [string]$DebugVisDir = "",
    [int]$DownloadFtDebugVisCount = 12,
    [string]$DownloadFtDebugVisSteps = "",
    [int]$CkptWaitTimeoutSec = 1800,
    [bool]$EnableExtendedCkptWaitOnMissing = $true,
    [int]$CkptExtendedWaitTimeoutSec = 1200,
    [int]$CkptWaitPollSec = 20,
    [int]$CkptMissingRetryCount = 1,
    [int]$CkptMissingRetrySleepSec = 20,
    [bool]$EnableResumeCkptFallbackOnShortCkptMissing = $true,
    [int]$NoSpaceRetryCount = 1,
    [string]$NoSpaceCleanupRoot = "/vggt/finetune",
    [int]$NoSpaceCleanupKeepRecentDirs = 220,
    [int]$NoSpaceCleanupDeleteLimit = 120,
    [int]$ModalRunTimeoutSec = 3600,
    [int]$ModalRunPollSec = 20,
    [int]$ModalRunNoOutputTimeoutSec = 600,
    [int]$PrecomputeNoOutputTimeoutSec = 1800,
    [int]$ModalRunNoOutputMaxRetries = 1,
    [bool]$ModalRunQuiet = $true,
    [bool]$ShortFinetuneAllowQuietNoOutputBypass = $true,
    [bool]$EnablePrecomputePointmapFallbackOnNoOutput = $true,
    [string]$PrecomputeFallbackPointmapSource = "point_head",
    [int]$PrecomputeFallbackNoOutputTimeoutSec = 1200,
    [int]$EvalNoOutputTimeoutSec = 2400,
    [string]$ModalRunHeartbeatPath = "logs/modal_phase5/modal_run_progress_latest.json",
    [int]$EarlyStopPatience = 1,
    [double]$MinImprove = 0.0001,
    [int]$MaxStepsPerEpoch = 80,
    [switch]$RunLongOnImprove
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

function Parse-Lrs([string]$Raw) {
    return @(
        ($Raw -split '[,\s]+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { $_.Trim() })
    )
}

function Find-ReuseFtDebugLocalPath(
    [string]$ReuseFtCkptPath,
    [ValidateSet("metrics", "summary")]
    [string]$Kind
) {
    if ([string]::IsNullOrWhiteSpace($ReuseFtCkptPath)) { return "" }
    $m = [regex]::Match([string]$ReuseFtCkptPath, "lr_[^/_]+_(\d{8}_\d{6})/ckpt/model_ft_zju(?:_last)?\.pt$")
    if (-not $m.Success) { return "" }
    $runTs = [string]$m.Groups[1].Value
    if ([string]::IsNullOrWhiteSpace($runTs)) { return "" }
    $suffix = if ($Kind -eq "metrics") { "metrics.jsonl" } else { "summary.json" }
    $matches = @(Get-ChildItem -Path (Join-Path $CodeDir "logs/modal_phase5") -Filter ("ftdebug_*_{0}_short_{1}" -f $runTs, $suffix) -ErrorAction SilentlyContinue)
    if ($matches.Count -gt 0) {
        return $matches[0].FullName
    }
    $legacy = Join-Path $CodeDir ("logs/modal_phase5/ftdebug_1e-6_{0}_short_{1}" -f $runTs, $suffix)
    if (Test-Path $legacy) {
        return $legacy
    }
    return ""
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

function Parse-NumSrcViewsFromInferArgs([string]$InferArgs) {
    if ([string]::IsNullOrWhiteSpace($InferArgs)) { return "" }
    $m = [regex]::Match([string]$InferArgs, "(?:^|\s)--num_src_views(?:\s+|=)(\d+)(?:\s|$)")
    if ($m.Success) { return [string]$m.Groups[1].Value }
    return ""
}

$pairModeBeforeNormalize = [string]$PointMvDepthPairMode
$PointMvDepthPairMode = Resolve-PointMvDepthPairMode -Raw $PointMvDepthPairMode -Default "adjacent"
if (([string]$pairModeBeforeNormalize).Trim().ToLowerInvariant() -ne ([string]$PointMvDepthPairMode)) {
    Write-Host "[lr-sweep] normalize point_mv_depth_pair_mode: raw='$pairModeBeforeNormalize' -> '$PointMvDepthPairMode'"
}

function Get-CamCountFromCamNames([string]$RawCamNames) {
    if ([string]::IsNullOrWhiteSpace($RawCamNames)) { return "" }
    $tokens = @(
        $RawCamNames -split "[,\s;|]+" |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_.Trim() }
    )
    if ($tokens.Count -le 0) { return "" }
    return [int]$tokens.Count
}

function Resolve-FtCandidateInvalidReason(
    [string]$Status,
    [string]$Reason
) {
    $reasonNorm = ([string]$Reason).Trim()
    if ([string]::IsNullOrWhiteSpace($reasonNorm)) { return "" }
    if ($reasonNorm -match "(?i)ckpt_handoff_failed|failed finding central directory|pytorchstreamreader failed reading zip archive|zip_probe_failed|checkpoint unreadable|invalid checkpoint zip") { return "ckpt_handoff_failed" }
    if ($reasonNorm -match "(?i)eval_num_src_views_mismatch|src_views_mismatch") { return "src_views_mismatch" }
    if ($reasonNorm -match "(?i)quality_guard") { return "quality_guard_blocked" }
    if ($reasonNorm -match "(?i)visual_guard") { return "visual_guard_blocked" }
    if ($reasonNorm -match "(?i)zero_samples\(n=0\)|eval_empty|post_eval_no_valid_metrics") { return "eval_empty" }
    if ($reasonNorm -match "(?i)failed to fetch/parse metrics|evaluate short candidates failed|eval_failed") { return "eval_failed" }
    if ($reasonNorm -match "(?i)(?:heartbeat_stall_timeout|no_output_timeout)_\d+s") { return "no_output_timeout" }
    if ($reasonNorm -match "(?i)ckpt_tmp_fallback|checkpoint tmp fallback") { return "ckpt_tmp_fallback" }
    if ($reasonNorm -match "(?i)checkpoint .*missing|ckpt.*missing|checkpoint wait timeout|checkpoint unavailable") { return "ckpt_missing" }
    if ($reasonNorm -match "(?i)precompute_empty|geom_subdir_empty|missing_seq_npz|no_npz") { return "precompute_empty" }
    if ($reasonNorm -match "(?i)precompute|cuda out of memory|out of memory|oom") { return "precompute_failed" }
    return ""
}

function Finalize-SweepRows([object[]]$Rows) {
    $evalNumSrcViews = Parse-NumSrcViewsFromInferArgs -InferArgs $EvalInferArgsExtra
    $camCount = Get-CamCountFromCamNames -RawCamNames $CamNames
    foreach ($row in @($Rows)) {
        if ($null -eq $row) { continue }
        if (-not $row.PSObject.Properties["cam_count_used"]) {
            $row | Add-Member -NotePropertyName cam_count_used -NotePropertyValue $camCount -Force
        } else {
            $row.cam_count_used = $camCount
        }
        if (-not $row.PSObject.Properties["eval_num_src_views"]) {
            $row | Add-Member -NotePropertyName eval_num_src_views -NotePropertyValue $evalNumSrcViews -Force
        } else {
            $row.eval_num_src_views = $evalNumSrcViews
        }
        if (-not $row.PSObject.Properties["eval_num_src_views_declared"]) {
            $row | Add-Member -NotePropertyName eval_num_src_views_declared -NotePropertyValue $evalNumSrcViews -Force
        } else {
            $row.eval_num_src_views_declared = $evalNumSrcViews
        }
        if (-not $row.PSObject.Properties["pointmap_source_requested"]) {
            $row | Add-Member -NotePropertyName pointmap_source_requested -NotePropertyValue ([string]$PointmapSource) -Force
        }
        if (-not $row.PSObject.Properties["pointmap_source_resolved"]) {
            $row | Add-Member -NotePropertyName pointmap_source_resolved -NotePropertyValue "" -Force
        }
        if (-not $row.PSObject.Properties["precompute_fallback_used"]) {
            $row | Add-Member -NotePropertyName precompute_fallback_used -NotePropertyValue $false -Force
        }
        if (-not $row.PSObject.Properties["precompute_timeout_hit"]) {
            $row | Add-Member -NotePropertyName precompute_timeout_hit -NotePropertyValue $false -Force
        }
        $supportGenerationActive = To-BoolLoose $(if ($row.PSObject.Properties["precompute_mv_support_on"]) { $row.precompute_mv_support_on } else { $PrecomputeMvSupportOn }) $false
        if (-not $row.PSObject.Properties["support_generation_active"]) {
            $row | Add-Member -NotePropertyName support_generation_active -NotePropertyValue $(if ($supportGenerationActive) { 1.0 } else { 0.0 }) -Force
        } else {
            $row.support_generation_active = $(if ($supportGenerationActive) { 1.0 } else { 0.0 })
        }
        if (-not $row.PSObject.Properties["point_support_mode"]) {
            $row | Add-Member -NotePropertyName point_support_mode -NotePropertyValue ([string]$PointSupportMode) -Force
        } else {
            $row.point_support_mode = [string]$PointSupportMode
        }
        if (-not $row.PSObject.Properties["point_mv_depth_support_mode"]) {
            $row | Add-Member -NotePropertyName point_mv_depth_support_mode -NotePropertyValue ([string]$PointMvDepthSupportMode) -Force
        } else {
            $row.point_mv_depth_support_mode = [string]$PointMvDepthSupportMode
        }
        if (-not $row.PSObject.Properties["point_mv_mask_support_mode"]) {
            $row | Add-Member -NotePropertyName point_mv_mask_support_mode -NotePropertyValue ([string]$PointMvMaskSupportMode) -Force
        } else {
            $row.point_mv_mask_support_mode = [string]$PointMvMaskSupportMode
        }
        if (-not $row.PSObject.Properties["fg_supervision_boost"]) {
            $row | Add-Member -NotePropertyName fg_supervision_boost -NotePropertyValue ([double]$FgSupervisionBoost) -Force
        } else {
            $row.fg_supervision_boost = [double]$FgSupervisionBoost
        }
        if (-not $row.PSObject.Properties["fg_supervision_bg_floor"]) {
            $row | Add-Member -NotePropertyName fg_supervision_bg_floor -NotePropertyValue ([double]$FgSupervisionBgFloor) -Force
        } else {
            $row.fg_supervision_bg_floor = [double]$FgSupervisionBgFloor
        }
        if (-not $row.PSObject.Properties["fg_supervision_region_mode"]) {
            $row | Add-Member -NotePropertyName fg_supervision_region_mode -NotePropertyValue ([string]$FgSupervisionRegionMode) -Force
        } else {
            $row.fg_supervision_region_mode = [string]$FgSupervisionRegionMode
        }
        if (-not $row.PSObject.Properties["fg_supervision_region_erode_px"]) {
            $row | Add-Member -NotePropertyName fg_supervision_region_erode_px -NotePropertyValue ([int]$FgSupervisionRegionErodePx) -Force
        } else {
            $row.fg_supervision_region_erode_px = [int]$FgSupervisionRegionErodePx
        }
        if (-not $row.PSObject.Properties["lambda_fg_conf_presence"]) {
            $row | Add-Member -NotePropertyName lambda_fg_conf_presence -NotePropertyValue ([double]$LambdaFgConfPresence) -Force
        } else {
            $row.lambda_fg_conf_presence = [double]$LambdaFgConfPresence
        }
        if (-not $row.PSObject.Properties["fg_conf_presence_target_ratio"]) {
            $row | Add-Member -NotePropertyName fg_conf_presence_target_ratio -NotePropertyValue ([double]$FgConfPresenceTargetRatio) -Force
        } else {
            $row.fg_conf_presence_target_ratio = [double]$FgConfPresenceTargetRatio
        }
        if (-not $row.PSObject.Properties["lambda_fg_structure_depth_edge"]) {
            $row | Add-Member -NotePropertyName lambda_fg_structure_depth_edge -NotePropertyValue ([double]$LambdaFgStructureDepthEdge) -Force
        } else {
            $row.lambda_fg_structure_depth_edge = [double]$LambdaFgStructureDepthEdge
        }
        if (-not $row.PSObject.Properties["fg_structure_bbox_margin_px"]) {
            $row | Add-Member -NotePropertyName fg_structure_bbox_margin_px -NotePropertyValue ([int]$FgStructureBboxMarginPx) -Force
        } else {
            $row.fg_structure_bbox_margin_px = [int]$FgStructureBboxMarginPx
        }
        if (-not $row.PSObject.Properties["fg_structure_bbox_min_side_px"]) {
            $row | Add-Member -NotePropertyName fg_structure_bbox_min_side_px -NotePropertyValue ([int]$FgStructureBboxMinSidePx) -Force
        } else {
            $row.fg_structure_bbox_min_side_px = [int]$FgStructureBboxMinSidePx
        }
        if (-not $row.PSObject.Properties["fg_structure_region_mode"]) {
            $row | Add-Member -NotePropertyName fg_structure_region_mode -NotePropertyValue ([string]$FgStructureRegionMode) -Force
        } else {
            $row.fg_structure_region_mode = [string]$FgStructureRegionMode
        }
        if (-not $row.PSObject.Properties["fg_structure_region_erode_px"]) {
            $row | Add-Member -NotePropertyName fg_structure_region_erode_px -NotePropertyValue ([int]$FgStructureRegionErodePx) -Force
        } else {
            $row.fg_structure_region_erode_px = [int]$FgStructureRegionErodePx
        }
        if (-not $row.PSObject.Properties["fg_structure_depth_edge_warmup_steps"]) {
            $row | Add-Member -NotePropertyName fg_structure_depth_edge_warmup_steps -NotePropertyValue ([int]$FgStructureDepthEdgeWarmupSteps) -Force
        } else {
            $row.fg_structure_depth_edge_warmup_steps = [int]$FgStructureDepthEdgeWarmupSteps
        }
        if (-not $row.PSObject.Properties["fg_structure_boundary_probe_px"]) {
            $row | Add-Member -NotePropertyName fg_structure_boundary_probe_px -NotePropertyValue ([int]$FgStructureBoundaryProbePx) -Force
        } else {
            $row.fg_structure_boundary_probe_px = [int]$FgStructureBoundaryProbePx
        }
        if (-not $row.PSObject.Properties["fg_structure_edge_support_mode"]) {
            $row | Add-Member -NotePropertyName fg_structure_edge_support_mode -NotePropertyValue ([string]$FgStructureEdgeSupportMode) -Force
        } else {
            $row.fg_structure_edge_support_mode = [string]$FgStructureEdgeSupportMode
        }
        if (-not $row.PSObject.Properties["fg_structure_edge_support_quantile"]) {
            $row | Add-Member -NotePropertyName fg_structure_edge_support_quantile -NotePropertyValue ([double]$FgStructureEdgeSupportQuantile) -Force
        } else {
            $row.fg_structure_edge_support_quantile = [double]$FgStructureEdgeSupportQuantile
        }
        if (-not $row.PSObject.Properties["fg_structure_edge_support_min_px"]) {
            $row | Add-Member -NotePropertyName fg_structure_edge_support_min_px -NotePropertyValue ([int]$FgStructureEdgeSupportMinPx) -Force
        } else {
            $row.fg_structure_edge_support_min_px = [int]$FgStructureEdgeSupportMinPx
        }
        if (-not $row.PSObject.Properties["fg_structure_edge_weight_mode"]) {
            $row | Add-Member -NotePropertyName fg_structure_edge_weight_mode -NotePropertyValue ([string]$FgStructureEdgeWeightMode) -Force
        } else {
            $row.fg_structure_edge_weight_mode = [string]$FgStructureEdgeWeightMode
        }
        if (-not $row.PSObject.Properties["fg_structure_boundary_falloff_px"]) {
            $row | Add-Member -NotePropertyName fg_structure_boundary_falloff_px -NotePropertyValue ([int]$FgStructureBoundaryFalloffPx) -Force
        } else {
            $row.fg_structure_boundary_falloff_px = [int]$FgStructureBoundaryFalloffPx
        }
        if (-not $row.PSObject.Properties["fg_structure_component_bias_mode"]) {
            $row | Add-Member -NotePropertyName fg_structure_component_bias_mode -NotePropertyValue ([string]$FgStructureComponentBiasMode) -Force
        } else {
            $row.fg_structure_component_bias_mode = [string]$FgStructureComponentBiasMode
        }
        if (-not $row.PSObject.Properties["fg_structure_component_bias_threshold_ratio"]) {
            $row | Add-Member -NotePropertyName fg_structure_component_bias_threshold_ratio -NotePropertyValue ([double]$FgStructureComponentBiasThresholdRatio) -Force
        } else {
            $row.fg_structure_component_bias_threshold_ratio = [double]$FgStructureComponentBiasThresholdRatio
        }
        if (-not $row.PSObject.Properties["fg_structure_component_bias_other_scale"]) {
            $row | Add-Member -NotePropertyName fg_structure_component_bias_other_scale -NotePropertyValue ([double]$FgStructureComponentBiasOtherScale) -Force
        } else {
            $row.fg_structure_component_bias_other_scale = [double]$FgStructureComponentBiasOtherScale
        }
        if (-not $row.PSObject.Properties["fg_structure_front_depth_bias_mode"]) {
            $row | Add-Member -NotePropertyName fg_structure_front_depth_bias_mode -NotePropertyValue ([string]$FgStructureFrontDepthBiasMode) -Force
        } else {
            $row.fg_structure_front_depth_bias_mode = [string]$FgStructureFrontDepthBiasMode
        }
        if (-not $row.PSObject.Properties["fg_structure_front_depth_bias_tau"]) {
            $row | Add-Member -NotePropertyName fg_structure_front_depth_bias_tau -NotePropertyValue ([double]$FgStructureFrontDepthBiasTau) -Force
        } else {
            $row.fg_structure_front_depth_bias_tau = [double]$FgStructureFrontDepthBiasTau
        }
        if (-not $row.PSObject.Properties["fg_structure_front_depth_bias_center_quantile"]) {
            $row | Add-Member -NotePropertyName fg_structure_front_depth_bias_center_quantile -NotePropertyValue ([double]$FgStructureFrontDepthBiasCenterQuantile) -Force
        } else {
            $row.fg_structure_front_depth_bias_center_quantile = [double]$FgStructureFrontDepthBiasCenterQuantile
        }
        if (-not $row.PSObject.Properties["lambda_point_mv_outside_ring"]) {
            $row | Add-Member -NotePropertyName lambda_point_mv_outside_ring -NotePropertyValue ([double]$LambdaPointMvOutsideRing) -Force
        } else {
            $row.lambda_point_mv_outside_ring = [double]$LambdaPointMvOutsideRing
        }
        if (-not $row.PSObject.Properties["point_mv_outside_ring_px"]) {
            $row | Add-Member -NotePropertyName point_mv_outside_ring_px -NotePropertyValue ([int]$PointMvOutsideRingPx) -Force
        } else {
            $row.point_mv_outside_ring_px = [int]$PointMvOutsideRingPx
        }
        if (-not $row.PSObject.Properties["tf32"]) {
            $row | Add-Member -NotePropertyName tf32 -NotePropertyValue ([bool]$Tf32) -Force
        } else {
            $row.tf32 = [bool]$Tf32
        }
        if (-not $row.PSObject.Properties["amp"]) {
            $row | Add-Member -NotePropertyName amp -NotePropertyValue ([bool]$Amp) -Force
        } else {
            $row.amp = [bool]$Amp
        }
        if (-not $row.PSObject.Properties["strict_deterministic"]) {
            $row | Add-Member -NotePropertyName strict_deterministic -NotePropertyValue ([bool]$StrictDeterministic) -Force
        } else {
            $row.strict_deterministic = [bool]$StrictDeterministic
        }
        foreach ($kv in @(
            @{ Name = "runner_tf32"; Value = [bool]$Tf32 },
            @{ Name = "runner_amp"; Value = [bool]$Amp },
            @{ Name = "runner_strict_deterministic"; Value = [bool]$StrictDeterministic },
            @{ Name = "precompute_tf32"; Value = [bool]$Tf32 },
            @{ Name = "precompute_amp"; Value = [bool]$Amp },
            @{ Name = "precompute_strict_deterministic"; Value = [bool]$StrictDeterministic },
            @{ Name = "teacher_tf32"; Value = [bool]$Tf32 },
            @{ Name = "teacher_amp"; Value = [bool]$Amp },
            @{ Name = "teacher_deterministic"; Value = [bool]$StrictDeterministic }
        )) {
            if (-not $row.PSObject.Properties[$kv.Name]) {
                $row | Add-Member -NotePropertyName $kv.Name -NotePropertyValue $kv.Value -Force
            } else {
                $row.($kv.Name) = $kv.Value
            }
        }
        $precomputeSource = ""
        try { $precomputeSource = [string]$row.pointmap_source_resolved } catch {}
        if ([string]::IsNullOrWhiteSpace($precomputeSource)) {
            try { $precomputeSource = [string]$row.pointmap_source_requested } catch {}
        }
        if (-not $row.PSObject.Properties["precompute_source"]) {
            $row | Add-Member -NotePropertyName precompute_source -NotePropertyValue $precomputeSource -Force
        } else {
            $row.precompute_source = $precomputeSource
        }
        $candidateInvalidReason = ""
        try {
            if ($row.PSObject.Properties["candidate_invalid_reason"]) {
                $candidateInvalidReason = [string]$row.candidate_invalid_reason
            }
        } catch {}
        if ([string]::IsNullOrWhiteSpace($candidateInvalidReason)) {
            $candidateInvalidReason = Resolve-FtCandidateInvalidReason `
                -Status ([string]$row.status) `
                -Reason ([string]$row.reason)
        }
        if (-not $row.PSObject.Properties["candidate_invalid_reason"]) {
            $row | Add-Member -NotePropertyName candidate_invalid_reason -NotePropertyValue $candidateInvalidReason -Force
        } else {
            $row.candidate_invalid_reason = $candidateInvalidReason
        }
        $supportMetricDefaults = [ordered]@{
            point_support_path_active = 0.0
            point_mv_depth_support_path_active = 0.0
            point_mv_mask_support_path_active = 0.0
            point_target_blend_mv_support_active = 0.0
            mv_support_raw_mean = [double]::NaN
            mv_support_valid_ratio = [double]::NaN
            mv_support_fg_valid_ratio = [double]::NaN
            mv_support_bg_valid_ratio = [double]::NaN
            mv_support_pair_count_eff = [double]::NaN
            mv_support_conf_mean = [double]::NaN
            mv_support_nan_ratio = [double]::NaN
            depth_conf_delta_mean = [double]::NaN
            depth_conf_fg_preserved_active = [double]::NaN
            depth_conf_fg_preserve_px = [double]::NaN
            depth_conf_fg_exact_ratio = [double]::NaN
            depth_conf_fg_preserve_ratio = [double]::NaN
            depth_conf_fg_raw_mean = [double]::NaN
            depth_conf_fg_after_support_mean = [double]::NaN
            depth_conf_fg_final_mean = [double]::NaN
            mv_support_generation_region_mode = ""
            mv_support_generation_fg_mask_source = ""
            point_mv_mode = ""
            point_mv_support_mean = [double]::NaN
            point_mv_support_p10 = [double]::NaN
            point_mv_support_p90 = [double]::NaN
            point_mv_support_fg_mean = [double]::NaN
            point_mv_support_fg_p10 = [double]::NaN
            point_mv_support_fg_p90 = [double]::NaN
            point_mv_support_bg_mean = [double]::NaN
            point_mv_support_bg_p10 = [double]::NaN
            point_mv_support_bg_p90 = [double]::NaN
            point_mv_pseudo_support_mean = [double]::NaN
            point_mv_pseudo_support_p10 = [double]::NaN
            point_mv_pseudo_support_p90 = [double]::NaN
            point_mv_pseudo_support_fg_mean = [double]::NaN
            point_mv_pseudo_support_fg_p10 = [double]::NaN
            point_mv_pseudo_support_fg_p90 = [double]::NaN
            point_mv_pseudo_support_bg_mean = [double]::NaN
            point_mv_pseudo_support_bg_p10 = [double]::NaN
            point_mv_pseudo_support_bg_p90 = [double]::NaN
            point_support_eff_mean = [double]::NaN
            point_support_eff_p10 = [double]::NaN
            point_support_eff_p90 = [double]::NaN
            point_support_eff_fg_mean = [double]::NaN
            point_support_eff_fg_p10 = [double]::NaN
            point_support_eff_fg_p90 = [double]::NaN
            point_support_eff_bg_mean = [double]::NaN
            point_support_eff_bg_p10 = [double]::NaN
            point_support_eff_bg_p90 = [double]::NaN
            point_mv_depth_support_eff_mean = [double]::NaN
            point_mv_depth_support_eff_p10 = [double]::NaN
            point_mv_depth_support_eff_p90 = [double]::NaN
            point_mv_depth_support_eff_fg_mean = [double]::NaN
            point_mv_depth_support_eff_fg_p10 = [double]::NaN
            point_mv_depth_support_eff_fg_p90 = [double]::NaN
            point_mv_depth_support_eff_bg_mean = [double]::NaN
            point_mv_depth_support_eff_bg_p10 = [double]::NaN
            point_mv_depth_support_eff_bg_p90 = [double]::NaN
            point_mv_mask_support_eff_mean = [double]::NaN
            point_mv_mask_support_eff_p10 = [double]::NaN
            point_mv_mask_support_eff_p90 = [double]::NaN
            point_mv_mask_support_eff_fg_mean = [double]::NaN
            point_mv_mask_support_eff_fg_p10 = [double]::NaN
            point_mv_mask_support_eff_fg_p90 = [double]::NaN
            point_mv_mask_support_eff_bg_mean = [double]::NaN
            point_mv_mask_support_eff_bg_p10 = [double]::NaN
            point_mv_mask_support_eff_bg_p90 = [double]::NaN
            fg_structure_depth_edge_active = 0.0
            fg_structure_bbox_cover = 0.0
            fg_structure_region_cover = 0.0
            fg_structure_effective_cover = 0.0
            fg_structure_boundary_probe_cover = 0.0
            fg_structure_bbox_active_ratio = 0.0
            fg_structure_region_active_ratio = 0.0
            fg_structure_boundary_band_active_ratio = 0.0
            fg_structure_depth_edge_active_views = 0.0
            fg_structure_depth_edge_boundary_active_views = 0.0
            fg_structure_depth_edge_loss = 0.0
            fg_structure_depth_edge_loss_main = 0.0
            fg_structure_depth_edge_loss_boundary_probe = 0.0
            fg_structure_depth_edge_loss_interior = 0.0
            fg_structure_depth_edge_loss_boundary_band = 0.0
            fg_structure_depth_edge_pred_mean = 0.0
            fg_structure_depth_edge_tgt_mean = 0.0
            fg_structure_depth_edge_boundary_probe_pred_mean = 0.0
            fg_structure_depth_edge_boundary_probe_tgt_mean = 0.0
            fg_structure_depth_edge_boundary_pred_mean = 0.0
            fg_structure_depth_edge_boundary_tgt_mean = 0.0
            fg_structure_target_edge_support_active = 0.0
            fg_structure_target_edge_support_views = 0.0
            fg_structure_target_edge_support_cover = 0.0
            fg_structure_target_edge_support_region_share = 0.0
            fg_structure_target_edge_support_threshold_mean = 0.0
            fg_structure_main_weight_mean = 0.0
            fg_structure_boundary_distance_weight_share = 1.0
            fg_structure_front_depth_bias_weight_share = 1.0
            fg_structure_front_depth_bias_active_views = 0.0
            main_support_depth_mode_count = 0.0
            main_support_back_mode_share = 0.0
            main_support_front_back_gap = 0.0
            main_support_depth_hist_peak_ratio = 0.0
            main_support_secondary_risk = 0.0
            main_support_depth_mode_active_views = 0.0
            lambda_fg_structure_depth_edge_scale = 1.0
            lambda_fg_structure_depth_edge_eff = [double]$LambdaFgStructureDepthEdge
            point_mv_outside_ring_active = 0.0
            point_mv_outside_ring_active_views = 0.0
            point_mv_outside_ring_hit_ratio = 0.0
            point_mv_outside_ring_loss = 0.0
            point_mv_outside_ring_base_proj_ratio = 0.0
            point_mv_outside_ring_valid_ratio = 0.0
        }
        foreach ($kv in $supportMetricDefaults.GetEnumerator()) {
            if (-not $row.PSObject.Properties[$kv.Key]) {
                $row | Add-Member -NotePropertyName $kv.Key -NotePropertyValue $kv.Value -Force
            }
        }
    }
}

function Get-FtMetricsSnapshot([string]$MetricsPath) {
    if ([string]::IsNullOrWhiteSpace($MetricsPath) -or (-not (Test-Path $MetricsPath))) {
        return $null
    }
    try {
        $lines = Get-Content $MetricsPath | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        if ($lines.Count -le 0) { return $null }
        $events = New-Object System.Collections.Generic.List[object]
        foreach ($ln in $lines) {
            try {
                $events.Add(($ln | ConvertFrom-Json))
            } catch {}
        }
        if ($events.Count -le 0) { return $null }
        for ($i = $events.Count - 1; $i -ge 0; $i--) {
            $ev = $events[$i]
            try {
                $evName = [string]$ev.event
                if ($evName -eq "epoch_end" -or $evName -eq "step_eval") {
                    return $ev
                }
            } catch {}
        }
        return $events[$events.Count - 1]
    } catch {
        return $null
    }
}

function Merge-FtMetricsSnapshotIntoRow(
    [object]$Row,
    [object]$Snapshot
) {
    if (($null -eq $Row) -or ($null -eq $Snapshot)) { return }
    $names = @(
        "point_support_path_active",
        "point_mv_depth_support_path_active",
        "point_mv_mask_support_path_active",
        "point_target_blend_mv_support_active",
        "point_mv_mode",
        "point_mv_support_mean","point_mv_support_p10","point_mv_support_p90",
        "point_mv_support_fg_mean","point_mv_support_fg_p10","point_mv_support_fg_p90",
        "point_mv_support_bg_mean","point_mv_support_bg_p10","point_mv_support_bg_p90",
        "point_mv_pseudo_support_mean","point_mv_pseudo_support_p10","point_mv_pseudo_support_p90",
        "point_mv_pseudo_support_fg_mean","point_mv_pseudo_support_fg_p10","point_mv_pseudo_support_fg_p90",
        "point_mv_pseudo_support_bg_mean","point_mv_pseudo_support_bg_p10","point_mv_pseudo_support_bg_p90",
        "point_support_eff_mean","point_support_eff_p10","point_support_eff_p90",
        "point_support_eff_fg_mean","point_support_eff_fg_p10","point_support_eff_fg_p90",
        "point_support_eff_bg_mean","point_support_eff_bg_p10","point_support_eff_bg_p90",
        "point_mv_depth_support_eff_mean","point_mv_depth_support_eff_p10","point_mv_depth_support_eff_p90",
        "point_mv_depth_support_eff_fg_mean","point_mv_depth_support_eff_fg_p10","point_mv_depth_support_eff_fg_p90",
        "point_mv_depth_support_eff_bg_mean","point_mv_depth_support_eff_bg_p10","point_mv_depth_support_eff_bg_p90",
        "point_mv_mask_support_eff_mean","point_mv_mask_support_eff_p10","point_mv_mask_support_eff_p90",
        "point_mv_mask_support_eff_fg_mean","point_mv_mask_support_eff_fg_p10","point_mv_mask_support_eff_fg_p90",
        "point_mv_mask_support_eff_bg_mean","point_mv_mask_support_eff_bg_p10","point_mv_mask_support_eff_bg_p90",
        "fg_supervision_boost","fg_supervision_boost_applied",
        "fg_supervision_bg_floor",
        "fg_supervision_region_mode","fg_supervision_region_erode_px",
        "fg_supervision_boost_cover","fg_supervision_boost_cover_ratio_in_fg",
        "fg_supervision_boundary_ring_cover","fg_supervision_boundary_ring_ratio_in_fg",
        "fg_supervision_boost_mask_mean","fg_supervision_boost_mask_p10","fg_supervision_boost_mask_p90",
        "fg_supervision_boost_mask_fg_mean","fg_supervision_boost_mask_fg_p10","fg_supervision_boost_mask_fg_p90",
        "fg_supervision_boost_mask_bg_mean","fg_supervision_boost_mask_bg_p10","fg_supervision_boost_mask_bg_p90",
        "fg_supervision_boundary_ring_mean","fg_supervision_boundary_ring_p10","fg_supervision_boundary_ring_p90",
        "fg_supervision_boundary_ring_fg_mean","fg_supervision_boundary_ring_fg_p10","fg_supervision_boundary_ring_fg_p90",
        "fg_supervision_boundary_ring_bg_mean","fg_supervision_boundary_ring_bg_p10","fg_supervision_boundary_ring_bg_p90",
        "fg_supervision_profile_mean","fg_supervision_profile_p10","fg_supervision_profile_p90",
        "fg_supervision_profile_fg_mean","fg_supervision_profile_fg_p10","fg_supervision_profile_fg_p90",
        "fg_supervision_profile_bg_mean","fg_supervision_profile_bg_p10","fg_supervision_profile_bg_p90",
        "fg_supervision_weight_mean",
        "supervision_valid_cover","supervision_valid_fg_mean","supervision_valid_bg_mean",
        "fg_conf_presence_enabled","fg_conf_presence_target_ratio",
        "fg_conf_presence_pred_mean","fg_conf_presence_tgt_mean","fg_conf_presence_target_floor",
        "fg_conf_presence_active_ratio","fg_conf_presence_loss",
        "tf32","amp","strict_deterministic",
        "runner_tf32","runner_amp","runner_strict_deterministic",
        "precompute_tf32","precompute_amp","precompute_strict_deterministic",
        "teacher_tf32","teacher_amp","teacher_deterministic",
        "lambda_fg_structure_depth_edge","fg_structure_bbox_margin_px","fg_structure_bbox_min_side_px",
        "fg_structure_region_mode","fg_structure_region_erode_px","fg_structure_depth_edge_warmup_steps","fg_structure_boundary_probe_px",
        "fg_structure_edge_support_mode","fg_structure_edge_support_quantile","fg_structure_edge_support_min_px",
        "fg_structure_edge_weight_mode","fg_structure_boundary_falloff_px",
        "fg_structure_component_bias_mode","fg_structure_component_bias_threshold_ratio","fg_structure_component_bias_other_scale",
        "fg_structure_front_depth_bias_mode","fg_structure_front_depth_bias_tau","fg_structure_front_depth_bias_center_quantile",
        "lambda_point_mv_outside_ring","point_mv_outside_ring_px",
        "fg_structure_depth_edge_active",
        "fg_structure_bbox_cover","fg_structure_region_cover","fg_structure_effective_cover","fg_structure_boundary_probe_cover",
        "fg_structure_bbox_active_ratio","fg_structure_region_active_ratio","fg_structure_boundary_band_active_ratio",
        "fg_structure_depth_edge_active_views","fg_structure_depth_edge_boundary_active_views","fg_structure_depth_edge_loss",
        "fg_structure_depth_edge_loss_main","fg_structure_depth_edge_loss_boundary_probe",
        "fg_structure_depth_edge_loss_interior","fg_structure_depth_edge_loss_boundary_band",
        "fg_structure_depth_edge_pred_mean","fg_structure_depth_edge_tgt_mean",
        "fg_structure_depth_edge_boundary_probe_pred_mean","fg_structure_depth_edge_boundary_probe_tgt_mean",
        "fg_structure_depth_edge_boundary_pred_mean","fg_structure_depth_edge_boundary_tgt_mean",
        "fg_structure_target_edge_support_active","fg_structure_target_edge_support_views","fg_structure_target_edge_support_cover",
        "fg_structure_target_edge_support_region_share","fg_structure_target_edge_support_threshold_mean",
        "fg_structure_main_weight_mean","fg_structure_boundary_distance_weight_share",
        "fg_structure_front_depth_bias_weight_share","fg_structure_front_depth_bias_active_views",
        "main_support_component_count","main_support_largest_component_share","main_support_top2_component_share",
        "main_support_centroid_distance_mean","main_support_component_active_views","main_support_component_bias_weight_share",
        "main_support_depth_mode_count","main_support_back_mode_share","main_support_front_back_gap",
        "main_support_depth_hist_peak_ratio","main_support_secondary_risk","main_support_depth_mode_active_views",
        "lambda_fg_structure_depth_edge_scale","lambda_fg_structure_depth_edge_eff",
        "point_mv_outside_ring_active",
        "point_mv_outside_ring_active_views","point_mv_outside_ring_hit_ratio","point_mv_outside_ring_loss",
        "point_mv_outside_ring_base_proj_ratio","point_mv_outside_ring_valid_ratio",
        "loss_fg_structure_depth_edge","loss_contrib_fg_structure_depth_edge","mean_loss_fg_structure_depth_edge",
        "loss_point_mv_outside_ring","loss_contrib_point_mv_outside_ring","mean_loss_point_mv_outside_ring",
        "loss_fg_conf_presence","loss_contrib_fg_conf_presence","mean_loss_fg_conf_presence"
    )
    foreach ($name in $names) {
        try {
            if ($Snapshot.PSObject.Properties[$name]) {
                $val = $Snapshot.$name
                if (-not $Row.PSObject.Properties[$name]) {
                    $Row | Add-Member -NotePropertyName $name -NotePropertyValue $val -Force
                } else {
                    $Row.$name = $val
                }
            }
        } catch {}
    }
}

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $abs = Join-Path (Resolve-Path ".").Path $Path
    $dir = Split-Path -Parent $abs
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 20
    $tmp = "$abs.tmp.$PID.$([DateTime]::UtcNow.Ticks)"
    [System.IO.File]::WriteAllText($tmp, $json, $enc)
    try {
        if ([System.IO.File]::Exists($abs)) {
            [System.IO.File]::Replace($tmp, $abs, $null, $true)
        } else {
            [System.IO.File]::Move($tmp, $abs)
        }
    } catch {
        try {
            [System.IO.File]::Copy($tmp, $abs, $true)
        } finally {
            if ([System.IO.File]::Exists($tmp)) {
                [System.IO.File]::Delete($tmp)
            }
        }
    }
}

function Write-ModalRunHeartbeat(
    [string]$State,
    [string]$ScriptPath,
    [int]$Attempt,
    [int]$MaxRetries,
    [int]$TimeoutSec,
    [int]$ProcId,
    [datetime]$StartedAt,
    [string]$StdoutFile,
    [string]$StderrFile,
    [int]$ExitCode = [int]::MinValue,
    [string]$Note = "",
    [object]$StructuredHeartbeat = $null
) {
    $now = Get-Date
    $elapsed = [double]::NaN
    if ($StartedAt -ne [datetime]::MinValue) {
        $elapsed = ($now - $StartedAt).TotalSeconds
    }
    $stdoutLen = 0
    $stderrLen = 0
    if (-not [string]::IsNullOrWhiteSpace($StdoutFile) -and (Test-Path $StdoutFile)) {
        try { $stdoutLen = [int64](Get-Item $StdoutFile).Length } catch {}
    }
    if (-not [string]::IsNullOrWhiteSpace($StderrFile) -and (Test-Path $StderrFile)) {
        try { $stderrLen = [int64](Get-Item $StderrFile).Length } catch {}
    }
    $procExists = $false
    if ($ProcId -gt 0) {
        try {
            $null = Get-Process -Id $ProcId -ErrorAction Stop
            $procExists = $true
        } catch {
            $procExists = $false
        }
    }

    $obj = [ordered]@{
        updated_at = $now.ToString("yyyy-MM-ddTHH:mm:ss")
        state = $State
        script_path = $ScriptPath
        attempt = $Attempt
        max_retries = $MaxRetries
        no_output_timeout_sec = [Math]::Max(0, [int]$ModalRunNoOutputTimeoutSec)
        no_output_max_retries = [Math]::Max(0, [int]$ModalRunNoOutputMaxRetries)
        timeout_sec = $TimeoutSec
        pid = $ProcId
        started_at = $(if ($StartedAt -eq [datetime]::MinValue) { "" } else { $StartedAt.ToString("yyyy-MM-ddTHH:mm:ss") })
        elapsed_sec = $elapsed
        stdout_file = $StdoutFile
        stderr_file = $StderrFile
        stdout_bytes = $stdoutLen
        stderr_bytes = $stderrLen
        proc_exists = $procExists
        writer_pid = $PID
        note = $Note
    }
    if ($null -ne $StructuredHeartbeat) {
        $obj.structured_heartbeat_source = [string]$StructuredHeartbeat.source
        $obj.structured_heartbeat_phase = [string]$StructuredHeartbeat.phase
        $obj.structured_heartbeat_progress_counter = $(if ($StructuredHeartbeat.PSObject.Properties["progress_counter"]) { $StructuredHeartbeat.progress_counter } else { $null })
        $obj.structured_heartbeat_items_done = $(if ($StructuredHeartbeat.PSObject.Properties["items_done"]) { $StructuredHeartbeat.items_done } else { $null })
        $obj.structured_heartbeat_items_total = $(if ($StructuredHeartbeat.PSObject.Properties["items_total"]) { $StructuredHeartbeat.items_total } else { $null })
        $obj.structured_heartbeat_scene_id = $(if ($StructuredHeartbeat.PSObject.Properties["scene_id"]) { [string]$StructuredHeartbeat.scene_id } else { "" })
        $obj.structured_heartbeat_batch_idx = $(if ($StructuredHeartbeat.PSObject.Properties["batch_idx"]) { $StructuredHeartbeat.batch_idx } else { $null })
        $obj.structured_heartbeat_frame_start = $(if ($StructuredHeartbeat.PSObject.Properties["frame_start"]) { $StructuredHeartbeat.frame_start } else { $null })
        $obj.structured_heartbeat_frame_end = $(if ($StructuredHeartbeat.PSObject.Properties["frame_end"]) { $StructuredHeartbeat.frame_end } else { $null })
        $obj.structured_heartbeat_time = $(if ($StructuredHeartbeat.PSObject.Properties["heartbeat_time"]) { [string]$StructuredHeartbeat.heartbeat_time } else { "" })
    }
    if ($ExitCode -ne [int]::MinValue) {
        $obj.exit_code = $ExitCode
    }
    Write-JsonNoBom -Path $ModalRunHeartbeatPath -Obj $obj
}

function Get-StructuredHeartbeatSignature([object]$Heartbeat) {
    if ($null -eq $Heartbeat) { return "" }
    $parts = @(
        [string]$Heartbeat.source,
        [string]$Heartbeat.phase,
        [string]$Heartbeat.progress_counter,
        [string]$Heartbeat.items_done,
        [string]$Heartbeat.items_total,
        [string]$Heartbeat.batch_idx,
        [string]$Heartbeat.frame_start,
        [string]$Heartbeat.frame_end,
        [string]$Heartbeat.heartbeat_time
    )
    return ($parts -join "|")
}

function Get-LatestStructuredHeartbeat([string]$StdoutFile) {
    if ([string]::IsNullOrWhiteSpace($StdoutFile) -or (-not (Test-Path $StdoutFile))) {
        return $null
    }
    $lines = @()
    try {
        $lines = @(Get-Content $StdoutFile -Tail 240 -Encoding UTF8)
    } catch {
        return $null
    }
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        $line = [string]$lines[$i]
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $mPre = [regex]::Match($line, '^\[precompute-heartbeat\]\s+(\{.*\})\s*$')
        if ($mPre.Success) {
            try {
                $obj = $mPre.Groups[1].Value | ConvertFrom-Json
                return [pscustomobject]@{
                    source = "precompute"
                    phase = [string]$obj.phase
                    progress_counter = $(if ($obj.PSObject.Properties["progress_counter"]) { [int]$obj.progress_counter } else { $null })
                    items_done = $(if ($obj.PSObject.Properties["items_done"]) { $obj.items_done } else { $null })
                    items_total = $(if ($obj.PSObject.Properties["items_total"]) { $obj.items_total } else { $null })
                    scene_id = $(if ($obj.PSObject.Properties["scene_id"]) { [string]$obj.scene_id } else { "" })
                    batch_idx = $(if ($obj.PSObject.Properties["batch_idx"]) { $obj.batch_idx } else { $null })
                    frame_start = $(if ($obj.PSObject.Properties["frame_start"]) { $obj.frame_start } else { $null })
                    frame_end = $(if ($obj.PSObject.Properties["frame_end"]) { $obj.frame_end } else { $null })
                    heartbeat_time = $(if ($obj.PSObject.Properties["time"]) { [string]$obj.time } else { "" })
                }
            } catch {}
        }
        $mFtJson = [regex]::Match($line, '^\[finetune\]\s+(\{.*\})\s*$')
        if ($mFtJson.Success) {
            $ftBlob = [string]$mFtJson.Groups[1].Value
            $ftObj = $null
            try {
                $ftObj = $ftBlob | ConvertFrom-Json
            } catch {
                try {
                    $ftNorm = ($ftBlob -replace "'", '"')
                    $ftObj = $ftNorm | ConvertFrom-Json
                } catch {}
            }
            if ($ftObj -ne $null) {
                $ftEvent = ""
                if ($ftObj.PSObject.Properties["event"]) {
                    $ftEvent = [string]$ftObj.event
                }
                if ($ftEvent -in @("step_heartbeat", "step_eval")) {
                    return [pscustomobject]@{
                        source = "finetune"
                        phase = $ftEvent
                        progress_counter = $(if ($ftObj.PSObject.Properties["step"]) { [int]$ftObj.step } else { $null })
                        items_done = $null
                        items_total = $null
                        scene_id = ""
                        batch_idx = $null
                        frame_start = $null
                        frame_end = $null
                        heartbeat_time = ""
                    }
                }
            }
        }
        $mRemote = [regex]::Match($line, '^\[remote\]\s+\[alive\]\s+child_pid=(\d+)\s+elapsed=([0-9.]+)s\s+idle=([0-9.]+)s\b')
        if ($mRemote.Success) {
            return [pscustomobject]@{
                source = "remote_alive"
                phase = "remote_alive"
                progress_counter = [int][Math]::Floor([double]$mRemote.Groups[2].Value)
                items_done = $null
                items_total = $null
                scene_id = ""
                batch_idx = $null
                frame_start = $null
                frame_end = $null
                heartbeat_time = ""
            }
        }
    }
    return $null
}

function Parse-StepList([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) {
        return @()
    }
    $out = New-Object System.Collections.Generic.List[int]
    foreach ($tok in ($Raw -split "[,\s;|]+" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
        $v = 0
        if ([int]::TryParse($tok.Trim(), [ref]$v)) {
            if ($v -ge 0) {
                $out.Add($v) | Out-Null
            }
        }
    }
    return @($out | Select-Object -Unique)
}

function Get-StepFromFilename([string]$Filename) {
    $m = [regex]::Match([string]$Filename, "step(\d+)")
    if ($m.Success) {
        return [int]$m.Groups[1].Value
    }
    return -1
}

function Get-VolumeFileNames([object[]]$Items) {
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($it in @($Items)) {
        if ($null -eq $it) { continue }
        $obj = $it
        $type = ""
        try {
            if ($obj.PSObject.Properties["Type"]) {
                $type = [string]$obj.Type
            }
        } catch {}
        if (-not [string]::IsNullOrWhiteSpace($type) -and ($type -ne "file")) {
            continue
        }

        $name = ""
        foreach ($k in @("Filename", "filename", "Path", "path", "Name", "name")) {
            try {
                if ($obj.PSObject.Properties[$k]) {
                    $name = [string]$obj.$k
                    if (-not [string]::IsNullOrWhiteSpace($name)) { break }
                }
            } catch {}
        }
        if ([string]::IsNullOrWhiteSpace($name)) {
            if ($obj -is [string]) {
                $name = [string]$obj
            }
        }
        if (-not [string]::IsNullOrWhiteSpace($name)) {
            $out.Add($name) | Out-Null
        }
    }
    return @($out | Select-Object -Unique)
}

function Resolve-VolumeRemoteFilePath(
    [string]$RemoteDir,
    [string]$ListedPath
) {
    $remoteDirNorm = ([string]$RemoteDir).Trim().Replace("\", "/")
    $listedPathNorm = ([string]$ListedPath).Trim().Replace("\", "/")
    if ([string]::IsNullOrWhiteSpace($listedPathNorm)) { return "" }
    if ($listedPathNorm.StartsWith("/")) { return $listedPathNorm }
    $remoteDirNorm = $remoteDirNorm.TrimEnd("/")
    if ([string]::IsNullOrWhiteSpace($remoteDirNorm)) {
        return "/" + $listedPathNorm.TrimStart("/")
    }
    $remoteDirRel = $remoteDirNorm.TrimStart("/")
    if ($listedPathNorm -eq $remoteDirRel) {
        return "/" + $listedPathNorm
    }
    if ($listedPathNorm.StartsWith($remoteDirRel + "/")) {
        return "/" + $listedPathNorm
    }
    return ($remoteDirNorm + "/" + $listedPathNorm.TrimStart("/"))
}

function New-LocalScratchFilePath(
    [string]$Prefix,
    [string]$Extension
) {
    $repoRoot = ""
    try {
        $repoRoot = [string](Resolve-Path ".").Path
    } catch {
        $repoRoot = [string]$CodeDir
    }
    if ([string]::IsNullOrWhiteSpace($repoRoot)) {
        $repoRoot = [string]$PWD.Path
    }
    $scratchRoot = Join-Path $repoRoot "logs/modal_phase5/_tmp_precompute_support_stats"
    New-Item -ItemType Directory -Force -Path $scratchRoot | Out-Null
    $safePrefix = Sanitize([string]$Prefix)
    $safeExt = [string]$Extension
    if (-not $safeExt.StartsWith(".")) {
        $safeExt = "." + $safeExt.TrimStart(".")
    }
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss_fffffff")
    return (Join-Path $scratchRoot "${safePrefix}_${stamp}_$PID$safeExt")
}

function Parse-VolumeLsItems([string]$ItemsJson) {
    $parsed = $null
    try {
        $parsed = $ItemsJson | ConvertFrom-Json
    } catch {
        return @()
    }
    $flat = New-Object System.Collections.Generic.List[object]
    $queue = New-Object System.Collections.Generic.Queue[object]
    foreach ($x in @($parsed)) {
        $queue.Enqueue($x)
    }
    while ($queue.Count -gt 0) {
        $cur = $queue.Dequeue()
        if ($null -eq $cur) { continue }
        if ($cur -is [System.Array]) {
            foreach ($y in @($cur)) {
                $queue.Enqueue($y)
            }
            continue
        }
        $flat.Add($cur) | Out-Null
    }
    return @($flat.ToArray())
}

function Get-PrecomputeGeomIntegrity(
    [string]$SeqNamesRaw,
    [string]$GeomSubdir
) {
    $seqTokens = @(
        $SeqNamesRaw -split "[,\s;|]+" |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_.Trim() }
    )
    if ($seqTokens.Count -le 0 -or [string]::IsNullOrWhiteSpace([string]$GeomSubdir)) {
        return [pscustomobject]@{
            ok = $false
            total_npz = 0
            missing_seqs = ""
            reason = "precompute_geom_inspect_failed(invalid_args)"
            candidate_invalid_reason = "precompute_failed"
        }
    }

    $missingSeqs = New-Object System.Collections.Generic.List[string]
    $inspectErrors = New-Object System.Collections.Generic.List[string]
    $totalNpz = 0
    foreach ($seq in $seqTokens) {
        $remoteDir = "/zju_mocap/$seq/$GeomSubdir"
        try {
            $itemsJson = modal volume ls --json vggt-zju-data $remoteDir 2>$null | Out-String
            $items = @(Parse-VolumeLsItems -ItemsJson $itemsJson)
            $files = @(Get-VolumeFileNames -Items $items)
            $npzFiles = @(
                $files |
                    Where-Object {
                        $name = [string]$_
                        ($name -match "(?i)(^|/)frame_\d+\.npz$") -or ($name -match "(?i)\.npz$")
                    }
            )
            $npzCount = $npzFiles.Count
            if ($npzCount -le 0) {
                $missingSeqs.Add($seq) | Out-Null
            } else {
                $totalNpz += [int]$npzCount
            }
        } catch {
            $inspectErrors.Add($seq) | Out-Null
        }
    }

    if ($inspectErrors.Count -gt 0) {
        return [pscustomobject]@{
            ok = $false
            total_npz = [int]$totalNpz
            missing_seqs = ($inspectErrors -join ",")
            reason = "precompute_geom_inspect_failed(seqs=$($inspectErrors -join ','))"
            candidate_invalid_reason = "precompute_failed"
        }
    }
    if ($missingSeqs.Count -gt 0 -or $totalNpz -le 0) {
        $missingNote = if ($missingSeqs.Count -gt 0) { $missingSeqs -join "," } else { "all" }
        return [pscustomobject]@{
            ok = $false
            total_npz = [int]$totalNpz
            missing_seqs = $missingNote
            reason = "precompute_empty(missing_seq_npz=$missingNote total_npz=$totalNpz geom_subdir=$GeomSubdir)"
            candidate_invalid_reason = "precompute_empty"
        }
    }
    return [pscustomobject]@{
        ok = $true
        total_npz = [int]$totalNpz
        missing_seqs = ""
        reason = ""
        candidate_invalid_reason = ""
    }
}

function Get-PrecomputeSupportStats(
    [string]$SeqNamesRaw,
    [string]$GeomSubdir
) {
    $result = [ordered]@{
        mv_support_raw_mean = [double]::NaN
        mv_support_valid_ratio = [double]::NaN
        mv_support_fg_valid_ratio = [double]::NaN
        mv_support_bg_valid_ratio = [double]::NaN
        mv_support_pair_count_eff = [double]::NaN
        mv_support_conf_mean = [double]::NaN
        mv_support_nan_ratio = [double]::NaN
        depth_conf_delta_mean = [double]::NaN
        mv_support_fg_mean = [double]::NaN
        mv_support_bg_mean = [double]::NaN
        depth_conf_delta_fg_mean = [double]::NaN
        depth_conf_delta_bg_mean = [double]::NaN
        depth_conf_fg_preserved_active = [double]::NaN
        depth_conf_fg_preserve_px = [double]::NaN
        depth_conf_fg_exact_ratio = [double]::NaN
        depth_conf_fg_preserve_ratio = [double]::NaN
        depth_conf_fg_raw_mean = [double]::NaN
        depth_conf_fg_after_support_mean = [double]::NaN
        depth_conf_fg_final_mean = [double]::NaN
        mv_support_generation_region_mode = ""
        mv_support_generation_fg_mask_source = ""
    }
    $helperScript = Join-Path $PSScriptRoot "fetch_precompute_support_stats.py"
    if ((Test-Path $helperScript) -and -not [string]::IsNullOrWhiteSpace([string]$GeomSubdir)) {
        try {
            $jsonText = @(
                & python $helperScript `
                    --seq-names $SeqNamesRaw `
                    --geom-subdir $GeomSubdir `
                    --volume-name "vggt-zju-data" `
                    --remote-root "/zju_mocap" 2>$null
            ) -join ""
            if (-not [string]::IsNullOrWhiteSpace($jsonText)) {
                $obj = $jsonText | ConvertFrom-Json
                foreach ($prop in @($obj.PSObject.Properties)) {
                    $result[$prop.Name] = $prop.Value
                }
                $nonEmptyKeys = @($obj.PSObject.Properties | ForEach-Object { $_.Name })
                if ($nonEmptyKeys.Count -gt 0) {
                    return [pscustomobject]$result
                }
            }
        } catch {
            Write-Host "[lr-sweep] support-stats helper failed geom_subdir=$GeomSubdir error=$($_.Exception.Message)"
        }
    }
    $seqTokens = @(
        $SeqNamesRaw -split "[,\s;|]+" |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_.Trim() }
    )
    foreach ($seq in $seqTokens) {
        if ([string]::IsNullOrWhiteSpace($seq)) { continue }
        $remoteDir = "/zju_mocap/$seq/$GeomSubdir"
        try {
            $itemsJson = modal volume ls --json vggt-zju-data $remoteDir 2>$null | Out-String
            $items = @(Parse-VolumeLsItems -ItemsJson $itemsJson)
            $files = @(Get-VolumeFileNames -Items $items)
            $statsRel = @(
                $files |
                    Where-Object {
                        $name = [string]$_
                        $name -match "(?i)(^|/)frame_\\d+\\.support_stats\\.json$"
                    } |
                    Select-Object -First 1
            )
            if ($statsRel.Count -gt 0) {
                $remoteStatsFile = Resolve-VolumeRemoteFilePath -RemoteDir $remoteDir -ListedPath ([string]$statsRel[0])
                $tmpJson = New-LocalScratchFilePath -Prefix "support_stats_json_$seq" -Extension ".json"
                try {
                    if (Test-Path $tmpJson) { Remove-Item $tmpJson -Force -ErrorAction SilentlyContinue }
                    modal volume get vggt-zju-data $remoteStatsFile $tmpJson 2>$null | Out-Null
                    if (Test-Path $tmpJson) {
                        $obj = Get-Content $tmpJson -Raw -Encoding UTF8 | ConvertFrom-Json
                        foreach ($prop in @($obj.PSObject.Properties)) {
                            $result[$prop.Name] = $prop.Value
                        }
                        return [pscustomobject]$result
                    }
                    Write-Host "[lr-sweep] support-stats sidecar not downloaded seq=$seq remote=$remoteStatsFile"
                } catch {
                    Write-Host "[lr-sweep] support-stats sidecar read failed seq=$seq remote=$remoteStatsFile error=$($_.Exception.Message)"
                } finally {
                    if (Test-Path $tmpJson) { Remove-Item $tmpJson -Force -ErrorAction SilentlyContinue }
                }
            }
            $npzRel = @(
                $files |
                    Where-Object {
                        $name = [string]$_
                        ($name -match "(?i)(^|/)frame_\\d+\\.npz$") -or ($name -match "(?i)\\.npz$")
                    } |
                    Select-Object -First 1
            )
            if ($npzRel.Count -le 0) { continue }
            $remoteFile = Resolve-VolumeRemoteFilePath -RemoteDir $remoteDir -ListedPath ([string]$npzRel[0])
            $tmpNp = New-LocalScratchFilePath -Prefix "support_stats_npz_$seq" -Extension ".npz"
            try {
                if (Test-Path $tmpNp) { Remove-Item $tmpNp -Force -ErrorAction SilentlyContinue }
                modal volume get vggt-zju-data $remoteFile $tmpNp 2>$null | Out-Null
                if (-not (Test-Path $tmpNp)) {
                    Write-Host "[lr-sweep] support-stats npz fallback not downloaded seq=$seq remote=$remoteFile"
                    continue
                }
                $py = @'
import json, math, numpy as np, sys
path = sys.argv[1]
keys = [
    "mv_support_raw_mean",
    "mv_support_valid_ratio",
    "mv_support_fg_valid_ratio",
    "mv_support_bg_valid_ratio",
    "mv_support_pair_count_eff",
    "mv_support_conf_mean",
    "mv_support_nan_ratio",
    "depth_conf_delta_mean",
    "mv_support_fg_mean",
    "mv_support_bg_mean",
    "depth_conf_delta_fg_mean",
    "depth_conf_delta_bg_mean",
    "depth_conf_fg_preserved_active",
    "depth_conf_fg_preserve_px",
    "depth_conf_fg_exact_ratio",
    "depth_conf_fg_preserve_ratio",
    "depth_conf_fg_raw_mean",
    "depth_conf_fg_after_support_mean",
    "depth_conf_fg_final_mean",
    "mv_support_generation_region_mode",
    "mv_support_generation_fg_mask_source",
]
z = np.load(path, allow_pickle=True)
out = {}
def scalarize(v):
    arr = np.asarray(v).reshape(-1)
    if arr.size <= 0:
        return None
    item = arr[0]
    if np.issubdtype(arr.dtype, np.number):
        num = float(item)
        if math.isnan(num) or math.isinf(num):
            return None
        return num
    return str(item)
for k in keys:
    if k not in z:
        continue
    try:
        value = scalarize(z[k])
        if value is None:
            continue
        out[k] = value
    except Exception:
        pass
print(json.dumps(out))
'@
                $jsonText = @($py | python - $tmpNp) -join ""
                if (-not [string]::IsNullOrWhiteSpace($jsonText)) {
                    $obj = $jsonText | ConvertFrom-Json
                    foreach ($prop in @($obj.PSObject.Properties)) {
                        $result[$prop.Name] = $prop.Value
                    }
                    return [pscustomobject]$result
                }
                Write-Host "[lr-sweep] support-stats npz fallback empty seq=$seq remote=$remoteFile"
            } catch {
                Write-Host "[lr-sweep] support-stats npz fallback failed seq=$seq remote=$remoteFile error=$($_.Exception.Message)"
            } finally {
                if (Test-Path $tmpNp) { Remove-Item $tmpNp -Force -ErrorAction SilentlyContinue }
            }
        } catch {}
    }
    return [pscustomobject]$result
}

function To-VolumePath([string]$PathInMntOut) {
    $s = [string]$PathInMntOut
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

function Resolve-VolumeRemotePath(
    [string]$EntryPath,
    [string]$BaseDir
) {
    $e = ([string]$EntryPath).Replace("\", "/")
    if ([string]::IsNullOrWhiteSpace($e)) { return $null }
    if ($e.StartsWith("/mnt/out/")) {
        return To-VolumePath -PathInMntOut $e
    }
    if ($e.StartsWith("mnt/out/")) {
        return To-VolumePath -PathInMntOut $e
    }
    if ($e.StartsWith("/")) {
        return $e
    }
    $b = To-VolumePath -PathInMntOut $BaseDir
    if ([string]::IsNullOrWhiteSpace($b)) { $b = "/" }
    if ($b -eq "/") {
        return "/" + $e.TrimStart("/")
    }
    return $b.TrimEnd("/") + "/" + $e.TrimStart("/")
}

function Test-ModalVolumeFile([string]$PathInMntOut) {
    $vp = To-VolumePath -PathInMntOut $PathInMntOut
    $norm = ([string]$vp).Replace("\", "/")
    if ($norm.EndsWith("/")) { $norm = $norm.TrimEnd("/") }
    $slash = $norm.LastIndexOf("/")
    $dir = "/"
    $name = $norm.Trim("/")
    if ($slash -gt 0) {
        $dir = $norm.Substring(0, $slash)
        $name = $norm.Substring($slash + 1)
    }
    if ([string]::IsNullOrWhiteSpace($dir)) { $dir = "/" }
    if ([string]::IsNullOrWhiteSpace($name)) { return $false }
    try {
        $itemsJson = modal volume ls --json vggt-out $dir 2>$null | Out-String
        $items = @(Parse-VolumeLsItems -ItemsJson $itemsJson)
        $files = @(Get-VolumeFileNames -Items $items)
        $hit = @($files | Where-Object { [string]$_ -like "*$name" })
        return ($hit.Count -gt 0)
    } catch {
        return $false
    }
}

function Convert-VolumeSizeToBytes([object]$Value) {
    if ($null -eq $Value) { return -1L }
    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) { return -1L }
    $direct = 0L
    if ([int64]::TryParse($text, [ref]$direct)) { return [int64]$direct }
    $m = [regex]::Match($text, '^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP]?i?B)\s*$')
    if (-not $m.Success) { return -1L }
    $valueNum = 0.0
    if (-not [double]::TryParse($m.Groups[1].Value, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$valueNum)) {
        return -1L
    }
    $unit = $m.Groups[2].Value
    $factor = switch -Regex ($unit) {
        '^B$' { 1.0; break }
        '^KiB$' { 1024.0; break }
        '^MiB$' { 1024.0 * 1024.0; break }
        '^GiB$' { 1024.0 * 1024.0 * 1024.0; break }
        '^TiB$' { 1024.0 * 1024.0 * 1024.0 * 1024.0; break }
        '^PiB$' { 1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0; break }
        default { -1.0 }
    }
    if ($factor -lt 0) { return -1L }
    return [int64][Math]::Round($valueNum * $factor)
}

function Get-ModalVolumeFileInfo([string]$PathInMntOut) {
    $vp = To-VolumePath -PathInMntOut $PathInMntOut
    $norm = ([string]$vp).Replace("\", "/")
    if ($norm.EndsWith("/")) { $norm = $norm.TrimEnd("/") }
    $slash = $norm.LastIndexOf("/")
    $dir = "/"
    $name = $norm.Trim("/")
    if ($slash -gt 0) {
        $dir = $norm.Substring(0, $slash)
        $name = $norm.Substring($slash + 1)
    }
    if ([string]::IsNullOrWhiteSpace($dir)) { $dir = "/" }
    if ([string]::IsNullOrWhiteSpace($name)) {
        return [pscustomobject]@{ exists = $false; size_bytes = -1L; size_text = ""; path = $PathInMntOut }
    }
    try {
        $itemsJson = modal volume ls --json vggt-out $dir 2>$null | Out-String
        $items = @(Parse-VolumeLsItems -ItemsJson $itemsJson)
        foreach ($obj in @($items)) {
            if ($null -eq $obj) { continue }
            $type = ""
            try { if ($obj.PSObject.Properties["Type"]) { $type = [string]$obj.Type } } catch {}
            if (-not [string]::IsNullOrWhiteSpace($type) -and ($type -ne "file")) { continue }
            $cand = ""
            foreach ($k in @("Filename", "filename", "Path", "path", "Name", "name")) {
                try {
                    if ($obj.PSObject.Properties[$k]) {
                        $cand = [string]$obj.$k
                        if (-not [string]::IsNullOrWhiteSpace($cand)) { break }
                    }
                } catch {}
            }
            if ([string]::IsNullOrWhiteSpace($cand)) { continue }
            $candName = [System.IO.Path]::GetFileName(([string]$cand).Replace("\","/"))
            if ($candName -ne $name) { continue }
            $sizeText = ""
            foreach ($k in @("Size", "size")) {
                try {
                    if ($obj.PSObject.Properties[$k]) {
                        $sizeText = [string]$obj.$k
                        if (-not [string]::IsNullOrWhiteSpace($sizeText)) { break }
                    }
                } catch {}
            }
            return [pscustomobject]@{
                exists = $true
                size_bytes = (Convert-VolumeSizeToBytes -Value $sizeText)
                size_text = $sizeText
                path = $PathInMntOut
            }
        }
    } catch {}
    return [pscustomobject]@{ exists = $false; size_bytes = -1L; size_text = ""; path = $PathInMntOut }
}

function Wait-ModalVolumeFileStable(
    [string]$PathInMntOut,
    [int]$TimeoutSec = 120,
    [int]$PollSec = 10,
    [int]$StablePollsRequired = 2
) {
    $deadline = (Get-Date).AddSeconds([Math]::Max(15, [int]$TimeoutSec))
    $poll = [Math]::Max(3, [int]$PollSec)
    $required = [Math]::Max(2, [int]$StablePollsRequired)
    $lastSize = -1L
    $stableCount = 0
    while ((Get-Date) -lt $deadline) {
        $info = Get-ModalVolumeFileInfo -PathInMntOut $PathInMntOut
        if ($info.exists -and ($info.size_bytes -gt 0)) {
            if ($info.size_bytes -eq $lastSize) {
                $stableCount += 1
            } else {
                $stableCount = 1
                $lastSize = [int64]$info.size_bytes
            }
            if ($stableCount -ge $required) {
                Write-Host "[lr-sweep] checkpoint stable on volume: $PathInMntOut size=$($info.size_text)"
                return $true
            }
        }
        Start-Sleep -Seconds $poll
    }
    Write-Host "[lr-sweep] checkpoint stability wait timeout on volume: $PathInMntOut"
    return $false
}

function Wait-ModalVolumeFile(
    [string]$PathInMntOut,
    [int]$TimeoutSec = 300,
    [int]$PollSec = 15
) {
    $deadline = (Get-Date).AddSeconds([Math]::Max(15, [int]$TimeoutSec))
    $started = Get-Date
    $poll = [Math]::Max(3, [int]$PollSec)
    $iter = 0
    while ((Get-Date) -lt $deadline) {
        $iter += 1
        if (Test-ModalVolumeFile -PathInMntOut $PathInMntOut) {
            $elapsedOk = [int][Math]::Round(((Get-Date) - $started).TotalSeconds)
            Write-Host "[lr-sweep] checkpoint available on volume: $PathInMntOut (wait=${elapsedOk}s)"
            return $true
        }
        if (($iter -eq 1) -or (($iter % 3) -eq 0)) {
            $elapsed = [int][Math]::Round(((Get-Date) - $started).TotalSeconds)
            Write-Host "[lr-sweep] waiting checkpoint on volume: $PathInMntOut (elapsed=${elapsed}s timeout=${TimeoutSec}s poll=${poll}s)"
        }
        Start-Sleep -Seconds $poll
    }
    Write-Host "[lr-sweep] checkpoint wait timeout on volume: $PathInMntOut (timeout=${TimeoutSec}s)"
    return $false
}

function Resolve-ModalCheckpointPath(
    [string]$PrimaryPathInMntOut,
    [int]$TimeoutSec = 300,
    [int]$PollSec = 15,
    [switch]$AllowTmpFallback
) {
    $script:LastCheckpointResolveReason = "missing"
    if (Wait-ModalVolumeFile -PathInMntOut $PrimaryPathInMntOut -TimeoutSec $TimeoutSec -PollSec $PollSec) {
        $stableOk = Wait-ModalVolumeFileStable -PathInMntOut $PrimaryPathInMntOut -TimeoutSec ([Math]::Min([Math]::Max(30, [int]$PollSec * 4), 180)) -PollSec $PollSec
        if ($stableOk) {
            $script:LastCheckpointResolveReason = "primary"
            return $PrimaryPathInMntOut
        }
        Write-Host "[lr-sweep] checkpoint unstable after availability: $PrimaryPathInMntOut"
    }
    if ($AllowTmpFallback) {
        $tmpPath = "$PrimaryPathInMntOut.tmp"
        if (Test-ModalVolumeFile -PathInMntOut $tmpPath) {
            $tmpStableOk = Wait-ModalVolumeFileStable -PathInMntOut $tmpPath -TimeoutSec ([Math]::Min([Math]::Max(30, [int]$PollSec * 4), 180)) -PollSec $PollSec
            if (-not $tmpStableOk) {
                Write-Host "[lr-sweep] checkpoint tmp unstable after availability: $tmpPath"
                $script:LastCheckpointResolveReason = "tmp_unstable"
                return $null
            }
            $promoted = $false
            try {
                $srcVp = To-VolumePath -PathInMntOut $tmpPath
                $dstVp = To-VolumePath -PathInMntOut $PrimaryPathInMntOut
                modal volume cp vggt-out $srcVp $dstVp 2>$null | Out-Null
                if (Test-ModalVolumeFile -PathInMntOut $PrimaryPathInMntOut) {
                    $promoted = $true
                }
            } catch {
                $promoted = $false
            }
            if ($promoted) {
                Write-Host "[lr-sweep] checkpoint tmp promoted: $tmpPath -> $PrimaryPathInMntOut"
                $script:LastCheckpointResolveReason = "tmp_promoted"
                return $PrimaryPathInMntOut
            }
            Write-Host "[lr-sweep] checkpoint fallback: use tmp file $tmpPath"
            $script:LastCheckpointResolveReason = "tmp"
            return $tmpPath
        }
    }
    $script:LastCheckpointResolveReason = "timeout_missing"
    return $null
}

function Test-NoSpaceError([string[]]$Lines) {
    if ($null -eq $Lines) { return $false }
    $blob = @($Lines) -join "`n"
    if ([string]::IsNullOrWhiteSpace($blob)) { return $false }
    return ([regex]::IsMatch($blob, "(?i)(Errno\s*28|No space left on device)"))
}

function Resolve-ModalRunFailureReason(
    [object]$RunResult,
    [string[]]$Lines,
    [string]$DefaultReason
) {
    $reason = [string]$DefaultReason
    $note = ""
    try {
        if ($null -ne $RunResult -and $RunResult.PSObject.Properties["TimeoutNote"]) {
            $note = [string]$RunResult.TimeoutNote
        }
    } catch {}
    $blob = @($Lines) -join "`n"

    if (-not [string]::IsNullOrWhiteSpace($note)) {
        if (($note -like "no_output_timeout_*") -or ($note -like "heartbeat_stall_timeout_*")) {
            return "${reason}: $note"
        }
        if ($note -ne "timeout") {
            return "${reason}: $note"
        }
    }
    if ($blob -match "(?i)(heartbeat_stall_timeout|no_output_timeout)_(\d+)s") {
        return "${reason}: $($Matches[1])_$($Matches[2])s"
    }
    if ($blob -match "(?i)ckpt_handoff_failed|failed finding central directory|pytorchstreamreader failed reading zip archive|zip_probe_failed|checkpoint unreadable|invalid checkpoint zip") {
        return "${reason}: ckpt_handoff_failed"
    }
    if ($blob -match "(?i)\[modal-run\]\s+timeout") {
        return "${reason}: timeout"
    }
    return $reason
}

function Get-RunDirFromMntOutPath([string]$PathInMntOut) {
    if ([string]::IsNullOrWhiteSpace($PathInMntOut)) { return $null }
    $vp = (To-VolumePath -PathInMntOut $PathInMntOut).Replace("\", "/")
    $m = [regex]::Match($vp, "^(/vggt/finetune/[^/]+)")
    if ($m.Success) {
        return $m.Groups[1].Value
    }
    return $null
}

function Get-PathTimestampUtc([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return [datetime]::MinValue }
    $leaf = [System.IO.Path]::GetFileName($Path.TrimEnd("/"))
    $m = [regex]::Match($leaf, "(\d{8}_\d{6})$")
    if (-not $m.Success) {
        $m = [regex]::Match($leaf, "(\d{8}_\d{6})")
    }
    if (-not $m.Success) { return [datetime]::MinValue }
    $dt = [datetime]::MinValue
    if ([datetime]::TryParseExact($m.Groups[1].Value, "yyyyMMdd_HHmmss", [System.Globalization.CultureInfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::AssumeUniversal, [ref]$dt)) {
        return $dt
    }
    return [datetime]::MinValue
}

function Invoke-NoSpaceCleanup(
    [string]$RootDir,
    [string[]]$PinnedRunDirs,
    [int]$KeepRecentDirs = 220,
    [int]$DeleteLimit = 120
) {
    $root = (To-VolumePath -PathInMntOut $RootDir).Replace("\", "/").TrimEnd("/")
    $keepN = [Math]::Max(1, [int]$KeepRecentDirs)
    $deleteN = [Math]::Max(1, [int]$DeleteLimit)

    $pins = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($p in @($PinnedRunDirs)) {
        if ([string]::IsNullOrWhiteSpace($p)) { continue }
        $norm = (To-VolumePath -PathInMntOut $p).Replace("\", "/").TrimEnd("/")
        if (-not [string]::IsNullOrWhiteSpace($norm)) {
            [void]$pins.Add($norm)
        }
    }

    $itemsJson = ""
    try {
        $itemsJson = modal volume ls --json vggt-out $root 2>$null | Out-String
    } catch {
        return [pscustomobject]@{
            root = $root
            listed = 0
            deleted = 0
            keep_recent = $keepN
            delete_limit = $deleteN
            reason = "list_failed"
            deleted_paths = @()
        }
    }

    $items = @(Parse-VolumeLsItems -ItemsJson $itemsJson)
    $dirs = @()
    foreach ($it in $items) {
        if ($null -eq $it) { continue }
        $type = ""
        try {
            if ($it.PSObject.Properties["Type"]) { $type = [string]$it.Type }
        } catch {}
        if ($type -ne "dir") { continue }
        $name = ""
        foreach ($k in @("Filename", "filename", "Path", "path", "Name", "name")) {
            try {
                if ($it.PSObject.Properties[$k]) {
                    $name = [string]$it.$k
                    if (-not [string]::IsNullOrWhiteSpace($name)) { break }
                }
            } catch {}
        }
        if ([string]::IsNullOrWhiteSpace($name)) { continue }
        $path = ("/" + $name.TrimStart("/")).Replace("\", "/").TrimEnd("/")
        if (-not $path.StartsWith($root + "/")) { continue }
        $ts = Get-PathTimestampUtc -Path $path
        $dirs += [pscustomobject]@{
            path = $path
            has_ts = ($ts -ne [datetime]::MinValue)
            ts = $ts
        }
    }

    if ($dirs.Count -le 0) {
        return [pscustomobject]@{
            root = $root
            listed = 0
            deleted = 0
            keep_recent = $keepN
            delete_limit = $deleteN
            reason = "empty"
            deleted_paths = @()
        }
    }

    $ordered = @(
        $dirs |
            Sort-Object `
                @{Expression = { [int]$_.has_ts }; Descending = $true}, `
                @{Expression = { if ($_.has_ts) { $_.ts } else { [datetime]::MinValue } }; Descending = $true}, `
                @{Expression = { $_.path }; Descending = $true}
    )

    $keepSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($d in @($ordered | Select-Object -First $keepN)) {
        [void]$keepSet.Add([string]$d.path)
    }
    foreach ($p in $pins) {
        [void]$keepSet.Add([string]$p)
    }

    $candidates = @(
        $ordered |
            Where-Object { $_.has_ts -and (-not $keepSet.Contains([string]$_.path)) }
    )
    $toDelete = @(
        $candidates |
            Sort-Object `
                @{Expression = { $_.ts }; Descending = $false}, `
                @{Expression = { $_.path }; Descending = $false} |
            Select-Object -First $deleteN
    )

    $deleted = New-Object System.Collections.Generic.List[string]
    $delIdx = 0
    foreach ($d in $toDelete) {
        $delIdx += 1
        $p = [string]$d.path
        if ([string]::IsNullOrWhiteSpace($p)) { continue }
        if (($delIdx -eq 1) -or (($delIdx % 10) -eq 0) -or ($delIdx -eq $toDelete.Count)) {
            Write-Host "[lr-sweep] no-space cleanup deleting (${delIdx}/$($toDelete.Count)): $p"
        }
        try {
            modal volume rm -r vggt-out $p 2>$null | Out-Null
            $deleted.Add($p) | Out-Null
        } catch {}
    }

    return [pscustomobject]@{
        root = $root
        listed = $dirs.Count
        deleted = $deleted.Count
        keep_recent = $keepN
        delete_limit = $deleteN
        reason = ""
        deleted_paths = @($deleted.ToArray())
    }
}

function Fetch-FtDebugArtifacts(
    [string]$LogDirInMntOut,
    [string]$LabelTag,
    [int]$VisCount = 12,
    [string]$VisStepsRaw = ""
) {
    $safeLabel = Sanitize($LabelTag)
    $visCountSafe = [Math]::Max(1, [int]$VisCount)
    $visSteps = @(Parse-StepList -Raw $VisStepsRaw)
    $volDir = To-VolumePath -PathInMntOut $LogDirInMntOut
    $localPrefix = "logs/modal_phase5/ftdebug_${safeLabel}"

    try {
        modal volume get vggt-out "$volDir/finetune_vggt_metrics.jsonl" "${localPrefix}_metrics.jsonl" 2>$null | Out-Null
    } catch {}
    try {
        modal volume get vggt-out "$volDir/finetune_vggt_summary.json" "${localPrefix}_summary.json" 2>$null | Out-Null
    } catch {}

    $visDir = "$volDir/ft_debug_vis"
    $items = @()
    try {
        $itemsJson = modal volume ls --json vggt-out $visDir 2>$null | Out-String
        $items = @(Parse-VolumeLsItems -ItemsJson $itemsJson)
    } catch {}
    if ($items.Count -le 0) {
        return
    }

    $allFiles = @(Get-VolumeFileNames -Items $items)
    if ($allFiles.Count -le 0) {
        return
    }
    $pngFiles = @($allFiles | Where-Object { [string]$_ -match "\.png$" })
    $jsonFiles = @($allFiles | Where-Object { [string]$_ -match "\.json$" })

    $selectedPng = @()
    if ($visSteps.Count -gt 0) {
        foreach ($s in $visSteps) {
            $pick = $pngFiles |
                Where-Object { (Get-StepFromFilename -Filename ([System.IO.Path]::GetFileName([string]$_))) -eq [int]$s } |
                Sort-Object |
                Select-Object -First 1
            if ($pick) { $selectedPng += @([string]$pick) }
        }
    } else {
        $selectedPng = @($pngFiles | Sort-Object | Select-Object -First $visCountSafe)
    }

    foreach ($f in $selectedPng) {
        $remote = Resolve-VolumeRemotePath -EntryPath ([string]$f) -BaseDir $visDir
        if ([string]::IsNullOrWhiteSpace($remote)) { continue }
        $base = [System.IO.Path]::GetFileName([string]$remote)
        $local = "${localPrefix}_$base"
        try {
            modal volume get vggt-out $remote $local 2>$null | Out-Null
        } catch {}
    }

    $selectedJson = @()
    if ($selectedPng.Count -gt 0) {
        $wantSteps = @()
        foreach ($f in $selectedPng) {
            $st = Get-StepFromFilename -Filename ([System.IO.Path]::GetFileName([string]$f))
            if ($st -ge 0) { $wantSteps += @($st) }
        }
        $wantSteps = @($wantSteps | Select-Object -Unique)
        foreach ($s in $wantSteps) {
            $pickJson = $jsonFiles |
                Where-Object { (Get-StepFromFilename -Filename ([System.IO.Path]::GetFileName([string]$_))) -eq [int]$s } |
                Sort-Object |
                Select-Object -First 1
            if ($pickJson) { $selectedJson += @([string]$pickJson) }
        }
    } else {
        $selectedJson = @($jsonFiles | Sort-Object | Select-Object -First $visCountSafe)
    }

    foreach ($f in $selectedJson) {
        $remote = Resolve-VolumeRemotePath -EntryPath ([string]$f) -BaseDir $visDir
        if ([string]::IsNullOrWhiteSpace($remote)) { continue }
        $base = [System.IO.Path]::GetFileName([string]$remote)
        $local = "${localPrefix}_$base"
        try {
            modal volume get vggt-out $remote $local 2>$null | Out-Null
        } catch {}
    }
}

function Invoke-ModalRun(
    [string]$ScriptPath = "modal_run_train.py",
    [int]$MaxRetries = 3,
    [int]$RetrySleepSec = 10,
    [int]$TimeoutSec = 3600,
    [int]$NoOutputMaxRetries = 1,
    [bool]$AllowQuietNoOutputBypass = $true
) {
    $attempt = 0
    $last = $null
    $maxRetriesSafe = [Math]::Max(1, [int]$MaxRetries)
    $noOutputMaxRetriesSafe = [Math]::Max(0, [int]$NoOutputMaxRetries)
    while ($attempt -lt $maxRetriesSafe) {
        $attempt += 1
        $stdoutFile = [System.IO.Path]::GetTempFileName()
        $stderrFile = [System.IO.Path]::GetTempFileName()
        try {
            $modalRunCmd = "modal run "
            if ([bool]$ModalRunQuiet) {
                $modalRunCmd += "-q "
            }
            $modalRunCmd += "`"$ScriptPath`""
            $proc = Start-Process `
                -FilePath "cmd.exe" `
                -ArgumentList @("/c", $modalRunCmd) `
                -WorkingDirectory $CodeDir `
                -NoNewWindow `
                -PassThru `
                -RedirectStandardOutput $stdoutFile `
                -RedirectStandardError $stderrFile
            $timeoutSecSafe = [Math]::Max(60, [int]$TimeoutSec)
            $pollSecSafe = [Math]::Max(5, [int]$ModalRunPollSec)
            $startedAt = Get-Date
            $noOutputTimeoutSecSafe = [Math]::Max(0, [int]$ModalRunNoOutputTimeoutSec)
            if ([bool]$AllowQuietNoOutputBypass -and [bool]$ModalRunQuiet -and ($noOutputTimeoutSecSafe -gt 0)) {
                # `modal run -q` is intentionally quiet for long spans; disable
                # no-output stall kill and rely on TimeoutSec for hang protection.
                Write-Host "[lr-sweep] modal run quiet mode -> disable no-output stall kill (threshold was ${noOutputTimeoutSecSafe}s)"
                $noOutputTimeoutSecSafe = 0
            }
            $lastIoChangeAt = $startedAt
            $lastStdoutLen = [int64]0
            $lastStderrLen = [int64]0
            $lastStructuredHeartbeat = $null
            $lastStructuredHeartbeatSig = ""
            $lastStructuredProgressAt = $startedAt
            $timeoutNote = "timeout"
            $timedOutByNoOutput = $false
            Write-ModalRunHeartbeat `
                -State "running" `
                -ScriptPath $ScriptPath `
                -Attempt $attempt `
                -MaxRetries $MaxRetries `
                -TimeoutSec $timeoutSecSafe `
                -ProcId $proc.Id `
                -StartedAt $startedAt `
                -StdoutFile $stdoutFile `
                -StderrFile $stderrFile `
                -Note "modal run launched" `
                -StructuredHeartbeat $lastStructuredHeartbeat

            $exited = $false
            $timedOut = $false
            $pollTick = 0
            while (-not $exited) {
                $exited = $proc.WaitForExit($pollSecSafe * 1000)
                if ($exited) { break }
                $pollTick += 1
                $elapsed = (Get-Date) - $startedAt
                if ($elapsed.TotalSeconds -ge $timeoutSecSafe) {
                    $timedOut = $true
                    break
                }
                $stdoutLenNow = [int64]0
                $stderrLenNow = [int64]0
                try {
                    if (Test-Path $stdoutFile) { $stdoutLenNow = [int64](Get-Item $stdoutFile).Length }
                    if (Test-Path $stderrFile) { $stderrLenNow = [int64](Get-Item $stderrFile).Length }
                } catch {}
                $structuredHeartbeatNow = Get-LatestStructuredHeartbeat -StdoutFile $stdoutFile
                if ($null -ne $structuredHeartbeatNow) {
                    $sigNow = Get-StructuredHeartbeatSignature -Heartbeat $structuredHeartbeatNow
                    if (($sigNow -ne "") -and ($sigNow -ne $lastStructuredHeartbeatSig)) {
                        $lastStructuredHeartbeat = $structuredHeartbeatNow
                        $lastStructuredHeartbeatSig = $sigNow
                        if ([string]$structuredHeartbeatNow.source -ne "remote_alive") {
                            $lastStructuredProgressAt = Get-Date
                        }
                    }
                }
                if (($stdoutLenNow -ne $lastStdoutLen) -or ($stderrLenNow -ne $lastStderrLen)) {
                    $lastIoChangeAt = Get-Date
                    $lastStdoutLen = $stdoutLenNow
                    $lastStderrLen = $stderrLenNow
                }
                if ($noOutputTimeoutSecSafe -gt 0) {
                    if (($null -ne $lastStructuredHeartbeat) -and ([string]$lastStructuredHeartbeat.source -ne "remote_alive")) {
                        $heartbeatAgeSec = ((Get-Date) - $lastStructuredProgressAt).TotalSeconds
                        if ($heartbeatAgeSec -ge $noOutputTimeoutSecSafe) {
                            $timedOut = $true
                            $timedOutByNoOutput = $true
                            $timeoutNote = "heartbeat_stall_timeout_${noOutputTimeoutSecSafe}s"
                            Write-Host "[lr-sweep] modal run heartbeat stall script=$ScriptPath attempt=$attempt source=$($lastStructuredHeartbeat.source) phase=$($lastStructuredHeartbeat.phase) progress=$($lastStructuredHeartbeat.progress_counter) age=${heartbeatAgeSec}s threshold=${noOutputTimeoutSecSafe}s"
                            break
                        }
                    } else {
                        $noIoElapsedSec = ((Get-Date) - $lastIoChangeAt).TotalSeconds
                        if ($noIoElapsedSec -ge $noOutputTimeoutSecSafe) {
                            $timedOut = $true
                            $timedOutByNoOutput = $true
                            $timeoutNote = "no_output_timeout_${noOutputTimeoutSecSafe}s"
                            Write-Host "[lr-sweep] modal run stall detected script=$ScriptPath attempt=$attempt no_output_elapsed=${noIoElapsedSec}s threshold=${noOutputTimeoutSecSafe}s"
                            break
                        }
                    }
                }
                if (($pollTick -eq 1) -or (($pollTick % 3) -eq 0)) {
                    $stdoutLen = 0
                    $stderrLen = 0
                    try {
                        if (Test-Path $stdoutFile) { $stdoutLen = [int64](Get-Item $stdoutFile).Length }
                        if (Test-Path $stderrFile) { $stderrLen = [int64](Get-Item $stderrFile).Length }
                    } catch {}
                    $elapsedSec = [int][Math]::Round($elapsed.TotalSeconds)
                    $hbSource = ""
                    $hbPhase = ""
                    $hbAge = ""
                    if ($null -ne $lastStructuredHeartbeat) {
                        $hbSource = [string]$lastStructuredHeartbeat.source
                        $hbPhase = [string]$lastStructuredHeartbeat.phase
                        $hbAge = [int][Math]::Round(((Get-Date) - $lastStructuredProgressAt).TotalSeconds)
                    }
                    Write-Host "[lr-sweep] modal run alive script=$ScriptPath attempt=$attempt elapsed=${elapsedSec}s timeout=${timeoutSecSafe}s stdout_bytes=${stdoutLen} stderr_bytes=${stderrLen} hb_source=$hbSource hb_phase=$hbPhase hb_age=${hbAge}s"
                }
                Write-ModalRunHeartbeat `
                    -State "running" `
                    -ScriptPath $ScriptPath `
                    -Attempt $attempt `
                    -MaxRetries $MaxRetries `
                    -TimeoutSec $timeoutSecSafe `
                    -ProcId $proc.Id `
                    -StartedAt $startedAt `
                    -StdoutFile $stdoutFile `
                    -StderrFile $stderrFile `
                    -Note "polling" `
                    -StructuredHeartbeat $lastStructuredHeartbeat
            }

            if ($timedOut) {
                try {
                    & taskkill /PID $proc.Id /T /F > $null 2>&1
                } catch {}
                Start-Sleep -Milliseconds 200
                try {
                    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                } catch {}
            }
            $output = @()
            if (Test-Path $stdoutFile) { $output += @(Get-Content $stdoutFile) }
            if (Test-Path $stderrFile) { $output += @(Get-Content $stderrFile) }
            if (-not $timedOut) {
                $rc = [int]$proc.ExitCode
            } else {
                $output += @("[modal-run] timeout after ${timeoutSecSafe}s ($timeoutNote)")
                $rc = 124
            }
            Write-ModalRunHeartbeat `
                -State $(if ($rc -eq 0) { "done" } elseif ($timedOut) { "timeout" } else { "failed" }) `
                -ScriptPath $ScriptPath `
                -Attempt $attempt `
                -MaxRetries $MaxRetries `
                -TimeoutSec $timeoutSecSafe `
                -ProcId $proc.Id `
                -StartedAt $startedAt `
                -StdoutFile $stdoutFile `
                -StderrFile $stderrFile `
                -ExitCode $rc `
                -Note "attempt finished" `
                -StructuredHeartbeat $lastStructuredHeartbeat
            $last = [pscustomobject]@{
                Output = $output
                ExitCode = $rc
                Attempt = $attempt
                TimeoutNote = $timeoutNote
                TimedOutByNoOutput = $timedOutByNoOutput
            }
            if ($rc -eq 0) {
                return $last
            }

            $blob = ($output -join "`n")
            $hasModalRunTimeout = ($blob -match "\[modal-run\] timeout")
            $isTransient = (
                ($blob -match "Connection lost") -or
                ($blob -match "WinError 10053") -or
                ($blob -match "WinError 10054") -or
                ($blob -match "SSL shutdown timed out") -or
                ($blob -match "Deadline exceeded") -or
                ($blob -match "heartbeat failed") -or
                ($blob -match "modal\.exception\.ConnectionError") -or
                ($blob -match "timed out waiting for final app logs") -or
                ($blob -match "Could not connect to the Modal server") -or
                ($blob -match "Cannot connect to host") -or
                ($blob -match "cloudflarestorage.com") -or
                ($blob -match "FETCH_HEAD was modified during build process") -or
                ($blob -match "\.git/HEAD was modified during build process") -or
                ($blob -match "was modified during build process")
            )
            if ($hasModalRunTimeout -and (-not $timedOutByNoOutput)) {
                $isTransient = $true
            }

            $retryCap = $maxRetriesSafe
            if ($timedOutByNoOutput) {
                $retryCap = [Math]::Min($maxRetriesSafe, $noOutputMaxRetriesSafe)
            }

            if ($isTransient -and $attempt -lt $retryCap) {
                Write-ModalRunHeartbeat `
                    -State "retrying" `
                    -ScriptPath $ScriptPath `
                    -Attempt $attempt `
                    -MaxRetries $MaxRetries `
                    -TimeoutSec $timeoutSecSafe `
                    -ProcId $proc.Id `
                    -StartedAt $startedAt `
                    -StdoutFile $stdoutFile `
                    -StderrFile $stderrFile `
                    -ExitCode $rc `
                    -Note "transient failure, waiting for retry"
                Write-Host "[modal-run] transient failure attempt=$attempt/$MaxRetries, retry in ${RetrySleepSec}s"
                Start-Sleep -Seconds $RetrySleepSec
                continue
            }
            if ($timedOutByNoOutput -and $attempt -ge $retryCap) {
                Write-Host "[modal-run] no-output timeout stop-retry attempt=$attempt cap=$retryCap timeout_note=$timeoutNote"
            }
            return $last
        } finally {
            Remove-Item $stdoutFile -ErrorAction SilentlyContinue
            Remove-Item $stderrFile -ErrorAction SilentlyContinue
        }
    }
    return $last
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$lrVals = Parse-Lrs -Raw $LrList
if ($lrVals.Count -eq 0) {
    throw "LrList is empty: $LrList"
}

if ($FreezeMode -ne "all_trainable") {
    Write-Host "[lr-sweep] warning: FreezeMode=$FreezeMode (mentor-aligned default is all_trainable)."
}

$shortInfos = @()
$sweepRows = @()

foreach ($lr in $lrVals) {
    $safeLr = Sanitize($lr)
    $label = "lr_$safeLr"
    $ftCkptDir = "/mnt/out/vggt/finetune/lr_$safeLr`_$timestamp/ckpt"
    $ftLogDir = "/mnt/out/vggt/finetune/lr_$safeLr`_$timestamp/logs"
    $ftModelPath = "$ftCkptDir/model_ft_zju.pt"
    $geomOut = "vggt_geom_ft_lr_$safeLr`_$timestamp"

    $argsExtra = @(
        "--epochs=$EpochsShort",
        "--max_frames=$MaxFramesShort",
        "--lr=$lr",
        "--freeze_mode=$FreezeMode",
        "--depth_scale_align=$DepthScaleAlign",
        "--lambda_depth=$LambdaDepth",
        "--lambda_point=$LambdaPoint",
        "--lambda_point_reproj=$LambdaPointReproj",
        "--lambda_point_normal_consis=$LambdaPointNormalConsis",
        "--lambda_point_mv_depth=$LambdaPointMvDepth",
        "--lambda_point_mv_mask=$LambdaPointMvMask",
        "--lambda_conf=$LambdaConf",
        "--lambda_conf_warmup_steps=$LambdaConfWarmupSteps",
        "--lambda_geom_cons=$LambdaGeomCons",
        "--lambda_cam=$LambdaCam",
        "--lambda_cam_warmup_steps=$LambdaCamWarmupSteps",
        "--cam_rot_weight=$CamRotWeight",
        "--cam_fov_weight=$CamFovWeight",
        "--cam_warmup_steps=$CamWarmupSteps",
        "--jitter=$Jitter",
        "--noise_std=$NoiseStd",
        "--robust_l1_eps=$RobustL1Eps",
        "--conf_weight_thr=$ConfWeightThr",
        "--conf_weight_gamma=$ConfWeightGamma",
        "--conf_weight_per_view_quantile=$ConfWeightPerViewQuantile",
        "--conf_weight_per_view_min_valid=$ConfWeightPerViewMinValid",
        "--gram_dyn_enable=$GramDynEnable",
        "--gram_dyn_layer_idx=$GramDynLayerIdx",
        "--gram_dyn_quantile=$GramDynQuantile",
        "--gram_dyn_weight_floor=$GramDynWeightFloor",
        "--gram_dyn_warmup_steps=$GramDynWarmupSteps",
        "--dyn_proxy_enable=$DynProxyEnable",
        "--dyn_proxy_mode=$DynProxyMode",
        "--dyn_proxy_use_gram=$DynProxyUseGram",
        "--dyn_proxy_use_support=$DynProxyUseSupport",
        "--dyn_proxy_floor=$DynProxyFloor",
        "--dyn_proxy_warmup_steps=$DynProxyWarmupSteps",
        "--point_cons_tau=$PointConsTau",
        "--point_cons_weight_floor=$PointConsWeightFloor",
        "--point_cons_clip_min_qv=$PointConsClipMinQv",
        "--point_cons_quantile=$PointConsQuantile",
        "--point_cons_focus=$PointConsFocus",
        "--point_residual_quantile=$PointResidualQuantile",
        "--point_residual_focus=$PointResidualFocus",
        "--point_residual_boost=$PointResidualBoost",
        "--point_residual_boost_cap=$PointResidualBoostCap",
        "--point_target_mode=$PointTargetMode",
        "--point_target_blend_alpha=$PointTargetBlendAlpha",
        "--point_target_blend_alpha_min=$PointTargetBlendAlphaMin",
        "--point_target_blend_alpha_max=$PointTargetBlendAlphaMax",
        "--point_target_blend_rel_gain=$PointTargetBlendRelGain",
        "--point_target_blend_mv_gain=$PointTargetBlendMvGain",
        "--point_target_blend_by_reliability=$PointTargetBlendByReliability",
        "--point_target_blend_by_mv_support=$PointTargetBlendByMvSupport",
        "--point_target_blend_mv_region_mode=$PointTargetBlendMvRegionMode",
        "--point_target_blend_mv_policy=$PointTargetBlendMvPolicy",
        "--point_target_consensus_alpha_floor=$PointTargetConsensusAlphaFloor",
        "--target_point_frame=$TargetPointFrame",
        "--pred_point_frame=$PredPointFrame",
        "--use_fg_mask=$UseFgMask",
        "--fg_mask_source=$FgMaskSource",
        "--fg_mask_erode_px=$FgMaskErodePx",
        "--point_loss_fg_erode_px=$PointLossFgErodePx",
        "--fg_supervision_boost=$FgSupervisionBoost",
        "--fg_supervision_bg_floor=$FgSupervisionBgFloor",
        "--fg_supervision_region_mode=$FgSupervisionRegionMode",
        "--fg_supervision_region_erode_px=$FgSupervisionRegionErodePx",
        "--lambda_fg_conf_presence=$LambdaFgConfPresence",
        "--fg_conf_presence_target_ratio=$FgConfPresenceTargetRatio",
        "--lambda_fg_structure_depth_edge=$LambdaFgStructureDepthEdge",
        "--fg_structure_bbox_margin_px=$FgStructureBboxMarginPx",
        "--fg_structure_bbox_min_side_px=$FgStructureBboxMinSidePx",
        "--fg_structure_region_mode=$FgStructureRegionMode",
        "--fg_structure_region_erode_px=$FgStructureRegionErodePx",
        "--fg_structure_depth_edge_warmup_steps=$FgStructureDepthEdgeWarmupSteps",
        "--fg_structure_boundary_probe_px=$FgStructureBoundaryProbePx",
        "--fg_structure_edge_support_mode=$FgStructureEdgeSupportMode",
        "--fg_structure_edge_support_quantile=$FgStructureEdgeSupportQuantile",
        "--fg_structure_edge_support_min_px=$FgStructureEdgeSupportMinPx",
        "--fg_structure_edge_weight_mode=$FgStructureEdgeWeightMode",
        "--fg_structure_boundary_falloff_px=$FgStructureBoundaryFalloffPx",
        "--fg_structure_component_bias_mode=$FgStructureComponentBiasMode",
        "--fg_structure_component_bias_threshold_ratio=$FgStructureComponentBiasThresholdRatio",
        "--fg_structure_component_bias_other_scale=$FgStructureComponentBiasOtherScale",
        "--fg_structure_front_depth_bias_mode=$FgStructureFrontDepthBiasMode",
        "--fg_structure_front_depth_bias_tau=$FgStructureFrontDepthBiasTau",
        "--fg_structure_front_depth_bias_center_quantile=$FgStructureFrontDepthBiasCenterQuantile",
        "--lambda_point_mv_outside_ring=$LambdaPointMvOutsideRing",
        "--point_mv_outside_ring_px=$PointMvOutsideRingPx",
        "--supervision_weight_mode=$SupervisionWeightMode",
        "--supervision_weight_mix_alpha=$SupervisionWeightMixAlpha",
        "--point_reproj_warmup_steps=$PointReprojWarmupSteps",
        "--point_reproj_clamp_px=$PointReprojClampPx",
        "--point_mv_consistency=$PointMvConsistency",
        "--point_mv_tol_abs=$PointMvTolAbs",
        "--point_mv_tol_rel=$PointMvTolRel",
        "--point_mv_weight_floor=$PointMvWeightFloor",
        "--point_mv_stride=$PointMvStride",
        "--point_mv_depth_max_pairs=$PointMvDepthMaxPairs",
        "--point_mv_depth_pair_mode=$PointMvDepthPairMode",
        "--point_mv_depth_warmup_steps=$PointMvDepthWarmupSteps",
        "--point_mv_depth_region_mode=$PointMvDepthRegionMode",
        "--point_mv_mask_warmup_steps=$PointMvMaskWarmupSteps",
        "--point_mv_depth_inlier_only=$PointMvDepthInlierOnly",
        "--point_mv_depth_err_quantile=$PointMvDepthErrQuantile",
        "--point_mv_depth_outlier_boost=$PointMvDepthOutlierBoost",
        "--point_mv_depth_outlier_cap=$PointMvDepthOutlierCap",
        "--point_mv_depth_tgt_valid_mode=$PointMvDepthTgtValidMode",
        "--point_mv_depth_tgt_valid_floor=$PointMvDepthTgtValidFloor",
        "--point_mv_depth_min_tgt_valid_ratio=$PointMvDepthMinTgtValidRatio",
        "--point_mv_mask_min_tgt_fg_ratio=$PointMvMaskMinTgtFgRatio",
        "--point_mv_mask_hit_thr=$PointMvMaskHitThr",
        "--point_mv_mask_soft_blur_px=$PointMvMaskSoftBlurPx",
        "--point_mv_mask_soft_blur_iters=$PointMvMaskSoftBlurIters",
        "--point_mv_mask_soft_mix=$PointMvMaskSoftMix",
        "--point_mv_mask_soft_hit_thr=$PointMvMaskSoftHitThr",
        "--point_mv_depth_tgt_valid_scale_mode=$PointMvDepthTgtValidScaleMode",
        "--point_mv_depth_tgt_valid_scale_thr=$PointMvDepthTgtValidScaleThr",
        "--point_mv_depth_adapt_mode=$PointMvDepthAdaptMode",
        "--point_mv_depth_adapt_target_valid=$PointMvDepthAdaptTargetValid",
        "--point_mv_depth_adapt_min_scale=$PointMvDepthAdaptMinScale",
        "--point_mv_depth_adapt_max_scale=$PointMvDepthAdaptMaxScale",
        "--point_support_mode=$PointSupportMode",
        "--point_support_floor=$PointSupportFloor",
        "--point_mv_depth_support_mode=$PointMvDepthSupportMode",
        "--point_mv_depth_support_floor=$PointMvDepthSupportFloor",
        "--point_mv_mask_support_mode=$PointMvMaskSupportMode",
        "--point_mv_mask_support_floor=$PointMvMaskSupportFloor",
        "--point_mv_depth_fg_erode_px=$PointMvDepthFgErodePx",
        "--point_loss_scale_depth_unproject=$PointLossScaleDepthUnproject",
        "--point_warmup_steps=$PointWarmupSteps",
        "--point_normal_consis_warmup_steps=$PointNormalConsisWarmupSteps",
        "--lr_backbone_scale=$LrBackboneScale",
        "--lr_head_scale=$LrHeadScale",
        "--lr_camera_scale=$LrCameraScale",
        "--grad_clip=$GradClip",
        "--geom_subdir=$PseudoGeomSubdir",
        "--log_dir=$ftLogDir",
        "--ckpt_dir=$ftCkptDir",
        "--eval_every_steps=$EvalEverySteps",
        "--debug_metrics_every_steps=$DebugMetricsEverySteps",
        "--debug_vis_every_steps=$DebugVisEverySteps",
        "--debug_vis_max_steps=$DebugVisMaxSteps",
        "--debug_vis_views=$DebugVisViews",
        "--early_stop_patience=$EarlyStopPatience",
        "--min_improve=$MinImprove",
        "--max_steps_per_epoch=$MaxStepsPerEpoch"
    ) -join " "
    if (-not [string]::IsNullOrWhiteSpace($DebugVisDir)) {
        $argsExtra = "$argsExtra --debug_vis_dir=$DebugVisDir"
    }
    if (-not [string]::IsNullOrWhiteSpace($ResumeCkpt)) {
        $argsExtra = "$argsExtra --resume_ckpt=$ResumeCkpt"
    }

    $ftModelPathResolved = ""
    $usedResumeFallbackForShort = $false
    $shortCkptSourceNote = ""
    $shortCkptResolveReason = ""
    $ftMetricsLocal = "logs/modal_phase5/ftdebug_${safeLr}_${timestamp}_short_metrics.jsonl"
    $meanStepUpdateRatio = $null
    $reuseShortFtActive = -not [string]::IsNullOrWhiteSpace($ReuseShortFtCkpt)

    if ($reuseShortFtActive) {
        Write-Host "[lr-sweep] reuse short finetuned ckpt lr=$lr source=$ReuseShortFtCkpt"
        $ftModelPathResolved = Resolve-ModalCheckpointPath -PrimaryPathInMntOut $ReuseShortFtCkpt -TimeoutSec ([Math]::Min([int]$CkptWaitTimeoutSec, 300)) -PollSec $CkptWaitPollSec -AllowTmpFallback
        $shortCkptResolveReason = [string]$script:LastCheckpointResolveReason
        if ([string]::IsNullOrWhiteSpace($ftModelPathResolved)) {
            $sweepRows += [pscustomobject]@{
                stage = "short"
                label = $label
                lr = $lr
                freeze_mode = $FreezeMode
                depth_scale_align = $DepthScaleAlign
                geom_subdir = ""
                ft_ckpt = $ReuseShortFtCkpt
                status = "error"
                reason = "reused short ft checkpoint missing: resolve=${shortCkptResolveReason}: $ReuseShortFtCkpt"
            }
            continue
        }
        $shortCkptSourceNote = "reused_short_ft_ckpt"
        $ftLogLocal = "logs/modal_phase5/vggt_ft_lr_${safeLr}_$timestamp.finetune.log"
        @("[lr-sweep] reused short finetuned ckpt: $ftModelPathResolved") | Tee-Object -FilePath $ftLogLocal | Out-Null
        $reuseMetricsSource = Find-ReuseFtDebugLocalPath -ReuseFtCkptPath $ReuseShortFtCkpt -Kind metrics
        if (-not [string]::IsNullOrWhiteSpace($reuseMetricsSource) -and (Test-Path $reuseMetricsSource)) {
            Copy-Item -Path $reuseMetricsSource -Destination $ftMetricsLocal -Force
        }
        $reuseSummarySource = Find-ReuseFtDebugLocalPath -ReuseFtCkptPath $ReuseShortFtCkpt -Kind summary
        $ftSummaryLocal = "logs/modal_phase5/ftdebug_${safeLr}_${timestamp}_short_summary.json"
        if (-not [string]::IsNullOrWhiteSpace($reuseSummarySource) -and (Test-Path $reuseSummarySource)) {
            Copy-Item -Path $reuseSummarySource -Destination $ftSummaryLocal -Force
        }
    } else {
    Write-Host "[lr-sweep] short finetune lr=$lr"
    $env:VGGT_CODE_DIR = $CodeDir
    $env:VGGT_MODE = "precompute"
    $env:VGGT_TF32 = $(if ([bool]$Tf32) { "1" } else { "0" })
    $env:VGGT_AMP = $(if ([bool]$Amp) { "1" } else { "0" })
    $env:VGGT_STRICT_DETERMINISTIC = $(if ([bool]$StrictDeterministic) { "1" } else { "0" })
    $env:VGGT_SEQ_NAMES = $SeqNames
    $env:VGGT_GEOM_SUBDIR = $PseudoGeomSubdir
    $env:VGGT_CAM_NAMES = $CamNames
    $env:VGGT_MAX_FRAMES = [string]$MaxFramesShort
    $env:VGGT_PRECOMPUTE_SCRIPT = "finetune_vggt_pseudo.py"
    $env:VGGT_PRECOMPUTE_CKPT = $PretrainedCkpt
    $env:VGGT_PRECOMPUTE_ARGS_EXTRA = $argsExtra
    Remove-Item Env:VGGT_POINTMAP_SOURCE -ErrorAction SilentlyContinue
    Remove-Item Env:VGGT_PROFILE -ErrorAction SilentlyContinue

    $ftRun = Invoke-ModalRun `
        -ScriptPath "modal_run_train.py" `
        -TimeoutSec $ModalRunTimeoutSec `
        -NoOutputMaxRetries $ModalRunNoOutputMaxRetries `
        -AllowQuietNoOutputBypass $ShortFinetuneAllowQuietNoOutputBypass
    $ftOutput = @($ftRun.Output)
    $ftRc = [int]$ftRun.ExitCode
    $ftLogLocal = "logs/modal_phase5/vggt_ft_lr_${safeLr}_$timestamp.finetune.log"
    $ftOutput | Tee-Object -FilePath $ftLogLocal | Out-Null
    $ftNoSpaceDetected = Test-NoSpaceError -Lines $ftOutput
    if ($ftNoSpaceDetected -and ($NoSpaceRetryCount -gt 0)) {
        $runPins = @(
            (Get-RunDirFromMntOutPath -PathInMntOut $ResumeCkpt),
            (Get-RunDirFromMntOutPath -PathInMntOut $ftModelPath),
            (Get-RunDirFromMntOutPath -PathInMntOut $ftCkptDir)
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique
        for ($nsRetry = 1; $nsRetry -le $NoSpaceRetryCount; $nsRetry++) {
            $clean = Invoke-NoSpaceCleanup `
                -RootDir $NoSpaceCleanupRoot `
                -PinnedRunDirs $runPins `
                -KeepRecentDirs $NoSpaceCleanupKeepRecentDirs `
                -DeleteLimit $NoSpaceCleanupDeleteLimit
            Write-Host "[lr-sweep] no-space recovery cleanup retry=${nsRetry}/${NoSpaceRetryCount} listed=$($clean.listed) deleted=$($clean.deleted) keep_recent=$($clean.keep_recent) delete_limit=$($clean.delete_limit)"
            if ($CkptMissingRetrySleepSec -gt 0) {
                Start-Sleep -Seconds $CkptMissingRetrySleepSec
            }
            $ftRun = Invoke-ModalRun `
                -ScriptPath "modal_run_train.py" `
                -TimeoutSec $ModalRunTimeoutSec `
                -NoOutputMaxRetries $ModalRunNoOutputMaxRetries `
                -AllowQuietNoOutputBypass $ShortFinetuneAllowQuietNoOutputBypass
            $ftOutput = @($ftRun.Output)
            $ftRc = [int]$ftRun.ExitCode
            $ftNoSpaceLog = "logs/modal_phase5/vggt_ft_lr_${safeLr}_$timestamp.finetune_nospace_retry${nsRetry}.log"
            $ftOutput | Tee-Object -FilePath $ftNoSpaceLog | Out-Null
            $ftNoSpaceDetected = Test-NoSpaceError -Lines $ftOutput
            if (-not $ftNoSpaceDetected) {
                Write-Host "[lr-sweep] no-space recovery cleared on retry ${nsRetry}/${NoSpaceRetryCount}"
                break
            }
        }
    }
    if ($ftNoSpaceDetected) {
        $sweepRows += [pscustomobject]@{
            stage = "short"
            label = $label
            lr = $lr
            freeze_mode = $FreezeMode
            depth_scale_align = $DepthScaleAlign
            geom_subdir = ""
            ft_ckpt = $ftModelPath
            status = "error"
            reason = "finetune no space left on device (after cleanup retry=$NoSpaceRetryCount)"
        }
        continue
    }
    if ($ftRc -ne 0) {
        $ftFailReason = Resolve-ModalRunFailureReason `
            -RunResult $ftRun `
            -Lines $ftOutput `
            -DefaultReason "finetune modal run failed"
        $sweepRows += [pscustomobject]@{
            stage = "short"
            label = $label
            lr = $lr
            freeze_mode = $FreezeMode
            depth_scale_align = $DepthScaleAlign
            geom_subdir = ""
            ft_ckpt = $ftModelPath
            status = "error"
            reason = $ftFailReason
        }
        continue
    }
    $ftModelPathResolved = Resolve-ModalCheckpointPath -PrimaryPathInMntOut $ftModelPath -TimeoutSec $CkptWaitTimeoutSec -PollSec $CkptWaitPollSec -AllowTmpFallback
    $usedResumeFallbackForShort = $false
    $shortCkptSourceNote = ""
    $shortCkptResolveReason = [string]$script:LastCheckpointResolveReason
    $ckptRetryNoSpaceHit = $false
    if ([string]::IsNullOrWhiteSpace($ftModelPathResolved) -and ($CkptMissingRetryCount -gt 0)) {
        for ($ckptRetry = 1; $ckptRetry -le $CkptMissingRetryCount; $ckptRetry++) {
            Write-Host "[lr-sweep] checkpoint missing after finetune success, retry short finetune (${ckptRetry}/${CkptMissingRetryCount}) lr=$lr"
            if ($CkptMissingRetrySleepSec -gt 0) {
                Start-Sleep -Seconds $CkptMissingRetrySleepSec
            }
            $ftRunRetry = Invoke-ModalRun `
                -ScriptPath "modal_run_train.py" `
                -TimeoutSec $ModalRunTimeoutSec `
                -NoOutputMaxRetries $ModalRunNoOutputMaxRetries `
                -AllowQuietNoOutputBypass $ShortFinetuneAllowQuietNoOutputBypass
            $ftOutputRetry = @($ftRunRetry.Output)
            $ftRcRetry = [int]$ftRunRetry.ExitCode
            $ftRetryLogLocal = "logs/modal_phase5/vggt_ft_lr_${safeLr}_$timestamp.finetune_ckpt_retry${ckptRetry}.log"
            $ftOutputRetry | Tee-Object -FilePath $ftRetryLogLocal | Out-Null
            $ftRetryNoSpace = Test-NoSpaceError -Lines $ftOutputRetry
            if ($ftRetryNoSpace) {
                Write-Host "[lr-sweep] checkpoint retry hit no-space (retry=${ckptRetry})"
                $ckptRetryNoSpaceHit = $true
                break
            }
            if ($ftRcRetry -ne 0) {
                Write-Host "[lr-sweep] retry short finetune failed rc=$ftRcRetry (retry=${ckptRetry})"
                continue
            }
            $ftModelPathResolved = Resolve-ModalCheckpointPath -PrimaryPathInMntOut $ftModelPath -TimeoutSec $CkptWaitTimeoutSec -PollSec $CkptWaitPollSec -AllowTmpFallback
            $shortCkptResolveReason = [string]$script:LastCheckpointResolveReason
            if (-not [string]::IsNullOrWhiteSpace($ftModelPathResolved)) {
                Write-Host "[lr-sweep] checkpoint resolved after retry (${ckptRetry}/${CkptMissingRetryCount})"
                break
            }
        }
    }
    if ($ckptRetryNoSpaceHit) {
        $sweepRows += [pscustomobject]@{
            stage = "short"
            label = $label
            lr = $lr
            freeze_mode = $FreezeMode
            depth_scale_align = $DepthScaleAlign
            geom_subdir = ""
            ft_ckpt = $ftModelPath
            status = "error"
            reason = "finetune no space left on device during checkpoint retry"
        }
        continue
    }
    if ([string]::IsNullOrWhiteSpace($ftModelPathResolved) -and [bool]$EnableExtendedCkptWaitOnMissing) {
        $extraWait = [Math]::Max(60, [int]$CkptExtendedWaitTimeoutSec)
        if ($extraWait -gt [int]$CkptWaitTimeoutSec) {
            Write-Host "[lr-sweep] extended checkpoint materialization wait: timeout=${CkptWaitTimeoutSec}s -> ${extraWait}s"
            $ftModelPathResolved = Resolve-ModalCheckpointPath -PrimaryPathInMntOut $ftModelPath -TimeoutSec $extraWait -PollSec $CkptWaitPollSec -AllowTmpFallback
            $shortCkptResolveReason = [string]$script:LastCheckpointResolveReason
            if (-not [string]::IsNullOrWhiteSpace($ftModelPathResolved)) {
                $shortCkptSourceNote = "extended_ckpt_wait_success"
            }
        }
    }
    if ([string]::IsNullOrWhiteSpace($ftModelPathResolved) -and [bool]$EnableResumeCkptFallbackOnShortCkptMissing -and (-not [string]::IsNullOrWhiteSpace($ResumeCkpt))) {
        $resumeWaitTimeout = [Math]::Min([int]$CkptWaitTimeoutSec, 300)
        $resumeWaitTimeout = [Math]::Max($resumeWaitTimeout, 60)
        $resumeResolved = Resolve-ModalCheckpointPath `
            -PrimaryPathInMntOut $ResumeCkpt `
            -TimeoutSec $resumeWaitTimeout `
            -PollSec $CkptWaitPollSec `
            -AllowTmpFallback
        if (-not [string]::IsNullOrWhiteSpace($resumeResolved)) {
            $ftModelPathResolved = $resumeResolved
            $usedResumeFallbackForShort = $true
            $shortCkptSourceNote = "fallback_resume_ckpt_due_missing_short_ckpt"
            Write-Host "[lr-sweep] checkpoint missing for short finetune; fallback to resume_ckpt=$resumeResolved"
        }
    }
    if ([string]::IsNullOrWhiteSpace($ftModelPathResolved)) {
        $sweepRows += [pscustomobject]@{
            stage = "short"
            label = $label
            lr = $lr
            freeze_mode = $FreezeMode
            depth_scale_align = $DepthScaleAlign
            geom_subdir = ""
            ft_ckpt = $ftModelPath
            status = "error"
            reason = "finetune checkpoint missing on volume after retry($CkptMissingRetryCount), resolve=${shortCkptResolveReason}: $ftModelPath"
        }
        continue
    }
    Fetch-FtDebugArtifacts `
        -LogDirInMntOut $ftLogDir `
        -LabelTag "${safeLr}_${timestamp}_short" `
        -VisCount $DownloadFtDebugVisCount `
        -VisStepsRaw $DownloadFtDebugVisSteps

    $ftMetricsLocal = "logs/modal_phase5/ftdebug_${safeLr}_${timestamp}_short_metrics.jsonl"
    $meanStepUpdateRatio = $null
    $ftMetricsSnapshot = Get-FtMetricsSnapshot -MetricsPath $ftMetricsLocal
    if ($null -ne $ftMetricsSnapshot) {
        try {
            if ($ftMetricsSnapshot.PSObject.Properties["mean_step_update_ratio"]) {
                $meanStepUpdateRatio = [double]$ftMetricsSnapshot.mean_step_update_ratio
            }
        } catch {}
    }
    if (($MinMeanStepUpdateRatio -gt 0.0) -and ($null -ne $meanStepUpdateRatio) -and ($meanStepUpdateRatio -lt $MinMeanStepUpdateRatio)) {
        Write-Host "[lr-sweep] stalled update: mean_step_update_ratio=$meanStepUpdateRatio < $MinMeanStepUpdateRatio (skip precompute)"
        $sweepRows += [pscustomobject]@{
            stage = "short"
            label = $label
            lr = $lr
            freeze_mode = $FreezeMode
            depth_scale_align = $DepthScaleAlign
            geom_subdir = ""
            ft_ckpt = $ftModelPath
            status = "error"
            reason = "stalled update: mean_step_update_ratio=$meanStepUpdateRatio < $MinMeanStepUpdateRatio"
        }
        continue
    }
    }

    if ($reuseShortFtActive) {
        Write-Host "[lr-sweep] precompute geom from reused short ft ckpt lr=$lr"
    } elseif ($usedResumeFallbackForShort) {
        Write-Host "[lr-sweep] precompute geom from resume fallback ckpt lr=$lr"
    } else {
        Write-Host "[lr-sweep] precompute geom from finetuned ckpt lr=$lr"
    }
    $pointmapSourceRequested = [string]$PointmapSource
    $pointmapSourceResolved = $PointmapSource
    if ([string]::IsNullOrWhiteSpace($pointmapSourceResolved) -or $pointmapSourceResolved -eq "auto") {
        if ($FreezeMode -eq "point_only") {
            $pointmapSourceResolved = "point_head"
        } else {
            $pointmapSourceResolved = "depth_unproject"
        }
    }
    $pointmapSourcePrimary = [string]$pointmapSourceResolved
    $precomputeFallbackUsed = $false
    $precomputeTimeoutHit = $false
    Write-Host "[lr-sweep] pointmap_source=$pointmapSourceResolved (freeze_mode=$FreezeMode)"
    $env:VGGT_MODE = "precompute"
    $env:VGGT_TF32 = $(if ([bool]$Tf32) { "1" } else { "0" })
    $env:VGGT_AMP = $(if ([bool]$Amp) { "1" } else { "0" })
    $env:VGGT_STRICT_DETERMINISTIC = $(if ([bool]$StrictDeterministic) { "1" } else { "0" })
    $env:VGGT_SEQ_NAMES = $SeqNames
    $env:VGGT_CAM_NAMES = $CamNames
    $env:VGGT_PRECOMPUTE_SCRIPT = "precompute_zju_vggt_geom.py"
    $env:VGGT_PRECOMPUTE_CKPT = $ftModelPathResolved
    $env:VGGT_GEOM_SUBDIR = $geomOut
    $env:VGGT_MAX_FRAMES = [string]$MaxFramesShort
    $env:VGGT_POINTMAP_SOURCE = $pointmapSourceResolved
    $env:VGGT_POINT_HEAD_FRAME = $PointHeadFrame
    $env:VGGT_UNPROJECT_IMPL = $UnprojectImpl
    $env:VGGT_MV_SUPPORT_ON = $PrecomputeMvSupportOn
    $env:VGGT_MV_SUPPORT_TOL_ABS = [string]$PrecomputeMvSupportTolAbs
    $env:VGGT_MV_SUPPORT_TOL_REL = [string]$PrecomputeMvSupportTolRel
    $env:VGGT_MV_SUPPORT_STRIDE = [string]$PrecomputeMvSupportStride
    $env:VGGT_MV_SUPPORT_MODE = $PrecomputeMvSupportMode
    $env:VGGT_MV_SUPPORT_FLOOR = [string]$PrecomputeMvSupportFloor
    $env:VGGT_MV_SUPPORT_GAMMA = [string]$PrecomputeMvSupportGamma
    $env:VGGT_MV_SUPPORT_CLIP_THR = [string]$PrecomputeMvSupportClipThr
    $env:VGGT_MV_SUPPORT_CLIP_FLOOR = [string]$PrecomputeMvSupportClipFloor
    $env:VGGT_MV_SUPPORT_HARD_THR = [string]$PrecomputeMvSupportHardThr
    $env:VGGT_MV_CONF_VALID_FLOOR = [string]$PrecomputeMvConfValidFloor
    $env:VGGT_MV_SUPPORT_SAVE = $PrecomputeMvSupportSave
    $env:VGGT_MV_SUPPORT_SAVE_RAW_CONF = $PrecomputeMvSupportSaveRawConf
    $env:VGGT_MV_SUPPORT_REGION_MODE = $PrecomputeMvSupportRegionMode
    $env:VGGT_MV_SUPPORT_FG_MASK_SOURCE = $PrecomputeMvSupportFgMaskSource
    $env:VGGT_MV_SUPPORT_FG_ERODE_PX = [string]$PrecomputeMvSupportFgErodePx
    $env:VGGT_MV_SUPPORT_FG_PRESERVE_PX = [string]$PrecomputeMvSupportFgPreservePx
    $env:VGGT_PRECOMPUTE_ARGS_EXTRA = ""

    $savedNoOutputTimeout = [int]$ModalRunNoOutputTimeoutSec
    $precomputeNoOutputTimeoutEffective = [int]$savedNoOutputTimeout
    if ([int]$PrecomputeNoOutputTimeoutSec -gt 0) {
        $precomputeNoOutputTimeoutEffective = [Math]::Max(300, [int]$PrecomputeNoOutputTimeoutSec)
    }
    try {
        $ModalRunNoOutputTimeoutSec = $precomputeNoOutputTimeoutEffective
        Write-Host "[lr-sweep] precompute no-output policy timeout_sec=$ModalRunNoOutputTimeoutSec base_timeout_sec=$savedNoOutputTimeout precompute_target_sec=$PrecomputeNoOutputTimeoutSec"
        $pcRun = Invoke-ModalRun `
            -ScriptPath "modal_run_train.py" `
            -TimeoutSec $ModalRunTimeoutSec `
            -NoOutputMaxRetries $ModalRunNoOutputMaxRetries `
            -AllowQuietNoOutputBypass $false
    } finally {
        $ModalRunNoOutputTimeoutSec = $savedNoOutputTimeout
    }
    $pcOutput = @($pcRun.Output)
    $pcRc = [int]$pcRun.ExitCode
    $pcLogLocal = "logs/modal_phase5/vggt_ft_lr_${safeLr}_$timestamp.precompute.log"
    $pcOutput | Tee-Object -FilePath $pcLogLocal | Out-Null
    $pcFailReason = ""
    if ($pcRc -ne 0) {
        $pcFailReason = Resolve-ModalRunFailureReason `
            -RunResult $pcRun `
            -Lines $pcOutput `
            -DefaultReason "precompute from finetuned ckpt failed"
        $pointmapSourcePrimaryNorm = [string]$pointmapSourcePrimary
        if (-not [string]::IsNullOrWhiteSpace($pointmapSourcePrimaryNorm)) {
            $pointmapSourcePrimaryNorm = $pointmapSourcePrimaryNorm.Trim().ToLowerInvariant()
        }
        $fallbackSource = [string]$PrecomputeFallbackPointmapSource
        if ([string]::IsNullOrWhiteSpace($fallbackSource) -or
            ((-not [string]::IsNullOrWhiteSpace($pointmapSourcePrimaryNorm)) -and
             ($fallbackSource.Trim().ToLowerInvariant() -eq $pointmapSourcePrimaryNorm))) {
            if ($pointmapSourcePrimaryNorm -eq "point_head") {
                $fallbackSource = "depth_unproject"
            } else {
                $fallbackSource = "point_head"
            }
        }
        $fallbackSourceNorm = [string]$fallbackSource
        if (-not [string]::IsNullOrWhiteSpace($fallbackSourceNorm)) {
            $fallbackSourceNorm = $fallbackSourceNorm.Trim().ToLowerInvariant()
        }
        $isPrecomputeNoOutputFailure = [regex]::IsMatch([string]$pcFailReason, "(?i)(?:heartbeat_stall_timeout|no_output_timeout)_\d+s")
        $precomputeTimeoutHit = [bool]$isPrecomputeNoOutputFailure
        $fallbackEnabled = [bool]$EnablePrecomputePointmapFallbackOnNoOutput
        if ((-not $fallbackEnabled) -and
            ($pointmapSourcePrimaryNorm -eq "point_head") -and
            $isPrecomputeNoOutputFailure) {
            $fallbackEnabled = $true
            Write-Host "[lr-sweep] enforce precompute fallback for point_head no-output"
        }
        $canFallbackPrecompute = $fallbackEnabled -and
            (-not [string]::IsNullOrWhiteSpace($fallbackSourceNorm)) -and
            (-not [string]::IsNullOrWhiteSpace($pointmapSourcePrimaryNorm)) -and
            ($fallbackSourceNorm -ne $pointmapSourcePrimaryNorm) -and
            $isPrecomputeNoOutputFailure
        if ($canFallbackPrecompute) {
            Write-Host "[lr-sweep] precompute fallback trigger=$pcFailReason source=$pointmapSourcePrimary -> $fallbackSource"
            $env:VGGT_POINTMAP_SOURCE = $fallbackSource
            $savedNoOutputTimeoutFb = [int]$ModalRunNoOutputTimeoutSec
            $fallbackTimeoutBase = [Math]::Max(300, [int]$PrecomputeFallbackNoOutputTimeoutSec)
            $fallbackNoOutputTimeoutSec = [Math]::Max(
                [int]$savedNoOutputTimeoutFb,
                [int]$fallbackTimeoutBase
            )
            $pcRunFb = $null
            try {
                $ModalRunNoOutputTimeoutSec = $fallbackNoOutputTimeoutSec
                Write-Host "[lr-sweep] precompute fallback no-output policy timeout_sec=$ModalRunNoOutputTimeoutSec base_timeout_sec=$savedNoOutputTimeoutFb fallback_target_sec=$fallbackTimeoutBase primary_target_sec=$PrecomputeNoOutputTimeoutSec"
                $pcRunFb = Invoke-ModalRun `
                    -ScriptPath "modal_run_train.py" `
                    -TimeoutSec $ModalRunTimeoutSec `
                    -NoOutputMaxRetries $ModalRunNoOutputMaxRetries `
                    -AllowQuietNoOutputBypass $false
            } finally {
                $ModalRunNoOutputTimeoutSec = $savedNoOutputTimeoutFb
            }
            $pcOutputFb = @($pcRunFb.Output)
            $pcRcFb = [int]$pcRunFb.ExitCode
            $pcLogLocalFb = "logs/modal_phase5/vggt_ft_lr_${safeLr}_$timestamp.precompute_fallback.log"
            $pcOutputFb | Tee-Object -FilePath $pcLogLocalFb | Out-Null
            if ($pcRcFb -eq 0) {
                $pcRun = $pcRunFb
                $pcOutput = $pcOutputFb
                $pcRc = 0
                $pointmapSourceResolved = $fallbackSource
                $precomputeFallbackUsed = $true
                Write-Host "[lr-sweep] precompute fallback succeeded source=$fallbackSource"
            } else {
                $pcFailReasonFb = Resolve-ModalRunFailureReason `
                    -RunResult $pcRunFb `
                    -Lines $pcOutputFb `
                    -DefaultReason "precompute fallback failed"
                $fallbackNoOutputFailure = [regex]::IsMatch([string]$pcFailReasonFb, "(?i)(?:heartbeat_stall_timeout|no_output_timeout)_\d+s")
                if ($isPrecomputeNoOutputFailure -and $fallbackNoOutputFailure) {
                    $pcFailReason = "precompute_dual_source_no_output_exhausted(primary=$pointmapSourcePrimaryNorm fallback=$fallbackSourceNorm reason=$pcFailReasonFb)"
                } else {
                    $pcFailReason = "$pcFailReason; fallback($fallbackSource)=$pcFailReasonFb"
                }
                $env:VGGT_POINTMAP_SOURCE = $pointmapSourcePrimary
                $pointmapSourceResolved = $pointmapSourcePrimary
                Write-Host "[lr-sweep] precompute fallback failed source=$fallbackSource reason=$pcFailReasonFb"
            }
        } elseif ($isPrecomputeNoOutputFailure) {
            Write-Host "[lr-sweep] precompute fallback skipped source=$pointmapSourcePrimary fallback=$fallbackSource enabled=$fallbackEnabled same_source=$($fallbackSourceNorm -eq $pointmapSourcePrimaryNorm)"
        }
    }
    if ($pcRc -ne 0) {
        $sweepRows += [pscustomobject]@{
            stage = "short"
            label = $label
            lr = $lr
            freeze_mode = $FreezeMode
            depth_scale_align = $DepthScaleAlign
            geom_subdir = $geomOut
            ft_ckpt = $ftModelPath
            status = "error"
            reason = $pcFailReason
            pointmap_source_requested = $pointmapSourceRequested
            pointmap_source_resolved = $pointmapSourceResolved
            precompute_mv_support_on = $PrecomputeMvSupportOn
            precompute_mv_support_region_mode = $PrecomputeMvSupportRegionMode
            precompute_mv_support_fg_mask_source = $PrecomputeMvSupportFgMaskSource
            precompute_mv_support_fg_erode_px = [string]$PrecomputeMvSupportFgErodePx
            precompute_mv_support_fg_preserve_px = [string]$PrecomputeMvSupportFgPreservePx
            point_target_mode = $PointTargetMode
            point_target_blend_by_mv_support = $PointTargetBlendByMvSupport
            point_target_blend_mv_region_mode = $PointTargetBlendMvRegionMode
            point_mv_depth_region_mode = $PointMvDepthRegionMode
            precompute_fallback_used = [bool]$precomputeFallbackUsed
            precompute_timeout_hit = [bool]$precomputeTimeoutHit
            candidate_invalid_reason = (Resolve-FtCandidateInvalidReason -Status "error" -Reason $pcFailReason)
        }
        Merge-FtMetricsSnapshotIntoRow -Row $sweepRows[$sweepRows.Count - 1] -Snapshot $ftMetricsSnapshot
        continue
    }

    $geomIntegrity = Get-PrecomputeGeomIntegrity -SeqNamesRaw $SeqNames -GeomSubdir $geomOut
    if (-not [bool]$geomIntegrity.ok) {
        $pcFailReason = [string]$geomIntegrity.reason
        Write-Host "[lr-sweep] reject precompute output before eval: $pcFailReason"
        $sweepRows += [pscustomobject]@{
            stage = "short"
            label = $label
            lr = $lr
            freeze_mode = $FreezeMode
            depth_scale_align = $DepthScaleAlign
            geom_subdir = $geomOut
            ft_ckpt = $ftModelPathResolved
            status = "error"
            reason = $pcFailReason
            pointmap_source_requested = $pointmapSourceRequested
            pointmap_source_resolved = $pointmapSourceResolved
            precompute_mv_support_on = $PrecomputeMvSupportOn
            precompute_mv_support_region_mode = $PrecomputeMvSupportRegionMode
            precompute_mv_support_fg_mask_source = $PrecomputeMvSupportFgMaskSource
            precompute_mv_support_fg_erode_px = [string]$PrecomputeMvSupportFgErodePx
            precompute_mv_support_fg_preserve_px = [string]$PrecomputeMvSupportFgPreservePx
            point_target_mode = $PointTargetMode
            point_target_blend_by_mv_support = $PointTargetBlendByMvSupport
            point_target_blend_mv_region_mode = $PointTargetBlendMvRegionMode
            point_mv_depth_region_mode = $PointMvDepthRegionMode
            precompute_fallback_used = [bool]$precomputeFallbackUsed
            precompute_timeout_hit = [bool]$precomputeTimeoutHit
            candidate_invalid_reason = [string]$geomIntegrity.candidate_invalid_reason
        }
        Merge-FtMetricsSnapshotIntoRow -Row $sweepRows[$sweepRows.Count - 1] -Snapshot $ftMetricsSnapshot
        continue
    }

    $precomputeSupportStats = Get-PrecomputeSupportStats -SeqNamesRaw $SeqNames -GeomSubdir $geomOut

    $shortInfos += [pscustomobject]@{
        label = $label
        lr = $lr
        geom_subdir = $geomOut
        ft_ckpt = $ftModelPathResolved
        pointmap_source_requested = $pointmapSourceRequested
        pointmap_source_resolved = $pointmapSourceResolved
        precompute_fallback_used = [bool]$precomputeFallbackUsed
        precompute_timeout_hit = [bool]$precomputeTimeoutHit
        precompute_mv_support_region_mode = $PrecomputeMvSupportRegionMode
        precompute_mv_support_fg_mask_source = $PrecomputeMvSupportFgMaskSource
        precompute_mv_support_fg_erode_px = [string]$PrecomputeMvSupportFgErodePx
        precompute_mv_support_fg_preserve_px = [string]$PrecomputeMvSupportFgPreservePx
        mv_support_raw_mean = $precomputeSupportStats.mv_support_raw_mean
        mv_support_valid_ratio = $precomputeSupportStats.mv_support_valid_ratio
        mv_support_fg_valid_ratio = $precomputeSupportStats.mv_support_fg_valid_ratio
        mv_support_bg_valid_ratio = $precomputeSupportStats.mv_support_bg_valid_ratio
        mv_support_pair_count_eff = $precomputeSupportStats.mv_support_pair_count_eff
        mv_support_conf_mean = $precomputeSupportStats.mv_support_conf_mean
        mv_support_nan_ratio = $precomputeSupportStats.mv_support_nan_ratio
        depth_conf_delta_mean = $precomputeSupportStats.depth_conf_delta_mean
        mv_support_fg_mean = $precomputeSupportStats.mv_support_fg_mean
        mv_support_bg_mean = $precomputeSupportStats.mv_support_bg_mean
        depth_conf_delta_fg_mean = $precomputeSupportStats.depth_conf_delta_fg_mean
        depth_conf_delta_bg_mean = $precomputeSupportStats.depth_conf_delta_bg_mean
        depth_conf_fg_preserved_active = $precomputeSupportStats.depth_conf_fg_preserved_active
        depth_conf_fg_preserve_px = $precomputeSupportStats.depth_conf_fg_preserve_px
        depth_conf_fg_exact_ratio = $precomputeSupportStats.depth_conf_fg_exact_ratio
        depth_conf_fg_preserve_ratio = $precomputeSupportStats.depth_conf_fg_preserve_ratio
        depth_conf_fg_raw_mean = $precomputeSupportStats.depth_conf_fg_raw_mean
        depth_conf_fg_after_support_mean = $precomputeSupportStats.depth_conf_fg_after_support_mean
        depth_conf_fg_final_mean = $precomputeSupportStats.depth_conf_fg_final_mean
        mv_support_generation_region_mode = $precomputeSupportStats.mv_support_generation_region_mode
        mv_support_generation_fg_mask_source = $precomputeSupportStats.mv_support_generation_fg_mask_source
    }
    $sweepRows += [pscustomobject]@{
        stage = "short"
        label = $label
        lr = $lr
        freeze_mode = $FreezeMode
        depth_scale_align = $DepthScaleAlign
        geom_subdir = $geomOut
        ft_ckpt = $ftModelPathResolved
        status = $(if ($usedResumeFallbackForShort) { "ok_fallback" } else { "ok" })
        reason = $shortCkptSourceNote
        pointmap_source_requested = $pointmapSourceRequested
        pointmap_source_resolved = $pointmapSourceResolved
        precompute_mv_support_on = $PrecomputeMvSupportOn
        precompute_mv_support_region_mode = $PrecomputeMvSupportRegionMode
        precompute_mv_support_fg_mask_source = $PrecomputeMvSupportFgMaskSource
        precompute_mv_support_fg_erode_px = [string]$PrecomputeMvSupportFgErodePx
        precompute_mv_support_fg_preserve_px = [string]$PrecomputeMvSupportFgPreservePx
        mv_support_raw_mean = $precomputeSupportStats.mv_support_raw_mean
        mv_support_valid_ratio = $precomputeSupportStats.mv_support_valid_ratio
        mv_support_fg_valid_ratio = $precomputeSupportStats.mv_support_fg_valid_ratio
        mv_support_bg_valid_ratio = $precomputeSupportStats.mv_support_bg_valid_ratio
        mv_support_pair_count_eff = $precomputeSupportStats.mv_support_pair_count_eff
        mv_support_conf_mean = $precomputeSupportStats.mv_support_conf_mean
        mv_support_nan_ratio = $precomputeSupportStats.mv_support_nan_ratio
        depth_conf_delta_mean = $precomputeSupportStats.depth_conf_delta_mean
        mv_support_fg_mean = $precomputeSupportStats.mv_support_fg_mean
        mv_support_bg_mean = $precomputeSupportStats.mv_support_bg_mean
        depth_conf_delta_fg_mean = $precomputeSupportStats.depth_conf_delta_fg_mean
        depth_conf_delta_bg_mean = $precomputeSupportStats.depth_conf_delta_bg_mean
        depth_conf_fg_preserved_active = $precomputeSupportStats.depth_conf_fg_preserved_active
        depth_conf_fg_preserve_px = $precomputeSupportStats.depth_conf_fg_preserve_px
        depth_conf_fg_exact_ratio = $precomputeSupportStats.depth_conf_fg_exact_ratio
        depth_conf_fg_preserve_ratio = $precomputeSupportStats.depth_conf_fg_preserve_ratio
        depth_conf_fg_raw_mean = $precomputeSupportStats.depth_conf_fg_raw_mean
        depth_conf_fg_after_support_mean = $precomputeSupportStats.depth_conf_fg_after_support_mean
        depth_conf_fg_final_mean = $precomputeSupportStats.depth_conf_fg_final_mean
        mv_support_generation_region_mode = $precomputeSupportStats.mv_support_generation_region_mode
        mv_support_generation_fg_mask_source = $precomputeSupportStats.mv_support_generation_fg_mask_source
        point_target_mode = $PointTargetMode
        point_target_blend_by_mv_support = $PointTargetBlendByMvSupport
        point_target_blend_mv_region_mode = $PointTargetBlendMvRegionMode
        point_mv_depth_region_mode = $PointMvDepthRegionMode
        precompute_fallback_used = [bool]$precomputeFallbackUsed
        precompute_timeout_hit = [bool]$precomputeTimeoutHit
        candidate_invalid_reason = ""
    }
    Merge-FtMetricsSnapshotIntoRow -Row $sweepRows[$sweepRows.Count - 1] -Snapshot $ftMetricsSnapshot
}

if ($shortInfos.Count -eq 0) {
    $sweepCsv = "logs/modal_phase5/vggt_ft_sweep_$timestamp.csv"
    $sweepLatest = "logs/modal_phase5/vggt_ft_sweep_latest.csv"
    Finalize-SweepRows -Rows $sweepRows
    $sweepRows | Export-Csv $sweepCsv -NoTypeInformation -Encoding UTF8
    $sweepRows | Export-Csv $sweepLatest -NoTypeInformation -Encoding UTF8
    Write-Host "[lr-sweep] no short finetune candidate succeeded."
    exit 3
}

$geomCand = ($shortInfos | ForEach-Object { "$($_.geom_subdir):$($_.label)" }) -join ";"
Write-Host "[lr-sweep] evaluate short candidates..."
$evalArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts/eval_geom_candidates.ps1",
    "-CodeDir", $CodeDir,
    "-SeqNames", $SeqNames,
    "-CamNames", $CamNames,
    "-GeomCandidates", $geomCand,
    "-NumSamples", [string]$EvalNumSamples,
    "-MinPSNR", [string]$MinPSNR,
    "-MinSSIM", [string]$MinSSIM,
    "-MaxWL1", [string]$MaxWL1,
    "-OutTag", "lr_sweep_short_$timestamp"
)
if (-not [string]::IsNullOrWhiteSpace($EvalInferArgsExtra)) {
    $evalArgs += @("-InferArgsExtra", $EvalInferArgsExtra)
}
if (-not [string]::IsNullOrWhiteSpace($DecoderCkpt)) {
    $evalArgs += @("-DecoderCkpt", $DecoderCkpt)
}
$evalTimeoutSec = [Math]::Max(600, [int]$ModalRunTimeoutSec)
$evalPollSec = [Math]::Max(5, [int]$ModalRunPollSec)
$evalNoOutputTimeoutSec = [Math]::Max(0, [int]$EvalNoOutputTimeoutSec)
$evalStdoutFile = [System.IO.Path]::GetTempFileName()
$evalStderrFile = [System.IO.Path]::GetTempFileName()
$evalStartedAt = Get-Date
$evalProc = $null
$evalRc = -1
$evalTimedOut = $false
$evalTimeoutNote = "timeout"
$evalOutput = @()
try {
    $evalProc = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $evalArgs `
        -WorkingDirectory $CodeDir `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardOutput $evalStdoutFile `
        -RedirectStandardError $evalStderrFile

    Write-ModalRunHeartbeat `
        -State "running" `
        -ScriptPath "scripts/eval_geom_candidates.ps1" `
        -Attempt 1 `
        -MaxRetries 1 `
        -TimeoutSec $evalTimeoutSec `
        -ProcId $evalProc.Id `
        -StartedAt $evalStartedAt `
        -StdoutFile $evalStdoutFile `
        -StderrFile $evalStderrFile `
        -Note "post_eval launched"

    $evalExited = $false
    $evalPollTick = 0
    $evalLastIoChangeAt = $evalStartedAt
    $evalLastStdoutLen = [int64]0
    $evalLastStderrLen = [int64]0
    while (-not $evalExited) {
        $evalExited = $evalProc.WaitForExit($evalPollSec * 1000)
        if ($evalExited) { break }
        $evalPollTick += 1
        $elapsedEval = (Get-Date) - $evalStartedAt
        if ($elapsedEval.TotalSeconds -ge $evalTimeoutSec) {
            $evalTimedOut = $true
            break
        }
        $evalStdoutLen = [int64]0
        $evalStderrLen = [int64]0
        try {
            if (Test-Path $evalStdoutFile) { $evalStdoutLen = [int64](Get-Item $evalStdoutFile).Length }
            if (Test-Path $evalStderrFile) { $evalStderrLen = [int64](Get-Item $evalStderrFile).Length }
        } catch {}
        if (($evalStdoutLen -ne $evalLastStdoutLen) -or ($evalStderrLen -ne $evalLastStderrLen)) {
            $evalLastIoChangeAt = Get-Date
            $evalLastStdoutLen = $evalStdoutLen
            $evalLastStderrLen = $evalStderrLen
        }
        if ($evalNoOutputTimeoutSec -gt 0) {
            $evalNoIoElapsedSec = ((Get-Date) - $evalLastIoChangeAt).TotalSeconds
            if ($evalNoIoElapsedSec -ge $evalNoOutputTimeoutSec) {
                $evalTimedOut = $true
                $evalTimeoutNote = "no_output_timeout_${evalNoOutputTimeoutSec}s"
                Write-Host "[lr-sweep] eval stall detected no_output_elapsed=${evalNoIoElapsedSec}s threshold=${evalNoOutputTimeoutSec}s"
                break
            }
        }
        if (($evalPollTick -eq 1) -or (($evalPollTick % 3) -eq 0)) {
            $elapsedEvalSec = [int][Math]::Round($elapsedEval.TotalSeconds)
            Write-Host "[lr-sweep] eval alive elapsed=${elapsedEvalSec}s timeout=${evalTimeoutSec}s stdout_bytes=${evalStdoutLen} stderr_bytes=${evalStderrLen}"
        }
        Write-ModalRunHeartbeat `
            -State "running" `
            -ScriptPath "scripts/eval_geom_candidates.ps1" `
            -Attempt 1 `
            -MaxRetries 1 `
            -TimeoutSec $evalTimeoutSec `
            -ProcId $evalProc.Id `
            -StartedAt $evalStartedAt `
            -StdoutFile $evalStdoutFile `
            -StderrFile $evalStderrFile `
            -Note "post_eval polling"
    }

    if ($evalTimedOut) {
        try {
            & taskkill /PID $evalProc.Id /T /F > $null 2>&1
        } catch {}
        Start-Sleep -Milliseconds 200
        try {
            Stop-Process -Id $evalProc.Id -Force -ErrorAction SilentlyContinue
        } catch {}
        $evalRc = 124
    } else {
        $evalRc = [int]$evalProc.ExitCode
    }

    if (Test-Path $evalStdoutFile) { $evalOutput += @(Get-Content $evalStdoutFile) }
    if (Test-Path $evalStderrFile) { $evalOutput += @(Get-Content $evalStderrFile) }
    if ($evalTimedOut) {
        $evalOutput += @("[lr-sweep] evaluate short candidates timeout after ${evalTimeoutSec}s ($evalTimeoutNote)")
    }

    $evalLogLocal = "logs/modal_phase5/vggt_ft_lr_${timestamp}.eval_short.log"
    $evalOutput | Tee-Object -FilePath $evalLogLocal | Out-Null

    Write-ModalRunHeartbeat `
        -State $(if ($evalRc -eq 0) { "done" } elseif ($evalTimedOut) { "timeout" } else { "failed" }) `
        -ScriptPath "scripts/eval_geom_candidates.ps1" `
        -Attempt 1 `
        -MaxRetries 1 `
        -TimeoutSec $evalTimeoutSec `
        -ProcId $(if ($evalProc -ne $null) { $evalProc.Id } else { 0 }) `
        -StartedAt $evalStartedAt `
        -StdoutFile $evalStdoutFile `
        -StderrFile $evalStderrFile `
        -ExitCode $evalRc `
        -Note "post_eval finished"
} finally {
    Remove-Item $evalStdoutFile -ErrorAction SilentlyContinue
    Remove-Item $evalStderrFile -ErrorAction SilentlyContinue
}
if ($evalRc -ne 0) {
    throw "evaluate short candidates failed (rc=$evalRc)"
}

$evalCsv = "logs/modal_phase5/baseline_compare_latest.csv"
$evalRows = @()
if (Test-Path $evalCsv) {
    $evalRows = Import-Csv $evalCsv
}

$bestRow = $null
$okRows = @($evalRows | Where-Object { $_.status -eq "ok" -and $_.pass -eq "True" })
if ($okRows.Count -gt 0) {
    $bestRow = $okRows | Sort-Object { [double]$_.mean_PSNR } -Descending | Select-Object -First 1
}

if ($RunLongOnImprove -and ($bestRow -ne $null)) {
    $bestLabel = [string]$bestRow.label
    $bestInfo = $shortInfos | Where-Object { $_.label -eq $bestLabel } | Select-Object -First 1
    if ($bestInfo -ne $null) {
        $bestSafe = Sanitize($bestLabel)
        $longLabel = "${bestLabel}_long"
        $longGeom = "vggt_geom_ft_${bestSafe}_long_$timestamp"
        $longCkptDir = "/mnt/out/vggt/finetune/${bestSafe}_long_$timestamp/ckpt"
        $longLogDir = "/mnt/out/vggt/finetune/${bestSafe}_long_$timestamp/logs"
        $longModel = "$longCkptDir/model_ft_zju.pt"
        $longPointmapSourceRequested = if ($bestInfo.PSObject.Properties["pointmap_source_requested"]) { [string]$bestInfo.pointmap_source_requested } else { [string]$PointmapSource }
        $longPointmapSourceResolved = if ($bestInfo.PSObject.Properties["pointmap_source_resolved"]) { [string]$bestInfo.pointmap_source_resolved } else { "" }
        if ([string]::IsNullOrWhiteSpace($longPointmapSourceResolved)) {
            $longPointmapSourceResolved = [string]$PointmapSource
        }
        $longPrecomputeFallbackUsed = if ($bestInfo.PSObject.Properties["precompute_fallback_used"]) { [bool]$bestInfo.precompute_fallback_used } else { $false }
        $longPrecomputeTimeoutHit = $false

        $maxFramesLongArg = $MaxFramesLong
        if ($maxFramesLongArg -le 0) { $maxFramesLongArg = $MaxFramesShort }

        $argsLong = @(
            "--epochs=$EpochsLong",
            "--max_frames=$maxFramesLongArg",
            "--lr=$($bestInfo.lr)",
            "--freeze_mode=$FreezeMode",
            "--depth_scale_align=$DepthScaleAlign",
            "--lambda_depth=$LambdaDepth",
            "--lambda_point=$LambdaPoint",
            "--lambda_point_reproj=$LambdaPointReproj",
            "--lambda_point_normal_consis=$LambdaPointNormalConsis",
            "--lambda_point_mv_depth=$LambdaPointMvDepth",
            "--lambda_point_mv_mask=$LambdaPointMvMask",
            "--lambda_conf=$LambdaConf",
            "--lambda_conf_warmup_steps=$LambdaConfWarmupSteps",
            "--lambda_geom_cons=$LambdaGeomCons",
            "--lambda_cam=$LambdaCam",
            "--lambda_cam_warmup_steps=$LambdaCamWarmupSteps",
            "--cam_rot_weight=$CamRotWeight",
            "--cam_fov_weight=$CamFovWeight",
            "--cam_warmup_steps=$CamWarmupSteps",
            "--jitter=$Jitter",
            "--noise_std=$NoiseStd",
            "--robust_l1_eps=$RobustL1Eps",
            "--conf_weight_thr=$ConfWeightThr",
            "--conf_weight_gamma=$ConfWeightGamma",
            "--conf_weight_per_view_quantile=$ConfWeightPerViewQuantile",
            "--conf_weight_per_view_min_valid=$ConfWeightPerViewMinValid",
            "--gram_dyn_enable=$GramDynEnable",
            "--gram_dyn_layer_idx=$GramDynLayerIdx",
            "--gram_dyn_quantile=$GramDynQuantile",
            "--gram_dyn_weight_floor=$GramDynWeightFloor",
            "--gram_dyn_warmup_steps=$GramDynWarmupSteps",
            "--dyn_proxy_enable=$DynProxyEnable",
            "--dyn_proxy_mode=$DynProxyMode",
            "--dyn_proxy_use_gram=$DynProxyUseGram",
            "--dyn_proxy_use_support=$DynProxyUseSupport",
            "--dyn_proxy_floor=$DynProxyFloor",
            "--dyn_proxy_warmup_steps=$DynProxyWarmupSteps",
            "--point_cons_tau=$PointConsTau",
            "--point_cons_weight_floor=$PointConsWeightFloor",
            "--point_cons_clip_min_qv=$PointConsClipMinQv",
            "--point_cons_quantile=$PointConsQuantile",
            "--point_cons_focus=$PointConsFocus",
            "--point_residual_quantile=$PointResidualQuantile",
            "--point_residual_focus=$PointResidualFocus",
            "--point_residual_boost=$PointResidualBoost",
            "--point_residual_boost_cap=$PointResidualBoostCap",
            "--point_target_mode=$PointTargetMode",
            "--point_target_blend_alpha=$PointTargetBlendAlpha",
            "--point_target_blend_alpha_min=$PointTargetBlendAlphaMin",
            "--point_target_blend_alpha_max=$PointTargetBlendAlphaMax",
            "--point_target_blend_rel_gain=$PointTargetBlendRelGain",
            "--point_target_blend_mv_gain=$PointTargetBlendMvGain",
            "--point_target_blend_by_reliability=$PointTargetBlendByReliability",
            "--point_target_blend_by_mv_support=$PointTargetBlendByMvSupport",
            "--point_target_blend_mv_region_mode=$PointTargetBlendMvRegionMode",
            "--point_target_blend_mv_policy=$PointTargetBlendMvPolicy",
            "--point_target_consensus_alpha_floor=$PointTargetConsensusAlphaFloor",
            "--target_point_frame=$TargetPointFrame",
            "--pred_point_frame=$PredPointFrame",
            "--use_fg_mask=$UseFgMask",
            "--fg_mask_source=$FgMaskSource",
            "--fg_mask_erode_px=$FgMaskErodePx",
            "--point_loss_fg_erode_px=$PointLossFgErodePx",
            "--fg_supervision_boost=$FgSupervisionBoost",
            "--fg_supervision_bg_floor=$FgSupervisionBgFloor",
            "--fg_supervision_region_mode=$FgSupervisionRegionMode",
            "--fg_supervision_region_erode_px=$FgSupervisionRegionErodePx",
            "--lambda_fg_conf_presence=$LambdaFgConfPresence",
            "--fg_conf_presence_target_ratio=$FgConfPresenceTargetRatio",
            "--lambda_fg_structure_depth_edge=$LambdaFgStructureDepthEdge",
            "--fg_structure_bbox_margin_px=$FgStructureBboxMarginPx",
            "--fg_structure_bbox_min_side_px=$FgStructureBboxMinSidePx",
            "--fg_structure_region_mode=$FgStructureRegionMode",
            "--fg_structure_region_erode_px=$FgStructureRegionErodePx",
            "--fg_structure_depth_edge_warmup_steps=$FgStructureDepthEdgeWarmupSteps",
            "--fg_structure_boundary_probe_px=$FgStructureBoundaryProbePx",
            "--fg_structure_edge_support_mode=$FgStructureEdgeSupportMode",
            "--fg_structure_edge_support_quantile=$FgStructureEdgeSupportQuantile",
            "--fg_structure_edge_support_min_px=$FgStructureEdgeSupportMinPx",
            "--fg_structure_edge_weight_mode=$FgStructureEdgeWeightMode",
            "--fg_structure_boundary_falloff_px=$FgStructureBoundaryFalloffPx",
            "--fg_structure_component_bias_mode=$FgStructureComponentBiasMode",
            "--fg_structure_component_bias_threshold_ratio=$FgStructureComponentBiasThresholdRatio",
            "--fg_structure_component_bias_other_scale=$FgStructureComponentBiasOtherScale",
            "--fg_structure_front_depth_bias_mode=$FgStructureFrontDepthBiasMode",
            "--fg_structure_front_depth_bias_tau=$FgStructureFrontDepthBiasTau",
            "--fg_structure_front_depth_bias_center_quantile=$FgStructureFrontDepthBiasCenterQuantile",
            "--lambda_point_mv_outside_ring=$LambdaPointMvOutsideRing",
            "--point_mv_outside_ring_px=$PointMvOutsideRingPx",
            "--supervision_weight_mode=$SupervisionWeightMode",
            "--supervision_weight_mix_alpha=$SupervisionWeightMixAlpha",
            "--point_reproj_warmup_steps=$PointReprojWarmupSteps",
            "--point_reproj_clamp_px=$PointReprojClampPx",
            "--point_mv_consistency=$PointMvConsistency",
            "--point_mv_tol_abs=$PointMvTolAbs",
            "--point_mv_tol_rel=$PointMvTolRel",
            "--point_mv_weight_floor=$PointMvWeightFloor",
            "--point_mv_stride=$PointMvStride",
            "--point_mv_depth_max_pairs=$PointMvDepthMaxPairs",
            "--point_mv_depth_pair_mode=$PointMvDepthPairMode",
            "--point_mv_depth_warmup_steps=$PointMvDepthWarmupSteps",
            "--point_mv_depth_region_mode=$PointMvDepthRegionMode",
            "--point_mv_mask_warmup_steps=$PointMvMaskWarmupSteps",
            "--point_mv_depth_inlier_only=$PointMvDepthInlierOnly",
            "--point_mv_depth_err_quantile=$PointMvDepthErrQuantile",
            "--point_mv_depth_outlier_boost=$PointMvDepthOutlierBoost",
            "--point_mv_depth_outlier_cap=$PointMvDepthOutlierCap",
            "--point_mv_depth_tgt_valid_mode=$PointMvDepthTgtValidMode",
            "--point_mv_depth_tgt_valid_floor=$PointMvDepthTgtValidFloor",
            "--point_mv_depth_min_tgt_valid_ratio=$PointMvDepthMinTgtValidRatio",
            "--point_mv_mask_min_tgt_fg_ratio=$PointMvMaskMinTgtFgRatio",
            "--point_mv_mask_hit_thr=$PointMvMaskHitThr",
            "--point_mv_mask_soft_blur_px=$PointMvMaskSoftBlurPx",
            "--point_mv_mask_soft_blur_iters=$PointMvMaskSoftBlurIters",
            "--point_mv_mask_soft_mix=$PointMvMaskSoftMix",
            "--point_mv_mask_soft_hit_thr=$PointMvMaskSoftHitThr",
            "--point_mv_depth_tgt_valid_scale_mode=$PointMvDepthTgtValidScaleMode",
            "--point_mv_depth_tgt_valid_scale_thr=$PointMvDepthTgtValidScaleThr",
            "--point_mv_depth_adapt_mode=$PointMvDepthAdaptMode",
            "--point_mv_depth_adapt_target_valid=$PointMvDepthAdaptTargetValid",
            "--point_mv_depth_adapt_min_scale=$PointMvDepthAdaptMinScale",
            "--point_mv_depth_adapt_max_scale=$PointMvDepthAdaptMaxScale",
            "--point_support_mode=$PointSupportMode",
            "--point_support_floor=$PointSupportFloor",
            "--point_mv_depth_support_mode=$PointMvDepthSupportMode",
            "--point_mv_depth_support_floor=$PointMvDepthSupportFloor",
            "--point_mv_mask_support_mode=$PointMvMaskSupportMode",
            "--point_mv_mask_support_floor=$PointMvMaskSupportFloor",
            "--point_mv_depth_fg_erode_px=$PointMvDepthFgErodePx",
            "--point_loss_scale_depth_unproject=$PointLossScaleDepthUnproject",
            "--point_warmup_steps=$PointWarmupSteps",
            "--point_normal_consis_warmup_steps=$PointNormalConsisWarmupSteps",
            "--lr_backbone_scale=$LrBackboneScale",
            "--lr_head_scale=$LrHeadScale",
            "--lr_camera_scale=$LrCameraScale",
            "--grad_clip=$GradClip",
            "--geom_subdir=$PseudoGeomSubdir",
            "--resume_ckpt=$($bestInfo.ft_ckpt)",
            "--log_dir=$longLogDir",
            "--ckpt_dir=$longCkptDir",
            "--eval_every_steps=$EvalEverySteps",
            "--debug_metrics_every_steps=$DebugMetricsEverySteps",
            "--debug_vis_every_steps=$DebugVisEverySteps",
            "--debug_vis_max_steps=$DebugVisMaxSteps",
            "--debug_vis_views=$DebugVisViews",
            "--early_stop_patience=$EarlyStopPatience",
            "--min_improve=$MinImprove",
            "--max_steps_per_epoch=$MaxStepsPerEpoch"
        ) -join " "
        if (-not [string]::IsNullOrWhiteSpace($DebugVisDir)) {
            $argsLong = "$argsLong --debug_vis_dir=$DebugVisDir"
        }

        Write-Host "[lr-sweep] long finetune from best short label=$bestLabel"
        $env:VGGT_CODE_DIR = $CodeDir
        $env:VGGT_MODE = "precompute"
        $env:VGGT_SEQ_NAMES = $SeqNames
        $env:VGGT_GEOM_SUBDIR = $PseudoGeomSubdir
        $env:VGGT_CAM_NAMES = $CamNames
        $env:VGGT_MAX_FRAMES = [string]$maxFramesLongArg
        $env:VGGT_PRECOMPUTE_SCRIPT = "finetune_vggt_pseudo.py"
        $env:VGGT_PRECOMPUTE_CKPT = $PretrainedCkpt
        $env:VGGT_PRECOMPUTE_ARGS_EXTRA = $argsLong
        Remove-Item Env:VGGT_PROFILE -ErrorAction SilentlyContinue

        $ftLongRun = Invoke-ModalRun `
            -ScriptPath "modal_run_train.py" `
            -TimeoutSec $ModalRunTimeoutSec `
            -NoOutputMaxRetries $ModalRunNoOutputMaxRetries
        $ftLongOutput = @($ftLongRun.Output)
        $ftLongRc = [int]$ftLongRun.ExitCode
        $ftLongLog = "logs/modal_phase5/vggt_ft_${bestSafe}_$timestamp.long_finetune.log"
        $ftLongOutput | Tee-Object -FilePath $ftLongLog | Out-Null
        if ($ftLongRc -eq 0) {
            $longModelResolved = Resolve-ModalCheckpointPath -PrimaryPathInMntOut $longModel -TimeoutSec $CkptWaitTimeoutSec -PollSec $CkptWaitPollSec -AllowTmpFallback
            if ([string]::IsNullOrWhiteSpace($longModelResolved)) {
                $sweepRows += [pscustomobject]@{
                    stage = "long"
                    label = $longLabel
                    lr = [string]$bestInfo.lr
                    freeze_mode = $FreezeMode
                    depth_scale_align = $DepthScaleAlign
                    geom_subdir = $longGeom
                    ft_ckpt = $longModel
                    status = "error"
                    reason = "long finetune checkpoint missing on volume: $longModel"
                    pointmap_source_requested = $longPointmapSourceRequested
                    pointmap_source_resolved = $longPointmapSourceResolved
                    precompute_fallback_used = [bool]$longPrecomputeFallbackUsed
                    precompute_timeout_hit = [bool]$longPrecomputeTimeoutHit
                }
                continue
            }
            Fetch-FtDebugArtifacts `
                -LogDirInMntOut $longLogDir `
                -LabelTag "${bestSafe}_${timestamp}_long" `
                -VisCount $DownloadFtDebugVisCount `
                -VisStepsRaw $DownloadFtDebugVisSteps
            $env:VGGT_MODE = "precompute"
            $env:VGGT_PRECOMPUTE_SCRIPT = "precompute_zju_vggt_geom.py"
            $env:VGGT_PRECOMPUTE_CKPT = $longModelResolved
            $env:VGGT_GEOM_SUBDIR = $longGeom
            $env:VGGT_MAX_FRAMES = [string]$maxFramesLongArg
            $env:VGGT_POINTMAP_SOURCE = $longPointmapSourceResolved
            $env:VGGT_POINT_HEAD_FRAME = $PointHeadFrame
            $env:VGGT_UNPROJECT_IMPL = $UnprojectImpl
            $env:VGGT_MV_SUPPORT_ON = $PrecomputeMvSupportOn
            $env:VGGT_MV_SUPPORT_TOL_ABS = [string]$PrecomputeMvSupportTolAbs
            $env:VGGT_MV_SUPPORT_TOL_REL = [string]$PrecomputeMvSupportTolRel
            $env:VGGT_MV_SUPPORT_STRIDE = [string]$PrecomputeMvSupportStride
            $env:VGGT_MV_SUPPORT_MODE = $PrecomputeMvSupportMode
            $env:VGGT_MV_SUPPORT_FLOOR = [string]$PrecomputeMvSupportFloor
            $env:VGGT_MV_SUPPORT_GAMMA = [string]$PrecomputeMvSupportGamma
            $env:VGGT_MV_SUPPORT_CLIP_THR = [string]$PrecomputeMvSupportClipThr
            $env:VGGT_MV_SUPPORT_CLIP_FLOOR = [string]$PrecomputeMvSupportClipFloor
            $env:VGGT_MV_SUPPORT_HARD_THR = [string]$PrecomputeMvSupportHardThr
            $env:VGGT_MV_CONF_VALID_FLOOR = [string]$PrecomputeMvConfValidFloor
            $env:VGGT_MV_SUPPORT_SAVE = $PrecomputeMvSupportSave
            $env:VGGT_MV_SUPPORT_SAVE_RAW_CONF = $PrecomputeMvSupportSaveRawConf
            $env:VGGT_MV_SUPPORT_REGION_MODE = $PrecomputeMvSupportRegionMode
            $env:VGGT_MV_SUPPORT_FG_MASK_SOURCE = $PrecomputeMvSupportFgMaskSource
            $env:VGGT_MV_SUPPORT_FG_ERODE_PX = [string]$PrecomputeMvSupportFgErodePx
            $env:VGGT_MV_SUPPORT_FG_PRESERVE_PX = [string]$PrecomputeMvSupportFgPreservePx
            $env:VGGT_PRECOMPUTE_ARGS_EXTRA = ""

            $pcLongRun = Invoke-ModalRun `
                -ScriptPath "modal_run_train.py" `
                -TimeoutSec $ModalRunTimeoutSec `
                -NoOutputMaxRetries $ModalRunNoOutputMaxRetries `
                -AllowQuietNoOutputBypass $false
            $pcLongOutput = @($pcLongRun.Output)
            $pcLongRc = [int]$pcLongRun.ExitCode
            $pcLongLog = "logs/modal_phase5/vggt_ft_${bestSafe}_$timestamp.long_precompute.log"
            $pcLongOutput | Tee-Object -FilePath $pcLongLog | Out-Null
            $pcLongFailReason = ""
            if ($pcLongRc -ne 0) {
                $pcLongFailReason = Resolve-ModalRunFailureReason `
                    -RunResult $pcLongRun `
                    -Lines $pcLongOutput `
                    -DefaultReason "long precompute failed"
                $longPrecomputeTimeoutHit = [bool]([regex]::IsMatch([string]$pcLongFailReason, "(?i)(?:heartbeat_stall_timeout|no_output_timeout)_\d+s"))
            }
            if ($pcLongRc -eq 0) {
                $geomCand2 = "$($bestInfo.geom_subdir):$bestLabel;$($longGeom):$longLabel"
    $evalArgs2 = @(
                    "-ExecutionPolicy", "Bypass",
                    "-File", "scripts/eval_geom_candidates.ps1",
                    "-CodeDir", $CodeDir,
                    "-SeqNames", $SeqNames,
                    "-CamNames", $CamNames,
                    "-GeomCandidates", $geomCand2,
        "-NumSamples", [string]$EvalNumSamples,
                    "-MinPSNR", [string]$MinPSNR,
                    "-MinSSIM", [string]$MinSSIM,
                    "-MaxWL1", [string]$MaxWL1,
        "-OutTag", "lr_sweep_long_$timestamp"
    )
    if (-not [string]::IsNullOrWhiteSpace($EvalInferArgsExtra)) {
        $evalArgs2 += @("-InferArgsExtra", $EvalInferArgsExtra)
    }
    if (-not [string]::IsNullOrWhiteSpace($DecoderCkpt)) {
        $evalArgs2 += @("-DecoderCkpt", $DecoderCkpt)
    }
                & powershell @evalArgs2

                $sweepRows += [pscustomobject]@{
                    stage = "long"
                    label = $longLabel
                    lr = [string]$bestInfo.lr
                    freeze_mode = $FreezeMode
                    depth_scale_align = $DepthScaleAlign
                    geom_subdir = $longGeom
                    ft_ckpt = $longModelResolved
                    status = "ok"
                    reason = ""
                    pointmap_source_requested = $longPointmapSourceRequested
                    pointmap_source_resolved = $longPointmapSourceResolved
                    precompute_fallback_used = [bool]$longPrecomputeFallbackUsed
                    precompute_timeout_hit = [bool]$longPrecomputeTimeoutHit
                }
            } else {
                $sweepRows += [pscustomobject]@{
                    stage = "long"
                    label = $longLabel
                    lr = [string]$bestInfo.lr
                    freeze_mode = $FreezeMode
                    depth_scale_align = $DepthScaleAlign
                    geom_subdir = $longGeom
                    ft_ckpt = $longModelResolved
                    status = "error"
                    reason = $pcLongFailReason
                    pointmap_source_requested = $longPointmapSourceRequested
                    pointmap_source_resolved = $longPointmapSourceResolved
                    precompute_fallback_used = [bool]$longPrecomputeFallbackUsed
                    precompute_timeout_hit = [bool]$longPrecomputeTimeoutHit
                }
            }
        } else {
            $sweepRows += [pscustomobject]@{
                stage = "long"
                label = $longLabel
                lr = [string]$bestInfo.lr
                freeze_mode = $FreezeMode
                depth_scale_align = $DepthScaleAlign
                geom_subdir = $longGeom
                ft_ckpt = $longModel
                status = "error"
                reason = "long finetune failed"
                pointmap_source_requested = $longPointmapSourceRequested
                pointmap_source_resolved = $longPointmapSourceResolved
                precompute_fallback_used = [bool]$longPrecomputeFallbackUsed
                precompute_timeout_hit = [bool]$longPrecomputeTimeoutHit
            }
        }
    }
}

$sweepCsv = "logs/modal_phase5/vggt_ft_sweep_$timestamp.csv"
$sweepLatest = "logs/modal_phase5/vggt_ft_sweep_latest.csv"
Finalize-SweepRows -Rows $sweepRows
$sweepRows | Export-Csv $sweepCsv -NoTypeInformation -Encoding UTF8
$sweepRows | Export-Csv $sweepLatest -NoTypeInformation -Encoding UTF8
Write-Host "[lr-sweep] wrote: $sweepLatest"
exit 0
