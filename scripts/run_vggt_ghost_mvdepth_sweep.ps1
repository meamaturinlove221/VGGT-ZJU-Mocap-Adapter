param(
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$PseudoGeomSubdir = "vggt_geom",
    [string]$PretrainedCkpt = "model.pt",
    [string]$ResumeCkpt = "",
    [string]$ReuseShortFtCkpt = "",
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
    [double]$LambdaPointNormalConsis = 0.0,
    [double]$LambdaConf = 0.002,
    [int]$LambdaConfWarmupSteps = 80,
    [double]$LambdaCam = 0.0,
    [int]$LambdaCamWarmupSteps = 0,
    [int]$EpochsShort = 1,
    [int]$MaxFramesShort = 400,
    [int]$MaxStepsPerEpoch = 80,
    [int]$EvalNumSamples = 40,
    [string]$EvalInferArgsExtra = "--num_src_views=6",
    [string]$EvalNumSrcViewsList = "",
    [string]$CamNames = "Camera_B1,Camera_B2,Camera_B3,Camera_B4,Camera_B5,Camera_B6,Camera_B7,Camera_B8,Camera_B9,Camera_B10,Camera_B11,Camera_B12,Camera_B13,Camera_B14,Camera_B15,Camera_B16,Camera_B17,Camera_B18,Camera_B19,Camera_B20,Camera_B21,Camera_B22,Camera_B23",
    [string]$PointTargetMode = "depth_unproject",
    [double]$PointTargetBlendAlpha = 0.85,
    [double]$PointTargetBlendAlphaMin = 0.0,
    [double]$PointTargetBlendAlphaMax = 1.0,
    [double]$PointTargetConsensusAlphaFloor = 0.35,
    [double]$PointTargetBlendRelGain = 1.0,
    [double]$PointTargetBlendMvGain = 1.0,
    [string]$PointTargetBlendByReliability = "on",
    [string]$PointTargetBlendByMvSupport = "on",
    [string]$PointTargetBlendMvRegionMode = "all",
    [string]$PointTargetBlendMvPolicy = "strong_to_depth",
    [string]$PointmapSource = "auto",
    [string]$TargetPointFrame = "auto",
    [string]$PredPointFrame = "auto",
    [string]$UseFgMask = "on",
    [string]$FgMaskSource = "mask",
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
    [string]$PrecomputeMvSupportRegionMode = "auto",
    [string]$PrecomputeMvSupportFgMaskSource = "auto",
    [int]$PrecomputeMvSupportFgErodePx = 5,
    [int]$PrecomputeMvSupportFgPreservePx = 5,
    [string]$SupervisionWeightMode = "mix",
    [double]$SupervisionWeightMixAlpha = 0.35,
    [double]$ConfWeightThr = 0.0,
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
    [bool]$EnableAnySplatAblationSixPack = $false,
    [int]$FgMaskErodePx = 0,
    [int]$PointLossFgErodePx = 1,
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
    [bool]$Tf32 = $true,
    [bool]$Amp = $true,
    [bool]$StrictDeterministic = $false,
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
    [int]$PointMvMaskSoftBlurPx = 0,
    [int]$PointMvMaskSoftBlurIters = 1,
    [double]$PointMvMaskSoftMix = 0.0,
    [double]$PointMvMaskSoftHitThr = 0.35,
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
    [string]$PointMvMaskSupportMode = "inverse",
    [double]$PointMvMaskSupportFloor = 0.0,
    [int]$PointMvDepthFgErodePx = 0,
    [double]$PointMvTolAbs = 0.06,
    [double]$PointMvTolRel = 0.10,
    [double]$PointMvWeightFloor = 0.5,
    [int]$PointMvStride = 2,
    [int]$PointMvDepthMaxPairs = 2,
    [string]$PointMvDepthPairMode = "adjacent",
    [string]$PointMvDepthRegionMode = "all",
    [double]$PointMvDepthErrQuantile = 1.0,
    [int]$PointNormalConsisWarmupSteps = 40,
    [int]$EvalEverySteps = 1,
    [int]$DebugMetricsEverySteps = 1,
    [int]$DebugVisEverySteps = 1,
    [int]$DebugVisMaxSteps = 60,
    [int]$DebugVisViews = 1,
    [string]$DebugVisDir = "",
    [int]$ModalRunTimeoutSec = 3600,
    [int]$ModalRunNoOutputTimeoutSec = 600,
    [int]$ModalRunNoOutputTimeoutSecPointHead = 780,
    [int]$ModalRunNoOutputMaxRetries = 1,
    [bool]$ModalRunQuiet = $true,
    [int]$CkptWaitTimeoutSec = 1200,
    [bool]$EnableExtendedCkptWaitOnMissing = $true,
    [int]$CkptExtendedWaitTimeoutSec = 1200,
    [int]$CkptMissingRetryCount = 0,
    [bool]$EnableResumeCkptFallbackOnShortCkptMissing = $false,
    [bool]$DisallowResumeFallbackResult = $true,
    [int]$NoSpaceRetryCount = 1,
    [int]$NoSpaceCleanupKeepRecentDirs = 220,
    [int]$NoSpaceCleanupDeleteLimit = 120,
    [int]$NoImprovePatience = 2,
    [double]$MinGhostImprove = 0.03,
    [double]$MinPSNRGuard = 20.2,
    [double]$MinSSIMGuard = 0.70,
    [double]$MaxWl1Guard = 0.08,
    [bool]$EnableAbsoluteQualityGuard = $true,
    [int]$MaskWorsenStopConsecutive = 1,
    [double]$MaskWorsenGhostDelta = 0.10,
    [double]$CatastrophicGhostRiseStop = 1.2,
    [int]$InfraNoOutputStopConsecutive = 2,
    [bool]$EnableNoOutputProbeRetry = $true,
    [int]$NoOutputProbeTriggerConsecutive = 1,
    [int]$NoOutputProbeTimeoutSec = 720,
    [int]$NoOutputProbeTimeoutSecDepthUnproject = 600,
    [int]$NoOutputProbeTimeoutSecPointHeadWeak = 720,
    [int]$NoOutputProbeMaxRetries = 1,
    [int]$NoOutputProbeRunTimeoutSec = 4200,
    [int]$NoOutputProbeMaxFramesShort = 280,
    [int]$NoOutputProbeMaxStepsPerEpoch = 64,
    [int]$NoOutputProbeMaxUsesPerStage = 3,
    [bool]$EnablePreemptiveProbeForPointHead = $true,
    [int]$PreemptiveProbeMaxCandidates = 2,
    [bool]$EnablePreemptiveProbeForStrongDepthUnproject = $true,
    [bool]$SkipReactiveProbeAfterPreemptive = $true,
    [int]$PrecomputeNoOutputTimeoutSec = 1800,
    [int]$PrecomputeNoOutputTimeoutSecPointHead = 900,
    [int]$PrecomputeNoOutputTimeoutSecPointHeadWeak = 900,
    [bool]$EnableDepthPrecomputeNoOutputRecovery = $true,
    [int]$DepthPrecomputeNoOutputRecoveryThreshold = 1,
    [int]$DepthPrecomputeNoOutputRecoveryMaxCount = 1,
    [int]$DepthPrecomputeNoOutputRecoveryTimeoutSec = 1200,
    [bool]$EnablePrecomputeNoOutputRetry = $true,
    [int]$PrecomputeNoOutputRetryMaxRetries = 1,
    [int]$PrecomputeNoOutputRetryTimeoutSec = 1200,
    [int]$PrecomputeNoOutputRetryTimeoutSecPointHead = 600,
    [bool]$EnableDepthPrecomputeNoOutputPenalty = $true,
    [int]$DepthPrecomputeNoOutputPenaltyThreshold = 1,
    [int]$DepthPrecomputeNoOutputPenaltyTimeoutSec = 420,
    [int]$DepthPrecomputeNoOutputSkipThreshold = 2,
    [int]$PrecomputeNoOutputTimeoutSecDepthUnproject = 900,
    [int]$Stage1DepthUnprojectPrecomputeFloorSec = 1500,
    [int]$EvalNoOutputTimeoutSec = 2400,
    [int]$EvalNoOutputTimeoutSecPointHead = 1200,
    [int]$EvalNoOutputTimeoutSecDepthUnproject = 1500,
    [int]$MetricReadRetry = 3,
    [int]$MetricReadRetrySleepSec = 5,
    [string]$LaneId = "lane_a",
    [string]$CandidateFamily = "stage2_training",
    [string]$GuardTier = "",
    [bool]$RollbackTriggered = $false,
    [bool]$EnableVisualAntiBlackGuard = $true,
    [double]$MinPredLumaMean = 0.045,
    [double]$MinPredNonBlackRatio = 0.10,
    [double]$MinAreaRatio = 0.55,
    [double]$MinWidthRatio = 0.65
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

function To-BoolLoose(
    [Parameter(ValueFromPipeline = $true)]
    $Value,
    [bool]$Default = $false
) {
    if ($null -eq $Value) { return $Default }
    if ($Value -is [bool]) { return [bool]$Value }
    $raw = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }
    switch -Regex ($raw.ToLowerInvariant()) {
        '^(1|true|yes|y|on)$' { return $true }
        '^(0|false|no|n|off)$' { return $false }
        default { return $Default }
    }
}

function To-DoubleOrNaN(
    [Parameter(ValueFromPipeline = $true)]
    $Value
) {
    if ($null -eq $Value) { return [double]::NaN }
    if ($Value -is [double]) { return [double]$Value }
    if ($Value -is [single]) { return [double]$Value }
    if ($Value -is [decimal]) { return [double]$Value }
    if ($Value -is [System.Int16]) { return [double]$Value }
    if ($Value -is [System.Int32]) { return [double]$Value }
    if ($Value -is [System.Int64]) { return [double]$Value }
    if ($Value -is [byte]) { return [double]$Value }
    $raw = [string]$Value
    if ([string]::IsNullOrWhiteSpace($raw)) { return [double]::NaN }
    $raw = $raw.Trim()
    $parsed = 0.0
    if ([double]::TryParse($raw, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) {
        return [double]$parsed
    }
    return [double]::NaN
}

function Write-JsonFileNoBomLocal(
    [string]$Path,
    [object]$Obj
) {
    $abs = Join-Path (Resolve-Path ".").Path $Path
    $dir = Split-Path -Parent $abs
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 20
    [System.IO.File]::WriteAllText($abs, $json, $enc)
}

function Write-CandidateResultJsonLocal(
    [string]$Path,
    [string]$LatestPath,
    [object]$Row,
    [string]$RunTag,
    [string]$Timestamp
) {
    if ($null -eq $Row) { return }
    $payload = [ordered]@{
        updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        source = "run_vggt_ghost_mvdepth_sweep"
        candidate_result_version = 1
        run_tag = $RunTag
        run_timestamp = $Timestamp
    }
    foreach ($prop in @($Row.PSObject.Properties)) {
        $payload[$prop.Name] = $prop.Value
    }
    Write-JsonFileNoBomLocal -Path $Path -Obj $payload
    if (-not [string]::IsNullOrWhiteSpace($LatestPath)) {
        Write-JsonFileNoBomLocal -Path $LatestPath -Obj $payload
    }
}

function Write-CandidateResultSeedJsonLocal(
    [string]$Path,
    [string]$LatestPath,
    [hashtable]$Seed
) {
    if ($null -eq $Seed) { return }
    $payload = [ordered]@{
        updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        source = "run_vggt_ghost_mvdepth_sweep"
        candidate_result_version = 1
    }
    foreach ($k in @($Seed.Keys)) {
        $payload[$k] = $Seed[$k]
    }
    Write-JsonFileNoBomLocal -Path $Path -Obj $payload
    if (-not [string]::IsNullOrWhiteSpace($LatestPath)) {
        Write-JsonFileNoBomLocal -Path $LatestPath -Obj $payload
    }
}

function Copy-SelectedPropertiesLocal(
    [object]$Source,
    [object]$Target,
    [string[]]$Names
) {
    if (($null -eq $Source) -or ($null -eq $Target) -or ($null -eq $Names)) { return }
    foreach ($name in @($Names)) {
        if ([string]::IsNullOrWhiteSpace($name)) { continue }
        try {
            if ($Source.PSObject.Properties[$name]) {
                $val = $Source.$name
                if (-not $Target.PSObject.Properties[$name]) {
                    $Target | Add-Member -NotePropertyName $name -NotePropertyValue $val -Force
                } else {
                    $Target.$name = $val
                }
            }
        } catch {}
    }
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

function San([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) { return "item" }
    return ([regex]::Replace($Raw, "[^A-Za-z0-9_.-]+", "_")).Trim("_")
}

$pairModeBeforeNormalize = [string]$PointMvDepthPairMode
$PointMvDepthPairMode = Resolve-PointMvDepthPairMode -Raw $PointMvDepthPairMode -Default "adjacent"
if (([string]$pairModeBeforeNormalize).Trim().ToLowerInvariant() -ne ([string]$PointMvDepthPairMode)) {
    Write-Host "[ghost-mv] normalize point_mv_depth_pair_mode: raw='$pairModeBeforeNormalize' -> '$PointMvDepthPairMode'"
}

function Get-StepFromName([string]$Name) {
    if ([string]::IsNullOrWhiteSpace($Name)) { return -1 }
    $m = [regex]::Match($Name, "step(\d+)\.png$")
    if ($m.Success) {
        return [int]$m.Groups[1].Value
    }
    return -1
}

function Get-RemoteCatFgFiles([string]$OutVolPath, [int]$MaxCount = 3) {
    if ([string]::IsNullOrWhiteSpace($OutVolPath)) { return @() }
    try {
        $items = modal volume ls --json vggt-out $OutVolPath | ConvertFrom-Json
    } catch {
        return @()
    }
    $hits = @(
        @($items) |
            Where-Object {
                $fn = [string]$_.Filename
                $fn -match "cat_fg_mask_pred_tgt_step\d+\.png$"
            } |
            Sort-Object { Get-StepFromName([string]$_.Filename) }
    )
    if ($hits.Count -le 0) { return @() }
    return @($hits | Select-Object -First ([Math]::Max(1, [int]$MaxCount)))
}

function Volume-GetSafe([string]$RemoteFile, [string]$LocalPath) {
    if ([string]::IsNullOrWhiteSpace($RemoteFile) -or [string]::IsNullOrWhiteSpace($LocalPath)) {
        return $false
    }
    try {
        $root = Split-Path -Parent $LocalPath
        if (-not [string]::IsNullOrWhiteSpace($root)) {
            New-Item -ItemType Directory -Force -Path $root | Out-Null
        }
        modal volume get vggt-out $RemoteFile $LocalPath | Out-Null
        if (Test-Path $LocalPath) { return $true }
    } catch {
        if (Test-Path $LocalPath) { return $true }
    }
    return $false
}

function Read-BestFromCompare([string]$CompareCsvPath) {
    if ([string]::IsNullOrWhiteSpace($CompareCsvPath) -or (-not (Test-Path $CompareCsvPath))) {
        return $null
    }
    try {
        $cand = @(
            Import-Csv $CompareCsvPath |
                Where-Object { $_.status -eq "ok" } |
                Sort-Object { [double]$_.mean_PSNR } -Descending
        )
        if ($cand.Count -gt 0) {
            $row = $cand[0]
            $row | Add-Member -NotePropertyName best_source -NotePropertyValue "compare" -Force
            return $row
        }
    } catch {
    }
    return $null
}

function Read-BestFromGate([string]$GateJsonPath) {
    if ([string]::IsNullOrWhiteSpace($GateJsonPath) -or (-not (Test-Path $GateJsonPath))) {
        return $null
    }
    try {
        $gate = Get-Content $GateJsonPath -Raw | ConvertFrom-Json
        $rows = @($gate.rows)
        if ($rows.Count -le 0) { return $null }
        $ok = @($rows | Where-Object { [string]$_.status -eq "ok" })
        if ($ok.Count -le 0) { $ok = $rows }
        if ($ok.Count -le 0) { return $null }
        $best = $ok[0]
        return [pscustomobject]@{
            label = [string]$best.label
            geom_subdir = [string]$best.geom_subdir
            mean_PSNR = [double]$best.mean_PSNR
            mean_SSIM = [double]$best.mean_SSIM
            mean_weighted_L1 = [double]$best.mean_weighted_L1
            infer_out_volume_path = [string]$best.infer_out_volume_path
            best_source = "gate"
        }
    } catch {
    }
    return $null
}

function Resolve-BestMetrics(
    [string]$CompareCsvPath,
    [string]$GateJsonPath,
    [int]$RetryCount,
    [int]$RetrySleepSec
) {
    $retryN = [Math]::Max(0, [int]$RetryCount)
    $sleepSec = [Math]::Max(1, [int]$RetrySleepSec)
    for ($i = 0; $i -le $retryN; $i++) {
        $best = Read-BestFromCompare -CompareCsvPath $CompareCsvPath
        if ($best -ne $null) { return $best }
        if ($i -lt $retryN) {
            Start-Sleep -Seconds $sleepSec
        }
    }
    return (Read-BestFromGate -GateJsonPath $GateJsonPath)
}

function Get-LatestFtSweepFailureReason([string]$SweepCsvPath) {
    if ([string]::IsNullOrWhiteSpace($SweepCsvPath) -or (-not (Test-Path $SweepCsvPath))) {
        return ""
    }
    try {
        $rows = @(Import-Csv $SweepCsvPath)
        if ($rows.Count -le 0) { return "" }
        $last = $rows[$rows.Count - 1]
        return [string]$last.reason
    } catch {
        return ""
    }
}

function Get-LatestFtSweepRow([string]$SweepCsvPath) {
    if ([string]::IsNullOrWhiteSpace($SweepCsvPath) -or (-not (Test-Path $SweepCsvPath))) {
        return $null
    }
    try {
        $rows = @(Import-Csv $SweepCsvPath)
        if ($rows.Count -le 0) { return $null }
        return $rows[$rows.Count - 1]
    } catch {
        return $null
    }
}

function Resolve-CandidateInvalidReason(
    [int]$ExitCode,
    [string]$FailureReason,
    [bool]$VisualGuardBlocked,
    [bool]$QualityGuardBlocked,
    [bool]$EvalNumSrcViewsMismatch,
    [object]$LatestFtRow
) {
    if ($EvalNumSrcViewsMismatch) { return "src_views_mismatch" }
    if ($QualityGuardBlocked) { return "quality_guard_blocked" }
    if ($VisualGuardBlocked) { return "visual_guard_blocked" }

    $latestFtCandidateInvalidReason = ""
    try {
        if (($LatestFtRow -ne $null) -and $LatestFtRow.PSObject.Properties["candidate_invalid_reason"]) {
            $latestFtCandidateInvalidReason = [string]$LatestFtRow.candidate_invalid_reason
        }
    } catch {}

    $reasonNorm = ([string]$FailureReason).Trim()
    if ([string]::IsNullOrWhiteSpace($reasonNorm)) {
        return $latestFtCandidateInvalidReason
    }
    if ($reasonNorm -match "(?i)zero_samples\(n=0\)|eval_empty|post_eval_no_valid_metrics|no valid metrics row") { return "eval_empty" }
    if ($reasonNorm -match "(?i)failed to fetch/parse metrics|evaluate short candidates failed|eval_failed") { return "eval_failed" }
    if ($reasonNorm -match "(?i)eval_num_src_views_mismatch|src_views_mismatch") { return "src_views_mismatch" }
    if ($reasonNorm -match "(?i)quality_guard") { return "quality_guard_blocked" }
    if ($reasonNorm -match "(?i)visual_guard") { return "visual_guard_blocked" }
    if ($reasonNorm -match "(?i)(?:heartbeat_stall_timeout|no_output_timeout)_\d+s") { return "no_output_timeout" }
    if ($reasonNorm -match "(?i)precompute_empty|geom_subdir_empty|missing_seq_npz|no_npz") { return "precompute_empty" }
    if ($reasonNorm -match "(?i)precompute|cuda out of memory|out of memory|oom") { return "precompute_failed" }
    if (-not [string]::IsNullOrWhiteSpace($latestFtCandidateInvalidReason)) {
        return $latestFtCandidateInvalidReason
    }
    return ""
}

function Resolve-ActualEvalNumSrcViews([object]$BestRow) {
    if ($null -eq $BestRow) { return "" }
    $probe = @()
    foreach ($k in @("infer_out_dir", "infer_out_volume_path", "run_url", "label")) {
        try {
            if ($BestRow.PSObject.Properties[$k]) {
                $v = [string]$BestRow.$k
                if (-not [string]::IsNullOrWhiteSpace($v)) { $probe += @($v) }
            }
        } catch {}
    }
    foreach ($s in @($probe)) {
        $m = [regex]::Match([string]$s, "(?i)multiview_eval(\d+)")
        if ($m.Success) { return [string]$m.Groups[1].Value }
        $m = [regex]::Match([string]$s, "(?i)(?:num_src_views|srcviews|srcview|views)(\d+)")
        if ($m.Success) { return [string]$m.Groups[1].Value }
    }
    return ""
}

function Set-InferArgsNumSrcViews(
    [string]$BaseInferArgs,
    [string]$NumSrcViews
) {
    if ([string]::IsNullOrWhiteSpace($NumSrcViews)) {
        return [string]$BaseInferArgs
    }
    $n = [string]$NumSrcViews.Trim()
    if (-not ($n -match "^\d+$")) {
        return [string]$BaseInferArgs
    }
    $base = [string]$BaseInferArgs
    if ([string]::IsNullOrWhiteSpace($base)) {
        return "--num_src_views=$n"
    }
    if ([regex]::IsMatch($base, "(?:^|\s)--num_src_views(?:\s+|=)\d+(?:\s|$)")) {
        return [regex]::Replace(
            $base,
            "(?:^|\s)--num_src_views(?:\s+|=)\d+(?:\s|$)",
            " --num_src_views=$n "
        ).Trim()
    }
    return ($base.Trim() + " --num_src_views=$n").Trim()
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

function Resolve-EvalNumSrcViewsForCandidate(
    [int]$CandidateIndex,
    [string[]]$ViewTokens
) {
    $tokens = @($ViewTokens | Where-Object { $_ -match "^\d+$" })
    if ($tokens.Count -le 0) { return "" }
    $idx = [Math]::Max(0, [int]$CandidateIndex - 1)
    $pick = $idx % $tokens.Count
    return [string]$tokens[$pick]
}

function Read-VisualMetricsFromSummary(
    [string]$GhostSummaryCsvPath
) {
    $out = [ordered]@{
        ghost_visual_score = [double]::NaN
        pred_luma_mean = [double]::NaN
        pred_nonblack_ratio_thr008 = [double]::NaN
        pred_nonblack_ratio_thr015 = [double]::NaN
        fg_pred_luma_mean = [double]::NaN
        fg_pred_nonblack_ratio = [double]::NaN
        fg_pred_contrast = [double]::NaN
        fg_pred_tgt_l1 = [double]::NaN
    }
    if ([string]::IsNullOrWhiteSpace($GhostSummaryCsvPath) -or (-not (Test-Path $GhostSummaryCsvPath))) {
        return [pscustomobject]$out
    }
    try {
        $rows = @(Import-Csv $GhostSummaryCsvPath)
        if ($rows.Count -le 0) { return [pscustomobject]$out }
        $r0 = $rows[0]
        if ($r0.PSObject.Properties["ghost_visual_score_mean"]) {
            $out.ghost_visual_score = [double]$r0.ghost_visual_score_mean
        }
        if ($r0.PSObject.Properties["pred_luma_mean_mean"]) {
            $out.pred_luma_mean = [double]$r0.pred_luma_mean_mean
        }
        if ($r0.PSObject.Properties["pred_nonblack_ratio_thr008_mean"]) {
            $out.pred_nonblack_ratio_thr008 = [double]$r0.pred_nonblack_ratio_thr008_mean
        }
        if ($r0.PSObject.Properties["pred_nonblack_ratio_thr015_mean"]) {
            $out.pred_nonblack_ratio_thr015 = [double]$r0.pred_nonblack_ratio_thr015_mean
        }
        if ($r0.PSObject.Properties["fg_pred_luma_mean_mean"]) {
            $out.fg_pred_luma_mean = [double]$r0.fg_pred_luma_mean_mean
        }
        if ($r0.PSObject.Properties["fg_pred_nonblack_ratio_mean"]) {
            $out.fg_pred_nonblack_ratio = [double]$r0.fg_pred_nonblack_ratio_mean
        }
        if ($r0.PSObject.Properties["fg_pred_contrast_mean"]) {
            $out.fg_pred_contrast = [double]$r0.fg_pred_contrast_mean
        }
        if ($r0.PSObject.Properties["fg_pred_tgt_l1_mean"]) {
            $out.fg_pred_tgt_l1 = [double]$r0.fg_pred_tgt_l1_mean
        }
    } catch {
    }
    return [pscustomobject]$out
}

function Invoke-FtLrSweepForCurrentSetting(
    [string]$MvValue,
    [string]$MvMaskValue,
    [string]$LocalEvalInferArgsExtra,
    [double]$LocalConfWeightPerViewQuantile = [double]::NaN,
    [double]$LocalLambdaPointNormalConsis = [double]::NaN,
    [string]$LocalPointMvDepthSupportMode = "",
    [double]$LocalPointMvDepthSupportFloor = [double]::NaN,
    [string]$LocalPointMvMaskSupportMode = "",
    [double]$LocalPointMvMaskSupportFloor = [double]::NaN,
    [int]$LocalMaxFramesShort,
    [int]$LocalMaxStepsPerEpoch,
    [int]$LocalModalRunTimeoutSec,
    [int]$LocalModalRunNoOutputTimeoutSec,
    [int]$LocalModalRunNoOutputMaxRetries,
    [bool]$LocalModalRunQuiet,
    [bool]$LocalShortFinetuneAllowQuietNoOutputBypass,
    [int]$LocalPrecomputeNoOutputTimeoutSec,
    [int]$LocalEvalNoOutputTimeoutSec,
    [int]$LocalCkptWaitTimeoutSec
) {
    $effConfWeightPerViewQuantile = if ([double]::IsNaN([double]$LocalConfWeightPerViewQuantile)) { [double]$ConfWeightPerViewQuantile } else { [double]$LocalConfWeightPerViewQuantile }
    $effLambdaPointNormalConsis = if ([double]::IsNaN([double]$LocalLambdaPointNormalConsis)) { [double]$LambdaPointNormalConsis } else { [double]$LocalLambdaPointNormalConsis }
    $effPointMvDepthSupportMode = if ([string]::IsNullOrWhiteSpace([string]$LocalPointMvDepthSupportMode)) { [string]$PointMvDepthSupportMode } else { [string]$LocalPointMvDepthSupportMode }
    $effPointMvDepthSupportFloor = if ([double]::IsNaN([double]$LocalPointMvDepthSupportFloor)) { [double]$PointMvDepthSupportFloor } else { [double]$LocalPointMvDepthSupportFloor }
    $effPointMvMaskSupportMode = if ([string]::IsNullOrWhiteSpace([string]$LocalPointMvMaskSupportMode)) { [string]$PointMvMaskSupportMode } else { [string]$LocalPointMvMaskSupportMode }
    $effPointMvMaskSupportFloor = if ([double]::IsNaN([double]$LocalPointMvMaskSupportFloor)) { [double]$PointMvMaskSupportFloor } else { [double]$LocalPointMvMaskSupportFloor }

    $pointmapSourceNormLocal = ([string]$PointmapSource).Trim().ToLowerInvariant()
    $candidateFamilyNormLocal = ([string]$CandidateFamily).Trim().ToLowerInvariant()
    $blendPolicyNormLocal = ([string]$PointTargetBlendMvPolicy).Trim().ToLowerInvariant()
    $enablePrecomputeFallback = $false
    $precomputeFallbackSource = "depth_unproject"
    if ($pointmapSourceNormLocal -eq "point_head") {
        # Only keep fallback from point_head -> depth_unproject.
        # This path remains the default across all stages.
        $enablePrecomputeFallback = $true
    } elseif (($pointmapSourceNormLocal -eq "depth_unproject") -and ($candidateFamilyNormLocal -match "stage1")) {
        # Stage1 stabilization: allow one source flip when depth_unproject
        # precompute stalls, so candidate production can continue.
        $enablePrecomputeFallback = $true
        $precomputeFallbackSource = "point_head"
    }
    $precomputeFallbackNoOutputTimeoutSec = [Math]::Max(300, [int]$PrecomputeNoOutputTimeoutSec)
    $precomputeFallbackSourceNorm = ([string]$precomputeFallbackSource).Trim().ToLowerInvariant()
    if ($precomputeFallbackSourceNorm -eq "point_head") {
        $precomputeFallbackNoOutputTimeoutSec = [Math]::Max(300, [int]$PrecomputeNoOutputTimeoutSecPointHead)
        if ($blendPolicyNormLocal -eq "weak_to_depth") {
            $precomputeFallbackNoOutputTimeoutSec = [Math]::Max(
                [int]$precomputeFallbackNoOutputTimeoutSec,
                [int]$PrecomputeNoOutputTimeoutSecPointHeadWeak
            )
        }
    } elseif ($precomputeFallbackSourceNorm -eq "depth_unproject") {
        $precomputeFallbackNoOutputTimeoutSec = [Math]::Max(300, [int]$PrecomputeNoOutputTimeoutSecDepthUnproject)
    }
    & "$CodeDir\scripts\run_vggt_ft_lr_sweep.ps1" `
        -CodeDir $CodeDir `
        -SeqNames $SeqNames `
        -PseudoGeomSubdir $PseudoGeomSubdir `
        -PretrainedCkpt $PretrainedCkpt `
        -ResumeCkpt $ResumeCkpt `
        -ReuseShortFtCkpt $ReuseShortFtCkpt `
        -LrList $Lr `
        -LrBackboneScale $LrBackboneScale `
        -LrHeadScale $LrHeadScale `
        -LrCameraScale $LrCameraScale `
        -GradClip $GradClip `
        -MinMeanStepUpdateRatio $MinMeanStepUpdateRatio `
        -FreezeMode all_trainable `
        -DepthScaleAlign median `
        -EpochsShort $EpochsShort `
        -MaxFramesShort $LocalMaxFramesShort `
        -MaxStepsPerEpoch $LocalMaxStepsPerEpoch `
        -EvalNumSamples $EvalNumSamples `
        -EvalInferArgsExtra $LocalEvalInferArgsExtra `
        -LambdaPoint $LambdaPoint `
        -LambdaConf $LambdaConf `
        -LambdaConfWarmupSteps $LambdaConfWarmupSteps `
        -LambdaCam $LambdaCam `
        -LambdaCamWarmupSteps $LambdaCamWarmupSteps `
        -PointTargetMode $PointTargetMode `
        -PointTargetBlendAlpha $PointTargetBlendAlpha `
        -PointTargetBlendAlphaMin $PointTargetBlendAlphaMin `
        -PointTargetBlendAlphaMax $PointTargetBlendAlphaMax `
        -PointTargetConsensusAlphaFloor $PointTargetConsensusAlphaFloor `
        -PointTargetBlendRelGain $PointTargetBlendRelGain `
        -PointTargetBlendMvGain $PointTargetBlendMvGain `
        -PointTargetBlendByReliability $PointTargetBlendByReliability `
        -PointTargetBlendByMvSupport $PointTargetBlendByMvSupport `
        -PointTargetBlendMvRegionMode $PointTargetBlendMvRegionMode `
        -PointTargetBlendMvPolicy $PointTargetBlendMvPolicy `
        -PointmapSource $PointmapSource `
        -EnablePrecomputePointmapFallbackOnNoOutput $enablePrecomputeFallback `
        -PrecomputeFallbackPointmapSource $precomputeFallbackSource `
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
        -PrecomputeMvSupportRegionMode $PrecomputeMvSupportRegionMode `
        -PrecomputeMvSupportFgMaskSource $PrecomputeMvSupportFgMaskSource `
        -PrecomputeMvSupportFgErodePx $PrecomputeMvSupportFgErodePx `
        -PrecomputeMvSupportFgPreservePx $PrecomputeMvSupportFgPreservePx `
        -EvalEverySteps $EvalEverySteps `
        -DebugMetricsEverySteps $DebugMetricsEverySteps `
        -DebugVisEverySteps $DebugVisEverySteps `
        -DebugVisMaxSteps $DebugVisMaxSteps `
        -DebugVisViews $DebugVisViews `
        -DebugVisDir $DebugVisDir `
        -ModalRunTimeoutSec $LocalModalRunTimeoutSec `
        -ModalRunNoOutputTimeoutSec $LocalModalRunNoOutputTimeoutSec `
        -ModalRunNoOutputMaxRetries $LocalModalRunNoOutputMaxRetries `
        -ModalRunQuiet:$LocalModalRunQuiet `
        -ShortFinetuneAllowQuietNoOutputBypass $LocalShortFinetuneAllowQuietNoOutputBypass `
        -PrecomputeNoOutputTimeoutSec $LocalPrecomputeNoOutputTimeoutSec `
        -PrecomputeFallbackNoOutputTimeoutSec $precomputeFallbackNoOutputTimeoutSec `
        -EvalNoOutputTimeoutSec $LocalEvalNoOutputTimeoutSec `
        -CkptWaitTimeoutSec $LocalCkptWaitTimeoutSec `
        -CkptMissingRetryCount $CkptMissingRetryCount `
        -NoSpaceRetryCount $NoSpaceRetryCount `
        -NoSpaceCleanupKeepRecentDirs $NoSpaceCleanupKeepRecentDirs `
        -NoSpaceCleanupDeleteLimit $NoSpaceCleanupDeleteLimit `
        -CamNames $CamNames `
        -UseFgMask $UseFgMask `
        -FgMaskSource $FgMaskSource `
        -FgMaskErodePx $FgMaskErodePx `
        -PointLossFgErodePx $PointLossFgErodePx `
        -FgSupervisionBoost $FgSupervisionBoost `
        -FgSupervisionBgFloor $FgSupervisionBgFloor `
        -FgSupervisionRegionMode $FgSupervisionRegionMode `
        -FgSupervisionRegionErodePx $FgSupervisionRegionErodePx `
        -LambdaFgConfPresence $LambdaFgConfPresence `
        -FgConfPresenceTargetRatio $FgConfPresenceTargetRatio `
        -LambdaFgStructureDepthEdge $LambdaFgStructureDepthEdge `
        -FgStructureBboxMarginPx $FgStructureBboxMarginPx `
        -FgStructureBboxMinSidePx $FgStructureBboxMinSidePx `
        -FgStructureRegionMode $FgStructureRegionMode `
        -FgStructureRegionErodePx $FgStructureRegionErodePx `
        -FgStructureDepthEdgeWarmupSteps $FgStructureDepthEdgeWarmupSteps `
        -FgStructureBoundaryProbePx $FgStructureBoundaryProbePx `
        -FgStructureEdgeSupportMode $FgStructureEdgeSupportMode `
        -FgStructureEdgeSupportQuantile $FgStructureEdgeSupportQuantile `
        -FgStructureEdgeSupportMinPx $FgStructureEdgeSupportMinPx `
        -FgStructureEdgeWeightMode $FgStructureEdgeWeightMode `
        -FgStructureBoundaryFalloffPx $FgStructureBoundaryFalloffPx `
        -FgStructureComponentBiasMode $FgStructureComponentBiasMode `
        -FgStructureComponentBiasThresholdRatio $FgStructureComponentBiasThresholdRatio `
        -FgStructureComponentBiasOtherScale $FgStructureComponentBiasOtherScale `
        -FgStructureFrontDepthBiasMode $FgStructureFrontDepthBiasMode `
        -FgStructureFrontDepthBiasTau $FgStructureFrontDepthBiasTau `
        -FgStructureFrontDepthBiasCenterQuantile $FgStructureFrontDepthBiasCenterQuantile `
        -LambdaPointMvOutsideRing $LambdaPointMvOutsideRing `
        -PointMvOutsideRingPx $PointMvOutsideRingPx `
        -Tf32 $Tf32 `
        -Amp $Amp `
        -StrictDeterministic $StrictDeterministic `
        -SupervisionWeightMode $SupervisionWeightMode `
        -SupervisionWeightMixAlpha $SupervisionWeightMixAlpha `
        -ConfWeightThr $ConfWeightThr `
        -ConfWeightGamma $ConfWeightGamma `
        -ConfWeightPerViewQuantile $effConfWeightPerViewQuantile `
        -ConfWeightPerViewMinValid $ConfWeightPerViewMinValid `
        -GramDynEnable $GramDynEnable `
        -GramDynLayerIdx $GramDynLayerIdx `
        -GramDynQuantile $GramDynQuantile `
        -GramDynWeightFloor $GramDynWeightFloor `
        -GramDynWarmupSteps $GramDynWarmupSteps `
        -DynProxyEnable $DynProxyEnable `
        -DynProxyMode $DynProxyMode `
        -DynProxyUseGram $DynProxyUseGram `
        -DynProxyUseSupport $DynProxyUseSupport `
        -DynProxyFloor $DynProxyFloor `
        -DynProxyWarmupSteps $DynProxyWarmupSteps `
        -LambdaPointReproj $LambdaPointReproj `
        -LambdaPointNormalConsis $effLambdaPointNormalConsis `
        -PointReprojWarmupSteps $PointReprojWarmupSteps `
        -PointReprojClampPx 64 `
        -PointMvConsistency on `
        -LambdaPointMvDepth ([double]$MvValue) `
        -LambdaPointMvMask ([double]$MvMaskValue) `
        -PointMvTolAbs $PointMvTolAbs `
        -PointMvTolRel $PointMvTolRel `
        -PointMvWeightFloor $PointMvWeightFloor `
        -PointMvStride $PointMvStride `
        -PointMvDepthMaxPairs $PointMvDepthMaxPairs `
        -PointMvDepthPairMode $PointMvDepthPairMode `
        -PointMvDepthWarmupSteps $PointMvDepthWarmupSteps `
        -PointMvDepthRegionMode $PointMvDepthRegionMode `
        -PointMvMaskWarmupSteps $PointMvMaskWarmupSteps `
        -PointMvDepthInlierOnly $PointMvDepthInlierOnly `
        -PointMvDepthErrQuantile $PointMvDepthErrQuantile `
        -PointMvDepthOutlierBoost $PointMvDepthOutlierBoost `
        -PointMvDepthOutlierCap $PointMvDepthOutlierCap `
        -PointMvDepthTgtValidMode $PointMvDepthTgtValidMode `
        -PointMvDepthTgtValidFloor $PointMvDepthTgtValidFloor `
        -PointMvDepthMinTgtValidRatio $PointMvDepthMinTgtValidRatio `
        -PointMvMaskMinTgtFgRatio $PointMvMaskMinTgtFgRatio `
        -PointMvMaskHitThr $PointMvMaskHitThr `
        -PointMvMaskSoftBlurPx $PointMvMaskSoftBlurPx `
        -PointMvMaskSoftBlurIters $PointMvMaskSoftBlurIters `
        -PointMvMaskSoftMix $PointMvMaskSoftMix `
        -PointMvMaskSoftHitThr $PointMvMaskSoftHitThr `
        -PointMvDepthTgtValidScaleMode $PointMvDepthTgtValidScaleMode `
        -PointMvDepthTgtValidScaleThr $PointMvDepthTgtValidScaleThr `
        -PointMvDepthAdaptMode $PointMvDepthAdaptMode `
        -PointMvDepthAdaptTargetValid $PointMvDepthAdaptTargetValid `
        -PointMvDepthAdaptMinScale $PointMvDepthAdaptMinScale `
        -PointMvDepthAdaptMaxScale $PointMvDepthAdaptMaxScale `
        -PointSupportMode $PointSupportMode `
        -PointSupportFloor $PointSupportFloor `
        -PointMvDepthSupportMode $effPointMvDepthSupportMode `
        -PointMvDepthSupportFloor $effPointMvDepthSupportFloor `
        -PointMvMaskSupportMode $effPointMvMaskSupportMode `
        -PointMvMaskSupportFloor $effPointMvMaskSupportFloor `
        -PointMvDepthFgErodePx $PointMvDepthFgErodePx `
        -PointWarmupSteps $PointWarmupSteps `
        -PointNormalConsisWarmupSteps $PointNormalConsisWarmupSteps `
        -PointLossScaleDepthUnproject $PointLossScaleDepthUnproject `
        -PointConsClipMinQv $PointConsClipMinQv `
        -PointConsQuantile $PointConsQuantile `
        -PointConsFocus $PointConsFocus `
        -PointResidualQuantile $PointResidualQuantile `
        -PointResidualFocus $PointResidualFocus `
        -PointResidualBoost $PointResidualBoost `
        -PointResidualBoostCap $PointResidualBoostCap `
        -EnableExtendedCkptWaitOnMissing $EnableExtendedCkptWaitOnMissing `
        -CkptExtendedWaitTimeoutSec $CkptExtendedWaitTimeoutSec `
        -EnableResumeCkptFallbackOnShortCkptMissing $EnableResumeCkptFallbackOnShortCkptMissing `
        -EarlyStopPatience 0 `
        -MinImprove 0.0
    return [int]$LASTEXITCODE
}

$mvVals = Parse-Tokens -Raw $LambdaPointMvDepthList
if ($mvVals.Count -eq 0) { throw "LambdaPointMvDepthList is empty" }
$mvMaskVals = Parse-Tokens -Raw $LambdaPointMvMaskList
if ($mvMaskVals.Count -eq 0) { $mvMaskVals = @([string]$LambdaPointMvMask) }
$evalNumSrcViewTokens = Parse-Tokens -Raw $EvalNumSrcViewsList
$camCountUsed = Get-CamCountFromCamNames -RawCamNames $CamNames

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outCsv = "logs/modal_phase5/ghost_mvdepth_sweep_$ts.csv"
$outLatest = "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv"
$outMd = "logs/modal_phase5/ghost_mvdepth_sweep_latest.md"
$rows = @()
$bestGhost = [double]::PositiveInfinity
$bestPsnr = [double]::NegativeInfinity
$bestWl1 = [double]::PositiveInfinity
$noImproveRounds = 0
$infraNoOutputConsecutive = 0
$stopDueInfraNoOutput = $false
$stopDueCatastrophicGhost = $false
$noOutputProbeUsedCount = 0
$candidateIndex = 0
$depthPrecomputeNoOutputFailures = @{}
$effectiveNoImprovePatience = [Math]::Max(1, [int]$NoImprovePatience)

$quantileOnValue = [double]$ConfWeightPerViewQuantile
if ($quantileOnValue -le 0.0) { $quantileOnValue = 0.65 }
$normalOnValue = [double]$LambdaPointNormalConsis
if ($normalOnValue -le 0.0) { $normalOnValue = 0.05 }
$splitDepthModeOn = "direct"
$splitDepthFloorOn = [Math]::Max(0.05, [double]$PointMvDepthSupportFloor)
$splitMaskModeOn = "inverse"
$splitMaskFloorOn = [Math]::Max(0.0, [double]$PointMvMaskSupportFloor)
$ablationProfiles = @(
    [pscustomobject]@{
        id = "default"
        conf_q = [double]::NaN
        normal = [double]::NaN
        depth_mode = ""
        depth_floor = [double]::NaN
        mask_mode = ""
        mask_floor = [double]::NaN
        support_split = $false
    }
)
if ([bool]$EnableAnySplatAblationSixPack) {
    $ablationProfiles = @(
        [pscustomobject]@{
            id = "ablation_baseline"
            conf_q = 0.0
            normal = 0.0
            depth_mode = "off"
            depth_floor = 0.0
            mask_mode = "off"
            mask_floor = 0.0
            support_split = $false
        },
        [pscustomobject]@{
            id = "ablation_quantile_only"
            conf_q = $quantileOnValue
            normal = 0.0
            depth_mode = "off"
            depth_floor = 0.0
            mask_mode = "off"
            mask_floor = 0.0
            support_split = $false
        },
        [pscustomobject]@{
            id = "ablation_normal_only"
            conf_q = 0.0
            normal = $normalOnValue
            depth_mode = "off"
            depth_floor = 0.0
            mask_mode = "off"
            mask_floor = 0.0
            support_split = $false
        },
        [pscustomobject]@{
            id = "ablation_support_split_only"
            conf_q = 0.0
            normal = 0.0
            depth_mode = $splitDepthModeOn
            depth_floor = $splitDepthFloorOn
            mask_mode = $splitMaskModeOn
            mask_floor = $splitMaskFloorOn
            support_split = $true
        },
        [pscustomobject]@{
            id = "ablation_quantile_normal"
            conf_q = $quantileOnValue
            normal = $normalOnValue
            depth_mode = "off"
            depth_floor = 0.0
            mask_mode = "off"
            mask_floor = 0.0
            support_split = $false
        },
        [pscustomobject]@{
            id = "ablation_all_on"
            conf_q = $quantileOnValue
            normal = $normalOnValue
            depth_mode = $splitDepthModeOn
            depth_floor = $splitDepthFloorOn
            mask_mode = $splitMaskModeOn
            mask_floor = $splitMaskFloorOn
            support_split = $true
        }
    )
    $effectiveNoImprovePatience = [Math]::Max($effectiveNoImprovePatience, ($ablationProfiles.Count * 2))
    Write-Host "[ghost-mv] AnySplat ablation six-pack enabled profiles=$($ablationProfiles.Count) effective_no_improve_patience=$effectiveNoImprovePatience"
}

foreach ($mv in $mvVals) {
    $prevMaskGhost = [double]::NaN
    $maskWorsenConsecutive = 0
    foreach ($mvMask in $mvMaskVals) {
    foreach ($ablationProfile in $ablationProfiles) {
    $candidateIndex += 1
    $maskWorsenTriggered = $false
    $ablationGroupId = [string]$ablationProfile.id
    if ([string]::IsNullOrWhiteSpace($ablationGroupId)) { $ablationGroupId = "default" }
    $runConfWeightPerViewQuantile = [double]$ablationProfile.conf_q
    $runLambdaPointNormalConsis = [double]$ablationProfile.normal
    $runPointMvDepthSupportMode = [string]$ablationProfile.depth_mode
    $runPointMvDepthSupportFloor = [double]$ablationProfile.depth_floor
    $runPointMvMaskSupportMode = [string]$ablationProfile.mask_mode
    $runPointMvMaskSupportFloor = [double]$ablationProfile.mask_floor
    $ablationSupportSplit = [bool]$ablationProfile.support_split
    $tag = "mv_" + (San($mv)) + "_mvmask_" + (San($mvMask)) + "_" + (San($ablationGroupId))
    Write-Host "[ghost-mv] run lambda_point_mv_depth=$mv lambda_point_mv_mask=$mvMask ablation=$ablationGroupId"
    $evalNumSrcViewsNow = Resolve-EvalNumSrcViewsForCandidate -CandidateIndex $candidateIndex -ViewTokens $evalNumSrcViewTokens
    $runEvalInferArgsExtra = Set-InferArgsNumSrcViews -BaseInferArgs $EvalInferArgsExtra -NumSrcViews $evalNumSrcViewsNow
    if (-not [string]::IsNullOrWhiteSpace($evalNumSrcViewsNow)) {
        Write-Host "[ghost-mv] candidate view profile: eval_num_src_views=$evalNumSrcViewsNow cam_count_used=$camCountUsed infer_args=`"$runEvalInferArgsExtra`""
    }
    Write-Host "[ghost-mv] ablation knobs: conf_q=$runConfWeightPerViewQuantile normal=$runLambdaPointNormalConsis depth_support=$runPointMvDepthSupportMode/$runPointMvDepthSupportFloor mask_support=$runPointMvMaskSupportMode/$runPointMvMaskSupportFloor"
    $candidateSkipReason = ""
    $depthKey = [string]$mv
    $depthPrecomputeNoOutputFailCount = 0
    if ($depthPrecomputeNoOutputFailures.ContainsKey($depthKey)) {
        try { $depthPrecomputeNoOutputFailCount = [int]$depthPrecomputeNoOutputFailures[$depthKey] } catch { $depthPrecomputeNoOutputFailCount = 0 }
    }

    $runMaxFramesShort = [int]$MaxFramesShort
    $runMaxStepsPerEpoch = [int]$MaxStepsPerEpoch
    $runModalRunTimeoutSec = [int]$ModalRunTimeoutSec
    $runModalRunNoOutputTimeoutSec = [int]$ModalRunNoOutputTimeoutSec
    $runModalRunNoOutputMaxRetries = [int]$ModalRunNoOutputMaxRetries
    # Over the overnight P0 window, silent short-finetune runs are treated as
    # chain failures that must be surfaced quickly instead of being hidden by
    # quiet mode. Checkpoint materialization is handled by the explicit wait
    # path below; the modal run itself should still obey no-output fast-fail.
    $runShortFinetuneAllowQuietNoOutputBypass = $false
    # Keep checkpoint materialization wait bounded for throughput.
    # For depth_unproject, allow a larger wait window to reduce
    # false fallback-to-resume when remote volume checkpoint materialization lags.
    $runCkptWaitTimeoutSec = [Math]::Max(60, [Math]::Min([int]$CkptWaitTimeoutSec, 900))
    $preemptiveProbeApplied = $false
    $pointmapSourceNorm = ([string]$PointmapSource).Trim().ToLowerInvariant()
    if ($pointmapSourceNorm -eq "point_head") {
        $runModalRunNoOutputTimeoutSec = [Math]::Max(
            [int]$runModalRunNoOutputTimeoutSec,
            [int]$ModalRunNoOutputTimeoutSecPointHead
        )
        # point_head short finetune often misses ckpt materialization; cap checkpoint
        # wait but do not over-aggressively fallback to resume_ckpt.
        $runCkptWaitTimeoutSec = [Math]::Max(60, [Math]::Min([int]$runCkptWaitTimeoutSec, 300))
    }
    $blendPolicyNorm = ([string]$PointTargetBlendMvPolicy).Trim().ToLowerInvariant()
    $runPrecomputeNoOutputTimeoutSec = [Math]::Max(300, [int]$PrecomputeNoOutputTimeoutSec)
    if ($pointmapSourceNorm -eq "point_head") {
        $runPrecomputeNoOutputTimeoutSec = [Math]::Max(300, [int]$PrecomputeNoOutputTimeoutSecPointHead)
    } elseif ($pointmapSourceNorm -eq "depth_unproject") {
        $runPrecomputeNoOutputTimeoutSec = [Math]::Max(300, [int]$PrecomputeNoOutputTimeoutSecDepthUnproject)
        if (([string]$CandidateFamily).Trim().ToLowerInvariant() -match "stage1") {
            $stage1DepthFloor = [Math]::Max(300, [int]$Stage1DepthUnprojectPrecomputeFloorSec)
            # Keep a stage1-specific floor, but allow autoloop emergency mode
            # to lower it for throughput-oriented fast fail.
            $runPrecomputeNoOutputTimeoutSec = [Math]::Max([int]$runPrecomputeNoOutputTimeoutSec, [int]$stage1DepthFloor)
        }
    }
    $runEvalNoOutputTimeoutSec = [Math]::Max(300, [int]$EvalNoOutputTimeoutSec)
    if ($pointmapSourceNorm -eq "point_head") {
        $runEvalNoOutputTimeoutSec = [Math]::Max(300, [int]$EvalNoOutputTimeoutSecPointHead)
    } elseif ($pointmapSourceNorm -eq "depth_unproject") {
        $runEvalNoOutputTimeoutSec = [Math]::Max(300, [int]$EvalNoOutputTimeoutSecDepthUnproject)
    }
    $precomputeNoOutputRecoveryApplied = $false
    if (($pointmapSourceNorm -eq "point_head") -and ($blendPolicyNorm -eq "weak_to_depth")) {
        $runPrecomputeNoOutputTimeoutSec = [Math]::Max(
            [int]$runPrecomputeNoOutputTimeoutSec,
            [int]$PrecomputeNoOutputTimeoutSecPointHeadWeak
        )
        # Weak lane point_head should not fail too aggressively at 600s.
        $runPrecomputeNoOutputTimeoutSec = [Math]::Max([int]$runPrecomputeNoOutputTimeoutSec, 900)
    }
    if ([bool]$EnableDepthPrecomputeNoOutputPenalty) {
        $skipThreshold = [Math]::Max(1, [int]$DepthPrecomputeNoOutputSkipThreshold)
        if ($depthPrecomputeNoOutputFailCount -ge $skipThreshold) {
            $candidateSkipReason = "depth_precompute_no_output_streak_$depthPrecomputeNoOutputFailCount"
            Write-Host "[ghost-mv] skip candidate due depth precompute no-output streak: mv=$mv mvmask=$mvMask count=$depthPrecomputeNoOutputFailCount skip_threshold=$skipThreshold"
        } else {
            # Recovery pass first: after the first no-output failure on the same depth,
            # try one longer precompute timeout before switching to penalty fast-fail.
            if ([bool]$EnableDepthPrecomputeNoOutputRecovery) {
                $recoveryThreshold = [Math]::Max(1, [int]$DepthPrecomputeNoOutputRecoveryThreshold)
                $recoveryMaxCount = [Math]::Max($recoveryThreshold, [int]$DepthPrecomputeNoOutputRecoveryMaxCount)
                if (($depthPrecomputeNoOutputFailCount -ge $recoveryThreshold) -and ($depthPrecomputeNoOutputFailCount -le $recoveryMaxCount)) {
                    $recoveryTimeout = [Math]::Max(300, [int]$DepthPrecomputeNoOutputRecoveryTimeoutSec)
                    if ($runPrecomputeNoOutputTimeoutSec -lt $recoveryTimeout) {
                        Write-Host "[ghost-mv] depth precompute timeout recovery applied: mv=$mv count=$depthPrecomputeNoOutputFailCount timeout=$runPrecomputeNoOutputTimeoutSec->$recoveryTimeout"
                        $runPrecomputeNoOutputTimeoutSec = $recoveryTimeout
                    }
                    $precomputeNoOutputRecoveryApplied = $true
                }
            }
            if (-not $precomputeNoOutputRecoveryApplied) {
                $penaltyThreshold = [Math]::Max(1, [int]$DepthPrecomputeNoOutputPenaltyThreshold)
                if ($depthPrecomputeNoOutputFailCount -ge $penaltyThreshold) {
                    $penaltyTimeout = [Math]::Max(300, [int]$DepthPrecomputeNoOutputPenaltyTimeoutSec)
                    if ($runPrecomputeNoOutputTimeoutSec -gt $penaltyTimeout) {
                        Write-Host "[ghost-mv] depth precompute timeout penalty applied: mv=$mv count=$depthPrecomputeNoOutputFailCount timeout=$runPrecomputeNoOutputTimeoutSec->$penaltyTimeout"
                        $runPrecomputeNoOutputTimeoutSec = $penaltyTimeout
                    }
                }
            }
        }
    }
    $effectiveNoOutputProbeTimeoutSec = [int]$NoOutputProbeTimeoutSec
    if (($pointmapSourceNorm -eq "point_head") -and ($blendPolicyNorm -eq "weak_to_depth")) {
        $effectiveNoOutputProbeTimeoutSec = [Math]::Max(
            [int]$effectiveNoOutputProbeTimeoutSec,
            [int]$NoOutputProbeTimeoutSecPointHeadWeak
        )
    }
    if ($pointmapSourceNorm -eq "depth_unproject") {
        $effectiveNoOutputProbeTimeoutSec = [Math]::Max(
            [int]$effectiveNoOutputProbeTimeoutSec,
            [int]$NoOutputProbeTimeoutSecDepthUnproject
        )
    }
    $allowPreemptiveProbeBySource = (($pointmapSourceNorm -eq "point_head") -or
        ([bool]$EnablePreemptiveProbeForStrongDepthUnproject -and
         ($pointmapSourceNorm -eq "depth_unproject") -and
         ($blendPolicyNorm -eq "strong_to_depth")))
    $preemptiveCap = [Math]::Max(0, [int]$PreemptiveProbeMaxCandidates)
    if ([bool]$EnableNoOutputProbeRetry -and
        [bool]$EnablePreemptiveProbeForPointHead -and
        ($preemptiveCap -gt 0) -and
        ($candidateIndex -le $preemptiveCap) -and
        $allowPreemptiveProbeBySource -and
        ($noOutputProbeUsedCount -lt [Math]::Max(0, [int]$NoOutputProbeMaxUsesPerStage))) {
        $runMaxFramesShort = [Math]::Max(80, [int]$NoOutputProbeMaxFramesShort)
        $runMaxStepsPerEpoch = [Math]::Max(20, [int]$NoOutputProbeMaxStepsPerEpoch)
        $runModalRunTimeoutSec = [Math]::Max(1800, [int]$NoOutputProbeRunTimeoutSec)
        $probeNoOutputFloor = [int]$ModalRunNoOutputTimeoutSec + 60
        if (($pointmapSourceNorm -eq "point_head") -and ($blendPolicyNorm -eq "weak_to_depth")) {
            # weak_to_depth in point_head path prefers faster failover; no extra +60 padding.
            $probeNoOutputFloor = [int]$ModalRunNoOutputTimeoutSec
        } elseif (($pointmapSourceNorm -eq "depth_unproject") -and
                  (($blendPolicyNorm -eq "strong_to_depth") -or ($blendPolicyNorm -eq "")) -and
                  (([string]$CandidateFamily).Trim().ToLowerInvariant() -match "stage1")) {
            # Overnight P0 fast-fail for stage1 strong depth should obey the
            # advertised stage timeout instead of silently widening by +60s.
            $probeNoOutputFloor = [int]$ModalRunNoOutputTimeoutSec
        }
        $runModalRunNoOutputTimeoutSec = [Math]::Max([int]$probeNoOutputFloor, [int]$effectiveNoOutputProbeTimeoutSec)
        $runModalRunNoOutputMaxRetries = [Math]::Max(0, [int]$NoOutputProbeMaxRetries)
        $noOutputProbeUsedCount += 1
        $preemptiveProbeApplied = $true
        Write-Host "[ghost-mv] preemptive probe profile: mv=$mv mvmask=$mvMask candidate=$candidateIndex/$preemptiveCap source=$pointmapSourceNorm policy=$blendPolicyNorm frames=$runMaxFramesShort steps=$runMaxStepsPerEpoch run_timeout=$runModalRunTimeoutSec no_output_timeout=$runModalRunNoOutputTimeoutSec precompute_no_output_timeout=$runPrecomputeNoOutputTimeoutSec eval_no_output_timeout=$runEvalNoOutputTimeoutSec retries=$runModalRunNoOutputMaxRetries probe_used=$noOutputProbeUsedCount/$NoOutputProbeMaxUsesPerStage"
        Write-Host "[ghost-mv] effective ckpt_wait_timeout_sec=$runCkptWaitTimeoutSec"
    }
    $rc = 0
    $latestFtReason = ""
    $precomputeNoOutputRetryApplied = $false
    $candidateResultPath = ("logs/modal_phase5/candidate_result_ghost_mv_{0}_cand{1:D2}_{2}.json" -f $tag, [int]$candidateIndex, $ts)
    $candidateResultLatest = "logs/modal_phase5/candidate_result_latest.json"
    Write-CandidateResultSeedJsonLocal `
        -Path $candidateResultPath `
        -LatestPath $candidateResultLatest `
        -Seed ([ordered]@{
            candidate_result_json = $candidateResultPath
            run_tag = $tag
            run_timestamp = $ts
            candidate_index = $candidateIndex
            candidate_family = $CandidateFamily
            lane_id = $LaneId
            guard_tier = $GuardTier
            stage_status = "running"
            ft_status = "running"
            exit_code = ""
            ft_failure_reason = ""
            candidate_invalid_reason = ""
            lambda_point_mv_depth = $mv
            lambda_point_mv_mask = $mvMask
            lr = $Lr
            modal_run_quiet = [bool]$ModalRunQuiet
            precompute_source = $PointmapSource
            precompute_source_requested = $PointmapSource
            precompute_source_resolved = ""
            precompute_fallback_used = $false
            precompute_timeout_hit = $false
            support_generation_active = $(if (([string]$PrecomputeMvSupportOn).Trim().ToLowerInvariant() -in @("1","true","yes","y","on")) { 1.0 } else { 0.0 })
            point_support_path_active = 0.0
            point_mv_depth_support_path_active = 0.0
            point_mv_mask_support_path_active = 0.0
            point_target_blend_mv_support_active = 0.0
            precompute_mv_support_on = $PrecomputeMvSupportOn
            point_support_mode = $PointSupportMode
            point_mv_depth_support_mode = $runPointMvDepthSupportMode
            point_mv_mask_support_mode = $runPointMvMaskSupportMode
            use_fg_mask = $UseFgMask
            fg_mask_source = $FgMaskSource
            point_target_mode = $PointTargetMode
            point_target_blend_by_mv_support = $PointTargetBlendByMvSupport
            point_target_blend_mv_region_mode = $PointTargetBlendMvRegionMode
            precompute_mv_support_region_mode = $PrecomputeMvSupportRegionMode
            precompute_mv_support_fg_mask_source = $PrecomputeMvSupportFgMaskSource
            precompute_mv_support_fg_erode_px = [string]$PrecomputeMvSupportFgErodePx
            precompute_mv_support_fg_preserve_px = [string]$PrecomputeMvSupportFgPreservePx
            point_mv_depth_region_mode = $PointMvDepthRegionMode
            fg_supervision_boost = $FgSupervisionBoost
            fg_supervision_bg_floor = $FgSupervisionBgFloor
            fg_supervision_region_mode = $FgSupervisionRegionMode
            fg_supervision_region_erode_px = $FgSupervisionRegionErodePx
            lambda_fg_conf_presence = $LambdaFgConfPresence
            fg_conf_presence_target_ratio = $FgConfPresenceTargetRatio
            lambda_fg_structure_depth_edge = $LambdaFgStructureDepthEdge
            fg_structure_bbox_margin_px = $FgStructureBboxMarginPx
            fg_structure_bbox_min_side_px = $FgStructureBboxMinSidePx
            fg_structure_region_mode = $FgStructureRegionMode
            fg_structure_region_erode_px = $FgStructureRegionErodePx
            fg_structure_depth_edge_warmup_steps = $FgStructureDepthEdgeWarmupSteps
            fg_structure_boundary_probe_px = $FgStructureBoundaryProbePx
            fg_structure_edge_support_mode = $FgStructureEdgeSupportMode
            fg_structure_edge_support_quantile = $FgStructureEdgeSupportQuantile
            fg_structure_edge_support_min_px = $FgStructureEdgeSupportMinPx
            fg_structure_edge_weight_mode = $FgStructureEdgeWeightMode
            fg_structure_boundary_falloff_px = $FgStructureBoundaryFalloffPx
            fg_structure_component_bias_mode = $FgStructureComponentBiasMode
            fg_structure_component_bias_threshold_ratio = $FgStructureComponentBiasThresholdRatio
            fg_structure_component_bias_other_scale = $FgStructureComponentBiasOtherScale
            fg_structure_front_depth_bias_mode = $FgStructureFrontDepthBiasMode
            fg_structure_front_depth_bias_tau = $FgStructureFrontDepthBiasTau
            fg_structure_front_depth_bias_center_quantile = $FgStructureFrontDepthBiasCenterQuantile
            lambda_point_mv_outside_ring = $LambdaPointMvOutsideRing
            point_mv_outside_ring_px = $PointMvOutsideRingPx
            tf32 = [bool]$Tf32
            amp = [bool]$Amp
            strict_deterministic = [bool]$StrictDeterministic
            runner_tf32 = [bool]$Tf32
            runner_amp = [bool]$Amp
            runner_strict_deterministic = [bool]$StrictDeterministic
            precompute_tf32 = [bool]$Tf32
            precompute_amp = [bool]$Amp
            precompute_strict_deterministic = [bool]$StrictDeterministic
            teacher_tf32 = [bool]$Tf32
            teacher_amp = [bool]$Amp
            teacher_deterministic = [bool]$StrictDeterministic
            eval_num_src_views = $evalNumSrcViewsNow
            eval_num_src_views_declared = $evalNumSrcViewsNow
            eval_num_src_views_actual = ""
            eval_num_src_views_mismatch = $false
            cam_count_used = $camCountUsed
            dyn_proxy_enable = $DynProxyEnable
            effective_run_timeout_sec = $runModalRunTimeoutSec
            effective_no_output_timeout_sec = $runModalRunNoOutputTimeoutSec
            effective_precompute_no_output_timeout_sec = $runPrecomputeNoOutputTimeoutSec
            effective_eval_no_output_timeout_sec = $runEvalNoOutputTimeoutSec
            effective_ckpt_wait_timeout_sec = $runCkptWaitTimeoutSec
        })
    if (-not [string]::IsNullOrWhiteSpace($candidateSkipReason)) {
        $rc = 3
        $latestFtReason = "skipped:$candidateSkipReason"
    } else {
        $rc = Invoke-FtLrSweepForCurrentSetting `
            -MvValue $mv `
            -MvMaskValue $mvMask `
            -LocalEvalInferArgsExtra $runEvalInferArgsExtra `
            -LocalConfWeightPerViewQuantile $runConfWeightPerViewQuantile `
            -LocalLambdaPointNormalConsis $runLambdaPointNormalConsis `
            -LocalPointMvDepthSupportMode $runPointMvDepthSupportMode `
            -LocalPointMvDepthSupportFloor $runPointMvDepthSupportFloor `
            -LocalPointMvMaskSupportMode $runPointMvMaskSupportMode `
            -LocalPointMvMaskSupportFloor $runPointMvMaskSupportFloor `
            -LocalMaxFramesShort $runMaxFramesShort `
            -LocalMaxStepsPerEpoch $runMaxStepsPerEpoch `
            -LocalModalRunTimeoutSec $runModalRunTimeoutSec `
            -LocalModalRunNoOutputTimeoutSec $runModalRunNoOutputTimeoutSec `
            -LocalModalRunNoOutputMaxRetries $runModalRunNoOutputMaxRetries `
            -LocalModalRunQuiet $ModalRunQuiet `
            -LocalShortFinetuneAllowQuietNoOutputBypass $runShortFinetuneAllowQuietNoOutputBypass `
            -LocalPrecomputeNoOutputTimeoutSec $runPrecomputeNoOutputTimeoutSec `
            -LocalEvalNoOutputTimeoutSec $runEvalNoOutputTimeoutSec `
            -LocalCkptWaitTimeoutSec $runCkptWaitTimeoutSec
    }
    if ($rc -ne 0) {
        if ([string]::IsNullOrWhiteSpace($latestFtReason)) {
            $latestFtReason = Get-LatestFtSweepFailureReason -SweepCsvPath "logs/modal_phase5/vggt_ft_sweep_latest.csv"
        }
        $isNoOutputFailure = [regex]::IsMatch([string]$latestFtReason, "(?i)(?:heartbeat_stall_timeout|no_output_timeout)_\d+s")
        $isPrecomputeNoOutputFailure = [regex]::IsMatch([string]$latestFtReason, "(?i)precompute.*(?:heartbeat_stall_timeout|no_output_timeout)_\d+s")
        $isDualSourcePrecomputeExhausted = [regex]::IsMatch([string]$latestFtReason, "(?i)precompute_dual_source_no_output_exhausted")
        $precomputeRetryCap = [Math]::Max(0, [int]$PrecomputeNoOutputRetryMaxRetries)
        if ($pointmapSourceNorm -eq "depth_unproject") {
            $familyNorm = ([string]$CandidateFamily).Trim().ToLowerInvariant()
            if ($familyNorm -match "stage1") {
                # For stage1 stabilization, allow up to two retries before giving up.
                $precomputeRetryCap = [Math]::Max($precomputeRetryCap, 2)
            } else {
                # For non-stage1, keep throughput-oriented fast fail.
                $precomputeRetryCap = 0
            }
        }
        if ($isDualSourcePrecomputeExhausted) {
            # The lr-sweep already burned both primary and fallback precompute
            # sources for this same candidate. Re-running another full short
            # finetune is pure churn, so fail fast and move on.
            $precomputeRetryCap = 0
            Write-Host "[ghost-mv] skip same-candidate precompute retry after dual-source exhaustion: mv=$mv mvmask=$mvMask reason=$latestFtReason"
        }
        if ($isNoOutputFailure -and $isPrecomputeNoOutputFailure -and [bool]$EnablePrecomputeNoOutputRetry -and ($precomputeRetryCap -gt 0)) {
            $retryPrecomputeTimeoutTarget = [int]$PrecomputeNoOutputRetryTimeoutSec
            if ($pointmapSourceNorm -eq "point_head") {
                $retryPrecomputeTimeoutTarget = [Math]::Min(
                    [int]$retryPrecomputeTimeoutTarget,
                    [Math]::Max(300, [int]$PrecomputeNoOutputRetryTimeoutSecPointHead)
                )
            }
            $retryPrecomputeTimeout = [Math]::Max(
                [int]$runPrecomputeNoOutputTimeoutSec,
                [Math]::Max(300, [int]$retryPrecomputeTimeoutTarget)
            )
            for ($preRetryAttempt = 1; $preRetryAttempt -le $precomputeRetryCap; $preRetryAttempt++) {
                $precomputeNoOutputRetryApplied = $true
                Write-Host "[ghost-mv] precompute no-output retry: mv=$mv mvmask=$mvMask attempt=$preRetryAttempt/$precomputeRetryCap precompute_timeout=$retryPrecomputeTimeout"
                $preRetryRc = Invoke-FtLrSweepForCurrentSetting `
                    -MvValue $mv `
                    -MvMaskValue $mvMask `
                    -LocalEvalInferArgsExtra $runEvalInferArgsExtra `
                    -LocalConfWeightPerViewQuantile $runConfWeightPerViewQuantile `
                    -LocalLambdaPointNormalConsis $runLambdaPointNormalConsis `
                    -LocalPointMvDepthSupportMode $runPointMvDepthSupportMode `
                    -LocalPointMvDepthSupportFloor $runPointMvDepthSupportFloor `
                    -LocalPointMvMaskSupportMode $runPointMvMaskSupportMode `
                    -LocalPointMvMaskSupportFloor $runPointMvMaskSupportFloor `
                    -LocalMaxFramesShort $runMaxFramesShort `
                    -LocalMaxStepsPerEpoch $runMaxStepsPerEpoch `
                    -LocalModalRunTimeoutSec $runModalRunTimeoutSec `
                    -LocalModalRunNoOutputTimeoutSec $runModalRunNoOutputTimeoutSec `
                    -LocalModalRunNoOutputMaxRetries $runModalRunNoOutputMaxRetries `
                    -LocalModalRunQuiet $ModalRunQuiet `
                    -LocalPrecomputeNoOutputTimeoutSec $retryPrecomputeTimeout `
                    -LocalEvalNoOutputTimeoutSec $runEvalNoOutputTimeoutSec `
                    -LocalCkptWaitTimeoutSec $runCkptWaitTimeoutSec
                if ($preRetryRc -eq 0) {
                    $rc = 0
                    $latestFtReason = ""
                    Write-Host "[ghost-mv] precompute no-output retry succeeded: mv=$mv mvmask=$mvMask"
                    break
                }
                $rc = $preRetryRc
                $latestFtReason = Get-LatestFtSweepFailureReason -SweepCsvPath "logs/modal_phase5/vggt_ft_sweep_latest.csv"
                $isNoOutputFailure = [regex]::IsMatch([string]$latestFtReason, "(?i)(?:heartbeat_stall_timeout|no_output_timeout)_\d+s")
                $isPrecomputeNoOutputFailure = [regex]::IsMatch([string]$latestFtReason, "(?i)precompute.*(?:heartbeat_stall_timeout|no_output_timeout)_\d+s")
                Write-Host "[ghost-mv] precompute no-output retry failed: mv=$mv mvmask=$mvMask attempt=$preRetryAttempt/$precomputeRetryCap rc=$preRetryRc reason=$latestFtReason"
                if (-not ($isNoOutputFailure -and $isPrecomputeNoOutputFailure)) {
                    break
                }
            }
        }
        if ($rc -ne 0 -and $isPrecomputeNoOutputFailure) {
            $depthPrecomputeNoOutputFailCount = [Math]::Max(0, $depthPrecomputeNoOutputFailCount) + 1
            $depthPrecomputeNoOutputFailures[$depthKey] = $depthPrecomputeNoOutputFailCount
            Write-Host "[ghost-mv] depth precompute no-output streak: mv=$mv count=$depthPrecomputeNoOutputFailCount"
        }
        if ($rc -ne 0 -and $isNoOutputFailure -and $isPrecomputeNoOutputFailure -and (-not $precomputeNoOutputRetryApplied)) {
            Write-Host "[ghost-mv] skip same-candidate probe retry on precompute no-output: mv=$mv mvmask=$mvMask reason=$latestFtReason"
        } elseif ($rc -ne 0 -and $isNoOutputFailure -and $isPrecomputeNoOutputFailure -and $precomputeNoOutputRetryApplied) {
            Write-Host "[ghost-mv] precompute no-output retry exhausted: mv=$mv mvmask=$mvMask reason=$latestFtReason"
        }
        $allowReactiveProbeRetry = (-not ([bool]$SkipReactiveProbeAfterPreemptive -and $preemptiveProbeApplied))
        if ($isNoOutputFailure -and (-not $isPrecomputeNoOutputFailure) -and [bool]$EnableNoOutputProbeRetry -and $allowReactiveProbeRetry -and ($noOutputProbeUsedCount -lt [Math]::Max(0, [int]$NoOutputProbeMaxUsesPerStage)) -and ($infraNoOutputConsecutive -ge ([Math]::Max(1, [int]$NoOutputProbeTriggerConsecutive) - 1))) {
            $probeMaxFrames = [Math]::Max(80, [int]$NoOutputProbeMaxFramesShort)
            $probeMaxSteps = [Math]::Max(20, [int]$NoOutputProbeMaxStepsPerEpoch)
            $probeRunTimeout = [Math]::Max(1800, [int]$NoOutputProbeRunTimeoutSec)
            $probeNoOutputFloor = [int]$ModalRunNoOutputTimeoutSec + 60
            if (($pointmapSourceNorm -eq "point_head") -and ($blendPolicyNorm -eq "weak_to_depth")) {
                $probeNoOutputFloor = [int]$ModalRunNoOutputTimeoutSec
            } elseif (($pointmapSourceNorm -eq "depth_unproject") -and
                      (($blendPolicyNorm -eq "strong_to_depth") -or ($blendPolicyNorm -eq "")) -and
                      (([string]$CandidateFamily).Trim().ToLowerInvariant() -match "stage1")) {
                $probeNoOutputFloor = [int]$ModalRunNoOutputTimeoutSec
            }
            $probeNoOutputTimeout = [Math]::Max([int]$probeNoOutputFloor, [int]$effectiveNoOutputProbeTimeoutSec)
            $probeRetries = [Math]::Max(0, [int]$NoOutputProbeMaxRetries)
            $noOutputProbeUsedCount += 1
            Write-Host "[ghost-mv] probe retry on no-output: mv=$mv mvmask=$mvMask frames=$probeMaxFrames steps=$probeMaxSteps run_timeout=$probeRunTimeout no_output_timeout=$probeNoOutputTimeout retries=$probeRetries probe_used=$noOutputProbeUsedCount/$NoOutputProbeMaxUsesPerStage"
            $probeRc = Invoke-FtLrSweepForCurrentSetting `
                -MvValue $mv `
                -MvMaskValue $mvMask `
                -LocalEvalInferArgsExtra $runEvalInferArgsExtra `
                -LocalConfWeightPerViewQuantile $runConfWeightPerViewQuantile `
                -LocalLambdaPointNormalConsis $runLambdaPointNormalConsis `
                -LocalPointMvDepthSupportMode $runPointMvDepthSupportMode `
                -LocalPointMvDepthSupportFloor $runPointMvDepthSupportFloor `
                -LocalPointMvMaskSupportMode $runPointMvMaskSupportMode `
                -LocalPointMvMaskSupportFloor $runPointMvMaskSupportFloor `
                -LocalMaxFramesShort $probeMaxFrames `
                -LocalMaxStepsPerEpoch $probeMaxSteps `
                -LocalModalRunTimeoutSec $probeRunTimeout `
                -LocalModalRunNoOutputTimeoutSec $probeNoOutputTimeout `
                -LocalModalRunNoOutputMaxRetries $probeRetries `
                -LocalModalRunQuiet $ModalRunQuiet `
                -LocalPrecomputeNoOutputTimeoutSec $runPrecomputeNoOutputTimeoutSec `
                -LocalEvalNoOutputTimeoutSec $runEvalNoOutputTimeoutSec `
                -LocalCkptWaitTimeoutSec $runCkptWaitTimeoutSec
            if ($probeRc -eq 0) {
                $rc = 0
                $latestFtReason = ""
                Write-Host "[ghost-mv] probe retry succeeded: mv=$mv mvmask=$mvMask"
            } else {
                $rc = $probeRc
                $latestFtReason = Get-LatestFtSweepFailureReason -SweepCsvPath "logs/modal_phase5/vggt_ft_sweep_latest.csv"
                Write-Host "[ghost-mv] probe retry failed: rc=$probeRc reason=$latestFtReason"
            }
            $isNoOutputFailure = [regex]::IsMatch([string]$latestFtReason, "(?i)(?:heartbeat_stall_timeout|no_output_timeout)_\d+s")
        } elseif ($isNoOutputFailure -and (-not $isPrecomputeNoOutputFailure) -and [bool]$EnableNoOutputProbeRetry -and (-not $allowReactiveProbeRetry)) {
            Write-Host "[ghost-mv] skip same-candidate probe retry after preemptive profile: mv=$mv mvmask=$mvMask reason=$latestFtReason"
        }

        if ($rc -ne 0 -and $isNoOutputFailure) {
            $infraNoOutputConsecutive += 1
            Write-Host "[ghost-mv] infra no-output failure streak=$infraNoOutputConsecutive reason=$latestFtReason"
            if ($infraNoOutputConsecutive -ge [Math]::Max(1, [int]$InfraNoOutputStopConsecutive)) {
                $stopDueInfraNoOutput = $true
            }
        } elseif ($rc -ne 0) {
            $infraNoOutputConsecutive = 0
        }
    } else {
        $infraNoOutputConsecutive = 0
        # Keep depth-level no-output streak within the same stage window so
        # subsequent candidates can apply penalty timeout instead of repeating
        # full precompute waits after fallback-driven success.
    }

    $cmpLatest = "logs/modal_phase5/baseline_compare_latest.csv"
    $gateLatest = "logs/modal_phase5/vggt_ft_gate_latest.json"
    $sweepLatest = "logs/modal_phase5/vggt_ft_sweep_latest.csv"
    $cmpOut = "logs/modal_phase5/baseline_compare_ghost_mv_${tag}_$ts.csv"
    $sweepOut = "logs/modal_phase5/vggt_ft_sweep_ghost_mv_${tag}_$ts.csv"
    if (Test-Path $cmpLatest) { Copy-Item $cmpLatest $cmpOut -Force }
    if (Test-Path $sweepLatest) { Copy-Item $sweepLatest $sweepOut -Force }
    $latestFtRow = Get-LatestFtSweepRow -SweepCsvPath $sweepLatest
    $latestFtStatus = ""
    $precomputeSourceRequested = ""
    $precomputeSourceResolved = ""
    $precomputeFallbackUsed = $false
    $precomputeTimeoutHit = $false
    if ($latestFtRow -ne $null) {
        try { $latestFtStatus = [string]$latestFtRow.status } catch { $latestFtStatus = "" }
        try {
            if ($latestFtRow.PSObject.Properties["pointmap_source_requested"]) {
                $precomputeSourceRequested = [string]$latestFtRow.pointmap_source_requested
            }
        } catch {}
        try {
            if ($latestFtRow.PSObject.Properties["pointmap_source_resolved"]) {
                $precomputeSourceResolved = [string]$latestFtRow.pointmap_source_resolved
            }
        } catch {}
        try {
            if ($latestFtRow.PSObject.Properties["precompute_fallback_used"]) {
                $precomputeFallbackUsed = To-BoolLoose -Value $latestFtRow.precompute_fallback_used -Default $false
            }
        } catch {}
        try {
            if ($latestFtRow.PSObject.Properties["precompute_timeout_hit"]) {
                $precomputeTimeoutHit = To-BoolLoose -Value $latestFtRow.precompute_timeout_hit -Default $false
            }
        } catch {}
    }

    if ($rc -ne 0) {
        try {
            $failReason = [string]$latestFtReason
            if ([string]::IsNullOrWhiteSpace($failReason)) {
                $failReason = "lr_sweep_failed_rc_$rc"
            }
            Write-JsonNoBom -Path $gateLatest -Obj ([ordered]@{
                    updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
                    status = "failed"
                    source = "run_vggt_ghost_mvdepth_sweep"
                    exit_code = $rc
                    reason = $failReason
                    lambda_point_mv_depth = $mv
                    lambda_point_mv_mask = $mvMask
                    infra_no_output_consecutive = $infraNoOutputConsecutive
                })
        } catch {
        }
    }

    $best = $null
    if ($rc -eq 0) {
        $best = Resolve-BestMetrics `
            -CompareCsvPath $cmpLatest `
            -GateJsonPath $gateLatest `
            -RetryCount $MetricReadRetry `
            -RetrySleepSec $MetricReadRetrySleepSec
        if ($best -eq $null) {
            $rc = 10
            $latestFtReason = "post_eval_no_valid_metrics(compare/gate had no ok rows)"
            Write-Host "[ghost-mv] no valid metrics row after eval: mv=$mv mvmask=$mvMask"
        }
        if ([bool]$DisallowResumeFallbackResult -and ($latestFtStatus -eq "ok_fallback")) {
            $rc = 8
            $latestFtReason = "resume_fallback_blocked(status=ok_fallback)"
            Write-Host "[ghost-mv] disallow fallback result: mv=$mv mvmask=$mvMask ablation=$ablationGroupId"
        }
    } else {
        Write-Host "[ghost-mv] skip metric parse because lr-sweep failed (rc=$rc)."
    }

    $ghostMean = ""
    $ghostP95 = ""
    $ghostSoftMean = ""
    $ghostSoftP95 = ""
    $ghostVisualScore = ""
    $predLumaMean = ""
    $predNonBlackRatio008 = ""
    $predNonBlackRatio015 = ""
    $fgPredLumaMean = ""
    $fgPredNonBlackRatio = ""
    $fgPredContrast = ""
    $fgPredTgtL1 = ""
    $widthRatioMean = ""
    $areaRatioMean = ""
    $visualGuardBlocked = $false
    $visualGuardReason = ""
    $ghostRowsCsv = ""
    $ghostSummaryCsv = ""
    $evalNumSrcViewsActual = ""
    $evalNumSrcViewsMismatch = $false
    $qualityGuardBlocked = $false
    $qualityGuardReason = ""
    if ($best -ne $null) {
        $evalNumSrcViewsActual = Resolve-ActualEvalNumSrcViews -BestRow $best
        if ([string]::IsNullOrWhiteSpace($evalNumSrcViewsActual) -and $latestFtRow -ne $null) {
            $evalNumSrcViewsActual = Resolve-ActualEvalNumSrcViews -BestRow $latestFtRow
        }
        if ((-not [string]::IsNullOrWhiteSpace($evalNumSrcViewsNow)) -and (-not [string]::IsNullOrWhiteSpace($evalNumSrcViewsActual))) {
            $evalNumSrcViewsMismatch = ([string]$evalNumSrcViewsNow -ne [string]$evalNumSrcViewsActual)
            if ($evalNumSrcViewsMismatch) {
                $rc = 9
                $latestFtReason = "eval_num_src_views_mismatch(requested=$evalNumSrcViewsNow actual=$evalNumSrcViewsActual)"
                Write-Host "[ghost-mv] reject candidate by eval view mismatch: $latestFtReason"
            }
        }
        if ($rc -eq 0) {
            $outVolPath = [string]$best.infer_out_volume_path
            $remoteFiles = @(Get-RemoteCatFgFiles -OutVolPath $outVolPath -MaxCount 3)
            if ($remoteFiles.Count -gt 0) {
                $ghostLocalDir = "logs/modal_phase5/_ghost_eval_${tag}_$ts"
                New-Item -ItemType Directory -Force -Path $ghostLocalDir | Out-Null
                foreach ($rf in $remoteFiles) {
                    $remoteName = [string]$rf.Filename
                    $base = [System.IO.Path]::GetFileName($remoteName)
                    $localPath = Join-Path $ghostLocalDir $base
                    [void](Volume-GetSafe -RemoteFile $remoteName -LocalPath $localPath)
                }
                $pattern = (Join-Path $ghostLocalDir "*.png").Replace("\", "/")
                $ghostCsv = "logs/modal_phase5/ghost_score_rows_${tag}_$ts.csv"
                $ghostSummaryCsv = "logs/modal_phase5/ghost_score_summary_${tag}_$ts.csv"
                $ghostJson = "logs/modal_phase5/ghost_score_${tag}_$ts.json"
                python "$CodeDir\tools\score_ghosting_from_cat_pred.py" `
                    --input "$tag=$pattern" `
                    --out_csv "$ghostCsv" `
                    --out_summary_csv "$ghostSummaryCsv" `
                    --out_json "$ghostJson" | Out-Null
                if (Test-Path $ghostCsv) {
                    $ghostRowsCsv = $ghostCsv
                }
                if (Test-Path $ghostSummaryCsv) {
                    $grows = @(Import-Csv $ghostSummaryCsv)
                    if ($grows.Count -gt 0) {
                        $ghostMean = [double]$grows[0].ghost_score_mean
                        $ghostP95 = [double]$grows[0].ghost_score_p95
                        if ($grows[0].PSObject.Properties["ghost_soft_score_mean"]) {
                            $ghostSoftMean = [double]$grows[0].ghost_soft_score_mean
                        }
                        if ($grows[0].PSObject.Properties["ghost_soft_score_p95"]) {
                            $ghostSoftP95 = [double]$grows[0].ghost_soft_score_p95
                        }
                        if ($grows[0].PSObject.Properties["ghost_visual_score_mean"]) {
                            $ghostVisualScore = [double]$grows[0].ghost_visual_score_mean
                        }
                        if ($grows[0].PSObject.Properties["pred_luma_mean_mean"]) {
                            $predLumaMean = [double]$grows[0].pred_luma_mean_mean
                        }
                        if ($grows[0].PSObject.Properties["pred_nonblack_ratio_thr008_mean"]) {
                            $predNonBlackRatio008 = [double]$grows[0].pred_nonblack_ratio_thr008_mean
                        }
                        if ($grows[0].PSObject.Properties["pred_nonblack_ratio_thr015_mean"]) {
                            $predNonBlackRatio015 = [double]$grows[0].pred_nonblack_ratio_thr015_mean
                        }
                        if ($grows[0].PSObject.Properties["fg_pred_luma_mean_mean"]) {
                            $fgPredLumaMean = [double]$grows[0].fg_pred_luma_mean_mean
                        }
                        if ($grows[0].PSObject.Properties["fg_pred_nonblack_ratio_mean"]) {
                            $fgPredNonBlackRatio = [double]$grows[0].fg_pred_nonblack_ratio_mean
                        }
                        if ($grows[0].PSObject.Properties["fg_pred_contrast_mean"]) {
                            $fgPredContrast = [double]$grows[0].fg_pred_contrast_mean
                        }
                        if ($grows[0].PSObject.Properties["fg_pred_tgt_l1_mean"]) {
                            $fgPredTgtL1 = [double]$grows[0].fg_pred_tgt_l1_mean
                        }
                        if ($grows[0].PSObject.Properties["width_ratio_mean"]) {
                            $widthRatioMean = [double]$grows[0].width_ratio_mean
                        }
                        if ($grows[0].PSObject.Properties["area_ratio_mean"]) {
                            $areaRatioMean = [double]$grows[0].area_ratio_mean
                        }
                    }
                }
            }
        }
    }

    if (($rc -eq 0) -and [bool]$EnableVisualAntiBlackGuard) {
        $vReasons = New-Object System.Collections.Generic.List[string]
        $lumaNow = To-DoubleOrNaN($predLumaMean)
        $nonBlackNow = To-DoubleOrNaN($predNonBlackRatio008)
        $widthNow = To-DoubleOrNaN($widthRatioMean)
        $areaNow = To-DoubleOrNaN($areaRatioMean)
        if ([double]::IsNaN($lumaNow) -or ($lumaNow -lt [double]$MinPredLumaMean)) {
            $vReasons.Add("pred_luma_mean<$MinPredLumaMean (now=$lumaNow)") | Out-Null
        }
        if ([double]::IsNaN($nonBlackNow) -or ($nonBlackNow -lt [double]$MinPredNonBlackRatio)) {
            $vReasons.Add("pred_nonblack_ratio_thr008<$MinPredNonBlackRatio (now=$nonBlackNow)") | Out-Null
        }
        if ([double]::IsNaN($areaNow) -or ($areaNow -lt [double]$MinAreaRatio)) {
            $vReasons.Add("area_ratio<$MinAreaRatio (now=$areaNow)") | Out-Null
        }
        if ([double]::IsNaN($widthNow) -or ($widthNow -lt [double]$MinWidthRatio)) {
            $vReasons.Add("width_ratio<$MinWidthRatio (now=$widthNow)") | Out-Null
        }
        if ($vReasons.Count -gt 0) {
            $visualGuardBlocked = $true
            $visualGuardReason = ($vReasons -join "; ")
            $rc = 7
            $latestFtReason = "visual_guard_failed: $visualGuardReason"
            Write-Host "[ghost-mv] visual anti-black guard rejected candidate: mv=$mv mvmask=$mvMask reason=$latestFtReason"
        }
    }
    if ([string]::IsNullOrWhiteSpace([string]$ghostVisualScore) -and (-not [string]::IsNullOrWhiteSpace([string]$ghostMean))) {
        $ghostVisualScore = $ghostMean
    }

    $psnrNow = [double]::NaN
    $ssimNow = [double]::NaN
    $wl1Now = [double]::NaN
    if ($best -ne $null) {
        $psnrNow = To-DoubleOrNaN($best.mean_PSNR)
        $ssimNow = To-DoubleOrNaN($best.mean_SSIM)
        $wl1Now = To-DoubleOrNaN($best.mean_weighted_L1)
    }
    if (($rc -eq 0) -and [bool]$EnableAbsoluteQualityGuard) {
        $absQualityReasons = New-Object System.Collections.Generic.List[string]
        if ([double]::IsNaN($psnrNow) -or ($psnrNow -lt [double]$MinPSNRGuard)) {
            $absQualityReasons.Add("psnr<$([double]$MinPSNRGuard) (now=$psnrNow)") | Out-Null
        }
        if (([double]$MinSSIMGuard -gt 0) -and ([double]::IsNaN($ssimNow) -or ($ssimNow -lt [double]$MinSSIMGuard))) {
            $absQualityReasons.Add("ssim<$([double]$MinSSIMGuard) (now=$ssimNow)") | Out-Null
        }
        if (([double]$MaxWl1Guard -gt 0) -and ([double]::IsNaN($wl1Now) -or ($wl1Now -gt [double]$MaxWl1Guard))) {
            $absQualityReasons.Add("wl1>$([double]$MaxWl1Guard) (now=$wl1Now)") | Out-Null
        }
        if ($absQualityReasons.Count -gt 0) {
            $qualityGuardBlocked = $true
            $qualityGuardReason = ($absQualityReasons -join "; ")
            $rc = 6
            $latestFtReason = "absolute_quality_guard_failed: " + $qualityGuardReason
            Write-Host "[ghost-mv] absolute quality guard rejected candidate: mv=$mv mvmask=$mvMask reason=$latestFtReason"
        }
    }

    $precomputeSource = [string]$precomputeSourceResolved
    if ([string]::IsNullOrWhiteSpace($precomputeSource)) {
        $precomputeSource = [string]$precomputeSourceRequested
    }
    $candidateInvalidReason = Resolve-CandidateInvalidReason `
        -ExitCode $rc `
        -FailureReason $latestFtReason `
        -VisualGuardBlocked:$visualGuardBlocked `
        -QualityGuardBlocked:$qualityGuardBlocked `
        -EvalNumSrcViewsMismatch:$evalNumSrcViewsMismatch `
        -LatestFtRow $latestFtRow

    $rows += [pscustomobject]@{
        lambda_point_mv_depth = $mv
        lambda_point_mv_mask = $mvMask
        lambda_conf = $LambdaConf
        lambda_conf_warmup_steps = $LambdaConfWarmupSteps
        lambda_cam = $LambdaCam
        lambda_cam_warmup_steps = $LambdaCamWarmupSteps
        lr = $Lr
        lr_head_scale = $LrHeadScale
        grad_clip = $GradClip
        precompute_mv_support_on = $PrecomputeMvSupportOn
        support_generation_active = $(if (([string]$PrecomputeMvSupportOn).Trim().ToLowerInvariant() -in @("1","true","yes","y","on")) { 1.0 } else { 0.0 })
        precompute_mv_support_tol_abs = $PrecomputeMvSupportTolAbs
        precompute_mv_support_tol_rel = $PrecomputeMvSupportTolRel
        precompute_mv_support_stride = $PrecomputeMvSupportStride
        precompute_mv_support_mode = $PrecomputeMvSupportMode
        precompute_mv_support_floor = $PrecomputeMvSupportFloor
        precompute_mv_support_gamma = $PrecomputeMvSupportGamma
        precompute_mv_support_clip_thr = $PrecomputeMvSupportClipThr
        precompute_mv_support_clip_floor = $PrecomputeMvSupportClipFloor
        precompute_mv_support_hard_thr = $PrecomputeMvSupportHardThr
        precompute_mv_conf_valid_floor = $PrecomputeMvConfValidFloor
        precompute_mv_support_region_mode = $PrecomputeMvSupportRegionMode
        precompute_mv_support_fg_mask_source = $PrecomputeMvSupportFgMaskSource
        precompute_mv_support_fg_erode_px = [string]$PrecomputeMvSupportFgErodePx
        precompute_mv_support_fg_preserve_px = [string]$PrecomputeMvSupportFgPreservePx
        point_target_consensus_alpha_floor = $PointTargetConsensusAlphaFloor
        use_fg_mask = $UseFgMask
        fg_mask_source = $FgMaskSource
        point_target_mode = $PointTargetMode
        point_target_blend_by_mv_support = $PointTargetBlendByMvSupport
        point_target_blend_mv_region_mode = $PointTargetBlendMvRegionMode
        point_mv_depth_min_tgt_valid_ratio = $PointMvDepthMinTgtValidRatio
        point_mv_depth_region_mode = $PointMvDepthRegionMode
        point_mv_mask_min_tgt_fg_ratio = $PointMvMaskMinTgtFgRatio
        point_mv_mask_hit_thr = $PointMvMaskHitThr
        point_mv_mask_soft_blur_px = $PointMvMaskSoftBlurPx
        point_mv_mask_soft_blur_iters = $PointMvMaskSoftBlurIters
        point_mv_mask_soft_mix = $PointMvMaskSoftMix
        point_mv_mask_soft_hit_thr = $PointMvMaskSoftHitThr
        conf_weight_per_view_quantile = $runConfWeightPerViewQuantile
        conf_weight_per_view_min_valid = $ConfWeightPerViewMinValid
        lambda_point_normal_consis = $runLambdaPointNormalConsis
        point_normal_consis_warmup_steps = $PointNormalConsisWarmupSteps
        point_mv_depth_pair_mode = $PointMvDepthPairMode
        point_mv_stride = $PointMvStride
        point_support_mode = $PointSupportMode
        point_mv_depth_support_mode = $runPointMvDepthSupportMode
        point_mv_depth_support_floor = $runPointMvDepthSupportFloor
        point_mv_mask_support_mode = $runPointMvMaskSupportMode
        point_mv_mask_support_floor = $runPointMvMaskSupportFloor
        point_mv_depth_tgt_valid_mode = $PointMvDepthTgtValidMode
        point_target_blend_mv_policy = $PointTargetBlendMvPolicy
        fg_supervision_boost = $FgSupervisionBoost
        fg_supervision_bg_floor = $FgSupervisionBgFloor
        fg_supervision_region_mode = $FgSupervisionRegionMode
        fg_supervision_region_erode_px = $FgSupervisionRegionErodePx
        lambda_fg_conf_presence = $LambdaFgConfPresence
        fg_conf_presence_target_ratio = $FgConfPresenceTargetRatio
        lambda_fg_structure_depth_edge = $LambdaFgStructureDepthEdge
        fg_structure_bbox_margin_px = $FgStructureBboxMarginPx
        fg_structure_bbox_min_side_px = $FgStructureBboxMinSidePx
        fg_structure_region_mode = $FgStructureRegionMode
        fg_structure_region_erode_px = $FgStructureRegionErodePx
        fg_structure_depth_edge_warmup_steps = $FgStructureDepthEdgeWarmupSteps
        fg_structure_boundary_probe_px = $FgStructureBoundaryProbePx
        fg_structure_edge_support_mode = $FgStructureEdgeSupportMode
        fg_structure_edge_support_quantile = $FgStructureEdgeSupportQuantile
        fg_structure_edge_support_min_px = $FgStructureEdgeSupportMinPx
        fg_structure_edge_weight_mode = $FgStructureEdgeWeightMode
        fg_structure_boundary_falloff_px = $FgStructureBoundaryFalloffPx
        fg_structure_component_bias_mode = $FgStructureComponentBiasMode
        fg_structure_component_bias_threshold_ratio = $FgStructureComponentBiasThresholdRatio
        fg_structure_component_bias_other_scale = $FgStructureComponentBiasOtherScale
        fg_structure_front_depth_bias_mode = $FgStructureFrontDepthBiasMode
        fg_structure_front_depth_bias_tau = $FgStructureFrontDepthBiasTau
        fg_structure_front_depth_bias_center_quantile = $FgStructureFrontDepthBiasCenterQuantile
        lambda_point_mv_outside_ring = $LambdaPointMvOutsideRing
        point_mv_outside_ring_px = $PointMvOutsideRingPx
        eval_num_src_views = $evalNumSrcViewsNow
        eval_num_src_views_declared = $evalNumSrcViewsNow
        eval_num_src_views_actual = $evalNumSrcViewsActual
        eval_num_src_views_mismatch = [bool]$evalNumSrcViewsMismatch
        cam_count_used = $camCountUsed
        ft_status = $latestFtStatus
        exit_code = $rc
        ft_failure_reason = $latestFtReason
        candidate_invalid_reason = $candidateInvalidReason
        precompute_source = $precomputeSource
        precompute_source_requested = $precomputeSourceRequested
        precompute_source_resolved = $precomputeSourceResolved
        precompute_fallback_used = [bool]$precomputeFallbackUsed
        precompute_timeout_hit = [bool]$precomputeTimeoutHit
        infra_no_output_consecutive = $infraNoOutputConsecutive
        candidate_index = $candidateIndex
        ablation_group_id = $ablationGroupId
        ablation_conf_quantile = $runConfWeightPerViewQuantile
        ablation_lambda_point_normal_consis = $runLambdaPointNormalConsis
        ablation_support_split = [bool]$ablationSupportSplit
        dyn_proxy_enable = $DynProxyEnable
        dyn_proxy_mode = $DynProxyMode
        dyn_proxy_use_gram = $DynProxyUseGram
        dyn_proxy_use_support = $DynProxyUseSupport
        dyn_proxy_floor = $DynProxyFloor
        dyn_proxy_warmup_steps = $DynProxyWarmupSteps
        candidate_skip_reason = $candidateSkipReason
        depth_precompute_no_output_failures = $depthPrecomputeNoOutputFailCount
        precompute_no_output_recovery_applied = $precomputeNoOutputRecoveryApplied
        precompute_no_output_retry_applied = $precomputeNoOutputRetryApplied
        preemptive_probe_applied = $preemptiveProbeApplied
        effective_run_timeout_sec = $runModalRunTimeoutSec
        effective_no_output_timeout_sec = $runModalRunNoOutputTimeoutSec
        effective_precompute_no_output_timeout_sec = $runPrecomputeNoOutputTimeoutSec
        effective_eval_no_output_timeout_sec = $runEvalNoOutputTimeoutSec
        effective_ckpt_wait_timeout_sec = $runCkptWaitTimeoutSec
        best_source = $(if ($best) { [string]$best.best_source } else { "" })
        best_label = $(if ($best) { [string]$best.label } else { "" })
        best_geom_subdir = $(if ($best) { [string]$best.geom_subdir } else { "" })
        mean_PSNR = $(if ($best) { [double]$best.mean_PSNR } else { "" })
        mean_SSIM = $(if ($best) { [double]$best.mean_SSIM } else { "" })
        mean_weighted_L1 = $(if ($best) { [double]$best.mean_weighted_L1 } else { "" })
        ghost_score_mean = $ghostMean
        ghost_score_p95 = $ghostP95
        baseline_compare_csv = $cmpOut
        sweep_csv = $sweepOut
        ghost_rows_csv = $ghostRowsCsv
        ghost_summary_csv = $ghostSummaryCsv
        lane_id = $LaneId
        candidate_family = $CandidateFamily
        guard_tier = $GuardTier
        rollback_triggered = [bool]$RollbackTriggered
        visual_guard_blocked = [bool]$visualGuardBlocked
        visual_guard_reason = $visualGuardReason
        quality_guard_blocked = [bool]$qualityGuardBlocked
        quality_guard_reason = $qualityGuardReason
        pred_luma_mean = $predLumaMean
        pred_nonblack_ratio_thr008 = $predNonBlackRatio008
        pred_nonblack_ratio_thr015 = $predNonBlackRatio015
        fg_pred_luma_mean = $fgPredLumaMean
        fg_pred_nonblack_ratio = $fgPredNonBlackRatio
        fg_pred_contrast = $fgPredContrast
        fg_pred_tgt_l1 = $fgPredTgtL1
        area_ratio_mean = $areaRatioMean
        width_ratio_mean = $widthRatioMean
        ghost_visual_score = $ghostVisualScore
        ghost_soft_score = $ghostSoftMean
        ghost_soft_score_mean = $ghostSoftMean
        ghost_soft_score_p95 = $ghostSoftP95
    }
    Write-Host "[ghost-mv] lambda_point_mv_depth=$mv lambda_point_mv_mask=$mvMask rc=$rc"

    $rowRef = $rows[$rows.Count - 1]
    if ($rowRef -ne $null) {
        Copy-SelectedPropertiesLocal `
            -Source $latestFtRow `
            -Target $rowRef `
            -Names @(
                "support_generation_active",
                "point_support_path_active",
                "point_mv_depth_support_path_active",
                "point_mv_mask_support_path_active",
                "point_target_blend_mv_support_active",
                "point_mv_mode",
                "point_mv_support_mean","point_mv_support_p10","point_mv_support_p90",
                "point_mv_support_fg_mean","point_mv_support_fg_p10","point_mv_support_fg_p90",
                "point_mv_support_bg_mean","point_mv_support_bg_p10","point_mv_support_bg_p90",
                "mv_support_raw_mean","mv_support_valid_ratio","mv_support_fg_valid_ratio","mv_support_bg_valid_ratio",
                "mv_support_pair_count_eff","mv_support_conf_mean","mv_support_nan_ratio","depth_conf_delta_mean",
                "mv_support_fg_mean","mv_support_bg_mean","depth_conf_delta_fg_mean","depth_conf_delta_bg_mean",
                "depth_conf_fg_preserved_active","depth_conf_fg_preserve_px","depth_conf_fg_exact_ratio",
                "depth_conf_fg_preserve_ratio","depth_conf_fg_raw_mean","depth_conf_fg_after_support_mean",
                "depth_conf_fg_final_mean",
                "mv_support_generation_region_mode","mv_support_generation_fg_mask_source",
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
                "lambda_fg_conf_presence","fg_conf_presence_target_ratio",
                "fg_conf_presence_enabled","fg_conf_presence_pred_mean","fg_conf_presence_tgt_mean",
                "fg_conf_presence_target_floor","fg_conf_presence_active_ratio","fg_conf_presence_loss",
                "loss_fg_conf_presence","loss_contrib_fg_conf_presence","mean_loss_fg_conf_presence",
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
                "point_mv_mask_support_eff_bg_mean","point_mv_mask_support_eff_bg_p10","point_mv_mask_support_eff_bg_p90"
            )
        $supportMetricDefaults = [ordered]@{
            support_generation_active = 0.0
            point_support_path_active = 0.0
            point_mv_depth_support_path_active = 0.0
            point_mv_mask_support_path_active = 0.0
            point_target_blend_mv_support_active = 0.0
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
            fg_supervision_boost = [double]::NaN
            fg_supervision_boost_applied = [double]::NaN
            fg_supervision_bg_floor = [double]::NaN
            fg_supervision_region_mode = ""
            fg_supervision_region_erode_px = [double]::NaN
            fg_supervision_boost_cover = [double]::NaN
            fg_supervision_boost_cover_ratio_in_fg = [double]::NaN
            fg_supervision_boundary_ring_cover = [double]::NaN
            fg_supervision_boundary_ring_ratio_in_fg = [double]::NaN
            fg_supervision_boost_mask_mean = [double]::NaN
            fg_supervision_boost_mask_p10 = [double]::NaN
            fg_supervision_boost_mask_p90 = [double]::NaN
            fg_supervision_boost_mask_fg_mean = [double]::NaN
            fg_supervision_boost_mask_fg_p10 = [double]::NaN
            fg_supervision_boost_mask_fg_p90 = [double]::NaN
            fg_supervision_boost_mask_bg_mean = [double]::NaN
            fg_supervision_boost_mask_bg_p10 = [double]::NaN
            fg_supervision_boost_mask_bg_p90 = [double]::NaN
            fg_supervision_boundary_ring_mean = [double]::NaN
            fg_supervision_boundary_ring_p10 = [double]::NaN
            fg_supervision_boundary_ring_p90 = [double]::NaN
            fg_supervision_boundary_ring_fg_mean = [double]::NaN
            fg_supervision_boundary_ring_fg_p10 = [double]::NaN
            fg_supervision_boundary_ring_fg_p90 = [double]::NaN
            fg_supervision_boundary_ring_bg_mean = [double]::NaN
            fg_supervision_boundary_ring_bg_p10 = [double]::NaN
            fg_supervision_boundary_ring_bg_p90 = [double]::NaN
            fg_supervision_profile_mean = [double]::NaN
            fg_supervision_profile_p10 = [double]::NaN
            fg_supervision_profile_p90 = [double]::NaN
            fg_supervision_profile_fg_mean = [double]::NaN
            fg_supervision_profile_fg_p10 = [double]::NaN
            fg_supervision_profile_fg_p90 = [double]::NaN
            fg_supervision_profile_bg_mean = [double]::NaN
            fg_supervision_profile_bg_p10 = [double]::NaN
            fg_supervision_profile_bg_p90 = [double]::NaN
            fg_supervision_weight_mean = [double]::NaN
            supervision_valid_cover = [double]::NaN
            supervision_valid_fg_mean = [double]::NaN
            supervision_valid_bg_mean = [double]::NaN
            lambda_fg_conf_presence = [double]::NaN
            fg_conf_presence_target_ratio = [double]::NaN
            fg_conf_presence_enabled = [double]::NaN
            fg_conf_presence_pred_mean = [double]::NaN
            fg_conf_presence_tgt_mean = [double]::NaN
            fg_conf_presence_target_floor = [double]::NaN
            fg_conf_presence_active_ratio = [double]::NaN
            fg_conf_presence_loss = [double]::NaN
            loss_fg_conf_presence = [double]::NaN
            loss_contrib_fg_conf_presence = [double]::NaN
            mean_loss_fg_conf_presence = [double]::NaN
            tf32 = [bool]$Tf32
            amp = [bool]$Amp
            strict_deterministic = [bool]$StrictDeterministic
            runner_tf32 = [bool]$Tf32
            runner_amp = [bool]$Amp
            runner_strict_deterministic = [bool]$StrictDeterministic
            precompute_tf32 = [bool]$Tf32
            precompute_amp = [bool]$Amp
            precompute_strict_deterministic = [bool]$StrictDeterministic
            teacher_tf32 = [bool]$Tf32
            teacher_amp = [bool]$Amp
            teacher_deterministic = [bool]$StrictDeterministic
            lambda_fg_structure_depth_edge = [double]::NaN
            fg_structure_bbox_margin_px = [double]::NaN
            fg_structure_bbox_min_side_px = [double]::NaN
            fg_structure_region_mode = ""
            fg_structure_region_erode_px = [double]::NaN
            fg_structure_depth_edge_warmup_steps = [double]::NaN
            fg_structure_boundary_probe_px = [double]::NaN
            fg_structure_edge_support_mode = "off"
            fg_structure_edge_support_quantile = 0.0
            fg_structure_edge_support_min_px = 32
            fg_structure_edge_weight_mode = "uniform"
            fg_structure_boundary_falloff_px = 0
            fg_structure_component_bias_mode = "off"
            fg_structure_component_bias_threshold_ratio = 0.25
            fg_structure_component_bias_other_scale = 1.0
            fg_structure_front_depth_bias_mode = "off"
            fg_structure_front_depth_bias_tau = 0.75
            fg_structure_front_depth_bias_center_quantile = 0.55
            lambda_point_mv_outside_ring = [double]::NaN
            point_mv_outside_ring_px = [double]::NaN
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
            main_support_component_count = 0.0
            main_support_largest_component_share = 0.0
            main_support_top2_component_share = 0.0
            main_support_centroid_distance_mean = 0.0
            main_support_component_active_views = 0.0
            main_support_component_bias_weight_share = 1.0
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
            loss_fg_structure_depth_edge = 0.0
            loss_contrib_fg_structure_depth_edge = 0.0
            mean_loss_fg_structure_depth_edge = 0.0
            loss_point_mv_outside_ring = 0.0
            loss_contrib_point_mv_outside_ring = 0.0
            mean_loss_point_mv_outside_ring = 0.0
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
        }
        foreach ($kv in $supportMetricDefaults.GetEnumerator()) {
            if (-not $rowRef.PSObject.Properties[$kv.Key]) {
                $rowRef | Add-Member -NotePropertyName $kv.Key -NotePropertyValue $kv.Value -Force
            }
        }
        if (-not $rowRef.PSObject.Properties["candidate_result_json"]) {
            $rowRef | Add-Member -NotePropertyName candidate_result_json -NotePropertyValue $candidateResultPath -Force
        } else {
            $rowRef.candidate_result_json = $candidateResultPath
        }
        Write-CandidateResultJsonLocal `
            -Path $candidateResultPath `
            -LatestPath $candidateResultLatest `
            -Row $rowRef `
            -RunTag $tag `
            -Timestamp $ts
    }

    $rows | Export-Csv $outCsv -NoTypeInformation -Encoding UTF8
    $rows | Export-Csv $outLatest -NoTypeInformation -Encoding UTF8

    $ghostNow = [double]::NaN
    if (-not [string]::IsNullOrWhiteSpace([string]$ghostMean)) {
        $ghostNow = [double]$ghostMean
    }

    $hasGhost = -not [double]::IsNaN($ghostNow)
    $ssimOk = (([double]$MinSSIMGuard -le 0.0) -or ((-not [double]::IsNaN($ssimNow)) -and ($ssimNow -ge [double]$MinSSIMGuard)))
    $wl1Ok = (([double]$MaxWl1Guard -le 0.0) -or ((-not [double]::IsNaN($wl1Now)) -and ($wl1Now -le [double]$MaxWl1Guard)))
    $isValidRound = ([int]$rc -eq 0) -and (-not [double]::IsNaN($psnrNow)) -and ($psnrNow -ge $MinPSNRGuard) -and $ssimOk -and $wl1Ok
    if ($isValidRound) {
        if ($hasGhost) {
            $prevBestGhost = $bestGhost
            $strictBetter = ($ghostNow -lt $bestGhost)
            if ($strictBetter) {
                $bestGhost = $ghostNow
                $bestPsnr = $psnrNow
                if (-not [double]::IsNaN($wl1Now)) { $bestWl1 = $wl1Now }
            }
            if ($ghostNow -le ($prevBestGhost - [Math]::Max(0.0, $MinGhostImprove))) {
                $noImproveRounds = 0
                Write-Host "[ghost-mv] improved ghost=$ghostNow psnr=$psnrNow"
            } else {
                $noImproveRounds += 1
                if ($strictBetter) {
                    Write-Host "[ghost-mv] micro-improve ghost=$ghostNow (threshold=$MinGhostImprove), no-improve round=$noImproveRounds"
                } else {
                    Write-Host "[ghost-mv] no-improve round=$noImproveRounds ghost=$ghostNow best=$bestGhost"
                }
            }
            if (([double]$CatastrophicGhostRiseStop -gt 0.0) -and
                (-not [double]::IsInfinity($prevBestGhost)) -and
                (-not [double]::IsNaN($prevBestGhost))) {
                $catRise = $ghostNow - $prevBestGhost
                if ($catRise -ge [double]$CatastrophicGhostRiseStop) {
                    $stopDueCatastrophicGhost = $true
                    Write-Host "[ghost-mv] catastrophic rise stop: ghost=$ghostNow ref_best=$prevBestGhost rise=$catRise thr=$CatastrophicGhostRiseStop"
                }
            }
        } else {
            $psnrImproved = $psnrNow -gt ($bestPsnr + 1e-6)
            $wl1Improved = (-not [double]::IsNaN($wl1Now)) -and ($wl1Now -lt ($bestWl1 - 1e-6))
            if ($psnrImproved -or $wl1Improved) {
                if ($psnrImproved) { $bestPsnr = $psnrNow }
                if ($wl1Improved) { $bestWl1 = $wl1Now }
                $noImproveRounds = 0
                Write-Host "[ghost-mv] ghost unavailable -> fallback improve psnr=$psnrNow wl1=$wl1Now"
            } else {
                $noImproveRounds += 1
                Write-Host "[ghost-mv] ghost unavailable -> fallback no-improve round=$noImproveRounds psnr=$psnrNow wl1=$wl1Now best_psnr=$bestPsnr best_wl1=$bestWl1"
            }
        }
    } else {
        $noImproveRounds += 1
        Write-Host "[ghost-mv] invalid round -> count as no-improve (rc=$rc, ghost=$ghostNow, psnr=$psnrNow, wl1=$wl1Now)"
    }

    if ((-not [bool]$EnableAnySplatAblationSixPack) -and $isValidRound -and $hasGhost) {
        if (-not [double]::IsNaN($prevMaskGhost)) {
            $maskGhostRise = $ghostNow - $prevMaskGhost
            if ($maskGhostRise -ge [Math]::Max(0.0, [double]$MaskWorsenGhostDelta)) {
                $maskWorsenConsecutive += 1
                Write-Host "[ghost-mv] mask worsen streak=$maskWorsenConsecutive rise=$maskGhostRise (thr=$MaskWorsenGhostDelta)"
            } else {
                $maskWorsenConsecutive = 0
            }
            if (($MaskWorsenStopConsecutive -gt 0) -and ($maskWorsenConsecutive -ge [Math]::Max(1, [int]$MaskWorsenStopConsecutive))) {
                $maskWorsenTriggered = $true
                Write-Host "[ghost-mv] early stop mask sweep: consecutive worsen >= $MaskWorsenStopConsecutive"
            }
        } else {
            $maskWorsenConsecutive = 0
        }
        $prevMaskGhost = $ghostNow
    }

    if ($maskWorsenTriggered) {
        break
    }
    if ($stopDueInfraNoOutput) {
        Write-Host "[ghost-mv] early stop: consecutive infra no-output failures reached $InfraNoOutputStopConsecutive"
        break
    }
    if ($stopDueCatastrophicGhost) {
        Write-Host "[ghost-mv] early stop: catastrophic ghost rise reached threshold=$CatastrophicGhostRiseStop"
        break
    }
    if ($noImproveRounds -ge [Math]::Max(1, [int]$effectiveNoImprovePatience)) {
        Write-Host "[ghost-mv] early stop: no-improve reached patience=$effectiveNoImprovePatience"
        break
    }
    }
    if ($stopDueInfraNoOutput) { break }
    if ($stopDueCatastrophicGhost) { break }
    if ($noImproveRounds -ge [Math]::Max(1, [int]$effectiveNoImprovePatience)) { break }
}
}

$rows | Export-Csv $outCsv -NoTypeInformation -Encoding UTF8
$rows | Export-Csv $outLatest -NoTypeInformation -Encoding UTF8

$lines = @("# Ghost MV-Depth Sweep (Latest)", "")
foreach ($r in $rows) {
    $lines += "- lane=$($r.lane_id), family=$($r.candidate_family), ablation=$($r.ablation_group_id), lambda_point_mv_depth=$($r.lambda_point_mv_depth), src_views=$($r.eval_num_src_views_declared), src_views_actual=$($r.eval_num_src_views_actual), rc=$($r.exit_code), PSNR=$($r.mean_PSNR), SSIM=$($r.mean_SSIM), wL1=$($r.mean_weighted_L1), ghost=$($r.ghost_score_mean), ghost_visual=$($r.ghost_visual_score), luma=$($r.pred_luma_mean), nonblack=$($r.pred_nonblack_ratio_thr008), fg_luma=$($r.fg_pred_luma_mean), fg_nonblack=$($r.fg_pred_nonblack_ratio), fg_contrast=$($r.fg_pred_contrast), fg_tgt_l1=$($r.fg_pred_tgt_l1), visual_blocked=$($r.visual_guard_blocked), quality_blocked=$($r.quality_guard_blocked), invalid=$($r.candidate_invalid_reason)"
}
$lines += ""
$lines += "- best_ghost=$bestGhost, best_psnr=$bestPsnr, patience=$effectiveNoImprovePatience"
$lines += "- catastrophic_ghost_rise_stop=$CatastrophicGhostRiseStop"
Set-Content -Path $outMd -Value ($lines -join "`n") -Encoding UTF8

if ((@($rows | Where-Object { [int]$_.exit_code -ne 0 }).Count) -gt 0) {
    exit 2
}
exit 0
