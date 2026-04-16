[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [ValidateSet("G0","F0","F1","F2","F3","P1","R0","R1","R2","H0","H1","H1a","H1b","H1c","H1d","H1e","H1s1","H1s2","H1s1_core","H1s2_core","H1sf1","H1sf2","H2","S0","S1","S2","S3","T1","T2","T4","B0","T3")]
    [string]$ProbeId,
    [string]$InheritContractPath = "",
    [string]$SeqNames = "CoreView_390",
    [string]$CamNames = "Camera_B1,Camera_B2,Camera_B3,Camera_B4,Camera_B5,Camera_B6,Camera_B7,Camera_B8,Camera_B9,Camera_B10,Camera_B11,Camera_B12,Camera_B13,Camera_B14,Camera_B15,Camera_B16,Camera_B17,Camera_B18,Camera_B19,Camera_B20,Camera_B21,Camera_B22,Camera_B23",
    [string]$ResumeCkpt = "/mnt/out/vggt/finetune/lr_1e-6_20260227_101300/ckpt/model_ft_zju.pt",
    [string]$ReuseShortFtCkpt = "",
    [string]$PseudoGeomSubdir = "vggt_geom_ft_lr_1e-6_20260227_101300",
    [int]$Seed = 0,
    [int]$MosaicSeed = -1,
    [switch]$DryRun,
    [string]$EvalNumSrcViews = "8",
    [string]$LambdaPointMvDepth = "0.001",
    [string]$BaselineLambdaPointMvMask = "0.0005",
    [ValidateRange(0, 64)]
    [int]$PrecomputeMvSupportFgPreservePx = 5,
    [double]$FgSupervisionBoost = 1.0,
    [double]$FgSupervisionBgFloor = 0.0,
    [ValidateSet("all","interior_only")]
    [string]$FgSupervisionRegionMode = "all",
    [ValidateRange(0, 64)]
    [int]$FgSupervisionRegionErodePx = 0,
    [double]$LambdaFgConfPresence = 0.0,
    [double]$FgConfPresenceTargetRatio = 0.9,
    [double]$LambdaFgStructureDepthEdge = 0.0,
    [ValidateRange(0, 512)]
    [int]$FgStructureBboxMarginPx = 12,
    [ValidateRange(1, 512)]
    [int]$FgStructureBboxMinSidePx = 24,
    [ValidateSet("bbox","bbox_fg_interior")]
    [string]$FgStructureRegionMode = "bbox",
    [ValidateRange(0, 64)]
    [int]$FgStructureRegionErodePx = 0,
    [ValidateRange(0, 50000)]
    [int]$FgStructureDepthEdgeWarmupSteps = 0,
    [ValidateRange(0, 64)]
    [int]$FgStructureBoundaryProbePx = 2,
    [ValidateSet("off","target_edge_quantile")]
    [string]$FgStructureEdgeSupportMode = "off",
    [double]$FgStructureEdgeSupportQuantile = 0.0,
    [ValidateRange(1, 4096)]
    [int]$FgStructureEdgeSupportMinPx = 32,
    [ValidateSet("uniform","target_edge_sqrt")]
    [string]$FgStructureEdgeWeightMode = "uniform",
    [ValidateRange(0, 64)]
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
    [ValidateRange(0, 64)]
    [int]$PointMvOutsideRingPx = 3,
    [bool]$Tf32 = $true,
    [bool]$Amp = $true,
    [bool]$StrictDeterministic = $false,
    [string]$SnapshotOutRoot = "logs/modal_phase5/snapshots",
    [int]$WatchPollSec = 10
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

function Resolve-OptionalRepoPath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoDir $Path
}

function Read-JsonUtf8Maybe([string]$Path) {
    $resolved = Resolve-OptionalRepoPath $Path
    if ([string]::IsNullOrWhiteSpace($resolved) -or -not (Test-Path $resolved)) {
        return $null
    }
    $raw = Get-Content -Raw -Path $resolved -Encoding UTF8
    if (-not [string]::IsNullOrEmpty($raw) -and [int][char]$raw[0] -eq 0xFEFF) {
        $raw = $raw.Substring(1)
    }
    return ($raw | ConvertFrom-Json)
}

function Get-ContractValue([object]$Contract, [string]$Name, $Default = $null) {
    if ($null -eq $Contract) { return $Default }
    $prop = $Contract.PSObject.Properties[$Name]
    if ($null -eq $prop) { return $Default }
    return $prop.Value
}

function Set-CfgValueFromContract([System.Collections.IDictionary]$Cfg, [object]$Contract, [string]$CfgKey, [string]$ContractKey, [string]$Type = "string") {
    $raw = Get-ContractValue -Contract $Contract -Name $ContractKey
    if ($null -eq $raw) { return }
    $text = ([string]$raw).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) { return }
    switch ($Type) {
        "int" { $Cfg[$CfgKey] = [int]$raw }
        "double" { $Cfg[$CfgKey] = [double]$raw }
        "bool" {
            $norm = $text.ToLowerInvariant()
            if ($norm -in @("1","true","yes","y","on")) {
                $Cfg[$CfgKey] = $true
            } elseif ($norm -in @("0","false","no","n","off")) {
                $Cfg[$CfgKey] = $false
            }
        }
        default { $Cfg[$CfgKey] = [string]$raw }
    }
}

$inheritContract = Read-JsonUtf8Maybe -Path $InheritContractPath
if ($null -ne $inheritContract) {
    $inheritResolved = Resolve-OptionalRepoPath $InheritContractPath
    Write-Host "[probe] inherit contract=$inheritResolved"
    $contractCamNames = [string](Get-ContractValue -Contract $inheritContract -Name "cam_names")
    if (-not [string]::IsNullOrWhiteSpace($contractCamNames)) {
        $CamNames = $contractCamNames
    }
}

if ($MosaicSeed -lt 0) { $MosaicSeed = $Seed }
$oldFtSeed = $env:VGGT_FT_SEED
$oldMosaicSeed = $env:VGGT_MOSAIC_SEED
$oldPrecomputeGpuSpec = $env:VGGT_GPU_SPEC_PRECOMPUTE
$oldInferGpuSpec = $env:VGGT_GPU_SPEC_INFER
$env:VGGT_FT_SEED = [string]$Seed
$env:VGGT_MOSAIC_SEED = [string]$MosaicSeed
$env:VGGT_GPU_SPEC_PRECOMPUTE = "A100-80GB"
$env:VGGT_GPU_SPEC_INFER = "A100-80GB"

$syncStatusScriptPath = Join-Path $RepoDir "scripts/sync_single_probe_latest.ps1"
$ghostSweepPath = Join-Path $RepoDir "scripts/run_vggt_ghost_mvdepth_sweep.ps1"
$snapshotScriptPath = Join-Path $RepoDir "scripts/snapshot_human_transparency_probe.ps1"
$refreshSummaryScriptPath = Join-Path $RepoDir "scripts/refresh_support_probe_summary.ps1"
if (-not (Test-Path $syncStatusScriptPath)) { throw "sync status script not found: $syncStatusScriptPath" }
if (-not (Test-Path $ghostSweepPath)) { throw "ghost sweep script not found: $ghostSweepPath" }
if (-not (Test-Path $snapshotScriptPath)) { throw "snapshot script not found: $snapshotScriptPath" }
if (-not (Test-Path $refreshSummaryScriptPath)) { throw "refresh summary script not found: $refreshSummaryScriptPath" }

$cfg = [ordered]@{
    PointmapSource = "depth_unproject"
    PointTargetMode = "depth_unproject"
    PrecomputeMvSupportOn = "off"
    PointSupportMode = "off"
    PointMvDepthSupportMode = "off"
    PointMvMaskSupportMode = "inverse"
    PointTargetBlendByMvSupport = "off"
    PointTargetBlendMvRegionMode = "all"
    PrecomputeMvSupportRegionMode = "all"
    PrecomputeMvSupportFgMaskSource = "mask"
    PrecomputeMvSupportFgErodePx = 5
    PrecomputeMvSupportFgPreservePx = [int]$PrecomputeMvSupportFgPreservePx
    FgSupervisionBoost = [double]$FgSupervisionBoost
    FgSupervisionBgFloor = [double]$FgSupervisionBgFloor
    FgSupervisionRegionMode = [string]$FgSupervisionRegionMode
    FgSupervisionRegionErodePx = [int]$FgSupervisionRegionErodePx
    LambdaFgConfPresence = [double]$LambdaFgConfPresence
    FgConfPresenceTargetRatio = [double]$FgConfPresenceTargetRatio
    LambdaFgStructureDepthEdge = [double]$LambdaFgStructureDepthEdge
    FgStructureBboxMarginPx = [int]$FgStructureBboxMarginPx
    FgStructureBboxMinSidePx = [int]$FgStructureBboxMinSidePx
    FgStructureRegionMode = [string]$FgStructureRegionMode
    FgStructureRegionErodePx = [int]$FgStructureRegionErodePx
    FgStructureDepthEdgeWarmupSteps = [int]$FgStructureDepthEdgeWarmupSteps
    FgStructureBoundaryProbePx = [int]$FgStructureBoundaryProbePx
    FgStructureEdgeSupportMode = [string]$FgStructureEdgeSupportMode
    FgStructureEdgeSupportQuantile = [double]$FgStructureEdgeSupportQuantile
    FgStructureEdgeSupportMinPx = [int]$FgStructureEdgeSupportMinPx
    FgStructureEdgeWeightMode = [string]$FgStructureEdgeWeightMode
    FgStructureBoundaryFalloffPx = [int]$FgStructureBoundaryFalloffPx
    FgStructureComponentBiasMode = [string]$FgStructureComponentBiasMode
    FgStructureComponentBiasThresholdRatio = [double]$FgStructureComponentBiasThresholdRatio
    FgStructureComponentBiasOtherScale = [double]$FgStructureComponentBiasOtherScale
    LambdaPointMvOutsideRing = [double]$LambdaPointMvOutsideRing
    PointMvOutsideRingPx = [int]$PointMvOutsideRingPx
    Tf32 = [bool]$Tf32
    Amp = [bool]$Amp
    StrictDeterministic = [bool]$StrictDeterministic
    PointMvDepthRegionMode = "all"
    UseFgMask = "on"
    FgMaskSource = "mask"
    LambdaPointMvMask = "0"
}

if ($null -ne $inheritContract) {
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PointmapSource" -ContractKey "pointmap_source"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PointTargetMode" -ContractKey "point_target_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PrecomputeMvSupportOn" -ContractKey "precompute_mv_support_on"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PointSupportMode" -ContractKey "point_support_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PointMvDepthSupportMode" -ContractKey "point_mv_depth_support_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PointMvMaskSupportMode" -ContractKey "point_mv_mask_support_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PointTargetBlendByMvSupport" -ContractKey "point_target_blend_by_mv_support"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PointTargetBlendMvRegionMode" -ContractKey "point_target_blend_mv_region_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PrecomputeMvSupportRegionMode" -ContractKey "precompute_mv_support_region_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PrecomputeMvSupportFgMaskSource" -ContractKey "precompute_mv_support_fg_mask_source"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PrecomputeMvSupportFgErodePx" -ContractKey "precompute_mv_support_fg_erode_px" -Type "int"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PrecomputeMvSupportFgPreservePx" -ContractKey "precompute_mv_support_fg_preserve_px" -Type "int"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PointMvDepthRegionMode" -ContractKey "point_mv_depth_region_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "UseFgMask" -ContractKey "use_fg_mask"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgMaskSource" -ContractKey "fg_mask_source"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "LambdaPointMvMask" -ContractKey "lambda_point_mv_mask"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgSupervisionBoost" -ContractKey "fg_supervision_boost" -Type "double"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgSupervisionBgFloor" -ContractKey "fg_supervision_bg_floor" -Type "double"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgSupervisionRegionMode" -ContractKey "fg_supervision_region_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgSupervisionRegionErodePx" -ContractKey "fg_supervision_region_erode_px" -Type "int"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "LambdaFgConfPresence" -ContractKey "lambda_fg_conf_presence" -Type "double"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgConfPresenceTargetRatio" -ContractKey "fg_conf_presence_target_ratio" -Type "double"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "LambdaFgStructureDepthEdge" -ContractKey "lambda_fg_structure_depth_edge" -Type "double"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureBboxMarginPx" -ContractKey "fg_structure_bbox_margin_px" -Type "int"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureBboxMinSidePx" -ContractKey "fg_structure_bbox_min_side_px" -Type "int"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureRegionMode" -ContractKey "fg_structure_region_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureRegionErodePx" -ContractKey "fg_structure_region_erode_px" -Type "int"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureDepthEdgeWarmupSteps" -ContractKey "fg_structure_depth_edge_warmup_steps" -Type "int"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureBoundaryProbePx" -ContractKey "fg_structure_boundary_probe_px" -Type "int"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureEdgeSupportMode" -ContractKey "fg_structure_edge_support_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureEdgeSupportQuantile" -ContractKey "fg_structure_edge_support_quantile" -Type "double"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureEdgeSupportMinPx" -ContractKey "fg_structure_edge_support_min_px" -Type "int"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureEdgeWeightMode" -ContractKey "fg_structure_edge_weight_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureBoundaryFalloffPx" -ContractKey "fg_structure_boundary_falloff_px" -Type "int"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureComponentBiasMode" -ContractKey "fg_structure_component_bias_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureComponentBiasThresholdRatio" -ContractKey "fg_structure_component_bias_threshold_ratio" -Type "double"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureComponentBiasOtherScale" -ContractKey "fg_structure_component_bias_other_scale" -Type "double"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureFrontDepthBiasMode" -ContractKey "fg_structure_front_depth_bias_mode"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureFrontDepthBiasTau" -ContractKey "fg_structure_front_depth_bias_tau" -Type "double"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "FgStructureFrontDepthBiasCenterQuantile" -ContractKey "fg_structure_front_depth_bias_center_quantile" -Type "double"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "LambdaPointMvOutsideRing" -ContractKey "lambda_point_mv_outside_ring" -Type "double"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "PointMvOutsideRingPx" -ContractKey "point_mv_outside_ring_px" -Type "int"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "Tf32" -ContractKey "tf32" -Type "bool"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "Amp" -ContractKey "amp" -Type "bool"
    Set-CfgValueFromContract -Cfg $cfg -Contract $inheritContract -CfgKey "StrictDeterministic" -ContractKey "strict_deterministic" -Type "bool"
}

switch ($ProbeId) {
    "G0" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
    }
    "F0" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
    }
    "F1" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.5
        $cfg.FgSupervisionBgFloor = 0.05
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
    }
    "F2" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 2.0
        $cfg.FgSupervisionBgFloor = 0.05
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
    }
    "F3" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.5
        $cfg.FgSupervisionBgFloor = 0.05
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.02
        $cfg.FgConfPresenceTargetRatio = 0.9
    }
    "P1" {
        $p1Lambda = $(if ($PSBoundParameters.ContainsKey("LambdaFgConfPresence")) { [double]$LambdaFgConfPresence } else { 0.005 })
        $p1TargetRatio = $(if ($PSBoundParameters.ContainsKey("FgConfPresenceTargetRatio")) { [double]$FgConfPresenceTargetRatio } else { 0.8 })
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = $p1Lambda
        $cfg.FgConfPresenceTargetRatio = $p1TargetRatio
    }
    "R0" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
    }
    "R1" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.5
        $cfg.FgSupervisionBgFloor = 0.05
        $cfg.FgSupervisionRegionMode = "interior_only"
        $cfg.FgSupervisionRegionErodePx = 3
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
    }
    "R2" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.3
        $cfg.FgSupervisionBgFloor = 0.05
        $cfg.FgSupervisionRegionMode = "interior_only"
        $cfg.FgSupervisionRegionErodePx = 5
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
    }
    "H0" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.0
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox"
        $cfg.FgStructureRegionErodePx = 0
        $cfg.FgStructureDepthEdgeWarmupSteps = 0
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "off"
        $cfg.FgStructureEdgeSupportQuantile = 0.0
        $cfg.FgStructureEdgeSupportMinPx = 32
        $cfg.FgStructureEdgeWeightMode = "uniform"
        $cfg.FgStructureBoundaryFalloffPx = 0
        $cfg.LambdaPointMvOutsideRing = 0.0
        $cfg.PointMvOutsideRingPx = 3
    }
    "H1" {
        throw "ProbeId=H1 is deprecated; use H1s1_core, H1s2_core, H1sf1, or H1sf2."
    }
    "H1a" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.003
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox_fg_interior"
        $cfg.FgStructureRegionErodePx = 3
        $cfg.FgStructureDepthEdgeWarmupSteps = 80
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "off"
        $cfg.FgStructureEdgeSupportQuantile = 0.0
        $cfg.FgStructureEdgeSupportMinPx = 32
        $cfg.FgStructureEdgeWeightMode = "uniform"
        $cfg.FgStructureBoundaryFalloffPx = 0
        $cfg.LambdaPointMvOutsideRing = 0.0
        $cfg.PointMvOutsideRingPx = 3
    }
    "H1b" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.005
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox_fg_interior"
        $cfg.FgStructureRegionErodePx = 3
        $cfg.FgStructureDepthEdgeWarmupSteps = 80
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "off"
        $cfg.FgStructureEdgeSupportQuantile = 0.0
        $cfg.FgStructureEdgeSupportMinPx = 32
        $cfg.FgStructureEdgeWeightMode = "uniform"
        $cfg.FgStructureBoundaryFalloffPx = 0
        $cfg.LambdaPointMvOutsideRing = 0.0
        $cfg.PointMvOutsideRingPx = 3
    }
    "H1c" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.005
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox_fg_interior"
        $cfg.FgStructureRegionErodePx = 5
        $cfg.FgStructureDepthEdgeWarmupSteps = 80
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "off"
        $cfg.FgStructureEdgeSupportQuantile = 0.0
        $cfg.FgStructureEdgeSupportMinPx = 32
        $cfg.FgStructureEdgeWeightMode = "uniform"
        $cfg.FgStructureBoundaryFalloffPx = 0
        $cfg.LambdaPointMvOutsideRing = 0.0
        $cfg.PointMvOutsideRingPx = 3
    }
    "H1d" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.003
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox_fg_interior"
        $cfg.FgStructureRegionErodePx = 3
        $cfg.FgStructureDepthEdgeWarmupSteps = 80
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "target_edge_quantile"
        $cfg.FgStructureEdgeSupportQuantile = 0.75
        $cfg.FgStructureEdgeSupportMinPx = 32
        $cfg.FgStructureEdgeWeightMode = "uniform"
        $cfg.FgStructureBoundaryFalloffPx = 0
        $cfg.LambdaPointMvOutsideRing = 0.0
        $cfg.PointMvOutsideRingPx = 3
    }
    "H1e" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.003
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox_fg_interior"
        $cfg.FgStructureRegionErodePx = 3
        $cfg.FgStructureDepthEdgeWarmupSteps = 80
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "target_edge_quantile"
        $cfg.FgStructureEdgeSupportQuantile = 0.85
        $cfg.FgStructureEdgeSupportMinPx = 24
        $cfg.FgStructureEdgeWeightMode = "uniform"
        $cfg.FgStructureBoundaryFalloffPx = 0
        $cfg.LambdaPointMvOutsideRing = 0.0
        $cfg.PointMvOutsideRingPx = 3
    }
    "H1s1" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.003
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox_fg_interior"
        $cfg.FgStructureRegionErodePx = 3
        $cfg.FgStructureDepthEdgeWarmupSteps = 80
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "off"
        $cfg.FgStructureEdgeSupportQuantile = 0.0
        $cfg.FgStructureEdgeSupportMinPx = 32
        $cfg.FgStructureEdgeWeightMode = "target_edge_sqrt"
        $cfg.FgStructureBoundaryFalloffPx = 2
        $cfg.FgStructureComponentBiasMode = "off"
        $cfg.FgStructureComponentBiasThresholdRatio = 0.25
        $cfg.FgStructureComponentBiasOtherScale = 1.0
        $cfg.LambdaPointMvOutsideRing = 0.0
        $cfg.PointMvOutsideRingPx = 3
    }
    "H1s2" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.003
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox_fg_interior"
        $cfg.FgStructureRegionErodePx = 3
        $cfg.FgStructureDepthEdgeWarmupSteps = 80
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "off"
        $cfg.FgStructureEdgeSupportQuantile = 0.0
        $cfg.FgStructureEdgeSupportMinPx = 32
        $cfg.FgStructureEdgeWeightMode = "target_edge_sqrt"
        $cfg.FgStructureBoundaryFalloffPx = 3
        $cfg.FgStructureComponentBiasMode = "off"
        $cfg.FgStructureComponentBiasThresholdRatio = 0.25
        $cfg.FgStructureComponentBiasOtherScale = 1.0
        $cfg.LambdaPointMvOutsideRing = 0.0
        $cfg.PointMvOutsideRingPx = 3
    }
    "H1s1_core" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.003
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox_fg_interior"
        $cfg.FgStructureRegionErodePx = 3
        $cfg.FgStructureDepthEdgeWarmupSteps = 80
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "off"
        $cfg.FgStructureEdgeSupportQuantile = 0.0
        $cfg.FgStructureEdgeSupportMinPx = 32
        $cfg.FgStructureEdgeWeightMode = "target_edge_sqrt"
        $cfg.FgStructureBoundaryFalloffPx = 2
        $cfg.FgStructureComponentBiasMode = "largest_soft"
        $cfg.FgStructureComponentBiasThresholdRatio = 0.25
        $cfg.FgStructureComponentBiasOtherScale = 0.35
        $cfg.FgStructureFrontDepthBiasMode = "off"
        $cfg.FgStructureFrontDepthBiasTau = 0.75
        $cfg.FgStructureFrontDepthBiasCenterQuantile = 0.55
        $cfg.LambdaPointMvOutsideRing = 0.0
        $cfg.PointMvOutsideRingPx = 3
    }
    "H1s2_core" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.003
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox_fg_interior"
        $cfg.FgStructureRegionErodePx = 3
        $cfg.FgStructureDepthEdgeWarmupSteps = 80
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "off"
        $cfg.FgStructureEdgeSupportQuantile = 0.0
        $cfg.FgStructureEdgeSupportMinPx = 32
        $cfg.FgStructureEdgeWeightMode = "target_edge_sqrt"
        $cfg.FgStructureBoundaryFalloffPx = 3
        $cfg.FgStructureComponentBiasMode = "largest_soft"
        $cfg.FgStructureComponentBiasThresholdRatio = 0.25
        $cfg.FgStructureComponentBiasOtherScale = 0.35
        $cfg.FgStructureFrontDepthBiasMode = "off"
        $cfg.FgStructureFrontDepthBiasTau = 0.75
        $cfg.FgStructureFrontDepthBiasCenterQuantile = 0.55
        $cfg.LambdaPointMvOutsideRing = 0.0
        $cfg.PointMvOutsideRingPx = 3
    }
    "H1sf1" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.003
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox_fg_interior"
        $cfg.FgStructureRegionErodePx = 3
        $cfg.FgStructureDepthEdgeWarmupSteps = 80
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "off"
        $cfg.FgStructureEdgeSupportQuantile = 0.0
        $cfg.FgStructureEdgeSupportMinPx = 32
        $cfg.FgStructureEdgeWeightMode = "target_edge_sqrt"
        $cfg.FgStructureBoundaryFalloffPx = 2
        $cfg.FgStructureComponentBiasMode = "largest_soft"
        $cfg.FgStructureComponentBiasThresholdRatio = 0.25
        $cfg.FgStructureComponentBiasOtherScale = 0.35
        $cfg.FgStructureFrontDepthBiasMode = "front_soft"
        $cfg.FgStructureFrontDepthBiasTau = 0.75
        $cfg.FgStructureFrontDepthBiasCenterQuantile = 0.55
        $cfg.LambdaPointMvOutsideRing = 0.0
        $cfg.PointMvOutsideRingPx = 3
    }
    "H1sf2" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.003
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox_fg_interior"
        $cfg.FgStructureRegionErodePx = 3
        $cfg.FgStructureDepthEdgeWarmupSteps = 80
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "off"
        $cfg.FgStructureEdgeSupportQuantile = 0.0
        $cfg.FgStructureEdgeSupportMinPx = 32
        $cfg.FgStructureEdgeWeightMode = "target_edge_sqrt"
        $cfg.FgStructureBoundaryFalloffPx = 3
        $cfg.FgStructureComponentBiasMode = "largest_soft"
        $cfg.FgStructureComponentBiasThresholdRatio = 0.25
        $cfg.FgStructureComponentBiasOtherScale = 0.35
        $cfg.FgStructureFrontDepthBiasMode = "front_soft"
        $cfg.FgStructureFrontDepthBiasTau = 0.75
        $cfg.FgStructureFrontDepthBiasCenterQuantile = 0.55
        $cfg.LambdaPointMvOutsideRing = 0.0
        $cfg.PointMvOutsideRingPx = 3
    }
    "H2" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PrecomputeMvSupportRegionMode = "bg_only"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.FgSupervisionBoost = 1.0
        $cfg.FgSupervisionBgFloor = 0.0
        $cfg.FgSupervisionRegionMode = "all"
        $cfg.FgSupervisionRegionErodePx = 0
        $cfg.LambdaFgConfPresence = 0.0
        $cfg.FgConfPresenceTargetRatio = 0.9
        $cfg.LambdaFgStructureDepthEdge = 0.01
        $cfg.FgStructureBboxMarginPx = 12
        $cfg.FgStructureBboxMinSidePx = 24
        $cfg.FgStructureRegionMode = "bbox"
        $cfg.FgStructureRegionErodePx = 0
        $cfg.FgStructureDepthEdgeWarmupSteps = 0
        $cfg.FgStructureBoundaryProbePx = 2
        $cfg.FgStructureEdgeSupportMode = "off"
        $cfg.FgStructureEdgeSupportQuantile = 0.0
        $cfg.FgStructureEdgeSupportMinPx = 32
        $cfg.LambdaPointMvOutsideRing = 0.002
        $cfg.PointMvOutsideRingPx = 3
    }
    "S0" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
    }
    "S1" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PointSupportMode = "direct"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
    }
    "S2" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "direct"
        $cfg.PointMvMaskSupportMode = "off"
    }
    "S3" {
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PointSupportMode = "off"
        $cfg.PointMvDepthSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "off"
        $cfg.PointMvMaskSupportMode = "inverse"
    }
    "T1" {
        $cfg.PrecomputeMvSupportOn = "on"
    }
    "T2" {
        $cfg.PointTargetMode = "blend"
        $cfg.PrecomputeMvSupportOn = "on"
        $cfg.PointTargetBlendByMvSupport = "on"
    }
    "T4" {
        $cfg.PointMvDepthRegionMode = "bg_only"
        $cfg.UseFgMask = "off"
        $cfg.FgMaskSource = "mask"
    }
    "B0" {
        $cfg.LambdaPointMvMask = [string]$BaselineLambdaPointMvMask
    }
    "T3" {
        $cfg.LambdaPointMvMask = "0"
    }
}

$contractLatestPath = Join-Path $RepoDir "logs/modal_phase5/probe_contract_latest.json"
$contractTag = Get-Date -Format "yyyyMMdd_HHmmss"
$contractStampedPath = Join-Path $RepoDir ("logs/modal_phase5/probe_contract_${ProbeId}_$contractTag.json")
$contract = [ordered]@{
    probe_id = $ProbeId
    repo_dir = $RepoDir
    seq_names = $SeqNames
    cam_names = $CamNames
    resume_ckpt = $ResumeCkpt
    reuse_short_ft_ckpt = $ReuseShortFtCkpt
    pseudo_geom_subdir = $PseudoGeomSubdir
    seed = [int]$Seed
    mosaic_seed = [int]$MosaicSeed
    eval_num_src_views = [string]$EvalNumSrcViews
    lambda_point_mv_depth = [string]$LambdaPointMvDepth
    baseline_lambda_point_mv_mask = [string]$BaselineLambdaPointMvMask
    generated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    precompute_gpu_spec = "A100-80GB"
    infer_gpu_spec = "A100-80GB"
    pointmap_source = [string]$cfg.PointmapSource
    point_target_mode = [string]$cfg.PointTargetMode
    precompute_mv_support_on = [string]$cfg.PrecomputeMvSupportOn
    precompute_mv_support_region_mode = [string]$cfg.PrecomputeMvSupportRegionMode
    precompute_mv_support_fg_mask_source = [string]$cfg.PrecomputeMvSupportFgMaskSource
    precompute_mv_support_fg_erode_px = [string]$cfg.PrecomputeMvSupportFgErodePx
    precompute_mv_support_fg_preserve_px = [string]$cfg.PrecomputeMvSupportFgPreservePx
    fg_supervision_boost = [string]$cfg.FgSupervisionBoost
    fg_supervision_bg_floor = [string]$cfg.FgSupervisionBgFloor
    fg_supervision_region_mode = [string]$cfg.FgSupervisionRegionMode
    fg_supervision_region_erode_px = [string]$cfg.FgSupervisionRegionErodePx
    lambda_fg_conf_presence = [string]$cfg.LambdaFgConfPresence
    fg_conf_presence_target_ratio = [string]$cfg.FgConfPresenceTargetRatio
    lambda_fg_structure_depth_edge = [string]$cfg.LambdaFgStructureDepthEdge
    fg_structure_bbox_margin_px = [string]$cfg.FgStructureBboxMarginPx
    fg_structure_bbox_min_side_px = [string]$cfg.FgStructureBboxMinSidePx
    fg_structure_region_mode = [string]$cfg.FgStructureRegionMode
    fg_structure_region_erode_px = [string]$cfg.FgStructureRegionErodePx
    fg_structure_depth_edge_warmup_steps = [string]$cfg.FgStructureDepthEdgeWarmupSteps
    fg_structure_boundary_probe_px = [string]$cfg.FgStructureBoundaryProbePx
    fg_structure_edge_support_mode = [string]$cfg.FgStructureEdgeSupportMode
    fg_structure_edge_support_quantile = [string]$cfg.FgStructureEdgeSupportQuantile
    fg_structure_edge_support_min_px = [string]$cfg.FgStructureEdgeSupportMinPx
    fg_structure_edge_weight_mode = [string]$cfg.FgStructureEdgeWeightMode
    fg_structure_boundary_falloff_px = [string]$cfg.FgStructureBoundaryFalloffPx
    fg_structure_component_bias_mode = [string]$cfg.FgStructureComponentBiasMode
    fg_structure_component_bias_threshold_ratio = [string]$cfg.FgStructureComponentBiasThresholdRatio
    fg_structure_component_bias_other_scale = [string]$cfg.FgStructureComponentBiasOtherScale
    fg_structure_front_depth_bias_mode = [string]$cfg.FgStructureFrontDepthBiasMode
    fg_structure_front_depth_bias_tau = [string]$cfg.FgStructureFrontDepthBiasTau
    fg_structure_front_depth_bias_center_quantile = [string]$cfg.FgStructureFrontDepthBiasCenterQuantile
    lambda_point_mv_outside_ring = [string]$cfg.LambdaPointMvOutsideRing
    point_mv_outside_ring_px = [string]$cfg.PointMvOutsideRingPx
    tf32 = [string]([int][bool]$cfg.Tf32)
    amp = [string]([int][bool]$cfg.Amp)
    strict_deterministic = [string]([int][bool]$cfg.StrictDeterministic)
    point_support_mode = [string]$cfg.PointSupportMode
    point_mv_depth_support_mode = [string]$cfg.PointMvDepthSupportMode
    point_mv_mask_support_mode = [string]$cfg.PointMvMaskSupportMode
    point_target_blend_by_mv_support = [string]$cfg.PointTargetBlendByMvSupport
    point_target_blend_mv_region_mode = [string]$cfg.PointTargetBlendMvRegionMode
    point_mv_depth_region_mode = [string]$cfg.PointMvDepthRegionMode
    use_fg_mask = [string]$cfg.UseFgMask
    fg_mask_source = [string]$cfg.FgMaskSource
    lambda_point_mv_mask = [string]$cfg.LambdaPointMvMask
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $contractLatestPath) | Out-Null
$contract | ConvertTo-Json -Depth 6 | Set-Content -Path $contractLatestPath -Encoding UTF8
$contract | ConvertTo-Json -Depth 6 | Set-Content -Path $contractStampedPath -Encoding UTF8

$runSucceeded = $false
$sweepExitCode = $null
try {
    Write-Host "[probe] start probe=$ProbeId seq=$SeqNames seed=$Seed mosaic_seed=$MosaicSeed"
    Write-Host "[probe] cfg pointmap_source=$($cfg.PointmapSource) point_target_mode=$($cfg.PointTargetMode) precompute_mv_support_on=$($cfg.PrecomputeMvSupportOn) precompute_mv_support_region_mode=$($cfg.PrecomputeMvSupportRegionMode) precompute_mv_support_fg_mask_source=$($cfg.PrecomputeMvSupportFgMaskSource) precompute_mv_support_fg_erode_px=$($cfg.PrecomputeMvSupportFgErodePx) precompute_mv_support_fg_preserve_px=$($cfg.PrecomputeMvSupportFgPreservePx) fg_supervision_boost=$($cfg.FgSupervisionBoost) fg_supervision_bg_floor=$($cfg.FgSupervisionBgFloor) fg_supervision_region_mode=$($cfg.FgSupervisionRegionMode) fg_supervision_region_erode_px=$($cfg.FgSupervisionRegionErodePx) lambda_fg_conf_presence=$($cfg.LambdaFgConfPresence) fg_conf_presence_target_ratio=$($cfg.FgConfPresenceTargetRatio) lambda_fg_structure_depth_edge=$($cfg.LambdaFgStructureDepthEdge) fg_structure_bbox_margin_px=$($cfg.FgStructureBboxMarginPx) fg_structure_bbox_min_side_px=$($cfg.FgStructureBboxMinSidePx) fg_structure_region_mode=$($cfg.FgStructureRegionMode) fg_structure_region_erode_px=$($cfg.FgStructureRegionErodePx) fg_structure_depth_edge_warmup_steps=$($cfg.FgStructureDepthEdgeWarmupSteps) fg_structure_boundary_probe_px=$($cfg.FgStructureBoundaryProbePx) fg_structure_edge_support_mode=$($cfg.FgStructureEdgeSupportMode) fg_structure_edge_support_quantile=$($cfg.FgStructureEdgeSupportQuantile) fg_structure_edge_support_min_px=$($cfg.FgStructureEdgeSupportMinPx) fg_structure_edge_weight_mode=$($cfg.FgStructureEdgeWeightMode) fg_structure_boundary_falloff_px=$($cfg.FgStructureBoundaryFalloffPx) fg_structure_component_bias_mode=$($cfg.FgStructureComponentBiasMode) fg_structure_component_bias_threshold_ratio=$($cfg.FgStructureComponentBiasThresholdRatio) fg_structure_component_bias_other_scale=$($cfg.FgStructureComponentBiasOtherScale) fg_structure_front_depth_bias_mode=$($cfg.FgStructureFrontDepthBiasMode) fg_structure_front_depth_bias_tau=$($cfg.FgStructureFrontDepthBiasTau) fg_structure_front_depth_bias_center_quantile=$($cfg.FgStructureFrontDepthBiasCenterQuantile) lambda_point_mv_outside_ring=$($cfg.LambdaPointMvOutsideRing) point_mv_outside_ring_px=$($cfg.PointMvOutsideRingPx) tf32=$($cfg.Tf32) amp=$($cfg.Amp) strict_deterministic=$($cfg.StrictDeterministic) point_support_mode=$($cfg.PointSupportMode) point_mv_depth_support_mode=$($cfg.PointMvDepthSupportMode) point_mv_mask_support_mode=$($cfg.PointMvMaskSupportMode) point_target_blend_by_mv_support=$($cfg.PointTargetBlendByMvSupport) point_target_blend_mv_region_mode=$($cfg.PointTargetBlendMvRegionMode) point_mv_depth_region_mode=$($cfg.PointMvDepthRegionMode) lambda_point_mv_mask=$($cfg.LambdaPointMvMask) use_fg_mask=$($cfg.UseFgMask) fg_mask_source=$($cfg.FgMaskSource)"
    if (-not [string]::IsNullOrWhiteSpace($ReuseShortFtCkpt)) {
        Write-Host "[probe] reuse_short_ft_ckpt=$ReuseShortFtCkpt"
    }
    if ($ProbeId -eq "T3") {
        Write-Host "[probe] note=T3 should be compared against B0, not against T0-smoke."
    }
    if ($DryRun) {
        Write-Host "[probe] dry_run_only=true"
        Write-Host "[probe] contract_latest=$contractLatestPath"
        Write-Host "[probe] contract_stamped=$contractStampedPath"
        $runSucceeded = $true
        return
    }
    & $ghostSweepPath `
        -CodeDir $RepoDir `
        -SeqNames $SeqNames `
        -CamNames $CamNames `
        -PseudoGeomSubdir $PseudoGeomSubdir `
        -ResumeCkpt $ResumeCkpt `
        -ReuseShortFtCkpt $ReuseShortFtCkpt `
        -EvalNumSrcViewsList $EvalNumSrcViews `
        -EvalInferArgsExtra "--num_src_views=$EvalNumSrcViews" `
        -Lr '1e-6' `
        -LambdaPointMvDepthList $LambdaPointMvDepth `
        -LambdaPointMvMask ([string]$cfg.LambdaPointMvMask) `
        -PointmapSource ([string]$cfg.PointmapSource) `
        -PointTargetMode ([string]$cfg.PointTargetMode) `
        -PrecomputeMvSupportOn ([string]$cfg.PrecomputeMvSupportOn) `
        -PrecomputeMvSupportRegionMode ([string]$cfg.PrecomputeMvSupportRegionMode) `
        -PrecomputeMvSupportFgMaskSource ([string]$cfg.PrecomputeMvSupportFgMaskSource) `
        -PrecomputeMvSupportFgErodePx ([int]$cfg.PrecomputeMvSupportFgErodePx) `
        -PrecomputeMvSupportFgPreservePx ([int]$cfg.PrecomputeMvSupportFgPreservePx) `
        -PointSupportMode ([string]$cfg.PointSupportMode) `
        -PointMvDepthSupportMode ([string]$cfg.PointMvDepthSupportMode) `
        -PointMvMaskSupportMode ([string]$cfg.PointMvMaskSupportMode) `
        -PointTargetBlendByMvSupport ([string]$cfg.PointTargetBlendByMvSupport) `
        -PointTargetBlendMvRegionMode ([string]$cfg.PointTargetBlendMvRegionMode) `
        -PointMvDepthRegionMode ([string]$cfg.PointMvDepthRegionMode) `
        -UseFgMask ([string]$cfg.UseFgMask) `
        -FgMaskSource ([string]$cfg.FgMaskSource) `
        -FgSupervisionBoost ([double]$cfg.FgSupervisionBoost) `
        -FgSupervisionBgFloor ([double]$cfg.FgSupervisionBgFloor) `
        -FgSupervisionRegionMode ([string]$cfg.FgSupervisionRegionMode) `
        -FgSupervisionRegionErodePx ([int]$cfg.FgSupervisionRegionErodePx) `
        -LambdaFgConfPresence ([double]$cfg.LambdaFgConfPresence) `
        -FgConfPresenceTargetRatio ([double]$cfg.FgConfPresenceTargetRatio) `
        -LambdaFgStructureDepthEdge ([double]$cfg.LambdaFgStructureDepthEdge) `
        -FgStructureBboxMarginPx ([int]$cfg.FgStructureBboxMarginPx) `
        -FgStructureBboxMinSidePx ([int]$cfg.FgStructureBboxMinSidePx) `
        -FgStructureRegionMode ([string]$cfg.FgStructureRegionMode) `
        -FgStructureRegionErodePx ([int]$cfg.FgStructureRegionErodePx) `
        -FgStructureDepthEdgeWarmupSteps ([int]$cfg.FgStructureDepthEdgeWarmupSteps) `
        -FgStructureBoundaryProbePx ([int]$cfg.FgStructureBoundaryProbePx) `
        -FgStructureEdgeSupportMode ([string]$cfg.FgStructureEdgeSupportMode) `
        -FgStructureEdgeSupportQuantile ([double]$cfg.FgStructureEdgeSupportQuantile) `
        -FgStructureEdgeSupportMinPx ([int]$cfg.FgStructureEdgeSupportMinPx) `
        -FgStructureEdgeWeightMode ([string]$cfg.FgStructureEdgeWeightMode) `
        -FgStructureBoundaryFalloffPx ([int]$cfg.FgStructureBoundaryFalloffPx) `
        -FgStructureComponentBiasMode ([string]$cfg.FgStructureComponentBiasMode) `
        -FgStructureComponentBiasThresholdRatio ([double]$cfg.FgStructureComponentBiasThresholdRatio) `
        -FgStructureComponentBiasOtherScale ([double]$cfg.FgStructureComponentBiasOtherScale) `
        -FgStructureFrontDepthBiasMode ([string]$cfg.FgStructureFrontDepthBiasMode) `
        -FgStructureFrontDepthBiasTau ([double]$cfg.FgStructureFrontDepthBiasTau) `
        -FgStructureFrontDepthBiasCenterQuantile ([double]$cfg.FgStructureFrontDepthBiasCenterQuantile) `
        -LambdaPointMvOutsideRing ([double]$cfg.LambdaPointMvOutsideRing) `
        -PointMvOutsideRingPx ([int]$cfg.PointMvOutsideRingPx) `
        -Tf32 ([bool]$cfg.Tf32) `
        -Amp ([bool]$cfg.Amp) `
        -StrictDeterministic ([bool]$cfg.StrictDeterministic) `
        -EnableAnySplatAblationSixPack:$false `
        -DynProxyEnable 'off' `
        -LaneId ('probe_' + $ProbeId.ToLowerInvariant()) `
        -CandidateFamily 'human_transparency_probe' `
        -GuardTier 'manual_probe' `
        -ModalRunQuiet:$false `
        -ModalRunNoOutputMaxRetries 0 `
        -EnableNoOutputProbeRetry:$false `
        -EnablePreemptiveProbeForPointHead:$false `
        -EnablePreemptiveProbeForStrongDepthUnproject:$false `
        -EnablePrecomputeNoOutputRetry:$false `
        -EnableDepthPrecomputeNoOutputRecovery:$false `
        -EnableResumeCkptFallbackOnShortCkptMissing:$false
    $sweepExitCode = $LASTEXITCODE
    if ($sweepExitCode -eq 0) {
        $runSucceeded = $true
    } else {
        throw "ghost sweep exited with code $sweepExitCode"
    }
}
finally {
    if (-not $DryRun) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $syncStatusScriptPath `
            -RepoDir $RepoDir `
            -ProbeId $ProbeId `
            -State $(if ($runSucceeded) { "done" } else { "error" }) `
            -ContractPath "logs/modal_phase5/probe_contract_latest.json" | Out-Null
        & powershell -NoProfile -ExecutionPolicy Bypass -File $snapshotScriptPath `
            -RepoDir $RepoDir `
            -ProbeId $ProbeId `
            -Label "manual_probe" `
            -ContractPath "logs/modal_phase5/probe_contract_latest.json" `
            -OutRoot $SnapshotOutRoot | Out-Null
        try {
            & powershell -NoProfile -ExecutionPolicy Bypass -File $refreshSummaryScriptPath `
                -RepoDir $RepoDir | Out-Null
        } catch {
            Write-Warning ("[probe] support summary refresh failed: " + $_.Exception.Message)
        }
    }
    if ($null -eq $oldFtSeed) { Remove-Item Env:VGGT_FT_SEED -ErrorAction SilentlyContinue } else { $env:VGGT_FT_SEED = $oldFtSeed }
    if ($null -eq $oldMosaicSeed) { Remove-Item Env:VGGT_MOSAIC_SEED -ErrorAction SilentlyContinue } else { $env:VGGT_MOSAIC_SEED = $oldMosaicSeed }
    if ($null -eq $oldPrecomputeGpuSpec) { Remove-Item Env:VGGT_GPU_SPEC_PRECOMPUTE -ErrorAction SilentlyContinue } else { $env:VGGT_GPU_SPEC_PRECOMPUTE = $oldPrecomputeGpuSpec }
    if ($null -eq $oldInferGpuSpec) { Remove-Item Env:VGGT_GPU_SPEC_INFER -ErrorAction SilentlyContinue } else { $env:VGGT_GPU_SPEC_INFER = $oldInferGpuSpec }
}
