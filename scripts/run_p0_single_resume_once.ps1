param(
    [string]$RepoDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$StartResumeCkpt = "/mnt/out/vggt/finetune/lr_1e-6_20260227_101300/ckpt/model_ft_zju.pt",
    [string]$StartPseudoGeomSubdir = "vggt_geom_ft_lr_1e-6_20260227_101300",
    [int]$Seed = 0,
    [int]$MosaicSeed = -1,
    [string]$SnapshotOutRoot = "logs/modal_phase5/snapshots",
    [int]$MaxCycles = 1,
    [int]$StopAfterHours = 2,
    [int]$WatchPollSec = 10
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

if ($MosaicSeed -lt 0) { $MosaicSeed = $Seed }
$oldFtSeed = $env:VGGT_FT_SEED
$oldMosaicSeed = $env:VGGT_MOSAIC_SEED
$oldPrecomputeGpuSpec = $env:VGGT_GPU_SPEC_PRECOMPUTE
$oldInferGpuSpec = $env:VGGT_GPU_SPEC_INFER
$env:VGGT_FT_SEED = [string]$Seed
$env:VGGT_MOSAIC_SEED = [string]$MosaicSeed
$env:VGGT_GPU_SPEC_PRECOMPUTE = "A100-80GB"
$env:VGGT_GPU_SPEC_INFER = "A100-80GB"

$scriptPath = Join-Path $RepoDir "scripts/run_overnight_ghost_autoloop.ps1"
$syncStatusScriptPath = Join-Path $RepoDir "scripts/sync_single_probe_latest.ps1"
$snapshotScriptPath = Join-Path $RepoDir "scripts/snapshot_human_transparency_probe.ps1"
$refreshSummaryScriptPath = Join-Path $RepoDir "scripts/refresh_support_probe_summary.ps1"
if (-not (Test-Path $scriptPath)) {
    throw "autoloop script not found: $scriptPath"
}
if (-not (Test-Path $syncStatusScriptPath)) {
    throw "sync status script not found: $syncStatusScriptPath"
}
if (-not (Test-Path $snapshotScriptPath)) {
    throw "snapshot script not found: $snapshotScriptPath"
}
if (-not (Test-Path $refreshSummaryScriptPath)) {
    throw "refresh summary script not found: $refreshSummaryScriptPath"
}

$contractLatestPath = Join-Path $RepoDir "logs/modal_phase5/probe_contract_latest.json"
$contractTag = Get-Date -Format "yyyyMMdd_HHmmss"
$contractStampedPath = Join-Path $RepoDir ("logs/modal_phase5/probe_contract_T0_smoke_$contractTag.json")
$contract = [ordered]@{
    probe_id = "T0_smoke"
    repo_dir = $RepoDir
    seq_names = $SeqNames
    start_resume_ckpt = $StartResumeCkpt
    start_pseudo_geom_subdir = $StartPseudoGeomSubdir
    seed = [int]$Seed
    mosaic_seed = [int]$MosaicSeed
    generated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    precompute_gpu_spec = "A100-80GB"
    infer_gpu_spec = "A100-80GB"
    pointmap_source = "depth_unproject"
    point_target_mode = "depth_unproject"
    precompute_mv_support_on = "off"
    point_target_blend_by_mv_support = "off"
    point_target_blend_mv_region_mode = "all"
    point_mv_depth_region_mode = "all"
    use_fg_mask = "on"
    fg_mask_source = "mask"
    lambda_point_mv_mask = "0"
    stage2_eval_num_src_views = "8"
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $contractLatestPath) | Out-Null
$contract | ConvertTo-Json -Depth 6 | Set-Content -Path $contractLatestPath -Encoding UTF8
$contract | ConvertTo-Json -Depth 6 | Set-Content -Path $contractStampedPath -Encoding UTF8

$runSucceeded = $false
$autoloopExitCode = $null
try {
    & $scriptPath `
        -SeqNames $SeqNames `
        -StartResumeCkpt $StartResumeCkpt `
        -StartPseudoGeomSubdir $StartPseudoGeomSubdir `
        -MaxCycles $MaxCycles `
        -StopAfterHours $StopAfterHours `
        -ForceStage2Only `
        -PointTargetMode 'depth_unproject' `
        -BaseUseFgMask 'on' `
        -BaseFgMaskSource 'mask' `
        -BasePointTargetBlendMvRegionMode 'all' `
        -BasePointMvDepthRegionMode 'all' `
        -EnablePersistentCycleState:$false `
        -EnableHistoricalSweepBootstrap:$false `
        -EmergencyGhostShockEnabled:$true `
        -Stage2DualLaneEnabled:$false `
        -PostRescueEnabled:$false `
        -Stage2EnableAnySplatAblationSixPack:$false `
        -Stage2DynProxyEnable 'off' `
        -Stage2EvalNumSrcViewsList '8' `
        -Stage2LambdaPointMvDepthList '0.001' `
        -Stage2LambdaPointMvMaskList '0' `
        -ModalRunQuiet:$false
    $autoloopExitCode = $LASTEXITCODE
    if ($autoloopExitCode -eq 0) {
        $runSucceeded = $true
    } else {
        throw "single-run autoloop exited with code $autoloopExitCode"
    }
}
finally {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $syncStatusScriptPath `
        -RepoDir $RepoDir `
        -ProbeId "T0_smoke" `
        -State $(if ($runSucceeded) { "done" } else { "error" }) `
        -ContractPath "logs/modal_phase5/probe_contract_latest.json" | Out-Null
    & powershell -NoProfile -ExecutionPolicy Bypass -File $snapshotScriptPath `
        -RepoDir $RepoDir `
        -ProbeId "T0_smoke" `
        -Label "manual_smoke" `
        -ContractPath "logs/modal_phase5/probe_contract_latest.json" `
        -OutRoot $SnapshotOutRoot | Out-Null
    try {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $refreshSummaryScriptPath `
            -RepoDir $RepoDir | Out-Null
    } catch {
        Write-Warning ("[t0-smoke] support summary refresh failed: " + $_.Exception.Message)
    }
    if ($null -eq $oldFtSeed) { Remove-Item Env:VGGT_FT_SEED -ErrorAction SilentlyContinue } else { $env:VGGT_FT_SEED = $oldFtSeed }
    if ($null -eq $oldMosaicSeed) { Remove-Item Env:VGGT_MOSAIC_SEED -ErrorAction SilentlyContinue } else { $env:VGGT_MOSAIC_SEED = $oldMosaicSeed }
    if ($null -eq $oldPrecomputeGpuSpec) { Remove-Item Env:VGGT_GPU_SPEC_PRECOMPUTE -ErrorAction SilentlyContinue } else { $env:VGGT_GPU_SPEC_PRECOMPUTE = $oldPrecomputeGpuSpec }
    if ($null -eq $oldInferGpuSpec) { Remove-Item Env:VGGT_GPU_SPEC_INFER -ErrorAction SilentlyContinue } else { $env:VGGT_GPU_SPEC_INFER = $oldInferGpuSpec }
}
