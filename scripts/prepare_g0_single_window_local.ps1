[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$ResumeCkpt = "/mnt/out/vggt/finetune/lr_1e-6_20260227_101300/ckpt/model_ft_zju.pt",
    [string]$PseudoGeomSubdir = "vggt_geom_ft_lr_1e-6_20260227_101300",
    [int]$Seed = 0,
    [int]$MosaicSeed = -1,
    [string]$EvalNumSrcViews = "8",
    [string]$LambdaPointMvDepth = "0.001",
    [string]$BaselineLambdaPointMvMask = "0.0005",
    [string]$OutRoot = "logs/modal_phase5/reports"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir
if ($MosaicSeed -lt 0) { $MosaicSeed = $Seed }

function Invoke-LocalScript([string]$RelativePath, [string[]]$ScriptArgs = @()) {
    $fullPath = Join-Path $RepoDir $RelativePath
    Write-Host "[g0-prep] running $RelativePath"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $fullPath @ScriptArgs
    if ($LASTEXITCODE -ne 0) {
        throw "local script failed: $RelativePath exit=$LASTEXITCODE"
    }
}

function Invoke-LocalScriptDirect([string]$RelativePath, [string[]]$ScriptArgs = @()) {
    $fullPath = Join-Path $RepoDir $RelativePath
    Write-Host "[g0-prep] running-direct $RelativePath"
    & $fullPath @ScriptArgs
    if ($LASTEXITCODE -ne 0) {
        throw "local script failed: $RelativePath exit=$LASTEXITCODE"
    }
}

function Copy-IfExists([string]$SourcePath, [string]$OutDir) {
    $resolved = $SourcePath
    if (-not [System.IO.Path]::IsPathRooted($resolved)) {
        $resolved = Join-Path $RepoDir $resolved
    }
    if (-not (Test-Path $resolved)) { return $null }
    $dest = Join-Path $OutDir (Split-Path $resolved -Leaf)
    Copy-Item $resolved $dest -Force
    return $dest
}

Invoke-LocalScript "scripts/run_p0_local_maintenance.ps1" @("-RepoDir", $RepoDir, "-SkipSnapshot", "-SkipManifest")
Invoke-LocalScript "scripts/check_candidate_result_consistency.ps1" @("-RepoDir", $RepoDir)
Invoke-LocalScript "scripts/preflight_p0_resume_local.ps1" @("-RepoDir", $RepoDir)
Invoke-LocalScript "scripts/verify_g0_contract_local.ps1" @(
    "-RepoDir", $RepoDir,
    "-SeqNames", $SeqNames,
    "-ResumeCkpt", $ResumeCkpt,
    "-PseudoGeomSubdir", $PseudoGeomSubdir,
    "-Seed", ([string]$Seed),
    "-MosaicSeed", ([string]$MosaicSeed),
    "-EvalNumSrcViews", $EvalNumSrcViews,
    "-LambdaPointMvDepth", $LambdaPointMvDepth,
    "-BaselineLambdaPointMvMask", $BaselineLambdaPointMvMask
)
Invoke-LocalScript "scripts/refresh_support_generation_multiframe.ps1" @("-RepoDir", $RepoDir)
Invoke-LocalScript "scripts/refresh_support_generation_g0_readiness.ps1" @("-RepoDir", $RepoDir)
Invoke-LocalScript "scripts/refresh_support_probe_summary.ps1" @("-RepoDir", $RepoDir)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $OutRoot ("g0_single_window_prep_" + $timestamp)
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Invoke-LocalScript "scripts/snapshot_human_transparency_probe.ps1" @(
    "-RepoDir", $RepoDir,
    "-ProbeId", "S0",
    "-Label", "before_G0_baseline"
)

$snapshotDir = $null
$snapshotRoot = Join-Path $RepoDir "logs/modal_phase5/snapshots"
if (Test-Path $snapshotRoot) {
    $snap = Get-ChildItem -Path $snapshotRoot -Directory -Filter "human_probe_S0_before_G0_baseline_*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($snap) {
        $snapshotDir = $snap.FullName
    }
}

$copied = @()
foreach ($path in @(
        "logs/modal_phase5/candidate_result_latest.json",
        "logs/modal_phase5/overnight_ghost_autoloop_latest.json",
        "logs/modal_phase5/watch_ghost_outputs_latest.json",
        "logs/modal_phase5/probe_contract_latest.json",
        "logs/modal_phase5/probe_contract_g0_latest.json",
        "logs/modal_phase5/human_probe_summary_latest.md",
        "logs/modal_phase5/human_probe_summary_latest.png",
        "logs/modal_phase5/human_probe_visual_grid_latest.png",
        "logs/modal_phase5/reports/support_generation_multiframe_latest.md",
        "logs/modal_phase5/reports/support_generation_multiframe_latest.png",
        "logs/modal_phase5/reports/support_generation_multiframe_latest.json",
        "logs/modal_phase5/reports/support_generation_g0_readiness_latest.md",
        "logs/modal_phase5/reports/support_generation_g0_readiness_latest.png",
        "logs/modal_phase5/reports/support_generation_g0_readiness_latest_zh.md",
        "logs/modal_phase5/reports/support_generation_g0_readiness_latest_zh.png"
    )) {
    $dest = Copy-IfExists -SourcePath $path -OutDir $outDir
    if ($dest) {
        $copied += [pscustomobject]@{
            source = $path
            copied_name = (Split-Path $dest -Leaf)
        }
    }
}

$summary = [ordered]@{
    generated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    mode = "local_only"
    ready_for_single_g0_cloud_window = $true
    seq_names = $SeqNames
    resume_ckpt = $ResumeCkpt
    pseudo_geom_subdir = $PseudoGeomSubdir
    seed = $Seed
    mosaic_seed = $MosaicSeed
    eval_num_src_views = $EvalNumSrcViews
    lambda_point_mv_depth = $LambdaPointMvDepth
    baseline_lambda_point_mv_mask = $BaselineLambdaPointMvMask
    frozen_candidate_stage = "single_probe_S0"
    prep_dir = (Join-Path $RepoDir $outDir)
    frozen_baseline_snapshot_dir = $snapshotDir
    key_contract = "logs/modal_phase5/probe_contract_g0_latest.json"
    copied_files = $copied
}

$summaryJson = Join-Path $outDir "g0_prep_summary.json"
$summaryMd = Join-Path $outDir "g0_prep_summary.md"
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryJson -Encoding UTF8

$snapshotLine = if ($snapshotDir) {
    "- frozen baseline snapshot: $snapshotDir"
} else {
    "- frozen baseline snapshot: <not found>"
}

$lines = @(
    "# G0 Single-Window Local Preparation",
    "",
    "- generated_at: $($summary.generated_at)",
    "- mode: local_only",
    "- ready_for_single_g0_cloud_window: true",
    "- seq_names: $SeqNames",
    "- resume_ckpt: $ResumeCkpt",
    "- pseudo_geom_subdir: $PseudoGeomSubdir",
    "- seed: $Seed",
    "- mosaic_seed: $MosaicSeed",
    "- eval_num_src_views: $EvalNumSrcViews",
    "",
    "## Key Contracts",
    "",
    '- frozen latest: `logs/modal_phase5/probe_contract_latest.json`',
    '- target G0: `logs/modal_phase5/probe_contract_g0_latest.json`',
    $snapshotLine,
    "",
    "## Copied Files"
)
foreach ($item in $copied) {
    $lines += "- $($item.source)"
}
$lines | Set-Content -Path $summaryMd -Encoding UTF8

Write-Host "[g0-prep] ready_for_single_g0_cloud_window=true"
Write-Host ("[g0-prep] out_dir=" + (Join-Path $RepoDir $outDir))
