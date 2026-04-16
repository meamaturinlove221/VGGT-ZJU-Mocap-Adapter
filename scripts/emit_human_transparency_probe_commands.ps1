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
    [string]$OutMdPath = "logs/modal_phase5/human_transparency_probe_commands_latest.md"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if ($MosaicSeed -lt 0) { $MosaicSeed = $Seed }
Set-Location $RepoDir

function Q([string]$Text) {
    return '"' + $Text + '"'
}

$maintCmd = "powershell -NoProfile -ExecutionPolicy Bypass -File " + (Q (Join-Path $RepoDir "scripts/run_p0_local_maintenance.ps1")) + " -RepoDir " + (Q $RepoDir)
$consistencyCmd = "powershell -NoProfile -ExecutionPolicy Bypass -File " + (Q (Join-Path $RepoDir "scripts/check_candidate_result_consistency.ps1")) + " -RepoDir " + (Q $RepoDir)
$preflightCmd = "powershell -NoProfile -ExecutionPolicy Bypass -File " + (Q (Join-Path $RepoDir "scripts/preflight_p0_resume_local.ps1")) + " -RepoDir " + (Q $RepoDir)
$g0PrepCmd = "powershell -NoProfile -ExecutionPolicy Bypass -File " + (Q (Join-Path $RepoDir "scripts/prepare_g0_single_window_local.ps1")) + " -RepoDir " + (Q $RepoDir) + " -SeqNames " + (Q $SeqNames) + " -ResumeCkpt " + (Q $ResumeCkpt) + " -PseudoGeomSubdir " + (Q $PseudoGeomSubdir) + " -Seed " + (Q ([string]$Seed)) + " -MosaicSeed " + (Q ([string]$MosaicSeed)) + " -EvalNumSrcViews " + (Q $EvalNumSrcViews) + " -LambdaPointMvDepth " + (Q $LambdaPointMvDepth) + " -BaselineLambdaPointMvMask " + (Q $BaselineLambdaPointMvMask)

$t0Cmd = @(
    "powershell -NoProfile -ExecutionPolicy Bypass -File",
    (Q (Join-Path $RepoDir "scripts/run_p0_single_resume_once.ps1")),
    "-RepoDir", (Q $RepoDir),
    "-SeqNames", (Q $SeqNames),
    "-StartResumeCkpt", (Q $ResumeCkpt),
    "-StartPseudoGeomSubdir", (Q $PseudoGeomSubdir),
    "-Seed", (Q ([string]$Seed)),
    "-MosaicSeed", (Q ([string]$MosaicSeed))
) -join " "

function New-ProbeCommand([string]$ProbeId) {
    return (@(
        "powershell -NoProfile -ExecutionPolicy Bypass -File",
        (Q (Join-Path $RepoDir "scripts/run_human_transparency_probe_once.ps1")),
        "-RepoDir", (Q $RepoDir),
        "-ProbeId", (Q $ProbeId),
        "-SeqNames", (Q $SeqNames),
        "-ResumeCkpt", (Q $ResumeCkpt),
        "-PseudoGeomSubdir", (Q $PseudoGeomSubdir),
        "-Seed", (Q ([string]$Seed)),
        "-MosaicSeed", (Q ([string]$MosaicSeed)),
        "-EvalNumSrcViews", (Q $EvalNumSrcViews),
        "-LambdaPointMvDepth", (Q $LambdaPointMvDepth),
        "-BaselineLambdaPointMvMask", (Q $BaselineLambdaPointMvMask)
    ) -join " ")
}

$g0Cmd = New-ProbeCommand -ProbeId "G0"
$s0Cmd = New-ProbeCommand -ProbeId "S0"
$s1Cmd = New-ProbeCommand -ProbeId "S1"
$s2Cmd = New-ProbeCommand -ProbeId "S2"
$s3Cmd = New-ProbeCommand -ProbeId "S3"

$lines = @(
    "# Human Transparency Probe Commands",
    "",
    "- generated_at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')",
    "- repo: $RepoDir",
    "- seq: $SeqNames",
    "- seed: $Seed",
    "- mosaic_seed: $MosaicSeed",
    "- eval_num_src_views: $EvalNumSrcViews",
    "- lambda_point_mv_depth: $LambdaPointMvDepth",
    "- baseline_lambda_point_mv_mask: $BaselineLambdaPointMvMask",
    "",
    "## Local Freeze-Line Checks",
    "",
    '```powershell',
    $maintCmd,
    $consistencyCmd,
    $preflightCmd,
    '```',
    "",
    "## Local G0 Contract Dry-Run",
    "",
    "Use this before giving `G0` a real cloud slot. It verifies the exact wrapper contract without starting cloud execution.",
    "",
    '```powershell',
    ("powershell -NoProfile -ExecutionPolicy Bypass -File " + (Q (Join-Path $RepoDir "scripts/verify_g0_contract_local.ps1")) + " -RepoDir " + (Q $RepoDir) + " -SeqNames " + (Q $SeqNames) + " -ResumeCkpt " + (Q $ResumeCkpt) + " -PseudoGeomSubdir " + (Q $PseudoGeomSubdir) + " -Seed " + (Q ([string]$Seed)) + " -MosaicSeed " + (Q ([string]$MosaicSeed)) + " -EvalNumSrcViews " + (Q $EvalNumSrcViews) + " -LambdaPointMvDepth " + (Q $LambdaPointMvDepth) + " -BaselineLambdaPointMvMask " + (Q $BaselineLambdaPointMvMask)),
    '```',
    "",
    "## Local G0 Single-Window Preparation",
    "",
    "Use this before a real `G0` cloud slot. It runs freeze-line checks, validates the `G0` contract, refreshes summaries, and snapshots the current `S0` baseline.",
    "",
    '```powershell',
    $g0PrepCmd,
    '```',
    "",
    "## Phase 1",
    "",
    "### T0-smoke",
    "",
    '```powershell',
    $t0Cmd,
    '```',
    "",
    "## Phase 2",
    "",
    'Run these only after `T0-smoke` passes both execution and audit.',
    "",
    "### G0",
    "",
    '```powershell',
    $g0Cmd,
    '```',
    "",
    "### S0",
    "",
    '```powershell',
    $s0Cmd,
    '```',
    "",
    "### S1",
    "",
    '```powershell',
    $s1Cmd,
    '```',
    "",
    "### S2",
    "",
    '```powershell',
    $s2Cmd,
    '```',
    "",
    "### S3",
    "",
    '```powershell',
    $s3Cmd,
    '```'
)

$outFull = Join-Path $RepoDir $OutMdPath
$outDir = Split-Path -Parent $outFull
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Set-Content -Path $outFull -Value $lines -Encoding UTF8
Write-Host ("[emit-probe-cmd] wrote " + $outFull)
