[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [ValidateSet("H0","H1","H1a","H1b","H1c","H1d","H1e","H1s1","H1s2","H1s1_core","H1s2_core","H1sf1","H1sf2","H2")]
    [string]$ProbeId,
    [string]$BaselineCandidatePath = "logs/modal_phase5/candidate_result_latest.json",
    [string]$BaselineLabel = "Working baseline F0 px=5",
    [string]$BaselineContractPath = "",
    [string]$SnapshotOutRoot = "logs/modal_phase5/snapshots",
    [string]$ReportOutRoot = "logs/modal_phase5/reports"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

$canonicalCamNames = "Camera_B1,Camera_B2,Camera_B3,Camera_B4,Camera_B5,Camera_B6,Camera_B7,Camera_B8,Camera_B9,Camera_B10,Camera_B11,Camera_B12,Camera_B13,Camera_B14,Camera_B15,Camera_B16,Camera_B17,Camera_B18,Camera_B19,Camera_B20,Camera_B21,Camera_B22,Camera_B23"

function Resolve-RepoPath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoDir $Path
}

function Read-JsonMaybe([string]$Path) {
    $resolved = Resolve-RepoPath $Path
    if (-not (Test-Path $resolved)) { return $null }
    try {
        return (Get-Content -Raw -Path $resolved -Encoding UTF8 | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function To-DoubleLoose($Value, [double]$Default = [double]::NaN) {
    if ($null -eq $Value) { return $Default }
    $raw = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }
    $out = 0.0
    if ([double]::TryParse($raw, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$out)) {
        return $out
    }
    return $Default
}

function To-BoolLoose($Value, [bool]$Default = $false) {
    if ($null -eq $Value) { return $Default }
    if ($Value -is [bool]) { return [bool]$Value }
    $raw = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }
    switch ($raw.ToLowerInvariant()) {
        "1" { return $true }
        "true" { return $true }
        "yes" { return $true }
        "y" { return $true }
        "on" { return $true }
        "0" { return $false }
        "false" { return $false }
        "no" { return $false }
        "n" { return $false }
        "off" { return $false }
        default { return $Default }
    }
}

function Ensure-ModalAppsStopped() {
    $appsRaw = modal app list --json | Out-String
    $apps = @()
    if (-not [string]::IsNullOrWhiteSpace($appsRaw)) {
        $parsed = $appsRaw | ConvertFrom-Json
        if ($parsed -is [System.Array]) {
            $apps = $parsed
        } elseif ($null -ne $parsed) {
            $apps = @($parsed)
        }
    }
    $active = @($apps | Where-Object { ([string]$_.'State').ToLowerInvariant() -ne 'stopped' })
    foreach ($app in $active) {
        $appId = [string]$app.'App ID'
        if (-not [string]::IsNullOrWhiteSpace($appId)) {
            Write-Host "[fg-structure-stage] stopping modal app $appId state=$($app.'State')"
            modal app stop $appId | Out-Null
        }
    }
}

function Resolve-BaselineContractPath([object]$Candidate, [string]$FallbackPath) {
    if (-not [string]::IsNullOrWhiteSpace($FallbackPath)) {
        return (Resolve-RepoPath $FallbackPath)
    }
    $runTimestamp = ""
    try { $runTimestamp = [string]$Candidate.run_timestamp } catch {}
    if (-not [string]::IsNullOrWhiteSpace($runTimestamp)) {
        $snapshotRoot = Resolve-RepoPath "logs/modal_phase5/snapshots"
        if (Test-Path $snapshotRoot) {
            foreach ($pattern in @("human_probe_F0_working_baseline_*", "human_probe_F0_*", "human_probe_*")) {
                $snapshotDirs = @(Get-ChildItem -Path $snapshotRoot -Directory -Filter $pattern -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending)
                foreach ($snapshotDir in $snapshotDirs) {
                    $snapshotContract = Join-Path $snapshotDir.FullName "probe_contract_latest.json"
                    if (-not (Test-Path $snapshotContract)) { continue }
                    $snapshotCandidates = @(Get-ChildItem -Path $snapshotDir.FullName -File -Filter "candidate_result*.json" -ErrorAction SilentlyContinue)
                    foreach ($snapshotCandidate in $snapshotCandidates) {
                        $snapshotCandidateJson = $null
                        try {
                            $snapshotCandidateJson = Get-Content -Raw -Path $snapshotCandidate.FullName -Encoding UTF8 | ConvertFrom-Json
                        } catch {
                            continue
                        }
                        if ([string]$snapshotCandidateJson.run_timestamp -eq $runTimestamp) {
                            return $snapshotContract
                        }
                    }
                }
            }
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($runTimestamp)) {
        $matches = @(Get-ChildItem -Path (Resolve-RepoPath "logs/modal_phase5") -Filter ("probe_contract_*_{0}.json" -f $runTimestamp) -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending)
        if ($matches.Count -gt 0) {
            return $matches[0].FullName
        }
    }
    return (Resolve-RepoPath "logs/modal_phase5/probe_contract_latest.json")
}

function Get-CandidateString([object]$Candidate, [string[]]$Names, [string]$Default = "") {
    if ($null -eq $Candidate) { return $Default }
    foreach ($name in $Names) {
        $prop = $Candidate.PSObject.Properties[$name]
        if ($null -eq $prop) { continue }
        $text = ([string]$prop.Value).Trim()
        if (-not [string]::IsNullOrWhiteSpace($text)) {
            return $text
        }
    }
    return $Default
}

$probeScript = Resolve-RepoPath "scripts/run_human_transparency_probe_once.ps1"
$consistencyScript = Resolve-RepoPath "scripts/check_candidate_result_consistency.ps1"
$renderScript = Resolve-RepoPath "scripts/render_fg_presence_validation_compare.py"

$baselineCandidateResolved = Resolve-RepoPath $BaselineCandidatePath
$baselineCandidate = Read-JsonMaybe $baselineCandidateResolved
if ($null -eq $baselineCandidate) {
    throw "baseline candidate missing: $baselineCandidateResolved"
}
$baselineCandidateStamped = Resolve-RepoPath ([string]$baselineCandidate.candidate_result_json)
if ([string]::IsNullOrWhiteSpace($baselineCandidateStamped) -or -not (Test-Path $baselineCandidateStamped)) {
    $baselineCandidateStamped = $baselineCandidateResolved
}
$baselineContractResolved = Resolve-BaselineContractPath -Candidate $baselineCandidate -FallbackPath $BaselineContractPath
$baselineContract = Read-JsonMaybe $baselineContractResolved
if ($null -eq $baselineContract) {
    throw "baseline contract missing: $baselineContractResolved"
}
$effectiveSeqNames = Get-CandidateString -Candidate $baselineContract -Names @("seq_names") -Default (Get-CandidateString -Candidate $baselineCandidate -Names @("seq_names") -Default "")
$effectiveCamNames = Get-CandidateString -Candidate $baselineContract -Names @("cam_names") -Default (Get-CandidateString -Candidate $baselineCandidate -Names @("cam_names") -Default $canonicalCamNames)
$effectiveResumeCkpt = Get-CandidateString -Candidate $baselineContract -Names @("resume_ckpt") -Default (Get-CandidateString -Candidate $baselineCandidate -Names @("resume_ckpt") -Default "")
$effectivePseudoGeomSubdir = Get-CandidateString -Candidate $baselineContract -Names @("pseudo_geom_subdir") -Default (Get-CandidateString -Candidate $baselineCandidate -Names @("pseudo_geom_subdir", "source_geom_subdir", "input_geom_subdir") -Default "")
$effectiveSeed = Get-CandidateString -Candidate $baselineContract -Names @("seed") -Default (Get-CandidateString -Candidate $baselineCandidate -Names @("seed") -Default "")
$effectiveMosaicSeed = Get-CandidateString -Candidate $baselineContract -Names @("mosaic_seed") -Default (Get-CandidateString -Candidate $baselineCandidate -Names @("mosaic_seed") -Default "")
$effectiveEvalNumSrcViews = Get-CandidateString -Candidate $baselineContract -Names @("eval_num_src_views") -Default (Get-CandidateString -Candidate $baselineCandidate -Names @("eval_num_src_views") -Default "")
$effectiveLambdaPointMvDepth = Get-CandidateString -Candidate $baselineContract -Names @("lambda_point_mv_depth") -Default (Get-CandidateString -Candidate $baselineCandidate -Names @("lambda_point_mv_depth") -Default "")
$effectiveFgPreservePx = Get-CandidateString -Candidate $baselineContract -Names @("precompute_mv_support_fg_preserve_px") -Default (Get-CandidateString -Candidate $baselineCandidate -Names @("precompute_mv_support_fg_preserve_px") -Default "")
$effectiveTf32 = Get-CandidateString -Candidate $baselineContract -Names @("tf32") -Default (Get-CandidateString -Candidate $baselineCandidate -Names @("tf32","runner_tf32") -Default "1")
$effectiveAmp = Get-CandidateString -Candidate $baselineContract -Names @("amp") -Default (Get-CandidateString -Candidate $baselineCandidate -Names @("amp","runner_amp") -Default "1")
$effectiveStrictDeterministic = Get-CandidateString -Candidate $baselineContract -Names @("strict_deterministic") -Default (Get-CandidateString -Candidate $baselineCandidate -Names @("strict_deterministic","runner_strict_deterministic") -Default "0")
$effectiveTf32Bool = To-BoolLoose -Value $effectiveTf32 -Default $true
$effectiveAmpBool = To-BoolLoose -Value $effectiveAmp -Default $true
$effectiveStrictDeterministicBool = To-BoolLoose -Value $effectiveStrictDeterministic -Default $false
$effectiveReuseShortFtCkpt = ""
if ($ProbeId -eq "H1") {
    throw "ProbeId=H1 is deprecated; use H1s1_core, H1s2_core, H1sf1, or H1sf2."
}
if ($ProbeId -eq "H0") {
    $effectiveReuseShortFtCkpt = Get-CandidateString -Candidate $baselineCandidate -Names @("ft_ckpt") -Default ""
}

Write-Host "[fg-structure-stage] baseline_contract=$baselineContractResolved"
Write-Host "[fg-structure-stage] effective_seq_names=$effectiveSeqNames"
Write-Host "[fg-structure-stage] effective_cam_names=$effectiveCamNames"
Write-Host "[fg-structure-stage] effective_resume_ckpt=$effectiveResumeCkpt"
Write-Host "[fg-structure-stage] effective_pseudo_geom_subdir=$effectivePseudoGeomSubdir"
Write-Host "[fg-structure-stage] effective_seed=$effectiveSeed mosaic_seed=$effectiveMosaicSeed eval_num_src_views=$effectiveEvalNumSrcViews"
Write-Host "[fg-structure-stage] effective_lambda_point_mv_depth=$effectiveLambdaPointMvDepth fg_preserve_px=$effectiveFgPreservePx"
Write-Host "[fg-structure-stage] effective_tf32=$effectiveTf32 effective_amp=$effectiveAmp effective_strict_deterministic=$effectiveStrictDeterministic"
if (-not [string]::IsNullOrWhiteSpace($effectiveReuseShortFtCkpt)) {
    Write-Host "[fg-structure-stage] effective_reuse_short_ft_ckpt=$effectiveReuseShortFtCkpt"
}

Ensure-ModalAppsStopped
& powershell -NoProfile -ExecutionPolicy Bypass -File $consistencyScript -RepoDir $RepoDir
if ($LASTEXITCODE -ne 0) {
    throw "pre-run consistency check failed with exit code $LASTEXITCODE"
}

$stageLogPath = Resolve-RepoPath ("logs/modal_phase5/{0}_fg_structure_{1}.local.out.log" -f $ProbeId.ToLowerInvariant(), (Get-Date -Format "yyyyMMdd_HHmmss"))
$probeParams = @{
    RepoDir = $RepoDir
    ProbeId = $ProbeId
    InheritContractPath = $baselineContractResolved
    SeqNames = $effectiveSeqNames
    CamNames = $effectiveCamNames
    ResumeCkpt = $effectiveResumeCkpt
    PseudoGeomSubdir = $effectivePseudoGeomSubdir
    Seed = $effectiveSeed
    MosaicSeed = $effectiveMosaicSeed
    EvalNumSrcViews = $effectiveEvalNumSrcViews
    LambdaPointMvDepth = $effectiveLambdaPointMvDepth
    PrecomputeMvSupportFgPreservePx = $effectiveFgPreservePx
    Tf32 = [bool]$effectiveTf32Bool
    Amp = [bool]$effectiveAmpBool
    StrictDeterministic = [bool]$effectiveStrictDeterministicBool
    SnapshotOutRoot = $SnapshotOutRoot
}
if (-not [string]::IsNullOrWhiteSpace($effectiveReuseShortFtCkpt)) {
    $probeParams["ReuseShortFtCkpt"] = $effectiveReuseShortFtCkpt
}
& $probeScript @probeParams *>&1 | Tee-Object -FilePath $stageLogPath
$probeExit = $LASTEXITCODE
if ($null -eq $probeExit) {
    $probeExit = $(if ($?) { 0 } else { 1 })
}
Ensure-ModalAppsStopped
if ($probeExit -ne 0) {
    throw "probe run failed with exit code $probeExit"
}

& powershell -NoProfile -ExecutionPolicy Bypass -File $consistencyScript -RepoDir $RepoDir
if ($LASTEXITCODE -ne 0) {
    throw "post-run consistency check failed with exit code $LASTEXITCODE"
}

$currentCandidate = Read-JsonMaybe "logs/modal_phase5/candidate_result_latest.json"
if ($null -eq $currentCandidate) {
    throw "candidate_result_latest.json missing after $ProbeId"
}
$currentCandidateStamped = Resolve-RepoPath ([string]$currentCandidate.candidate_result_json)
if ([string]::IsNullOrWhiteSpace($currentCandidateStamped) -or -not (Test-Path $currentCandidateStamped)) {
    $currentCandidateStamped = Resolve-RepoPath "logs/modal_phase5/candidate_result_latest.json"
}
$reportMd = Resolve-RepoPath ("{0}/fg_structure_{1}_compare_en_latest.md" -f $ReportOutRoot, $ProbeId.ToLowerInvariant())
$reportPng = Resolve-RepoPath ("{0}/fg_structure_{1}_compare_en_latest.png" -f $ReportOutRoot, $ProbeId.ToLowerInvariant())
python $renderScript `
    --repo-dir $RepoDir `
    --baseline-json $baselineCandidateStamped `
    --baseline-label $BaselineLabel `
    --compare ("{0}={1}" -f $ProbeId, $currentCandidateStamped) `
    --out-md $reportMd `
    --out-png $reportPng
if ($LASTEXITCODE -ne 0) {
    throw "compare render failed for $ProbeId"
}

$summary = [ordered]@{
    probe_id = $ProbeId
    baseline_json = $baselineCandidateStamped
    baseline_contract = $baselineContractResolved
    current_candidate_json = $currentCandidateStamped
    log_path = $stageLogPath
    report_md = $reportMd
    report_png = $reportPng
    ghost_visual_score = To-DoubleLoose $currentCandidate.ghost_visual_score
    fg_pred_luma_mean = To-DoubleLoose $currentCandidate.fg_pred_luma_mean
    fg_pred_contrast = To-DoubleLoose $currentCandidate.fg_pred_contrast
    fg_pred_tgt_l1 = To-DoubleLoose $currentCandidate.fg_pred_tgt_l1
    lambda_fg_structure_depth_edge = To-DoubleLoose $currentCandidate.lambda_fg_structure_depth_edge
    fg_structure_bbox_margin_px = To-DoubleLoose $currentCandidate.fg_structure_bbox_margin_px
    fg_structure_bbox_min_side_px = To-DoubleLoose $currentCandidate.fg_structure_bbox_min_side_px
    fg_structure_region_mode = [string]$currentCandidate.fg_structure_region_mode
    fg_structure_region_erode_px = To-DoubleLoose -Value $currentCandidate.fg_structure_region_erode_px -Default 0.0
    fg_structure_depth_edge_warmup_steps = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_warmup_steps -Default 0.0
    fg_structure_boundary_probe_px = To-DoubleLoose -Value $currentCandidate.fg_structure_boundary_probe_px -Default 2.0
    fg_structure_edge_support_mode = [string]$currentCandidate.fg_structure_edge_support_mode
    fg_structure_edge_support_quantile = To-DoubleLoose -Value $currentCandidate.fg_structure_edge_support_quantile -Default 0.0
    fg_structure_edge_support_min_px = To-DoubleLoose -Value $currentCandidate.fg_structure_edge_support_min_px -Default 32.0
    fg_structure_edge_weight_mode = [string]$currentCandidate.fg_structure_edge_weight_mode
    fg_structure_boundary_falloff_px = To-DoubleLoose -Value $currentCandidate.fg_structure_boundary_falloff_px -Default 0.0
    fg_structure_component_bias_mode = [string]$currentCandidate.fg_structure_component_bias_mode
    fg_structure_component_bias_threshold_ratio = To-DoubleLoose -Value $currentCandidate.fg_structure_component_bias_threshold_ratio -Default 0.25
    fg_structure_component_bias_other_scale = To-DoubleLoose -Value $currentCandidate.fg_structure_component_bias_other_scale -Default 1.0
    fg_structure_front_depth_bias_mode = [string]$currentCandidate.fg_structure_front_depth_bias_mode
    fg_structure_front_depth_bias_tau = To-DoubleLoose -Value $currentCandidate.fg_structure_front_depth_bias_tau -Default 0.75
    fg_structure_front_depth_bias_center_quantile = To-DoubleLoose -Value $currentCandidate.fg_structure_front_depth_bias_center_quantile -Default 0.55
    lambda_point_mv_outside_ring = To-DoubleLoose $currentCandidate.lambda_point_mv_outside_ring
    point_mv_outside_ring_px = To-DoubleLoose $currentCandidate.point_mv_outside_ring_px
    effective_seq_names = $effectiveSeqNames
    effective_cam_names = $effectiveCamNames
    effective_resume_ckpt = $effectiveResumeCkpt
    effective_pseudo_geom_subdir = $effectivePseudoGeomSubdir
    effective_seed = $effectiveSeed
    effective_mosaic_seed = $effectiveMosaicSeed
    effective_eval_num_src_views = $effectiveEvalNumSrcViews
    effective_lambda_point_mv_depth = To-DoubleLoose -Value $effectiveLambdaPointMvDepth -Default 0.0
    effective_fg_preserve_px = To-DoubleLoose -Value $effectiveFgPreservePx -Default 0.0
    effective_tf32 = To-DoubleLoose -Value $effectiveTf32 -Default 1.0
    effective_amp = To-DoubleLoose -Value $effectiveAmp -Default 1.0
    effective_strict_deterministic = To-DoubleLoose -Value $effectiveStrictDeterministic -Default 0.0
    effective_reuse_short_ft_ckpt = $effectiveReuseShortFtCkpt
    tf32 = $(if (To-BoolLoose -Value $currentCandidate.tf32 -Default (To-BoolLoose -Value $currentCandidate.runner_tf32 -Default $false)) { 1.0 } else { 0.0 })
    amp = $(if (To-BoolLoose -Value $currentCandidate.amp -Default (To-BoolLoose -Value $currentCandidate.runner_amp -Default $false)) { 1.0 } else { 0.0 })
    strict_deterministic = $(if (To-BoolLoose -Value $currentCandidate.strict_deterministic -Default (To-BoolLoose -Value $currentCandidate.runner_strict_deterministic -Default $false)) { 1.0 } else { 0.0 })
    runner_tf32 = $(if (To-BoolLoose -Value $currentCandidate.runner_tf32 -Default $false) { 1.0 } else { 0.0 })
    runner_amp = $(if (To-BoolLoose -Value $currentCandidate.runner_amp -Default $false) { 1.0 } else { 0.0 })
    runner_strict_deterministic = $(if (To-BoolLoose -Value $currentCandidate.runner_strict_deterministic -Default $false) { 1.0 } else { 0.0 })
    precompute_tf32 = $(if (To-BoolLoose -Value $currentCandidate.precompute_tf32 -Default $false) { 1.0 } else { 0.0 })
    precompute_amp = $(if (To-BoolLoose -Value $currentCandidate.precompute_amp -Default $false) { 1.0 } else { 0.0 })
    precompute_strict_deterministic = $(if (To-BoolLoose -Value $currentCandidate.precompute_strict_deterministic -Default $false) { 1.0 } else { 0.0 })
    teacher_tf32 = $(if (To-BoolLoose -Value $currentCandidate.teacher_tf32 -Default $false) { 1.0 } else { 0.0 })
    teacher_amp = $(if (To-BoolLoose -Value $currentCandidate.teacher_amp -Default $false) { 1.0 } else { 0.0 })
    teacher_deterministic = $(if (To-BoolLoose -Value $currentCandidate.teacher_deterministic -Default $false) { 1.0 } else { 0.0 })
    fg_structure_depth_edge_active = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_active -Default 0.0
    fg_structure_bbox_cover = To-DoubleLoose -Value $currentCandidate.fg_structure_bbox_cover -Default 0.0
    fg_structure_region_cover = To-DoubleLoose -Value $currentCandidate.fg_structure_region_cover -Default 0.0
    fg_structure_effective_cover = To-DoubleLoose -Value $currentCandidate.fg_structure_effective_cover -Default 0.0
    fg_structure_boundary_probe_cover = To-DoubleLoose -Value $currentCandidate.fg_structure_boundary_probe_cover -Default 0.0
    fg_structure_bbox_active_ratio = To-DoubleLoose -Value $currentCandidate.fg_structure_bbox_active_ratio -Default 0.0
    fg_structure_region_active_ratio = To-DoubleLoose -Value $currentCandidate.fg_structure_region_active_ratio -Default 0.0
    fg_structure_boundary_band_active_ratio = To-DoubleLoose -Value $currentCandidate.fg_structure_boundary_band_active_ratio -Default 0.0
    fg_structure_depth_edge_active_views = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_active_views -Default 0.0
    fg_structure_depth_edge_boundary_active_views = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_boundary_active_views -Default 0.0
    fg_structure_depth_edge_loss = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_loss -Default 0.0
    fg_structure_depth_edge_loss_main = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_loss_main -Default 0.0
    fg_structure_depth_edge_loss_boundary_probe = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_loss_boundary_probe -Default 0.0
    fg_structure_depth_edge_loss_interior = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_loss_interior -Default 0.0
    fg_structure_depth_edge_loss_boundary_band = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_loss_boundary_band -Default 0.0
    fg_structure_depth_edge_boundary_probe_pred_mean = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_boundary_probe_pred_mean -Default 0.0
    fg_structure_depth_edge_boundary_probe_tgt_mean = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_boundary_probe_tgt_mean -Default 0.0
    fg_structure_depth_edge_boundary_pred_mean = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_boundary_pred_mean -Default 0.0
    fg_structure_depth_edge_boundary_tgt_mean = To-DoubleLoose -Value $currentCandidate.fg_structure_depth_edge_boundary_tgt_mean -Default 0.0
    fg_structure_target_edge_support_active = To-DoubleLoose -Value $currentCandidate.fg_structure_target_edge_support_active -Default 0.0
    fg_structure_target_edge_support_views = To-DoubleLoose -Value $currentCandidate.fg_structure_target_edge_support_views -Default 0.0
    fg_structure_target_edge_support_cover = To-DoubleLoose -Value $currentCandidate.fg_structure_target_edge_support_cover -Default 0.0
    fg_structure_target_edge_support_region_share = To-DoubleLoose -Value $currentCandidate.fg_structure_target_edge_support_region_share -Default 0.0
    fg_structure_target_edge_support_threshold_mean = To-DoubleLoose -Value $currentCandidate.fg_structure_target_edge_support_threshold_mean -Default 0.0
    fg_structure_main_weight_mean = To-DoubleLoose -Value $currentCandidate.fg_structure_main_weight_mean -Default 0.0
    fg_structure_boundary_distance_weight_share = To-DoubleLoose -Value $currentCandidate.fg_structure_boundary_distance_weight_share -Default 1.0
    main_support_component_count = To-DoubleLoose -Value $currentCandidate.main_support_component_count -Default 0.0
    main_support_largest_component_share = To-DoubleLoose -Value $currentCandidate.main_support_largest_component_share -Default 0.0
    main_support_top2_component_share = To-DoubleLoose -Value $currentCandidate.main_support_top2_component_share -Default 0.0
    main_support_centroid_distance_mean = To-DoubleLoose -Value $currentCandidate.main_support_centroid_distance_mean -Default 0.0
    main_support_component_active_views = To-DoubleLoose -Value $currentCandidate.main_support_component_active_views -Default 0.0
    main_support_component_bias_weight_share = To-DoubleLoose -Value $currentCandidate.main_support_component_bias_weight_share -Default 1.0
    fg_structure_front_depth_bias_weight_share = To-DoubleLoose -Value $currentCandidate.fg_structure_front_depth_bias_weight_share -Default 1.0
    fg_structure_front_depth_bias_active_views = To-DoubleLoose -Value $currentCandidate.fg_structure_front_depth_bias_active_views -Default 0.0
    main_support_depth_mode_count = To-DoubleLoose -Value $currentCandidate.main_support_depth_mode_count -Default 0.0
    main_support_back_mode_share = To-DoubleLoose -Value $currentCandidate.main_support_back_mode_share -Default 0.0
    main_support_front_back_gap = To-DoubleLoose -Value $currentCandidate.main_support_front_back_gap -Default 0.0
    main_support_depth_hist_peak_ratio = To-DoubleLoose -Value $currentCandidate.main_support_depth_hist_peak_ratio -Default 0.0
    main_support_secondary_risk = To-DoubleLoose -Value $currentCandidate.main_support_secondary_risk -Default 0.0
    main_support_depth_mode_active_views = To-DoubleLoose -Value $currentCandidate.main_support_depth_mode_active_views -Default 0.0
    point_mv_outside_ring_active = To-DoubleLoose -Value $currentCandidate.point_mv_outside_ring_active -Default 0.0
    point_mv_outside_ring_active_views = To-DoubleLoose -Value $currentCandidate.point_mv_outside_ring_active_views -Default 0.0
    point_mv_outside_ring_hit_ratio = To-DoubleLoose -Value $currentCandidate.point_mv_outside_ring_hit_ratio -Default 0.0
    point_mv_outside_ring_loss = To-DoubleLoose -Value $currentCandidate.point_mv_outside_ring_loss -Default 0.0
    updated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
}
$summaryPath = Resolve-RepoPath ("{0}/fg_structure_{1}_summary_latest.json" -f $ReportOutRoot, $ProbeId.ToLowerInvariant())
$summary | ConvertTo-Json -Depth 6 | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host "[fg-structure-stage] summary=$summaryPath"
