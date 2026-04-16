[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$BaselineCandidatePath = "logs/modal_phase5/candidate_result_latest.json",
    [string]$BaselineLabel = "Working baseline F0 px=5",
    [string]$BaselineContractPath = "",
    [string]$ReportOutRoot = "logs/modal_phase5/reports",
    [string]$ZjuRoot = "F:\datasets\ZJU_MoCap\data\zju_mocap",
    [string]$LocalDiagCamera = "Camera_B1",
    [string]$LocalDiagFrameIndices = "0,1,2"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

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

function Invoke-StrictDryRunProbe(
    [string]$ProbeId,
    [object]$BaselineContract
) {
    $probeScript = Resolve-RepoPath "scripts/run_human_transparency_probe_once.ps1"
    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $probeScript,
        "-RepoDir", $RepoDir,
        "-ProbeId", $ProbeId,
        "-InheritContractPath", $baselineContract.contract_path,
        "-SeqNames", ([string]$BaselineContract.seq_names),
        "-ResumeCkpt", ([string]$BaselineContract.resume_ckpt),
        "-PseudoGeomSubdir", ([string]$BaselineContract.pseudo_geom_subdir),
        "-Seed", ([string]$BaselineContract.seed),
        "-MosaicSeed", ([string]$BaselineContract.mosaic_seed),
        "-EvalNumSrcViews", ([string]$BaselineContract.eval_num_src_views),
        "-LambdaPointMvDepth", ([string]$BaselineContract.lambda_point_mv_depth),
        "-PrecomputeMvSupportFgPreservePx", ([string]$BaselineContract.precompute_mv_support_fg_preserve_px),
        "-DryRun"
    )
    $output = & powershell @args
    if ($LASTEXITCODE -ne 0) {
        throw "dry-run failed for $ProbeId"
    }
    $latest = ""
    $stamped = ""
    foreach ($line in @($output)) {
        if ([string]$line -match 'contract_latest=(.+)$') {
            $latest = $matches[1].Trim()
        }
        if ([string]$line -match 'contract_stamped=(.+)$') {
            $stamped = $matches[1].Trim()
        }
    }
    if ([string]::IsNullOrWhiteSpace($stamped)) {
        throw "failed to resolve stamped contract path for $ProbeId"
    }
    return [pscustomobject]@{
        probe_id = $ProbeId
        contract_latest = $latest
        contract_stamped = $stamped
        contract = (Read-JsonMaybe $stamped)
    }
}

function Test-PowerShellParse([string[]]$Paths) {
    $errors = @()
    foreach ($path in $Paths) {
        $resolved = Resolve-RepoPath $path
        $tokens = $null
        $parseErrors = $null
        [System.Management.Automation.Language.Parser]::ParseFile($resolved, [ref]$tokens, [ref]$parseErrors) | Out-Null
        if ($parseErrors -and $parseErrors.Count -gt 0) {
            foreach ($err in $parseErrors) {
                $errors += ("{0}:{1}:{2} {3}" -f $resolved, $err.Extent.StartLineNumber, $err.Extent.StartColumnNumber, $err.Message)
            }
        }
    }
    if ($errors.Count -gt 0) {
        throw ("powershell parser errors:`n" + ($errors -join "`n"))
    }
}

$consistencyScript = Resolve-RepoPath "scripts/check_candidate_result_consistency.ps1"
$renderDiagScript = Resolve-RepoPath "scripts/render_fg_structure_diagnostics.py"
$baselineCandidateResolved = Resolve-RepoPath $BaselineCandidatePath
$baselineCandidate = Read-JsonMaybe $baselineCandidateResolved
if ($null -eq $baselineCandidate) {
    throw "baseline candidate missing: $baselineCandidateResolved"
}
$baselineContractResolved = Resolve-BaselineContractPath -Candidate $baselineCandidate -FallbackPath $BaselineContractPath
$baselineContract = Read-JsonMaybe $baselineContractResolved
if ($null -eq $baselineContract) {
    throw "baseline contract missing: $baselineContractResolved"
}
$baselineContract | Add-Member -NotePropertyName contract_path -NotePropertyValue $baselineContractResolved -Force

& powershell -NoProfile -ExecutionPolicy Bypass -File $consistencyScript -RepoDir $RepoDir
if ($LASTEXITCODE -ne 0) {
    throw "baseline consistency check failed with exit code $LASTEXITCODE"
}

$probeLatestPath = Resolve-RepoPath "logs/modal_phase5/probe_contract_latest.json"
$probeLatestBackup = $null
if (Test-Path $probeLatestPath) {
    $probeLatestBackup = Get-Content -Raw -Path $probeLatestPath -Encoding UTF8
}

try {
    $dryRuns = @()
    foreach ($stage in @("H0", "H1s1_core", "H1s2_core", "H1sf1", "H1sf2")) {
        $dryRuns += (Invoke-StrictDryRunProbe -ProbeId $stage -BaselineContract $baselineContract)
    }
} finally {
    if ($null -ne $probeLatestBackup) {
        Set-Content -Path $probeLatestPath -Value $probeLatestBackup -Encoding UTF8
    }
}

$psFiles = @(
    "scripts/run_human_transparency_probe_once.ps1",
    "scripts/run_vggt_ft_lr_sweep.ps1",
    "scripts/run_vggt_ghost_mvdepth_sweep.ps1",
    "scripts/run_fg_structure_local_plan.ps1",
    "scripts/run_fg_structure_stage.ps1"
)
Test-PowerShellParse -Paths $psFiles

$pyFiles = @(
    "finetune_vggt_pseudo.py",
    "scripts/render_fg_structure_diagnostics.py"
)
$pyCompileArgs = @("-m", "py_compile") + ($pyFiles | ForEach-Object { Resolve-RepoPath $_ })
& python @pyCompileArgs
if ($LASTEXITCODE -ne 0) {
    throw "py_compile failed"
}

$pytestArgs = @(
    "-m", "pytest",
    (Resolve-RepoPath "tests/test_fg_presence_supervision.py"),
    (Resolve-RepoPath "tests/test_fg_structure_supervision.py"),
    "-q"
)
& python @pytestArgs
if ($LASTEXITCODE -ne 0) {
    throw "pytest failed"
}

$localDiagMd = Resolve-RepoPath ("{0}/fg_structure_local_diagnostics_en_latest.md" -f $ReportOutRoot)
$localDiagPng = Resolve-RepoPath ("{0}/fg_structure_local_diagnostics_en_latest.png" -f $ReportOutRoot)
$boundaryProbePx = "2"
if ($dryRuns.Count -gt 0 -and $null -ne $dryRuns[0].contract -and -not [string]::IsNullOrWhiteSpace([string]$dryRuns[0].contract.fg_structure_boundary_probe_px)) {
    $boundaryProbePx = [string]$dryRuns[0].contract.fg_structure_boundary_probe_px
}
$renderDiagArgs = @(
    $renderDiagScript,
    "--repo-dir", $RepoDir,
    "--contract-json", $baselineContractResolved,
    "--candidate-json", $baselineCandidateResolved,
    "--zju-root", $ZjuRoot,
    "--camera", $LocalDiagCamera,
    "--frame-indices", $LocalDiagFrameIndices,
    "--boundary-probe-px", $boundaryProbePx,
    "--out-md", $localDiagMd,
    "--out-png", $localDiagPng
)
foreach ($item in $dryRuns) {
    $c = $item.contract
    $frontDepthBiasMode = [string]$c.fg_structure_front_depth_bias_mode
    if ([string]::IsNullOrWhiteSpace($frontDepthBiasMode)) { $frontDepthBiasMode = "off" }
    $frontDepthBiasTau = [string]$c.fg_structure_front_depth_bias_tau
    if ([string]::IsNullOrWhiteSpace($frontDepthBiasTau)) { $frontDepthBiasTau = "0.75" }
    $frontDepthBiasCenterQuantile = [string]$c.fg_structure_front_depth_bias_center_quantile
    if ([string]::IsNullOrWhiteSpace($frontDepthBiasCenterQuantile)) { $frontDepthBiasCenterQuantile = "0.55" }
    $stageSpec = "{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}|{10}|{11}|{12}|{13}|{14}|{15}|{16}|{17}|{18}|{19}" -f `
        $item.probe_id, `
        [string]$c.lambda_fg_structure_depth_edge, `
        [string]$c.lambda_point_mv_outside_ring, `
        [string]$c.fg_structure_bbox_margin_px, `
        [string]$c.fg_structure_bbox_min_side_px, `
        [string]$c.point_mv_outside_ring_px, `
        [string]$c.fg_structure_region_mode, `
        [string]$c.fg_structure_region_erode_px, `
        [string]$c.fg_structure_depth_edge_warmup_steps, `
        [string]$c.fg_structure_edge_support_mode, `
        [string]$c.fg_structure_edge_support_quantile, `
        [string]$c.fg_structure_edge_support_min_px, `
        [string]$c.fg_structure_edge_weight_mode, `
        [string]$c.fg_structure_boundary_falloff_px, `
        [string]$c.fg_structure_component_bias_mode, `
        [string]$c.fg_structure_component_bias_threshold_ratio, `
        [string]$c.fg_structure_component_bias_other_scale, `
        $frontDepthBiasMode, `
        $frontDepthBiasTau, `
        $frontDepthBiasCenterQuantile
    $renderDiagArgs += @("--stage", $stageSpec)
}
& python @renderDiagArgs
if ($LASTEXITCODE -ne 0) {
    throw "local H-family diagnostics render failed"
}

$dryRunSummaries = @()
foreach ($item in $dryRuns) {
    $c = $item.contract
    $dryRunSummaries += [pscustomobject]@{
        probe_id = $item.probe_id
        contract_stamped = $item.contract_stamped
        fg_supervision_boost = [string]$c.fg_supervision_boost
        fg_supervision_bg_floor = [string]$c.fg_supervision_bg_floor
        lambda_fg_conf_presence = [string]$c.lambda_fg_conf_presence
        lambda_fg_structure_depth_edge = [string]$c.lambda_fg_structure_depth_edge
        fg_structure_bbox_margin_px = [string]$c.fg_structure_bbox_margin_px
        fg_structure_bbox_min_side_px = [string]$c.fg_structure_bbox_min_side_px
        fg_structure_region_mode = [string]$c.fg_structure_region_mode
        fg_structure_region_erode_px = [string]$c.fg_structure_region_erode_px
        fg_structure_depth_edge_warmup_steps = [string]$c.fg_structure_depth_edge_warmup_steps
        fg_structure_boundary_probe_px = [string]$c.fg_structure_boundary_probe_px
        fg_structure_edge_support_mode = [string]$c.fg_structure_edge_support_mode
        fg_structure_edge_support_quantile = [string]$c.fg_structure_edge_support_quantile
        fg_structure_edge_support_min_px = [string]$c.fg_structure_edge_support_min_px
        fg_structure_edge_weight_mode = [string]$c.fg_structure_edge_weight_mode
        fg_structure_boundary_falloff_px = [string]$c.fg_structure_boundary_falloff_px
        fg_structure_component_bias_mode = [string]$c.fg_structure_component_bias_mode
        fg_structure_component_bias_threshold_ratio = [string]$c.fg_structure_component_bias_threshold_ratio
        fg_structure_component_bias_other_scale = [string]$c.fg_structure_component_bias_other_scale
        fg_structure_front_depth_bias_mode = $(if ([string]::IsNullOrWhiteSpace([string]$c.fg_structure_front_depth_bias_mode)) { "off" } else { [string]$c.fg_structure_front_depth_bias_mode })
        fg_structure_front_depth_bias_tau = $(if ([string]::IsNullOrWhiteSpace([string]$c.fg_structure_front_depth_bias_tau)) { "0.75" } else { [string]$c.fg_structure_front_depth_bias_tau })
        fg_structure_front_depth_bias_center_quantile = $(if ([string]::IsNullOrWhiteSpace([string]$c.fg_structure_front_depth_bias_center_quantile)) { "0.55" } else { [string]$c.fg_structure_front_depth_bias_center_quantile })
        lambda_point_mv_outside_ring = [string]$c.lambda_point_mv_outside_ring
        point_mv_outside_ring_px = [string]$c.point_mv_outside_ring_px
        tf32 = [string]$c.tf32
        amp = [string]$c.amp
        strict_deterministic = [string]$c.strict_deterministic
    }
}

$summary = [pscustomobject]@{
    baseline_label = $BaselineLabel
    baseline_candidate_json = $baselineCandidateResolved
    baseline_contract_json = $baselineContractResolved
    baseline_tf32 = [string]$baselineContract.tf32
    baseline_amp = [string]$baselineContract.amp
    baseline_strict_deterministic = [string]$baselineContract.strict_deterministic
    boundary_probe_px = $boundaryProbePx
    dry_runs = $dryRunSummaries
    parser_checked = $psFiles
    py_compile_files = $pyFiles
    pytest_targets = @(
        "tests/test_fg_presence_supervision.py",
        "tests/test_fg_structure_supervision.py"
    )
    local_diagnostic_md = $localDiagMd
    local_diagnostic_png = $localDiagPng
    cloud_run_started = $false
    updated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
}
$summaryPath = Resolve-RepoPath ("{0}/fg_structure_local_plan_latest.json" -f $ReportOutRoot)
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host "[fg-structure-local-plan] summary=$summaryPath"
