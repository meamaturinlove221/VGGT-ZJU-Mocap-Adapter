[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [ValidateSet("F0","F1","F2","P1","R1","R2")]
    [string]$ProbeId,
    [string]$BaselineCandidatePath = "logs/modal_phase5/candidate_result_latest.json",
    [string]$BaselineLabel = "Working baseline F0 px=5",
    [string]$BaselineContractPath = "",
    [string]$SnapshotOutRoot = "logs/modal_phase5/snapshots",
    [string]$ReportOutRoot = "logs/modal_phase5/reports",
    [double]$LambdaFgConfPresenceOverride = [double]::NaN,
    [double]$FgConfPresenceTargetRatioOverride = [double]::NaN
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
            Write-Host "[fg-stage] stopping modal app $appId state=$($app.'State')"
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
        $matches = @(Get-ChildItem -Path (Resolve-RepoPath "logs/modal_phase5") -Filter ("probe_contract_*_{0}.json" -f $runTimestamp) -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending)
        if ($matches.Count -gt 0) {
            return $matches[0].FullName
        }
    }
    return (Resolve-RepoPath "logs/modal_phase5/probe_contract_latest.json")
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

Write-Host "[fg-stage] baseline_json=$baselineCandidateStamped"
Write-Host "[fg-stage] baseline_contract=$baselineContractResolved"
Write-Host "[fg-stage] probe=$ProbeId"

Ensure-ModalAppsStopped
& powershell -NoProfile -ExecutionPolicy Bypass -File $consistencyScript -RepoDir $RepoDir
if ($LASTEXITCODE -ne 0) {
    throw "pre-run consistency check failed with exit code $LASTEXITCODE"
}

$stageLogPath = Resolve-RepoPath ("logs/modal_phase5/{0}_fg_presence_{1}.local.out.log" -f $ProbeId.ToLowerInvariant(), (Get-Date -Format "yyyyMMdd_HHmmss"))
$probeArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $probeScript,
    "-RepoDir", $RepoDir,
    "-ProbeId", $ProbeId,
    "-InheritContractPath", $baselineContractResolved,
    "-SeqNames", ([string]$baselineContract.seq_names),
    "-ResumeCkpt", ([string]$baselineContract.resume_ckpt),
    "-PseudoGeomSubdir", ([string]$baselineContract.pseudo_geom_subdir),
    "-Seed", ([string]$baselineContract.seed),
    "-MosaicSeed", ([string]$baselineContract.mosaic_seed),
    "-EvalNumSrcViews", ([string]$baselineContract.eval_num_src_views),
    "-LambdaPointMvDepth", ([string]$baselineContract.lambda_point_mv_depth),
    "-PrecomputeMvSupportFgPreservePx", ([string]$baselineContract.precompute_mv_support_fg_preserve_px),
    "-SnapshotOutRoot", $SnapshotOutRoot
)
if (-not [double]::IsNaN([double]$LambdaFgConfPresenceOverride)) {
    $probeArgs += @("-LambdaFgConfPresence", ([string]$LambdaFgConfPresenceOverride))
}
if (-not [double]::IsNaN([double]$FgConfPresenceTargetRatioOverride)) {
    $probeArgs += @("-FgConfPresenceTargetRatio", ([string]$FgConfPresenceTargetRatioOverride))
}
Write-Host "[fg-stage] log=$stageLogPath"
& powershell @probeArgs *>&1 | Tee-Object -FilePath $stageLogPath
$probeExit = $LASTEXITCODE
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
$stageSnapshotDir = $null
$snapshotRootResolved = Resolve-RepoPath $SnapshotOutRoot
if (Test-Path $snapshotRootResolved) {
    $snap = Get-ChildItem -Path $snapshotRootResolved -Directory -Filter ("human_probe_{0}_manual_probe_*" -f $ProbeId) |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $snap) {
        $stageSnapshotDir = $snap.FullName
    }
}

$reportMd = Resolve-RepoPath ("{0}/fg_presence_{1}_compare_en_latest.md" -f $ReportOutRoot, $ProbeId.ToLowerInvariant())
$reportPng = Resolve-RepoPath ("{0}/fg_presence_{1}_compare_en_latest.png" -f $ReportOutRoot, $ProbeId.ToLowerInvariant())
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
    snapshot_dir = $stageSnapshotDir
    log_path = $stageLogPath
    report_md = $reportMd
    report_png = $reportPng
    ghost_visual_score = To-DoubleLoose $currentCandidate.ghost_visual_score
    fg_pred_luma_mean = To-DoubleLoose $currentCandidate.fg_pred_luma_mean
    fg_pred_contrast = To-DoubleLoose $currentCandidate.fg_pred_contrast
    fg_pred_tgt_l1 = To-DoubleLoose $currentCandidate.fg_pred_tgt_l1
    fg_supervision_boost = To-DoubleLoose $currentCandidate.fg_supervision_boost
    fg_supervision_bg_floor = To-DoubleLoose $currentCandidate.fg_supervision_bg_floor
    fg_supervision_region_mode = [string]$currentCandidate.fg_supervision_region_mode
    fg_supervision_region_erode_px = To-DoubleLoose $currentCandidate.fg_supervision_region_erode_px
    fg_conf_presence_target_ratio = To-DoubleLoose $currentCandidate.fg_conf_presence_target_ratio
    fg_conf_presence_enabled = To-DoubleLoose $currentCandidate.fg_conf_presence_enabled
    fg_conf_presence_pred_mean = To-DoubleLoose $currentCandidate.fg_conf_presence_pred_mean
    fg_conf_presence_tgt_mean = To-DoubleLoose $currentCandidate.fg_conf_presence_tgt_mean
    fg_conf_presence_target_floor = To-DoubleLoose $currentCandidate.fg_conf_presence_target_floor
    fg_conf_presence_active_ratio = To-DoubleLoose $currentCandidate.fg_conf_presence_active_ratio
    fg_conf_presence_loss = To-DoubleLoose $currentCandidate.fg_conf_presence_loss
    mean_loss_fg_conf_presence = To-DoubleLoose $currentCandidate.mean_loss_fg_conf_presence
    width_ratio_mean = To-DoubleLoose $currentCandidate.width_ratio_mean
    area_ratio_mean = To-DoubleLoose $currentCandidate.area_ratio_mean
    delta_ghost_visual_score = (To-DoubleLoose $currentCandidate.ghost_visual_score) - (To-DoubleLoose $baselineCandidate.ghost_visual_score)
    delta_fg_pred_luma_mean = (To-DoubleLoose $currentCandidate.fg_pred_luma_mean) - (To-DoubleLoose $baselineCandidate.fg_pred_luma_mean)
    delta_fg_pred_contrast = (To-DoubleLoose $currentCandidate.fg_pred_contrast) - (To-DoubleLoose $baselineCandidate.fg_pred_contrast)
    delta_fg_pred_tgt_l1 = (To-DoubleLoose $currentCandidate.fg_pred_tgt_l1) - (To-DoubleLoose $baselineCandidate.fg_pred_tgt_l1)
    updated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
}
$summaryPath = Resolve-RepoPath ("{0}/fg_presence_{1}_summary_latest.json" -f $ReportOutRoot, $ProbeId.ToLowerInvariant())
$summary | ConvertTo-Json -Depth 6 | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host "[fg-stage] summary=$summaryPath"
