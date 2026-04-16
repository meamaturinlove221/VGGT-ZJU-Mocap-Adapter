[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$BaselineCandidatePath = "logs/modal_phase5/candidate_result_latest.json",
    [string]$BaselineLabel = "Working baseline F0 px=5",
    [string]$BaselineContractPath = "",
    [string]$SnapshotOutRoot = "logs/modal_phase5/snapshots",
    [string]$ReportOutRoot = "logs/modal_phase5/reports",
    [double]$LambdaFgConfPresence = 0.005,
    [double]$FgConfPresenceTargetRatio = 0.8
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
            Write-Host "[fg-presence-plan] stopping modal app $appId state=$($app.'State')"
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

function Get-ImprovementCount([object]$Summary) {
    $count = 0
    if ((To-DoubleLoose $Summary.delta_fg_pred_luma_mean) -gt 0.0) { $count += 1 }
    if ((To-DoubleLoose $Summary.delta_fg_pred_contrast) -gt 0.0) { $count += 1 }
    if ((To-DoubleLoose $Summary.delta_fg_pred_tgt_l1) -lt 0.0) { $count += 1 }
    return $count
}

function Test-PromotionPass([object]$Summary) {
    $ghost = To-DoubleLoose $Summary.ghost_visual_score
    $improvements = Get-ImprovementCount $Summary
    return (($ghost -le 4.98) -and ($improvements -ge 2))
}

$snapshotScript = Resolve-RepoPath "scripts/snapshot_human_transparency_probe.ps1"
$restoreScript = Resolve-RepoPath "scripts/restore_human_transparency_snapshot.ps1"
$stageScript = Resolve-RepoPath "scripts/run_fg_presence_stage.ps1"
$renderCompareScript = Resolve-RepoPath "scripts/render_fg_presence_validation_compare.py"
$renderDiagScript = Resolve-RepoPath "scripts/render_fg_presence_only_diagnostics.py"
$consistencyScript = Resolve-RepoPath "scripts/check_candidate_result_consistency.ps1"

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

Ensure-ModalAppsStopped
& powershell -NoProfile -ExecutionPolicy Bypass -File $consistencyScript -RepoDir $RepoDir
if ($LASTEXITCODE -ne 0) {
    throw "pre-plan consistency check failed with exit code $LASTEXITCODE"
}

$snapshotLog = & powershell -NoProfile -ExecutionPolicy Bypass -File $snapshotScript `
    -RepoDir $RepoDir `
    -ProbeId "F0" `
    -Label "working_baseline" `
    -ContractPath $baselineContractResolved `
    -OutRoot $SnapshotOutRoot
$baselineSnapshotDir = ""
foreach ($line in @($snapshotLog)) {
    if ([string]$line -match 'out_dir=(.+?)\s+files=') {
        $baselineSnapshotDir = $matches[1]
    }
}
if ([string]::IsNullOrWhiteSpace($baselineSnapshotDir)) {
    $snap = Get-ChildItem -Path (Resolve-RepoPath $SnapshotOutRoot) -Directory -Filter "human_probe_F0_working_baseline_*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $snap) {
        $baselineSnapshotDir = $snap.FullName
    }
}
if ([string]::IsNullOrWhiteSpace($baselineSnapshotDir)) {
    throw "failed to locate baseline snapshot directory"
}

$localDiagMd = Resolve-RepoPath ("{0}/fg_presence_only_local_diagnostics_en_latest.md" -f $ReportOutRoot)
$localDiagPng = Resolve-RepoPath ("{0}/fg_presence_only_local_diagnostics_en_latest.png" -f $ReportOutRoot)
python $renderDiagScript `
    --repo-dir $RepoDir `
    --baseline-json $baselineCandidateStamped `
    --baseline-label $BaselineLabel `
    --lambda-presence $LambdaFgConfPresence `
    --target-ratio $FgConfPresenceTargetRatio `
    --out-md $localDiagMd `
    --out-png $localDiagPng
if ($LASTEXITCODE -ne 0) {
    throw "local presence diagnostics render failed"
}

& powershell -NoProfile -ExecutionPolicy Bypass -File $stageScript `
    -RepoDir $RepoDir `
    -ProbeId "P1" `
    -BaselineCandidatePath $baselineCandidateStamped `
    -BaselineLabel $BaselineLabel `
    -BaselineContractPath $baselineContractResolved `
    -SnapshotOutRoot $SnapshotOutRoot `
    -ReportOutRoot $ReportOutRoot `
    -LambdaFgConfPresenceOverride $LambdaFgConfPresence `
    -FgConfPresenceTargetRatioOverride $FgConfPresenceTargetRatio
if ($LASTEXITCODE -ne 0) {
    throw "P1 stage failed"
}
$p1SummaryPath = Resolve-RepoPath ("{0}/fg_presence_p1_summary_latest.json" -f $ReportOutRoot)
$p1Summary = Read-JsonMaybe $p1SummaryPath
if ($null -eq $p1Summary) {
    throw "P1 summary missing: $p1SummaryPath"
}

$compareMd = Resolve-RepoPath ("{0}/fg_presence_only_validation_en_latest.md" -f $ReportOutRoot)
$comparePng = Resolve-RepoPath ("{0}/fg_presence_only_validation_en_latest.png" -f $ReportOutRoot)
python $renderCompareScript `
    --repo-dir $RepoDir `
    --baseline-json $baselineCandidateStamped `
    --baseline-label $BaselineLabel `
    --compare ("P1={0}" -f ([string]$p1Summary.current_candidate_json)) `
    --out-md $compareMd `
    --out-png $comparePng
if ($LASTEXITCODE -ne 0) {
    throw "presence compare render failed"
}

$winner = "baseline"
$winnerSnapshotDir = $baselineSnapshotDir
if (Test-PromotionPass $p1Summary) {
    $winner = "P1"
    $winnerSnapshotDir = [string]$p1Summary.snapshot_dir
} else {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $restoreScript -RepoDir $RepoDir -SnapshotDir $baselineSnapshotDir
    if ($LASTEXITCODE -ne 0) {
        throw "baseline restore failed"
    }
}

Ensure-ModalAppsStopped
& powershell -NoProfile -ExecutionPolicy Bypass -File $consistencyScript -RepoDir $RepoDir
if ($LASTEXITCODE -ne 0) {
    throw "final consistency check failed with exit code $LASTEXITCODE"
}

$latest = Read-JsonMaybe "logs/modal_phase5/candidate_result_latest.json"
$planSummary = [pscustomobject]@{
    baseline_candidate_json = $baselineCandidateStamped
    baseline_contract_json = $baselineContractResolved
    baseline_snapshot_dir = $baselineSnapshotDir
    local_diagnostic_md = $localDiagMd
    local_diagnostic_png = $localDiagPng
    combined_report_md = $compareMd
    combined_report_png = $comparePng
    winner = $winner
    winner_snapshot_dir = $winnerSnapshotDir
    stage = $p1Summary
    latest_candidate_json = Resolve-RepoPath ([string]$latest.candidate_result_json)
    latest_ghost_visual_score = To-DoubleLoose $latest.ghost_visual_score
    latest_fg_pred_luma_mean = To-DoubleLoose $latest.fg_pred_luma_mean
    latest_fg_pred_contrast = To-DoubleLoose $latest.fg_pred_contrast
    latest_fg_pred_tgt_l1 = To-DoubleLoose $latest.fg_pred_tgt_l1
    updated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
}
$planSummaryPath = Resolve-RepoPath ("{0}/fg_presence_only_plan_latest.json" -f $ReportOutRoot)
$planSummary | ConvertTo-Json -Depth 8 | Set-Content -Path $planSummaryPath -Encoding UTF8
Write-Host "[fg-presence-plan] summary=$planSummaryPath winner=$winner"
