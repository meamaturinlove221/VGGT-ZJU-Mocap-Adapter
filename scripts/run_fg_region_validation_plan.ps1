[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$BaselineCandidatePath = "logs/modal_phase5/candidate_result_latest.json",
    [string]$BaselineLabel = "Working baseline F0 px=5",
    [string]$BaselineContractPath = "",
    [string]$SnapshotOutRoot = "logs/modal_phase5/snapshots",
    [string]$ReportOutRoot = "logs/modal_phase5/reports",
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
            Write-Host "[fg-region-plan] stopping modal app $appId state=$($app.'State')"
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

function Test-StableButWeak([object]$Summary) {
    $ghost = To-DoubleLoose $Summary.ghost_visual_score
    $improvements = Get-ImprovementCount $Summary
    return (($ghost -le 4.98) -and ($improvements -lt 2))
}

$snapshotScript = Resolve-RepoPath "scripts/snapshot_human_transparency_probe.ps1"
$restoreScript = Resolve-RepoPath "scripts/restore_human_transparency_snapshot.ps1"
$stageScript = Resolve-RepoPath "scripts/run_fg_presence_stage.ps1"
$renderCompareScript = Resolve-RepoPath "scripts/render_fg_presence_validation_compare.py"
$renderDiagScript = Resolve-RepoPath "scripts/render_fg_supervision_region_diagnostics.py"
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
    -ProbeId "R0" `
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
    $snap = Get-ChildItem -Path (Resolve-RepoPath $SnapshotOutRoot) -Directory -Filter "human_probe_R0_working_baseline_*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $snap) {
        $baselineSnapshotDir = $snap.FullName
    }
}
if ([string]::IsNullOrWhiteSpace($baselineSnapshotDir)) {
    throw "failed to locate baseline snapshot directory"
}

$localDiagMd = Resolve-RepoPath ("{0}/fg_region_local_diagnostics_en_latest.md" -f $ReportOutRoot)
$localDiagPng = Resolve-RepoPath ("{0}/fg_region_local_diagnostics_en_latest.png" -f $ReportOutRoot)
python $renderDiagScript `
    --repo-dir $RepoDir `
    --contract-json $baselineContractResolved `
    --camera $LocalDiagCamera `
    --frame-indices $LocalDiagFrameIndices `
    --stage "R0|1.0|0.0|all|0" `
    --stage "F1-ref|1.5|0.05|all|0" `
    --stage "R2|1.3|0.05|interior_only|5" `
    --stage "R1|1.5|0.05|interior_only|3" `
    --out-md $localDiagMd `
    --out-png $localDiagPng
if ($LASTEXITCODE -ne 0) {
    throw "local region diagnostics render failed"
}

$stageResults = New-Object System.Collections.Generic.List[object]
$winner = "baseline"
$winnerSnapshotDir = $baselineSnapshotDir

& powershell -NoProfile -ExecutionPolicy Bypass -File $stageScript `
    -RepoDir $RepoDir `
    -ProbeId "R2" `
    -BaselineCandidatePath $baselineCandidateStamped `
    -BaselineLabel $BaselineLabel `
    -BaselineContractPath $baselineContractResolved `
    -SnapshotOutRoot $SnapshotOutRoot `
    -ReportOutRoot $ReportOutRoot
if ($LASTEXITCODE -ne 0) {
    throw "R2 stage failed"
}
$r2SummaryPath = Resolve-RepoPath ("{0}/fg_presence_r2_summary_latest.json" -f $ReportOutRoot)
$r2Summary = Read-JsonMaybe $r2SummaryPath
if ($null -eq $r2Summary) {
    throw "R2 summary missing: $r2SummaryPath"
}
$stageResults.Add($r2Summary) | Out-Null

if (Test-PromotionPass $r2Summary) {
    $winner = "R2"
    $winnerSnapshotDir = [string]$r2Summary.snapshot_dir
} elseif (Test-StableButWeak $r2Summary) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $stageScript `
        -RepoDir $RepoDir `
        -ProbeId "R1" `
        -BaselineCandidatePath $baselineCandidateStamped `
        -BaselineLabel $BaselineLabel `
        -BaselineContractPath $baselineContractResolved `
        -SnapshotOutRoot $SnapshotOutRoot `
        -ReportOutRoot $ReportOutRoot
    if ($LASTEXITCODE -ne 0) {
        throw "R1 stage failed"
    }
    $r1SummaryPath = Resolve-RepoPath ("{0}/fg_presence_r1_summary_latest.json" -f $ReportOutRoot)
    $r1Summary = Read-JsonMaybe $r1SummaryPath
    if ($null -eq $r1Summary) {
        throw "R1 summary missing: $r1SummaryPath"
    }
    $stageResults.Add($r1Summary) | Out-Null
    if (Test-PromotionPass $r1Summary) {
        $winner = "R1"
        $winnerSnapshotDir = [string]$r1Summary.snapshot_dir
    }
}

$compareArgs = @(
    "--repo-dir", $RepoDir,
    "--baseline-json", $baselineCandidateStamped,
    "--baseline-label", $BaselineLabel,
    "--out-md", (Resolve-RepoPath ("{0}/fg_region_validation_en_latest.md" -f $ReportOutRoot)),
    "--out-png", (Resolve-RepoPath ("{0}/fg_region_validation_en_latest.png" -f $ReportOutRoot))
)
foreach ($stageSummary in $stageResults) {
    $compareArgs += @("--compare", ("{0}={1}" -f ([string]$stageSummary.probe_id), ([string]$stageSummary.current_candidate_json)))
}
python $renderCompareScript @compareArgs
if ($LASTEXITCODE -ne 0) {
    throw "combined compare render failed"
}

if ($winner -eq "baseline") {
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

$stageArray = @()
foreach ($stageEntry in $stageResults) {
    $stageArray += $stageEntry
}

$planSummary = [pscustomobject]@{
    baseline_candidate_json = $baselineCandidateStamped
    baseline_contract_json = $baselineContractResolved
    baseline_snapshot_dir = $baselineSnapshotDir
    local_diagnostic_md = $localDiagMd
    local_diagnostic_png = $localDiagPng
    combined_report_md = (Resolve-RepoPath ("{0}/fg_region_validation_en_latest.md" -f $ReportOutRoot))
    combined_report_png = (Resolve-RepoPath ("{0}/fg_region_validation_en_latest.png" -f $ReportOutRoot))
    winner = $winner
    winner_snapshot_dir = $winnerSnapshotDir
    stages = $stageArray
    updated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
}
$planSummaryPath = Resolve-RepoPath ("{0}/fg_region_validation_plan_latest.json" -f $ReportOutRoot)
$planSummary | ConvertTo-Json -Depth 8 | Set-Content -Path $planSummaryPath -Encoding UTF8
Write-Host "[fg-region-plan] summary=$planSummaryPath winner=$winner"
