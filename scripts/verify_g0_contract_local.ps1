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
    [string]$BaselineLambdaPointMvMask = "0.0005"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir
if ($MosaicSeed -lt 0) { $MosaicSeed = $Seed }

powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoDir "scripts/run_p0_local_maintenance.ps1") -RepoDir $RepoDir | Out-Null
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoDir "scripts/check_candidate_result_consistency.ps1") -RepoDir $RepoDir | Out-Null
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoDir "scripts/preflight_p0_resume_local.ps1") -RepoDir $RepoDir | Out-Null

$latestContract = Join-Path $RepoDir "logs/modal_phase5/probe_contract_latest.json"
$g0LatestContract = Join-Path $RepoDir "logs/modal_phase5/probe_contract_g0_latest.json"
$backupContract = Join-Path $RepoDir "logs/modal_phase5/probe_contract_latest.before_g0_verify.json"
$hadLatestContract = Test-Path $latestContract
$restoreContract = $null
try {
    $candidateLatest = Get-Content (Join-Path $RepoDir "logs/modal_phase5/candidate_result_latest.json") -Encoding UTF8 -Raw | ConvertFrom-Json
    $candidateStage = [string]$candidateLatest.current_stage
    $candidateRunTs = [string]$candidateLatest.run_timestamp
    if ($candidateStage -match '^single_probe_(.+)$' -and -not [string]::IsNullOrWhiteSpace($candidateRunTs)) {
        $candidateProbeId = $Matches[1]
        $candidateContract = Join-Path $RepoDir ("logs/modal_phase5/probe_contract_{0}_{1}.json" -f $candidateProbeId, $candidateRunTs)
        if (Test-Path $candidateContract) {
            $restoreContract = $candidateContract
        }
    }
} catch {}
if ($hadLatestContract) {
    Copy-Item $latestContract $backupContract -Force
}

try {
    powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoDir "scripts/run_human_transparency_probe_once.ps1") `
        -RepoDir $RepoDir `
        -ProbeId G0 `
        -SeqNames $SeqNames `
        -ResumeCkpt $ResumeCkpt `
        -PseudoGeomSubdir $PseudoGeomSubdir `
        -Seed $Seed `
        -MosaicSeed $MosaicSeed `
        -EvalNumSrcViews $EvalNumSrcViews `
        -LambdaPointMvDepth $LambdaPointMvDepth `
        -BaselineLambdaPointMvMask $BaselineLambdaPointMvMask `
        -DryRun

    Copy-Item $latestContract $g0LatestContract -Force
    Write-Host ("[verify-g0] g0_contract_latest=" + $g0LatestContract)
    Get-Content $g0LatestContract -Encoding UTF8
}
finally {
    if ($restoreContract -and (Test-Path $restoreContract)) {
        Copy-Item $restoreContract $latestContract -Force
    } elseif ($hadLatestContract -and (Test-Path $backupContract)) {
        Copy-Item $backupContract $latestContract -Force
        Remove-Item $backupContract -Force -ErrorAction SilentlyContinue
    } elseif (-not $hadLatestContract) {
        Remove-Item $latestContract -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $backupContract -Force -ErrorAction SilentlyContinue
}

powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoDir "scripts/check_candidate_result_consistency.ps1") -RepoDir $RepoDir | Out-Null
