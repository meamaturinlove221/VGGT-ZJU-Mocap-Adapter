[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

$checks = @(
    @{ name = "paused_state"; cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_paused_state.ps1" },
    @{ name = "p0_local_readiness"; cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p0_local_readiness.ps1" },
    @{ name = "p0_source_contract"; cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p0_source_contract.ps1" },
    @{ name = "candidate_result_consistency"; cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_candidate_result_consistency.ps1" }
)

$failed = $false
foreach ($check in $checks) {
    Write-Host "[preflight] running $($check.name)"
    Invoke-Expression $check.cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[preflight] FAIL $($check.name) exit=$LASTEXITCODE"
        $failed = $true
    } else {
        Write-Host "[preflight] OK   $($check.name)"
    }
}

if ($failed) {
    Write-Host "[preflight] result=not_ready"
    exit 2
}

Write-Host "[preflight] result=ready_for_manual_single_p0_resume"
