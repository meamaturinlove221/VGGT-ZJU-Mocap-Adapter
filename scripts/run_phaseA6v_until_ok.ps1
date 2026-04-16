param(
    [int]$MaxAttempts = 12,
    [int]$SleepSec = 30,
    [int]$NumSamples = 40,
    [string]$OutTag = "phaseA6v_fixed",
    [string]$InferArgsExtra = "--num_src_views=5",
    [int]$MetricsWaitTimeoutSec = 300
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$geomCandidates = "vggt_geom_orig6v_400_dbg:orig6v400_6vsrc5;" +
                  "vggt_geom_ft_lr_2e-6_20260209_022529:ft2e6_6vsrc5;" +
                  "vggt_geom_ft_lr_1e-6_20260209_022529:ft1e6_6vsrc5"

function Test-AllOk([string]$CsvPath) {
    if (-not (Test-Path $CsvPath)) {
        return $false
    }
    $rows = @(Import-Csv $CsvPath)
    if ($rows.Count -ne 3) {
        return $false
    }
    foreach ($r in $rows) {
        if ($r.status -ne "ok") {
            return $false
        }
    }
    return $true
}

$ok = $false
for ($i = 1; $i -le [Math]::Max(1, [int]$MaxAttempts); $i++) {
    Write-Host "[phaseA6v] attempt $i/$MaxAttempts"
    try {
        .\scripts\eval_geom_candidates.ps1 `
            -GeomCandidates $geomCandidates `
            -NumSamples $NumSamples `
            -OutTag $OutTag `
            -InferArgsExtra $InferArgsExtra `
            -DownloadVisSteps "0,2,4" `
            -DownloadVisCount 3 `
            -MinPSNR 0 `
            -MinSSIM 0 `
            -MaxWL1 999 `
            -MetricsWaitTimeoutSec $MetricsWaitTimeoutSec
    } catch {
        Write-Host "[phaseA6v] eval script threw: $($_.Exception.Message)"
    }

    if (Test-AllOk -CsvPath "logs/modal_phase5/baseline_compare_latest.csv") {
        $ok = $true
        $ts = Get-Date -Format "yyyyMMdd_HHmmss"
        Copy-Item "logs/modal_phase5/baseline_compare_latest.csv" "logs/modal_phase5/phaseA6v_strict_ok_$ts.csv" -Force
        Copy-Item "logs/modal_phase5/vggt_ft_gate_latest.json" "logs/modal_phase5/phaseA6v_strict_ok_$ts.json" -Force
        Write-Host "[phaseA6v] success on attempt $i"
        break
    }

    if ($i -lt $MaxAttempts) {
        Write-Host "[phaseA6v] not ready, sleep $SleepSec sec..."
        Start-Sleep -Seconds ([Math]::Max(1, [int]$SleepSec))
    }
}

if (-not $ok) {
    Write-Error "[phaseA6v] failed after $MaxAttempts attempts"
    exit 2
}

exit 0
