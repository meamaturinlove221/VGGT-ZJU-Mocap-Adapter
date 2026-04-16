param(
    [string]$CodeDir = "F:\vggt",
    [string]$LocalZjuRoot = "F:\datasets\ZJU_MoCap\data\zju_mocap",
    [string]$Phase2ResultJson = "logs/modal_phase5/phase2_pixelsplat_prep_latest.json",
    [string]$OutRoot = "external/pixelsplat/datasets/zju_phase2_smoke",
    [int]$MaxScenes = 120,
    [double]$TrainRatio = 0.8,
    [int]$ChunkSize = 64,
    [int]$CheckMaxExamples = 20
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 20
    $root = (Resolve-Path ".").Path
    $abs = Join-Path $root $Path
    [System.IO.File]::WriteAllText($abs, $json, $enc)
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$resultNow = "logs/modal_phase5/pixelsplat_dataset_smoke_$ts.json"
$resultLatest = "logs/modal_phase5/pixelsplat_dataset_smoke_latest.json"
$resultMdLatest = "logs/modal_phase5/pixelsplat_dataset_smoke_latest.md"

$status = "ok"
$reason = ""
$manifestPath = ""
$localConvertedRoot = ""
$convertSummary = $null
$checkSummary = $null

try {
    if (-not (Test-Path $Phase2ResultJson)) {
        throw "phase2 result json not found: $Phase2ResultJson"
    }
    $phase2 = Get-Content $Phase2ResultJson -Raw | ConvertFrom-Json

    if ($phase2.remote -and $phase2.remote.manifest_local) {
        $manifestPath = [string]$phase2.remote.manifest_local
        if ($phase2.remote.local_download_root) {
            $candConverted = Join-Path ([string]$phase2.remote.local_download_root) "converted"
            if (Test-Path $candConverted) {
                $localConvertedRoot = $candConverted
            }
        }
    }
    if ([string]::IsNullOrWhiteSpace($manifestPath) -and $phase2.run_mode -eq "local") {
        $cand = Join-Path ([string]$phase2.out_root) "converted\pixelsplat_manifest.json"
        if (Test-Path $cand) { $manifestPath = $cand }
    }
    if ([string]::IsNullOrWhiteSpace($manifestPath) -or (-not (Test-Path $manifestPath))) {
        throw "manifest path not found in phase2 result (manifest_local missing or file absent)"
    }

    $outAbs = (Resolve-Path ".").Path | ForEach-Object { Join-Path $_ $OutRoot }
    New-Item -ItemType Directory -Path $outAbs -Force | Out-Null

    $cmd1 = @(
        "python", "tools/convert_pixelsplat_manifest_to_re10k_chunk.py",
        "--manifest", $manifestPath,
        "--local_zju_root", $LocalZjuRoot,
        "--out_root", $outAbs,
        "--max_scenes", [string]$MaxScenes,
        "--train_ratio", [string]$TrainRatio,
        "--chunk_size", [string]$ChunkSize
    )
    if (-not [string]::IsNullOrWhiteSpace($localConvertedRoot)) {
        $cmd1 += @("--local_converted_root", $localConvertedRoot)
    }
    $convertOut = & $cmd1[0] $cmd1[1..($cmd1.Length-1)]
    if ($LASTEXITCODE -ne 0) {
        throw "convert_pixelsplat_manifest_to_re10k_chunk failed"
    }
    $convertSummaryPath = Join-Path $outAbs "conversion_summary.json"
    if (Test-Path $convertSummaryPath) {
        $convertSummary = Get-Content $convertSummaryPath -Raw | ConvertFrom-Json
    }

    $cmd2 = @(
        "python", "tools/check_pixelsplat_re10k_chunk.py",
        "--dataset_root", $outAbs,
        "--max_examples", [string]$CheckMaxExamples
    )
    $checkOut = & $cmd2[0] $cmd2[1..($cmd2.Length-1)]
    if ($LASTEXITCODE -ne 0) {
        throw "check_pixelsplat_re10k_chunk failed"
    }
    $checkSummary = ($checkOut -join "`n") | ConvertFrom-Json
} catch {
    $status = "error"
    $reason = "$_"
}

$result = [ordered]@{
    timestamp = $ts
    status = $status
    reason = $reason
    code_dir = $CodeDir
    local_zju_root = $LocalZjuRoot
    phase2_result_json = $Phase2ResultJson
    manifest_path = $manifestPath
    local_converted_root = $localConvertedRoot
    out_root = $OutRoot
    convert_summary = $convertSummary
    check_summary = $checkSummary
}
Write-JsonNoBom -Path $resultNow -Obj $result
Write-JsonNoBom -Path $resultLatest -Obj $result

$md = @()
$md += "# PixelSplat Dataset Smoke"
$md += ""
$md += "- timestamp: $ts"
$md += "- status: $status"
$md += "- reason: $reason"
$md += "- manifest_path: $manifestPath"
$md += "- local_converted_root: $localConvertedRoot"
$md += "- out_root: $OutRoot"
if ($convertSummary) {
    $md += "- total_examples: $($convertSummary.num_examples_total)"
    $md += "- train_examples: $($convertSummary.train.num_examples)"
    $md += "- test_examples: $($convertSummary.test.num_examples)"
}
if ($checkSummary) {
    $md += "- check_ok: $($checkSummary.ok)"
    $md += "- check_train_entries: $($checkSummary.train.num_index_entries)"
    $md += "- check_test_entries: $($checkSummary.test.num_index_entries)"
}
Set-Content -Path $resultMdLatest -Value ($md -join "`n") -Encoding UTF8

Write-Host "[pixelsplat-smoke] wrote: $resultLatest"
Write-Host "[pixelsplat-smoke] wrote: $resultMdLatest"
if ($status -ne "ok") {
    exit 2
}
exit 0
