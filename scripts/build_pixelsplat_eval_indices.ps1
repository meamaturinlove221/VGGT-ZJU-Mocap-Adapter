param(
    [string]$DatasetRoot = "external/pixelsplat/datasets/zju_phase2_300_6v",
    [string]$Stage = "test",
    [string]$OutUniform = "external/pixelsplat/assets/evaluation_index_zju_uniform_6v.json",
    [string]$OutRandom = "external/pixelsplat/assets/evaluation_index_zju_random_6v.json",
    [int]$NumContextViews = 2,
    [int]$Seed = 2026,
    [int]$MaxScenes = 0
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
$resultNow = "logs/modal_phase5/pixelsplat_eval_index_$ts.json"
$resultLatest = "logs/modal_phase5/pixelsplat_eval_index_latest.json"
$resultMdLatest = "logs/modal_phase5/pixelsplat_eval_index_latest.md"

$status = "ok"
$reason = ""
$summary = $null

try {
    $cmd = @(
        "python", "tools/build_pixelsplat_eval_index.py",
        "--dataset_root", $DatasetRoot,
        "--stage", $Stage,
        "--out_uniform", $OutUniform,
        "--out_random", $OutRandom,
        "--num_context_views", [string]$NumContextViews,
        "--seed", [string]$Seed,
        "--max_scenes", [string]$MaxScenes
    )
    $out = & $cmd[0] $cmd[1..($cmd.Length-1)]
    if ($LASTEXITCODE -ne 0) {
        throw "build_pixelsplat_eval_index failed"
    }
    $summary = ($out -join "`n") | ConvertFrom-Json
} catch {
    $status = "error"
    $reason = "$_"
}

$result = [ordered]@{
    timestamp = $ts
    status = $status
    reason = $reason
    dataset_root = $DatasetRoot
    stage = $Stage
    out_uniform = $OutUniform
    out_random = $OutRandom
    num_context_views = $NumContextViews
    seed = $Seed
    max_scenes = $MaxScenes
    summary = $summary
}
Write-JsonNoBom -Path $resultNow -Obj $result
Write-JsonNoBom -Path $resultLatest -Obj $result

$md = @()
$md += "# PixelSplat Eval Index"
$md += ""
$md += "- timestamp: $ts"
$md += "- status: $status"
$md += "- reason: $reason"
$md += "- dataset_root: $DatasetRoot"
$md += "- stage: $Stage"
$md += "- out_uniform: $OutUniform"
$md += "- out_random: $OutRandom"
if ($summary) {
    $md += "- num_scenes_uniform: $($summary.num_scenes_uniform)"
    $md += "- num_scenes_random: $($summary.num_scenes_random)"
}
Set-Content -Path $resultMdLatest -Value ($md -join "`n") -Encoding UTF8

Write-Host "[pixelsplat-index] wrote: $resultLatest"
Write-Host "[pixelsplat-index] wrote: $resultMdLatest"
if ($status -ne "ok") {
    exit 2
}
exit 0
