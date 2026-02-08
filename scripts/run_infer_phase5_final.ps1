param(
    [string]$CodeDir = "F:\vggt",
    [string]$InferOutDir = ""
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$env:VGGT_CODE_DIR = $CodeDir
$env:VGGT_PROFILE = "phase5_final"

if ([string]::IsNullOrWhiteSpace($InferOutDir)) {
    Remove-Item Env:VGGT_INFER_OUT_DIR -ErrorAction SilentlyContinue
} else {
    $env:VGGT_INFER_OUT_DIR = $InferOutDir
}

modal run modal_run_train.py
