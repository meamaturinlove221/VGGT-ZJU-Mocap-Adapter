param(
    [string]$CodeDir = "F:\vggt",
    [string]$InferOutDir = "",
    [double]$MinPSNR = 21.15,
    [double]$MinSSIM = 0.860,
    [double]$MaxWL1 = 0.061,
    [int]$ExpectN = 118,
    [string]$ExpectedFgKey = "tgt_fg",
    [switch]$SkipRun
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 16
    $root = (Resolve-Path ".").Path
    $abs = Join-Path $root $Path
    [System.IO.File]::WriteAllText($abs, $json, $enc)
}

function To-VolumePath([string]$OutDir) {
    $s = [string]$OutDir
    if ($s.StartsWith("/mnt/out/")) {
        return "/" + $s.Substring("/mnt/out/".Length)
    }
    if ($s.StartsWith("mnt/out/")) {
        return "/" + $s.Substring("mnt/out/".Length)
    }
    if ($s.StartsWith("/")) {
        return $s
    }
    return "/" + $s
}

function Get-ModalRunUrl([string[]]$Lines) {
    $url = ""
    foreach ($line in $Lines) {
        if ($line -match "https://modal\.com/apps/\S+") {
            $url = $Matches[0]
        }
    }
    return $url
}

function Invoke-ModalRun([string]$ScriptPath = "modal_run_train.py") {
    $stdoutFile = [System.IO.Path]::GetTempFileName()
    $stderrFile = [System.IO.Path]::GetTempFileName()
    try {
        $proc = Start-Process `
            -FilePath "cmd.exe" `
            -ArgumentList @("/c", "modal run $ScriptPath") `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutFile `
            -RedirectStandardError $stderrFile
        $output = @()
        if (Test-Path $stdoutFile) { $output += @(Get-Content $stdoutFile) }
        if (Test-Path $stderrFile) { $output += @(Get-Content $stderrFile) }
        return [pscustomobject]@{
            Output = $output
            ExitCode = [int]$proc.ExitCode
        }
    } finally {
        Remove-Item $stdoutFile -ErrorAction SilentlyContinue
        Remove-Item $stderrFile -ErrorAction SilentlyContinue
    }
}

function Get-FirstOverlayJsonRemotePath([string]$RemoteOutDir) {
    $itemsJson = modal volume ls --json vggt-out $RemoteOutDir | Out-String
    $items = $itemsJson | ConvertFrom-Json
    $filtered = $items | Where-Object { $_.Type -eq "file" -and $_.Filename -match "gt_with_fg_overlay_step000000\.json$" } | Select-Object -ExpandProperty Filename
    $cand = @($filtered)
    if ($cand.Count -eq 0) {
        return ""
    }
    return "/" + [string]$cand[0]
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($InferOutDir)) {
    $InferOutDir = "/mnt/out/infer_viewdec/CoreView_390_phase5_final_gate_$timestamp"
}
$InferOutVolPath = To-VolumePath -OutDir $InferOutDir

$logPath = "logs/modal_phase5/release_gate_modal_$timestamp.log"
$metricsPath = "logs/modal_phase5/release_gate_metrics_$timestamp.json"
$overlayPath = "logs/modal_phase5/release_gate_overlay_$timestamp.json"
$resultPath = "logs/modal_phase5/release_gate_result_$timestamp.json"
$resultLatestPath = "logs/modal_phase5/release_gate_result_latest.json"

$modalOutput = @()
$runUrl = ""
$runExitCode = 0

if (-not $SkipRun) {
    $env:VGGT_CODE_DIR = $CodeDir
    $env:VGGT_PROFILE = "phase5_final"
    $env:VGGT_INFER_OUT_DIR = $InferOutDir
    $env:VGGT_INFER_NUM_SAMPLES = "-1"
    $env:VGGT_INFER_SPLIT = "val"

    Write-Host "[gate] running modal infer..."
    $run = Invoke-ModalRun -ScriptPath "modal_run_train.py"
    $modalOutput = @($run.Output)
    $runExitCode = [int]$run.ExitCode
    $modalOutput | Tee-Object -FilePath $logPath | Out-Null
    $runUrl = Get-ModalRunUrl -Lines $modalOutput
} else {
    Write-Host "[gate] SkipRun enabled; only validating existing out dir: $InferOutDir"
}

if (($runExitCode -ne 0) -and (-not $SkipRun)) {
    $fail = [ordered]@{
        pass = $false
        stage = "modal_run"
        reason = "modal run failed"
        run_exit_code = $runExitCode
        run_url = $runUrl
        infer_out_dir = $InferOutDir
        log_path = $logPath
        timestamp = $timestamp
    }
    Write-JsonNoBom -Path $resultPath -Obj $fail
    Write-JsonNoBom -Path $resultLatestPath -Obj $fail
    Write-Error "[gate] modal run failed."
    exit 2
}

Write-Host "[gate] downloading metrics..."
modal volume get vggt-out "$InferOutVolPath/metrics_val.json" $metricsPath | Out-Null
$overlayRemote = Get-FirstOverlayJsonRemotePath -RemoteOutDir $InferOutVolPath
if (-not [string]::IsNullOrWhiteSpace($overlayRemote)) {
    modal volume get vggt-out $overlayRemote $overlayPath | Out-Null
}

$metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
$overlay = $null
if (Test-Path $overlayPath) {
    $overlay = Get-Content $overlayPath -Raw | ConvertFrom-Json
}

$n = [int]$metrics.N
$psnr = [double]$metrics.mean_PSNR
$ssim = [double]$metrics.mean_SSIM
$wl1 = [double]$metrics.mean_weighted_L1
$fgKey = if ($overlay -ne $null) { [string]$overlay.source_fg_key } else { "" }
$maskPath = if ($overlay -ne $null) { [string]$overlay.tgt_mask_path } else { "" }
$overlayApplied = if ($overlay -ne $null) { [bool]$overlay.overlay_applied } else { $false }

$checks = [ordered]@{
    n_eq_expect = ($n -eq [int]$ExpectN)
    psnr_ge_threshold = ($psnr -ge [double]$MinPSNR)
    ssim_ge_threshold = ($ssim -ge [double]$MinSSIM)
    wl1_le_threshold = ($wl1 -le [double]$MaxWL1)
    source_fg_key_ok = ($fgKey -eq $ExpectedFgKey)
    tgt_mask_path_present = (-not [string]::IsNullOrWhiteSpace($maskPath))
    overlay_present = $overlayApplied
}

$pass = $true
foreach ($k in $checks.Keys) {
    if (-not [bool]$checks[$k]) {
        $pass = $false
        break
    }
}

$overlayPathOut = ""
if (Test-Path $overlayPath) {
    $overlayPathOut = $overlayPath
}

$result = [ordered]@{
    pass = $pass
    thresholds = [ordered]@{
        min_psnr = [double]$MinPSNR
        min_ssim = [double]$MinSSIM
        max_wl1 = [double]$MaxWL1
        expect_n = [int]$ExpectN
        expected_source_fg_key = $ExpectedFgKey
    }
    values = [ordered]@{
        N = $n
        mean_PSNR = $psnr
        mean_SSIM = $ssim
        mean_weighted_L1 = $wl1
        source_fg_key = $fgKey
        tgt_mask_path = $maskPath
        overlay_applied = $overlayApplied
    }
    checks = $checks
    artifacts = [ordered]@{
        infer_out_dir = $InferOutDir
        infer_out_volume_path = $InferOutVolPath
        run_url = $runUrl
        modal_log = $logPath
        metrics_path = $metricsPath
        overlay_path = $overlayPathOut
    }
    timestamp = $timestamp
}

Write-JsonNoBom -Path $resultPath -Obj $result
Write-JsonNoBom -Path $resultLatestPath -Obj $result

Write-Host "[gate] result: pass=$pass PSNR=$psnr SSIM=$ssim wL1=$wl1 N=$n"
Write-Host "[gate] wrote: $resultLatestPath"

if (-not $pass) {
    exit 3
}

exit 0
