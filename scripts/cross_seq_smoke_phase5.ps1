param(
    [string]$CodeDir = "F:\vggt",
    [string]$GeomSubdir = "vggt_geom_ft_20260208_044454",
    [string]$ExcludeSeq = "CoreView_390",
    [int]$NumSeq = 2,
    [int]$NumSamples = 30,
    [double]$MinPSNR = 19.5,
    [double]$MinSSIM = 0.82,
    [double]$MaxWL1 = 0.085,
    [string]$SeqNames = ""
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

function Split-Seqs([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) {
        return @()
    }
    return @(
        ($Raw -split '[,\s]+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { $_.Trim() })
    )
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

function Discover-SeqsFromVolume([string]$geomSubdir) {
    $out = @()
    try {
        $jsonText = modal volume ls --json vggt-zju-data /zju_mocap | Out-String
        $items = $jsonText | ConvertFrom-Json
        $set = New-Object System.Collections.Generic.HashSet[string]
        foreach ($it in $items) {
            $name = [string]$it.Filename
            if ($name -match "^zju_mocap/(CoreView_[^/]+)/$([regex]::Escape($geomSubdir))/") {
                $null = $set.Add($Matches[1])
            }
        }
        $out = @($set.ToArray() | Sort-Object)
    } catch {
        $out = @()
    }
    return $out
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$csvPath = "logs/modal_phase5/cross_seq_smoke_$timestamp.csv"
$csvLatestPath = "logs/modal_phase5/cross_seq_smoke_latest.csv"
$jsonPath = "logs/modal_phase5/cross_seq_smoke_$timestamp.json"
$jsonLatestPath = "logs/modal_phase5/cross_seq_smoke_latest.json"

$manualSeqs = Split-Seqs -Raw $SeqNames
if ($manualSeqs.Count -gt 0) {
    $discovered = @($manualSeqs)
} else {
    Write-Host "[smoke] discovering sequences from data volume..."
    $discovered = @(Discover-SeqsFromVolume -geomSubdir $GeomSubdir)
}

$selected = @(
    $discovered | Where-Object { $_ -ne $ExcludeSeq } | Sort-Object | Select-Object -First ([int][Math]::Max(0, $NumSeq))
)

$rows = @()

if ($selected.Count -eq 0) {
    $rows += [pscustomobject]@{
        seq = ""
        status = "skipped"
        pass = $false
        N = ""
        mean_PSNR = ""
        mean_SSIM = ""
        mean_weighted_L1 = ""
        run_url = ""
        infer_out_dir = ""
        infer_out_volume_path = ""
        reason = "no sequence with geom_subdir='$GeomSubdir' found (excluding $ExcludeSeq)"
    }
} else {
    foreach ($seq in $selected) {
        Write-Host "[smoke] running seq=$seq"
        $outDir = "/mnt/out/infer_viewdec/$($seq)_phase5_cross_smoke_$timestamp"
        $outVolPath = To-VolumePath -OutDir $outDir
        $runLog = "logs/modal_phase5/cross_seq_smoke_${seq}_$timestamp.log"
        $metricsPath = "logs/modal_phase5/cross_seq_smoke_${seq}_$timestamp.metrics.json"

        $env:VGGT_CODE_DIR = $CodeDir
        $env:VGGT_PROFILE = "phase5_final"
        $env:VGGT_SEQ_NAMES = $seq
        $env:VGGT_GEOM_SUBDIR = $GeomSubdir
        $env:VGGT_INFER_OUT_DIR = $outDir
        $env:VGGT_INFER_NUM_SAMPLES = [string]$NumSamples
        $env:VGGT_INFER_SPLIT = "val"

        $run = Invoke-ModalRun -ScriptPath "modal_run_train.py"
        $modalOutput = @($run.Output)
        $runExitCode = [int]$run.ExitCode
        $modalOutput | Tee-Object -FilePath $runLog | Out-Null
        $runUrl = Get-ModalRunUrl -Lines $modalOutput

        if ($runExitCode -ne 0) {
            $rows += [pscustomobject]@{
                seq = $seq
                status = "error"
                pass = $false
                N = ""
                mean_PSNR = ""
                mean_SSIM = ""
                mean_weighted_L1 = ""
                run_url = $runUrl
                infer_out_dir = $outDir
                reason = "modal run failed"
            }
            continue
        }

        try {
            modal volume get vggt-out "$outVolPath/metrics_val.json" $metricsPath | Out-Null
            $m = Get-Content $metricsPath -Raw | ConvertFrom-Json
            $n = [int]$m.N
            $psnr = [double]$m.mean_PSNR
            $ssim = [double]$m.mean_SSIM
            $wl1 = [double]$m.mean_weighted_L1
            $pass = ($psnr -ge $MinPSNR) -and ($ssim -ge $MinSSIM) -and ($wl1 -le $MaxWL1)
            $reasonText = ""
            if (-not $pass) {
                $reasonText = "below threshold"
            }
            $rows += [pscustomobject]@{
                seq = $seq
                status = "ok"
                pass = [bool]$pass
                N = $n
                mean_PSNR = $psnr
                mean_SSIM = $ssim
                mean_weighted_L1 = $wl1
                run_url = $runUrl
                infer_out_dir = $outDir
                infer_out_volume_path = $outVolPath
                reason = $reasonText
            }
        } catch {
            $rows += [pscustomobject]@{
                seq = $seq
                status = "error"
                pass = $false
                N = ""
                mean_PSNR = ""
                mean_SSIM = ""
                mean_weighted_L1 = ""
                run_url = $runUrl
                infer_out_dir = $outDir
                infer_out_volume_path = $outVolPath
                reason = "failed to fetch/parse metrics"
            }
        }
    }
}

$rows | Export-Csv $csvPath -NoTypeInformation -Encoding UTF8
$rows | Export-Csv $csvLatestPath -NoTypeInformation -Encoding UTF8

$summary = [ordered]@{
    timestamp = $timestamp
    geom_subdir = $GeomSubdir
    exclude_seq = $ExcludeSeq
    num_requested = [int]$NumSeq
    num_samples = [int]$NumSamples
    thresholds = [ordered]@{
        min_psnr = [double]$MinPSNR
        min_ssim = [double]$MinSSIM
        max_wl1 = [double]$MaxWL1
    }
    discovered = $discovered
    selected = $selected
    rows = $rows
}
Write-JsonNoBom -Path $jsonPath -Obj $summary
Write-JsonNoBom -Path $jsonLatestPath -Obj $summary

$failed = @($rows | Where-Object { $_.status -eq "error" -or ($_.status -eq "ok" -and -not [bool]$_.pass) }).Count
$skippedOnly = @($rows | Where-Object { $_.status -eq "skipped" }).Count -eq $rows.Count

Write-Host "[smoke] wrote: $csvLatestPath"
if ($skippedOnly) {
    Write-Host "[smoke] skipped: no eligible sequences found."
    exit 0
}
if ($failed -gt 0) {
    exit 4
}
exit 0
