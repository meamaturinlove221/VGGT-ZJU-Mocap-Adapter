param(
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$CamNames = "Camera_B1,Camera_B2,Camera_B3,Camera_B4,Camera_B5,Camera_B6,Camera_B7,Camera_B8,Camera_B9,Camera_B10,Camera_B11,Camera_B12,Camera_B13,Camera_B14,Camera_B15,Camera_B16,Camera_B17,Camera_B18,Camera_B19,Camera_B20,Camera_B21,Camera_B22,Camera_B23",
    [string]$GeomCandidates = "vggt_geom:baseline_orig;vggt_geom_ft_20260208_044454:baseline_ft",
    [string]$DecoderCkpt = "",
    [string]$Split = "val",
    [int]$NumSamples = -1,
    [double]$MinPSNR = 20.9,
    [double]$MinSSIM = 0.84,
    [double]$MaxWL1 = 0.08,
    [string]$OutTag = "baseline_compare",
    [string]$InferArgsExtra = "",
    [int]$MetricsWaitTimeoutSec = 7200,
    [string]$DownloadVisSteps = "",
    [int]$DownloadVisCount = 3,
    [switch]$SkipRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
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

function Get-ModalRunFailureReason([string[]]$Lines, [int]$ExitCode) {
    $joined = (@($Lines) -join "`n")
    if ($joined -match "OutOfMemoryError" -or $joined -match "CUDA out of memory") {
        return "infer modal run failed: cuda_oom"
    }
    if ($joined -match "metrics not ready before timeout") {
        return "infer modal run failed: metrics_timeout"
    }
    if ($joined -match "command failed \(rc=") {
        return "infer modal run failed: remote_command_failed(rc=$ExitCode)"
    }
    return "infer modal run failed: exit_code_$ExitCode"
}

function Invoke-ModalRun(
    [string]$ScriptPath = "modal_run_train.py",
    [int]$MaxRetries = 3,
    [int]$RetrySleepSec = 10
) {
    $attempt = 0
    $last = $null
    while ($attempt -lt [Math]::Max(1, $MaxRetries)) {
        $attempt += 1
        $stdoutFile = [System.IO.Path]::GetTempFileName()
        $stderrFile = [System.IO.Path]::GetTempFileName()
        try {
            $proc = Start-Process `
                -FilePath "cmd.exe" `
                -ArgumentList @("/c", "modal run -q $ScriptPath") `
                -NoNewWindow `
                -Wait `
                -PassThru `
                -RedirectStandardOutput $stdoutFile `
                -RedirectStandardError $stderrFile
            $output = @()
            if (Test-Path $stdoutFile) { $output += @(Get-Content $stdoutFile) }
            if (Test-Path $stderrFile) { $output += @(Get-Content $stderrFile) }
            $rc = [int]$proc.ExitCode
            $last = [pscustomobject]@{
                Output = $output
                ExitCode = $rc
                Attempt = $attempt
            }
            if ($rc -eq 0) {
                return $last
            }

            $blob = ($output -join "`n")
            $isTransient = (
                ($blob -match "Connection lost") -or
                ($blob -match "WinError 10053") -or
                ($blob -match "WinError 10054") -or
                ($blob -match "SSL shutdown timed out") -or
                ($blob -match "Deadline exceeded") -or
                ($blob -match "heartbeat failed") -or
                ($blob -match "modal\.exception\.ConnectionError") -or
                ($blob -match "timed out waiting for final app logs") -or
                ($blob -match "Could not connect to the Modal server") -or
                ($blob -match "Cannot connect to host") -or
                ($blob -match "cloudflarestorage.com") -or
                ($blob -match "FETCH_HEAD was modified during build process") -or
                ($blob -match "\.git/HEAD was modified during build process") -or
                ($blob -match "was modified during build process")
            )
            if ($isTransient -and $attempt -lt $MaxRetries) {
                Write-Host "[modal-run] transient failure attempt=$attempt/$MaxRetries, retry in ${RetrySleepSec}s"
                Start-Sleep -Seconds $RetrySleepSec
                continue
            }
            return $last
        } finally {
            Remove-Item $stdoutFile -ErrorAction SilentlyContinue
            Remove-Item $stderrFile -ErrorAction SilentlyContinue
        }
    }
    return $last
}

function Wait-FetchMetrics(
    [string]$OutVolPath,
    [string]$Split,
    [string]$LocalMetricsPath,
    [int]$TimeoutSec = 7200,
    [int]$PollSec = 20
) {
    $deadline = (Get-Date).AddSeconds([Math]::Max(30, [int]$TimeoutSec))
    while ((Get-Date) -lt $deadline) {
        try {
            modal volume get vggt-out "$OutVolPath/metrics_$Split.json" $LocalMetricsPath 2>$null | Out-Null
            if (Test-Path $LocalMetricsPath) {
                return $true
            }
        } catch {
            # Not ready yet or transient error; keep polling.
        }
        Start-Sleep -Seconds ([Math]::Max(3, [int]$PollSec))
    }
    return $false
}

function Sanitize([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) { return "item" }
    return ([regex]::Replace($Raw, "[^A-Za-z0-9_.-]+", "_")).Trim("_")
}

function Parse-Candidates([string]$Raw) {
    $rows = @()
    $parts = @($Raw -split ";" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    foreach ($p in $parts) {
        $x = $p.Trim()
        $geom = ""
        $label = ""
        if ($x -match "^([^:]+):(.+)$") {
            $geom = $Matches[1].Trim()
            $label = $Matches[2].Trim()
        } else {
            $geom = $x
            $label = $x
        }
        if (-not [string]::IsNullOrWhiteSpace($geom)) {
            $rows += [pscustomobject]@{
                geom = $geom
                label = $label
            }
        }
    }
    return @($rows)
}

function Parse-StepList([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) {
        return @()
    }
    $out = New-Object System.Collections.Generic.List[int]
    foreach ($tok in ($Raw -split "[,\s;|]+" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
        $v = 0
        if ([int]::TryParse($tok.Trim(), [ref]$v)) {
            if ($v -ge 0) {
                $out.Add($v) | Out-Null
            }
        }
    }
    return @($out | Select-Object -Unique)
}

function Get-StepFromFilename([string]$Filename) {
    $m = [regex]::Match([string]$Filename, "step(\d+)\.png$")
    if ($m.Success) {
        return [int]$m.Groups[1].Value
    }
    return -1
}

function Get-VolumeFileNames([object[]]$Items) {
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($it in @($Items)) {
        if ($null -eq $it) { continue }
        $obj = $it
        $type = ""
        try {
            if ($obj.PSObject.Properties["Type"]) {
                $type = [string]$obj.Type
            }
        } catch {}
        if (-not [string]::IsNullOrWhiteSpace($type) -and ($type -ne "file")) {
            continue
        }
        $name = ""
        foreach ($k in @("Filename", "filename", "Path", "path", "Name", "name")) {
            try {
                if ($obj.PSObject.Properties[$k]) {
                    $name = [string]$obj.$k
                    if (-not [string]::IsNullOrWhiteSpace($name)) { break }
                }
            } catch {}
        }
        if ([string]::IsNullOrWhiteSpace($name)) {
            if ($obj -is [string]) { $name = [string]$obj }
        }
        if (-not [string]::IsNullOrWhiteSpace($name)) {
            $out.Add($name) | Out-Null
        }
    }
    return @($out | Select-Object -Unique)
}

function Parse-VolumeLsItems([string]$ItemsJson) {
    $parsed = $null
    try {
        $parsed = $ItemsJson | ConvertFrom-Json
    } catch {
        return @()
    }
    $flat = New-Object System.Collections.Generic.List[object]
    $queue = New-Object System.Collections.Generic.Queue[object]
    foreach ($x in @($parsed)) {
        $queue.Enqueue($x)
    }
    while ($queue.Count -gt 0) {
        $cur = $queue.Dequeue()
        if ($null -eq $cur) { continue }
        if ($cur -is [System.Array]) {
            foreach ($y in @($cur)) {
                $queue.Enqueue($y)
            }
            continue
        }
        $flat.Add($cur) | Out-Null
    }
    return @($flat.ToArray())
}

function Find-RemoteFiles([string]$RemoteDir, [string]$Pattern) {
    try {
        $itemsJson = modal volume ls --json vggt-out $RemoteDir | Out-String
        $items = @(Parse-VolumeLsItems -ItemsJson $itemsJson)
        $files = @(
            Get-VolumeFileNames -Items $items |
                Where-Object { [string]$_ -match $Pattern }
        )
        if ($files.Count -gt 0) {
            $rows = @()
            foreach ($f in $files) {
                $rows += [pscustomobject]@{
                    Filename = [string]$f
                    Step = Get-StepFromFilename -Filename ([string]$f)
                }
            }
            return @(
                $rows |
                    Sort-Object Step, Filename |
                    ForEach-Object { "/" + [string]$_.Filename }
            )
        }
    } catch {
    }
    return @()
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$seqTag = Sanitize($SeqNames.Replace(",", "_").Replace(" ", "_"))
$csvPath = "logs/modal_phase5/baseline_compare_$timestamp.csv"
$csvLatestPath = "logs/modal_phase5/baseline_compare_latest.csv"
$jsonPath = "logs/modal_phase5/vggt_ft_gate_$timestamp.json"
$jsonLatestPath = "logs/modal_phase5/vggt_ft_gate_latest.json"
$visSteps = @(Parse-StepList -Raw $DownloadVisSteps)
$visCount = [Math]::Max(1, [int]$DownloadVisCount)

$cands = Parse-Candidates -Raw $GeomCandidates
if ($cands.Count -eq 0) {
    throw "No valid GeomCandidates parsed from: $GeomCandidates"
}

$rows = @()
foreach ($cand in $cands) {
    $geom = [string]$cand.geom
    $label = [string]$cand.label
    $safeLabel = Sanitize($label)
    $outDir = "/mnt/out/infer_viewdec/$seqTag`_$($OutTag)_$safeLabel`_$timestamp"
    $outVolPath = To-VolumePath -OutDir $outDir
    $runUrl = ""

    if (-not $SkipRun) {
        $env:VGGT_CODE_DIR = $CodeDir
        $env:VGGT_MODE = "infer"
        $env:VGGT_SEQ_NAMES = $SeqNames
        $env:VGGT_CAM_NAMES = $CamNames
        $env:VGGT_GEOM_SUBDIR = $geom
        $env:VGGT_INFER_SPLIT = $Split
        $env:VGGT_INFER_NUM_SAMPLES = [string]$NumSamples
        $env:VGGT_INFER_OUT_DIR = $outDir
        $env:VGGT_INFER_USE_EMA = "0"
        if ([string]::IsNullOrWhiteSpace($InferArgsExtra)) {
            Remove-Item Env:VGGT_INFER_ARGS_EXTRA -ErrorAction SilentlyContinue
        } else {
            $env:VGGT_INFER_ARGS_EXTRA = [string]$InferArgsExtra
        }
        if ([string]::IsNullOrWhiteSpace($DecoderCkpt)) {
            Remove-Item Env:VGGT_INFER_CKPT -ErrorAction SilentlyContinue
        } else {
            $env:VGGT_INFER_CKPT = $DecoderCkpt
        }
        Remove-Item Env:VGGT_PROFILE -ErrorAction SilentlyContinue

        Write-Host "[eval] run infer label=$label geom=$geom"
        $run = Invoke-ModalRun -ScriptPath "modal_run_train.py"
        $modalOutput = @($run.Output)
        $rc = [int]$run.ExitCode
        $runLog = "logs/modal_phase5/eval_geom_${safeLabel}_$timestamp.log"
        $modalOutput | Tee-Object -FilePath $runLog | Out-Null
        $runUrl = Get-ModalRunUrl -Lines $modalOutput
        if ($rc -ne 0) {
            $reason = Get-ModalRunFailureReason -Lines $modalOutput -ExitCode $rc
            Write-Host "[eval] infer failed label=$label reason=$reason"
            $rows += [pscustomobject]@{
                label = $label
                geom_subdir = $geom
                status = "error"
                pass = $false
                N = ""
                mean_weighted_L1 = ""
                mean_PSNR = ""
                mean_SSIM = ""
                run_url = $runUrl
                infer_out_dir = $outDir
                infer_out_volume_path = $outVolPath
                reason = $reason
                candidate_invalid_reason = "eval_failed"
            }
            continue
        }
    }

    $metricsPath = "logs/modal_phase5/eval_geom_${safeLabel}_$timestamp.metrics.json"
    try {
        $waitSec = [Math]::Max(30, [int]$MetricsWaitTimeoutSec)
        $metricsReady = Wait-FetchMetrics `
            -OutVolPath $outVolPath `
            -Split $Split `
            -LocalMetricsPath $metricsPath `
            -TimeoutSec $waitSec `
            -PollSec 20
        if (-not $metricsReady) {
            throw "metrics not ready before timeout"
        }
        $m = Get-Content $metricsPath -Raw | ConvertFrom-Json
        $n = [int]$m.N
        $psnr = [double]$m.mean_PSNR
        $ssim = [double]$m.mean_SSIM
        $wl1 = [double]$m.mean_weighted_L1
        $hasSamples = ($n -gt 0)
        $ok = $hasSamples -and ($psnr -ge $MinPSNR) -and ($ssim -ge $MinSSIM) -and ($wl1 -le $MaxWL1)

        $visPatterns = @(
            [pscustomobject]@{ name = "cat_pred_tgt"; pattern = "cat_pred_tgt_step\d+\.png$" },
            [pscustomobject]@{ name = "cat_fg_mask_pred_tgt"; pattern = "cat_fg_mask_pred_tgt_step\d+\.png$" },
            [pscustomobject]@{ name = "gt_with_fg_overlay"; pattern = "gt_with_fg_overlay_step\d+\.png$" }
        )
        foreach ($vp in $visPatterns) {
            $allRemote = @(Find-RemoteFiles -RemoteDir $outVolPath -Pattern $vp.pattern)
            if ($allRemote.Count -le 0) {
                continue
            }

            $selected = @()
            if ($visSteps.Count -gt 0) {
                foreach ($want in $visSteps) {
                    $pick = $allRemote | Where-Object {
                        (Get-StepFromFilename -Filename ([System.IO.Path]::GetFileName([string]$_))) -eq [int]$want
                    } | Select-Object -First 1
                    if ($pick) {
                        $selected += @([string]$pick)
                    }
                }
            } else {
                $selected = @($allRemote | Select-Object -First $visCount)
            }

            foreach ($remoteFile in $selected) {
                if ([string]::IsNullOrWhiteSpace($remoteFile)) {
                    continue
                }
                $base = [System.IO.Path]::GetFileName([string]$remoteFile)
                $local = "logs/modal_phase5/baseline_${safeLabel}_${timestamp}_$base"
                modal volume get vggt-out ([string]$remoteFile) $local | Out-Null
            }
        }

        $rows += [pscustomobject]@{
            label = $label
            geom_subdir = $geom
            status = $(if ($hasSamples) { "ok" } else { "error" })
            pass = [bool]$ok
            N = $n
            mean_weighted_L1 = $wl1
            mean_PSNR = $psnr
            mean_SSIM = $ssim
            run_url = $runUrl
            infer_out_dir = $outDir
            infer_out_volume_path = $outVolPath
            reason = $(
                if (-not $hasSamples) {
                    "zero_samples(N=0)"
                } elseif ($ok) {
                    ""
                } else {
                    "below threshold"
                }
            )
            candidate_invalid_reason = $(
                if (-not $hasSamples) {
                    "eval_empty"
                } else {
                    ""
                }
            )
        }
    } catch {
        $rows += [pscustomobject]@{
            label = $label
            geom_subdir = $geom
            status = "error"
            pass = $false
            N = ""
            mean_weighted_L1 = ""
            mean_PSNR = ""
            mean_SSIM = ""
            run_url = $runUrl
            infer_out_dir = $outDir
            infer_out_volume_path = $outVolPath
            reason = "failed to fetch/parse metrics"
            candidate_invalid_reason = "eval_failed"
        }
    }
}

$rows | Export-Csv $csvPath -NoTypeInformation -Encoding UTF8
$rows | Export-Csv $csvLatestPath -NoTypeInformation -Encoding UTF8

$failed = @($rows | Where-Object { $_.status -ne "ok" -or (-not [bool]$_.pass) })
$summary = [ordered]@{
    timestamp = $timestamp
    seq_names = $SeqNames
    split = $Split
    num_samples = [int]$NumSamples
    thresholds = [ordered]@{
        min_psnr = [double]$MinPSNR
        min_ssim = [double]$MinSSIM
        max_wl1 = [double]$MaxWL1
    }
    pass = ($failed.Count -eq 0)
    failed_count = $failed.Count
    rows = $rows
    csv = $csvLatestPath
}
Write-JsonNoBom -Path $jsonPath -Obj $summary
Write-JsonNoBom -Path $jsonLatestPath -Obj $summary

Write-Host "[eval] wrote: $csvLatestPath"
Write-Host "[eval] wrote: $jsonLatestPath"
if ($failed.Count -gt 0) {
    exit 4
}
exit 0
