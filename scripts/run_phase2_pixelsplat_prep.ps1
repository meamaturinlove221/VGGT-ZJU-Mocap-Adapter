param(
    [string]$CodeDir = "F:\vggt",
    [string]$ZjuRoot = "F:\datasets\ZJU_MoCap\data\zju_mocap",
    [string]$SeqNames = "CoreView_390",
    [string]$GeomSubdir = "auto",
    [string]$PretrainedCkpt = "model.pt",
    [int]$FrameStride = 1,
    [int]$MaxFrames = 300,
    [int]$MaxViews = 6,
    [int]$MosaicNumSamples = 2,
    [int]$MosaicNumTargets = 3,
    [int]$MosaicNumSrcViews = 6,
    [int]$CameraSanityNumFrames = 10,
    [string]$OutRoot = "",
    [switch]$ForceRemote,
    [switch]$NoRemoteFallbackOnLocalError,
    [string]$RemoteOutPrefix = "/mnt/out/phase2_pixelsplat_prep",
    [int]$RemoteRunRetries = 3
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

function To-VolumePath([string]$OutPath) {
    $s = [string]$OutPath
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

function Invoke-ModalVolumeGet([string]$RemotePath, [string]$LocalDest, [switch]$Force) {
    if ($Force) {
        & modal volume get vggt-out $RemotePath $LocalDest --force | Out-Null
    } else {
        & modal volume get vggt-out $RemotePath $LocalDest | Out-Null
    }
    if ($LASTEXITCODE -ne 0) {
        throw "modal volume get failed: remote=$RemotePath local=$LocalDest"
    }
}

function Resolve-GeomSubdir([string]$Raw) {
    $v = [string]$Raw
    if (-not [string]::IsNullOrWhiteSpace($v) -and $v -ne "auto") {
        return $v.Trim()
    }
    $gatePath = "logs/modal_phase5/vggt_ft_gate_latest.json"
    if (Test-Path $gatePath) {
        try {
            $g = Get-Content $gatePath -Raw | ConvertFrom-Json
            if ($g.rows -and $g.rows.Count -gt 0) {
                $x = [string]$g.rows[0].geom_subdir
                if (-not [string]::IsNullOrWhiteSpace($x)) {
                    return $x.Trim()
                }
            }
        } catch {
        }
    }
    return "vggt_geom_ft_20260208_044454"
}

function First-SeqName([string]$SeqNamesText) {
    $x = @($SeqNamesText -split '[,\s]+' | Where-Object { $_ })
    if ($x.Count -eq 0) { return "" }
    return [string]$x[0]
}

function Resolve-GeomSubdirOnDisk([string]$Root, [string]$SeqNamesText, [string]$Preferred) {
    $seq = First-SeqName -SeqNamesText $SeqNamesText
    if ([string]::IsNullOrWhiteSpace($seq)) { return $Preferred }
    $seqDir = Join-Path $Root $seq
    if (-not (Test-Path $seqDir)) { return $Preferred }

    $prefDir = Join-Path $seqDir $Preferred
    if (Test-Path $prefDir) { return $Preferred }

    $candFt = @(
        Get-ChildItem -Path $seqDir -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "vggt_geom_ft*" } |
            Sort-Object LastWriteTime -Descending
    )
    if ($candFt.Count -gt 0) { return [string]$candFt[0].Name }

    $fallbacks = @("vggt_geom", "vggt_geom_4v_backup", "vggt_geom_test6")
    foreach ($f in $fallbacks) {
        if (Test-Path (Join-Path $seqDir $f)) { return $f }
    }
    return $Preferred
}

function First-Npz([string]$Root, [string]$Seq, [string]$Geom) {
    $d = Join-Path (Join-Path $Root $Seq) $Geom
    if (-not (Test-Path $d)) { return "" }
    $f = Get-ChildItem -Path $d -Filter "*.npz" -File | Sort-Object Name | Select-Object -First 1
    if ($null -eq $f) { return "" }
    return $f.FullName
}

function Run-LocalStep([string]$StepName, [string[]]$Cmd, [string]$OutPath) {
    & $Cmd[0] $Cmd[1..($Cmd.Length-1)]
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed"
    }
    return [pscustomobject]@{
        step = $StepName
        ok = $true
        out = $OutPath
        mode = "local"
    }
}

function Run-RemoteStep(
    [string]$StepName,
    [string]$SeqNamesText,
    [string]$GeomDirName,
    [int]$MaxFramesArg,
    [string]$ScriptRelPath,
    [string]$ArgsExtra,
    [string]$TsTag
) {
    $env:VGGT_CODE_DIR = $CodeDir
    $env:VGGT_MODE = "precompute"
    $env:VGGT_SEQ_NAMES = $SeqNamesText
    $env:VGGT_GEOM_SUBDIR = $GeomDirName
    $env:VGGT_MAX_FRAMES = [string]$MaxFramesArg
    $env:VGGT_PRECOMPUTE_SCRIPT = $ScriptRelPath
    $env:VGGT_PRECOMPUTE_CKPT = $PretrainedCkpt
    $env:VGGT_PRECOMPUTE_ARGS_EXTRA = $ArgsExtra
    Remove-Item Env:VGGT_PROFILE -ErrorAction SilentlyContinue

    $safeStep = (($StepName -replace "[^A-Za-z0-9_.-]+", "_").Trim("_"))
    if ([string]::IsNullOrWhiteSpace($safeStep)) { $safeStep = "step" }
    $lastLogPath = ""
    $maxTry = [Math]::Max(1, [int]$RemoteRunRetries)
    for ($attempt = 1; $attempt -le $maxTry; $attempt++) {
        $run = Invoke-ModalRun -ScriptPath "modal_run_train.py"
        $logPath = "logs/modal_phase5/phase2_${safeStep}_$TsTag.remote.attempt$attempt.log"
        @($run.Output) | Tee-Object -FilePath $logPath | Out-Null
        $lastLogPath = $logPath
        if ([int]$run.ExitCode -eq 0) {
            break
        }
        $joined = (@($run.Output) -join "`n")
        $isBuildRace = $joined -match "modified during build process"
        if ($isBuildRace -and $attempt -lt $maxTry) {
            Start-Sleep -Seconds (5 * $attempt)
            continue
        }
        throw "$StepName remote modal run failed (rc=$($run.ExitCode), attempt=$attempt)"
    }

    return [pscustomobject]@{
        step = $StepName
        ok = $true
        mode = "remote"
        log = $lastLogPath
    }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$geom0 = Resolve-GeomSubdir -Raw $GeomSubdir
$geomLocal = Resolve-GeomSubdirOnDisk -Root $ZjuRoot -SeqNamesText $SeqNames -Preferred $geom0
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
    $OutRoot = "logs/phase2_pixelsplat_prep/$ts"
}

$outAbs = Resolve-Path "." | ForEach-Object { Join-Path $_ $OutRoot }
New-Item -ItemType Directory -Path $outAbs -Force | Out-Null
$convOut = Join-Path $outAbs "converted"
$sanityOut = Join-Path $outAbs "camera_sanity"
$mosaicOut = Join-Path $outAbs "mosaic"
$selectOutJson = Join-Path $outAbs "uniform_yaw_pick.json"
$resultJson = "logs/modal_phase5/phase2_pixelsplat_prep_$ts.json"
$resultJsonLatest = "logs/modal_phase5/phase2_pixelsplat_prep_latest.json"
$resultMdLatest = "logs/modal_phase5/phase2_pixelsplat_prep_latest.md"

$steps = @()
$status = "ok"
$reason = ""
$runMode = "local"
$remote = $null
$localFirstSeq = First-SeqName -SeqNamesText $SeqNames
$localFirstNpz = ""
if (-not [string]::IsNullOrWhiteSpace($localFirstSeq)) {
    $localFirstNpz = First-Npz -Root $ZjuRoot -Seq $localFirstSeq -Geom $geomLocal
}
$useRemote = [bool]$ForceRemote -or [string]::IsNullOrWhiteSpace($localFirstNpz)
$geom = $(if ($useRemote) { $geom0 } else { $geomLocal })
if ([string]::IsNullOrWhiteSpace($geom)) { $geom = $geom0 }

function Try-LocalPrep() {
    $localSteps = @()
    $cmd1 = @(
        "python", "tools/convert_to_pixelsplat_format.py",
        "--zju_root", $ZjuRoot,
        "--seq_names", $SeqNames,
        "--geom_subdir", $geom,
        "--out_root", $convOut,
        "--frame_stride", [string]$FrameStride,
        "--max_frames", [string]$MaxFrames,
        "--max_views", [string]$MaxViews,
        "--copy_mode", "none"
    )
    $localSteps += Run-LocalStep -StepName "convert_to_pixelsplat_format" -Cmd $cmd1 -OutPath $convOut

    $cmd2 = @(
        "python", "camera_sanity_check.py",
        "--zju_root", $ZjuRoot,
        "--seq_names", $SeqNames,
        "--geom_subdir", $geom,
        "--out", $sanityOut,
        "--num_frames", [string]$CameraSanityNumFrames
    )
    $localSteps += Run-LocalStep -StepName "camera_sanity_check" -Cmd $cmd2 -OutPath $sanityOut

    $cmd3 = @(
        "python", "inspect_batch_mosaic.py",
        "--zju_root", $ZjuRoot,
        "--seq_names", $SeqNames,
        "--geom_subdir", $geom,
        "--out", $mosaicOut,
        "--num_samples", [string]$MosaicNumSamples,
        "--num_targets", [string]$MosaicNumTargets,
        "--num_src_views", [string]$MosaicNumSrcViews
    )
    $localSteps += Run-LocalStep -StepName "inspect_batch_mosaic" -Cmd $cmd3 -OutPath $mosaicOut

    $seq = First-SeqName -SeqNamesText $SeqNames
    $firstNpz = First-Npz -Root $ZjuRoot -Seq $seq -Geom $geom
    if (-not [string]::IsNullOrWhiteSpace($firstNpz)) {
        $cmd4 = @(
            "python", "select_views_uniform_yaw.py",
            "--npz", $firstNpz,
            "--zju_root", $ZjuRoot,
            "--num_src_views", [string]$MosaicNumSrcViews,
            "--dump_json", $selectOutJson
        )
        $localSteps += Run-LocalStep -StepName "select_views_uniform_yaw_demo" -Cmd $cmd4 -OutPath $selectOutJson
    } else {
        $localSteps += [pscustomobject]@{
            step = "select_views_uniform_yaw_demo"
            ok = $false
            reason = "no npz found"
            mode = "local"
        }
    }
    return @($localSteps)
}

function Run-RemotePrep() {
    $remoteSteps = @()
    $remoteTag = "phase2_pixelsplat_prep_$ts"
    $remoteOutRoot = "$RemoteOutPrefix/$remoteTag"
    $remoteConv = "$remoteOutRoot/converted"
    $remoteSanity = "$remoteOutRoot/camera_sanity"
    $remoteMosaic = "$remoteOutRoot/mosaic"
    $remoteSelectJson = "$remoteOutRoot/uniform_yaw_pick.json"
    $remoteVolRoot = To-VolumePath -OutPath $remoteOutRoot

    $remoteSteps += Run-RemoteStep `
        -StepName "convert_to_pixelsplat_format" `
        -SeqNamesText $SeqNames `
        -GeomDirName $geom `
        -MaxFramesArg $MaxFrames `
        -ScriptRelPath "tools/convert_to_pixelsplat_format.py" `
        -ArgsExtra (
            "--zju_root /mnt/data/zju_mocap --seq_names $SeqNames --geom_subdir $geom --out_root $remoteConv " +
            "--frame_stride $FrameStride --max_frames $MaxFrames --max_views $MaxViews --copy_mode none"
        ) `
        -TsTag $ts
    $remoteSteps[-1] | Add-Member -NotePropertyName out -NotePropertyValue $remoteConv -Force

    $manifestLocal = Join-Path $outAbs "pixelsplat_manifest.remote.json"
    Invoke-ModalVolumeGet -RemotePath "$remoteVolRoot/converted/pixelsplat_manifest.json" -LocalDest $manifestLocal -Force

    $seq = First-SeqName -SeqNamesText $SeqNames
    $frameId = ""
    try {
        $manifestObj = Get-Content $manifestLocal -Raw | ConvertFrom-Json
        $seqObj = $null
        if ($manifestObj.sequences) {
            $seqObj = $manifestObj.sequences.$seq
            if ($null -eq $seqObj) {
                $firstProp = $manifestObj.sequences.PSObject.Properties | Select-Object -First 1
                if ($firstProp) {
                    $seq = [string]$firstProp.Name
                    $seqObj = $firstProp.Value
                }
            }
        }
        if ($seqObj -and $seqObj.frames -and $seqObj.frames.Count -gt 0) {
            $frameId = [string]$seqObj.frames[0].frame_id
        }
    } catch {
    }
    if (-not [string]::IsNullOrWhiteSpace($frameId) -and -not $frameId.EndsWith(".npz")) {
        $frameId = "$frameId.npz"
    }

    $remoteSteps += Run-RemoteStep `
        -StepName "camera_sanity_check" `
        -SeqNamesText $SeqNames `
        -GeomDirName $geom `
        -MaxFramesArg $MaxFrames `
        -ScriptRelPath "camera_sanity_check.py" `
        -ArgsExtra (
            "--zju_root /mnt/data/zju_mocap --seq_names $SeqNames --geom_subdir $geom --out $remoteSanity --num_frames $CameraSanityNumFrames"
        ) `
        -TsTag $ts
    $remoteSteps[-1] | Add-Member -NotePropertyName out -NotePropertyValue $remoteSanity -Force

    $remoteSteps += Run-RemoteStep `
        -StepName "inspect_batch_mosaic" `
        -SeqNamesText $SeqNames `
        -GeomDirName $geom `
        -MaxFramesArg $MaxFrames `
        -ScriptRelPath "inspect_batch_mosaic.py" `
        -ArgsExtra (
            "--zju_root /mnt/data/zju_mocap --seq_names $SeqNames --geom_subdir $geom --out $remoteMosaic " +
            "--num_samples $MosaicNumSamples --num_targets $MosaicNumTargets --num_src_views $MosaicNumSrcViews"
        ) `
        -TsTag $ts
    $remoteSteps[-1] | Add-Member -NotePropertyName out -NotePropertyValue $remoteMosaic -Force

    if (-not [string]::IsNullOrWhiteSpace($frameId)) {
        $remoteNpz = "/mnt/data/zju_mocap/$seq/$geom/$frameId"
        $remoteSteps += Run-RemoteStep `
            -StepName "select_views_uniform_yaw_demo" `
            -SeqNamesText $SeqNames `
            -GeomDirName $geom `
            -MaxFramesArg $MaxFrames `
            -ScriptRelPath "select_views_uniform_yaw.py" `
            -ArgsExtra (
                "--npz $remoteNpz --zju_root /mnt/data/zju_mocap --num_src_views $MosaicNumSrcViews --dump_json $remoteSelectJson"
            ) `
            -TsTag $ts
        $remoteSteps[-1] | Add-Member -NotePropertyName out -NotePropertyValue $remoteSelectJson -Force
    } else {
        $remoteSteps += [pscustomobject]@{
            step = "select_views_uniform_yaw_demo"
            ok = $false
            mode = "remote"
            reason = "failed to infer remote frame npz from manifest"
        }
    }

    $fetchRoot = Join-Path $outAbs "remote_fetch"
    New-Item -ItemType Directory -Path $fetchRoot -Force | Out-Null
    Invoke-ModalVolumeGet -RemotePath $remoteVolRoot -LocalDest $fetchRoot -Force
    $downloadedRoot = Join-Path $fetchRoot $remoteTag

    return [pscustomobject]@{
        steps = $remoteSteps
        remote_out_root = $remoteOutRoot
        remote_out_volume_path = $remoteVolRoot
        local_fetch_root = $fetchRoot
        local_download_root = $downloadedRoot
        manifest_local = $manifestLocal
    }
}

try {
    if (-not $useRemote) {
        try {
            $steps = Try-LocalPrep
            $runMode = "local"
        } catch {
            if ($NoRemoteFallbackOnLocalError) {
                throw
            }
            $runMode = "remote_fallback"
            $reason = "local prep failed, fallback to remote: $_"
            $geom = $geom0
            $remote = Run-RemotePrep
            $steps = @($remote.steps)
        }
    } else {
        $runMode = "remote"
        $remote = Run-RemotePrep
        $steps = @($remote.steps)
    }
} catch {
    $status = "error"
    $reason = "$_"
}

$pixelsplatRepo = Join-Path $CodeDir "external/pixelsplat"
$pixelsplatReady = Test-Path $pixelsplatRepo
$pixelsplatStatus = if ($pixelsplatReady) { "ready_repo_found" } else { "blocked_missing_repo" }

$result = [ordered]@{
    timestamp = $ts
    status = $status
    reason = $reason
    run_mode = $runMode
    code_dir = $CodeDir
    zju_root = $ZjuRoot
    seq_names = $SeqNames
    geom_subdir = $geom
    out_root = $outAbs
    local_first_npz = $localFirstNpz
    steps = $steps
    remote = $remote
    pixelsplat = [ordered]@{
        repo_path = $pixelsplatRepo
        status = $pixelsplatStatus
    }
}

Write-JsonNoBom -Path $resultJson -Obj $result
Write-JsonNoBom -Path $resultJsonLatest -Obj $result

$md = @()
$md += "# Phase2 PixelSplat Prep"
$md += ""
$md += "- timestamp: $ts"
$md += "- status: $status"
$md += "- reason: $reason"
$md += "- run_mode: $runMode"
$md += "- geom_subdir: $geom"
$md += "- out_root: $outAbs"
$md += "- local_first_npz: $localFirstNpz"
$md += "- pixelsplat_status: $pixelsplatStatus"
if ($remote) {
    $md += "- remote_out_root: $($remote.remote_out_root)"
    $md += "- remote_out_volume_path: $($remote.remote_out_volume_path)"
    $md += "- local_fetch_root: $($remote.local_fetch_root)"
    $md += "- local_download_root: $($remote.local_download_root)"
}
foreach ($s in $steps) {
    $md += "- step: $($s.step) ok=$($s.ok) mode=$($s.mode)"
    if ($s.out) { $md += "  out: $($s.out)" }
    if ($s.reason) { $md += "  reason: $($s.reason)" }
    if ($s.log) { $md += "  log: $($s.log)" }
}
Set-Content -Path $resultMdLatest -Value ($md -join "`n") -Encoding UTF8

Write-Host "[phase2] wrote: $resultJsonLatest"
Write-Host "[phase2] wrote: $resultMdLatest"
if ($status -ne "ok") {
    exit 2
}
exit 0
