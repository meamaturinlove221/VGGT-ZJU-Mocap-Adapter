param(
    [string]$CodeDir = "F:\vggt",
    [string]$PythonExe = "",
    [string]$SeqName = "CoreView_390",
    [int]$FrameId = 1080,
    [string]$TgtCamera = "Camera_B5",
    [string]$ZjuRoot = "F:\datasets\ZJU_MoCap\data\zju_mocap",
    [string]$RemoteZjuRoot = "/mnt/data/zju_mocap",
    [string]$Ckpt = "model.pt",
    [string]$ReportsDir = "logs/modal_phase5/reports",
    [string]$LocalOutRoot = "infer_out/vggt_raw_viewcount",
    [string]$StateJson = "logs/modal_phase5/orig_vggt_viewcount_task_latest.json",
    [string]$StateMd = "logs/modal_phase5/orig_vggt_viewcount_task_latest.md",
    [switch]$SkipModal
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Resolve-PythonExe([string]$Preferred) {
    if (-not [string]::IsNullOrWhiteSpace($Preferred) -and (Test-Path $Preferred)) {
        return (Resolve-Path $Preferred).Path
    }
    $hist = "D:\anaconda\envs\vggt-colmap\python.exe"
    if (Test-Path $hist) {
        return (Resolve-Path $hist).Path
    }
    return "python"
}

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $abs = Join-Path (Resolve-Path ".").Path $Path
    $dir = Split-Path -Parent $abs
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 20
    [System.IO.File]::WriteAllText($abs, $json, $enc)
}

function Write-TextNoBom([string]$Path, [string]$Text) {
    $abs = Join-Path (Resolve-Path ".").Path $Path
    $dir = Split-Path -Parent $abs
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($abs, $Text, $enc)
}

function To-VolumePath([string]$Raw) {
    $s = [string]$Raw
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

function Get-TaggedLineValue([string[]]$Lines, [string]$Prefix) {
    foreach ($line in @($Lines)) {
        if ([string]$line -like "$Prefix*") {
            return ([string]$line).Substring($Prefix.Length).Trim()
        }
    }
    return ""
}

function Invoke-Capture {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory
    )
    $stdoutFile = [System.IO.Path]::GetTempFileName()
    $stderrFile = [System.IO.Path]::GetTempFileName()
    try {
        $proc = Start-Process `
            -FilePath $FilePath `
            -ArgumentList $ArgumentList `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -WorkingDirectory $WorkingDirectory `
            -RedirectStandardOutput $stdoutFile `
            -RedirectStandardError $stderrFile
        $output = @()
        if (Test-Path $stdoutFile) { $output += @(Get-Content $stdoutFile) }
        if (Test-Path $stderrFile) { $output += @(Get-Content $stderrFile) }
        return [pscustomobject]@{
            ExitCode = [int]$proc.ExitCode
            Output = $output
        }
    } finally {
        Remove-Item $stdoutFile -ErrorAction SilentlyContinue
        Remove-Item $stderrFile -ErrorAction SilentlyContinue
    }
}

function Invoke-ModalRun {
    param(
        [string]$ScriptPath = "modal_run_train.py",
        [int]$MaxRetries = 3,
        [int]$RetrySleepSec = 10
    )
    $attempt = 0
    $last = $null
    while ($attempt -lt [Math]::Max(1, $MaxRetries)) {
        $attempt += 1
        $result = Invoke-Capture -FilePath "cmd.exe" -ArgumentList @("/c", "modal run -q $ScriptPath") -WorkingDirectory (Resolve-Path ".").Path
        $last = [pscustomobject]@{
            ExitCode = [int]$result.ExitCode
            Output = @($result.Output)
            Attempt = $attempt
        }
        if ($result.ExitCode -eq 0) {
            return $last
        }
        $blob = (@($result.Output) -join "`n")
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
            ($blob -match "Cannot connect to host")
        )
        if ($isTransient -and $attempt -lt $MaxRetries) {
            Start-Sleep -Seconds $RetrySleepSec
            continue
        }
        return $last
    }
    return $last
}

function Get-VolumeFileNames([object[]]$Items) {
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($it in @($Items)) {
        if ($null -eq $it) { continue }
        $type = ""
        try {
            if ($it.PSObject.Properties["Type"]) { $type = [string]$it.Type }
        } catch {}
        if (-not [string]::IsNullOrWhiteSpace($type) -and ($type -ne "file")) {
            continue
        }
        $name = ""
        foreach ($k in @("Filename", "filename", "Path", "path", "Name", "name")) {
            try {
                if ($it.PSObject.Properties[$k]) {
                    $name = [string]$it.$k
                    if (-not [string]::IsNullOrWhiteSpace($name)) { break }
                }
            } catch {}
        }
        if (-not [string]::IsNullOrWhiteSpace($name)) {
            $out.Add($name) | Out-Null
        }
    }
    return @($out | Select-Object -Unique)
}

function Get-VolumeDirNames([object[]]$Items) {
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($it in @($Items)) {
        if ($null -eq $it) { continue }
        $type = ""
        try {
            if ($it.PSObject.Properties["Type"]) { $type = [string]$it.Type }
        } catch {}
        if ($type -ne "dir") {
            continue
        }
        $name = ""
        foreach ($k in @("Filename", "filename", "Path", "path", "Name", "name")) {
            try {
                if ($it.PSObject.Properties[$k]) {
                    $name = [string]$it.$k
                    if (-not [string]::IsNullOrWhiteSpace($name)) { break }
                }
            } catch {}
        }
        if (-not [string]::IsNullOrWhiteSpace($name)) {
            $out.Add($name) | Out-Null
        }
    }
    return @($out | Select-Object -Unique)
}

function Download-ModalDirectory {
    param(
        [string]$RemoteDir,
        [string]$LocalDir
    )
    New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
    $itemsJson = modal volume ls --json vggt-out $RemoteDir 2>$null | Out-String
    if ([string]::IsNullOrWhiteSpace($itemsJson)) {
        throw "modal volume ls returned empty for $RemoteDir"
    }
    $items = $itemsJson | ConvertFrom-Json
    $fileNames = @(Get-VolumeFileNames $items)
    if ($fileNames.Count -le 0) {
        throw "no files listed at $RemoteDir"
    }
    foreach ($name in $fileNames) {
        $nameText = [string]$name
        if ($nameText.StartsWith("/")) {
            $remote = $nameText
        } elseif ($nameText.StartsWith($RemoteDir.TrimStart("/"))) {
            $remote = "/" + $nameText.TrimStart("/")
        } else {
            $remote = "$RemoteDir/$nameText"
        }
        $local = Join-Path $LocalDir ([System.IO.Path]::GetFileName([string]$name))
        modal volume get vggt-out $remote $local 2>$null | Out-Null
    }
}

function Read-Json([string]$Path) {
    return Get-Content $Path -Raw | ConvertFrom-Json
}

function Update-State(
    [string]$CurrentStage,
    [string]$Status = "running",
    [string]$Message = "",
    [hashtable]$Extra = @{}
) {
    $payload = [ordered]@{
        updated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        status = $Status
        current_stage = $CurrentStage
        message = $Message
        python = $script:PythonResolved
        seq_name = $SeqName
        frame_id = $FrameId
        tgt_camera = $TgtCamera
        results = $script:RunResults
    }
    foreach ($k in $Extra.Keys) {
        $payload[$k] = $Extra[$k]
    }
    Write-JsonNoBom $StateJson $payload
    $md = @(
        "# Original VGGT Viewcount Task",
        "",
        "- status: $Status",
        "- current_stage: $CurrentStage",
        "- message: $Message",
        "- updated_at: $($payload.updated_at)",
        "",
        "## Results",
        ""
    )
    foreach ($key in $script:RunResults.Keys) {
        $row = $script:RunResults[$key]
        $md += "- ${key}: $($row | ConvertTo-Json -Compress)"
    }
    Write-TextNoBom $StateMd ($md -join "`n")
}

function Run-PythonStep(
    [string]$Stage,
    [string[]]$PyArgs
) {
    Update-State -CurrentStage $Stage -Message ("python " + ($PyArgs -join " "))
    $result = Invoke-Capture -FilePath $script:PythonResolved -ArgumentList $PyArgs -WorkingDirectory (Resolve-Path ".").Path
    if ($result.ExitCode -ne 0) {
        Update-State -CurrentStage $Stage -Status "failed" -Message (@($result.Output) -join "`n")
        throw "python step failed: $Stage"
    }
    return $result.Output
}

Push-Location $CodeDir
try {
    $script:PythonResolved = Resolve-PythonExe $PythonExe
    $script:RunResults = [ordered]@{}
    $histAnchorPath = ""
    $histBackup = "G:\项目备份\vggt原版60°相机推理结果\infer_out\vggt_raw_middle_compare\CoreView_390\frame_001080_Camera_B5\run_20260207_234410\meta.json"
    if (Test-Path $histBackup) {
        $histMeta = Read-Json $histBackup
        if ($histMeta.anchor_cat_path -and (Test-Path ([string]$histMeta.anchor_cat_path))) {
            $histAnchorPath = [string]$histMeta.anchor_cat_path
        }
    }

    Update-State -CurrentStage "preflight" -Message "resolved python and historical backup metadata"

    $auditJson = Join-Path $ReportsDir "visual_taxonomy_latest.json"
    $auditCsv = Join-Path $ReportsDir "visual_taxonomy_latest.csv"
    $auditMd = Join-Path $ReportsDir "visual_taxonomy_latest.md"
    Run-PythonStep -Stage "taxonomy_audit" -PyArgs @(
        "scripts/orig_vggt_viewcount/audit_visual_taxonomy.py",
        "--reports_dir", $ReportsDir,
        "--out_json", $auditJson,
        "--out_csv", $auditCsv,
        "--out_md", $auditMd
    ) | Out-Null
    $script:RunResults["taxonomy"] = [ordered]@{
        json = $auditJson
        csv = $auditCsv
        md = $auditMd
    }
    Update-State -CurrentStage "taxonomy_audit" -Message "taxonomy audit finished"

    $smokeOutBase = Join-Path $LocalOutRoot "smoke_6src_hist"
    $smokeArgs = @(
        "scripts/orig_vggt_viewcount/render_raw_compare.py",
        "--seq_name", $SeqName,
        "--frame_id", "$FrameId",
        "--tgt_camera", $TgtCamera,
        "--view_profile", "6src_hist",
        "--zju_root", $ZjuRoot,
        "--ckpt", $Ckpt,
        "--out_dir", $smokeOutBase
    )
    if (-not [string]::IsNullOrWhiteSpace($histAnchorPath)) {
        $smokeArgs += @("--anchor_cat_path", $histAnchorPath)
    }
    $smokeOutput = Run-PythonStep -Stage "local_smoke_6src" -PyArgs $smokeArgs
    $smokeRunDir = Get-TaggedLineValue -Lines $smokeOutput -Prefix "RUN_DIR:"
    if ([string]::IsNullOrWhiteSpace($smokeRunDir)) {
        throw "local smoke did not emit RUN_DIR"
    }
    $smokeReport = Read-Json (Join-Path $smokeRunDir "report.json")
    $cov = [double]$smokeReport.render.coverage_ratio
    $meanConf = [double]$smokeReport.render.mean_conf
    $validContrib = [int]$smokeReport.render.valid_contrib
    if (($cov -lt 0.70) -or ($cov -gt 0.90)) {
        throw "local smoke coverage out of expected band: $cov"
    }
    if (($meanConf -lt 2.0) -or ($meanConf -gt 4.6)) {
        throw "local smoke mean_conf out of expected band: $meanConf"
    }
    if ($validContrib -lt 400000) {
        throw "local smoke valid_contrib too small: $validContrib"
    }
    $script:RunResults["smoke_6src_hist"] = [ordered]@{
        run_dir = $smokeRunDir
        coverage_ratio = $cov
        mean_conf = $meanConf
        valid_contrib = $validContrib
    }
    Update-State -CurrentStage "local_smoke_6src" -Message "local smoke passed"

    $profiles = @(
        [pscustomobject]@{ Label = "12src_nested"; RemoteBase = "/mnt/out/orig_vggt_viewcount/12src_nested"; LocalBase = (Join-Path $LocalOutRoot "12src_nested") },
        [pscustomobject]@{ Label = "6src_hist"; RemoteBase = "/mnt/out/orig_vggt_viewcount/6src_hist"; LocalBase = (Join-Path $LocalOutRoot "6src_hist") },
        [pscustomobject]@{ Label = "23cam_fullset"; RemoteBase = "/mnt/out/orig_vggt_viewcount/23cam_fullset"; LocalBase = (Join-Path $LocalOutRoot "23cam_fullset") }
    )

    foreach ($profile in $profiles) {
        $label = [string]$profile.Label
        if ($SkipModal) {
            continue
        }
        $env:VGGT_CODE_DIR = (Resolve-Path ".").Path
        $env:VGGT_MODE = "precompute"
        $env:VGGT_PRECOMPUTE_SCRIPT = "scripts/orig_vggt_viewcount/render_raw_compare.py"
        $env:VGGT_PRECOMPUTE_CKPT = $Ckpt
        $env:VGGT_ZJU_ROOT = $RemoteZjuRoot
        $env:VGGT_SEQ_NAMES = $SeqName
        $env:VGGT_GEOM_SUBDIR = [string]$profile.RemoteBase
        $env:VGGT_POINTMAP_SOURCE = "point_head"
        $env:VGGT_POINT_HEAD_FRAME = "world"
        $extraArgs = @(
            "--seq_name", $SeqName,
            "--frame_id", "$FrameId",
            "--tgt_camera", $TgtCamera,
            "--view_profile", $label
        )
        if (-not [string]::IsNullOrWhiteSpace($histAnchorPath)) {
            $extraArgs += @("--anchor_cat_path", $histAnchorPath)
        }
        $env:VGGT_PRECOMPUTE_ARGS_EXTRA = ($extraArgs | ForEach-Object {
            if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
        }) -join " "

        Update-State -CurrentStage ("modal_" + $label) -Message ("launch modal precompute for " + $label)
        $modalResult = Invoke-ModalRun
        if ($modalResult.ExitCode -ne 0) {
            Update-State -CurrentStage ("modal_" + $label) -Status "failed" -Message (@($modalResult.Output) -join "`n")
            throw "modal run failed for $label"
        }
        $frameTag = "frame_{0:D6}_{1}" -f $FrameId, $TgtCamera
        $remoteRunDir = Get-TaggedLineValue -Lines $modalResult.Output -Prefix "RUN_DIR:"
        if ([string]::IsNullOrWhiteSpace($remoteRunDir)) {
            $remoteFrameVolDir = To-VolumePath ("$($profile.RemoteBase)/$SeqName/$frameTag")
            $dirJson = modal volume ls --json vggt-out $remoteFrameVolDir 2>$null | Out-String
            if ([string]::IsNullOrWhiteSpace($dirJson)) {
                throw "modal run did not emit RUN_DIR and volume listing was empty for $label"
            }
            $dirItems = $dirJson | ConvertFrom-Json
            $dirNames = @((Get-VolumeDirNames $dirItems) | ForEach-Object { Split-Path -Leaf ([string]$_) })
            if ($dirNames.Count -le 0) {
                throw "modal run did not emit RUN_DIR and no run directories found for $label"
            }
            $runLeaf = @($dirNames | Sort-Object)[-1]
            $remoteVolDir = "$remoteFrameVolDir/$runLeaf"
            $remoteRunDir = "/mnt/out" + $remoteVolDir
        } else {
            $remoteVolDir = To-VolumePath $remoteRunDir
            $runLeaf = Split-Path -Leaf $remoteRunDir
        }
        $localFrameRoot = Join-Path $profile.LocalBase (Join-Path $SeqName $frameTag)
        $localRunDir = Join-Path $localFrameRoot $runLeaf
        Update-State -CurrentStage ("download_" + $label) -Message ("download " + $remoteVolDir)
        Download-ModalDirectory -RemoteDir $remoteVolDir -LocalDir $localRunDir

        $ghostRows = Join-Path $localRunDir "ghost_score_rows.csv"
        $ghostSummary = Join-Path $localRunDir "ghost_score_summary.csv"
        $ghostJson = Join-Path $localRunDir "ghost_score.json"
        Run-PythonStep -Stage ("ghost_" + $label) -PyArgs @(
            "tools/score_ghosting_from_cat_pred.py",
            "--input", "$label=$(Join-Path $localRunDir 'cat_fg_mask_pred_tgt_step000000.png')",
            "--out_csv", $ghostRows,
            "--out_summary_csv", $ghostSummary,
            "--out_json", $ghostJson
        ) | Out-Null

        $report = Read-Json (Join-Path $localRunDir "report.json")
        $ghostPayload = Read-Json $ghostJson
        $ghostBest = $null
        if ($ghostPayload.summary.Count -gt 0) {
            $ghostBest = $ghostPayload.summary[0]
        }
        $script:RunResults[$label] = [ordered]@{
            run_dir = $localRunDir
            remote_run_dir = $remoteRunDir
            coverage_ratio = [double]$report.render.coverage_ratio
            mean_conf = [double]$report.render.mean_conf
            valid_contrib = [int]$report.render.valid_contrib
            native_psnr = [double]$report.metrics.native.psnr
            native_ssim = [double]$report.metrics.native.ssim
            ghost_visual_score = if ($ghostBest) { [double]$ghostBest.ghost_visual_score_mean } else { [double]::NaN }
        }
        Update-State -CurrentStage ("ghost_" + $label) -Message ("completed " + $label)
    }

    $summaryArgs = @(
        "scripts/orig_vggt_viewcount/summarize_viewcount_runs.py",
        "--taxonomy_json", $auditJson,
        "--out_json", (Join-Path $ReportsDir "orig_vggt_viewcount_summary_latest.json"),
        "--out_csv", (Join-Path $ReportsDir "orig_vggt_viewcount_summary_latest.csv"),
        "--out_md", (Join-Path $ReportsDir "orig_vggt_viewcount_summary_latest.md")
    )
    foreach ($profileKey in @("12src_nested", "6src_hist", "23cam_fullset")) {
        if ($script:RunResults.Contains($profileKey)) {
            $summaryArgs += @("--run", "$profileKey=$($script:RunResults[$profileKey].run_dir)")
        }
    }
    Run-PythonStep -Stage "summary" -PyArgs $summaryArgs | Out-Null
    $script:RunResults["summary"] = [ordered]@{
        json = (Join-Path $ReportsDir "orig_vggt_viewcount_summary_latest.json")
        csv = (Join-Path $ReportsDir "orig_vggt_viewcount_summary_latest.csv")
        md = (Join-Path $ReportsDir "orig_vggt_viewcount_summary_latest.md")
    }
    Update-State -CurrentStage "done" -Status "completed" -Message "all requested tasks completed"
}
finally {
    Pop-Location
}
