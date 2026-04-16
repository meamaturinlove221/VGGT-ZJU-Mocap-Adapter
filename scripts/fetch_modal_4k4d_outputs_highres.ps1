param(
    [string]$Seq = "0012_11",
    [string]$RemoteDataRoot = "/4k4d_bridge",
    [string]$MountedZjuRoot = "/mnt/data/4k4d_bridge",
    [string]$GeomSubdir = "vggt_geom_4k4d_0012_11_96f7v_20260414",
    [string]$RemoteRunDir = "",
    [string]$LocalBridgeRoot = "",
    [string]$CamNames = "",
    [int[]]$Frames = @(0),
    [string]$TargetView = "Camera_00",
    [double]$DepthUpsampleFactor = 2.0,
    [int]$PreviewTileSize = 1400,
    [int]$MaxPoints = 180000,
    [int]$PreviewPoints = 90000,
    [double]$ConfPercentile = 72.0,
    [int]$SplatRadius = 1,
    [bool]$RenderOriginalResolution = $true,
    [string]$GpuSpecTrain = "A100-80GB",
    [int]$Epochs = 6,
    [int]$MaxFrames = 96,
    [double]$Lr = 1e-6,
    [string]$HumanPriorSubdir = "human_prior",
    [double]$HumanPriorBlendAlpha = 0.30,
    [double]$HumanPriorWeightBoost = 1.50,
    [double]$HumanPriorCompleteWeight = 0.35,
    [double]$LambdaPointPrior = 0.10,
    [switch]$RunTraining,
    [switch]$AllowMissingHumanPrior,
    [switch]$DownloadBridgeIfMissing,
    [string[]]$CleanupAppNames = @("vggt-zju-runner", "vggt-4k4d-train", "vggt-zju-geometry-minimal-finetune"),
    [bool]$RequireCleanModalStop = $true,
    [bool]$RequireReturnedArtifacts = $true,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Initialize-ModalCliEnvironment {
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    try {
        [Console]::OutputEncoding = $utf8
    } catch {
    }
    $global:OutputEncoding = $utf8
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:NO_COLOR = "1"
}

function Invoke-ModalJson {
    param([string[]]$CliArgs)
    $raw = & modal $CliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "modal command failed: modal $($CliArgs -join ' ')"
    }
    $blob = ($raw | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($blob)) {
        return @()
    }
    return ($blob | ConvertFrom-Json)
}

function Get-RemoteRunDirs {
    param([string]$SeqName)
    $items = Invoke-ModalJson -CliArgs @("volume", "ls", "--json", "vggt-out", "/viewdec_ablation")
    $dirs = @($items | Where-Object {
        [string]$_.Type -eq "dir"
    } | ForEach-Object {
        $rawName = [string]$_.Filename
        $normalized = $rawName.Replace("\", "/").TrimStart("/")
        $leaf = ($normalized.Split("/") | Select-Object -Last 1)
        if ((-not [string]::IsNullOrWhiteSpace($leaf)) -and ($leaf -like "$SeqName_*")) {
            if ($normalized -match "^viewdec_ablation/") {
                $normalized
            } else {
                "viewdec_ablation/$leaf"
            }
        }
    })
    return @($dirs | Sort-Object)
}

function Resolve-RemoteRunDir {
    param(
        [string]$SeqName,
        [string]$ExplicitRemoteRunDir,
        [string[]]$BeforeRuns
    )
    if (-not [string]::IsNullOrWhiteSpace($ExplicitRemoteRunDir)) {
        $clean = ([string]$ExplicitRemoteRunDir).Trim().TrimStart("/")
        if ($clean -notmatch "/") {
            $clean = "viewdec_ablation/$clean"
        }
        return $clean
    }
    $afterRuns = @(Get-RemoteRunDirs -SeqName $SeqName)
    $newRuns = @($afterRuns | Where-Object { ([string[]]$BeforeRuns) -notcontains ([string]$_) })
    if ($newRuns.Count -gt 0) {
        return [string]($newRuns | Sort-Object | Select-Object -Last 1)
    }
    if ($afterRuns.Count -gt 0) {
        return [string]($afterRuns | Sort-Object | Select-Object -Last 1)
    }
    throw "No remote run directory found for seq=$SeqName under /viewdec_ablation"
}

function Stop-LingeringModalApps {
    param([string[]]$AppNames)
    $targets = @($AppNames | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ } | Select-Object -Unique)
    $items = @(Invoke-ModalJson -CliArgs @("app", "list", "--json"))
    $active = @($items | Where-Object {
        ($targets -contains [string]$_."Description") -and ([string]$_."State" -ne "stopped")
    })
    foreach ($app in $active) {
        $appId = [string]$app."App ID"
        if (-not [string]::IsNullOrWhiteSpace($appId)) {
            Write-Host "[modal-return] stopping lingering app: $appId ($($app.Description))"
            & modal app stop $appId | Out-Null
        }
    }
    Start-Sleep -Seconds 3
    $post = @(Invoke-ModalJson -CliArgs @("app", "list", "--json"))
    $stillActive = @($post | Where-Object {
        ($targets -contains [string]$_."Description") -and ([string]$_."State" -ne "stopped")
    })
    return @{
        app_names = @($targets)
        stopped_count = [int]$active.Count
        remaining_active = [int]$stillActive.Count
        remaining_ids = @($stillActive | ForEach-Object { [string]$_."App ID" })
        remaining_descriptions = @($stillActive | ForEach-Object { [string]$_."Description" })
    }
}

function Resolve-LocalBridgeRoot {
    param(
        [string]$SeqName,
        [string]$RequestedRoot,
        [string]$RepoRoot,
        [switch]$AllowDownload
    )
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        $candidates += $RequestedRoot
    }
    $candidates += @(
        (Join-Path $RepoRoot "out_vis\\bridge_4k4d_med96\\$SeqName"),
        (Join-Path $RepoRoot "out_vis\\bridge_4k4d_run\\$SeqName"),
        (Join-Path $RepoRoot "out_vis\\bridge_4k4d_smoke\\$SeqName"),
        (Join-Path $RepoRoot "out_vis\\bridge_4k4d_smoke_range\\$SeqName")
    )
    foreach ($cand in $candidates) {
        if ([string]::IsNullOrWhiteSpace($cand)) { continue }
        if (Test-Path $cand) {
            return (Resolve-Path $cand).Path
        }
    }
    if (-not $AllowDownload) {
        throw "Local bridge root not found for seq=$SeqName. Pass -LocalBridgeRoot or -DownloadBridgeIfMissing."
    }
    return ""
}

function Download-ModalDirectory {
    param(
        [string]$VolumeName,
        [string]$RemotePath,
        [string]$LocalDestination
    )
    New-Item -ItemType Directory -Force -Path $LocalDestination | Out-Null
    & modal volume get $VolumeName $RemotePath $LocalDestination --force | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "modal volume get failed: volume=$VolumeName remote=$RemotePath local=$LocalDestination"
    }
}

function Download-ModalFile {
    param(
        [string]$VolumeName,
        [string]$RemotePath,
        [string]$LocalFile
    )
    $parent = Split-Path -Parent $LocalFile
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    & modal volume get $VolumeName $RemotePath $LocalFile --force | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "modal volume get failed: volume=$VolumeName remote=$RemotePath local=$LocalFile"
    }
}

function Resolve-ReturnedArtifact {
    param(
        [string]$RootDir,
        [string]$LeafName
    )
    if ([string]::IsNullOrWhiteSpace($RootDir) -or -not (Test-Path $RootDir)) {
        return ""
    }
    $hit = @(Get-ChildItem -Path $RootDir -Recurse -File -Filter $LeafName -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($hit.Count -le 0) {
        return ""
    }
    return [string]$hit[0].FullName
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot
Initialize-ModalCliEnvironment

if ($DryRun) {
    $dry = [ordered]@{
        seq = $Seq
        remote_data_root = $RemoteDataRoot
        mounted_zju_root = $MountedZjuRoot
        geom_subdir = $GeomSubdir
        cam_names = $CamNames
        frames = ($Frames -join ",")
        target_view = $TargetView
        depth_upsample_factor = $DepthUpsampleFactor
        preview_tile_size = $PreviewTileSize
        max_points = $MaxPoints
        preview_points = $PreviewPoints
        conf_percentile = $ConfPercentile
        render_original_resolution = [bool]$RenderOriginalResolution
        run_training = [bool]$RunTraining
        gpu_spec_train = $GpuSpecTrain
        epochs = $Epochs
        max_frames = $MaxFrames
        lr = $Lr
        human_prior_subdir = $HumanPriorSubdir
        human_prior_blend_alpha = $HumanPriorBlendAlpha
        human_prior_weight_boost = $HumanPriorWeightBoost
        human_prior_complete_weight = $HumanPriorCompleteWeight
        lambda_point_prior = $LambdaPointPrior
        cleanup_app_names = ($CleanupAppNames -join ",")
        require_clean_modal_stop = [bool]$RequireCleanModalStop
        require_returned_artifacts = [bool]$RequireReturnedArtifacts
    }
    Write-Host "[modal-return] dry-run summary:"
    $dry.GetEnumerator() | ForEach-Object {
        Write-Host ("  {0} = {1}" -f $_.Key, $_.Value)
    }
    exit 0
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$returnRoot = Join-Path $repoRoot "out_vis\\modal_returns\\${Seq}_$timestamp"
$remoteRunBefore = @(Get-RemoteRunDirs -SeqName $Seq)

if ($RunTraining) {
    Write-Host "[modal-return] launching training first..."
    $launcher = Join-Path $PSScriptRoot "run_modal_4k4d_human_prior_finetune.ps1"
    if (-not (Test-Path $launcher)) {
        throw "training launcher missing: $launcher"
    }
    & powershell -ExecutionPolicy Bypass -File $launcher `
        -Seq $Seq `
        -ZjuRoot $MountedZjuRoot `
        -GeomSubdir $GeomSubdir `
        -HumanPriorSubdir $HumanPriorSubdir `
        -CamNames $CamNames `
        -GpuSpecTrain $GpuSpecTrain `
        -Epochs $Epochs `
        -MaxFrames $MaxFrames `
        -Lr $Lr `
        -HumanPriorBlendAlpha $HumanPriorBlendAlpha `
        -HumanPriorWeightBoost $HumanPriorWeightBoost `
        -HumanPriorCompleteWeight $HumanPriorCompleteWeight `
        -LambdaPointPrior $LambdaPointPrior `
        -AllowMissingHumanPrior:$AllowMissingHumanPrior
    if ($LASTEXITCODE -ne 0) {
        throw "training launcher failed"
    }
}

$resolvedRemoteRunDir = Resolve-RemoteRunDir -SeqName $Seq -ExplicitRemoteRunDir $RemoteRunDir -BeforeRuns $remoteRunBefore
Write-Host "[modal-return] remote run dir: $resolvedRemoteRunDir"

$remoteRunLocal = Join-Path $returnRoot "remote_run"
Download-ModalDirectory -VolumeName "vggt-out" -RemotePath ("/" + $resolvedRemoteRunDir.TrimStart("/")) -LocalDestination $remoteRunLocal
$remoteSummary = Resolve-ReturnedArtifact -RootDir $remoteRunLocal -LeafName "finetune_vggt_summary.json"
$remoteMetrics = Resolve-ReturnedArtifact -RootDir $remoteRunLocal -LeafName "finetune_vggt_metrics.jsonl"
$remoteCkptBest = Resolve-ReturnedArtifact -RootDir $remoteRunLocal -LeafName "model_ft_zju.pt"
$remoteCkptLast = Resolve-ReturnedArtifact -RootDir $remoteRunLocal -LeafName "model_ft_zju_last.pt"

$bridgeRootResolved = Resolve-LocalBridgeRoot -SeqName $Seq -RequestedRoot $LocalBridgeRoot -RepoRoot $repoRoot -AllowDownload:$DownloadBridgeIfMissing
if ([string]::IsNullOrWhiteSpace($bridgeRootResolved)) {
    $bridgeRootResolved = Join-Path $returnRoot "bridge_seq"
    Write-Host "[modal-return] local bridge missing; downloading remote bridge to $bridgeRootResolved"
    Download-ModalDirectory -VolumeName "vggt-zju-data" -RemotePath "$RemoteDataRoot/$Seq" -LocalDestination $bridgeRootResolved
}
Write-Host "[modal-return] local bridge root: $bridgeRootResolved"

$geomLocalRoot = Join-Path $returnRoot "geom_cache"
New-Item -ItemType Directory -Force -Path $geomLocalRoot | Out-Null

$downloadedNpz = @()
foreach ($frameId in $Frames) {
    $stem = ("frame_{0:D6}.npz" -f [int]$frameId)
    $remoteNp = "$RemoteDataRoot/$Seq/$GeomSubdir/$stem"
    $localNp = Join-Path $geomLocalRoot $stem
    Write-Host "[modal-return] downloading geom npz frame=$frameId"
    Download-ModalFile -VolumeName "vggt-zju-data" -RemotePath $remoteNp -LocalFile $localNp
    $downloadedNpz += $localNp
}

$pointcloudOut = Join-Path $returnRoot "pointcloud_highres"
$reprojOut = Join-Path $returnRoot "reprojection_highres"
New-Item -ItemType Directory -Force -Path $pointcloudOut | Out-Null
New-Item -ItemType Directory -Force -Path $reprojOut | Out-Null

$npzArgs = @()
foreach ($npz in $downloadedNpz) {
    $npzArgs += $npz
}

Write-Host "[modal-return] exporting high-res fused point cloud previews..."
& python .\scripts\export_4k4d_pointcloud_previews.py `
    --seq-root $bridgeRootResolved `
    --output-dir $pointcloudOut `
    --npz-paths $npzArgs `
    --rebuild-from-depth `
    --depth-upsample-factor $DepthUpsampleFactor `
    --conf-percentile $ConfPercentile `
    --max-points $MaxPoints `
    --preview-points $PreviewPoints `
    --preview-tile-size $PreviewTileSize
if ($LASTEXITCODE -ne 0) {
    throw "high-res pointcloud export failed"
}

$reprojMeta = @()
foreach ($npz in $downloadedNpz) {
    Write-Host "[modal-return] exporting high-res target-view reprojection from $(Split-Path -Leaf $npz)..."
    $reprojArgs = @(
        ".\scripts\render_4k4d_target_reprojection.py",
        "--npz-path", $npz,
        "--seq-root", $bridgeRootResolved,
        "--output-dir", $reprojOut,
        "--target-view", $TargetView,
        "--conf-percentile", [string]$ConfPercentile,
        "--splat-radius", [string]$SplatRadius,
        "--rebuild-from-depth",
        "--depth-upsample-factor", [string]$DepthUpsampleFactor
    )
    if ($RenderOriginalResolution) {
        $reprojArgs += "--render-original-resolution"
    }
    & python @reprojArgs
    if ($LASTEXITCODE -ne 0) {
        throw "high-res reprojection export failed for $npz"
    }
}

$cleanup = Stop-LingeringModalApps -AppNames $CleanupAppNames
if ($RequireCleanModalStop -and ([int]$cleanup.remaining_active -gt 0)) {
    throw "Modal cleanup incomplete: remaining_active=$($cleanup.remaining_active) ids=$($cleanup.remaining_ids -join ',') descriptions=$($cleanup.remaining_descriptions -join ',')"
}

$pointcloudSummary = Get-ChildItem -Path $pointcloudOut -Filter "*_export_summary.json" -File -ErrorAction SilentlyContinue | Select-Object -First 1
$reprojMetaFiles = @(Get-ChildItem -Path $reprojOut -Filter "*_reprojection_meta.json" -File -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
$pointcloudPlyFiles = @(Get-ChildItem -Path $pointcloudOut -Filter "*_fused_pointcloud.ply" -File -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
$pointcloudPreviewFiles = @(Get-ChildItem -Path $pointcloudOut -Filter "*_preview_contact_sheet.png" -File -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)

if ($RequireReturnedArtifacts) {
    $missing = @()
    if ([string]::IsNullOrWhiteSpace($remoteSummary)) { $missing += "remote_summary" }
    if ([string]::IsNullOrWhiteSpace($remoteMetrics)) { $missing += "remote_metrics" }
    if ([string]::IsNullOrWhiteSpace($remoteCkptBest)) { $missing += "remote_ckpt_best" }
    if ([string]::IsNullOrWhiteSpace($remoteCkptLast)) { $missing += "remote_ckpt_last" }
    if (-not $pointcloudSummary) { $missing += "pointcloud_summary" }
    if ($pointcloudPlyFiles.Count -le 0) { $missing += "pointcloud_ply" }
    if ($pointcloudPreviewFiles.Count -le 0) { $missing += "pointcloud_preview_png" }
    if ($reprojMetaFiles.Count -le 0) { $missing += "reprojection_meta" }
    if ($missing.Count -gt 0) {
        throw "Returned artifact check failed. Missing: $($missing -join ', ')"
    }
}

$manifest = [ordered]@{
    seq = $Seq
    remote_run_dir = $resolvedRemoteRunDir
    local_return_root = $returnRoot
    local_bridge_root = $bridgeRootResolved
    geom_subdir = $GeomSubdir
    frames = @($Frames)
    target_view = $TargetView
    depth_upsample_factor = $DepthUpsampleFactor
    render_original_resolution = [bool]$RenderOriginalResolution
    preview_tile_size = $PreviewTileSize
    max_points = $MaxPoints
    preview_points = $PreviewPoints
    remote_run_local = $remoteRunLocal
    remote_summary = $remoteSummary
    remote_metrics = $remoteMetrics
    remote_ckpt_best = $remoteCkptBest
    remote_ckpt_last = $remoteCkptLast
    downloaded_npz = @($downloadedNpz)
    pointcloud_output_dir = $pointcloudOut
    pointcloud_summary = $(if ($pointcloudSummary) { $pointcloudSummary.FullName } else { "" })
    pointcloud_ply = @($pointcloudPlyFiles)
    pointcloud_preview_png = @($pointcloudPreviewFiles)
    reprojection_output_dir = $reprojOut
    reprojection_meta = @($reprojMetaFiles)
    modal_cleanup = $cleanup
    artifacts_verified = [bool]$RequireReturnedArtifacts
    cleanup_verified = [bool]($cleanup.remaining_active -eq 0)
}

$manifestPath = Join-Path $returnRoot "return_manifest.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestPath -Encoding UTF8

$md = @()
$md += "# Modal Return Summary"
$md += ""
$md += "- seq: $Seq"
$md += "- remote_run_dir: $resolvedRemoteRunDir"
$md += "- local_return_root: $returnRoot"
$md += "- geom_subdir: $GeomSubdir"
$md += "- frames: $($Frames -join ', ')"
$md += "- target_view: $TargetView"
$md += "- depth_upsample_factor: $DepthUpsampleFactor"
$md += "- render_original_resolution: $([bool]$RenderOriginalResolution)"
$md += "- remote_summary: $remoteSummary"
$md += "- remote_metrics: $remoteMetrics"
$md += "- remote_ckpt_best: $remoteCkptBest"
$md += "- remote_ckpt_last: $remoteCkptLast"
$md += "- pointcloud_output_dir: $pointcloudOut"
$md += "- reprojection_output_dir: $reprojOut"
$md += "- cleanup_app_names: $($CleanupAppNames -join ', ')"
$md += "- cleanup_stopped_count: $($cleanup.stopped_count)"
$md += "- cleanup_remaining_active: $($cleanup.remaining_active)"
$md += "- cleanup_verified: $([bool]($cleanup.remaining_active -eq 0))"
if ($pointcloudSummary) {
    $md += "- pointcloud_summary: $($pointcloudSummary.FullName)"
}
$md += "- pointcloud_ply_count: $($pointcloudPlyFiles.Count)"
$md += "- pointcloud_preview_png_count: $($pointcloudPreviewFiles.Count)"
foreach ($metaPath in $reprojMetaFiles) {
    $md += "- reprojection_meta: $metaPath"
}
$mdPath = Join-Path $returnRoot "return_manifest.md"
$md | Set-Content -Path $mdPath -Encoding UTF8

Write-Host "[modal-return] completed."
Write-Host "[modal-return] manifest json: $manifestPath"
Write-Host "[modal-return] manifest md:   $mdPath"
