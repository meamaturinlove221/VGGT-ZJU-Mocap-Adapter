param(
    [string]$Seq = "0012_11",
    [string]$ZjuRoot = "/mnt/data/4k4d_bridge",
    [string]$GeomSubdir = "vggt_geom_4k4d_0012_11_96f7v_20260414",
    [string]$HumanPriorSubdir = "human_prior",
    [string]$CamNames = "",
    [string]$GpuSpecTrain = "A100-80GB",
    [int]$Epochs = 6,
    [int]$MaxFrames = 96,
    [double]$Lr = 1e-6,
    [double]$HumanPriorBlendAlpha = 0.30,
    [double]$HumanPriorWeightBoost = 1.50,
    [double]$HumanPriorCompleteWeight = 0.35,
    [double]$LambdaPointPrior = 0.10,
    [switch]$AllowMissingHumanPrior,
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

function Resolve-RemoteVolumeRootFromMountedPath {
    param([string]$MountedPath)
    $raw = ([string]$MountedPath).Trim()
    if ($raw -notmatch '^/mnt/data(/.*)?$') {
        return ""
    }
    $suffix = $raw.Substring("/mnt/data".Length)
    if ([string]::IsNullOrWhiteSpace($suffix)) {
        return "/"
    }
    if (-not $suffix.StartsWith("/")) {
        $suffix = "/" + $suffix
    }
    return $suffix.TrimEnd("/")
}

function Assert-HumanPriorReady {
    param(
        [string]$SeqName,
        [string]$MountedZjuRoot,
        [string]$PriorSubdir
    )
    $cleanSubdir = ([string]$PriorSubdir).Trim().Trim('/','\')
    if ([string]::IsNullOrWhiteSpace($cleanSubdir)) {
        throw "Human prior subdir is empty."
    }

    $remoteRoot = Resolve-RemoteVolumeRootFromMountedPath -MountedPath $MountedZjuRoot
    if (-not [string]::IsNullOrWhiteSpace($remoteRoot)) {
        $remotePriorDir = ($remoteRoot.TrimEnd("/") + "/" + $SeqName + "/" + $cleanSubdir).Replace("//", "/")
        $items = @(Invoke-ModalJson -CliArgs @("volume", "ls", "--json", "vggt-zju-data", $remotePriorDir))
        $npz = @($items | Where-Object {
            ([string]$_.Type -eq "file") -and ([string]$_.Filename -like "*.npz")
        })
        if ($npz.Count -le 0) {
            throw "No human-prior sidecar npz found in remote volume path: $remotePriorDir"
        }
        return @{
            mode = "remote_volume"
            root = $remotePriorDir
            npz_count = [int]$npz.Count
        }
    }

    $localPriorDir = Join-Path (Join-Path $MountedZjuRoot $SeqName) $cleanSubdir
    if (-not (Test-Path $localPriorDir)) {
        throw "Human-prior sidecar directory missing: $localPriorDir"
    }
    $npz = @(Get-ChildItem -Path $localPriorDir -Filter *.npz -File -ErrorAction SilentlyContinue)
    if ($npz.Count -le 0) {
        throw "No human-prior sidecar npz found in local directory: $localPriorDir"
    }
    return @{
        mode = "local_fs"
        root = $localPriorDir
        npz_count = [int]$npz.Count
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot
Initialize-ModalCliEnvironment

$env:VGGT_CODE_DIR = $repoRoot
$env:VGGT_MODE = "train"
$env:VGGT_TRAIN_SCRIPT = "finetune_vggt_pseudo.py"

$env:VGGT_ZJU_ROOT = $ZjuRoot
$env:VGGT_SEQ_NAMES = $Seq
$env:VGGT_GEOM_SUBDIR = $GeomSubdir
if ([string]::IsNullOrWhiteSpace($CamNames)) {
    Remove-Item Env:VGGT_CAM_NAMES -ErrorAction SilentlyContinue
} else {
    $env:VGGT_CAM_NAMES = $CamNames
}

$env:VGGT_GPU_SPEC_TRAIN = $GpuSpecTrain
$env:VGGT_TRAIN_CPU = "16"
$env:VGGT_TRAIN_MEMORY_MB = "65536"
$env:VGGT_TIMEOUT_SEC = "86400"

$env:VGGT_EPOCHS = "$Epochs"
$env:VGGT_MAX_FRAMES = "$MaxFrames"
$env:VGGT_LR = "$Lr"
$env:VGGT_TF32 = "true"
$env:VGGT_AMP = "true"

$env:VGGT_FT_FREEZE_MODE = "depth_point"
$env:VGGT_FT_DEPTH_SCALE_ALIGN = "median"
$env:VGGT_FT_USE_FG_MASK = "on"
$env:VGGT_FT_FG_MASK_SOURCE = "mask"
$env:VGGT_FT_LAMBDA_DEPTH = "1.0"
$env:VGGT_FT_LAMBDA_POINT = "0.5"
$env:VGGT_FT_LAMBDA_CONF = "0.05"
$env:VGGT_FT_EARLY_STOP_PATIENCE = "2"
$env:VGGT_FT_MIN_IMPROVE = "0.00005"
$env:VGGT_FT_EVAL_EVERY_STEPS = "24"
$env:VGGT_FT_LOG_HEARTBEAT_SEC = "30"

$env:VGGT_FT_HUMAN_PRIOR_ENABLE = "on"
$env:VGGT_FT_HUMAN_PRIOR_SUBDIR = $HumanPriorSubdir
$env:VGGT_FT_HUMAN_PRIOR_STRICT = $(if ($AllowMissingHumanPrior) { "off" } else { "on" })
$env:VGGT_FT_HUMAN_PRIOR_POINT_BLEND_ALPHA = "$HumanPriorBlendAlpha"
$env:VGGT_FT_HUMAN_PRIOR_POINT_BLEND_REGION = "head_face"
$env:VGGT_FT_HUMAN_PRIOR_WEIGHT_BOOST = "$HumanPriorWeightBoost"
$env:VGGT_FT_HUMAN_PRIOR_WEIGHT_REGION = "body"
$env:VGGT_FT_HUMAN_PRIOR_COMPLETE_WEIGHT = "$HumanPriorCompleteWeight"
$env:VGGT_FT_HUMAN_PRIOR_COMPLETE_REGION = "body"
$env:VGGT_FT_LAMBDA_POINT_PRIOR = "$LambdaPointPrior"
$env:VGGT_FT_HUMAN_PRIOR_LOSS_REGION = "head_face"
$env:VGGT_FT_HUMAN_PRIOR_REGION_ERODE_PX = "0"

$env:VGGT_TRAIN_ARGS_EXTRA = ""

$summary = [ordered]@{
    repo_root = $repoRoot
    seq = $Seq
    zju_root = $ZjuRoot
    geom_subdir = $GeomSubdir
    cam_names = $CamNames
    gpu_spec_train = $GpuSpecTrain
    epochs = $Epochs
    max_frames = $MaxFrames
    lr = $Lr
    human_prior_subdir = $HumanPriorSubdir
    human_prior_strict = $env:VGGT_FT_HUMAN_PRIOR_STRICT
    human_prior_blend_alpha = $HumanPriorBlendAlpha
    human_prior_weight_boost = $HumanPriorWeightBoost
    human_prior_complete_weight = $HumanPriorCompleteWeight
    lambda_point_prior = $LambdaPointPrior
}

if (-not $AllowMissingHumanPrior) {
    $priorReady = Assert-HumanPriorReady -SeqName $Seq -MountedZjuRoot $ZjuRoot -PriorSubdir $HumanPriorSubdir
    $summary.human_prior_ready_mode = $priorReady.mode
    $summary.human_prior_ready_root = $priorReady.root
    $summary.human_prior_npz_count = $priorReady.npz_count
}

Write-Host "[modal-ft] launch summary:"
$summary.GetEnumerator() | ForEach-Object {
    Write-Host ("  {0} = {1}" -f $_.Key, $_.Value)
}

if ($DryRun) {
    Write-Host "[modal-ft] dry-run only; not submitting Modal job."
    exit 0
}

Write-Host "[modal-ft] submitting Modal job..."
modal run modal_run_train.py
