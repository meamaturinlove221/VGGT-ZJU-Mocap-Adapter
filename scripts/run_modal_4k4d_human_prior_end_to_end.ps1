param(
    [string]$Seq = "0012_11",
    [string]$RemoteDataRoot = "/4k4d_bridge",
    [string]$MountedZjuRoot = "/mnt/data/4k4d_bridge",
    [string]$GeomSubdir = "vggt_geom_4k4d_0012_11_96f7v_20260414",
    [string]$HumanPriorSubdir = "human_prior",
    [string]$CamNames = "",
    [int[]]$Frames = @(0),
    [string]$TargetView = "Camera_00",
    [double]$DepthUpsampleFactor = 2.0,
    [int]$PreviewTileSize = 1400,
    [int]$MaxPoints = 180000,
    [int]$PreviewPoints = 90000,
    [double]$ConfPercentile = 72.0,
    [int]$SplatRadius = 1,
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

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$runner = Join-Path $PSScriptRoot "fetch_modal_4k4d_outputs_highres.ps1"
if (-not (Test-Path $runner)) {
    throw "missing script: $runner"
}

$psArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $runner,
    "-Seq", $Seq,
    "-RemoteDataRoot", $RemoteDataRoot,
    "-MountedZjuRoot", $MountedZjuRoot,
    "-GeomSubdir", $GeomSubdir,
    "-Frames"
)
$psArgs += @($Frames | ForEach-Object { [string]$_ })
$psArgs += @(
    "-TargetView", $TargetView,
    "-DepthUpsampleFactor", [string]$DepthUpsampleFactor,
    "-PreviewTileSize", [string]$PreviewTileSize,
    "-MaxPoints", [string]$MaxPoints,
    "-PreviewPoints", [string]$PreviewPoints,
    "-ConfPercentile", [string]$ConfPercentile,
    "-SplatRadius", [string]$SplatRadius,
    "-GpuSpecTrain", $GpuSpecTrain,
    "-Epochs", [string]$Epochs,
    "-MaxFrames", [string]$MaxFrames,
    "-Lr", [string]$Lr,
    "-HumanPriorSubdir", $HumanPriorSubdir,
    "-HumanPriorBlendAlpha", [string]$HumanPriorBlendAlpha,
    "-HumanPriorWeightBoost", [string]$HumanPriorWeightBoost,
    "-HumanPriorCompleteWeight", [string]$HumanPriorCompleteWeight,
    "-LambdaPointPrior", [string]$LambdaPointPrior,
    "-RunTraining"
)
if (-not [string]::IsNullOrWhiteSpace($CamNames)) {
    $psArgs += @("-CamNames", $CamNames)
}
if ($AllowMissingHumanPrior) {
    $psArgs += "-AllowMissingHumanPrior"
}
if ($DryRun) {
    $psArgs += "-DryRun"
}

& powershell @psArgs

if ($LASTEXITCODE -ne 0) {
    throw "end-to-end modal return pipeline failed"
}
