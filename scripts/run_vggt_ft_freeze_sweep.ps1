param(
    [string]$CodeDir = "F:\vggt",
    [string]$SeqNames = "CoreView_390",
    [string]$PseudoGeomSubdir = "vggt_geom",
    [string]$PretrainedCkpt = "model.pt",
    [double]$Lr = 1e-7,
    [object]$FreezeModes = "depth_point",
    [int]$Epochs = 3,
    [int]$MaxFrames = 400,
    [int]$EvalNumSamples = 60,
    [string]$DecoderCkpt = "",
    [double]$MinPSNR = 20.9,
    [double]$MinSSIM = 0.84,
    [double]$MaxWL1 = 0.08,
    [string]$CamNames = "Camera_B1,Camera_B5,Camera_B10,Camera_B14,Camera_B19,Camera_B23",
    [double]$LambdaDepth = 1.0,
    [double]$LambdaPoint = 0.5,
    [double]$LambdaConf = 0.02,
    [double]$LambdaGeomCons = 0.05,
    [double]$LambdaCam = 0.03,
    [double]$CamRotWeight = 1.0,
    [double]$CamFovWeight = 0.2,
    [int]$CamWarmupSteps = 40,
    [double]$RobustL1Eps = 0.01,
    [double]$ConfWeightThr = 0.05,
    [double]$ConfWeightGamma = 1.0,
    [double]$PointConsTau = 0.03,
    [double]$PointConsWeightFloor = 0.2,
    [double]$PointConsClipMinQv = 1e-6,
    [double]$PointConsQuantile = 0.5,
    [string]$PointConsFocus = "inlier",
    [double]$PointResidualQuantile = 1.0,
    [string]$PointResidualFocus = "inlier",
    [string]$PointMvConsistency = "off",
    [double]$PointMvTolAbs = 0.03,
    [double]$PointMvTolRel = 0.05,
    [double]$PointMvWeightFloor = 0.2,
    [int]$PointMvStride = 2,
    [double]$PointLossScaleDepthUnproject = 0.5,
    [int]$PointWarmupSteps = 40,
    [double]$LrBackboneScale = 0.05,
    [double]$LrHeadScale = 1.0,
    [double]$LrCameraScale = 0.1,
    [double]$GradClip = 0.5,
    [string]$DepthScaleAlign = "median",
    [int]$EarlyStopPatience = 1,
    [double]$MinImprove = 0.0001,
    [int]$MaxStepsPerEpoch = 80,
    [switch]$RunSupSweep,
    [object]$SupDepthScaleAlignList = "off,median",
    [object]$SupLambdaConfList = "0.02,0.01"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Sanitize([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) { return "item" }
    return ([regex]::Replace($Raw, "[^A-Za-z0-9_.-]+", "_")).Trim("_")
}

function Split-List([object]$Raw) {
    if ($null -eq $Raw) {
        return @()
    }
    $parts = @()
    if ($Raw -is [System.Array]) {
        $parts = @($Raw)
    } else {
        $parts = @($Raw)
    }
    $out = @()
    foreach ($p in $parts) {
        $s = [string]$p
        if ([string]::IsNullOrWhiteSpace($s)) { continue }
        $out += @(
            $s -split '[,\s]+' |
                Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
                ForEach-Object { $_.Trim() }
        )
    }
    return @($out)
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

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$freezeModes = Split-List -Raw $FreezeModes
if ($freezeModes.Count -eq 0) {
    throw "FreezeModes is empty: $FreezeModes"
}

$nonDual = @($freezeModes | Where-Object { $_ -ne "depth_point" })
if ($nonDual.Count -gt 0) {
    Write-Host "[freeze-sweep] warning: non-dual freeze modes detected: $($nonDual -join ',')"
    Write-Host "[freeze-sweep] mentor-aligned recommendation: keep FreezeModes=depth_point so depth+point losses are both active."
}

$rows = @()
$freezeInfos = @()

foreach ($mode in $freezeModes) {
    $safeMode = Sanitize($mode)
    $label = "freeze_$safeMode"
    $ftCkptDir = "/mnt/out/vggt/finetune/freeze_$safeMode`_$timestamp/ckpt"
    $ftLogDir = "/mnt/out/vggt/finetune/freeze_$safeMode`_$timestamp/logs"
    $ftModelPath = "$ftCkptDir/model_ft_zju.pt"
    $geomOut = "vggt_geom_ft_freeze_$safeMode`_$timestamp"

    $argsExtra = @(
        "--epochs=$Epochs",
        "--max_frames=$MaxFrames",
        "--lr=$Lr",
        "--freeze_mode=$mode",
        "--depth_scale_align=$DepthScaleAlign",
        "--lambda_depth=$LambdaDepth",
        "--lambda_point=$LambdaPoint",
        "--lambda_conf=$LambdaConf",
        "--lambda_geom_cons=$LambdaGeomCons",
        "--lambda_cam=$LambdaCam",
        "--cam_rot_weight=$CamRotWeight",
        "--cam_fov_weight=$CamFovWeight",
        "--cam_warmup_steps=$CamWarmupSteps",
        "--robust_l1_eps=$RobustL1Eps",
        "--conf_weight_thr=$ConfWeightThr",
        "--conf_weight_gamma=$ConfWeightGamma",
            "--point_cons_tau=$PointConsTau",
            "--point_cons_weight_floor=$PointConsWeightFloor",
            "--point_cons_clip_min_qv=$PointConsClipMinQv",
            "--point_cons_quantile=$PointConsQuantile",
            "--point_cons_focus=$PointConsFocus",
            "--point_residual_quantile=$PointResidualQuantile",
            "--point_residual_focus=$PointResidualFocus",
            "--point_mv_consistency=$PointMvConsistency",
            "--point_mv_tol_abs=$PointMvTolAbs",
            "--point_mv_tol_rel=$PointMvTolRel",
            "--point_mv_weight_floor=$PointMvWeightFloor",
            "--point_mv_stride=$PointMvStride",
            "--point_loss_scale_depth_unproject=$PointLossScaleDepthUnproject",
        "--point_warmup_steps=$PointWarmupSteps",
        "--lr_backbone_scale=$LrBackboneScale",
        "--lr_head_scale=$LrHeadScale",
        "--lr_camera_scale=$LrCameraScale",
        "--grad_clip=$GradClip",
        "--geom_subdir=$PseudoGeomSubdir",
        "--log_dir=$ftLogDir",
        "--ckpt_dir=$ftCkptDir",
        "--eval_every_steps=20",
        "--early_stop_patience=$EarlyStopPatience",
        "--min_improve=$MinImprove",
        "--max_steps_per_epoch=$MaxStepsPerEpoch"
    ) -join " "

    Write-Host "[freeze-sweep] finetune mode=$mode"
    $env:VGGT_CODE_DIR = $CodeDir
    $env:VGGT_MODE = "precompute"
    $env:VGGT_SEQ_NAMES = $SeqNames
    $env:VGGT_GEOM_SUBDIR = $PseudoGeomSubdir
    $env:VGGT_CAM_NAMES = $CamNames
    $env:VGGT_MAX_FRAMES = [string]$MaxFrames
    $env:VGGT_PRECOMPUTE_SCRIPT = "finetune_vggt_pseudo.py"
    $env:VGGT_PRECOMPUTE_CKPT = $PretrainedCkpt
    $env:VGGT_PRECOMPUTE_ARGS_EXTRA = $argsExtra
    Remove-Item Env:VGGT_PROFILE -ErrorAction SilentlyContinue

    $ftRun = Invoke-ModalRun -ScriptPath "modal_run_train.py"
    $ftOutput = @($ftRun.Output)
    $ftRc = [int]$ftRun.ExitCode
    $ftLogLocal = "logs/modal_phase5/vggt_ft_freeze_${safeMode}_$timestamp.finetune.log"
    $ftOutput | Tee-Object -FilePath $ftLogLocal | Out-Null
    if ($ftRc -ne 0) {
        $rows += [pscustomobject]@{
            stage = "freeze"
            label = $label
            lr = $Lr
            freeze_mode = $mode
            depth_scale_align = $DepthScaleAlign
            lambda_conf = $LambdaConf
            geom_subdir = ""
            ft_ckpt = $ftModelPath
            status = "error"
            reason = "finetune failed"
        }
        continue
    }

    $env:VGGT_MODE = "precompute"
    $env:VGGT_PRECOMPUTE_SCRIPT = "precompute_zju_vggt_geom.py"
    $env:VGGT_PRECOMPUTE_CKPT = $ftModelPath
    $env:VGGT_GEOM_SUBDIR = $geomOut
    $env:VGGT_MAX_FRAMES = [string]$MaxFrames
    $env:VGGT_PRECOMPUTE_ARGS_EXTRA = ""
    $pcRun = Invoke-ModalRun -ScriptPath "modal_run_train.py"
    $pcOutput = @($pcRun.Output)
    $pcRc = [int]$pcRun.ExitCode
    $pcLogLocal = "logs/modal_phase5/vggt_ft_freeze_${safeMode}_$timestamp.precompute.log"
    $pcOutput | Tee-Object -FilePath $pcLogLocal | Out-Null
    if ($pcRc -ne 0) {
        $rows += [pscustomobject]@{
            stage = "freeze"
            label = $label
            lr = $Lr
            freeze_mode = $mode
            depth_scale_align = $DepthScaleAlign
            lambda_conf = $LambdaConf
            geom_subdir = $geomOut
            ft_ckpt = $ftModelPath
            status = "error"
            reason = "precompute failed"
        }
        continue
    }

    $freezeInfos += [pscustomobject]@{
        label = $label
        lr = $Lr
        freeze_mode = $mode
        geom_subdir = $geomOut
        ft_ckpt = $ftModelPath
    }
    $rows += [pscustomobject]@{
        stage = "freeze"
        label = $label
        lr = $Lr
        freeze_mode = $mode
        depth_scale_align = $DepthScaleAlign
        lambda_conf = $LambdaConf
        geom_subdir = $geomOut
        ft_ckpt = $ftModelPath
        status = "ok"
        reason = ""
    }
}

if ($freezeInfos.Count -eq 0) {
    $csvNow = "logs/modal_phase5/vggt_ft_sweep_$timestamp.csv"
    $csvLatest = "logs/modal_phase5/vggt_ft_sweep_latest.csv"
    $rows | Export-Csv $csvNow -NoTypeInformation -Encoding UTF8
    $rows | Export-Csv $csvLatest -NoTypeInformation -Encoding UTF8
    throw "No freeze candidate succeeded."
}

$candStr = ($freezeInfos | ForEach-Object { "$($_.geom_subdir):$($_.label)" }) -join ";"
Write-Host "[freeze-sweep] evaluate freeze candidates..."
$evalArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts/eval_geom_candidates.ps1",
    "-CodeDir", $CodeDir,
    "-SeqNames", $SeqNames,
    "-GeomCandidates", $candStr,
    "-NumSamples", [string]$EvalNumSamples,
    "-MinPSNR", [string]$MinPSNR,
    "-MinSSIM", [string]$MinSSIM,
    "-MaxWL1", [string]$MaxWL1,
    "-OutTag", "freeze_sweep_$timestamp"
)
if (-not [string]::IsNullOrWhiteSpace($DecoderCkpt)) {
    $evalArgs += @("-DecoderCkpt", $DecoderCkpt)
}
& powershell @evalArgs

$bestInfo = $null
$evalCsv = "logs/modal_phase5/baseline_compare_latest.csv"
if (Test-Path $evalCsv) {
    $evalRows = Import-Csv $evalCsv
    $okRows = @($evalRows | Where-Object { $_.status -eq "ok" -and $_.pass -eq "True" })
    if ($okRows.Count -gt 0) {
        $best = $okRows | Sort-Object { [double]$_.mean_PSNR } -Descending | Select-Object -First 1
        if ($best -ne $null) {
            $bestInfo = $freezeInfos | Where-Object { $_.label -eq [string]$best.label } | Select-Object -First 1
        }
    }
}

if ($RunSupSweep -and ($bestInfo -ne $null)) {
    $supDepthModes = Split-List -Raw $SupDepthScaleAlignList
    $supLambdaConfs = Split-List -Raw $SupLambdaConfList
    $supInfos = @()

    foreach ($dsa in $supDepthModes) {
        foreach ($lc in $supLambdaConfs) {
            $safeDsa = Sanitize($dsa)
            $safeLc = Sanitize($lc)
            $label = "sup_${safeDsa}_lc_${safeLc}"
            $ftCkptDir = "/mnt/out/vggt/finetune/sup_${safeDsa}_$safeLc`_$timestamp/ckpt"
            $ftLogDir = "/mnt/out/vggt/finetune/sup_${safeDsa}_$safeLc`_$timestamp/logs"
            $ftModelPath = "$ftCkptDir/model_ft_zju.pt"
            $geomOut = "vggt_geom_ft_sup_${safeDsa}_$safeLc`_$timestamp"

            $argsExtra = @(
                "--epochs=$Epochs",
                "--max_frames=$MaxFrames",
                "--lr=$Lr",
                "--freeze_mode=$($bestInfo.freeze_mode)",
                "--depth_scale_align=$dsa",
                "--lambda_depth=$LambdaDepth",
                "--lambda_point=$LambdaPoint",
                "--lambda_conf=$lc",
                "--lambda_geom_cons=$LambdaGeomCons",
                "--lambda_cam=$LambdaCam",
                "--cam_rot_weight=$CamRotWeight",
                "--cam_fov_weight=$CamFovWeight",
                "--cam_warmup_steps=$CamWarmupSteps",
                "--robust_l1_eps=$RobustL1Eps",
                "--conf_weight_thr=$ConfWeightThr",
                "--conf_weight_gamma=$ConfWeightGamma",
                    "--point_cons_tau=$PointConsTau",
                    "--point_cons_weight_floor=$PointConsWeightFloor",
                    "--point_cons_clip_min_qv=$PointConsClipMinQv",
                    "--point_cons_quantile=$PointConsQuantile",
                    "--point_cons_focus=$PointConsFocus",
                    "--point_residual_quantile=$PointResidualQuantile",
                    "--point_residual_focus=$PointResidualFocus",
                    "--point_loss_scale_depth_unproject=$PointLossScaleDepthUnproject",
                "--point_warmup_steps=$PointWarmupSteps",
                "--lr_backbone_scale=$LrBackboneScale",
                "--lr_head_scale=$LrHeadScale",
                "--lr_camera_scale=$LrCameraScale",
                "--grad_clip=$GradClip",
                "--geom_subdir=$PseudoGeomSubdir",
                "--resume_ckpt=$($bestInfo.ft_ckpt)",
                "--log_dir=$ftLogDir",
                "--ckpt_dir=$ftCkptDir",
                "--eval_every_steps=20",
                "--early_stop_patience=$EarlyStopPatience",
                "--min_improve=$MinImprove",
                "--max_steps_per_epoch=$MaxStepsPerEpoch"
            ) -join " "

            $env:VGGT_CODE_DIR = $CodeDir
            $env:VGGT_MODE = "precompute"
            $env:VGGT_SEQ_NAMES = $SeqNames
            $env:VGGT_GEOM_SUBDIR = $PseudoGeomSubdir
            $env:VGGT_CAM_NAMES = $CamNames
            $env:VGGT_MAX_FRAMES = [string]$MaxFrames
            $env:VGGT_PRECOMPUTE_SCRIPT = "finetune_vggt_pseudo.py"
            $env:VGGT_PRECOMPUTE_CKPT = $PretrainedCkpt
            $env:VGGT_PRECOMPUTE_ARGS_EXTRA = $argsExtra
            Remove-Item Env:VGGT_PROFILE -ErrorAction SilentlyContinue

            $ftRun = Invoke-ModalRun -ScriptPath "modal_run_train.py"
            $ftOutput = @($ftRun.Output)
            $ftRc = [int]$ftRun.ExitCode
            $ftLogLocal = "logs/modal_phase5/vggt_ft_sup_${safeDsa}_$safeLc`_$timestamp.finetune.log"
            $ftOutput | Tee-Object -FilePath $ftLogLocal | Out-Null
            if ($ftRc -ne 0) {
                $rows += [pscustomobject]@{
                    stage = "sup"
                    label = $label
                    lr = $Lr
                    freeze_mode = $bestInfo.freeze_mode
                    depth_scale_align = $dsa
                    lambda_conf = $lc
                    geom_subdir = ""
                    ft_ckpt = $ftModelPath
                    status = "error"
                    reason = "sup finetune failed"
                }
                continue
            }

            $env:VGGT_MODE = "precompute"
            $env:VGGT_PRECOMPUTE_SCRIPT = "precompute_zju_vggt_geom.py"
            $env:VGGT_PRECOMPUTE_CKPT = $ftModelPath
            $env:VGGT_GEOM_SUBDIR = $geomOut
            $env:VGGT_MAX_FRAMES = [string]$MaxFrames
            $env:VGGT_PRECOMPUTE_ARGS_EXTRA = ""
            $pcRun = Invoke-ModalRun -ScriptPath "modal_run_train.py"
            $pcOutput = @($pcRun.Output)
            $pcRc = [int]$pcRun.ExitCode
            $pcLogLocal = "logs/modal_phase5/vggt_ft_sup_${safeDsa}_$safeLc`_$timestamp.precompute.log"
            $pcOutput | Tee-Object -FilePath $pcLogLocal | Out-Null
            if ($pcRc -ne 0) {
                $rows += [pscustomobject]@{
                    stage = "sup"
                    label = $label
                    lr = $Lr
                    freeze_mode = $bestInfo.freeze_mode
                    depth_scale_align = $dsa
                    lambda_conf = $lc
                    geom_subdir = $geomOut
                    ft_ckpt = $ftModelPath
                    status = "error"
                    reason = "sup precompute failed"
                }
                continue
            }

            $supInfos += [pscustomobject]@{
                geom_subdir = $geomOut
                label = $label
            }
            $rows += [pscustomobject]@{
                stage = "sup"
                label = $label
                lr = $Lr
                freeze_mode = $bestInfo.freeze_mode
                depth_scale_align = $dsa
                lambda_conf = $lc
                geom_subdir = $geomOut
                ft_ckpt = $ftModelPath
                status = "ok"
                reason = ""
            }
        }
    }

    if ($supInfos.Count -gt 0) {
        $supCandStr = ($supInfos | ForEach-Object { "$($_.geom_subdir):$($_.label)" }) -join ";"
        $evalArgs2 = @(
            "-ExecutionPolicy", "Bypass",
            "-File", "scripts/eval_geom_candidates.ps1",
            "-CodeDir", $CodeDir,
            "-SeqNames", $SeqNames,
            "-GeomCandidates", $supCandStr,
            "-NumSamples", [string]$EvalNumSamples,
            "-MinPSNR", [string]$MinPSNR,
            "-MinSSIM", [string]$MinSSIM,
            "-MaxWL1", [string]$MaxWL1,
            "-OutTag", "sup_sweep_$timestamp"
        )
        if (-not [string]::IsNullOrWhiteSpace($DecoderCkpt)) {
            $evalArgs2 += @("-DecoderCkpt", $DecoderCkpt)
        }
        & powershell @evalArgs2
    }
}

$csvNow = "logs/modal_phase5/vggt_ft_sweep_$timestamp.csv"
$csvLatest = "logs/modal_phase5/vggt_ft_sweep_latest.csv"
$rows | Export-Csv $csvNow -NoTypeInformation -Encoding UTF8
$rows | Export-Csv $csvLatest -NoTypeInformation -Encoding UTF8
Write-Host "[freeze-sweep] wrote: $csvLatest"
exit 0
