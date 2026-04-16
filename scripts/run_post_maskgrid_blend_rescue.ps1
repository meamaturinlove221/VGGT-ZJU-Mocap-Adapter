param(
    [string]$CodeDir = "F:\vggt",
    [string]$MaskgridMetaPath = "logs/modal_phase5/post_upstream_maskgrid_latest.json",
    [int]$WaitMetaTimeoutSec = 21600,
    [int]$PollSec = 20,
    [string]$SeqNames = "CoreView_390",
    [string]$PseudoGeomSubdir = "vggt_geom",
    [string]$Lr = "1e-6",
    [string]$AlphaList = "0.55,0.70,0.85",
    [int]$TopKCkpt = 2,
    [int]$EpochsShort = 1,
    [int]$MaxFramesShort = 400,
    [int]$MaxStepsPerEpoch = 80,
    [int]$EvalNumSamples = 40,
    [string]$UnprojectImpl = "upstream433",
    [string]$PointmapSource = "point_head",
    [string]$PrecomputeMvSupportOn = "on",
    [int]$ModalRunTimeoutSec = 3600,
    [int]$ModalRunNoOutputTimeoutSec = 600,
    [int]$ModalRunNoOutputTimeoutSecPointHead = 600
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $CodeDir

function Parse-Tokens([string]$Raw) {
    return @(
        $Raw -split "[,\s;|]+" |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_.Trim() }
    )
}

function To-DoubleOrNaN($x) {
    try {
        if ($null -eq $x -or [string]::IsNullOrWhiteSpace([string]$x)) { return [double]::NaN }
        return [double]$x
    } catch {
        return [double]::NaN
    }
}

function Read-JsonSafe([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try {
        return (Get-Content $Path -Raw | ConvertFrom-Json)
    } catch {
        return $null
    }
}

$deadline = (Get-Date).AddSeconds([Math]::Max(60, [int]$WaitMetaTimeoutSec))
$maskgridPid = 0
$meta = $null
while ((Get-Date) -lt $deadline) {
    $meta = Read-JsonSafe -Path $MaskgridMetaPath
    if ($meta -ne $null -and $meta.PSObject.Properties["proc_pid"]) {
        try { $maskgridPid = [int]$meta.proc_pid } catch { $maskgridPid = 0 }
        if ($maskgridPid -gt 0) {
            Write-Host "[blend-rescue] found upstream pid=$maskgridPid from $MaskgridMetaPath"
            break
        }
    }
    Write-Host "[blend-rescue] waiting upstream pid from $MaskgridMetaPath ..."
    Start-Sleep -Seconds ([Math]::Max(3, [int]$PollSec))
}

if ($maskgridPid -gt 0) {
    try {
        Write-Host "[blend-rescue] waiting upstream pid=$maskgridPid to finish ..."
        Wait-Process -Id $maskgridPid -ErrorAction SilentlyContinue
    } catch {
    }
}

$latestSweep = "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv"
if (-not (Test-Path $latestSweep)) {
    throw "missing $latestSweep (cannot run blend rescue)"
}

$baseRows = @(
    Import-Csv $latestSweep |
        Where-Object { [int]$_.exit_code -eq 0 -and -not [string]::IsNullOrWhiteSpace([string]$_.ghost_score_mean) } |
        Sort-Object {
            To-DoubleOrNaN($_.ghost_score_mean)
        }, {
            -1.0 * (To-DoubleOrNaN($_.mean_PSNR))
        } |
        Select-Object -First ([Math]::Max(1, [int]$TopKCkpt))
)
if ($baseRows.Count -le 0) {
    throw "no valid row in $latestSweep"
}

$alphas = Parse-Tokens -Raw $AlphaList
if ($alphas.Count -eq 0) {
    throw "AlphaList is empty"
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$rows = @()
$baseCandidates = @()

$baseRank = 0
foreach ($baseRow in $baseRows) {
    $baseRank += 1
    $baseMvDepth = [string]$baseRow.lambda_point_mv_depth
    $baseMvMask = [string]$baseRow.lambda_point_mv_mask
    $baseSweepCsv = [string]$baseRow.sweep_csv
    $baseResumeCkpt = ""
    if (Test-Path $baseSweepCsv) {
        $sr = @(
            Import-Csv $baseSweepCsv |
                Where-Object { $_.status -eq "ok" -and $_.stage -eq "short" } |
                Select-Object -First 1
        )
        if ($sr.Count -gt 0) {
            $baseResumeCkpt = [string]$sr[0].ft_ckpt
        }
    }
    $baseCandidates += [pscustomobject]@{
        base_rank = $baseRank
        lambda_point_mv_depth = $baseMvDepth
        lambda_point_mv_mask = $baseMvMask
        base_ghost_score_mean = To-DoubleOrNaN($baseRow.ghost_score_mean)
        base_mean_PSNR = To-DoubleOrNaN($baseRow.mean_PSNR)
        base_resume_ckpt = $baseResumeCkpt
        base_sweep_csv = $baseSweepCsv
    }

    foreach ($alphaRaw in $alphas) {
        $alpha = [double]$alphaRaw
        $alphaTag = ("{0:F2}" -f $alpha).Replace(".", "p")

        $callArgs = @(
            "-CodeDir", $CodeDir,
            "-SeqNames", $SeqNames,
            "-PseudoGeomSubdir", $PseudoGeomSubdir,
            "-Lr", $Lr,
            "-LambdaPointMvDepthList", $baseMvDepth,
            "-LambdaPointMvMaskList", $baseMvMask,
            "-EpochsShort", [string]$EpochsShort,
            "-MaxFramesShort", [string]$MaxFramesShort,
            "-MaxStepsPerEpoch", [string]$MaxStepsPerEpoch,
            "-EvalNumSamples", [string]$EvalNumSamples,
            "-PointTargetMode", "blend",
            "-PointTargetBlendAlpha", [string]$alpha,
            "-PointTargetBlendAlphaMin", [string]$alpha,
            "-PointTargetBlendAlphaMax", [string]$alpha,
            "-PointTargetConsensusAlphaFloor", "0.0",
            "-PointTargetBlendRelGain", "0.0",
            "-PointTargetBlendMvGain", "0.0",
            "-PointTargetBlendByReliability", "off",
            "-PointTargetBlendByMvSupport", "off",
            "-PointTargetBlendMvPolicy", "strong_to_depth",
            "-PointSupportMode", "inverse",
            "-PointSupportFloor", "0.2",
            "-PointMvDepthSupportMode", "inverse",
            "-PointMvDepthSupportFloor", "0.2",
            "-PointmapSource", $PointmapSource,
            "-UnprojectImpl", $UnprojectImpl,
            "-PrecomputeMvSupportOn", $PrecomputeMvSupportOn,
            "-NoImprovePatience", "99",
            "-MinGhostImprove", "0.0",
            "-ModalRunTimeoutSec", [string]$ModalRunTimeoutSec,
            "-ModalRunNoOutputTimeoutSec", [string]$ModalRunNoOutputTimeoutSec,
            "-ModalRunNoOutputTimeoutSecPointHead", [string]$ModalRunNoOutputTimeoutSecPointHead,
            "-NoOutputProbeTimeoutSec", [string]([Math]::Max(300, [int]$ModalRunNoOutputTimeoutSec + 60)),
            "-NoOutputProbeTimeoutSecPointHeadWeak", [string]([Math]::Max(300, [int]$ModalRunNoOutputTimeoutSec + 60)),
            "-LaneId", "lane_b",
            "-CandidateFamily", "stage2_post_rescue",
            "-GuardTier", "exploration"
        )
        if (-not [string]::IsNullOrWhiteSpace($baseResumeCkpt)) {
            $callArgs += @("-ResumeCkpt", $baseResumeCkpt)
        }

        & "$CodeDir\scripts\run_vggt_ghost_mvdepth_sweep.ps1" @callArgs
        <# Equivalent expanded args:
           -PointTargetMode blend + fixed alpha, with inverse support to focus low-support ghost regions.
        #>
        $rc = [int]$LASTEXITCODE

        $sweepLatest = "logs/modal_phase5/ghost_mvdepth_sweep_latest.csv"
        $cmpLatest = "logs/modal_phase5/baseline_compare_latest.csv"
        $sweepOut = "logs/modal_phase5/ghost_mvdepth_blend_rank${baseRank}_alpha_${alphaTag}_$ts.csv"
        $cmpOut = "logs/modal_phase5/baseline_compare_blend_rank${baseRank}_alpha_${alphaTag}_$ts.csv"
        if (Test-Path $sweepLatest) { Copy-Item $sweepLatest $sweepOut -Force }
        if (Test-Path $cmpLatest) { Copy-Item $cmpLatest $cmpOut -Force }

        $cand = @()
        if (Test-Path $sweepLatest) {
            $cand = @(
                Import-Csv $sweepLatest |
                    Where-Object { [int]$_.exit_code -eq 0 -and -not [string]::IsNullOrWhiteSpace([string]$_.ghost_score_mean) } |
                    Sort-Object {
                        To-DoubleOrNaN($_.ghost_score_mean)
                    }, {
                        -1.0 * (To-DoubleOrNaN($_.mean_PSNR))
                    } |
                    Select-Object -First 1
            )
        }

        $ghost = [double]::NaN
        $ghostSoft = [double]::NaN
        $psnr = [double]::NaN
        $ssim = [double]::NaN
        $wl1 = [double]::NaN
        if ($cand.Count -gt 0) {
            $ghost = To-DoubleOrNaN($cand[0].ghost_score_mean)
            $ghostSoft = To-DoubleOrNaN($cand[0].ghost_soft_score)
            $psnr = To-DoubleOrNaN($cand[0].mean_PSNR)
            $ssim = To-DoubleOrNaN($cand[0].mean_SSIM)
            $wl1 = To-DoubleOrNaN($cand[0].mean_weighted_L1)
        }

        $rows += [pscustomobject]@{
            base_rank = $baseRank
            alpha = $alpha
            exit_code = $rc
            ghost_score_mean = $ghost
            ghost_soft_score = $ghostSoft
            mean_PSNR = $psnr
            mean_SSIM = $ssim
            mean_weighted_L1 = $wl1
            lambda_point_mv_depth = $baseMvDepth
            lambda_point_mv_mask = $baseMvMask
            resume_ckpt = $baseResumeCkpt
            sweep_csv = $sweepOut
            baseline_compare_csv = $cmpOut
            base_ghost_score_mean = To-DoubleOrNaN($baseRow.ghost_score_mean)
            base_mean_PSNR = To-DoubleOrNaN($baseRow.mean_PSNR)
            base_sweep_csv = $baseSweepCsv
        }
    }
}

$outCsv = "logs/modal_phase5/blend_rescue_sweep_$ts.csv"
$outLatest = "logs/modal_phase5/blend_rescue_sweep_latest.csv"
$rows | Export-Csv $outCsv -NoTypeInformation -Encoding UTF8
$rows | Export-Csv $outLatest -NoTypeInformation -Encoding UTF8

$bestBlend = @(
    $rows |
        Where-Object { [int]$_.exit_code -eq 0 -and -not [double]::IsNaN([double]$_.ghost_score_mean) } |
        Sort-Object {
            [double]$_.ghost_score_mean
        }, {
            -1.0 * [double]$_.mean_PSNR
        } |
        Select-Object -First 1
)

$gate = [ordered]@{
    generated_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    maskgrid_meta_path = $MaskgridMetaPath
    maskgrid_pid = $maskgridPid
    base_candidates = $baseCandidates
    rows = $rows
    best = $(if ($bestBlend.Count -gt 0) { $bestBlend[0] } else { $null })
}
$gate | ConvertTo-Json -Depth 8 | Set-Content -Path "logs/modal_phase5/blend_rescue_gate_latest.json" -Encoding UTF8

$outMd = "logs/modal_phase5/blend_rescue_sweep_latest.md"
$lines = @("# Blend Rescue Sweep (Latest)", "")
foreach ($r in $rows) {
    $lines += "- alpha=$($r.alpha), rc=$($r.exit_code), ghost=$($r.ghost_score_mean), PSNR=$($r.mean_PSNR), SSIM=$($r.mean_SSIM), wL1=$($r.mean_weighted_L1)"
}
Set-Content -Path $outMd -Value ($lines -join "`n") -Encoding UTF8

if ((@($rows | Where-Object { [int]$_.exit_code -ne 0 }).Count) -gt 0) {
    exit 2
}
exit 0
