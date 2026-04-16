param(
    [string]$CompareCsv = "logs/modal_phase5/baseline_compare_latest.csv",
    [string]$OutMd = "logs/modal_phase5/lr_debug_visual_report_latest.md",
    [string]$OutCsv = "logs/modal_phase5/lr_debug_visual_latest.csv",
    [string]$BaselineLabel = "",
    [double]$MaxPSNRDrop = 0.3,
    [string]$VisSteps = "",
    [int]$VisCount = 3
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Ensure-ParentDir([string]$PathLike) {
    $root = (Resolve-Path ".").Path
    $abs = Join-Path $root $PathLike
    $dir = Split-Path -Path $abs -Parent
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
}

function Parse-StepList([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) {
        return @()
    }
    $vals = New-Object System.Collections.Generic.List[int]
    foreach ($tok in ($Raw -split "[,\s;|]+" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
        $v = 0
        if ([int]::TryParse($tok.Trim(), [ref]$v)) {
            if ($v -ge 0) {
                $vals.Add($v) | Out-Null
            }
        }
    }
    return @($vals | Select-Object -Unique)
}

function Get-StepFromName([string]$Name) {
    $m = [regex]::Match([string]$Name, "step(\d+)\.png$")
    if ($m.Success) {
        return [int]$m.Groups[1].Value
    }
    return -1
}

function Sanitize([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) { return "item" }
    return ([regex]::Replace($Raw, "[^A-Za-z0-9_.-]+", "_")).Trim("_")
}

function Pick-Visuals([string]$Label, [string]$Kind, [int[]]$StepList, [int]$Count) {
    $safe = Sanitize($Label)
    $pat = "baseline_${safe}_*_${Kind}_step*.png"
    $all = @(
        Get-ChildItem -Path "logs/modal_phase5" -File -Filter $pat -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending
    )
    if ($all.Count -eq 0) {
        return @()
    }

    $rows = @()
    foreach ($f in $all) {
        $rows += [pscustomobject]@{
            Path = $f.FullName
            Name = $f.Name
            Step = Get-StepFromName -Name $f.Name
            LastWriteTime = $f.LastWriteTime
        }
    }

    if ($StepList.Count -gt 0) {
        $picked = @()
        foreach ($s in $StepList) {
            $r = $rows | Where-Object { $_.Step -eq [int]$s } | Select-Object -First 1
            if ($r -ne $null) {
                $picked += @($r)
            }
        }
        return @($picked)
    }

    return @($rows | Select-Object -First ([Math]::Max(1, $Count)))
}

if (-not (Test-Path $CompareCsv)) {
    throw "compare csv not found: $CompareCsv"
}

$rows = @(Import-Csv $CompareCsv)
if ($rows.Count -eq 0) {
    throw "compare csv is empty: $CompareCsv"
}

$okRows = @($rows | Where-Object { $_.status -eq "ok" })
if ($okRows.Count -eq 0) {
    throw "no ok rows in compare csv: $CompareCsv"
}

$baseline = $null
if (-not [string]::IsNullOrWhiteSpace($BaselineLabel)) {
    $baseline = $okRows | Where-Object { $_.label -eq $BaselineLabel } | Select-Object -First 1
}
if ($baseline -eq $null) {
    $baseline = $okRows | Sort-Object { [double]$_.mean_PSNR } -Descending | Select-Object -First 1
}
if ($baseline -eq $null) {
    throw "failed to resolve baseline row"
}

$baselinePsnr = [double]$baseline.mean_PSNR
$steps = @(Parse-StepList -Raw $VisSteps)
$visCount = [Math]::Max(1, [int]$VisCount)

$reportRows = @()
$visRows = @()
foreach ($r in $okRows) {
    $label = [string]$r.label
    $psnr = [double]$r.mean_PSNR
    $ssim = [double]$r.mean_SSIM
    $wl1 = [double]$r.mean_weighted_L1
    $n = [int]$r.N
    $drop = $baselinePsnr - $psnr
    $pruned = ($drop -gt [double]$MaxPSNRDrop)

    $reportRows += [pscustomobject]@{
        label = $label
        geom_subdir = [string]$r.geom_subdir
        N = $n
        mean_PSNR = $psnr
        mean_SSIM = $ssim
        mean_weighted_L1 = $wl1
        baseline_label = [string]$baseline.label
        delta_psnr_vs_baseline = ($psnr - $baselinePsnr)
        psnr_drop_vs_baseline = $drop
        drop_over_threshold = [bool]$pruned
        run_url = [string]$r.run_url
        infer_out_dir = [string]$r.infer_out_dir
    }

    $kinds = @("cat_pred_tgt", "cat_fg_mask_pred_tgt", "gt_with_fg_overlay")
    foreach ($k in $kinds) {
        $picked = @(Pick-Visuals -Label $label -Kind $k -StepList $steps -Count $visCount)
        foreach ($p in $picked) {
            $visRows += [pscustomobject]@{
                label = $label
                kind = $k
                step = [int]$p.Step
                path = [string]$p.Path
                name = [string]$p.Name
                last_write = [string]$p.LastWriteTime
            }
        }
    }
}

Ensure-ParentDir -PathLike $OutCsv
$reportRows | Export-Csv $OutCsv -NoTypeInformation -Encoding UTF8

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# LR Debug Visual Report") | Out-Null
$md.Add("") | Out-Null
$md.Add("- compare_csv: " + $CompareCsv) | Out-Null
$md.Add("- baseline_label: " + [string]$baseline.label) | Out-Null
$md.Add("- baseline_psnr: " + $baselinePsnr.ToString("F6")) | Out-Null
$md.Add("- max_psnr_drop: " + [string][double]$MaxPSNRDrop) | Out-Null
$md.Add("- vis_steps: " + [string]$VisSteps) | Out-Null
$md.Add("- vis_count: " + [string]$visCount) | Out-Null
$md.Add("") | Out-Null
$md.Add("## Metrics") | Out-Null
$md.Add("") | Out-Null
$md.Add("| label | N | PSNR | SSIM | wL1 | delta_psnr_vs_baseline | drop_over_threshold |") | Out-Null
$md.Add("| --- | ---: | ---: | ---: | ---: | ---: | :---: |") | Out-Null
foreach ($r in $reportRows) {
    $md.Add("| $($r.label) | $($r.N) | $([double]$r.mean_PSNR) | $([double]$r.mean_SSIM) | $([double]$r.mean_weighted_L1) | $([double]$r.delta_psnr_vs_baseline) | $($r.drop_over_threshold) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Visuals") | Out-Null
$md.Add("") | Out-Null

foreach ($label in ($reportRows.label | Select-Object -Unique)) {
    $md.Add("### $label") | Out-Null
    $sub = @($visRows | Where-Object { $_.label -eq $label } | Sort-Object kind, step)
    if ($sub.Count -eq 0) {
        $md.Add("- no visuals found") | Out-Null
        $md.Add("") | Out-Null
        continue
    }
    foreach ($v in $sub) {
        $md.Add("- [$($v.kind) step=$($v.step)]($($v.path))") | Out-Null
    }
    $md.Add("") | Out-Null
}

Ensure-ParentDir -PathLike $OutMd
$enc = New-Object System.Text.UTF8Encoding($false)
$root = (Resolve-Path ".").Path
$absMd = Join-Path $root $OutMd
[System.IO.File]::WriteAllLines($absMd, $md, $enc)

Write-Host "[report] wrote csv: $OutCsv"
Write-Host "[report] wrote md: $OutMd"
