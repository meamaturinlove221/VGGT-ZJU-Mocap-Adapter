param(
    [string]$Seq = "0012_11",
    [string]$BridgeRoot = "",
    [string]$GeomSubdir = "vggt_geom_4k4d_0012_11_96f7v_20260414",
    [string]$RemoteDataRoot = "/4k4d_bridge",
    [string]$HumanPriorSubdir = "human_prior",
    [int[]]$VerifyFrames = @(0, 47, 95),
    [int]$DownloadTimeoutSecPerFile = 240,
    [int]$DownloadRetries = 3,
    [int]$UploadRetries = 3,
    [switch]$SkipDownload,
    [switch]$SkipExport,
    [switch]$SkipVerify,
    [switch]$SkipUpload,
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

function Test-NpzReadable {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $false
    }
    & python -c "import numpy as np,sys; data=np.load(sys.argv[1], allow_pickle=True); data.close()" $Path | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Wait-NpzReadable {
    param(
        [string]$Path,
        [int]$TimeoutSec = 120
    )
    $deadline = (Get-Date).AddSeconds([Math]::Max(5, [int]$TimeoutSec))
    while ((Get-Date) -lt $deadline) {
        if (Test-NpzReadable -Path $Path) {
            return $true
        }
        Start-Sleep -Seconds 3
    }
    return (Test-NpzReadable -Path $Path)
}

function Get-RemoteFileEntries {
    param(
        [string]$VolumeName,
        [string]$RemoteDir
    )
    $items = @(Invoke-ModalJson -CliArgs @("volume", "ls", "--json", $VolumeName, $RemoteDir))
    return @(
        $items |
        Where-Object { [string]$_.Type -eq "file" } |
        Sort-Object { [string]$_.Filename } |
        ForEach-Object {
            [ordered]@{
                remote = "/" + ([string]$_.Filename).TrimStart("/")
                leaf = Split-Path ([string]$_.Filename) -Leaf
            }
        }
    )
}

function Sync-ModalFiles {
    param(
        [string]$VolumeName,
        [string]$RemoteDir,
        [string]$LocalDir,
        [int]$TimeoutSecPerFile,
        [int]$Retries
    )
    if ((Test-Path $LocalDir) -and -not (Get-Item $LocalDir).PSIsContainer) {
        Remove-Item -LiteralPath $LocalDir -Force
    }
    New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
    $files = @(Get-RemoteFileEntries -VolumeName $VolumeName -RemoteDir $RemoteDir)
    $done = 0
    foreach ($entry in $files) {
        $localPath = Join-Path $LocalDir $entry.leaf
        if (Test-NpzReadable -Path $localPath) {
            $done += 1
            continue
        }
        $ok = $false
        for ($attempt = 1; $attempt -le $Retries; $attempt++) {
            if (Test-Path $localPath) {
                Remove-Item -LiteralPath $localPath -Force -ErrorAction SilentlyContinue
            }
            $proc = Start-Process -FilePath "modal" -ArgumentList @("volume", "get", $VolumeName, $entry.remote, $localPath, "--force") -PassThru -WindowStyle Hidden
            $deadline = (Get-Date).AddSeconds([Math]::Max(30, [int]$TimeoutSecPerFile))
            $lastLen = -1L
            $stableRounds = 0
            while ((Get-Date) -lt $deadline) {
                Start-Sleep -Seconds 2
                if (Test-Path $localPath) {
                    $len = (Get-Item $localPath).Length
                    if (($len -eq $lastLen) -and ($len -gt 0)) {
                        $stableRounds += 1
                    } else {
                        $stableRounds = 0
                        $lastLen = $len
                    }
                    if (($stableRounds -ge 2) -and (Test-NpzReadable -Path $localPath)) {
                        $ok = $true
                        break
                    }
                }
                if ($proc.HasExited) {
                    if (Test-NpzReadable -Path $localPath) {
                        $ok = $true
                    }
                    break
                }
            }
            if (-not $proc.HasExited) {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            }
            if (-not $ok -and (Test-Path $localPath)) {
                $ok = Wait-NpzReadable -Path $localPath -TimeoutSec 120
            }
            if ($ok) {
                break
            }
        }
        if (-not $ok) {
            throw "failed to sync remote file: $($entry.remote)"
        }
        $done += 1
        if (($done % 8) -eq 0) {
            Write-Host "[human-prior-case] synced $done / $($files.Count) geom files"
        }
    }
    return $files.Count
}

function Upload-ModalFiles {
    param(
        [string]$VolumeName,
        [string]$LocalDir,
        [string]$RemoteDir,
        [int]$Retries
    )
    $files = @(Get-ChildItem -Path $LocalDir -Filter *.npz -File | Sort-Object Name)
    $done = 0
    foreach ($file in $files) {
        $ok = $false
        for ($attempt = 1; $attempt -le $Retries; $attempt++) {
            & modal volume put $VolumeName $file.FullName ($RemoteDir.TrimEnd("/") + "/") --force | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $ok = $true
                break
            }
        }
        if (-not $ok) {
            throw "failed to upload sidecar: $($file.FullName)"
        }
        $done += 1
        if (($done % 16) -eq 0) {
            Write-Host "[human-prior-case] uploaded $done / $($files.Count) sidecars"
        }
    }
    return $files.Count
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot
Initialize-ModalCliEnvironment

if ([string]::IsNullOrWhiteSpace($BridgeRoot)) {
    $BridgeRoot = Join-Path $repoRoot "out_vis\\bridge_4k4d_med96\\$Seq"
}
$bridgeRootResolved = (Resolve-Path $BridgeRoot).Path
$geomRoot = Join-Path $bridgeRootResolved $GeomSubdir
$remoteGeomDir = ($RemoteDataRoot.TrimEnd("/") + "/" + $Seq + "/" + $GeomSubdir).Replace("//", "/")
$localPriorRoot = Join-Path $bridgeRootResolved $HumanPriorSubdir
$remotePriorDir = ($RemoteDataRoot.TrimEnd("/") + "/" + $Seq + "/" + $HumanPriorSubdir).Replace("//", "/")
$manifestPath = Join-Path $localPriorRoot "prepare_modal_4k4d_human_prior_case_summary.json"

$summary = [ordered]@{
    seq = $Seq
    bridge_root = $bridgeRootResolved
    geom_root = $geomRoot
    remote_geom_dir = $remoteGeomDir
    human_prior_root = $localPriorRoot
    remote_prior_dir = $remotePriorDir
    verify_frames = @($VerifyFrames)
    skip_download = [bool]$SkipDownload
    skip_export = [bool]$SkipExport
    skip_verify = [bool]$SkipVerify
    skip_upload = [bool]$SkipUpload
}

if ($DryRun) {
    Write-Host "[human-prior-case] dry-run summary:"
    $summary.GetEnumerator() | ForEach-Object {
        Write-Host ("  {0} = {1}" -f $_.Key, $_.Value)
    }
    exit 0
}

if (-not $SkipDownload) {
    $geomCount = Sync-ModalFiles `
        -VolumeName "vggt-zju-data" `
        -RemoteDir $remoteGeomDir `
        -LocalDir $geomRoot `
        -TimeoutSecPerFile $DownloadTimeoutSecPerFile `
        -Retries $DownloadRetries
    $summary.geom_file_count = [int]$geomCount
}

if (-not $SkipExport) {
    & python .\tools\export_4k4d_human_prior.py `
        --seq-root $bridgeRootResolved `
        --geom-root $geomRoot `
        --output-subdir $HumanPriorSubdir `
        --overwrite
    if ($LASTEXITCODE -ne 0) {
        throw "human_prior export failed"
    }
}

if (-not $SkipVerify) {
    $verifyArgs = @(
        ".\tools\check_4k4d_human_prior_loss.py",
        "--seq-root", $bridgeRootResolved,
        "--geom-root", $geomRoot,
        "--human-prior-subdir", $HumanPriorSubdir
    )
    if ($VerifyFrames.Count -gt 0) {
        $verifyArgs += "--frames"
        $verifyArgs += @($VerifyFrames | ForEach-Object { [string]$_ })
    }
    & python @verifyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "human_prior local verification failed"
    }
}

if (-not $SkipUpload) {
    $priorCount = Upload-ModalFiles `
        -VolumeName "vggt-zju-data" `
        -LocalDir $localPriorRoot `
        -RemoteDir $remotePriorDir `
        -Retries $UploadRetries
    $summary.uploaded_prior_count = [int]$priorCount
    $remoteItems = @(Invoke-ModalJson -CliArgs @("volume", "ls", "--json", "vggt-zju-data", $remotePriorDir))
    $summary.remote_prior_npz_count = [int]@($remoteItems | Where-Object {
        ([string]$_.Type -eq "file") -and ([string]$_.Filename -like "*.npz")
    }).Count
}

New-Item -ItemType Directory -Force -Path $localPriorRoot | Out-Null
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestPath -Encoding UTF8
Write-Host "[human-prior-case] summary saved to $manifestPath"
