[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [Parameter(Mandatory = $true)]
    [string]$SnapshotDir
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

function Read-JsonMaybe([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try {
        return (Get-Content -Raw -Path $Path -Encoding UTF8 | ConvertFrom-Json)
    } catch {
        return $null
    }
}

$resolvedSnapshotDir = if ([System.IO.Path]::IsPathRooted($SnapshotDir)) {
    $SnapshotDir
} else {
    Join-Path $RepoDir $SnapshotDir
}
$snapshotMatches = @()
if ($resolvedSnapshotDir.IndexOfAny(@([char]'*', [char]'?')) -ge 0) {
    $snapshotMatches = @(Get-ChildItem -Path $resolvedSnapshotDir -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
} elseif (Test-Path $resolvedSnapshotDir) {
    $snapshotMatches = @((Get-Item $resolvedSnapshotDir))
}
if ($snapshotMatches.Count -le 0) {
    throw "snapshot dir not found: $resolvedSnapshotDir"
}
$resolvedSnapshotDir = $snapshotMatches[0].FullName

$manifestPath = Join-Path $resolvedSnapshotDir "manifest.json"
$manifest = Read-JsonMaybe -Path $manifestPath
if ($null -eq $manifest) {
    throw "snapshot manifest missing or unreadable: $manifestPath"
}

$restored = 0
foreach ($entry in @($manifest)) {
    if ($null -eq $entry) { continue }
    if ([string]$entry.tag -ne "base") { continue }
    $copiedName = [string]$entry.copied_name
    $source = [string]$entry.source
    if ([string]::IsNullOrWhiteSpace($copiedName) -or [string]::IsNullOrWhiteSpace($source)) { continue }
    $snapshotFile = Join-Path $resolvedSnapshotDir $copiedName
    if (-not (Test-Path $snapshotFile)) { continue }
    $target = $source
    if (-not [System.IO.Path]::IsPathRooted($target)) {
        $target = Join-Path $RepoDir $target
    }
    $targetDir = Split-Path $target -Parent
    if (-not [string]::IsNullOrWhiteSpace($targetDir)) {
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    }
    Copy-Item -Path $snapshotFile -Destination $target -Force
    $restored += 1
}

Write-Host "[probe-restore] snapshot_dir=$resolvedSnapshotDir restored=$restored"
