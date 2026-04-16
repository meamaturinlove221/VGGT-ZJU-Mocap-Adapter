[CmdletBinding()]
param(
    [string]$RepoDir = "F:\vggt",
    [string]$StatusDir = "logs/modal_phase5"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $RepoDir

function Ensure-CsvColumns(
    [string]$Path,
    [string[]]$Columns
) {
    if (-not (Test-Path $Path)) { return [pscustomobject]@{ path = $Path; changed = $false; added = @() } }
    $rows = @(Import-Csv $Path)
    $added = New-Object System.Collections.Generic.List[string]
    foreach ($col in $Columns) {
        $exists = $false
        if ($rows.Count -gt 0) {
            $exists = ($rows[0].PSObject.Properties.Name -contains $col)
        } else {
            $header = Get-Content $Path -TotalCount 1 -Encoding UTF8
            $exists = ([string]$header -split '","') -contains $col
        }
        if ($exists) { continue }
        $added.Add($col) | Out-Null
        foreach ($row in $rows) {
            Add-Member -InputObject $row -NotePropertyName $col -NotePropertyValue ""
        }
    }
    if ($added.Count -gt 0) {
        if ($rows.Count -gt 0) {
            $rows | Export-Csv $Path -NoTypeInformation -Encoding UTF8
        } else {
            $obj = [pscustomobject]([ordered]@{})
            foreach ($col in $Columns) { Add-Member -InputObject $obj -NotePropertyName $col -NotePropertyValue "" -Force }
            @($obj) | Select-Object -First 0 | Export-Csv $Path -NoTypeInformation -Encoding UTF8
        }
        return [pscustomobject]@{ path = $Path; changed = $true; added = @($added) }
    }
    return [pscustomobject]@{ path = $Path; changed = $false; added = @() }
}

$targets = @(
    [pscustomobject]@{
        path = (Join-Path $StatusDir "ghost_mvdepth_sweep_latest.csv")
        cols = @("point_target_blend_by_mv_support")
    },
    [pscustomobject]@{
        path = (Join-Path $StatusDir "ghost_autoloop_latest.csv")
        cols = @("precompute_mv_support_on", "point_target_blend_by_mv_support")
    }
)

foreach ($t in $targets) {
    $res = Ensure-CsvColumns -Path $t.path -Columns $t.cols
    Write-Host ("[repair-csv-schema] path={0} changed={1} added={2}" -f $res.path, $res.changed, ($res.added -join ","))
}
