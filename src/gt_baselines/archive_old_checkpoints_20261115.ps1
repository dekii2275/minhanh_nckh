$ErrorActionPreference = 'Stop'
$outputRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\outputs')).Path
$archiveRoot = Join-Path $outputRoot '_checkpoint_archive_20261115'
if (Test-Path -LiteralPath $archiveRoot) { throw 'Archive already exists; choose a new destination.' }
$prefix = $outputRoot.TrimEnd('\') + '\'
$items = @(Get-ChildItem -LiteralPath $outputRoot -Filter '*.pt' -File -Recurse)
if ($items.Count -eq 0) { throw 'No checkpoints found; refusing to proceed.' }
$inventory = foreach ($item in $items) {
    $resolved = (Resolve-Path -LiteralPath $item.FullName).Path
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Outside output root: $resolved" }
    $relative = $resolved.Substring($prefix.Length)
    $destination = Join-Path $archiveRoot $relative
    if (-not $destination.StartsWith($archiveRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Invalid destination: $destination" }
    [PSCustomObject]@{ source = $resolved; destination = $destination; sha256 = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash; bytes = $item.Length }
}
New-Item -ItemType Directory -Path $archiveRoot | Out-Null
$inventory | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $archiveRoot 'inventory.json') -Encoding utf8
foreach ($entry in $inventory) {
    $parent = Split-Path -Parent $entry.destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Move-Item -LiteralPath $entry.source -Destination $entry.destination
    if ((Get-FileHash -LiteralPath $entry.destination -Algorithm SHA256).Hash -ne $entry.sha256) { throw "Hash mismatch: $($entry.destination)" }
}
Write-Output "Archived $($inventory.Count) checkpoints under $archiveRoot; original report links to moved checkpoints now require the archive paths."
