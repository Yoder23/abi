param(
    [ValidateSet('Audit','Execute','Verify')][string]$Mode = 'Audit',
    [string]$ExpectedPlanSha256 = ''
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$workspace = (Resolve-Path -LiteralPath (Join-Path $repo '..')).Path
$out = Join-Path $repo 'results\review_2026_11_19'
$planPath = Join-Path $out 'r112_cleanup_plan.json'
$journalPath = Join-Path $out 'r112_cleanup_actions.jsonl'
$completePath = Join-Path $out 'r112_cleanup_complete.json'
$targetNames = @(
    'layercakeogwithdecoder', 'abi_release\clean_run', 'abi_release\.release-build',
    '.strict-hostile-r3-a', '.strict-hostile-r3-b', '.reexec', '.archive-audit', '.red-team'
)
$targetRoots = @($targetNames | ForEach-Object { (Resolve-Path -LiteralPath (Join-Path $workspace $_)).Path })
$primaryRoots = @($repo, (Resolve-Path -LiteralPath (Join-Path $workspace 'layercake_release')).Path)
$utf8 = [System.Text.UTF8Encoding]::new($false)

function Digest([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Within([string]$Path,[string]$Root) { $Path.StartsWith($Root+'\',[StringComparison]::OrdinalIgnoreCase) }
function Target([string]$Path) {
    if (-not (Within $Path $workspace) -or $Path -match '\\.git\\') { return $false }
    return @($targetRoots | Where-Object { Within $Path $_ }).Count -gt 0
}
function PlainFile([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    if (-not (Within $full $workspace)) { throw "Path outside intended workspace: $full" }
    $item = Get-Item -LiteralPath $full -Force
    if ($item.PSIsContainer) { throw "Not a file: $full" }
    $cursor = $item
    while ($cursor.FullName -ne $workspace) {
        if (($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse path: $full" }
        $cursor = Get-Item -LiteralPath (Split-Path -Parent $cursor.FullName) -Force
    }
    return $item
}
function WriteNew([string]$Path,$Value) {
    $stream = [IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {
        $bytes = $utf8.GetBytes(($Value | ConvertTo-Json -Depth 15)+'`n'.Replace('`n',"`n"))
        $stream.Write($bytes,0,$bytes.Length)
    } finally { $stream.Dispose() }
}
function ProtectedPaths {
    $set = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($root in $primaryRoots) {
        $names = @(& git -C $root ls-files)
        if ($LASTEXITCODE -ne 0) { throw "Cannot resolve tracked files: $root" }
        foreach ($name in $names) { [void]$set.Add([IO.Path]::GetFullPath((Join-Path $root $name))) }
    }
    $manifest = Join-Path $repo 'results\review_2026_11_18\r111_manifest.json'
    if ((Digest $manifest) -ne '13a93229d5d9f13f19df38a8eeb60017d57e6a6ebc42ccd226647304287e0f76') {
        throw 'R111 manifest changed'
    }
    $m = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
    foreach($name in @($m.files.PSObject.Properties.Name)+@($m.host_inputs.PSObject.Properties.Name)) {
        [void]$set.Add([IO.Path]::GetFullPath((Join-Path $repo $name)))
    }
    return ,$set
}

New-Item -ItemType Directory -Path $out -Force | Out-Null
if ($Mode -eq 'Audit') {
    if (Test-Path -LiteralPath $planPath) { throw 'Preserve the existing audit plan' }
    $protected = ProtectedPaths
    $candidates = @($targetRoots | ForEach-Object { Get-ChildItem -LiteralPath $_ -File -Recurse -Force } |
        Where-Object { $_.Length -ge 16MB -and (Target $_.FullName) -and -not $protected.Contains($_.FullName) } |
        Sort-Object FullName -Unique)
    $sizes = @{}
    foreach($f in $candidates) { $sizes[$f.Length.ToString()]=$true }
    $retained = @($primaryRoots | ForEach-Object { Get-ChildItem -LiteralPath $_ -File -Recurse -Force } |
        Where-Object { $_.Length -ge 16MB -and -not (Target $_.FullName) -and $sizes.ContainsKey($_.Length.ToString()) } |
        Sort-Object FullName -Unique)
    $candidatePaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach($f in $candidates) { [void]$candidatePaths.Add($f.FullName) }
    $duplicates = [Collections.Generic.List[object]]::new()
    $hashed = 0
    foreach ($group in @(@($retained)+@($candidates) | Group-Object Length | Where-Object Count -gt 1)) {
        $byHash = @{}
        foreach ($f in ($group.Group | Sort-Object @{Expression={$candidatePaths.Contains($_.FullName)}},FullName)) {
            [void](PlainFile $f.FullName)
            $digest = Digest $f.FullName
            $hashed++
            if (-not $byHash.ContainsKey($digest)) { $byHash[$digest] = $f; continue }
            if (-not $candidatePaths.Contains($f.FullName)) { continue }
            $keep = $byHash[$digest]
            $duplicates.Add([ordered]@{remove=$f.FullName;retain=$keep.FullName;bytes=$f.Length;sha256=$digest})
        }
        Write-Output "Hashed $hashed candidate/reference files; exact duplicates found: $($duplicates.Count)"
    }
    $selected = [Collections.Generic.List[object]]::new()
    [long]$bytes = 0
    [long]$free = (Get-PSDrive C).Free
    foreach($row in ($duplicates | Sort-Object @{Expression={$_.bytes};Descending=$true},remove)) {
        if ($bytes+$free -ge 12GB) { break }
        $selected.Add($row); $bytes += $row.bytes
    }
    WriteNew $planPath ([ordered]@{
        format='r112-exact-duplicate-cleanup/1'; created_utc=[DateTime]::UtcNow.ToString('o');
        workspace=$workspace; target_roots=$targetRoots; minimum_candidate_bytes=16MB;
        script_sha256=(Digest $PSCommandPath); candidate_files=$candidates.Count; hashed_files=$hashed;
        duplicate_files_found=$duplicates.Count; selected_files=$selected.Count;
        selected_logical_bytes=$bytes; free_bytes_before=$free; target_free_bytes=12GB;
        protected_paths=$protected.Count; deletions=@($selected);
        scope='Only exact SHA-256 duplicates; retain every unique file and all primary tracked/R111 dependencies; no recursive deletion'
    })
    Write-Output "Plan SHA256: $(Digest $planPath); selected $($selected.Count) files / $bytes logical bytes"
    exit 0
}

if ($ExpectedPlanSha256 -notmatch '^[0-9a-f]{64}$' -or (Digest $planPath) -ne $ExpectedPlanSha256) {
    throw 'Supply the independently checked exact plan SHA256'
}
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if ($plan.workspace -ne $workspace -or $plan.script_sha256 -ne (Digest $PSCommandPath) -or
    (@($plan.target_roots) -join '|') -ne ($targetRoots -join '|')) { throw 'Changed cleanup scope or implementation' }
$protected = ProtectedPaths
$deletedSet = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach($row in $plan.deletions) {
    $target = [IO.Path]::GetFullPath($row.remove)
    if ($target -ne $row.remove -or -not (Target $target) -or $protected.Contains($target) -or -not $deletedSet.Add($target)) {
        throw "Invalid, protected or repeated target: $target"
    }
}
foreach($row in $plan.deletions) {
    if ($deletedSet.Contains($row.retain)) { throw 'Retained copy also scheduled for deletion' }
    $keep = PlainFile $row.retain
    if ($keep.Length -ne $row.bytes -or (Digest $row.retain) -ne $row.sha256) { throw 'Retained copy missing or changed' }
}
if ($Mode -eq 'Verify') {
    $complete = Get-Content -LiteralPath $completePath -Raw | ConvertFrom-Json
    if ($complete.plan_sha256 -ne $ExpectedPlanSha256 -or $complete.removed_files -ne $plan.selected_files) { throw 'Incomplete cleanup receipt' }
    foreach($row in $plan.deletions) { if (Test-Path -LiteralPath $row.remove) { throw 'Removal incomplete' } }
    Write-Output "Verified $($plan.selected_files) removals and every retained SHA256; plan $ExpectedPlanSha256"
    exit 0
}
if ((Test-Path -LiteralPath $journalPath) -or (Test-Path -LiteralPath $completePath)) { throw 'Preserve existing cleanup execution' }
foreach($row in $plan.deletions) {
    $target = PlainFile $row.remove
    if ($target.Length -ne $row.bytes -or (Digest $row.remove) -ne $row.sha256) { throw 'Target changed since audit' }
}
$journal = [IO.StreamWriter]::new([IO.File]::Open($journalPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read),$utf8)
$journal.AutoFlush = $true
[long]$before = (Get-PSDrive C).Free
$removed = 0
try {
    foreach($row in $plan.deletions) {
        [void](PlainFile $row.remove); [void](PlainFile $row.retain)
        if ((Digest $row.retain) -ne $row.sha256 -or (Digest $row.remove) -ne $row.sha256) { throw 'Copy changed immediately before removal' }
        $journal.WriteLine((@{action='verified_before_remove';remove=$row.remove;retain=$row.retain;sha256=$row.sha256;bytes=$row.bytes} | ConvertTo-Json -Compress))
        Remove-Item -LiteralPath $row.remove -Force
        if (Test-Path -LiteralPath $row.remove) { throw 'Removal did not complete' }
        $removed++
        $journal.WriteLine((@{action='removed';remove=$row.remove;retained_sha256=(Digest $row.retain);utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress))
        Write-Output "Removed verified duplicate $removed/$($plan.selected_files)"
    }
} finally { $journal.Dispose() }
WriteNew $completePath ([ordered]@{format='r112-exact-duplicate-cleanup-complete/1';plan_sha256=$ExpectedPlanSha256;
    removed_files=$removed;removed_logical_bytes=$plan.selected_logical_bytes;free_bytes_before=$before;
    free_bytes_after=(Get-PSDrive C).Free;journal_sha256=(Digest $journalPath);completed_utc=[DateTime]::UtcNow.ToString('o');
    recovery='Copy the retained file to its original remove path using the plan; all retained bytes hash-identical'})
Write-Output (Get-Content -LiteralPath $completePath -Raw)
