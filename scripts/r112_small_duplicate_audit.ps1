$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$workspace = (Resolve-Path -LiteralPath (Join-Path $repo '..')).Path
$destination = Join-Path $repo 'results\review_2026_11_19\r112_small_duplicate_audit.json'
if (Test-Path -LiteralPath $destination) { throw 'Preserve the existing audit' }
$primary = @($repo, (Join-Path $workspace 'layercake_release'))
$tracked = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($root in $primary) {
    $names = @(& git -C $root ls-files)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve tracked paths' }
    foreach ($name in $names) { [void]$tracked.Add([IO.Path]::GetFullPath((Join-Path $root $name))) }
}
$protected = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$manifestPath = Join-Path $repo 'results\review_2026_11_18\r111_manifest.json'
if ((Get-FileHash -LiteralPath $manifestPath).Hash.ToLowerInvariant() -ne '13a93229d5d9f13f19df38a8eeb60017d57e6a6ebc42ccd226647304287e0f76') { throw 'Changed R111 manifest' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($name in @($manifest.files.PSObject.Properties.Name)+@($manifest.host_inputs.PSObject.Properties.Name)) {
    [void]$protected.Add([IO.Path]::GetFullPath((Join-Path $repo $name)))
}
$files = @(Get-ChildItem -LiteralPath $workspace -File -Recurse -Force -ErrorAction Stop |
    Where-Object { $_.Length -ge 16KB -and $_.Length -lt 1MB -and $_.FullName -notmatch '\\(\.git|hf_models|\.cache|\.datasets_cache)\\' -and ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0 })
$groups = [Collections.Generic.List[object]]::new()
$hashed = 0
foreach ($size in ($files | Group-Object Length | Where-Object Count -gt 1)) {
    $hashes = @{}
    foreach ($file in $size.Group) {
        $digest = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $hashed++
        if (-not $hashes.ContainsKey($digest)) { $hashes[$digest] = [Collections.Generic.List[object]]::new() }
        $hashes[$digest].Add([ordered]@{
            path=$file.FullName; tracked=$tracked.Contains($file.FullName);
            r111_dependency=$protected.Contains($file.FullName);
            git_metadata=($file.FullName -match '\\.git\\');
            primary_repository=(@($primary | Where-Object { $file.FullName.StartsWith($_+'\',[StringComparison]::OrdinalIgnoreCase) }).Count -gt 0)
        })
    }
    foreach ($digest in $hashes.Keys) {
        if ($hashes[$digest].Count -gt 1) {
            $groups.Add([ordered]@{sha256=$digest; bytes=[long]$size.Name; copies=@($hashes[$digest]); count=$hashes[$digest].Count})
        }
    }
    Write-Output "Hashed $hashed files; duplicate groups $($groups.Count)"
}
$result = [ordered]@{
    format='r112-readonly-remaining-duplicate-audit/1'; created_utc=[DateTime]::UtcNow.ToString('o');
    workspace=$workspace; minimum_bytes=16KB; maximum_bytes_exclusive=1MB; files_considered=$files.Count; hashed_files=$hashed;
    free_bytes=(Get-PSDrive C).Free; groups=@($groups | Sort-Object bytes -Descending);
    script_sha256=(Get-FileHash -LiteralPath $PSCommandPath).Hash.ToLowerInvariant();
    caution='Read-only candidates, NOT a deletion plan. Primary paths, Git objects, source caches, dependency bindings and retained cleanup references require protection. Size/hash identity does not establish dispensability.'
}
$stream = [IO.File]::Open($destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try {
    $data = [Text.UTF8Encoding]::new($false).GetBytes(($result | ConvertTo-Json -Depth 12)+"`n")
    $stream.Write($data,0,$data.Length)
} finally { $stream.Dispose() }
Write-Output "Audit SHA256: $((Get-FileHash -LiteralPath $destination).Hash.ToLowerInvariant())"
