param(
    [ValidateSet('Audit', 'Execute')][string]$Mode = 'Audit',
    [string]$ExpectedPlanSha256 = ''
)
$ErrorActionPreference = 'Stop'
$workspace = 'C:\Python310\layercake_merged_nextgen_perfectA_option1_full_ready'
$oldRoot = Join-Path $workspace '_public_r102_20261110'
$auditRoot = Join-Path $oldRoot 'preservation_20261118'
$repoNames = @('abi_release', 'layercake_release')
$env:GIT_NO_LAZY_FETCH = '1'
$env:GIT_OPTIONAL_LOCKS = '0'
function Digest($path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
function SaveJson($path, $value) {
    if (Test-Path -LiteralPath $path) { throw "Preserve existing record: $path" }
    $value | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $path -Encoding utf8
}
function Target($name) {
    if ($name -notin $repoNames) { throw 'Unapproved repository target' }
    $exact = Join-Path $oldRoot $name
    $resolved = (Resolve-Path -LiteralPath $exact).Path
    if ($resolved -ne $exact -or -not $resolved.StartsWith("$workspace\_public_r102_20261110\", [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Target escaped exact review-checkout boundary'
    }
    $items = @((Get-Item -LiteralPath $resolved -Force)) + @(Get-ChildItem -LiteralPath $resolved -Force -Recurse)
    if (@($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Refuse reparse points' }
    return $resolved
}
function GitObjects($repo) {
    $values = @(git -C $repo cat-file --batch-all-objects '--batch-check=%(objectname) %(objecttype) %(objectsize)')
    if ($LASTEXITCODE -ne 0 -or $values.Count -eq 0) { throw "Cannot enumerate Git objects: $repo" }
    return $values
}
function RetainedObjects($source, $main) {
    $values = @($source | ForEach-Object { $_.Split(' ')[0] } | git -C $main cat-file '--batch-check=%(objectname) %(objecttype) %(objectsize)')
    if ($LASTEXITCODE -ne 0 -or @(Compare-Object $source $values).Count) { throw "Objects missing from primary: $main" }
}
if ($Mode -eq 'Audit') {
    if (Test-Path -LiteralPath $auditRoot) { throw 'Preserve earlier audit; do not overwrite it' }
    New-Item -ItemType Directory -Path $auditRoot | Out-Null
    $freeBefore = (Get-PSDrive C).Free
    $rootReceipts = @(Get-ChildItem -LiteralPath $oldRoot -File -Force | ForEach-Object {
        [pscustomobject]@{path=$_.FullName;sha256=(Digest $_.FullName);bytes=$_.Length}
    })
    $plan = [System.Collections.Generic.List[object]]::new()
    $objects = [System.Collections.Generic.List[object]]::new()
    foreach ($name in $repoNames) {
        $target = Target $name
        $main = Join-Path $workspace $name
        $sourceObjects = @(GitObjects $target)
        RetainedObjects $sourceObjects $main
        $fsck = @(git -C $main fsck --full --no-reflogs --no-dangling 2>&1 | ForEach-Object { "$_" })
        if ($LASTEXITCODE -ne 0) { throw "Primary Git integrity failed: $name" }
        $objects.Add([pscustomobject]@{repo=$name;retained_in=$main;head=(git -C $target rev-parse HEAD);objects=$sourceObjects;primary_fsck_output=$fsck})
        foreach ($file in Get-ChildItem -LiteralPath $target -File -Force -Recurse) {
            $relative = $file.FullName.Substring($target.Length+1)
            if ($relative.StartsWith('.git\objects\')) { continue }
            $hash = Digest $file.FullName
            $candidate = Join-Path $main $relative
            $action = 'preserve_unique'
            $retained = Join-Path (Join-Path $auditRoot $name) $relative
            if ((Test-Path -LiteralPath $candidate -PathType Leaf) -and
                (Get-Item -LiteralPath $candidate -Force).Length -eq $file.Length -and (Digest $candidate) -eq $hash) {
                $retained = $candidate
                $action = 'duplicate'
            }
            $plan.Add([pscustomobject]@{repo=$name;source=$file.FullName;retained=$retained;sha256=$hash;bytes=$file.Length;action=$action})
        }
        Write-Output "Audited $name with $($sourceObjects.Count) retained Git objects"
    }
    $unique = @($plan | Where-Object action -eq 'preserve_unique')
    SaveJson (Join-Path $auditRoot 'plan.json') ([ordered]@{
        format='r110-targeted-cleanup/1';utc=[DateTime]::UtcNow.ToString('o');targets=$repoNames;
        free_bytes_before=$freeBefore;root_receipts=$rootReceipts;git_object_retention=$objects;files=$plan
    })
    [pscustomobject]@{files=$plan.Count;unique_files=$unique.Count;unique_bytes=($unique | Measure-Object bytes -Sum).Sum;
        duplicate_bytes=($plan | Where-Object action -eq 'duplicate' | Measure-Object bytes -Sum).Sum;
        plan_sha256=(Digest (Join-Path $auditRoot 'plan.json'));deletion_performed=$false} | ConvertTo-Json
    exit
}
$planPath = Join-Path $auditRoot 'plan.json'
if ($ExpectedPlanSha256 -notmatch '^[a-f0-9]{64}$' -or (Digest $planPath) -ne $ExpectedPlanSha256) { throw 'Explicit verified plan identity required' }
if (Test-Path -LiteralPath (Join-Path $auditRoot 'completion.json')) { throw 'Cleanup already completed' }
$document = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if ($document.format -ne 'r110-targeted-cleanup/1' -or @(Compare-Object $repoNames $document.targets).Count) { throw 'Wrong audit targets' }
foreach ($group in $document.git_object_retention) {
    $target = Target $group.repo
    $sourceObjects = @(GitObjects $target)
    if (@(Compare-Object $sourceObjects $group.objects).Count) { throw 'Source object inventory changed' }
    RetainedObjects $sourceObjects (Join-Path $workspace $group.repo)
    $actual = @(Get-ChildItem -LiteralPath $target -File -Force -Recurse | Where-Object { -not $_.FullName.StartsWith("$target\.git\objects\") } | ForEach-Object FullName)
    $expected = @($document.files | Where-Object repo -eq $group.repo | ForEach-Object source)
    if (@(Compare-Object $actual $expected).Count) { throw 'Checkout contents changed after audit' }
}
# Check every source and every duplicate survivor before moving or deleting any file.
foreach ($row in $document.files) {
    $target = Join-Path $oldRoot $row.repo
    if (-not $row.source.StartsWith("$target\", [StringComparison]::OrdinalIgnoreCase)) { throw 'Source outside selected checkout' }
    if ((Digest $row.source) -ne $row.sha256) { throw "Source changed: $($row.source)" }
    if ($row.action -eq 'duplicate') {
        $primary = Join-Path $workspace $row.repo
        if (-not $row.retained.StartsWith("$primary\", [StringComparison]::OrdinalIgnoreCase) -or (Digest $row.retained) -ne $row.sha256) { throw 'Duplicate retention mismatch' }
    } elseif ($row.action -eq 'preserve_unique') {
        $resolvedDestination = [IO.Path]::GetFullPath($row.retained)
        if (-not $resolvedDestination.StartsWith("$auditRoot\", [StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $resolvedDestination)) { throw 'Invalid unique-evidence destination' }
    } else { throw 'Unrecognized preservation action' }
}
$unique = @($document.files | Where-Object action -eq 'preserve_unique')
foreach ($row in $unique) {
    $destinationParent = Split-Path -Parent $row.retained
    if (-not (Test-Path -LiteralPath $destinationParent)) { New-Item -ItemType Directory -Path $destinationParent | Out-Null }
    Move-Item -LiteralPath $row.source -Destination $row.retained
    if ((Digest $row.retained) -ne $row.sha256) { throw 'Moved unique evidence hash mismatch' }
}
foreach ($row in $document.files) { if ((Digest $row.retained) -ne $row.sha256) { throw 'Final survivor hash mismatch' } }
foreach ($name in $repoNames) {
    $target = Target $name
    Remove-Item -LiteralPath $target -Recurse -Force
}
foreach ($receipt in $document.root_receipts) { if ((Digest $receipt.path) -ne $receipt.sha256) { throw 'Reconstruction receipt changed' } }
SaveJson (Join-Path $auditRoot 'completion.json') ([ordered]@{
    format='r110-targeted-cleanup-completion/1';utc=[DateTime]::UtcNow.ToString('o');plan_sha256=$ExpectedPlanSha256;
    removed_checkouts=$repoNames;preserved_unique_files=$unique.Count;preserved_unique_bytes=($unique | Measure-Object bytes -Sum).Sum;
    free_bytes_after=(Get-PSDrive C).Free;root_receipts_unchanged=$document.root_receipts;
    retained_cache_directories=@((Join-Path $oldRoot 'hf_cache'), (Join-Path $oldRoot 'hf_cache_copy'))
})
Get-Content -LiteralPath (Join-Path $auditRoot 'completion.json')
