param(
    [ValidateSet('Plan','Execute','Verify')][string]$Mode = 'Plan',
    [string]$ExpectedPlanSha256 = ''
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$workspace = (Resolve-Path -LiteralPath (Join-Path $repo '..')).Path
$out = Join-Path $repo 'results\review_2026_11_19'
$planPath = Join-Path $out 'r112_hardlink_aux_plan.json'
$journalPath = Join-Path $out 'r112_hardlink_aux_actions.jsonl'
$completePath = Join-Path $out 'r112_hardlink_aux_complete.json'
$auditPath = Join-Path $out 'r112_remaining_duplicate_audit.json'
$manifestPath = Join-Path $repo 'results\review_2026_11_18\r111_manifest.json'
$roots = @((Join-Path $repo 'results'),(Join-Path $workspace 'layercake_release\artifacts'),(Join-Path $workspace 'layercake_release\runs_experiment'))
$auxNames = @('.strict-hostile-r3-a','.strict-hostile-r3-b','.reexec','.archive-audit','.red-team','.blind-redteam-r2-tag-a','abi_public_r3_tag_clone','abi_public_r4_tag_clone')
$auxRoots = @($auxNames | ForEach-Object { (Resolve-Path -LiteralPath (Join-Path $workspace $_)).Path })
$roots += $auxRoots
$utf8 = [Text.UTF8Encoding]::new($false)
function Digest([string]$p) { (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() }
function Within([string]$p,[string]$r) { $p.StartsWith($r+'\',[StringComparison]::OrdinalIgnoreCase) }
function WriteNew([string]$p,$v) {
    $s=[IO.File]::Open($p,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try { $b=$utf8.GetBytes(($v | ConvertTo-Json -Depth 15)+"`n");$s.Write($b,0,$b.Length) } finally {$s.Dispose()}
}
if ((Digest $auditPath) -ne '2eea09c52caec9a54eaf8d85138eb73830e56695310d7fda9dd4c7cec24f6aca' -or
    (Digest $manifestPath) -ne '13a93229d5d9f13f19df38a8eeb60017d57e6a6ebc42ccd226647304287e0f76') { throw 'Changed audit or R111 custody anchor' }
$protected = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach($name in @($manifest.files.PSObject.Properties.Name)+@($manifest.host_inputs.PSObject.Properties.Name)) {
    [void]$protected.Add([IO.Path]::GetFullPath((Join-Path $repo $name)))
}
function Eligible([string]$p) {
    $p=[IO.Path]::GetFullPath($p)
    return (Within $p $workspace) -and (@($roots | Where-Object { Within $p $_ }).Count -gt 0) -and
        -not $protected.Contains($p) -and $p -notmatch '\\(\.git|hf_models|\.cache|\.datasets_cache|data|scope_revision_r111|count_feedback_r112|review_2026_11_18|review_2026_11_19)\\'
}
function PlainFile([string]$p) {
    if (-not (Eligible $p)) { throw "Protected/out-of-scope file: $p" }
    $f=Get-Item -LiteralPath $p -Force
    if($f.PSIsContainer){throw 'Directory is not a payload'}
    $cursor=$f
    while($cursor.FullName -ne $workspace){
        if(($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw "Reparse path: $p"}
        $cursor=Get-Item -LiteralPath (Split-Path -Parent $cursor.FullName) -Force
    }
    return $f
}
function Links([string]$p) {
    $lines=@(& fsutil hardlink list $p)
    if($LASTEXITCODE -ne 0){throw "Cannot verify physical file links: $p"}
    $paths=@($lines | Where-Object { $_.Trim() } | ForEach-Object {
        $line=$_.Trim()
        if(-not $line.StartsWith('\')){throw 'Unexpected fsutil hardlink output'}
        [IO.Path]::GetFullPath([IO.Path]::GetPathRoot($p).TrimEnd('\')+$line)
    })
    if($paths -notcontains $p){throw 'Hardlink inventory does not contain queried path'}
    return $paths
}
if($Mode -eq 'Plan'){
    if(Test-Path -LiteralPath $planPath){throw 'Preserve existing hardlink plan'}
    $audit=Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
    $actions=[Collections.Generic.List[object]]::new()
    foreach($g in ($audit.groups | Sort-Object bytes -Descending)){
        $copies=@($g.copies | Where-Object { Eligible $_.path } | Sort-Object @{Expression={$_.primary_repository};Descending=$true},@{Expression={$_.tracked};Descending=$true},path)
        if($copies.Count -lt 2){continue}
        $valid=@($copies | Where-Object {
            $f=PlainFile $_.path
            if($f.Length -ne $g.bytes -or (Digest $f.FullName) -ne $g.sha256){throw 'Historical copy changed since audit'}
            $links=@(Links $f.FullName)
            # Retain an existing linked canonical only when every alias is in scope.
            @($links | Where-Object { -not (Eligible $_) }).Count -eq 0
        })
        if($valid.Count -lt 2){continue}
        $keep=$valid[0].path
        foreach($copy in ($valid | Select-Object -Skip 1)){
            $candidate = $copy.path
            if(@($auxRoots | Where-Object { Within $candidate $_ }).Count -eq 0 -or @(Links $candidate) -contains $keep){continue}
            $f=Get-Item -LiteralPath $copy.path -Force
            $actions.Add([pscustomobject]@{target=$copy.path;retain=$keep;bytes=$g.bytes;sha256=$g.sha256;
                original_attributes=$f.Attributes.ToString();original_last_write_utc=$f.LastWriteTimeUtc.ToString('o');
                backup=(Join-Path $f.DirectoryName '.r112-link-backup')})
        }
    }
    WriteNew $planPath ([ordered]@{format='r112-hardlink-plan/1';created_utc=[DateTime]::UtcNow.ToString('o');
        workspace=$workspace;target_roots=$roots;script_sha256=(Digest $PSCommandPath);
        source_audit_sha256=(Digest $auditPath);r111_manifest_sha256=(Digest $manifestPath);
        free_bytes_before=(Get-PSDrive C).Free;files=$actions.Count;logical_duplicate_bytes=($actions | Measure-Object bytes -Sum).Sum;
        actions=@($actions);scope='Keep every path and byte; historical payloads only; no R111 dependency, source cache or Git metadata; shared payloads become read-only'})
    Write-Output "Plan SHA256: $(Digest $planPath); files: $($actions.Count)"
    exit 0
}
if($ExpectedPlanSha256 -notmatch '^[0-9a-f]{64}$' -or (Digest $planPath) -ne $ExpectedPlanSha256){throw 'Exact reviewed plan SHA256 required'}
$plan=Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if($plan.workspace -ne $workspace -or $plan.script_sha256 -ne (Digest $PSCommandPath) -or ($plan.target_roots -join '|') -ne ($roots -join '|')){throw 'Changed hardlink scope/implementation'}
$targets=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach($row in $plan.actions){$t=$row.target;if(@($auxRoots | Where-Object { Within $t $_ }).Count -eq 0 -or -not $targets.Add($t)){throw 'Invalid/repeated auxiliary target'}}
foreach($row in $plan.actions){
    if($targets.Contains($row.retain) -or $row.backup -ne (Join-Path (Split-Path -Parent $row.target) '.r112-link-backup')){throw 'Unsafe retained/backup path'}
    foreach($p in @($row.target,$row.retain)){
        $f=PlainFile $p
        if($f.Length -ne $row.bytes -or (Digest $p) -ne $row.sha256){throw 'Missing/changed payload'}
        foreach($link in @(Links $p)){if(-not (Eligible $link)){throw 'Existing link reaches protected file'}}
    }
    if(Test-Path -LiteralPath $row.backup){throw 'Preserve interrupted rollback file'}
}
if($Mode -eq 'Verify'){
    $complete=Get-Content -LiteralPath $completePath -Raw | ConvertFrom-Json
    if($complete.plan_sha256 -ne $ExpectedPlanSha256 -or $complete.completed_files -ne $plan.files -or $complete.journal_sha256 -ne (Digest $journalPath)){throw 'Incomplete/changed completion'}
    foreach($row in $plan.actions){
        if(@(Links $row.target) -notcontains $row.retain -or -not (Get-Item -LiteralPath $row.target -Force).IsReadOnly){throw 'Missing physical link or immutability flag'}
    }
    Write-Output "Verified all $($plan.files) byte-identical physical hardlinks; R111/cache/Git exclusions intact"
    exit 0
}
if((Test-Path -LiteralPath $journalPath) -or (Test-Path -LiteralPath $completePath)){throw 'Preserve prior execution'}
$writer=[IO.StreamWriter]::new([IO.File]::Open($journalPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read),$utf8)
$writer.AutoFlush=$true
$before=(Get-PSDrive C).Free
$count=0
try{
    foreach($row in $plan.actions){
        [void](PlainFile $row.target);[void](PlainFile $row.retain)
        if((Digest $row.target) -ne $row.sha256 -or (Digest $row.retain) -ne $row.sha256){throw 'Payload changed immediately before linking'}
        $writer.WriteLine((@{action='before_link';target=$row.target;retain=$row.retain;sha256=$row.sha256;bytes=$row.bytes;backup=$row.backup} | ConvertTo-Json -Compress))
        Move-Item -LiteralPath $row.target -Destination $row.backup
        try{
            New-Item -ItemType HardLink -Path $row.target -Target $row.retain -ErrorAction Stop | Out-Null
            if((Digest $row.target) -ne $row.sha256 -or @(Links $row.target) -notcontains $row.retain){throw 'New hardlink verification failed'}
            (Get-Item -LiteralPath $row.target -Force).IsReadOnly=$true
            if((Digest $row.backup) -ne $row.sha256){throw 'Rollback copy changed'}
            Remove-Item -LiteralPath $row.backup -Force
        }catch{
            if(Test-Path -LiteralPath $row.backup){
                if(Test-Path -LiteralPath $row.target){Remove-Item -LiteralPath $row.target -Force}
                Move-Item -LiteralPath $row.backup -Destination $row.target
            }
            throw
        }
        $count++
        $writer.WriteLine((@{action='linked';target=$row.target;retain=$row.retain;sha256=(Digest $row.target);physical_links=@(Links $row.target);utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress))
        if($count % 25 -eq 0){Write-Output "Linked $count/$($plan.files)"}
    }
}finally{$writer.Dispose()}
WriteNew $completePath ([ordered]@{format='r112-hardlink-complete/1';plan_sha256=$ExpectedPlanSha256;
    completed_files=$count;logical_duplicate_bytes=$plan.logical_duplicate_bytes;free_bytes_before=$before;
    free_bytes_after=(Get-PSDrive C).Free;journal_sha256=(Digest $journalPath);completed_utc=[DateTime]::UtcNow.ToString('o');
    recovery='Every original path remains. To restore independent storage, copy a linked payload to a new file and replace one path after verifying its hash; never modify shared historical bytes in place.'})
Write-Output (Get-Content -LiteralPath $completePath -Raw)
