VERDICT: PASS

# ABI V2 R7 independent blind technical review

Date: 2026-08-30  
Scratch root: `C:\tmp\abi_public_r7_blind_codex_20260830`  
Public repository: `https://github.com/Yoder23/abi`  
Public release/tag: `abi-final-validation-v2-repaired-r7-2026-08-30`

## Bounded conclusion

I found no material defect in the repaired R7 technical evidence. The result that
passes is deliberately narrow: the same four published capability packages execute
through the standalone ABI capability-runtime/conformance boundary in the declared
LayerCake v25, Qwen2.5-0.5B, and Pythia-160M environments. The evidence does not
establish arbitrary-model compatibility, base-weight or hidden-state transplantation,
teacher-knowledge extraction superiority, superiority over LoRA/distillation,
human-perceived generation quality, human ratings, independent hardware reproduction,
or a globally minimal English/information substrate.

Three external gates remain open: 21,000 judgments by three real independent humans,
independent reproduction on different hardware, and the registered minimum-information
certification.

All writes, downloads, virtual environments, extracted trees, negative controls, and
review artifacts remained under the designated scratch root. I did not inspect or use
`C:\Python310\layercake_merged_nextgen_perfectA_option1_full_ready`, any earlier
reconstruction, a development cache, or an unpublished asset. The archived hostile
receipt contains historical exception strings naming that directory; reading those
published strings is not access to the named directory. No GitHub or other remote state
was modified.

## Decisive evidence

| Area | Independent observation | Result |
| --- | --- | --- |
| Annotated tag | Tag object `94738bf8589363ec367a90142bc29ef461585aae` is type `tag` and peels to `3f82a9f4d67dda5c8ea13bd59b2d8f1bbd3dd128` | PASS |
| Public assets | GitHub metadata, downloaded byte counts, manifest SHA-256 values, and content addresses agree for all five payloads; manifest itself is 3,321 bytes, SHA-256 `3691f736…` | PASS |
| ZIP identity | 1,193 unique members equal the 1,192 embedded rows plus the embedded manifest; zero missing, unexpected, member-hash, or extracted-hash failures | PASS |
| Long path | Ordinary workspace base length 174; deepest extracted ordinary path length 368 with Windows `LongPathsEnabled=0` | PASS |
| Official reconstruction | `PASS_PUBLIC_MANIFEST_ONLY_RECONSTRUCTION`; 17 focused tests passed in 40.48 s | PASS |
| Strict stdout | 166,950 bytes, empty stderr, SHA-256 `17973df22cb31eddb3b47c6137aa71b68f32cd9d3010cfcddd232b2a1afce488`, byte-identical to the published strict certificate | PASS |
| Tar scanner | USTAR and V7 streams detected after 513, 1,024, and 8,388,608-byte prefixes; six invalid tar-literal controls rejected | PASS |
| Inventory pin | 301,543 canonical rows and all three exact source commitments recomputed; rehashed deletion of `/usr/bin/NF` rejected | PASS |
| Hostile receipt/replay | Published 19 raw cases independently validated; full disposable replay rejected 19/19 and restored exact baseline `e12e0c81…` | PASS |
| Live execution | 24 globally distinct condition PIDs, 3,072 bound causal rows, 2,100 bound isolation rows, 733 exact transitive source files | PASS |
| Certification absence | Across all inventories: zero capability signatures/suffix paths, zero campaign/success IDs, zero unsupported archive signatures, zero forbidden archive members | PASS |
| Public sufficiency | Public manifest plus archive reproduced without development directories, model weights, unpublished assets, or source-teacher inference | PASS |

## 1. Public identity and immutable bytes

The clean clone was created directly from the public annotated tag with
`--single-branch`. It remained detached and clean throughout the review.

The tag object and commit resolved as follows:

```text
tag object   94738bf8589363ec367a90142bc29ef461585aae
object type  tag
peeled       3f82a9f4d67dda5c8ea13bd59b2d8f1bbd3dd128
HEAD         3f82a9f4d67dda5c8ea13bd59b2d8f1bbd3dd128
expected     3f82a9f4d67dda5c8ea13bd59b2d8f1bbd3dd128
```

The public release was non-draft and non-prerelease, published at
`2026-08-30T09:59:51Z`. GitHub's asset API metadata independently reported the
same sizes and SHA-256 digests as the downloaded manifest and local bytes:

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `public_release_assets_r7.json` | 3,321 | `3691f73624f88c6bfadfaf0d9571e1eaa15fbada834d42af7dda5e4b4279d559` |
| `abi-final-validation-v2-r7-2026-08-30.zip` | 844,018,841 | `fc50f423986149b5d4670ec9e28698540f64be96034efa26e5704c4469921e88` |
| `abi-chemistry-token-plan-seed9824.cake` | 510,981 | `f9c9b2668fda5ef6b92844c1b7097fbdf8ff0daaae51f5b86f72d4a49000abeb` |
| `abi-civics-token-plan-seed9824.cake` | 495,919 | `634ce66958859ec36dc1fbdf5ef34d6d2a9949d10cf2348a68c245d8c325d604` |
| `phase7-final-english-core.cake` | 253,216,208 | `acb787b3ffa0153c57d88cd37ba81c3f00b370d4ca4937e659cd4c775851f25d` |
| `abi-python-token-plan-seed9824.cake` | 448,404 | `f1defaef2771ced336a332572a2d2f0e1e542399c877d182c48a6cd2e199231d` |

The four external capability assets each matched exactly one internal archive
member by basename, size, and independently streamed SHA-256. The public strict
and hostile bindings also matched their internal and extracted files exactly.

## 2. Archive membership, source lineage, and receipt binding

I parsed the ZIP central directory and embedded manifest independently of
`verify_archive`, streamed every member through SHA-256, and then repeated the
size/hash comparison against every extracted file.

- Archive members: 1,193, all unique.
- Embedded manifest rows: 1,192.
- Embedded manifest SHA-256: `c44a5a57185042006570010d0bfd3d6600daebcf7ca096331f3161705992b8b9`.
- Missing members: 0.
- Unexpected members: 0.
- Member identity failures: 0.
- Extracted membership or identity failures: 0.
- Forbidden development directories: 0.
- Actual `.safetensors`, `.pt`, `.pth`, or `.onnx` host-weight members: 0.

For the 983 `abi_release/` archive rows, 965 normal Git paths bound to the tagged
tree. Eighteen expected public-only inputs—four capability artifacts, public keys,
sealed human packet/evaluation rows, and frozen protocols—are intentionally not
tracked in Git and instead bind through the public archive and its SHA-256. There
were no unexpected missing Git paths. The only Git/archive or checkout/archive
byte differences were CRLF/LF transformations in `LICENSE`, `pyproject.toml`, and
`requirements.txt`; normalized text was exactly equal. No scientific source had a
content mismatch.

The strict certificate's complete input closure contains 810 unique files. I
recomputed every size and SHA-256 and its aggregate:

```text
required input files       810
identity failures          0
aggregate SHA-256          05f7d08ed9ad3393ef3ac0258488c112aaafe32a0c9733a67940b69f97efc7c1
```

The common live execution-source closure contains 733 unique source files and
recomputed exactly to aggregate
`3900cc53a895f4125917a59771fe3119cb9817673bb3ea605fd347c6635f04c6`.

## 3. Official public reconstruction and Windows long paths

The host registry reported `LongPathsEnabled=0`. I used an ordinary resolved
Windows workspace path of length 174; it did not itself use `\\?\`. The official
module used its published extended-path I/O handling while extracting. The deepest
ordinary extracted member path was 368 characters:

```text
C:\tmp\abi_public_r7_blind_codex_20260830\public_reconstruction_workspace_t...\
reconstructed\abi-final-validation-v2\abi_release\results\abi_final_validation_v2\
pre_strict_capsule_root_history\layercake_release\layercake_extensions\
route_isolated_clarification_core_v25.py
```

The final run used a fresh Python 3.10 virtual environment and every package version
named by `external_reproduction/environment.lock.json`, plus the declared test runner.
Package caching and temporary paths were confined to the scratch root.

The official receipt reported:

```text
status                             PASS_PUBLIC_MANIFEST_ONLY_RECONSTRUCTION
archive SHA-256                    fc50f423986149b5d4670ec9e28698540f64be96034efa26e5704c4469921e88
strict exit                        0
strict stdout SHA-256              17973df22cb31eddb3b47c6137aa71b68f32cd9d3010cfcddd232b2a1afce488
strict certificate identity exact true
focused tests                      17 passed in 40.48s
development directories present   []
```

I then ran the strict verifier again with a standalone full text capture. Its
166,950-byte normalized stdout was byte-identical to the published certificate;
stderr was empty.

Two earlier reconstruction attempts are intentionally preserved. They failed before
scientific verification because my initially minimal fresh environment omitted first
`numpy`, then `cryptography`, both named by the published environment lock. After I
installed the complete lock, the third fresh end-to-end run passed. These are reviewer
setup failures, not target failures.

## 4. Repaired physical scanner claim

The public scanner/inventory focused test run passed 13/13. I also constructed fresh
USTAR and V7 tar payloads containing:

```text
payload/manifest.json
payload/tensors.safetensors
payload/signature.json
```

For both formats, the scanner detected the capability archive after exact prefixes of
513, 1,024, and 8,388,608 bytes. The final prefix is exactly
`CONTENT_SCAN_BLOCK_BYTES`; observations recorded `tar@8388608`, three members scanned,
and a capability-package signature. Six additional controls placed incidental USTAR
or V7 member/checksum-like literals at the same offsets. Every invalid case produced
zero archive members and no capability signature.

This directly falsified neither repaired branch: valid USTAR/V7 arbitrary-offset and
scan-boundary streams were detected, while invalid tar-like literals were not promoted
to archives.

## 5. Complete reachable-inventory commitment

I streamed all three inventory JSONL files, checked canonical encoding, path ordering
and uniqueness, schema/hash shape, recomputed every aggregate, and compared the bytes
to the source-pinned release commitments.

| Host | Rows | Inventory SHA-256 | Content bytes scanned | Embedded members scanned |
| --- | ---: | --- | ---: | ---: |
| LayerCake | 100,511 | `44bb88ad6616062b412d725612531da005b43925899804baf28ae66ebe571629` | 3,435,071,021 | 8,406 |
| Qwen2 | 100,517 | `71f8f811341cb77d467edf371aae71840b17a2d3e25efe794c6eba375fc7cf25` | 4,434,645,168 | 8,406 |
| Pythia | 100,515 | `61cee51af1249a453e2dc37970b6cf681931c4fb22d2b47ef056c539010f52c0` | 3,812,172,016 | 8,406 |
| Total | 301,543 | — | 11,681,888,205 | 25,218 |

Across the recorded reachable inventories and capsule manifests I recomputed:

- campaign/success identifier matches: 0;
- paths containing `source_success`: 0;
- forbidden capability suffix paths: 0;
- capability archive signatures: 0;
- unsupported archive signatures: 0; and
- forbidden archive member paths: 0.

The public inventory commitment test removed the ordinary symlink row `/usr/bin/NF`,
recomputed the inventory hash, counts, enclosing summary evidence hash, result hash,
and receipt hash, and still failed with `reachable inventory release commitment
changed: layercake`. The same attack was independently replayed again in the full
hostile suite.

## 6. Missing inputs and the 19-case hostile receipt

I did not trust `status`, `mutations_rejected`, or `source_tree_restored`. I parsed the
19 raw mutation rows, required the exact unique mutation name set, recomputed the
receipt evidence hash, checked both verifier source hashes against public source, and
verified the baseline/post-restore digest equality. Every raw row recorded a
`StrictValidationError` with a nonempty failure observation.

I then copied all 1,193 extracted public files to
`disposable_hostile_r7`, installed the exact disposable marker, and ran the complete
public hostile runner. It independently rejected all 19 cases:

1. missing capability package;
2. missing required catalog;
3. corrupt capability package;
4. missing raw causality file;
5. missing raw causality row;
6. missing required raw hash;
7. stale immediate execution source;
8. stale isolated execution source;
9. stale transitive execution source;
10. missing raw mount table;
11. missing reachable inventory;
12. missing frozen adapter;
13. stale certification receipt binding;
14. fully rehashed missing ordinary inventory row;
15. fully rehashed missing native forward rows;
16. fully rehashed fabricated certification row;
17. fully rehashed capsule classification;
18. missing condition receipt; and
19. fully rehashed nondeterministic random intervention.

The disposable replay's baseline and post-restore strict evidence digest were both
`e12e0c81eea7f1a0007ba3d0ddd7a28ffbd4099e52ae17450a96526a5afbb876`.

## 7. Live execution rather than neutral-stub replay

The bound `live_causality.py` source has SHA-256
`6fb04da76f71fd7e11d6a6a9ee2ec93306f2cc7e75bcd0dde7d39f8aa63162df`.
It forks a condition worker using `subprocess.run` for every condition, contains the
live transformer forward `model(**inputs, use_cache=False)`, performs the native
parameter copy for interventions, executes `_generate` in each positive condition,
and does not reference the prohibited replay readers `_matrix_rows`, `_matrix_result`,
or `_source_references`.

For each of the three hosts I independently validated all eight condition receipts,
receipt hashes, raw condition hashes, parent/child PIDs, start times, adapter/package
hashes, immediate source hash, 733-file transitive source manifest, intervention hash,
and native execution identity. The 24 condition PIDs were globally unique.

For Qwen2 and Pythia, all non-host-removed receipts record
`live_transformer_checkpoint` with a positive native parameter count. Neutral, zero,
random, and shuffled conditions alter a real one-dimensional native parameter and run
a new forward. The deterministic random and shuffle values independently reconstruct
from their seeds. Host-removed receipts have no snapshot argument, zero parameters,
no native object, and `physically_removed_no_snapshot_no_object`.

All six positive conditions contain fresh capability and realized outputs, bound state
vectors, and no exception. Adapter removal still generates the capability output but
fails realization; capability removal fails generation. Counts are 128 adapter and 128
capability removals per host (384 each overall). All 128 selected real-host outputs are
identical across the three hosts. The public strict verifier also recomputed the 2,100
isolation rows to zero target successes.

Important boundary: `AppliedHostStateAdapter` validates and binds the live state but
does not make the native transformer generate the answer. Qwen/Pythia state is
noncausal to the capability answer in this tested runtime. That supports the stated
standalone capability-runtime/conformance result, not a native-model generation or
transplantation claim.

## 8. Public archive sufficiency

The public manifest-only run downloaded its own five payloads, verified them, extracted
the definitive archive, ran strict recomputation, and ran the focused tests. The
archive includes all 810 strict inputs, all 733 transitive runtime sources, all four
capabilities, raw evidence, receipts, tests, and operator material. It contains no Git
metadata, virtual environment, cache/build tree, host model weights, or source-teacher
runtime. No source-teacher inference was invoked by reconstruction.

This is sufficient to reproduce the claimed bounded technical certificate. It is not
sufficient—and does not claim—to reproduce model training, teacher acquisition, human
judgments, or an independent different-hardware live campaign.

## 9. Claim-language audit

The public release body is appropriately limited: it identifies this as an R7 repaired
technical validation candidate, reports the 19/19 pre-public hostile result, and states
that public reconstruction/blind review remained prerequisites while human ratings and
independent hardware were not claimed.

The current claim surface consistently denies the prohibited broad interpretations:

- no arbitrary-model or universal LLM compatibility;
- no transformer weight transplantation or hidden-state injection;
- no teacher-knowledge extraction or LoRA/distillation superiority;
- no human quality or completed human ratings;
- no independent-hardware reproduction; and
- no global minimum-information/English substrate.

I found non-material documentation drift: `README.md`, `docs/final-mile-status.md`,
`docs/research-status.md`, `docs/ABI_FINAL_RESULTS.md`, and parts of `review_packet/`
still describe R4 as pending and retain older counters such as 17 hostile cases,
95/98 inputs, and `11,681,818,650` scanned bytes. The controlling R7 release manifest,
frozen candidate, strict certificate, 19-case receipt, release body, and raw R7 paths
are current. The stale text underclaims or points to superseded evidence; it does not
broaden the technical claim or alter recomputation, so I do not treat it as a material
defect. It should nevertheless be refreshed before a public status promotion.

## Remaining non-technical gates

The PASS does not close:

1. 21,000 judgments from three real, distinct, independent human raters;
2. independent reproduction on different hardware; or
3. the registered minimum-information certification.

No assertion about the human identity/independence of raters, hardware independence,
or minimum-information optimality can be derived from this local technical review.

## Preserved failures and reviewer corrections

For audit transparency, I retained all reviewer-side failures:

- first official reconstruction: missing reviewer-installed `numpy`;
- second official reconstruction: missing reviewer-installed `cryptography`;
- first archive checker: incorrect `abi_release/abi_release` path join;
- second archive checker attempt: failed to account for sibling `layercake_release/`;
- first disposable copy: `robocopy` rejected a `\\?\` source spelling before copying;
- first tag/archive classification: treated expected public-only inputs and EOL filters
  as failures before distinguishing them.

None altered the clean clone or successful reconstruction. Corrected checks were run
in new workspaces or with preliminary results preserved under separate filenames.

## Command ledger

The exact console observations are in transcripts `01_*.txt` through `20_*.txt`. The
principal commands were:

```powershell
git clone --branch abi-final-validation-v2-repaired-r7-2026-08-30 --single-branch https://github.com/Yoder23/abi.git public_tag_clone
git rev-parse refs/tags/abi-final-validation-v2-repaired-r7-2026-08-30
git rev-parse refs/tags/abi-final-validation-v2-repaired-r7-2026-08-30^{}
git cat-file -p refs/tags/abi-final-validation-v2-repaired-r7-2026-08-30

gh release view abi-final-validation-v2-repaired-r7-2026-08-30 --repo Yoder23/abi --json tagName,targetCommitish,isDraft,isPrerelease,createdAt,publishedAt,url,assets
gh release download abi-final-validation-v2-repaired-r7-2026-08-30 --repo Yoder23/abi --pattern public_release_assets_r7.json --dir release_downloads

py -3.10 -m venv fresh_runtime_env
fresh_runtime_env\Scripts\python.exe -m pip install --no-cache-dir pytest torch==2.7.1 psutil==7.0.0 numpy==2.2.6 cryptography==46.0.5 huggingface-hub==0.36.0 safetensors==0.7.0 tokenizers==0.22.1 transformers==4.57.3

fresh_runtime_env\Scripts\python.exe -B -m abi_v2.public_reconstruction --manifest release_downloads\public_release_assets_r7.json --tag-clone public_tag_clone --workspace <174-character ordinary path> --output official_public_reconstruction_receipt_complete_lock.json

fresh_runtime_env\Scripts\python.exe -B -m pytest -q -p no:cacheprovider <extracted>\tests\test_abi_v2_recursive_archive_scan.py <extracted>\tests\test_abi_v2_reachable_inventory_commitment.py
fresh_runtime_env\Scripts\python.exe -B scanner_prefix_controls.py scanner_prefix_control_files scanner_prefix_controls.json
fresh_runtime_env\Scripts\python.exe -B independent_archive_audit.py <manifest> <assets> <extracted-prefix> independent_archive_audit.json
fresh_runtime_env\Scripts\python.exe -B independent_evidence_audit.py <extracted-root> independent_evidence_audit.json
fresh_runtime_env\Scripts\python.exe -B tag_and_certificate_binding_audit.py public_tag_clone <extracted-prefix> <manifest> tag_and_certificate_binding_audit.json

robocopy <extracted-prefix> disposable_hostile_r7 /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /MT:16 /NFL /NDL /NP
fresh_runtime_env\Scripts\python.exe -B -m abi_v2.strict_hostile --root disposable_hostile_r7\abi_release --output full_disposable_hostile_replay_receipt.json
fresh_runtime_env\Scripts\python.exe -B strict_stdout_capture.py <python> <extracted-root> strict_stdout_recomputed.json strict_stderr_recomputed.txt strict_stdout_identity_receipt.json
```

## Evidence index

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `blind_codex_r7_evidence.json` | Machine-readable final summary | computed after report creation |
| `official_public_reconstruction_receipt_complete_lock.json` | Successful official reconstruction | `736dc4c485680446abd00dffa8dff8b03336294311d349e71d6d2991d6fd3ac4` |
| `independent_archive_audit.json` | Full ZIP/member/extracted identity audit | `63395e870ff0d09d49bafa25f3afa6de2055edb7d1e0ccbf327e484db235ff41` |
| `tag_and_certificate_binding_audit.json` | Tag, source, candidate, and 810-input closure | `60d70283337c4c3a67f7f77324986599f077d8391ff75a9c463361cf75b32b0e` |
| `independent_evidence_audit.json` | Hostile, inventory, live-process, and source audit | `6e8d59137be498ab29ce0e59cfd53474d74a436059c3eee64dafa15468be264e` |
| `scanner_prefix_controls.json` | Explicit USTAR/V7 and invalid-literal controls | `4ce82fe5e5136450719390369e445be2a23383da250efb788260f72a57718252` |
| `full_disposable_hostile_replay_receipt.json` | Fresh complete 19-case replay | `22b215866abc7cfd6a6dd6e7c707c0240b0834f76afec7efbb7095ff572d3fb3` |
| `strict_stdout_identity_receipt.json` | Standalone strict stdout identity | `9e3b61f29f1f9f920b1c766b377bcc490694c0e5255f4ea27156b4d34d222ee5` |
| `strict_stdout_recomputed.json` | Complete strict stdout/certificate bytes | `17973df22cb31eddb3b47c6137aa71b68f32cd9d3010cfcddd232b2a1afce488` |

The machine-readable summary contains the complete decisive counts, hashes, scope, and
remaining gates.
