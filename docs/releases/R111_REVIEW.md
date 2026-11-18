# R111 review packet — scope coverage and unconditional revision

Research snapshot: `abi-research-r111-2026-11-18`.
Result: **`CONTROL_QUALIFICATION_FAILED`**, with all 384 calls complete and exact
scientific replay. Sentence adherence improves, but the control does not qualify.
The finite evaluator prerequisite passed 800 new cases and 793 development
obligations with byte-identical replay and no false-safe decision.

| Measure | Draft | One revision | Gate |
| --- | --- | --- | --- |
| Panel A | 39/48 | 42/48 | 44/48 |
| Panel B | 38/48 | 44/48 | 44/48 |
| Both members pass | 37/48 | 41/48 | 44/48 |
| Sentence adherence | 7/24 | 15/24 | Unchanged 2–3-sentence obligation |
| Unresolved scope | 3/96 | 3/96 | Non-pass |

The primary sentence effect is +33.33 percentage points, 95% paired CI
[+16.67, +50.00]. All-task difference is +9.375 points, CI [+2.083, +17.708].
Neither relative gain substitutes for the absolute gates: revision misses panel
A, paired quality, summary and tone floors. Nine revised outputs still miss
sentence count; three remain scope-unresolved. There is zero collapse/truncation.
The [full study guide](../R111_SCOPE_REVISION.md) gives task scores, costs and
limits. Failure belongs to the experimental prerequisite, not measured ABI
acquisition or native LayerCake execution, neither of which ran.

## Scope and limits

This is a prerequisite experiment, not an ABI acquisition or native LayerCake
test. The frozen Qwen3-4B-Instruct-2507 source makes a draft, then exactly one
unconditional revision from its original request and complete decoded draft.
The final workflow output is always the revision, even when worse. No scorer
feedback, answer, output repair, best-of choice, training or recipient execution.
Historical matched LoRA/distillation checkpoints remain unchanged, not rerun here.

The declared 2048-token input envelope retains the full request and draft plus
the 640-token generation allowance. It is larger than R110's 1024-token envelope;
the extra call and context cost preclude a one-pass or compute-efficiency claim.
All quality thresholds and hardware settings remain unchanged.

The 96 prompts are fresh identifier/day instances of known templates, not hidden
semantic families. The 48 counterfactual pairs are the independent analysis units;
replay does not add independent observations. Primary sentence analysis covers
24 responses in 12 pair clusters with a fixed 10,000-draw paired bootstrap.
Finite construction-labeled scope validation is not independent human semantics.

## Immutable chain

| Stage | Public tag |
| --- | --- |
| Preregistration | `abi-prereg-r111-2026-11-18` |
| Implementation and exact inputs | `abi-r111-implementation-2026-11-18` |
| Actual challenge, before scoring | `abi-r111-challenge-2026-11-18` |
| Passing scope packet, before source execution | `abi-r111-scope-2026-11-18` |
| Complete result and publication index | `abi-research-r111-2026-11-18` |

The [protocol](../../experiments/scope_revision_r111/PROTOCOL.md) and
[study guide](../R111_SCOPE_REVISION.md) define the gates, resource budget,
analysis, binding chain and failure ownership. R110 is preserved, never rescored
into a pass. No automatic next experiment is authorized.

## Verification and dependencies

The complete verifier requires the pinned source/tokenizer cache, historical
Git/LFS payloads and exact sibling host bindings in the freeze. New R111 evidence
alone is not a self-contained model/dependency archive. Do not delete historical
payloads just because this study makes no recipient calls.

The [public manifest](../../results/review_2026_11_18/r111_manifest.json) indexes
every required repository payload and separate host/source binding. The
[strict recomputation receipt](../../results/review_2026_11_18/r111_strict_recomputation.json)
records another invocation of the frozen verifier, with zero model calls.
The [archive receipt](../../results/review_2026_11_18/r111_archive.json) records
the SHA-256 and every verified member of the release attachment `r111_evidence.zip`.
That attachment contains only new R111 evidence, not historical dependencies or
source weights. Obtain those through Git/LFS and the pinned cache in the manifest.
No new clean-clone or independent-hardware reproduction is claimed by this seal.

```powershell
python -m experiments.scope_revision_r111.run verify
python -m pytest -q tests/test_scope_revision_r111.py tests/test_r111_publication.py
python scripts/check_docs.py
```

`verify` is a raw recomputation, not new inference. It must fail on an incomplete
or altered packet. Do not rerun write-once `freeze`, `materialize`, `scope` or
`live` stages over preserved evidence. Scope replay is byte-identical; source
scientific replay excludes only `wall_seconds` and `storage_before`.

Same-machine verification and software tests are not independent hardware or
human evaluation. The whole-moonshot verdict remains `FULL_MOONSHOT_NOT_PROVEN`.

## Software-check environment

The first final suite recorded 629 passes, four declared skips and one Windows
temporary-path failure while creating the unchanged physical-certification
fixture. Its [receipt](../../results/review_2026_11_18/r111-final-default.xml) is
preserved. The [second receipt](../../results/review_2026_11_18/r111-final-a2-default.xml)
preserves the same failure after backslashes in `PYTEST_ADDOPTS` were consumed
by its shell-like parsing, producing another overly long relative path. A direct
short-path [focused check](../../results/review_2026_11_18/r111-capsule-short-path.xml)
passes the unchanged test. Generated temporary fixtures were retained outside
the repository, not published as source or scientific evidence.

These are test-environment failures, not source-quality or native LayerCake
execution failures. The full unchanged suite then passed **630 tests**, with
four declared skips, plus **16 exact historical Phase 7 checks**, using short
workspace temporary paths and forward slashes in `PYTEST_ADDOPTS`. No test was
modified or newly skipped. Receipts: [supported suite](../../results/review_2026_11_18/r111-final-a3-default.xml)
and [historical checks](../../results/review_2026_11_18/r111-final-a3-phase7.xml).
Seven publication-custody tests are included in the 630; 68 frozen R111 tests
cover finite development obligations and hostile integrity cases.
On Windows, use a short `--basetemp` and short `TEMP`/`TMP` when the default
sandbox path would exceed legacy filesystem path limits.
