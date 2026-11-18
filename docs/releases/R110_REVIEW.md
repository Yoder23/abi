# R110 review packet — scope and sentence adherence

Research snapshot: `abi-research-r110-2026-11-18`.
Result: **`CONTROL_QUALIFICATION_FAILED`**, with all 384 calls complete and exact
scientific replay. This is a completed negative prerequisite result, not a
qualified English control, acquired core or product certificate.

| Measure | Plain | Response contract | Gate |
| --- | --- | --- | --- |
| Panel A | 33/48 | 31/48 | 44/48 |
| Panel B | 34/48 | 34/48 | 44/48 |
| Both counterfactual members pass | 31/48 | 28/48 | 44/48 |
| Sentence-limit adherence | 6/24 | 8/24 | Unchanged 2–3-sentence obligation |

The sentence effect is +8.33 percentage points, 95% paired CI [-20.83, +37.50];
no improvement is supported. There are zero unchanged-input token mismatches
across 72 controls and zero collapse/budget stops. The scope grammar leaves
14 plain and 24 contract outputs unresolved; unresolved is a non-pass, not proof
of an actual contradiction. Sentence failures alone already preclude
qualification. See the [complete analysis](../R110_SCOPE_ADHERENCE.md).

## What this study can establish

R110 is a prerequisite experiment: a finite approval-scope evaluator challenge,
then one fixed plain-versus-response-contract comparison using the unchanged
Qwen3-4B-Instruct-2507 source. It does not acquire or train a LayerCake English
core. Neither ABI acquisition nor native LayerCake execution is tested. Matched
historical LoRA/distillation checkpoints are retained, not rerun or promoted here.

The prospective 600-case author-constructed scope challenge passed all six
100-case classes. All 164 historical fixtures and seven disclosed regressions
also passed. The two raw scope files are byte-identical. This is a finite
syntactic prerequisite, not an independent human semantic evaluation.

## Immutable chain

| Stage | Public tag |
| --- | --- |
| Design and exact inputs | `abi-prereg-r110-2026-11-17` |
| Evaluator, generator, execution and verifier freeze | `abi-r110-implementation-2026-11-17` |
| Actual challenge, published before scoring | `abi-r110-challenge-2026-11-18` |
| Complete passing scope packet, published before source execution | `abi-r110-scope-2026-11-18` |
| Complete source result and publication index | `abi-research-r110-2026-11-18` |

The [implementation contract](../../experiments/scope_adherence_r110/IMPLEMENTATION.md)
and [protocol](../../experiments/scope_adherence_r110/PROTOCOL.md) define the fixed
process order, source identity, resource limits and every quality gate. Neither
the cleanup nor publication helpers amend those files.

## Review without new model calls

The [publication manifest](../../results/review_2026_11_18/r110_manifest.json)
indexes exact bytes; the [strict recomputation receipt](../../results/review_2026_11_18/r110_strict_recomputation.json)
records a separate invocation of the unchanged full verifier and its elapsed time.
The [descriptive failure inventory](../../results/review_2026_11_18/r110_failure_inventory.json)
lists sentence-count failures and scope abstentions without rescoring or changing
any gate. These are same-machine checks, not independent-hardware reproduction.

The release attachment `r110_evidence.zip` contains the 40 new R110 evidence
files (340,196 bytes compressed), with SHA-256
`b681dc6f281943c21be0b660dc98f3c413888b33c18fa8edf5746119374b8c4f`.
Every archive member was compared by SHA-256 to its local evidence file before
publication. This archive does not replace the full dependency manifest.

Use the exact published repository, Git LFS payloads, historical host bindings
and pinned source cache listed in the manifest. The archive of new R110 evidence
alone does not contain that historical dependency closure or the source weights.
The source is `Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`.

```powershell
python -m experiments.scope_adherence_r110.run verify
python -m pytest -q tests/test_scope_adherence_r110.py tests/test_scope_adherence_r110_execution.py tests/test_r110_publication.py
python scripts/check_docs.py
```

The frozen verifier requires all four source executions, raw token IDs, decoded
outputs, receipts, complete source hashes, scope evidence and original dependencies.
It recomputes the scores and paired confidence intervals, rather than trusting
stored pass fields. It makes no model calls, but its token and payload validation
can be CPU-intensive. Do not rerun write-once `live`, `scope`, `freeze`, or
`materialize` commands over this packet.

Scope replay is byte-identical. Source scientific replay excludes only
`wall_seconds` and `storage_before`; the whole source JSONL files are not expected
to have identical bytes. Replay observations are not extra independent samples.
The 96 prompts instantiate known templates, not unseen semantic task families.

## Storage and historical custody

[Targeted cleanup](../R110_CLEANUP.md) freed about 16.24 GiB net by retiring only
two redundant local review checkouts. All 416 differing files were relocated and
rehashed, all old Git objects remain in the primary repositories, and all 17 root
receipts/scripts were retained. No primary checkpoint, shared source cache, R109
evidence, GPU setting or quality gate changed. Local preservation mappings are
published; checkout metadata itself is kept locally.

The supported final suite passed 555 tests with four declared skips, plus
16 exact historical Phase 7 tests and five separate publication-custody tests.
These are software checks, not English-quality
or independent-hardware proof. The whole-moonshot verdict remains
`FULL_MOONSHOT_NOT_PROVEN`.

Receipts: [supported suite](../../results/review_2026_11_18/r110-final-default.xml),
[historical checks](../../results/review_2026_11_18/r110-final-phase7.xml), and
[publication checks](../../results/review_2026_11_18/r110-publication-final.xml).
