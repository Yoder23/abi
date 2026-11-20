# R114 — scope and sentence-structured revision

Status: **CONTROL_QUALIFICATION_FAILED**, complete execution (2026-11-20).

All 576 calls completed across two fresh processes with exact scientific replay.
All 72 unchanged-input controls match within each worker. Frozen raw recomputation
passes and all 14 actual-row hostile mutations reject. These validate evidence,
not the failed quality endpoint. [Review and costs](releases/R114_REVIEW.md).

| Arm | Panel A /48 | Panel B /48 | Paired /48 | Tone pairs /6 | Sentence adherence /24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Draft | 41 | 38 | 38 | 0 | 7 |
| Ordinary revision | 44 | 45 | 44 | 2 | 18 |
| Sentence-array revision | 41 | 40 | 39 | 1 | 12 |

The unchanged floors are 44/48 per panel and paired, and 5/6 per task. Ordinary
revision meets the aggregate floors but fails tone. Structured revision also
fails summary (2/6 pairs); twelve arrays contain one multi-sentence string and
are correctly rejected by the registered codec. No output is split or repaired.
All arms have zero collapse/truncation. Both revisions cost the draft plus a
second source call; neither is an acquired English core.

Structured minus ordinary sentence adherence is -0.25, 95% paired-cluster CI
[-0.58333, +0.125], so no supported improvement. All-task row-quality difference
averaged within paired clusters is -0.08333, CI [-0.16667, -0.01042]; its lower
bound fails the registered -0.10 noninferiority margin. These are not CIs for
the separate both-panels-pass counts in the table.

Finite scope passes, but general live-prose validity does not. Two structured
outputs remain unresolved; another says `no changes are confirmed` and is
apparently falsely classified affirmative. One ordinary revision has unresolved
scope, separately from six genuine sentence violations. Preserve all frozen
scores and [raw failure evidence](../results/review_2026_11_20/r114_error_audit.json).
This is prerequisite/interface failure with evaluator limitations, not an ABI
acquisition or native LayerCake result. R113 and matched controls remain intact.
GPU settings and all gates are unchanged. No automatic successor.

Decision SHA-256: `b18d9692e22ec6d8c8740ded5db54b4c838ac0d4a304fd10daa00e5a3c24b59c`.
Raw result: [decision](../results/sentence_structure_r114/live/decision.json).
Research snapshot: `abi-research-r114-2026-11-20`.

## Published prerequisite and implementation history

All 800 prospective construction cases and 2,411 development obligations pass,
with zero errors/false-safe classifications and byte-identical fresh-process
replay. Raw SHA-256: `d96fe902d2bcd72d290507356970ab9b5e95214dcbd01fc1ee6f9d9e4a30da51`.
The [scope decision](../results/sentence_structure_r114/scope/decision.json) and
complete packet are published at `abi-r114-scope-2026-11-20` before GPU work.
This is finite author-constructed scope validation, not general semantic or
English qualification. This packet preceded the now-completed source comparison.

Source commit: `ab161c4f734799adae7e69d305a8a0aeb8bdbe94`.
Freeze SHA-256: `69ffc52a25b63bbbfb4173960bc1c8f3c9db6dc56ca08a0aac6a3012c92c1e01`.
Publication tag: `abi-r114-implementation-2026-11-20`. The freeze binds the full
R113 closure, new scientific code/tests, exact inputs, source files and package
versions. Its current preflight measures 21.53 GiB free. R113 recomputes unchanged;
an old R112 storage-block line printed during ancestry verification is historical,
not a current block. All 1,026 bound file identities passed. The actual 800-case
challenge is published before scoring at `abi-r114-challenge-2026-11-20`:
SHA-256 `e855fb6ba904e51011ad0f3ec8322a5271c91d8f4f3c963305f50c885bb7afde`.

The combined offline suite passes 920 tests with four existing skips; all 16
historical-host tests pass in their exact fixture. The 91 focused R114 checks
include all 2,411 historical/disclosed development cases, strict decoder rejection
and missing/forged evidence. These are software/development checks, not the
prospective scope result or English qualification.

Preserved pre-freeze failures: the first development run found an `as soon as`
scope regression and a missing test-fixture binding; the third run found another
test-fixture binding error. Both fixtures and the parser were corrected before
freezing; every failed receipt remains in `results/review_2026_11_20/r114-*.xml`.
Those development failures preceded the frozen prospective pass above. No model
generation had occurred when the passing scope packet was published.

Implementation entry points: [design and codec](../experiments/sentence_structure_r114/design.py),
[scope grammar](../experiments/sentence_structure_r114/evaluator.py),
[construction generator](../experiments/sentence_structure_r114/challenge.py),
[development provenance](../experiments/sentence_structure_r114/known.py), and
[write-once driver/verifier](../experiments/sentence_structure_r114/run.py).
Use `python -m experiments.sentence_structure_r114.run verify` to recompute the
completed result. Do not overwrite write-once evidence or retry a stop.

The [protocol](../experiments/sentence_structure_r114/PROTOCOL.md) separates
finite evaluator falsification from a shared-draft comparison of prose versus
strict sentence-array revision on fresh instances of known tasks. The decoder
joins valid unmodified sentence strings; it cannot repair outputs. Malformed
arrays fail. All quality gates, GPU settings and resource floors remain fixed.
Scope must pass before generation; no training, recipients or host edits.

[R113](R113_STORAGE_RECOVERED_REPLICATION.md) stays completed and failed; R112
stays incomplete. No historical rescore or fluent-English artifact claim.
Historical matched controls remain preserved; future learning comparisons must
account for this conventional representation and the full two-call cost.
Full moonshot: **FULL_MOONSHOT_NOT_PROVEN**. No automatic successor.
