# R114 — scope and sentence-structured revision

Status: implementation frozen; prospective challenge not yet scored (2026-11-20).

Source commit: `ab161c4f734799adae7e69d305a8a0aeb8bdbe94`.
Freeze SHA-256: `69ffc52a25b63bbbfb4173960bc1c8f3c9db6dc56ca08a0aac6a3012c92c1e01`.
Publication tag: `abi-r114-implementation-2026-11-20`. The freeze binds the full
R113 closure, new scientific code/tests, exact inputs, source files and package
versions. Its current preflight measures 21.53 GiB free. R113 recomputes unchanged;
an old R112 storage-block line printed during ancestry verification is historical,
not a current block. The actual new challenge must be published before scoring.

The combined offline suite passes 920 tests with four existing skips; all 16
historical-host tests pass in their exact fixture. The 91 focused R114 checks
include all 2,411 historical/disclosed development cases, strict decoder rejection
and missing/forged evidence. These are software/development checks, not the
prospective scope result or English qualification.

Preserved pre-freeze failures: the first development run found an `as soon as`
scope regression and a missing test-fixture binding; the third run found another
test-fixture binding error. Both fixtures and the parser were corrected before
freezing; every failed receipt remains in `results/review_2026_11_20/r114-*.xml`.
No prospective challenge has been scored and no model has been run.

Implementation entry points: [design and codec](../experiments/sentence_structure_r114/design.py),
[scope grammar](../experiments/sentence_structure_r114/evaluator.py),
[construction generator](../experiments/sentence_structure_r114/challenge.py),
[development provenance](../experiments/sentence_structure_r114/known.py), and
[write-once driver/verifier](../experiments/sentence_structure_r114/run.py).
Use `python -m experiments.sentence_structure_r114.run verify` only after the
registered stages complete. Do not overwrite write-once evidence or retry a stop.

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
