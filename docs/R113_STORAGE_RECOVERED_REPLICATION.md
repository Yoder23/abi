# R113 — storage-recovered count-feedback replication

Status: **IMPLEMENTATION AND INPUTS FROZEN; no generation yet** (2026-11-20).

Source implementation: `5efaf9f`. Freeze SHA-256:
`54b113c52f9dbb1b3a344cee9a10567d3d46c157a9f06a0150c3e70144529bdc`.
The freeze binds 983 repository/host paths plus the complete pinned source cache.
Inherited scope recomputation and exact input-token identity pass. Before launch,
publish the freeze at `abi-r113-implementation-2026-11-20`.

Software checks: 823 supported tests pass with four existing skips, four separate
publication checks pass, and all 16 historical Phase 7 tests pass in their exact
46-file fixture. The two direct Phase 7 invocations correctly rejected the newer
host's stale historical bindings; their failed receipts are preserved. No host
bytes, historical verifier or quality gate was changed. Focused execution tests
pass 133 checks; lint and documentation links pass. These are not model results.

[Protocol](../experiments/count_feedback_r113/PROTOCOL.md). This separately
authorized attempt repeats the frozen R112 comparison from the beginning in two
fresh processes. It does not resume R112 or splice its incomplete replay.
Prompts, model, evaluator, quality gates, GPU settings and budgets stay fixed.
The old panel is exposed; this is not fresh blind validation.

Storage cleanup recovered about 22.15 GiB without losing unique evidence:
1.45 GiB of verified public ZIP copies/disposable test fixtures were removed,
and two legacy token arrays were transparently compressed with identical hashes.
Source caches and Git were untouched. The local audit is outside both repos in
`storage_audit_20261120`; pre-inference telemetry, not this summary, controls launch.

R113 revalidates the inherited finite scope prerequisite, then performs 576
source calls including complete replay. Historical matched LoRA/distillation
checkpoints remain preserved. There is no training, recipient or native-host run.
A successful result could qualify only a conventional count-assisted workflow
on the known mechanical panel—not an English core, ABI novelty, or a breakthrough.
The whole-moonshot verdict remains `FULL_MOONSHOT_NOT_PROVEN`.

R112 remains [resource-incomplete](R112_COUNT_FEEDBACK.md). Every negative and
interrupted result stays independently addressable. No automatic successor.
