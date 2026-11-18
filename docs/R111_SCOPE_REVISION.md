# R111 — scope coverage and unconditional revision

Status: finite scope prerequisite passed; source execution pending.
No model calls, gate changes or GPU setting changes yet. [R110](R110_SCOPE_ADHERENCE.md) remains
an immutable negative prerequisite result, not an ABI acquisition or host test.

The [protocol](../experiments/scope_revision_r111/PROTOCOL.md) separates a new
800-case finite parser challenge from one conventional draft/revision comparison.
The revision always receives its own draft and original request, not scorer
feedback or a reference answer. Both stages retain the same quality gates.
The extra source call and larger 2048-token input envelope are explicitly
accounted for; no one-pass fluency, acquisition-efficiency or novelty claim.

The actual challenge must be published after the implementation freeze and
before scoring. A failed challenge closes GPU execution. If it passes, two
fresh source processes cover 96 paired prompts, draft + revision, with exact
replay (384 calls maximum). No training, recipient execution, native LayerCake
change or automatic follow-on. All existing matched controls are preserved.

## Implementation and custody

The published 800-case challenge passed all eight classes (100/100 each), plus
793 development obligations: 164 historical fixtures, seven regressions, 600
R110 cases and 22 disclosed R110 clauses. Fresh-process raw replay is byte-identical,
with no errors or false-safe decisions. This is finite author-constructed scope
validation, not independent semantic validation or English qualification.
The [scope decision](../results/scope_revision_r111/scope/decision.json) is published
under `abi-r111-scope-2026-11-18` before any GPU execution.
The actual challenge was published at `abi-r111-challenge-2026-11-18`.

The implementation code commit is `80dc070`; the publication tag is
`abi-r111-implementation-2026-11-18`. The [freeze](../results/scope_revision_r111/freeze.json)
has SHA-256 `8bfd26cccedcb07fd99197a1ae753e286d6357c010e15a1395ed0201abafeaa9`.
It binds the original preregistration, exact [population](../results/scope_revision_r111/population.json),
[baseline token inputs](../results/scope_revision_r111/baseline_inputs.json), source
cache, implementation/tests and full R110 public-manifest dependency closure.

The supported suite passed 623 tests (four declared skips), including 68 R111
development/integrity checks, plus 16 exact historical Phase 7 tests. These are
software/development checks, not the prospective challenge or an English-control
certificate. [Supported receipt](../results/review_2026_11_18/r111-implementation-default.xml);
[historical receipt](../results/review_2026_11_18/r111-implementation-phase7.xml).

Commands are write-once stages, not an automatic retry pipeline:

```powershell
python -m experiments.scope_revision_r111.run materialize
# Publish the actual challenge before scoring.
python -m experiments.scope_revision_r111.run scope
# Only after a passing scope decision is published:
python -m experiments.scope_revision_r111.run live
python -m experiments.scope_revision_r111.run verify
```

Stop on the first failed/incomplete prerequisite. Do not alter the frozen code,
repair a failed challenge, choose another seed or overwrite an execution directory.
