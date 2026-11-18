# R111 — scope coverage and unconditional revision

Status: completed, **`CONTROL_QUALIFICATION_FAILED`**, with exact scientific replay.
All 384 source calls completed; no gate or GPU setting changes. [R110](R110_SCOPE_ADHERENCE.md) remains
an immutable negative prerequisite result, not an ABI acquisition or host test.

The [protocol](../experiments/scope_revision_r111/PROTOCOL.md) separates a new
800-case finite parser challenge from one conventional draft/revision comparison.
The revision always receives its own draft and original request, not scorer
feedback or a reference answer. Both stages retain the same quality gates.
The extra source call and larger 2048-token input envelope are explicitly
accounted for; no one-pass fluency, acquisition-efficiency or novelty claim.

The actual challenge was published after the implementation freeze and before
scoring; its passing evidence was published before source execution. Two fresh
source processes covered 96 prompts in 48 pairs, draft + revision, with exact
replay (384 calls). No training, recipient execution, native LayerCake change or
automatic follow-on. All existing matched controls are preserved, not rerun here.

## Result and interpretation

| Measure | Draft | One revision | Unchanged gate |
| --- | --- | --- | --- |
| Panel A | 39/48 | 42/48 | 44/48 |
| Panel B | 38/48 | 44/48 | 44/48 |
| Both counterfactual members pass | 37/48 | 41/48 | 44/48 |
| Sentence-limit adherence | 7/24 | 15/24 | 2–3 sentences where requested |
| Unresolved scope classifications | 3/96 | 3/96 | Unresolved is a non-pass |
| Collapse / truncation | 0 / 0 | 0 / 0 | Zero |

The registered primary sentence effect is **+0.33333**, paired 95% bootstrap CI
**[+0.16667, +0.50000]** across 12 pair clusters. This supports a sentence-adherence
gain within this known-template, two-call workflow. It does **not** satisfy the
experiment's combined improvement/qualification endpoint because absolute quality
gates fail. The all-task difference is +0.09375, CI [+0.02083, +0.17708] across
48 pair clusters, satisfying the registered noninferiority margin only as a
component. Replays are not additional independent observations.

Revision misses panel A and paired floors. Summary scores are 3/6 and 5/6 per
panel (3/6 paired); tone scores are 3/6 and 3/6 (2/6 paired), below the 5/6
task floor. Nine revised responses still violate sentence count, versus 17
drafts. Scope remains unresolved on three outputs in each arm; those flags are
not proof of actual contradictions. The unresolved revision IDs are
`r111-011-b`, `r111-021-a`, and `r111-034-a`; the complete outputs and scope spans
remain in the [initial raw rows](../results/scope_revision_r111/live/initial/raw.jsonl).

The frozen R110 scorer, applied descriptively to these **new** outputs, gives
32/48 draft pairs and 38/48 revision pairs. R111's corresponding 37/48 and 41/48
reflect the versioned finite grammar, not a historical R110 rescore or a pass.
The finite challenge's perfect score does not establish complete live-prose scope.

Failure owner: `EXPERIMENTAL_PREREQUISITE`; evaluator coverage limitations remain
explicit. ABI acquisition and native LayerCake are `NOT_TESTED`, with no English
artifact created. No broad-English, novelty, superiority, acquisition-efficiency
or product claim is promoted. Preserve this result; no repair/retry in R111 and
no automatic next experiment. See the [review guide](releases/R111_REVIEW.md).

## Measured execution cost

Both fresh processes together generated 34,694 tokens in 384 calls. Process wall
time was 1,698.641 seconds; CPU process time was 1,595.047 seconds. The controller
loop took 1,724.609 seconds, excluding its final offline derivation. Generation
time was 791.447 seconds for draft calls and 810.471 seconds for revision calls;
the two-call workflow incurs both, not just the revision cost.

Peak 50-ms sampled process RSS was 4,667,396,096 bytes; peak CUDA allocation was
8,279,356,928 bytes. Source cache preparation was reused and hash-verified, with
zero new downloads, training steps or recipient calls. These are source-study
costs, not LayerCake inference or acquisition-efficiency measurements. No GPU
power, clock, thermal or driver setting was changed by the experiment.

## Implementation and custody

The published 800-case challenge passed all eight classes (100/100 each), plus
793 development obligations: 164 historical fixtures, seven regressions, 600
R110 cases and 22 disclosed R110 clauses. Fresh-process raw replay is byte-identical,
with no errors or false-safe decisions. This is finite author-constructed scope
validation, not independent semantic validation or English qualification.
The [scope decision](../results/scope_revision_r111/scope/decision.json) is published
under `abi-r111-scope-2026-11-18` before any GPU execution.
The actual challenge was published at `abi-r111-challenge-2026-11-18`.

The [final decision](../results/scope_revision_r111/live/decision.json) has SHA-256
`2a26b2ec4751e81835a7d0c47fa2aed9ad96786f7b710247a48516b52e7f22ce`.
The research snapshot is `abi-research-r111-2026-11-18`.

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

The final supported suite passed 630 tests (four declared skips), including
seven publication-custody checks, plus 16 exact historical tests. Two earlier
Windows temporary-path failures and their receipts are preserved in the
[review guide](releases/R111_REVIEW.md); no test or scientific implementation
was changed to obtain the final pass.

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
