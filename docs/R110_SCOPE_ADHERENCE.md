# R110 — scope and adherence (execution pending)

Status: **implementation frozen, not executed**. No new model calls, training, recipient
execution or GPU setting changes. The user authorized implementation/freeze and
conditional execution. The evaluator, challenge generator, driver and verifiers
now exist under the [implementation contract](../experiments/scope_adherence_r110/IMPLEMENTATION.md).
The unchanged 10 GiB storage floor was restored by the authorized
[verified cleanup](R110_CLEANUP.md) on 2026-11-18. Development tests are not
the prospective challenge, and implementation is not a scientific certificate.

## Execution freeze and preserved storage blocks

The immutable implementation code commit is
`7e8496df219171827a2d7586f73766eafefcffba`. Its
[freeze manifest](../results/scope_adherence_r110/execution/implementation_freeze.json)
has SHA-256
`b7958188919f7b2b572ebba37661898b6d5ada48685b737e5f03f9998671d0d1`.
The publication tag is `abi-r110-implementation-2026-11-17`; the earlier
preregistration tag remains unchanged. The one content-derived 600-case challenge
is now [materialized](../results/scope_adherence_r110/execution/challenge.json),
SHA-256 `3af120c5be55f441732314033b8b38afdf5b384aef52c7bb4554f9e4c2107023`.
It must be published before scoring. No challenge score or model call is claimed
by this input publication.

The [first storage preflight](../results/scope_adherence_r110/execution/preflights/1794945749670695600.json)
observed 10,634,899,456 free bytes versus the required 10,737,418,240 bytes.
This is `BLOCKED_STORAGE_BEFORE_EXECUTION`, not a launched/failed scientific
attempt. No files were deleted. Recheck the floor before materialization, scope
evaluation and every source prompt; do not lower it to get a result.
The [post-freeze preflight](../results/scope_adherence_r110/execution/preflights/1794945943081190100.json)
remained blocked at 10,555,985,920 free bytes, 181,432,320 bytes below the floor.

Development checks passed 63 new implementation tests, including the historical
164 fixtures and seven known regressions, plus 75 preserved registration/R109
tests (138 focused tests total). The full local suite passed 555 tests with four
declared skips and 16 exact historical Phase 7 checks. See the
[focused receipt](../results/review_2026_11_17/r110-implementation-development-a2.xml),
[full-suite receipt](../results/review_2026_11_17/r110-implementation-final-default.xml)
and [historical receipt](../results/review_2026_11_17/r110-implementation-final-phase7.xml).
None scores the prospective challenge or qualifies English generation.

## Why this study

[R109](R109_TRAINED_GROUNDING.md) failed qualification with exact replay. It
separated 20 sentence-count failures from seven approval-detector scope errors.
Correcting the latter alone could not clear the fixed gates. R110 preserves
every R109 score and raw file; it neither blames nor changes LayerCake.

## Fixed sequence

| Stage | Required evidence | Stop rule |
| --- | --- | --- |
| Registration (preserved) | Protocol, numeric contract, 96 exact task inputs, 192 renderings and tokenizer IDs, R109 hashes | Missing or changed inputs fail integrity checks; no execution follows automatically |
| Implementation freeze (current) | New scope classifier, complete challenge generator, verifier, live driver and tests published before challenge scoring | No code/input freeze or storage floor, no challenge or GPU execution |
| Scope challenge (future) | All 164 historical fixtures, seven known regressions, then 600 constructed cases with fresh-process replay | Any challenge error stops R110 before GPU work; no retry or seed selection |
| Source comparison (conditional) | Plain versus one fixed response-contract interface, 96 prompts each plus replay | Any resource/integrity failure stops; quality failure cannot promote a control |

The response contract changes only the 24 summarization/tone inputs. It repeats
existing obligations in explicit metadata; it deletes no notes and supplies no
answer or example. The other 72 messages are identical and their output tokens
must match across interfaces. There is no constrained decoding, sentence clipping,
forced EOS, self-correction, best-of sampling or extra turn.

The primary endpoint is sentence-count adherence on 12 counterfactual-pair
clusters: 10,000 paired bootstrap draws, two-sided 95% interval, lower bound
strictly above zero. Replay is not an extra independent sample. Improvement
also requires the unchanged 44/48 panel and pair gates, 5/6 every-task floors,
zero collapse/truncation, exact literals, all grounding/format/polarity checks,
and exact scientific replay. Old and new evaluator scores are both reported on
new outputs. The all-task quality interval must also retain the -0.10 lower-bound
noninferiority safeguard. This separates interface effects from scoring changes.

These inputs instantiate known task templates with new identifiers/days; they
are not new semantic-family holdouts. The scope challenge is author-constructed,
not independent human evaluation or general semantic validation. A qualified
control would still not be an acquired English core or a host certificate.

## Bounds and ownership

The source remains the exact cached R109 Qwen3-4B snapshot and generation setup.
The conditional live budget is four processes, at most 384 calls, four hours
per process and 16.5 hours total. Offline evaluator checking is capped at 30
minutes. The evidence cap is 1 GiB with a 10 GiB free-space floor; no downloads,
new weights, optimizer states or GPU power/clock/thermal changes are allowed.
Timing ceilings are not performance claims. See the complete
[protocol](../experiments/scope_adherence_r110/PROTOCOL.md) and
[machine-readable contract](../experiments/scope_adherence_r110/contract.json).

Evaluator failure belongs to the evaluator; source/adherence failure is a
prerequisite failure. Neither establishes ABI acquisition failure or native
LayerCake failure. Matched LoRA/distillation histories and all recipient
checkpoints remain unchanged; no recipient run is part of R110. LayerCake code,
packages and seals are untouched, and its existing handoff status remains valid.

## Registration integrity, not experiment verification

The sealed [registration](../results/scope_adherence_r110/preregistration/registration.json)
binds 767 files, including the preserved historical dependencies,
[96 task inputs](../results/scope_adherence_r110/preregistration/population.json)
and [192 exact renderings/token sequences](../results/scope_adherence_r110/preregistration/rendered_inputs.json).
Its SHA-256 is
`a689dc73219d2783e4129e2cf92dc4af3e5c6363441197094d0fa7ecdbd47697`.
The design commit is `8bd821a381533e5d0b132e57ca59d2539148fa2f`;
the publication tag is `abi-prereg-r110-2026-11-17`.
The largest plain and response-contract inputs are respectively 232 and 302
tokens; both leave the complete, unchanged 640-token generation allowance.

Local checks passed 43 R110 registration tests and 32 preserved R109 tests.
The final supported suite passed 492 tests with four declared skips, followed
by 16 exact historical Phase 7 checks. The
[supported-suite receipt](../results/review_2026_11_17/r110-final-default.xml)
and [historical-check receipt](../results/review_2026_11_17/r110-final-phase7.xml)
are software-check evidence, not evaluator validation or scientific promotion.
R109's decision remains byte-identical at SHA-256
`57126b03256ca20623e6a527d9b2f38110f755ff003f0ccf170b3042e1762725`.

Registration integrity can be recomputed:

```powershell
python -m experiments.scope_adherence_r110.register verify
python -m pytest -q tests/test_scope_adherence_r110.py
```

The registration verifier uses the pinned local source cache and retained R109
dependencies. It hashes payloads and tokenizes inputs; it does not generate text.
Its successful status is `REGISTRATION_INTEGRITY_ONLY`, never a scientific pass.
Do not rerun the write-once `seal` command over published inputs. Implementation
and conditional execution are now authorized, but the published freezes, storage
floor and prospective evaluator prerequisite must all pass before GPU work.
