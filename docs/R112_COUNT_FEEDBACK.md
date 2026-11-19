# R112 — sentence-count feedback and remaining scope cases

Status: **`RESOURCE_INCOMPLETE_NOT_QUALIFIED`**.

The source study stopped at the unchanged 10 GiB floor after **530/576 calls**:
288 initial and 242 replay. The worker measured 10,729,582,592 free bytes; the
controller then measured 10,726,658,048, both below 10,737,418,240. New R112 evidence
was only about 7.1 MB at the stop. The 242-row replay prefix is scientifically
exact, but the missing 46 calls and replay receipt prevent qualification. The
frozen complete verifier rejects it. No partial pass rate, confidence interval,
source qualification or ABI/host promotion is substituted for completion.

The [post-stop observation](../results/count_feedback_r112/interruption.json)
checks frozen inputs, the full initial rows, exact replay prefix, raw hashes and
measured storage exceptions. It is custody only, not a replacement certification.
See the [review packet](releases/R112_REVIEW.md). No restart or successor is authorized.

The [scope decision](../results/count_feedback_r112/scope/decision.json) derives
800/800 prospective cases and all 1,598 disclosed obligations correct, with zero
errors or false-safe decisions and byte-identical fresh-process replay. Both raw
files have SHA-256 `da8ff7f60370cc5081130de3088020c10a9cec77181ee4310464c5aba8beb544`.
Publication tag: `abi-r112-scope-2026-11-19`. This is finite author-constructed
validation, not general semantic or English certification. The source study is
incomplete as recorded above; no quality result or qualification is claimed.

Source commit: `fae959c9b0b3f0fee510b3c7d8f710acf9a1968a`.
[Freeze](../results/count_feedback_r112/freeze.json) SHA-256:
`6204639907ed39eeb50f7614eef6a4b64a4bb2c255117f2cc1f990a98b09b43b`.
The publication tag is `abi-r112-implementation-2026-11-19`. The actual challenge
must be published before its first scoring; no scientific outcome is implied.

The [actual 800-case challenge](../results/count_feedback_r112/challenge.json)
was materialized and published before scoring. SHA-256:
`73511ddee6f7de84be0a65553c661a5cdfdb45b103e0d7c7cbcb8eb0be708a8f`.
Its publication tag is `abi-r112-challenge-2026-11-19`.
The unchanged execution floor is 10 GiB free. The original preflight found less
than 300 MiB on the only available local volume; no scope challenge or model
call ran. The write-once observation is **288,468,992 bytes free (275.11 MiB)**,
which is **10,448,949,248 bytes (9.731 GiB) short** of the execution floor.
That original observation is preserved. The later user-authorized
[duplicate cleanup](R112_CLEANUP.md) removed 61 byte-identical historical copies,
with every retained copy rehashed and recovery paths recorded. The second
preflight found **5,854,650,368 bytes free (5.45 GiB)**, still **4,882,767,872 bytes
(4.55 GiB) short**. User-approved hard links then consolidated 606 additional
historical copies while preserving every path and byte. The third preflight
records **11,009,318,912 bytes free**, above 10 GiB. Earlier blocks remain
historical, not English-quality, ABI acquisition or native LayerCake results.
No GPU settings changed; every later execution stage must recheck the floor.
A final smaller-file batch consolidated another 790 copies without removing any
path: 1,396 total hard links / 6,302,684,385 logical duplicate bytes eliminated.
The complete plans and recovery guidance are in the linked cleanup report.
Two further historical-only plans completed during initial generation. Across all
five plans, 2,915 distinct target paths remain intact; all links and the full R111
closure rechecked successfully. These operations overlapped the first worker, so
timings are observed costs, not an isolated performance benchmark.

The [protocol](../experiments/count_feedback_r112/PROTOCOL.md) and
[machine-readable contract](../results/count_feedback_r112/contract.json) define
one bounded follow-up. R111's completed result and all its dependencies remain
unchanged. The registration tag is `abi-prereg-r112-2026-11-19`; the latest
completed research result remains [R111](R111_SCOPE_REVISION.md), not R112.

## Planned comparison

R111's one revision improved mechanical sentence adherence but left nine sentence
failures and three unresolved scope outputs in that arm. R112 isolates an explicit
counter signal: from one shared draft, compare the unchanged R111 self-revision
instruction against that same instruction plus the measured sentence count.
No semantic, literal, scope or pass/fail feedback is provided.

Only the 24 summary/tone prompts receive the extra scalar count; the other 72
inputs must be identical and produce exact greedy token matches. Both arms keep
their full raw outputs. No output repair, best-of selection or adaptive retry.
The fixed 96 prompts/48 pairs have new identifiers/days but known task templates.
Two fresh workers would make 576 calls total, including shared drafts, both
revision arms and exact replay. Each workflow's cost includes draft + revision.

This is conventional **counter-assisted revision**, not evaluator-independent
English, acquisition efficiency, ABI novelty or LoRA/distillation superiority.
Any later learner comparison must give every arm the same tool and call budget.
The historical matched checkpoints are preserved, not rerun in R112.

A separately frozen parser must first pass 800 new construction-labeled cases
and 1,598 known development obligations. The five newly disclosed construction
examples come from R111, not independent human labels. Unresolved remains a
non-pass; no quality threshold changes. The primary contrast is feedback minus
ordinary revision sentence adherence, with the same paired bootstrap rule and
all unchanged absolute gates. Neither relative improvement nor finite parser
success can replace those gates.

## What exists now

- [Design and exact feedback rendering](../experiments/count_feedback_r112/design.py).
- [Fixed prompt population](../results/count_feedback_r112/population.json) and
  [five disclosed scope cases](../results/count_feedback_r112/known_scope_cases.json).
- [Registration and parent custody](../results/count_feedback_r112/registration.json).
- [Write-once blocked preflight](../results/count_feedback_r112/preflight_001.json).
- [Post-cleanup blocked preflight](../results/count_feedback_r112/preflight_002.json)
  and [cleanup/recovery evidence](R112_CLEANUP.md). The original observation is
  not overwritten or relabeled as current.
- [Post-hardlink preflight](../results/count_feedback_r112/preflight_003.json)
  and [implementation contract](../experiments/count_feedback_r112/IMPLEMENTATION.md).
- The additive evaluator, independent challenge generator, bounded source driver
  and raw verifier under `execution/`. All 120 focused checks pass, including
  1,598 disclosed scope obligations; [receipt](../results/review_2026_11_19/r112-implementation-a2.xml).
  Preserve the [initial stale call-count test expectation](../results/review_2026_11_19/r112-implementation-a1.xml).
- Twenty-four lightweight design/custody tests, including unchanged gates,
  matched inputs, complete-draft retention, resource rejection and missing/
  changed parent dependencies. [Receipt](../results/review_2026_11_19/r112-registration-a2.xml).
  The earlier [20-test receipt](../results/review_2026_11_19/r112-registration-a1.xml)
  is preserved too. These are not scientific challenge or generation results.

All 693 repository files and 182 host bindings in R111's immutable public index
were hash-checked without alteration. Registration SHA-256:
`82788fa350c8c80f561c4d6ab55595b78ab2a09645ee64513c70b2e3a7929704`;
preflight SHA-256:
`561f42cfaf2828d1246e46760857c46b289c1a16e2346408f57b23228fbc0297`.
The separate registration-verifier invocation reproduced the recorded blocked
verdict. This checks custody and recorded arithmetic, not a new scientific replay.

The parser, challenge generator, live driver and scientific verifier are now
**implemented, unit-tested and frozen**. The finite scope stage has passed;
the source-generation comparison is resource-incomplete, not qualified.
The final offline software checks are recorded in the review packet; they are
not a quality claim. The write-once registrar itself still contains
no model runner and its immutable source set is unchanged.

```powershell
# Recompute the registration and recorded resource verdict; no model calls:
python -m experiments.count_feedback_r112.register verify
```

The recorded free-space observation describes that preflight only, not current
or independent hardware. Its verifier checks byte custody, input recomputation
and arithmetic, not retrospective physical disk state. Do not rerun `register`
over the preserved directory.

## Closed execution; no automatic resume

All publication prerequisites were met before the single authorized run. Its
resource stop is preserved, not erased by later disk recovery. Any completion
attempt needs separate prospective authorization and adequate storage headroom;
do not overwrite or silently continue the stopped workers. Unique evidence,
R111, source caches and all gates remain protected. No sweep, training, recipient,
native-host change or automatic follow-on is authorized.

Failure ownership remains explicit: resource, evaluator, experimental prerequisite;
ABI acquisition and native LayerCake are `NOT_TESTED`. The whole moonshot remains
`FULL_MOONSHOT_NOT_PROVEN`.
