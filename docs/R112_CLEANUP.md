# R112 storage cleanup and evidence custody

Date: 2026-11-19. This is storage/custody work, not a scientific experiment result.
The first two preflights were blocked. User-approved hard-link consolidation
restored the floor in the third observation; implementation development is now
tested, but scientific execution still requires its registered publication and
scope prerequisites. GPU settings, quality gates, R111, and LayerCake code are unchanged.

## Completed removal

Under the user's duplicate-only authorization, 61 historical copies totaling
**5,925,099,441 logical bytes (5.52 GiB)** were removed. Targets were exact files
in old clean-room/review trees, never a recursive directory deletion. Every
removal required matching SHA-256 and length against a retained copy, checked
again immediately before deletion and after it. No primary tracked path, Git
object, R111 dependency, source cache or unique checkpoint was removed.

The legacy `layercakeogwithdecoder` large-file candidates did not yield verified
duplicates in this plan. Matching names or file sizes were not accepted as proof.
Its unique data/checkpoints remain untouched.

Recovery is possible: copy each `retain` path back to its corresponding `remove`
path from the [exact plan](../results/review_2026_11_19/r112_cleanup_plan.json),
then verify the listed SHA-256. Retained copies must not be pruned. Some are Git
LFS objects; do not run LFS pruning without considering this recovery map.

Evidence:

- [Removal plan](../results/review_2026_11_19/r112_cleanup_plan.json), SHA-256
  `3d09bacb6059ceec83a479d74a3cc8bc575e6102557806ef44220c8928c76d23`.
- [Raw per-file action journal](../results/review_2026_11_19/r112_cleanup_actions.jsonl).
- [Completion receipt](../results/review_2026_11_19/r112_cleanup_complete.json).
- [Read-only remaining-duplicate audit](../results/review_2026_11_19/r112_remaining_duplicate_audit.json).
- [Second resource/custody preflight](../results/count_feedback_r112/preflight_002.json).

The immediate completion receipt observed 6,045,646,848 free bytes. The later
preflight observed **5,854,650,368 free bytes (5.45 GiB)**, a **4,882,767,872-byte
(4.55 GiB) deficit** against the unchanged 10 GiB floor. Concurrent disk usage
can change; neither number promises current free space. Logical removed bytes
and physical free-space change are reported separately.

## Preservation and validation

All 693 files and 182 host bindings in R111's immutable public manifest were
rehashed successfully after cleanup. R112's original registration and first
blocked preflight are byte-unchanged. No challenge was materialized/scored, no
model ran, and no implementation freeze is claimed.

Eleven offline custody tests pass, including ten malformed/tampered-packet
rejections: [passing receipt](../results/review_2026_11_19/r112-cleanup-custody-a2.xml).
The [initial receipt](../results/review_2026_11_19/r112-cleanup-custody-a1.xml)
preserves a Python 3.10 parsing failure on .NET's seven-digit UTC timestamps.
Only the new custody verifier was corrected; historical evidence was not edited.
These are software checks, not scientific replay or semantic certification.
The combined registration/custody check passes all 35 focused tests
([receipt](../results/review_2026_11_19/r112-cleanup-focused-a1.xml)); full local
supported testing remains deferred under the storage constraint.

```powershell
# Original workspace: rehash all retained copies and R111 dependencies.
python -m scripts.r112_cleanup_receipt verify
# Any checkout containing this compact packet: test custody consistency.
python -m pytest tests/test_r112_cleanup_custody.py -q
```

Absolute paths describe this local operation. A fresh checkout can audit the
compact records but cannot pretend to verify the original disk's physical state.
The `deleted_paths` field in the second preflight concerns that observation
itself; the linked cleanup journal explicitly records the earlier 61 removals.

## Approved hard-link consolidation

The follow-up read-only scan considered 3,122 files of at least 1 MiB and hashed
1,696 files with matching sizes, finding 240 duplicate-content groups. Many are
at historical scientific evidence paths that should remain available. The scan
is **not a deletion plan** and does not prove every duplicate path dispensable.

The user subsequently approved NTFS hard links. Two exact plans consolidated
**537 + 69 = 606 copies**, removing **5,870,357,993 logical duplicate bytes**
without removing an original evidence path. Already-linked files were excluded
from claimed new savings. Every link was verified by SHA-256, filesystem identity
and read-only status. No R111 dependency, source cache or Git metadata was linked
or removed. Historical tracked bytes remain unchanged; the LayerCake worktree is clean.

The first sandboxed creation was denied and rolled back; its one-row journal is
preserved. The same main plan then completed with elevated filesystem permission.
This was an operational retry, not a scientific retry. Rollback copies were
removed only after each new link was verified. Shared payloads are now read-only;
do not edit them in place. To restore separate storage, copy a payload into a new
file and replace one linked path only after verifying its hash.

- [Main plan](../results/review_2026_11_19/r112_hardlink_plan.json),
  [successful action journal](../results/review_2026_11_19/r112_hardlink_retry_actions.jsonl),
  [completion](../results/review_2026_11_19/r112_hardlink_retry_complete.json).
- [Preserved denied attempt](../results/review_2026_11_19/r112_hardlink_actions.jsonl)
  and [permission retry receipt](../results/review_2026_11_19/r112_hardlink_retry_authorization.json).
- [Auxiliary review-copy plan](../results/review_2026_11_19/r112_hardlink_aux_plan.json),
  [actions](../results/review_2026_11_19/r112_hardlink_aux_actions.jsonl),
  [completion](../results/review_2026_11_19/r112_hardlink_aux_complete.json).
- [Third preflight](../results/count_feedback_r112/preflight_003.json):
  **11,009,318,912 bytes free**, above the unchanged **10,737,418,240-byte** floor.
  All 693 indexed R111 files and 182 host bindings rechecked successfully.

This is one observed disk state, not a promise of later free space. Every
scientific stage and generation call must independently meet the original floor.
The new implementation's first unit-test receipt preserves one stale 384-call
expectation; it was corrected to the registered 576 before freezing anything.
[All 120 focused tests then passed](../results/review_2026_11_19/r112-implementation-a2.xml).

The unchanged implementation and challenge were subsequently published, and the
finite scope prerequisite passed 800 new cases plus all 1,598 known obligations
with exact replay. That packet is published at `abi-r112-scope-2026-11-19`.
Failure ownership remains separated; ABI acquisition and native LayerCake remain
**NOT_TESTED**.

## Final smaller-file hard-link batch

A separate read-only audit of smaller historical files identified another exact
plan: **790 copies / 432,326,392 logical bytes**. All original paths remain;
R111's full dependency closure, source caches, Git metadata and current R112
evidence were excluded. As before, SHA-256 and physical link identity are checked,
shared payloads become read-only, and rollback copies are removed only after
verification. The completion observed 10,718,302,208 free bytes before and
11,126,280,192 after; these are observations, not guaranteed future headroom.

- [Smaller-file inventory](../results/review_2026_11_19/r112_small_duplicate_audit.json).
- [Exact plan](../results/review_2026_11_19/r112_hardlink_small_plan.json), SHA-256
  `d3d04db85a4ad280ffe7d6a92a00218cca81ebf7f8ddacc77926210ace0e64f6`.
- [Action journal](../results/review_2026_11_19/r112_hardlink_small_actions.jsonl)
  and [completion](../results/review_2026_11_19/r112_hardlink_small_complete.json).

Total hard-link consolidation: **1,396 historical copies**, preserving every
path and byte while eliminating **6,302,684,385 logical duplicate bytes (5.87 GiB)**.
This is separate from the earlier 61 removed duplicate paths with retained
recovery copies. No unique evidence was removed. Do not modify hard-linked
historical files in place. The unchanged 10 GiB floor still applies before each
scientific stage and call.
