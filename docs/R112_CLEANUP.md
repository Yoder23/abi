# R112 storage cleanup and remaining blocker

Date: 2026-11-19. This is storage/custody work, not a scientific experiment result.
R112 remains `RESOURCE_BLOCKED_BEFORE_IMPLEMENTATION`. GPU settings, quality
gates, R111, and the LayerCake codebase are unchanged.

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

## Remaining storage decision

The follow-up read-only scan considered 3,122 files of at least 1 MiB and hashed
1,696 files with matching sizes, finding 240 duplicate-content groups. Many are
at historical scientific evidence paths that should remain available. The scan
is **not a deletion plan** and does not prove every duplicate path dispensable.

NTFS hard-link consolidation has been proposed, not performed: retain every path
and byte while storing identical historical content once. Linked paths would
share storage and must be treated as immutable. R111 dependencies, source caches
and Git metadata would remain untouched. This needs the user's decision and a
separate exact plan before execution. Do not erase evidence paths or relax the
10 GiB gate to force a launch.

Once storage is restored, implement, test and publish the unchanged
[R112 protocol](../experiments/count_feedback_r112/PROTOCOL.md), then publish and
pass its finite scope prerequisite before any source generation. Failure ownership
is currently **RESOURCE**; ABI acquisition and native LayerCake remain **NOT_TESTED**.
