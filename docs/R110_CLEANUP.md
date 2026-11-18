# R110 targeted storage cleanup — 2026-11-18

The user authorized a targeted audit and cleanup, preserving all unique evidence,
to meet R110's unchanged 10 GiB free-space floor. No scientific code, quality
threshold, GPU setting, primary checkpoint or historical result was changed.

## What was removed and retained

Only these two redundant local review checkouts were removed:

- `_public_r102_20261110/abi_release`, pinned at `15570c2ef788b14770781b50984bd07217a7a38f`;
- `_public_r102_20261110/layercake_release`, pinned at `763e6da3cca1aa7564e6bce4b64dbee4effa552d`.

Both are under the shared workspace, **not** the primary repositories. The audit
resolved their exact absolute paths, rejected reparse points, checked for active
process references, and recorded every non-object file before any deletion.

| Preservation check | Result |
| --- | --- |
| ABI Git objects present in primary | 12,064 / 12,064 |
| LayerCake Git objects present in primary | 5,379 / 5,379 |
| These objects also reachable from primary refs | All; zero unreachable |
| Primary full Git integrity checks | Both completed with exit code zero |
| Non-object files inventoried and hashed | 7,932 |
| Files differing from their corresponding primary paths | 416; 46,481,806 bytes moved and rehashed |
| Exact duplicate non-object bytes retained in primary | 14,660,233,277 |
| Root reconstruction receipts/scripts retained unchanged | 17 |

The 416 unmatched/differing files are conservatively preserved, not asserted to
be globally unique. They include older catalog/data/ledger versions and checkout
metadata. They remain at the local path
`_public_r102_20261110/preservation_20261118/`, outside both deletion targets.
The audit maps every original file to its retained path and SHA-256. Duplicate
content and all Git history remain in the primary repositories; the checkout
directories themselves were removed, not sent to the Recycle Bin. Their content
can be reconstructed from retained objects/files and the preservation mapping.

The sibling `hf_cache` and `hf_cache_copy` directories were retained. R110's pinned
Qwen source is in the separate shared Hugging Face cache and was not a cleanup
target. The R109 reconstruction receipt remains byte-identical, as do its raw
results, freeze and decision. Retiring a local reconstruction workspace does not
turn the old same-machine reconstruction into independent reproduction.

## Space and evidence

Free space was 421,888,000 bytes at the audit start and 17,860,808,704 bytes after
cleanup: an observed net increase of 17,438,920,704 bytes (about 16.24 GiB).
Other machine activity can affect free-space observations. The subsequent
[R110 preflight](../results/scope_adherence_r110/execution/preflights/1795025977422592300.json)
measured 17,860,509,696 bytes free and returned `STORAGE_READY`. Every later
stage/prompt must still recheck the floor; this is not a permanent space guarantee.

- [Audit plan and file/hash mapping](../results/review_2026_11_18/cleanup_plan.json),
  SHA-256 `78abc5a544f8d0ce690a4f65370541a1d6caac66ae4754e410471055d6e5d941`.
- [Completion receipt](../results/review_2026_11_18/cleanup_completion.json),
  SHA-256 `dfe775059de543ccf75d2bf82d6b077befee462a3dd976fe3904e86565eff396`.
- [Two-stage cleanup script](../reviews/cleanup_r110_20261118.ps1).

The public plan and completion are byte-identical copies of the local custody
records. Preserved checkout metadata is retained locally, not indiscriminately
published. This is storage/custody evidence, not an ABI or LayerCake quality result.
The [frozen R110 study](R110_SCOPE_ADHERENCE.md) subsequently completed all 384
source calls but failed control qualification. No new experiment, prompt sweep,
training or gate relaxation follows from successful cleanup or that negative result.
