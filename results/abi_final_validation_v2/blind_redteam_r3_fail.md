# Blind Codex r3 red-team — preserved failure

Date: 2026-08-25

Tag audited: `abi-final-validation-v2-repaired-r3-2026-08-25`

Commit audited: `fcbaf847ff8bc302a50e1c9b72486bc9295c4e33`

Verdict: **FAIL**

The first attempted blind session was unable to create any local process under
its policy. It correctly failed closed because no claim was available to it.
Its raw transcript is retained locally as
`blind_codex_redteam_r3.txt`, 3,374 bytes, SHA-256
`40fe99ccf30573c796c6d177b1372d0513da2f9aa0142282ddef91337e303b48`.

A second fresh blind session obtained read-only access and independently
recomputed the substantive release. It passed the eight-condition live causal
execution, strict raw recomputation, rehashed fail-closed attacks, capability
and success-ID scan across the capsule/snapshots, local public reconstruction,
and closure of human/hardware gates. It nevertheless returned `FAIL` for two
mandatory blockers:

1. The capsule was exact, but the admitted `/usr`, `/etc`, and Python package
   runtime trees were not retained as a per-file content-bound raw inventory.
   The historical scanner inspected forbidden path names and suffixes rather
   than every file's contents. Physical absence across the complete reachable
   filesystem was therefore unrecomputable.
2. The review environment could not independently query the exact r3 GitHub
   release because outbound access was denied/cache-stale. Local downloaded
   assets and receipts were not accepted as proof of current public durability.

The second raw transcript is retained locally as
`blind_codex_redteam_r3_retry2.txt`, 6,248 bytes, SHA-256
`13cfdcb286f51b6ff6e73b850d6db9366f4056de168fb163806ca0ab15c30d75`.

R4 is an additive repair. It does not rewrite or relabel this failure.
