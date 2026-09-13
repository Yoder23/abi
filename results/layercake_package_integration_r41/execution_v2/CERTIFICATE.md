# R41 bounded certificate

Verdict: **PASS — real LayerCake package-boundary integration**

This certificate is bound to:

- LayerCake host commit
  `f8a4569508f9f4a20e4babe3b4a942ac887da147` (post-release, unsealed);
- R40 result SHA-256
  `50c6a5aabb8e416ec7e959919591246eca1193b36c0c6b707ad0c217f7f324d6`;
- R41 result SHA-256
  `d964ef7d591788211a6d54911b126c1e392a0b535fd2af8bc3027a6811f427c4`;
- R41 evidence digest
  `2bc135700c3164a3544d2da938467a3b1d90aea8ad7d2d776403ef27baceb314`.

The fail-closed independent verifier loaded all 12 signed `.cake` archives,
recompared every packaged tensor with its frozen R36/R39 source tensor,
recomputed all 288 raw CPU/CUDA observations, and independently replayed all
288 generations. It also repeated 12 remove/reinstall controls, rejected all
12 corrupted archives, and confirmed that one explicit request loaded exactly
one neural module while all 12 packages were installed.

Recomputed gates:

- CPU functional: 144/144
- CUDA functional: 144/144
- CPU/CUDA output identity: 144/144
- exact match to frozen prospective R40 output: 288/288
- tensor identity: 12/12
- signed package verification: 12/12
- remove rejection and exact restoration: 12/12
- corrupted-package rejection: 12/12
- receiver training steps: 0
- teacher present at execution: no

LayerCake's focused extension tests pass (7/7 across the new and directly
related contracts). Its current-HEAD repository run reports 690 passed and 11
fail-closed sealed-lineage invalidation tests. Those failures are not R41
functional failures: LayerCake's charter explicitly identifies current HEAD as
a post-release unsealed lineage whose governed component changes must not
inherit the old Phase 6–8 seal. R41 does not reseal LayerCake or claim that the
old release certificate covers this new host format.

Claim ceiling: these twelve bounded teacher-derived neural components cross
the real signed LayerCake install/execute/remove/restore boundary unchanged and
retain their prospective supplied-content behavior. This is not proof of
unrestricted fluent English, arbitrary teacher/domain extraction, global
minimality, teacher-quality parity, or superiority to LoRA/distillation. The
full ABI moonshot remains **OPEN**.

