# Independent review handoff — 2026-09-14

## Review status

This repository is a repaired candidate for a fresh bounded-evidence review.
It is not certified as the full ABI moonshot, and repository reproducibility
must be re-established from a new public clone.

The prior independent verdict is preserved unchanged at
[`../reviews/independent_2026-09-14/ABI_MOONSHOT_INDEPENDENT_AUDIT_2026-09-14.md`](../reviews/independent_2026-09-14/ABI_MOONSHOT_INDEPENDENT_AUDIT_2026-09-14.md).
Its machine-readable companion and receipts are in the same directory. The
later `bf674f8` fresh-clone failure and its repair criteria are preserved there
as `FRESH_CLONE_REPRODUCIBILITY_AUDIT_BF674F8.md`.

## What changed after the audit

1. Exact retained Phase 0-8 protocols, manifests, catalogs, raw rows,
   certificates, and compact package bytes were force-added at the paths and
   hashes expected by the frozen evidence. Historical evidence was not
   regenerated.
2. Default pytest collection now includes the Phase 0 certificate, bounded
   Phase 4 frontier, Phase 6-8 verifiers, R97 replay, and a real V1089 Phase 8
   manifest integration test. The integration test reads all 52 entries and
   verifies exact ABI and LayerCake Git objects plus published LFS payloads.
3. The retained R97 parent/tokenizer and V1089 Phase 7 handoff files were
   added to the Git-LFS review surface.
4. The JSON proof ledger now matches the Markdown ledger for R13-B and R97.
5. The exact Phase 5 replay deficit is machine-readable. Six baseline tensors
   were deleted in the recorded storage cleanup and are not locally
   recoverable. The original verifier continues to fail closed.

## Fresh reviewer prerequisites

- Clone with Git LFS enabled and run `git lfs pull`.
- Use Python 3.10 and install the project test dependencies.
- For the default repaired-surface suite, place a full, non-sparse LayerCake
  checkout beside ABI as `../layercake_release` and detach it at
  `662c5a9b7264a1a5478c9dfb656f35c450e2504f`. The R7 archive reconstruction
  separately binds LayerCake `a87a653dbdb1a4e5f713baf7bc508d508277e00d`;
  use a separate worktree for that lineage rather than swapping evidence.
- Do not reuse caches, ignored development files, or another checkout's
  generated evidence.
- Treat missing artifacts, hashes, raw rows, packages, or independent records
  as failures. Do not trust stored `pass` booleans.

## Required local commands

```powershell
git status --short
git lfs fsck
C:\Python310\python.exe -m pytest -q
C:\Python310\python.exe -m pytest -q tests/test_independent_audit_replay_surface.py
C:\Python310\python.exe -m pytest -q tests/test_capability_compiler_phase4_b40_frontier_verify.py
C:\Python310\python.exe -m pytest -q tests/test_capability_compiler_phase6_verify.py
C:\Python310\python.exe -m pytest -q tests/test_capability_compiler_phase7_verify.py
C:\Python310\python.exe -m pytest -q tests/test_capability_compiler_phase8_release_readiness.py tests/test_capability_compiler_phase8_local_rehearsal_verify.py tests/test_capability_compiler_phase8_manifest_integration.py
C:\Python310\python.exe -m pytest -q tests/test_capability_compiler_phase5_verify.py
```

The last command is expected to fail until the six exact Phase 5 baseline
tensors are restored or Phase 5 is prospectively rerun and resealed. If it
passes without those tensors, treat that as a verifier regression.

The local branch-only sterile rehearsal for this repair used ABI and LayerCake
as clean sibling checkouts and produced 173 passed, 3 skipped in the default
suite plus 76/76 in the focused replay group. The manual Phase 5 command
produced 13 expected setup failures, all beginning at the same first disclosed
tensor:
`headline_v997/L1_r8_lr1e-4_exp4_seed104729/adapters.safetensors`.
These local counts are navigation aids, not a substitute for the fresh public
clone requested here.

## Required verdict vocabulary

Return exactly one whole-project verdict:

- `FULL_MOONSHOT_PROVEN`
- `BOUNDED_RESULTS_PROVEN_FULL_MOONSHOT_NOT_PROVEN`
- `REPOSITORY_NOT_REPRODUCIBLE`

A local result cannot complete human review or independent different-hardware
Phase 8. A full pass additionally requires one coherent final artifact—not a
composition of R7 runtime evidence, V1089 systems evidence, and R97 quality
evidence.

## Copyable fresh-review prompt

```text
Act as an independent principal research scientist, ML systems auditor, and
hostile reproducibility reviewer. Audit the ABI repository from a fresh clone;
do not inherit conclusions from prior sessions and do not modify evidence.

Start with README.md, AGENTS.md, CURRENT_PROJECT_STATUS.md, ACTIVE_MISSION.md,
CLAIMS.md, docs/abi_proof_ledger.md, docs/abi_proof_ledger.json,
docs/INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md, and the complete untouched audit
under reviews/independent_2026-09-14/. Verify every cited SHA-256 yourself.

Clone with Git LFS enabled. Work in a new short-path directory with no ignored
development assets. Run git status, git lfs fsck, the default pytest suite,
the review-surface test, Phase 4 and Phase 6-8 verifier tests, the real Phase 8
manifest integration test, and the Phase 5 verifier
command exactly as specified in the handoff. Independently inspect
pyproject.toml collection rules. Confirm whether the earlier discrepancy was
test exclusion, order dependence, or both. Do not accept a stored scientific
boolean when raw evidence can be recomputed.

Use a full non-sparse LayerCake sibling detached at
662c5a9b7264a1a5478c9dfb656f35c450e2504f for the default/V1089 replay. Use a
separate LayerCake worktree at a87a653dbdb1a4e5f713baf7bc508d508277e00d for
R7 reconstruction. Never treat one checkout as both lineages.

For R7, V1089, and R97, build separate causal/evidence tables. Do not combine
quality, speed, memory, portability, human, or public-reconstruction evidence
across lineages. For each claim, identify the exact final artifact, source
teacher, host, package, checkpoint, raw rows, verifier, hardware, and public
payload. Verify R97 strict replay and all hostile mutations from the exact
published parent/tokenizer and package bytes. Verify whether the V1089 Phase 8
manifest can be reconstructed solely from public manifests and sibling tagged
LayerCake sources. Treat the declared Phase 5 tensor deficit as a failed clean
replay unless the exact hashes are physically present.

Audit these mandatory scientific gates separately: teacher-origin causality;
prospective prompt-disjoint generalization; teacher-absent LayerCake execution;
unchanged-parent/random/removal/shuffle controls; English/domain segregation;
broad open-ended English; automatic capability discovery; latent purity;
smallest sufficient information; same-artifact CPU/GPU/TTFT/RSS evidence;
equal-information, equal-compute, and matched-quality LoRA/distillation
comparisons; human ratings; different-hardware execution; and clean public
reconstruction.

Return exactly one verdict token: FULL_MOONSHOT_PROVEN,
BOUNDED_RESULTS_PROVEN_FULL_MOONSHOT_NOT_PROVEN, or
REPOSITORY_NOT_REPRODUCIBLE. Follow it with a claim-by-claim table, reproduced
commands and counts, every missing/stale artifact, critical/high/medium
findings, and the smallest exact next experiment or repair for each blocker.
Scientific incompleteness is not a software failure; software/replay defects
are not scientific disproof. Never infer external human or hardware evidence
from local files.
```
