# Independent review handoff — 2026-09-14

## Review status

This repository is ready for a fresh bounded-evidence review after the
2026-09-14 audit response. It is not certified as the full ABI moonshot.

The prior independent verdict is preserved unchanged at
[`../reviews/independent_2026-09-14/ABI_MOONSHOT_INDEPENDENT_AUDIT_2026-09-14.md`](../reviews/independent_2026-09-14/ABI_MOONSHOT_INDEPENDENT_AUDIT_2026-09-14.md).
Its machine-readable companion and receipts are in the same directory. The
repair manifest pins every byte that was added to the public review surface.

## What changed after the audit

1. Exact historical certificates and the V6 catalog were restored at the
   root-relative locations expected by frozen verifiers. The curated originals
   remain under `evidence/current/`; the test requires byte identity.
2. Default pytest collection now includes the supported Phase 0 and Phase 6-8
   verifier modules. The earlier 84-test pass did not run those historical
   modules; that was exclusion by `python_files`, not demonstrated test-order
   dependence.
3. The retained R97 parent/tokenizer and V1089 Phase 7 handoff files were
   added to the Git-LFS review surface.
4. The JSON proof ledger now matches the Markdown ledger for R13-B and R97.
5. The exact Phase 5 replay deficit is machine-readable. Six baseline tensors
   were deleted in the recorded storage cleanup and are not locally
   recoverable. The original verifier continues to fail closed.

## Fresh reviewer prerequisites

- Clone with Git LFS enabled and run `git lfs pull`.
- Use Python 3.10 and install the project test dependencies.
- Place a clean LayerCake clone beside ABI as `../layercake_release` when a
  replay explicitly requires that repository; verify the requested commit for
  the evidence lineage being tested.
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
C:\Python310\python.exe -m pytest -q tests/test_capability_compiler_phase6_verify.py
C:\Python310\python.exe -m pytest -q tests/test_capability_compiler_phase7_verify.py
C:\Python310\python.exe -m pytest -q tests/test_capability_compiler_phase8_release_readiness.py tests/test_capability_compiler_phase8_local_rehearsal_verify.py
C:\Python310\python.exe -m pytest -q tests/test_capability_compiler_phase5_verify.py
```

The last command is expected to fail until the six exact Phase 5 baseline
tensors are restored or Phase 5 is prospectively rerun and resealed. If it
passes without those tensors, treat that as a verifier regression.

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
the review-surface test, Phase 6-8 verifier tests, and the Phase 5 verifier
command exactly as specified in the handoff. Independently inspect
pyproject.toml collection rules. Confirm whether the earlier discrepancy was
test exclusion, order dependence, or both. Do not accept a stored scientific
boolean when raw evidence can be recomputed.

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
