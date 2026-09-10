# R15B blind red-team report

**Verdict: PASS** for the bounded local claim
`BOUNDED_PREEXISTING_REPRESENTATION_CAPABILITY_RECOVERY`

Severity count: **Critical 0 · High 0 · Medium 4 · Low 2**

## Verified

- Exact clean HEAD: `c173e140953baa3c4220180baa42ef8e66c82edd`;
  `git status --short` returned empty.
- Linear chronology:
  - `0d4a75c` implementation freeze — 2026-09-09 23:17:53 -04:00
  - `8a51dae` preregistration — 23:19:45
  - `dafefc9` reveal — 23:19:53
  - `c173e14` certificate — 23:47:49
- Prereg commit added only `heldout_v1.json`; reveal commit added only the
  reveal.
- SHA-256 of revealed secret exactly matched commitment
  `7f3f17a85faa6785d232ae3e5b127cc460f438b59a142c571c8d59d0bace4e5a`.
- All 11 preregistered core code hashes matched the freeze, preregistration,
  and current tree. All four public-preflight bindings matched; the LFS object
  OID matched the configured public tensor hash. No bound file changed after
  preregistration.
- R11 freeze verification passed.
- Strict recomputation:
  - command: direct call to `verify(...)`
  - result: `PASS`, 4 capabilities, 40,000 package rows, recipients
    `pythia/qwen2/t5`.
- Focused tests:
  - `python -B -m pytest -q -p no:cacheprovider tests/test_preexisting_representation_r15b.py`
  - result: **9 passed in 1.59s**.
- Fresh GPU provenance replay against pinned local Qwen revision:
  - 24/24 regenerated source rows exactly matched;
  - all four regenerated residual tensors and output-head rows were bit-exact
    to committed bundles;
  - result: `bundle_tensors_exact=[true,true,true,true]`, 93.135 seconds.
- Standalone recipient replay in a new process, without loading the source
  model:
  - Pythia/Qwen2/T5 each reproduced 11,264 rows exactly;
  - all recipient states remained frozen;
  - total 33,792 rows, 28.28 seconds.
- Full `seal(...)` recomputation matched the committed certificate exactly,
  including strict PASS, live evidence, information accounting, and **10/10
  hostile mutations rejected**.
- Isolation transcripts show pivoted tmpfs root, read-only `/usr`, no old root
  or Windows mount, isolated network namespace, and a capsule inventory limited
  to bundle, specification, and frozen decoder.
- Packages are hash-addressed and strictly recompute byte-equivalent transitions
  from the bundles. Negative extraction controls remained at or below 0.2534;
  recipient AFTER/RESTORED were exact and removal conditions equaled BASE.

## Findings

### Medium

1. “Pre-answer” means only before the terminal `FINAL:` digit. The answer digit
   already appeared earlier in **24/24** reasoning prefixes; an answer/result
   phrase exposed it in 16/24. This is answer-conditioned next-token
   representation recovery, not extraction from a clean pre-solution latent
   state. The documentation discloses this.
2. The four held-out capabilities are permutations of only three public
   operations and reuse only **six unique prompts, completions, residuals, and
   probability vectors** across 24 rows. This supports secret slot-order
   recovery, not discovery of unseen semantics or knowledge.
3. The repository’s own live verifier regenerates source residuals but discards
   them and reuses the original bundles; strict verification also does not
   compare stored `residual_sha256`/probabilities to bundle tensors. My direct
   fresh replay closed that gap for this local audit, but the verifier should
   enforce it.
4. The source is revision-named but its complete snapshot contents and
   before/after model state are not hash-bound. Zero training is strongly
   supported by frozen code (`inference_mode`, `requires_grad=False`, no
   optimizer) and deterministic bit-exact replay, but canonical source-file
   identity is weaker than the recipient state hashing.

### Low

1. `verify_live.py`, `hostile_audit.py`, `accounting.py`, and `seal.py` were
   added post-reveal and are not preregistered code-bound. The primary runner,
   strict verifier, decoder, isolation code, gates, and protocol were frozen.
2. Public clean reconstruction remains pending. Strict/seal verification
   depends on locally ignored recipient rows, live rows/extractions, and a
   copied reveal—about 24.57 MB for the primary recipient rows alone—and on the
   external local model cache. The certificate correctly labels itself
   `LOCAL_SEAL_PENDING_PUBLIC_RECONSTRUCTION`; this is not public/independent
   reproducibility.

## Claim ceiling

This evidence supports only narrow recovery of a pretrained
modular-arithmetic transition under a registered ontology and secret slot
permutation, followed by exact frozen-package execution with zero R15B source
or recipient training.

**It does not prove pre-existing English or domain extraction.** It also does
not prove autonomous labeling, deep teacher-behavior cloning, LayerCake
ingestion, minimality, independent hardware reproduction, or superiority to
LoRA/distillation.
