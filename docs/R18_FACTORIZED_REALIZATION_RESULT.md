# R18 bounded factorized English realization

Status: `PUBLIC_PREREQUISITE_PASSED; HELD-OUT REPLICATION FAILED`

R18 replaces R17's failed all-or-nothing teacher-string prerequisite with a
feature-factorized grammar package and separate functional, source-agreement,
and causal scores. The compiler was frozen at commit `de8e54b` before the
public verdict and consumed only 72 anonymous teacher outputs, semantic
signatures, and lexical slots. It received no prompts, expected answers,
evaluation rows, success IDs, source model, or source weights.

## Public result

The unchanged R17-v2 source evidence was reused without another teacher run.
Compilation occurred in Linux pivot-root/no-network capsules. The resulting
3,881-byte package contains 24 slot templates and their evidence counts.

| Gate | Result |
| --- | ---: |
| Evaluation rows | 48 |
| Teacher functional exactness | 47/48 |
| Independent per-signature modal package | 46/48 |
| Factorized package functional exactness | 48/48 |
| Regressions on teacher-correct rows | 0 |
| Exact package/source string agreement | 47/48 |
| Mood-permutation control | 0/48 |
| Package-removed abstention | 48/48 |
| Final package bytes | 3,881 |
| Source/host training steps | 0/0 |
| Hostile mutations rejected | 12/12 |

A fresh physical replay reproduced the source bundle, control bundle,
evaluation rows, primary package, and control package byte-for-byte. The
teacher is absent at compilation and package execution.

## Interpretation

This is positive evidence for a small teacher-derived compositional English
surface-realization package in one registered family. Pooling only across the
subject-number feature repaired a noisy local majority that an independent
template package got wrong. The package also corrected one evaluation error
made by the source while preserving every source-correct evaluation.

It is not evidence of unrestricted English fluency, autonomous capability
discovery, arbitrary-domain extraction, production LayerCake ingestion,
global minimality, or superiority to LoRA or distillation. A preregistered
lexically hidden replication was the next gate.

## Held-out result

The implementation was frozen at `b56ef63`, the hidden seed was committed at
`db14613`, and it was revealed at `64612c8`. The unchanged GPU teacher then
produced 120 sealed outputs before compilation. All 24 signatures had 3/3
parseable extraction rows, so compilation was authorized.

The package scored 46/48 and regressed on two teacher-correct evaluation rows.
Both failures were present-positive plural questions: noisy extraction evidence
caused cross-number pooling to choose singular `Does` instead of plural `Do`.
The independent modal baseline was also 46/48, the teacher was 45/48, the
control remained 0/48, and removal remained 48/48. Strict recomputation
confirmed the negative verdict. R18 is therefore not certified.

This isolates a specific architectural error: number pooling is unsafe for
number-sensitive auxiliaries. A successor may test polarity-contrast
factorization that learns the positive form from its same-number negative
counterpart; it may not relabel or rerun this hidden selection as a pass.

The full ABI moonshot remains open, and R7 remains the controlling public
release.
