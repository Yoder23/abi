# R54 six-block layerwise sparse control — negative certificate

Status: **FAILED ON ZERO-COLLAPSE; VECTOR-ONLY BRANCH CLOSED**

R54 tested the preregistered causal correction to R53: add one
capability-selected width-768 vector before each frozen transformer block while
retaining one post-transformer capability cake.

## Frozen artifacts

- candidate checkpoint SHA-256:
  `ed8e5269cfa2ed95141b1f71f0cced11dd971af42418dcb509aa8dc1a2288e48`
- candidate metadata SHA-256:
  `4d137f27ce793d6448420928d8c3e1c1f6ece7fa8cd5a67807abe31617efbab1`
- response-blind router SHA-256:
  `ba88cf2a18ada898d739aa14a25805dae1c5212ffdcb4bfbfbd99c1c43b682db`
- raw validation rows SHA-256:
  `df8edaa727fd86f8d8cafd119c9f271b81bb85f6e31371b5ac268bdfbbfe3ca6`
- result SHA-256:
  `ad0593d13d99356bebde4ab8791e90cf20dc9476f52d0d4758f6da6f38f44773`

The endpoint completed 6,000 successful CUDA steps.  Exactly 1,593,806
parameters trained.  The active in-stack control is only 4,608 scalars;
81,923,342 parameters remained frozen and their before/after hashes match.

## Disclosed validation outcome

- candidate: 1,302/1,400
- R47 parent gate: 1,277/1,400
- pinned source comparator: 1,220/1,400
- source-pass retention: 95.57%
- paired candidate-minus-source: +5.86 percentage points,
  bootstrap 95% CI [+3.93, +7.79]
- every capability: at least 73/100
- exact routes and physical one-cake rows: 1,400/1,400
- generation errors: 0
- collapses: 9 (eight abstention, one clarification)

The fail-closed verifier recomputed every raw row and freshly re-executed all
1,400 neural generations exactly.  R54 passed every gate except zero collapse.
The failures are degenerate token/subtoken loops such as repeated identifier
fragments, not an aggregation artifact.

## Decision

The disclosed final-test is prohibited.  R54 proves that a tiny in-stack
route-local signal has enough causal authority to exceed both parent and source
aggregate functional scores while preserving the shared host.  It does not
pass certification because repetition collapse is non-negotiable.  Per the
preregistered rule, vector-only control is closed; the successor must add
nonlinear, capability-local capacity inside the frozen stack rather than tune
R54 or relax decoding/evaluation after observing this split.

This result does not establish unrestricted English, global minimality, human
quality, independent hardware, or superiority to LoRA or distillation.  The
ABI moonshot remains **OPEN**.
