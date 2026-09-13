# R58 preregistration: sparse learned termination controller

R55 passed its disclosed validation quality gates and missed its disclosed
final result only through three nonterminating generations. R56 showed that
changing token selection with a no-repeat rule destroys legitimate copying.
R57 showed that globally increasing EOS loss inside the language model reduces
quality and still leaves seven collapses. The combined evidence isolates the
remaining defect: token realization and response termination must not share a
single decision surface.

R58 freezes every R55 model tensor and trains a separate route-selected binary
termination controller. The controller reads the current final hidden vector
and may stop generation before the next token; it cannot select, replace,
mask, rescore, or rewrite any token. Each of fourteen routes owns a rank-16
two-layer controller. Exactly one controller (12,321 weights) is active per
sequence; 172,494 weights are installed. Threshold is fixed at logit zero
(probability 0.5), with no minimum-length or capability-specific threshold.

## Frozen inputs and training

- R55 host checkpoint SHA-256
  `4f302650a86964042996db1990e753388c84af7326253088f2a4bbeb698f1be6`;
- R55 metadata SHA-256
  `262fa8994dad29b8ded7039c5ce52a6e32af5ec91e2581b1bce220f06682b4ae`;
- broad archive SHA-256
  `82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150`;
- anchor archive SHA-256
  `f6a27cae529d990ba67f1a26bb9cd79f97ba5ca0965c9debd0fc98dab8dba820`;
- only rows whose complete source response and EOS fit in 256 tokens supervise
  the controller: 16,134 broad rows and all 1,438 anchor rows;
- seed 58,001; CUDA; 1,200 successful steps; main batch 16 and anchor batch 4;
- main sampling is deterministic shuffled-epoch, guaranteeing all 16,134
  broad rows are seen before reuse; anchor sampling is capability-balanced;
- AdamW, learning rate 1e-3, weight decay 0.01, gradient norm 1.0; and
- each complete record assigns equal binary-loss mass to its one positive
  terminal position and all nonterminal response positions together.

The R55 model runs under no-gradient inference and its checkpoint hash must be
unchanged before and after training. R55/R56/R57 validation and final outputs,
catalog labels, evaluators, and source results are absent from training.

## Decision rule

The controller and metadata are hash-bound before screening. Disclosed
validation must reach at least 1,277/1,400, at least 65/100 per capability, at
least 94% source-pass retention, zero collapse/errors, exact external routing,
exact model sparse execution, and exactly one termination-controller route per
decision. If it passes, disclosed final-test must reach at least 1,261/1,400
under the same gates. Neither disclosed split can promote R58.

Passing both authorizes exactly one new prospective catalog generated only
after the frozen endpoint and screen hashes exist. That fresh split must pass
the same gates with a new live pinned-source capture. Failure closes this
rank-16 hidden-state termination-controller mechanism; no rank, threshold,
step, rate, or minimum-length sweep is permitted.

R58 cannot by itself establish unrestricted English, global minimality, human
quality, independent hardware, or superiority to LoRA/distillation. The full
ABI moonshot remains open until separately registered gates pass.
