# R74/R75 preregistration: causal host acquisition and disjoint reasoning validation

R73 is the first ABI artifact derived from a robust conditional-weight source
interface. R74 tests whether it causes new behavior in LayerCake rather than
merely existing as a valid archive.

## Frozen acquisition

- untouched R47 parent checkpoint SHA-256
  `65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b`;
- R73-v2 main artifact SHA-256
  `3450f510430ab4402b9a052d1a6b928f594bf2b759abc6e3c5baa872e8758b54`;
  v1 is preserved but training-ineligible because its selection declared a
  survey rather than the exact reasoning capability subset;
- broad-English preservation anchor SHA-256
  `82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150`;
- unchanged canonical LayerCake ABI SHA-256
  `d024de52144a2d797d0501acb7deb55575ffca7e33f72900beff599cf0a97761`;
- CUDA, seed 74001, six-block rank-32 deep capability adapters, 6,000
  successful steps, main batch 8, anchor batch 8, uniform main and
  capability-balanced anchor sampling, shared/cake learning rates 2e-5/1e-4,
  classifier weight 0.25, prompt-overlap weight 1.0, balanced terminal loss,
  parent-logit preservation weight 0.5, max 256 tokens, and autonomous prefix
  recovery from step 400 every eight steps over horizons 8/16/32. The same two
  previously enumerated overlength rows in the broad anchor are excluded;
  every R73 main row remains in scope.

No source teacher, R72 scalar score, expected validation output, validation
evaluator result, or prior candidate output enters training. The source model
must be absent from the deployed candidate.

## Frozen validation design

R75 contains 700 validation-only prompts: 100 examples for each of the seven
R72-supported premise families. It uses new numeric ranges and D/E/F class
codes instead of the A/B/C training surface. This tests functional transfer of
the supplied two-hop rule and output role rather than exact target replay.

The source and candidate are both evaluated on the same R75 rows. Source
answers are conditional-likelihood argmaxes over D/E/F candidates; LayerCake
answers are autonomous greedy generations on the single reasoning route.
Pass requires source and candidate each at least 665/700 (95%), every family
at least 90/100, candidate retention of source-passing rows at least 95%, zero
candidate repetition collapse, no errors, and physical execution of only the
reasoning route's six adapters plus one terminal cake.

The untouched R47 parent is also run live on the same rows as a causal negative
control using its canonical reasoning route. R74 additionally requires at
least a 50 percentage-point candidate improvement over that parent. The parent
result is descriptive outside this bounded interface and is not a claim that
R47 lacks general reasoning.

R75 is a bounded reasoning-transfer result, not general-English certification,
minimality, independent-hardware reproduction, human review, or the complete
ABI moonshot.
