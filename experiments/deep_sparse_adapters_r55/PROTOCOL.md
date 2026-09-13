# R55 preregistration: six-block nonlinear sparse capability adapters

R54 exceeded the parent and source functional aggregates with only linear
route-local control, but produced nine verified repetition collapses.  Under
R54's decision rule, R55 closes the measured capacity defect by replacing each
pre-block control vector with a zero-initialized nonlinear rank-32 adapter
specific to that capability and layer.  This is not an R54 hyperparameter or
decoder sweep.

The fourteen post-transformer rank-64 cakes are still copied bit-exactly from
their canonical R47 parent routes.  The 84 in-stack adapters have zero output
at initialization, so construction is function-preserving.  At inference one
route selects six adapters plus one final cake.  Only 304,128 adapter
parameters and one rank-64 cake are active for a sequence; no foreign model,
extra token, or extra KV position is retained.

## Frozen inputs and training

- R47 parent, broad archive, anchor archive, canonical ABI, catalog, and R53
  response-blind router retain their previously registered hashes;
- R54 weights are not copied or used as initialization;
- seed 55,001;
- `deep_capability_adapter_cakes`, CUDA, 6,000 successful steps;
- main microbatch 8, anchor microbatch 4, capability-balanced sampling;
- shared/cake rates 2e-5/1e-4, classifier 0.25, prompt overlap 1.0,
  parent preservation 0.5, weight decay 0.01;
- max tokens 256 with the same two enumerated exclusions; and
- autonomous recovery 400/every-8/[8,16,32].

No source teacher, evaluator outcome, validation output, R53 output, or R54
output is training material.  The sealed external label router controls
evaluation routes.

## Decision rule

Disclosed validation must reach at least 1,277/1,400, at least 65/100 per
capability, at least 94% source-pass retention, zero collapse/errors, exact
routing, and exactly one capability route's adapters/cake per row.  If it
passes, disclosed R50 final-test must reach at least 1,261/1,400 under the same
gates.  Neither split can promote R55.

Passing both freezes R55 and authorizes one new prospective catalog and live
pinned-source capture.  Failure closes the rank-32 per-layer adapter design;
nearby ranks, rates, steps, or loss weights are prohibited unless profiling
isolates a joint quality/speed bottleneck.

R55 cannot establish unrestricted English, global minimality, human quality,
independent hardware, or LoRA/distillation superiority.  The ABI moonshot
remains open.
