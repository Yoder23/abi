# R54 preregistration: six-block layerwise sparse capability control

R53 localized the failure: capability labeling, external routing, and physical
post-transformer cake selection were exact, but an isolated late residual
could not reliably govern autonomous generation.  R54 changes the causal
injection point rather than sweeping the failed representation.  It adds one
zero-initialized, route-selected width-768 control vector before each of the
six frozen transformer blocks while retaining the copied post-transformer
capability cake.

The construction is function-preserving at initialization.  For each sequence
only one capability's six vectors and one post-transformer cake are selected.
The controls add 4,608 active scalar parameters and no cache positions,
attention tokens, or source-model dependency.  Embeddings, all transformer
weights, final normalization, and output head remain bit-exact to R47.

## Frozen inputs and training

- R47 parent checkpoint and metadata: the R53-registered hashes;
- broad and anchor archives: the R53-registered hashes;
- canonical ABI and catalog: the R53-registered hashes;
- response-blind R53 router:
  `ba88cf2a18ada898d739aa14a25805dae1c5212ffdcb4bfbfbd99c1c43b682db`;
- seed 54,001;
- `layerwise_capability_control_cakes`, CUDA, 6,000 successful steps;
- main microbatch 8, anchor microbatch 4, capability-balanced sampling;
- shared/cake rates 2e-5/1e-4, classifier 0.25, prompt overlap 1.0,
  parent preservation 0.5, weight decay 0.01;
- max tokens 256 with exactly the two previously enumerated exclusions; and
- autonomous recovery 400/every-8/[8,16,32].

The source teacher, validation outputs, evaluator outcomes, and R53 weights
are absent from training.  The external router is frozen and bypasses the
internal training-only router during evaluation.

## Decision rule

Disclosed validation must reach at least 1,277/1,400, at least 65/100 per
capability, at least 94% source-pass retention, zero collapse/errors, exact
routing, and exact sparse path selection.  If it passes, disclosed R50
final-test must reach at least 1,261/1,400 under the same gates.  Both are
development screens and cannot promote R54.

Passing both authorizes a new prospective catalog and live pinned-source
capture only after checkpoint, router, and verifier are frozen.  Failure
closes vector-only in-stack control; a successor must add nonlinear
capability-local capacity rather than tune this run.

R54 cannot establish unrestricted English, global minimality, human quality,
independent hardware, or LoRA/distillation superiority.  The ABI moonshot
remains open.
