# R57 preregistration: termination-balanced sparse capability adapters

R55's six-block rank-32 sparse capability adapters passed every disclosed
validation gate. On the frozen disclosed final split, they also exceeded the
parent aggregate and source-retention gates, but failed solely because three
of 1,400 generations did not terminate before the locked horizon. R56 proved
that a response-only no-repeat-four-gram decoder guard removes collapse only by
destroying valid copying and coherence. R56's decision rule therefore
authorizes one successor that changes training supervision for termination.

R57 changes no architecture, decoder, data, route, parent, or evaluation gate.
For a training record whose EOS target lies inside the 256-token context, its
equal-record language loss assigns exactly one-half of the record's mass to
non-EOS content and one-half to EOS. Prompt-overlap weighting remains confined
to content. A response truncated before EOS retains the ordinary content loss;
no fabricated terminal target replaces omitted source content. This is one
fixed mechanism, not an EOS-weight sweep.

## Frozen inputs and training

- untouched R47 parent checkpoint SHA-256
  `65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b`;
- broad archive SHA-256
  `82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150`;
- anchor archive SHA-256
  `f6a27cae529d990ba67f1a26bb9cd79f97ba5ca0965c9debd0fc98dab8dba820`;
- canonical ABI SHA-256
  `d024de52144a2d797d0501acb7deb55575ffca7e33f72900beff599cf0a97761`;
- R53 external response-blind router SHA-256
  `ba88cf2a18ada898d739aa14a25805dae1c5212ffdcb4bfbfbd99c1c43b682db`;
- R55 and R56 weights and outputs are not initialization or training data;
- seed 57,001, `deep_capability_adapter_cakes`, CUDA, 6,000 successful steps;
- main microbatch 8, anchor microbatch 4, capability-balanced sampling;
- shared/cake rates 2e-5/1e-4, classifier 0.25, prompt overlap 1.0,
  parent preservation 0.5, weight decay 0.01;
- max tokens 256 with the same two enumerated overlength-prompt exclusions;
- autonomous recovery 400/every-8/[8,16,32]; and
- greedy decoding with every repetition guard disabled.

The engine must inventory and hash all rows that do and do not expose EOS in
the training context. No source teacher, evaluator outcome, validation output,
R55 output, or R56 output is training material.

## Decision rule

Disclosed validation must reach at least 1,277/1,400, at least 65/100 per
capability, at least 94% source-pass retention, zero collapse/errors, exact
routing, and exactly one capability route's six adapters plus cake per row. If
it passes, disclosed final-test must reach at least 1,261/1,400 under the same
gates. Neither disclosed split can promote R57.

Passing both freezes the endpoint and authorizes exactly one newly generated
prospective 100-per-capability catalog, captured after the endpoint hash and
screen code are sealed. That fresh split must independently pass the same
quality, retention, collapse, routing, sparse-execution, source-removal, and
live-replay gates before this mechanism can be promoted.

Failure closes equal-mass terminal balancing. Nearby EOS fractions, steps,
rates, ranks, or decoder guards are prohibited. Any successor must use the raw
R57 evidence to isolate a different causal mechanism.

R57 cannot by itself establish unrestricted English, global minimality, human
quality, independent hardware, or superiority to LoRA/distillation. The full
ABI moonshot remains open until its separately registered gates pass.
