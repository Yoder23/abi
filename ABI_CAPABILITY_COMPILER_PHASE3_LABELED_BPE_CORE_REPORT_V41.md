# ABI Phase 3 Labeled BPE Core V41

Status: **COMPLETE FAILED — INTEGRATED LABELED CANDIDATE CLOSED**

V41 tested whether ABI's certified capability labels could materially improve
the failed plain sequence-imitation path without changing the deployed
LayerCake v3 graph. It added a 2,702-parameter training-only capability head,
50% deterministic synthetic-header dropout, and 10% causal-history corruption.
The final inference artifact retained exactly 4,174,280 deployed parameters and
contains no auxiliary head.

Training completed 4,000 steps in 99.50 seconds. The sampled capability-head
accuracy reached 100%, header views were balanced (31,824 body / 32,176 full),
and realized causal corruption was 9.93%. Full-prompt fit nevertheless declined
to 92.10% action accuracy and 80.09% exact sequences.

The locked autonomous suite remained 0/1,400 with no generation errors and no
locked repetition detections. As in V38, 1,394/1,400 raw outputs used the wrong
fact-free-reasoning mode. The paired candidate-minus-teacher functional
difference was -88.36 points (95% CI -89.93 to -86.79).

This shows that the training encoder can recover ABI capability labels, but a
discarded auxiliary objective does not make the unchanged causal decoder use
that separation on held-out prompts. An explicit, independently validated
route from inferred capability state into decoding is now the measured design
question. V41 does not authorize that architecture or any nearby ablation.

Decision SHA-256: `da9d7c961c50d0a3a49d67865be23fe3b647a48f9dd9e6bf3d079309e461e6f2`

Evidence SHA-256: `7302f8ff0ecc942eac127d7575de629a98d707bca034bb9309a8969ebeda716f`

Checkpoint SHA-256: `8e9e9f8351e8c7190f395dae6161bf9ddb433494c37b67263a7f8c266975743f`

Phase 3 remains uncertified; Phase 2 human ratings remain deferred; Phase 4 is
locked; no LayerCake performance or ABI superiority claim is permitted.
