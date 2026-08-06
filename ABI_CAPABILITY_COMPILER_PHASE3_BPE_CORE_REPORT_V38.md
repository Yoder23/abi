# ABI Phase 3 BPE Core V38

Status: **COMPLETE FAILED — FIXED-ACTION BPE CANDIDATE CLOSED**

V38 trained the one preregistered 4,174,280-parameter candidate on the bound
7,000-record acquisition IR. The repaired deterministic CUDA run completed all
4,000 optimizer steps in 94.41 seconds, used 321,081,344 peak allocated GPU
bytes, and produced a 16,704,448-byte checkpoint. It copied no teacher weights,
logits, activations, or transformer blocks; the teacher is absent at inference.

Teacher-forced training fit reached 95.13% action accuracy and 85.23% exact
sequences. This is a large exact-fit improvement over the sealed V23/V24
lexical candidates, but it misses the diagnostic 99% and 90% references.

Autonomous evaluation failed every functional gate: 0/1,400 passes, zero
reported generation errors, and zero detections by the locked repetition
metric. Outputs were non-empty but generalized into incorrect training-style
templates rather than the requested held-out capabilities. The paired
candidate-minus-teacher functional difference was -88.36 percentage points
(capability-stratified bootstrap 95% CI -89.93 to -86.79).

The independent recomputation binds the protocol, checkpoint, tokenizer,
configuration, all 7,000 fit rows, all 1,400 raw outputs, receipts, Wilson
intervals, and 10,000-replicate paired bootstrap.

Failure ownership is ABI acquisition/model generalization, not a LayerCake host
regression. LayerCake v3 had already passed independent construct and exact
16,800-sequence conformance checks. No LayerCake quality or speed is inherited.

Decision file SHA-256: `a2e713dba0b6d7682ef38322a6533de2b127231ced9a6273047be838f096ac04`

Evidence SHA-256: `4288e63e314506509a64e91aa3babe2caf717ffe5ae43cfdd720acb3471e55b3`

Checkpoint SHA-256: `3b607d5c4650549ce93407965372e9f755816f72a044832d2b1d6c779ee2135d`

This is negative development evidence. Phase 3 remains uncertified, Phase 2
human ratings remain deferred, Phase 4 remains locked, and no ABI superiority
claim is permitted.
