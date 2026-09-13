# R56 preregistration: response-only token-state repetition safeguard

R55 passed every disclosed quality, retention, routing, and sparse-execution
gate on validation.  It also cleared those gates on final-test, but three
verified subtoken loops violated zero collapse.  The loops repeat response
token four-grams; they are not failures of capability labeling, source
removal, routing, or aggregate functional acquisition.

R56 freezes the exact R55 checkpoint and router.  It makes no weight update
and selects no alternate checkpoint.  Greedy decoding is changed only by the
standard response-only no-repeat-four-gram rule already implemented in the ABI
runtime: when the current three-token response suffix has appeared before, the
previous continuation token is masked.  Prompt tokens are not part of this
history, so required copying from supplied text is not prohibited.  Existing
lexical truncation remains unchanged.

This rule is selected because the locked collapse metric is based on repeated
four-gram occurrences, not because of any particular observed identifier or
capability.  It is deterministic, model-agnostic, answer-agnostic, and stores
only response token IDs already required for decoding.

## Frozen inputs and decision rule

- exact R55 checkpoint, metadata, R53 router, catalog, source evidence, broad
  archive, anchor archive, and canonical ABI hashes remain frozen;
- training steps and teacher calls: zero;
- no-repeat n-gram size: exactly 4; no smaller/larger sweep is permitted;
- validation gates: R55's unchanged 1,277 aggregate, 65 per capability, 94%
  source retention, zero collapse/errors, exact routing/sparsity;
- final-test gates: unchanged 1,261 aggregate and all remaining gates.

Neither disclosed split can promote R56.  Passing both freezes the combined
neural/runtime artifact and authorizes one new prospective catalog and live
pinned-source capture created only afterward.  Failure closes this safeguard;
the next branch must change training supervision for sequence termination.

R56 cannot establish unrestricted English, global minimality, human quality,
independent hardware, or LoRA/distillation superiority.  The ABI moonshot
remains open.
