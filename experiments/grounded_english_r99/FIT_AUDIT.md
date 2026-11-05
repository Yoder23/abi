# R99-D1: fit versus autonomous-grounding diagnostic

Registered 2026-11-05 after the failed R99 FP32 comparison and before this
diagnostic. This is not a new learner, a retry, or a replacement evaluation.
R99 checkpoints, development outputs, source data and verdict remain immutable.

## Question and budget

Is the measured missing-fact problem present on training examples, or primarily
on fresh identifiers? Does correct teacher-forced token prediction conceal a
free-generation failure? These observations select a subsequent representation
hypothesis; they cannot certify fluency, acquisition, or a LayerCake regression.

For each of the three exact final checkpoints and unchanged parent, measure:

- Autonomous greedy generation on 24 training rows, three per task selected by
  the ascending SHA-256 of `r99-fit-v1:` plus record ID. No selection by score.
- Teacher-forced response-token predictions on those 24 rows and all 48 frozen
  R99 development rows. Store every target token, argmax prediction, NLL,
  offset-derived response mask and supplied-fact mask. Report generated-plan
  positions separately from the visible response; do not let plan-copy tokens
  inflate response accuracy.
- The original 48 development generations are referenced, not rerun or replaced.

The supplied-fact mask includes exact occurrences of the publicly requested
identifier, coordinator and day in the source response, using the existing
`required` fields. It is an evaluation-only mask. Teacher forcing is an
oracle-prefix diagnostic, never a deployed or autonomous quality result.

No optimization, new teacher queries, decoding constraints, answer retrieval,
threshold change, precision sweep, or LayerCake source change. FP32, greedy
generation, unchanged maximum 640 tokens and route labels. Fresh process per
checkpoint; pinned tokenizer and all parent source bindings. Limit one hour
wall time and 1 GiB new disk; abort and preserve partial rows on failure.

## Interpretation rule

Report all four systems and both populations. A response-token accuracy of
at least 90% with fewer than 75% autonomous mechanical passes is an
exposure/realization warning, not evidence that more data or a larger model is
necessary. A larger train/development gap suggests representation/generalization
work; poor training fit suggests an acquisition-fit issue. These are diagnostic
hypotheses, not causal certainty or permission for nearby training sweeps.

Any successor must be separately preregistered, materially change the failed
generated-plan representation, keep matched LoRA/distillation controls, use new
evaluation prompts and retain the original promotion gates. Human quality,
source-free signed host acceptance and same-artifact CPU gates remain open.
