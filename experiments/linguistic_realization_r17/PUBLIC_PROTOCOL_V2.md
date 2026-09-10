# R17 public compositional English-realization prerequisite — interface v2

## Preserved v1 result

The frozen v1 source interface failed before compilation: 38/72 extraction
and 28/48 evaluation generations matched the functional oracle. All 120 raw
rows are preserved under `results/linguistic_realization_r17/public_v3/`.
Failures clustered in question mood, negative polarity, and, less often,
future tense. The v1 oracle also placed `not` before the subject in negative
questions, an unnatural uncontracted construction.

## Single bounded repair

V2 changes only the source-facing wording and the negative-question surface
form:

- semantic features use explicit natural descriptions;
- question, negative, future, and no-contraction constraints are stated
  independently;
- negative questions use standard uncontracted subject-before-`not` order,
  for example `Does Mira not inspect the alcove?`.

The 24 feature signatures, extraction/evaluation counts, lexical split,
teacher revision, compiler algorithm, physical isolation, package schema,
causal controls, accounting, and exact gates remain unchanged. V2 still does
not place the expected sentence in a prompt and does not supply prompts,
oracles, or evaluation rows to the compiler.

## Decision rule and claim ceiling

If the unchanged source does not reach 72/72 extraction and 48/48 evaluation
exactness, preserve the result and stop this interface branch. If it passes,
run the unchanged physical compiler and causal controls, then authorize a
hidden replication only if every original public gate passes.

A pass remains only a public prerequisite for bounded compositional surface
realization. It is not unrestricted English, autonomous discovery, LayerCake
acceptance, global minimality, or superiority to LoRA or distillation.
