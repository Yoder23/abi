# R51 preregistration: broad-payload neural reconstruction

R50 established stable near-threshold held-out behavior from an 82.9M
LayerCake core trained on only 46,306 teacher tokens, but failed source-pass
retention by five cases.  The failures concentrate in reasoning, email,
conversation, instruction, grammar, and summarization.  R51 makes one
material, evidence-supported change: train the same LayerCake neural graph on
the already immutable 2.78M-token, 14-capability English-only ABI archive,
while retaining the earlier 46K-token corpus as an explicit anchor and the
frozen parent distribution as an anti-forgetting objective.

No R49 validation output, R50 final-test prompt/output/evaluator outcome, or
live R50 source output is training material.

## Frozen inputs

- Parent neural checkpoint SHA-256:
  `65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b`.
- Parent metadata SHA-256:
  `9590374b0afd3184dd75bcf08d8b9f0876ed7bcc0db7f7ec1f4abf147093d1d6`.
- Broad labeled English archive SHA-256:
  `82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150`.
  Its complete budget contains 24,421 rows and 2,784,714 authoritative Phi
  output tokens, all assigned to `english_core`, with zero domain-cake tokens,
  zero final-test rows, and no stored logits, activations, or source weights.
- Anchor archive SHA-256:
  `f6a27cae529d990ba67f1a26bb9cd79f97ba5ca0965c9debd0fc98dab8dba820`,
  containing 1,438 rows and 46,306 teacher tokens.
- Canonical LayerCake ABI SHA-256:
  `d024de52144a2d797d0501acb7deb55575ffca7e33f72900beff599cf0a97761`.

## Frozen training

- seed 51,001;
- CUDA, full existing core, unchanged tokenizer and graph;
- 6,000 successful optimizer steps;
- main microbatch 8 from the complete broad budget, capability-balanced;
- anchor microbatch 4 from the complete 46K-token archive,
  capability-balanced, weight 1.0;
- shared learning rate 2e-5, cake learning rate 1e-4, weight decay 0.01;
- classifier loss 0.25 and prompt-overlap loss 1.0;
- frozen-parent top-64 preservation weight 0.5;
- maximum sequence 256; and
- autonomous-prefix recovery begins at step 400 every eight steps at horizons
  8, 16, and 32.

The foreign Phi teacher is not loaded.  The only source information is the
fully accounted, labeled, normalized text in the two immutable archives.

## Decision sequence

1. Verify all input hashes, archive segregation, absence of final-test rows,
   and training accounting.
2. Use disclosed R49 validation and R50 final-test evidence only as
   developmental falsification screens.  R51 is rejected if it loses the
   parent's absolute quality, produces collapse, violates sparse execution,
   or does not address the measured retention failures.
3. If the development screen passes, freeze the R51 checkpoint and router.
4. Generate and commit a new, disjoint catalog lineage and capture its pinned
   Phi comparator before evaluating R51.  No R51 changes are permitted after
   that commitment.

No result on R49 or R50 can promote R51 because those splits are disclosed.
Only a new prospective replication, strict raw recomputation, live replay,
independent task families, human review, and later minimization/baseline work
can expand the claim.  The ABI moonshot remains open.
