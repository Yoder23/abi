# R46 causal English-transfer feasibility certificate

Scientific verdict: **FAIL**.

R46 trained every parameter of the 7,176,097-parameter sparse LayerCake Phase
2 core on the immutable, semantically segregated Phi-3 archive.  Training used
24,421 source records, 2,784,714 authoritative teacher tokens, 1,745 balanced
GPU steps, and no logits, activations, source weights, or source model at
inference.  The constrained English planner was physically excluded from all
generation.

On 280 prompt-, output-, and record-disjoint validation prompts, the candidate
passed 6 functional evaluators (2.14%) versus 11 for the parent and 216 for the
stored source outputs.  It produced 38 repetition/collapse failures and three
invalid UTF-8 decodes.  Its paired candidate-minus-source difference was
-0.7500 (95% prompt-bootstrap CI -0.8000 to -0.6964).

This closes naive full-model causal fine-tuning of the 7.18M/2,816-vocabulary
host as an English acquisition route.  It does **not** invalidate the upstream
teacher extraction, labeling, segregation, or package evidence.  It does not
certify English transfer, model promotion, minimality, or the ABI moonshot.

The fail-closed verifier independently recomputed 840 raw evaluation rows,
24,430 scheduled training selections, all evaluator and collapse decisions,
the gate vector, input and artifact hashes, and 73 checkpoint tensor shapes.
It passed verification of the negative result.

Immutable receipts:

- result SHA-256: `637323763db55a5f802db0101e5da98633a347609d970571a33bb3572f12cc40`
- evidence SHA-256: `0d7f5807a4f42d89e5fba0c54e4b4f92af294e49a5a1e304fa97f07105b1f4f7`
- evaluation SHA-256: `fc6c68a005d5e01e654f79554b8dba442a3b1b6981914d10c3235adc706fbcc9`
- failed checkpoint SHA-256: `d7afc98fe49b6b8df2c6c087fdbf823f79ea4daaa04d50025042c687b1b26b0d`

Full ABI moonshot: **OPEN**.
