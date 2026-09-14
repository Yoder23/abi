# R97 bounded prospective transfer certificate

Verdict: `PASS_R97_BOUNDED_PROSPECTIVE_TRANSFER`

The frozen 530,050-parameter R96 bridge passed all preregistered R97 gates on
1,400 prompts created after the candidate was frozen. The candidate scored
1,399/1,400 (99.93%); the live frozen teacher scored 1,088, the unchanged
LayerCake parent scored 60, and the seeded randomized bridge scored 17. Family
scores were 350, 350, 349, and 350 of 350.

The candidate retained 1,087/1,088 teacher-correct rows (99.91%) and exceeded
the teacher by 311 rows. The paired candidate-minus-source bootstrap interval
was [0.20071, 0.24429]. It showed zero collapse, physically executed the single
registered cake and six registered deep adapters on every row, and retained no
teacher parameters, logits, hidden activations, or transformer blocks.

Strict raw recomputation independently reconstructed all evaluator decisions,
token spans, token counts, collapse flags, invocation traces, aggregates,
bootstrap intervals, and gates without trusting stored scientific booleans. A
12-case hostile shadow-filesystem audit rejected every missing or altered
evidence class. A second live execution reproduced the evaluation JSONL
byte-for-byte.

- Candidate checkpoint: `9d3d6b38b1d437b5cbbed50c1470b4df1aa0371ffa51b1abd2adefbb645806e4`
- Candidate binding: `216f4092c2145d0ac1e63ebbb5ee28dc2ba726261c9929404d449bec11828397`
- Primary raw evaluation: `b4a4999228af61a509a7d22f8f6906b3075ce649a52e531b959646fbbdb40b8c`
- Primary evidence digest: `5caa7f0a46b2c0b377e08a3eb032d122a2515097ce9c6fa22720e9896ebaf9f9`
- Strict receipt: `59438fbff7e07a2d81f78b9c9905d473b09678892026b33cbd30cc0a02dfe30c`
- Hostile receipt: `cc34388e3b5ebcc11935da2a6f147df2247ebb61724b84957874b5354f1fb12e`
- Live replay raw evaluation: `b4a4999228af61a509a7d22f8f6906b3075ce649a52e531b959646fbbdb40b8c`

Claim boundary: bounded prospective, teacher-derived, two-hop nonce reasoning
transfer into a frozen LayerCake host. This does not establish broad English,
arbitrary-domain transfer, global minimality, fair superiority to LoRA or
distillation, human quality, independent hardware, or the full ABI moonshot.
Those claims remain `OPEN`.
