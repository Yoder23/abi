# R12-A public preflight v1: negative result

The immutable public attempt completed on 2026-08-31 and failed its frozen
teacher prerequisite. It is not an R12 transfer result and no held-out secret
was created.

- Qwen BEFORE exact accuracy: `0 / 512 = 0.0`
- Qwen AFTER exact accuracy: `68 / 512 = 0.1328125`
- AFTER atomic accuracy: `16 / 24 = 0.6666666666666666`
- extracted R11 package execution accuracy: `0.1953125`
- Qwen base state before/after: byte-identical by canonical tensor hash
- training-set accuracy: `2867 / 2904 = 0.9872589531680441`
- depth-4/depth-5 training accuracy: `1.0 / 0.9974279835390947`
- depth-6/depth-7 evaluation accuracy: `0.04716981132075472 / 0.15517241379310345`

The row-uniform source-training sampler disproportionately selected the 1,944
depth-5 rows while the atomic stratum contained only 24 rows. The saved source
adapter fit the long training programs but did not acquire the reusable atomic
transition or generalize compositionally. This attributes v1 to conventional
teacher acquisition, before the R11 transfer boundary.

Evidence identities:

- receipt evidence SHA-256:
  `b7ac0aa6c0681194817741aba87cf7498be5942b56b4db731507b0ff0fb3e557`
- failure-analysis evidence SHA-256:
  `d83151c9f7abfa1d2f1a71092f01969c0428b1964d55d6fbe1a86a0a27a9e537`
- saved adapter SHA-256:
  `381467bde2552497b5c89d53461c0176c1e66e47b7f4cb260a1b7d157c7b09b9`
- extracted negative package SHA-256:
  `7a4da9ba369a7a46eab294afdeaab6f94df9ffd0d1500e7a7f5e5f44f4c8465a`

V2 changes only the source-teacher optimization justified by this result:
equal per-depth sampling, learning rate `0.0002`, batch size `10`, and a
3,000-step ceiling. Data, exact gates, extractor, R11 package/runtime, frozen
host bindings, and evaluator semantics remain unchanged.
