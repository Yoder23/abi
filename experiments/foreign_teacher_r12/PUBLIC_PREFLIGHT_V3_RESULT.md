# R12-A public preflight v3: near-exact teacher, exact extracted package

The immutable prompt-disjoint same-length attempt completed on 2026-08-31. It
did not pass the exact Qwen-teacher prerequisite, so it is not R12
certification and no held-out secret was created.

- training prompts: `15,192` unique depth-1–7 programs
- evaluation prompts: `512` unique depth-6/7 programs
- training/evaluation prompt overlap: `0`
- Qwen BEFORE exact accuracy: `0 / 512 = 0.0`
- Qwen AFTER final exact accuracy: `505 / 512 = 0.986328125`
- Qwen AFTER atomic accuracy: `24 / 24 = 1.0`
- extracted R11 package size: `2,053 bytes`
- extracted R11 package execution accuracy: `512 / 512 = 1.0`
- Qwen base state before/after: byte-identical by canonical tensor hash

The conventional Qwen teacher learned the registered public capability to
near-exact quality on unseen, prompt-disjoint same-length compositions. The
fixed zero-parameter extractor compiled its exact atomic behavior into an
unchanged R11 package that was exact on the full evaluation. This remains a
strict teacher-gate failure because seven native Qwen predictions differed.

Evidence identities:

- receipt evidence SHA-256:
  `eda8f77250e5500e0d184896474a1e5272d17d7b3551342db9c2f2051045098b`
- saved adapter SHA-256:
  `c1eaca410d8a36ee6eb22559dde910c933998bd0ed4012b414f866e75b1641d1`
- exact extracted package SHA-256:
  `c14095ba3e81d36b2799063f60c14ddf343fe370e1cb39fe5fdbaf89d07704d1`

V4 performs one bounded continuation from the exact V3 adapter hash with a
10x lower learning rate (`0.00001`), no new data, and a 1,000-step ceiling.
All gates and downstream artifacts remain unchanged.
