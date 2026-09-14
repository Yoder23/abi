# R92 source-prerequisite failure certificate

Verdict: `FAIL_R92_SOURCE`

The pinned unchanged Phi-3 source reached 1,080/1,400 prior-corrected
candidate selections (77.14%), below the locked 1,330 authorization gate.
Family scores were 68, 189, 129, 197, 104, 197, and 196 out of 200. Corrected
ties were zero and all display positions were represented.

- Catalog SHA-256: `ae5686361a74f1539571f03568883532b17d139c65e77ba245cdea67b1ee6fee`
- Raw 1,400-row source evidence: `83a38803d2bcdd628b903e718a685a677d67d98ad5a74d1671aac076d7b9edc4`
- Evidence digest: `083db9a549abdc4022dc4c1be559b5de0707d94966323c4ba8b8d0cb93eb2a26`

The source read 3,821,079,552 frozen parameters and stored 8,400 scalar
candidate scores, zero full-vocabulary logit vectors, and zero hidden
activations. Source inference took 964.62 seconds on the declared GPU.

The R91 candidate and LayerCake host were not loaded, no R92 candidate binding
was created, and no candidate row was generated. R92 therefore supplies no
evidence for or against R91 realization. It is a source-interface failure and
does not promote a transfer claim. The full ABI moonshot remains `OPEN`.
