# R84 failure certificate

R84 is closed and **not promoted**.

- Candidate checkpoint: `45a8a4c39bee2ad1e5492df2a4906e80ade09eb7ac0fac9378a9947cb344bf0c`
- Candidate binding: `6552de71e58dd77508ade5474a2015eedee6129daaab398a1e1223417c7f3582`
- Screen result file: `0eb8c3e42884333d3502bafb87ac6c7cab893a361347416a93c8126696c12b28`
- Raw 1,400-row evidence: `695e82a06fc3623ea7cd55600175af8d93440ba5cb887bd4383a2c6e9fa1acb5`
- Recomputable evidence digest: `d6eadb401028f9f6733166efe6b2420834c651a857061808ddeaf1d2edfb9e0e`

The corrected R81 source passed 1,382/1,400. R84 passed 87/1,400 and
unchanged R78 passed 64/1,400, a paired gain of 0.01643 with 95% bootstrap
interval `[0.00857, 0.025]`. That small positive effect is far below the
predeclared 0.50 gate. Source retention was 87/1,382 (0.06295), conditional
selection was 608/1,400, and 36 candidate rows collapsed. Quality, family,
retention, causal-gain, and collapse gates failed. Physical sparsity, artifact
immutability, source quality, and teacher absence passed.

R83 and R84 jointly reject a stateless token-mixture pointer for this bounded
task: aligning the exact deployed mixture improves generation slightly but
does not supply the missing multi-step selection state or atomic realization.
The evidence-supported successor is a stateful neural span selector that learns
start and length from ABI records, then copies only its selected prompt span.
The full ABI moonshot remains `OPEN`.
