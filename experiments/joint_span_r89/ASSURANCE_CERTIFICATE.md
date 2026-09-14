# R88/R89 bounded-result assurance certificate

The frozen R88 package and its R89 holdout result passed the repaired strict
verifier, a ten-case hostile mutation audit, and fresh live execution replays.
This certificate is additive: the first hostile-audit failure is retained in
`HOSTILE_AUDIT_V1_FAILURE.md` and was not relabeled.

## Strict raw recomputation

- Receipt: `results/joint_span_r89/strict_verification_v3.json`
- Receipt SHA-256: `3a0ccd40b94f31322eb10552c165bf1113d821fc85353f1828a35054b4ea2da3`
- Evidence digest: `d7d4d96612e67e3bb2815be48cf2f366e83ab5fb3b8b7e517d1cb978615fbecd`
- Stored scientific booleans trusted: no
- Raw rows independently recomputed: 2,800

The verifier requires exact byte lengths and hashes for the candidate bridge,
candidate metadata, parent checkpoint and metadata, all five parent tokenizer
files, imported artifact, catalogs, source rows, evaluation rows, source and
candidate results, and prospective bindings. It recomputes evaluators, span
realization, collapse, family totals, retention, aggregate gates, and paired
bootstrap intervals from raw rows.

## Hostile audit

- Receipt: `results/joint_span_r89/hostile_audit_v2.json`
- Receipt SHA-256: `f91fdaedd417e36636a8ea3a6469701de396296f069fa17f3c661c6c37b4933e`
- Evidence digest: `8498195c3cd4cf50a73e1a55fffd6484cf7052097d33f9382121cb9f3ca22f4b`
- Clean baseline: accepted with a receipt
- Hostile mutations rejected without a receipt: 10/10

The rejected cases cover a missing or truncated raw matrix, a forged gate,
mutated candidate, parent, imported artifact, candidate binding, holdout code,
holdout catalog, and a missing tokenizer file. The audit operates on disposable
same-volume hardlink trees and unlinks before mutation; original evidence was
not changed.

## Fresh live execution

The unchanged R88 package was executed again against both frozen catalogs. No
training or calibration occurred. Runtime-dependent result receipts differ as
expected, while the complete output rows reproduce byte-for-byte:

- R88 replay result SHA-256: `e032fa1b613f76faac01e8abe41f5dc8fef68a93f265963bc9d09eee0fcb64ca`
- R88 replay evidence digest: `867cfb6b9d0bbd9b1001a44859edd793dcd48e55499980a4d4a0b7b19a06eb2f`
- R88 replay raw SHA-256: `758643dd8eb2384f7aa8db757f0da82f69bbeea62816a34425cd4be0bc326dbb`
- R89 replay result SHA-256: `5417d5752cde98fcfa34288f397043ea8dd83a006e49e11e439d59019f49abc0`
- R89 replay evidence digest: `19fcb67fb889e2bd0afdd9364d0702fb2ae1c30fd6aad549784d82dbb6cec5a2`
- R89 replay raw SHA-256: `23dd0fd2b02a2526498255d72a2639eb2242a3d99a52816777a2c6550a9e7708`

Both replays passed 1,400/1,400, retained every source-passing row, recorded no
collapse, rejected the random bridge at 0/1,400, and physically traced sparse
package execution on all rows.

## Claim boundary

This certifies assurance and live reproducibility for the bounded R88/R89
extractive capability result. It does not certify broad generative English,
unrestricted domains, globally minimal information, fair superiority to LoRA
or distillation, human preference, independent hardware, or the full ABI
moonshot. The full ABI moonshot remains `OPEN`.
