# R60 prospective frozen-endpoint certificate

Status: **FAIL. R59 is not promoted. The full ABI moonshot remains OPEN.**

R60 was generated only after the R59 endpoint and both disclosed matrices were
sealed. It contains 1,400 unique prompts (100 per capability), has zero exact
overlap with 179,477 prompt strings found in all earlier JSON catalogs, and was
sealed before the teacher or LayerCake endpoint was loaded.

The pinned Phi-3 source was captured live first: 837/1,400 functional, 43,597
authoritative output tokens, zero collapse. Candidate access during source
capture was zero. The frozen R59 endpoint and its ordinary R55 parent were then
executed live on every row.

## Result

- candidate: 813/1,400;
- parent: 813/1,400;
- source: 837/1,400;
- source-pass retention: 81.720%;
- paired candidate-minus-source: -1.714 percentage points, bootstrap 95% CI
  [-4.071, +0.714];
- candidate raw collapses: 53;
- candidate final-output collapses: 2;
- R59 identical-token boundary stops: 67;
- router: 1,310/1,400, with all 90 errors in format control;
- exact physical selected-cake execution: 1,400/1,400;
- naturally non-boundary parent parity: 1,333/1,333; and
- generation errors: zero.

Failed gates: absolute functional quality, per-capability quality, exact route,
source point noninferiority, source bootstrap noninferiority, source-pass
retention, zero raw collapse, and zero final collapse.

The live fail-closed verifier regenerated both candidate and parent for every
row: 1,400/1,400 candidate exact and 1,400/1,400 parent exact. It recomputed
routes, functional evaluators, source pairing, raw/final collapse, runtime
boundaries, parity, and physical execution from raw evidence.

## Failure localization

The most severe functional strata were abstention 0/100,
domain-independent reasoning 0/100, format control 0/100, coherence 35/100,
and tone control 55/100. Format control contains all 90 route errors and 29 of
53 raw collapses. The remaining raw collapses occur despite correct routing,
so routing alone cannot repair this endpoint. The evidence supports separate
work on prompt/route invariance and autonomous raw termination.

## Immutable evidence

- catalog: `4b0087c9a7fa0e0fd6f607fdbd94fffbdd3cb375f5880a0588c97414583a2ad7`
- source rows: `35a5a6e1035035db1663d793257274e3be2b766aab79a00a0476478d0428b596`
- screen result: `755c285d52bf50b3a6826ef5112cc111346740e04b8f75f74e15fa60d6ef4140`
- screen raw rows: `e509484babfab685c91fee3a6165bfea0b2881213fdb7b3a71fde9266e2a25cf`
- live strict receipt: `9e9070d820ed241d6c6f14590110338c38914a9935840c0817c864e64a513db2`

No prompt, row, evaluator, or negative result was removed. R60 is now a
development surface only and cannot later serve as untouched promotion data.

