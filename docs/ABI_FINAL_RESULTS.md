# ABI R7 final technical results

Updated: 2026-08-30

R7 passes the frozen, bounded capability-runtime and host-conformance protocol.
It does not yet pass human quality, independent-hardware, teacher-extraction, or
minimum-information gates.

## Release identity

- Tag: `abi-final-validation-v2-repaired-r7-2026-08-30`
- Commit: `3f82a9f4d67dda5c8ea13bd59b2d8f1bbd3dd128`
- Release archive SHA-256:
  `fc50f423986149b5d4670ec9e28698540f64be96034efa26e5704c4469921e88`
- Strict certificate SHA-256:
  `17973df22cb31eddb3b47c6137aa71b68f32cd9d3010cfcddd232b2a1afce488`
- Blind report SHA-256:
  `98b76579ed8486a41e1f5ac00c970738464cdc470dc2ccac18d6c9a89b5b9cea`

## Recomputed results

| Measurement | Result |
| --- | ---: |
| Physical certification inventory | 301,543 rows |
| Reachable regular-file bytes scanned | 11,681,888,205 |
| Capability archives/success IDs found | 0 / 0 |
| Locked host-capability matrix | 5,043 rows |
| Live causality | 3,072 rows, 24 distinct processes |
| Live isolation | 2,100 rows, 0 target successes |
| Transitive source files bound | 733 |
| Pre-public hostile controls | 19/19 rejected |
| Post-public hostile controls | 19/19 rejected |
| Public reconstruction focused tests | 17/17 passed |
| Blind USTAR/V7 prefix controls | 12/12 passed |

The three exact physical inventories contain 100,511 LayerCake, 100,517
Qwen2, and 100,515 Pythia rows. The strict verifier consumes no stored
scientific pass/fail booleans.

## Runtime conformance overhead

Twenty repeated observations per declared environment produced median wrapper
overhead fractions of 0.026818 for LayerCake, -0.007342 for Qwen2, and
0.081364 for Pythia. These are conformance-path overhead measurements, not
end-to-end generation benchmarks and not evidence of ABI inference dominance.

## Open gates

- Human ratings: `0/21,000` complete.
- Independent different-hardware reproduction: pending.
- Registered minimum-information certification: pending.
- Teacher-to-artifact extraction, labeling, and knowledge quality: not proven
  by R7.
- Superiority over LoRA or distillation: not proven.

See [R7_PUBLIC_VALIDATION.md](R7_PUBLIC_VALIDATION.md) for the public and blind
reproduction record.

## Additive R8 result

R8 is a Level 0 negative result and does not replace this R7 release. Public
source state was extracted exactly into the fixed `3 x 8 x 8` canonical
artifact, but the frozen Pythia recipient showed no package-specific gain:
AFTER−BASE was +0.001953 across 1,024 paired raw rows (95% bootstrap CI
[-0.019531, +0.024414]). The held-out commitment was never revealed. The exact
R8 answer is `NO` for the registered v10 system.
