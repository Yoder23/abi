# ABI R7 final technical results

Updated: 2026-09-02

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
The R8 public verifier hostile audit produced 8/8 expected outcomes: seven
malformed, missing, stale, or contaminated cases were rejected, and a
hash-consistent forged scientific boolean was ignored with the `NO` verdict
unchanged.

## Additive R9 result

R9's capability-specific Gate A is negative and does not replace R7. The v2
backend contained 1,039,880 parameters and trained for 10,000 steps without
changing Pythia. Strict recomputation produced:

| Measurement | R9 v2 result |
| --- | ---: |
| Training fit | 0.128472 |
| Unseen-depth BASE | 0.073242 |
| Unseen-depth AFTER | 0.125000 |
| AFTER - BASE | +0.051758 |
| Paired bootstrap 95% CI | [0.036133, 0.068359] |
| ZERO | 0.125000 |
| Mean AFTER teacher-recipient TV | 0.873909 |
| Live-replayed evaluation rows | 8,192 |
| Live-replayed training rows | 288 |
| Hostile controls | 5/5 rejected |

The registered 0.98 training-fit, 0.95 unseen-depth accuracy, +0.70 gain, and
negative-control gates failed. The universal capability-blind backend was not
run. See `results/neural_isa_r9/revision_002/capability_specific_pythia/`.

## Additive R11 result

R11 is a bounded synthetic construction pass and does not replace the R7
release. Four held-out capabilities learned in an ABI-native teacher substrate
were packaged as four 2,053-byte neural states. Across 90,112 source and
recipient rows, teacher AFTER and every Pythia/Qwen2/T5 AFTER and RESTORED score
were 1.0; all canonical UTF-8 teacher/recipient comparisons were exact;
negative controls were at most 0.19140625; and package, backend, and codec
removal returned exactly to BASE. Recipient optimization was zero and frozen
model/codec hashes did not change.

Seven hostile controls failed closed. A fresh live replay copied the exact
revision-001 packages by SHA-256, did not retrain them, and reproduced both raw
JSONL files byte-for-byte. Independent strict verification passed the replay.
See [R11_NATIVE_NEURAL_CONSTRUCTION_RESULT.md](R11_NATIVE_NEURAL_CONSTRUCTION_RESULT.md).

This proves only the ABI-native neural copy/paste construction. Extraction of
English or domain knowledge already encoded in a conventional open-weight LLM
remains open, as do LayerCake product acceptance and LoRA/distillation
comparison.

## Additive R12-A public result

R12-A does not replace R7 and is not held-out certification. It trained
Qwen2.5-0.5B conventionally on a fresh synthetic capability, while keeping the
R11 package schema, interpreter, host codecs, and execution boundary frozen.
On 512 unseen prompt-disjoint public compositions, native Qwen peaked at
510/512 and therefore failed the registered exact source prerequisite.

The source's 24 atomic probes were exact. A fixed zero-parameter extractor
compiled them into a 2,053-byte R11 package, and that package executed all
512/512 public compositions exactly without the source teacher. This is a
bounded public frontend construction result plus an explicit strict
teacher-gate failure. No held-out secret was created or revealed. See
[R12_FOREIGN_FRONTEND_PUBLIC_RESULT.md](R12_FOREIGN_FRONTEND_PUBLIC_RESULT.md).

## Additive R13-B held-out result

R13-B passes its preregistered local claim of bounded enumerable capability
extraction. Four hidden finite capabilities produced 96/96 source atomic
answers, four exact 2,053-byte packages, 2,048/2,048 package/oracle rows, and
6,144/6,144 AFTER plus 6,144/6,144 RESTORED recipient rows across Pythia,
Qwen2, and T5. Seven fresh-process replay files were byte-exact. The repaired
hostile verifier passed 9/9 controls after two failed audit revisions were
preserved and repaired.

Native Qwen matched only 372/2,048 long-composition oracle answers. This is
therefore capability canonicalization from an exhaustively enumerable atomic
interface, not exact teacher behavior copy. Public asset publication,
pre-existing knowledge, English/domain extraction, and all broader claims
remain open. See
[R13_BOUNDED_CAPABILITY_EXTRACTION_RESULT.md](R13_BOUNDED_CAPABILITY_EXTRACTION_RESULT.md).

## Additive R14 held-out result

R14 is a preregistered negative result and does not replace R7. Its frontend
received 256 mixed, non-atomic Qwen observations per capability rather than the
complete R13 transition table. The evaluation space contained 4,642,668,576
possible cases; 10,000 unseen-depth and 1,000 order-counterfactual rows were
scored per capability.

The frontend recovered one of three fresh capabilities exactly. The other two
latent selections were wrong, yielding package unseen/order-counterfactual
accuracies of 0.2449/0.2520 and 0.2657/0.2340. The all-capability gate failed,
so recipient execution correctly did not run. All 36,840 source observations
replayed byte-for-byte across three fresh processes, and 7/7 hostile controls
passed. See
[R14_NON_EXHAUSTIVE_CAPABILITY_RESULT.md](R14_NON_EXHAUSTIVE_CAPABILITY_RESULT.md).

## Additive R15A held-out result

R15A passes its preregistered bounded local claim. A generic frontend frozen
before reveal recovered 8/8 fresh synthetic capabilities from anonymous Qwen
before/after effective output-weight deltas without behavioral queries or
answers. Eight 2,053-byte packages were exact on 80,000 unseen and 8,000
counterfactual rows and produced 1.0 AFTER/RESTORED behavior across 135,168
Pythia, Qwen2, and T5 recipient rows. Full-state delta controls stayed at or
below 0.265.

Strict recomputation passed, a fresh live run regenerated the source and eight
physical extraction capsules and reproduced every recipient row byte-for-byte,
and 9/9 hostile mutations failed closed. The result is limited to a capability
deliberately taught through complete atomic supervision and extracted from a
before/after delta. Pre-existing English/domain extraction, labeling,
teacher-quality generation, after-only extraction, minimality, LayerCake
ingestion, and LoRA/distillation superiority remain open. See
[R15_FOREIGN_NEURAL_STATE_RESULT.md](R15_FOREIGN_NEURAL_STATE_RESULT.md).
