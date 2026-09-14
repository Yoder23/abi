# ABI R7 final technical results

Updated: 2026-09-14

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

## Later V1089 local compiler campaign

The post-R7 compiler campaign conditionally completes its registered machine
gates through Phase 7. Phase 4 establishes B40 as the smallest stable passing
tested budget among B20/B40 for the exact five-route architecture: all three
1,400-observation screens pass the machine-quality contract (4,200 observations
total), using 4,112 record memberships, 4,005 unique source attempts, and
123,167 authoritative teacher-output tokens. At the same exact B40 information
budget, the registered L0, L1, and D0 controls do not pass the all-seed quality
contract.

Phase 5 certifies 300/300 selected-domain results and 300/300 missing-domain
abstentions per seed for chemistry, civics, and Python, with unchanged English
outputs. Phase 6 certifies all three packages simultaneously across three fresh
host initializations: 900/900 selected specialist rows, 900/900 structured
components, 300/300 quarantined conflicts, zero collapse, and zero receiver
training. Phase 7 binds that exact quality/composition lineage to one product:
13.24x optimized CPU-transformer throughput and 13.51x registered GPU-comparator
throughput, 244 cross-device identities, lower active memory, and no teacher at
inference.

Phase 8 has only a local clean-export certificate. The exact 281,108,851-byte
handoff was reconstructed and replayed from detached source trees on the
development machine, but an independent operator and different CPU/CUDA
hardware have not run it. Phase 2 also remains at 0/21,000 human preferences.
Accordingly, V1089 is strong bounded machine evidence, not an unconditional
release, human-quality result, global minimum, or universal superiority claim.

## Additive R97 result

R97 is a bounded prospective local pass and does not replace R7. The frozen
R96 neural bridge scored 1,399/1,400 on a catalog created only after its
checkpoint was committed, versus 1,088 for the live pinned teacher, 60 for the
unchanged LayerCake parent, and 17 for a seeded random bridge. Its four family
scores were 350, 350, 349, and 350; teacher-correct retention was 1,087/1,088;
the paired teacher-advantage interval was [0.20071, 0.24429]; collapse was zero;
and physical sparse execution was 1,400/1,400.

The strict verifier recomputed raw spans, tokenization, evaluator outcomes,
collapses, invocation counts, aggregates, confidence intervals, and gates. A
12-case hostile shadow-tree audit rejected all mutations, and the second live
execution reproduced the raw JSONL byte-for-byte. The result proves bounded
teacher-derived two-hop reasoning transfer into LayerCake only. The full ABI
moonshot and external Phase 2/8 gates remain open.

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

## Additive R15B held-out result

R15B passes its preregistered bounded local claim for pre-existing
representation recovery. Qwen2-7B-Instruct remained unchanged and supplied 24
correct source-generated atomic answers across four secret slot mappings. A
generic zero-parameter decoder recovered all four transitions from anonymous
pre-answer residuals plus the corresponding frozen digit output-head rows in
physical Linux pivot-root/no-network capsules.

Four 2,053-byte packages scored 40,000/40,000 on registered depth-13 through
depth-18 oracle cases and produced byte-exact live replay across 33,792 Pythia,
Qwen2, and T5 recipient rows. Strict verification passed, fresh live execution
repeated the source generations and four physical extractions, and 10/10
hostile mutations failed closed. The source and all recipients received zero
training; the final packages contain no frozen source parameters.

A fresh blind review of evidence commit `c173e14` passed with zero
critical/high findings. Its direct provenance replay regenerated all 24 source
rows and all four tensor bundles bit-exactly. The subsequent additive verifier
repair enforces those comparisons and binds all 10 files/15.24 GB of the pinned
source snapshot by SHA-256.

The result is limited to a registered arithmetic representation. Prompts state
the operation, source reasoning can expose the answer before the captured
position, semantic labels are externally registered, and the package's deep
oracle behavior comes from the registered affine transition interpreter rather
than exact deep source behavior. R15B therefore does not prove English or
open-world domain extraction, autonomous labeling, minimality, teacher-quality
generation, production LayerCake ingestion, or superiority over LoRA or
distillation. See
[R15B_PREEXISTING_REPRESENTATION_RESULT.md](R15B_PREEXISTING_REPRESENTATION_RESULT.md).

## Additive R16 held-out result

R16 passes a preregistered bounded local claim for factual extraction and
semantic segregation. An unchanged Qwen2-7B-Instruct source supplied 16 hidden
facts across registered chemistry and geography namespaces. Two physically
isolated compilers emitted immutable packages totaling 1,222 bytes. The
packages matched the source on 48/48 disjoint evaluation questions, while
other-domain and removed-package conditions abstained 48/48 and a rotated-
score control scored 0/48.

The complete run and fresh live rerun agree byte-for-byte on all 96 source
rows, 48 evaluation rows, source bundles, and package bytes. Expanded strict
verification binds all 13 declared artifacts and 48 residual rows to the
complete pinned source snapshot. The expanded verifier rejected all 21 hostile
mutations. Source training was zero, the generic package executor required no
training, no source parameters are in the final packages, and the teacher was
absent during package execution. Blind review passed with no Critical or High
finding.

The result remains a structured closed-candidate memory test. Candidate
vocabularies and the two-domain ontology were registered, and the package is
not a fluent neural English substrate. Autonomous discovery, arbitrary-domain
coverage, free-form teacher-quality generation, production LayerCake
ingestion, native neural transplantation, minimality, and superiority to LoRA
or distillation remain open. See
[R16_FACTUAL_SEMANTIC_RESULT.md](R16_FACTUAL_SEMANTIC_RESULT.md).

The blind review found no Critical or High blocker but identified an
unpreserved public-protocol digest. The original result and all findings remain
historical. A new secret selection was then preregistered against the corrected
public receipt before reveal. That clean replication again passed 16/16 facts,
all 48-row source/package/segregation gates, exact live replay, direct source
rehashing, and 21/21 hostile controls. The two selections cover 27 distinct
facts. Blind review of exact replication commit `26fb029` passed with zero
findings. Durable publication and public reconstruction remain open.

## Additive R17-R19 linguistic-realization result

R17's exact-template source prerequisite failed twice and was closed. R18's
factorized successor passed its disclosed prerequisite at 48/48 but failed its
hidden replication at 46/48 because cross-number pooling produced two plural-
question regressions. Both failures remain authoritative.

R19 materially replaces that pooling rule with same-number polarity contrasts.
It passed 96/96 disclosed development rows and then passed a separately
committed fresh lexical replication at 48/48, versus 43/48 for the unchanged
teacher and 48/48 for independent modal consensus. It had zero teacher-correct
regressions, a 0/48 rejected control, and 48/48 removal abstention. The 4,263-
byte package contains no source, host, or bridge parameters. Strict
recomputation, 22/22 hostile mutations, and a fresh byte-exact physical replay
passed.

This is a bounded supplied-slot, 24-signature English surface-realization
result. It is not unrestricted English, autonomous prompt understanding,
conversation, summarization, arbitrary-domain extraction, LayerCake product
ingestion, global minimality, or superiority to LoRA/distillation. See
[R19_CONTRASTIVE_REALIZATION_RESULT.md](R19_CONTRASTIVE_REALIZATION_RESULT.md).

R20 tested the structural representation on six broader raw-instruction tasks.
The source produced all 192 registered rows but only 10/72 extraction and
19/120 evaluation strings exactly, and six sufficiently supported programs
could not be recovered. The prerequisite failed before compiler, package, or
LayerCake execution. This closes the tested exact delexicalized-program route;
it does not alter R19's narrower pass. See
[R20_INSTRUCTIONAL_REALIZATION_RESULT.md](R20_INSTRUCTIONAL_REALIZATION_RESULT.md).
