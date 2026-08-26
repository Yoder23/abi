# ABI repaired technical result tables

These values are recomputed from the repaired raw evidence by
`abi_v2.strict_validation`. The prior 18/18 certificate is historical and
superseded.

## Physical host certification

| Environment | Capsule files | Roundtrips | Native forwards | Adapter SHA-256 | Median overhead |
| --- | ---: | ---: | ---: | --- | ---: |
| LayerCake v25 | 9 | 128 | 0 | `d1f3a9d6…317f04` | 4.57% |
| Qwen2.5-0.5B | 15 | 128 | 16 | `b13a75b8…0291f` | -2.45% |
| Pythia-160M | 13 | 128 | 16 | `df3598b6…ceafa` | 8.47% |

Every capsule has zero capability archives and zero source-success ledgers.
Raw mount evidence shows the development drive replaced by private `tmpfs`
during worker execution.

## Live and locked evidence

| Evidence | Rows | Recomputed result |
| --- | ---: | --- |
| Locked three-host matrix | 5,043 | 5,043 functional source-byte matches |
| Cross-host matrix identity | 1,681 tasks | all output bytes identical |
| Specialist action identity | 300 tasks | all action sequences identical |
| Live causal interventions | 3,072 | all six positive states identical; removals fail live |
| Live isolation | 2,100 | 0 target successes; 700 cross-host identities |
| Supported tests | 70 | all pass |
| Disposable archive hostile audit | 15 mutations | all rejected; exact restore |

The live causal run uses eight fresh processes per host. Qwen/Pythia state
conditions mutate a native parameter, run a new forward, and pass the resulting
state to the conformance adapter. Canonical output remains invariant. This is
evidence for a standalone capability-runtime boundary, not host-model answer
generation.

## Integrity

The strict certificate binds 95 required files with per-file hashes and
aggregate SHA-256 `4387142fa72a266ec7f8624c161d803f474d911271c3501945b85aa86c599416`.
Its current evidence SHA-256 is
`75c6611d98bc1cc4988651659db1088529f5c30324a18f1b88cb997f65e314e0`.
No experiment status/gate/frozen-policy boolean is accepted as scientific
evidence.

## Pending gates

| Gate or limitation | State |
| --- | --- |
| Immutable public assets and definitive archive | Pending publication |
| Fresh public-manifest reconstruction | Pending |
| Fresh blind Codex red-team | Pending |
| Independent different-hardware reproduction | Closed until the above pass |
| Blinded human quality | Closed until the above pass; 0/21,000 |
| Stable minimum-information frontier | `PENDING_AFTER_EXTERNAL_VALIDATION` |
| Compatibility beyond the three named environments | Not proven |
| Base-weight tensor transplantation | Not implemented or claimed |
| Universal LoRA/distillation/fine-tuning superiority | Not proven |
| Hidden blind holdout | None; exact-retention suite is public |

Raw source: `results/abi_final_validation_v2/strict_validation.json`.
