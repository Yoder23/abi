# ABI repaired technical result tables

These values are recomputed from the repaired raw evidence by
`abi_v2.strict_validation`. The prior 18/18 certificate is historical and
superseded.

## Physical host certification

| Environment | Capsule files | Roundtrips | Native forwards | Adapter SHA-256 | Median overhead |
| --- | ---: | ---: | ---: | --- | ---: |
| LayerCake v25 | 8 | 128 | 0 | `d1f3a9d6…317f04` | 0.89% |
| Qwen2.5-0.5B | 14 | 128 | 16 | `b13a75b8…0291f` | 0.05% |
| Pythia-160M | 12 | 128 | 16 | `df3598b6…ceafa` | 7.45% |

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
| Supported tests | 66 | all pass |
| Disposable archive hostile audit | 9 mutations | all rejected; exact restore |

The live causal run loads the real host and obtains actual host state for each
task, but the frozen adapter exposes no host-state semantic channel. This is
evidence for a standalone capability-runtime boundary, not host-model answer
generation.

## Integrity

The strict certificate binds 59 required files with per-file hashes and
aggregate SHA-256 `3a56667dd2dffdf8c48b5de967933b968889ccc3cea6af762a567f9850414b46`.
Its current evidence SHA-256 is
`7f11a154ad5475ff71a99074ecee7cad35cd27bfa966d3c99cdd1f786e1a12d4`.
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
