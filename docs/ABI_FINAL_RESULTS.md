# ABI final technical result tables

Every value below is generated or checked from raw locked evidence by
`abi_v2.final_validation`; certificate summaries are not trusted as inputs.
Final internal technical readiness is 18/18. Independent hardware and human
validation remain open.

## Tested environments and capabilities

| Environment | Adapter bytes | Params | Cert examples | Cert UTF-8 bytes | Cert time | English | Python | Chemistry | Civics | Frozen retention | Math equality | Isolation | Teacher absent | Adapter overhead | Package hashes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| LayerCake v25 (CPU cert) | 1,556 | 0 | 128 | 5,953 | 0.040 s | 1,381/1,381 | 100/100 | 100/100 | 100/100 | 1,681/1,681 | Pass | Pass | Yes | 3.96% | Unchanged |
| Qwen2.5-0.5B (CUDA cert) | 1,637 | 0 | 128 | 5,953 | 9.163 s | 1,381/1,381 | 100/100 | 100/100 | 100/100 | 1,681/1,681 | Pass | Pass | Yes | -0.55% | Unchanged |
| Pythia-160M (CUDA cert) | 1,646 | 0 | 128 | 5,953 | 6.185 s | 1,381/1,381 | 100/100 | 100/100 | 100/100 | 1,681/1,681 | Pass | Pass | Yes | 3.12% | Unchanged |

The CPU/GPU labels identify where each certification was measured; they are not
cross-device superiority claims. Qwen/Pythia base hidden states are noncausal to
the answers. See the host-causality audit before interpreting “environment.”

## Limitations and pending external gates

| Gate or limitation | State |
| --- | --- |
| Independent different-hardware reproduction | Pending external operator |
| Blinded human quality | Pending three humans; 0/21,000 complete |
| Stable minimum-information frontier | `PENDING_AFTER_EXTERNAL_VALIDATION` |
| Compatibility beyond the three named environments | Not proven |
| Host-model answer generation | Falsified by neutral-stub causal audit |
| Base-weight tensor transplantation | Not implemented or claimed |
| Global LoRA/distillation/fine-tuning superiority | Not proven or claimed |
| Hidden blind holdout | None; exact-retention suite is public |
| Clean isolated reproduction | Pass; 62 tests, fresh 12-cell CPU matrix |

Raw source: `results/abi_final_validation/headline_recomputation.json`.
