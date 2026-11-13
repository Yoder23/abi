# R107: natural-text retention and residual intervention

This is a bounded no-training diagnostic, not an English-core or moonshot
certificate. It does not reopen or rescore R103-R106. The product-quality,
teacher-absence, CPU/GPU, memory and TTFT gates remain unchanged.

## Completed live result

**`PARENT_RESIDUAL_INTERFERENCE_SUPPORTED`**, limited to this registered
natural-text population. All 18 executions completed; each of the nine cells
replayed raw evidence byte-for-byte in a fresh process. No training occurred.

| Checkpoint | Active BPB | Neutral BPB | Active / neutral repeated-4-gram fraction |
| --- | ---: | ---: | ---: |
| Native ancestral reference | 1.2804 | — | 0.571 / — |
| R76 parent | 1.2793 | 1.2486 | 0.197 / 0.596 |
| R100 full-model control | 1.4927 | 1.4421 | 0.539 / 0.421 |
| R99 matched LoRA | 1.3847 | 1.3815 | 0.393 / 0.428 |
| R99 matched adapter distillation | 1.2793 | 1.2486 | 0.197 / 0.596 |

Lower BPB is better predictive loss, not a fluency certificate. The primary
paired passage-level ACTIVE-minus-NEUTRAL effect is **+0.03224 BPB**, 95%
bootstrap CI **[+0.02064, +0.04419]**. The table reports byte-weighted aggregate
BPB; it is intentionally not substituted for the passage-level contrast.
Full-model and distillation intervention contrasts are descriptive; the LoRA
interval crosses zero. No multiple-opportunity winner is declared.

The parent improved likelihood when neutralized but produced more repetitive,
longer continuations. Its active/neutral EOS counts were 13/16 and 1/16; lengths
therefore also affect the repetition statistic. Even the ancestral reference
has high greedy repetition. No neutralized state is qualified or saved as a core.

Crucial scope limitation: coherence route 1 was fixed before inference, and it
was not among R99's acquisition tasks. Adapter distillation's matching retention
there must not be generalized to its trained routes, broad English, or universal
superiority over LoRA. A separate post-hoc tensor/row audit documents this boundary
without changing the primary hypothesis or thresholds.

That tensor audit finds **zero changed tensors selected by coherence** for
adapter distillation and **64/64 identical active raw rows** relative to parent,
excluding only the cell label. Full-model and LoRA have 100 and 24 changed
selected tensors respectively, with zero identical rows. This is consistent
with isolated versus shared-weight updates in these particular historical
controls. It is not a comparison against every LoRA design: route-local LoRA
can also be isolated and remains an important future matched control.

Recorded execution-body totals (initial plus replay): **1,562.91 wall seconds**
and **1,529.56 CPU seconds**. Maximum sampled process RSS was 1,390,628,864 bytes;
maximum allocated CUDA memory was 426,620,928 bytes on the local RTX 3080 Laptop
GPU. These include loading/state-audit work inside each recorded body, exclude
some launcher/preflight overhead, and are not throughput/TTFT benchmarks or
acquisition-efficiency claims. There were 110,592 scored target-token observations
and 15,090 generated tokens across both executions. Each scored token stores one
selected logit and one normalizer, not a full logit vector. No new weights,
training data ingestion, copied substrate, bridge fitting or deployable package
was produced; all inherited model/information costs still count.

## What is being tested

The [registered protocol](../experiments/retention_intervention_r107/PROTOCOL.md)
compares four existing research checkpoints, including matched LoRA and
sequence-distillation controls, with their task residuals active versus zeroed
only in memory. All original checkpoint files remain unchanged. The native
DistilGPT2 ancestor is a reference, not a newly qualified instruction-following
control. Its published weight hash was independently matched before inference.

The fixed corpus is 64 unmodified WikiText validation passages selected without
model scores. Each condition scores the same 6,144 target tokens / 28,862 UTF-8
bytes and greedily continues 16 supplied prefixes. Selection, source licensing,
all token evidence and all continuations are public. The dataset contains facts
and specialist content; likelihood here is not a pure-English or purity metric.
Unseen-to-pretraining status is not established. Greedy repetition is descriptive,
not a human fluency score.

## Evidence and verification

- [Protocol and stopping rules](../experiments/retention_intervention_r107/PROTOCOL.md)
- [Preparation custody](../experiments/retention_intervention_r107/PREPARATION_NOTE.md)
- [Frozen inputs](../results/retention_intervention_r107/freeze.json)
- [Fixed passage population](../results/retention_intervention_r107/population.json)
- [Public source hashes and licenses](../results/retention_intervention_r107/sources/provenance.json)
- [Primary recomputed decision](../results/retention_intervention_r107/decision.json)
- [Complete parameter custody](../results/retention_intervention_r107/complete.json)
- [Descriptive tensor/route comparison](../results/retention_intervention_r107/descriptive_scope.json)
- [Post-run verifier notes](../experiments/retention_intervention_r107/VERIFICATION_NOTE.md)
- [Release/reproduction guide](releases/R107_REVIEW.md)

Recomputation requires all raw rows, both fresh-process executions, receipts,
original checkpoint files, source/tokenizer hashes and the pinned sibling host.
The additive `complete` verifier recomputes active/neutral state hashes and each
residual intervention from the original safetensors. It does not accept saved
scientific pass flags. No neutralized checkpoint is saved or promoted.

Local focused validation: 55 tests, including 26 hostile complete-packet
mutations. These test evidence handling, not English quality or independent
scientific review. All four active/neutral recipient state pairs recompute from
their original 87,710,428-parameter checkpoints. The fixed ancestral reference
retains all 81,912,576 of its parameters; no compact extraction is implied.
The supported repository suite also passes 391 tests plus 16 exact historical
Phase 7 checks, with four skips. The preserved uncached checker and the optimized
completion wrapper reproduce the same custody result; no frozen code was edited.

## Failure ownership and next decision

This measures the ABI research recipient/acquisition lineage, not LayerCake's
separately sealed product. A predictive-loss regression can motivate a future
retention-constrained learning control, but does not prove the cause of failed
grounding/instruction following or authorize another automatic sweep. Better
natural-text likelihood cannot substitute for autonomous quality. Native host
conformance and same-artifact product recertification remain separate gates.

`FULL_MOONSHOT_NOT_PROVEN` remains the controlling verdict. No new capability
package, compression, minimum-information result, speed advantage or superiority
over LoRA/distillation follows from this audit.
