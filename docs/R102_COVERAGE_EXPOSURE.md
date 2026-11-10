# R102 entity coverage and optimization exposure

Status: 2026-11-10. Preregistration/preflight; no R102 quality result yet.
`FULL_MOONSHOT_NOT_PROVEN` remains controlling. Publication awaits the user's
current privacy preference; this document does not imply a new public release.

The initial corpus freeze stopped before any teacher call or optimizer step.
After 317 aliases, variant 1 of `r98-train-summarization-20` exhausted its
4,096-proposal bound. The [preserved rejection trace](../results/coverage_exposure_r102_preflight_v1/candidates.jsonl)
contains no jointly length-matched candidate. A read-only analysis found 46
compatible project-ID components and 69 compatible person-ID components in
that same pool; combining recorded proposals 40 and 144 preserves the exact
119 prompt / 44 target token counts and inverse text identity. This is a sampler
construction issue, not student failure, data insufficiency or host incapacity.
No successful R102 freeze or GPU experiment exists yet.

## Question

[R101](R101_EXECUTION_CONTINUATION.md) found measured execution-path agreement,
exact unassisted replay and a null primary answer-prefix intervention. The
full-model control fitted its training references more closely than development.
R102 tests whether entity coverage or greater fixed optimization exposure helps
under matched workloads. It does not assume the answer or blame LayerCake.

Read the [locked study protocol](../experiments/coverage_exposure_r102/PROTOCOL.md).

| Factor | Levels | Held constant |
| --- | --- | --- |
| Entity coverage | Exact 192 teacher examples; four invertible entity-renamed versions per original | Teacher wording and knowledge, labels, prompt/target token lengths |
| Optimization exposure | 384 and 1,536 updates | Parent, optimizer, learning rate, sample order, batch/context, precision |
| Method controls | Conventional LoRA; deep-adapter sequence distillation | Same data policy, source information and example schedule |

There are four training trajectories and eight predetermined checkpoints, not
an open sweep. The unchanged parent and retained R100 full model are additional
evaluation-only references. There is no new full-model training or generated-plan
candidate. Parameter counts and FLOPs differ between methods and are disclosed.

## Evaluation and claim limits

Ninety-six new development inputs cover the existing eight supplied-content
tasks, with 48 original-shape and 48 shifted-shape identifiers. A fixed source
gate precedes all training. Evaluation answers never enter training. The two
identifier variants of each base case remain together in cluster-bootstrap
comparisons; they are not treated as independent replicates.

The autonomous screen remains >=44/48, >=5/6 per task, zero collapse and
teacher-relative noninferiority, separately in both strata. These mechanical
proxies cannot certify fluency or semantic entailment. One seed and one bounded
task family cannot prove broad English, universal ABI superiority or a minimum
English core. Any positive result remains a conventional acquisition-control
result until it earns the separate product and scientific gates.

The budget is three hours, 96 new teacher evaluation calls, four trajectories
totalling 6,144 optimizer steps and at most 6 GiB new evidence. No threshold,
learning-rate or architecture sweep follows automatically. LayerCake source,
historical evidence and prior checkpoints remain unchanged.
