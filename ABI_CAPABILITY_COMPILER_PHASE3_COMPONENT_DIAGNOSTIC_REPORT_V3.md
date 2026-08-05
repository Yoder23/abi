# ABI Phase 3 component diagnostic report

Status date: 2026-08-05

## Decision

The V9 read-only component diagnostic is **COMPLETE**. It performed no training,
persisted no altered checkpoint, accessed no final material, and cannot promote
a model. Its purpose was to localize V6 B1's near-teacher development behavior
before another architecture is designed.

The controlling decision is
`results/abi_capability_compiler_phase3_component_diagnostic/decision_v1.json`
(file SHA-256
`4ceb59e72e991595bbc39ce0a83fd09112234ab2185be9b59b96d623d11ab50a`,
internal evidence SHA-256
`9e50e69ad44745e28bbc64343153bbf6f67bf3ab945dd5debc1eb51579d5f560`).
An independent recomputation is byte-identical.

## Preserved preflight failure

V8 attempt 1 failed before evaluating a prompt or creating an output because
the diagnostic addressed the host's `ModuleList` using a string index. The
failure is preserved in
`ABI_CAPABILITY_COMPILER_PHASE3_COMPONENT_DIAGNOSTIC_ATTEMPT1_FAILURE.json`.
V9 changed only the index type, added a regression test, and was committed and
tagged before retry. No scientific field changed.

## Results

R0 is the sealed, unmodified B1 evaluation. Each ablation used the same B1
checkpoint and 1,400 paired development prompts.

| Variant | In-memory intervention | Passes | Rate | Collapses | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| R0 | none | 1,224/1,400 | 87.43% | 60 | 36,543 |
| R1 | bypass all six output cakes | 1,148/1,400 | 82.00% | 49 | 36,810 |
| R2 | bypass route embedding | 1,225/1,400 | 87.50% | 62 | 36,653 |
| R3 | bypass output cakes and route embedding | 1,148/1,400 | 82.00% | 50 | 36,765 |

| Paired comparison | Difference | 95% stratified-bootstrap CI |
| --- | ---: | ---: |
| R1 - R0 | -5.43 points | -7.07 to -3.79 |
| R2 - R0 | +0.07 points | -0.14 to +0.29 |
| R3 - R0 | -5.43 points | -7.07 to -3.79 |

All interventions leave the on-disk checkpoint SHA-256 exactly
`e25905096ecf047c8a4f98021288e68d9b5142d2150c4ce7bada576616ba11ef`.
No ablated checkpoint exists.

## Interpretation

The learned route embedding is not measurably responsible for B1's aggregate
quality. The shared prompt projection and three pre-block sequence adapters
retain 82.00% without either route-specific component. The output cakes add a
statistically supported 5.43 points but also coincide with 11 additional
collapse events relative to R1. Because this is a post-training ablation,
coadaptation prevents inferring the exact outcome of retraining without a
component.

Combined with the V6 causal matrix, the evidence rejects another six-way
English output-cake sweep. The next trainable hypothesis should:

1. keep the frozen host and shared prompt-conditioned sequence transforms;
2. replace six route-specific English output cakes with one shared generic
   output residual;
3. use capability labels only to condition the shared sequence transform, so
   labeled, label-free, and monolithic controls remain matched;
4. directly target the measured repetition failure under one preregistered
   loss, not a decoding-only quality transplant; and
5. retain deranged-target, no-response, parent, teacher, identical-exposure,
   three-seed, and final-data-firewall controls.

No such training is authorized by this report. It requires a new protocol,
implementation identity, tests, parameter accounting, and initial-seed stop
rule sealed before execution.

Verification reports 494 in-scope ABI tests passed with the known
separate-repository identity check explicitly deselected rather than weakened.

## Claim boundary

This diagnostic establishes post-training component dependence only. It does
not make B1 an ABI candidate, prove semantic-label value, certify Phase 3,
predict a retrained architecture, waive Phase 2 human ratings, or open Phase 4.
