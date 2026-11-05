# R100 acquisition-capacity control

Status: 2026-11-05. Completed and verified; the capacity control failed both
the useful-quality and training-fit screens. No candidate is promoted.
`FULL_MOONSHOT_NOT_PROVEN` remains the controlling verdict.

## Why this control comes before another ABI mechanism

The [R99 pilot](R99_ENGLISH_PILOT.md) failed with all three learners. Its
additive R99-D1 diagnostic then tested preserved checkpoints without training
or new teacher queries. Twenty-four training examples were selected by a frozen
hash rule, balanced across eight tasks; all 48 R99 development rows were also
measured under teacher forcing.

| System | Autonomous training passes | Correct-prefix response-token accuracy, training | Correct-prefix fact-token accuracy, training | Original development passes |
| --- | ---: | ---: | ---: | ---: |
| ABI generated plan | 7/24 | 76.48% | 97.03% | 10/48 |
| Adapter sequence distillation | 12/24 | 77.79% | 97.03% | 20/48 |
| LoRA | 13/24 | 79.72% | 98.51% | 27/48 |
| Unchanged parent | 0/24 | 47.65% | 67.82% | 0/48 |

The correct-prefix metrics supply the true prior target tokens. They are not
autonomous generation quality. Plan-copy positions are excluded from response
accuracy. High fact-token prediction does not mean the model reliably chooses
to mention the facts when generating freely.

The limited training fit and small response-accuracy train/development gaps
do not justify attributing the problem solely to unfamiliar identifiers. They
also do not isolate a LayerCake runtime defect. Before adding another compiler
representation, R100 tests whether the restricted trainable parameter scope
contributes to the acquisition bottleneck.

Read the [diagnostic protocol](../experiments/grounded_english_r99/FIT_AUDIT.md),
[raw token and generation evidence](../results/grounded_english_r99_fit_audit/),
and [recomputed summary](../results/grounded_english_r99_fit_audit/summary.json).

## The single intervention

The [R100 preregistration](../experiments/acquisition_capacity_r100/PROTOCOL.md)
allows all 87,710,428 parent parameters into the optimizer. The parent,
192 training examples, targets, seed, sample order, 384 updates, FP32 precision
and response-only loss remain identical to R99's distillation control.
The retained LoRA, adapter-distillation and ABI-plan checkpoints are evaluated
alongside it and the unchanged parent on 48 newly frozen prompts.

This is ordinary full-model fine-tuning/sequence distillation. It is explicitly
a diagnostic baseline, not an invention renamed ABI. Equal examples and updates
are not equal compute. The complete inherited model and extra optimization
cost are counted. No LayerCake source or canonical interface changes.

Fresh source qualification must pass before training. New evaluation answers
never enter the optimizer. There is one full-scope run, no best-checkpoint
selection or follow-on sweep. A favorable control result would inform ABI's
acquisition design; it would not certify a production English artifact.

## Frozen identities and execution

- Diagnostic preregistration: `853a7b7`; freeze: `6ccf646`.
- Diagnostic freeze SHA-256:
  `84b29ae86b7df793afb7f894fbd19e3e7348e98b01d671bc54334dfe7c153a2d`.
- R100 preregistration: `69f0922`; freeze: `54b35f6`.
- R100 freeze SHA-256:
  `cbcda0222ac18bd5a2a54486ef69ae5b8b5c4b82d3611ca724260a6a543f0922`.
- Parent and teacher identities are the same as R99; every required source,
  checkpoint and reference row is bound by the 784-input manifest.

Outputs are additive under `results/acquisition_capacity_r100/`. `stop.json`
records a prerequisite/operational stop; `decision.json` is generated only
after the complete comparison. Neither is a whole-moonshot certificate.

## Completed result

The normalized source gate passed 232/240 with zero collapse. Every optimizer
step was finite. The initial loss and example IDs matched the original
response-only distillation run exactly; all 384 sample batches and all 43,208
loss-bearing training tokens were matched by the verifier.

| System | Fresh development passes | Collapse cases |
| --- | ---: | ---: |
| Raw teacher | 44/48 | 0 |
| Normalized teacher | 45/48 | 0 |
| Full-model control | 20/48 | 1 |
| Retained LoRA | 20/48 | 1 |
| Retained adapter distillation | 14/48 | 6 |
| Retained ABI generated plan | 6/48 | 3 |
| Unchanged parent | 0/48 | 2 |

The required student score remained 44/48, at least 5/6 per family and zero
collapse. Full-model training-example generation was 17/24, below the 22/24
fit threshold. Its paired training gain over distillation had 95% interval
[-0.0417, 0.4167]; over LoRA, [0, 0.375]. Neither supports the preregistered
strictly-positive-lower-bound scope-sensitivity claim. Its development score
equaled LoRA, with paired difference interval [-0.1667, 0.1667].

The [decision](../results/acquisition_capacity_r100/decision.json) recomputes
all three negative screens from raw rows. Full scope changed 444 tensors /
86,969,088 parameter elements; optimizer enrollment was 87,710,428. Thus the
intervention really executed, but simply unfreezing more weights did not solve
the bounded acquisition problem. This does not prove lack of model capacity,
data insufficiency, a native host defect, or universal LoRA superiority.

The one training loop took 92.93 seconds, with 4,018,226,176 peak CUDA allocated
bytes and 1,450,946,560 sampled peak process-RSS bytes. Concurrent CPU checks,
unequal trainable scope and unequal FLOPs prevent efficiency claims. The new
48 source requests consumed 8,162 input tokens and 3,791 generated tokens,
with 209.90 seconds of source inference. Across R98/C1/R99/R100, retain all
367 source requests and 29,947 generated tokens, including repeated prefixes.
No teacher weights, logits or activations were newly imported; the complete
inherited parent still counts.

Final control checkpoint SHA-256:
`c6e32c20bdda3b0f350620254670a79f03400f0c44f6d1ee4ef02df1082c9c8b`.
This is an unsigned failed research control, not an installable English cake.

## Consequence for the next experiment

Do not repeat the generated-plan branch or run another nearby full-unfreezing
sweep. The immediate problem is autonomous acquisition/fit under this small
supplied-content corpus, not a demonstrated install/transfer/runtime failure.
Source coverage/diversity, optimization adequacy, parent suitability and
free-generation drift remain unresolved alternatives. A next study should
separate those causes with a prospectively fixed data-sufficiency/realization
control and matched information budgets before claiming a new ABI mechanism.
No such successor is implemented or certified by this result.

Read-only checks after completion:

```bash
python -m experiments.grounded_english_r99.fit_audit verify
python -m experiments.acquisition_capacity_r100.run verify
python scripts/run_review_tests.py
```

Do not rerun `freeze`, `pilot`, `source`, `train` or `evaluate` into an existing
evidence path. Required checkpoints are real Git LFS payloads, not pointer text.
The three intermediate full-scope training snapshots remain local; the final
checkpoint and raw logs are the required review artifacts.

## Remaining product gates

Broad English and human quality, isolation from specialist knowledge, compactness,
three paired seeds, immutable signed host acceptance, and the same final
artifact's CPU speed, TTFT and memory advantages remain open. Neither a passing
software suite nor this full-model control can fill those gates.
