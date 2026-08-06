# ABI capability-compiler Phase 3 direct-core V23 report

Status date: 2026-08-06

V23 is complete and failed its preregistered initial-seed absolute development
screen. The exact V23 architecture is closed. Phase 3 remains uncertified and
Phases 4 through 8 remain locked.

## Result

The 4,011,040-parameter self-causal candidate trained on the certified 7,000-
record English IR in 143.95 seconds on the declared RTX 3080 Laptop GPU. It
stored no source logits or activations, copied no source parameters, retained
no source blocks, and used no teacher at inference.

Autonomous development evaluation produced 504/1,400 functional passes
(36.00%; Wilson 95% CI 33.53%-38.55%), 77 repetition collapses, and zero
generation errors. Thirteen of fourteen capabilities missed the ordinary
point-and-Wilson gate. Prompt grounding scored 50/100, instruction following
22/100, and abstention 34/100, so all three critical capabilities missed their
stricter gates. Coherence and fact-free reasoning each scored 0/100.

The protocol required early closure after any absolute-screen miss. Therefore
matched causal controls, teacher-relative noninferiority, and the other paired
seeds were not run and are not inferred.

## Failure ownership

This is an ABI acquisition/representation failure, not a LayerCake regression.
The separate LayerCake repository had already passed a construct-only test of
the signed `lc-direct-neural-core/1` external-core interface, including exact
install/remove identity and CPU/CUDA execution. V23 autonomously generated
through the LayerCake-native plan but failed the ABI quality gates. It inherits
no LayerCake quality, speed, memory, or TTFT claim.

The measured bottleneck is narrower than “LayerCake cannot host the result.”
V23 trained only fixed-vocabulary output actions even though the host plan also
supports source-pointer actions. Outputs were often fluent templates, but they
substituted, duplicated, or lost prompt names, times, and places. One separately
preregistered pointer-supervised target representation is scientifically
supported. A V23 data, step, seed, rank, or nearby hyperparameter sweep is not.

## Verification and preserved correction

The verifier recomputes functional scoring and repetition collapse from all
1,400 raw outputs, reconstructs every aggregate and Wilson interval, binds the
protocol, checkpoint, tokenizer, model configuration, receipt, and output
hashes, and rejects receipt, raw-output, candidate-governance, and decision
mutations.

The first generated analysis draft is preserved but superseded. Its numbers
and failure verdict were correct, but its narrative incorrectly said that no
capability met the ordinary gate; clarification did. The corrected immutable
decision states that 13 of 14 capabilities failed and is the only controlling
V23 decision.

## Claim boundary

V23 does not establish teacher-relative quality, causal superiority, Phase 3,
or ABI superiority over LoRA or distillation. Phase 2 human ratings remain
deferred rather than passed, final material remains unopened, and Phases 4
through 8 remain locked.
