# R12-A conventional-teacher extraction protocol

R12-A tests only the missing frontend:

`ordinary open-weight teacher -> frozen R11 neural ABI package`

Every R11 execution-side artifact remains immutable. R12-A may not modify the
R11 package schema, recurrent executor, host codecs, host scales, model
revisions, interventions, or evaluator semantics.

## Public feasibility prerequisite

Before any R12 held-out secret exists, an ordinary frozen Qwen2.5-0.5B base
receives a standard source-local LoRA fine-tuning on one public synthetic
capability. The capability is not represented as an R11 transition during
teacher training. The teacher sees only prompt/next-token examples.

The public gate requires:

- base-teacher accuracy at most 0.25;
- fine-tuned-teacher accuracy exactly 1.0 on 512 unseen depth-6/7 programs;
- exactly 1.0 on all 24 atomic probes;
- gain at least 0.70;
- unchanged Qwen base-weight hash; and
- an extractor using no training rows, answers, rule seed, held-out rows, or
  learned parameters.

Failure closes that teacher-training configuration before held-out use.

## Foreign extractor

The bounded R12-A extractor is a fixed compiler for the registered opaque
micro-language ABI. It sends exactly 24 canonical atomic prompts to the frozen
teacher, reads the teacher's native full-vocabulary logits, selects the native
canonical output for each state/operator pair, and emits a one-hot normalized
`3 x 8 x 8` R11 transition.

This is behavioral extraction from a frozen conventional model, not analytic
tensor-delta inversion. It is allowed only as the R12-A construction gate and
must be described that way. It may not inspect training/evaluation answers or
the hidden capability rule.

## Held-out promotion gates

After the public prerequisite passes, code, source method, extractor, R11
bindings, data sizes, gates, and a fresh secret commitment must be committed
before reveal. A held-out pass requires:

- conventional Qwen BEFORE at most 0.25 and AFTER exactly 1.0 on 512 new
  depth-6/7 prompts;
- the fixed extractor emits the unchanged R11 package format;
- the package alone reproduces Qwen AFTER canonical UTF-8 exactly through all
  frozen R11 hosts on the held-out matrix;
- zero recipient optimization and no teacher during recipient execution;
- exact removal/restoration and all R11 negative controls;
- exact fresh live replay of the frozen package bytes; and
- fail-closed hostile verification.

A pass remains bounded to this synthetic family and behavioral extractor. It
does not establish pretrained English/domain extraction, universal weight
compilation, or superiority to LoRA/distillation.
