# R98-C1: complete truncated acquisition, then test the matched learners

Registered 2026-11-05 before any completion or student run. This is one bounded
operational follow-up, not a new independent confirmation or a repair of R98's
historical verdict. R98 remains `ABI_SOURCE_PREREQUISITE_FAILED` (206/240).

## Measured reason

All 30 email outputs and one tone-control output reached the fixed 144-token
limit. The source gate forbids any truncation, including the two email prefixes
that already passed the lexical screen. No student was trained. This isolates
an acquisition-budget problem; it does not demonstrate a LayerCake defect.

## One permitted intervention

Re-execute exactly those 31 truncated requests, in their original order, with
the same frozen source, prompt, system message, FP16 precision and greedy
decoding, but a maximum of 384 source tokens. Require exact equality of every
original generated token prefix. Preserve all original rows and all completion
attempts. Reuse the other 209 complete responses unchanged. Never filter rows,
change prompts, alter the evaluator, relax thresholds, or retry a failed request.
The merged 240-row corpus must pass the original source gate: at least 216
mechanical passes, at least 75% per task, no collapse, and no truncated rows.
Otherwise stop this follow-up before any student training. No further cap or
prompt sweep is authorized by this protocol.

The lexical screen has known limitations: it can flag conditional confirmation
language as contradictory and misses synonyms such as "awaits". Those observed
limitations are disclosed, not corrected after seeing this corpus. Neither a
pass nor a failure is a semantic or human-fluency judgment.

## Conditional learner comparison

Only if the source prerequisite passes, execute the R98 ABI-plan, direct
sequence-distillation and rank-32 LoRA arms, plus the unchanged parent, using
the immutable R98 training/evaluation implementation. The new runner explicitly
supplies a separate configuration/output namespace to that engine; it does not
edit historical code or evidence. Some engine-emitted format names retain R98;
their C1 lineage is the enclosing freeze manifest and output path.

Keep the same 192 training / 48 development split, seed 98001, sample order,
384 updates, batch 2, optimizer, trainable scopes and continuation thresholds.
Use the parent's full 1024-token context (all three arms equally padded) and
a 640-token generation cap (including the candidate's generated plan). These
changes accommodate completed targets without dropping or truncating training
examples. An example that still overflows stops training. The earlier run did
not train any arm, so there is no competing R98 learner result to combine.

The development set has already been inspected during source diagnosis. It is
not a hidden test. A successful pilot only supports a subsequent newly frozen
broader evaluation with three paired seeds. No generalization certification,
teacher superiority, or unrestricted English claim is allowed here.

All arms share source information, examples, optimizer steps and padded token
positions; they do not have equal trainable parameters, FLOPs or loss-bearing
tokens. Report those differences. The ABI intervention uses a supervised
distillation loss; do not describe it as an alternative to distillation already
proven superior. No inference teacher is loaded in student processes.

## Costs, scope and attribution

Record the original acquisition costs plus all completion costs, including
repeated prefix tokens. Report unique imported output tokens separately from
total teacher-generated tokens. Count the entire 87,710,428-parameter parent,
not just the trainable bridge. No fact-free core, minimum footprint or source
weight extraction is established by this experiment.

The bound remains six hours total, one hour per training arm, 20 GiB additional
disk, and at least 30 GiB initially free. GPU acquisition/training; CPU product
certification only after real quality. Preserve partial results and stop on
non-finite optimization, missing identities, prefix disagreement, source failure,
or resource-budget exhaustion. A budget stop is not proof of a hard limit.

Use unchanged LayerCake Python sources at the R98-pinned host identity.
Documentation-only successor commits are allowed only when every frozen host
source hash still matches. No LayerCake source edits or host certification occur.
Source failure is ABI acquisition failure; learner failure is acquisition/fit/
generalization evidence. Neither alone establishes a native host regression.

Broad English, independent human quality, source/domain isolation, signed
production acceptance, three paired seeds, independent hardware, same-artifact
CPU >=2x, >=95% throughput retention, TTFT and memory remain open gates.

## Commands and immutability

Run `python -m experiments.plan_english_r98_completion.run audit-r98` to
recompute the historical source failure. `freeze` binds old and new code,
catalogs, parent, source evidence and this protocol. `pilot` performs the one
completion pass and conditionally runs the learners. `verify-source` rechecks
the completed raw corpus and diagnoses a negative result without certifying a
student. `verify` is valid only after all learner arms finish.

The whole-moonshot verdict remains `FULL_MOONSHOT_NOT_PROVEN`.
