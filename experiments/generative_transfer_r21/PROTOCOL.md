# R21 public label-separated generative transfer bake-off

R21 is a disclosed public prerequisite, not an ABI moonshot certificate. It
tests whether a teacher-labeled, capability-factorized set of genuinely neural
LayerCake packages retains supplied-content behavior better than two matched
sequence-distillation controls.

The immutable source is the pinned Qwen2-7B-Instruct revision used by R19 and
R20. It produces 600 new training responses (100 per registered behavior) and
one registered label for every response. The six-label ontology is disclosed:
prose, summary, email, bullets, clarification, and abstention. This is not
autonomous ontology discovery. R20's 120 public evaluation prompts and source
outputs are reused without modification; no evaluation output is used for
training or model selection.

Three routes receive the same 600 teacher responses. `raw_sequence` is one
LayerCake token-plan student over raw prompts. `labeled_monolith` is one
student over the normalized label-plus-data boundary. `abi_factorized` uses
the same normalized boundary but installs one separately signed package per
label and activates only the selected package. All routes use the same three
seeds and the same total row-exposure budget. Parameter counts and actual
action exposure are reported rather than assumed equal.

A multinomial word model is fit only from teacher-labeled training
instructions and frozen into the ABI runtime manifest. It receives no
evaluation labels. Every package is a signed `lc-direct-neural-decoder/1`
LayerCake `.cake`; the teacher is absent at execution, and removal must stop
the selected package.

Quality is recomputed per row on five separate axes: supplied-content
retention, task adherence, numeric non-hallucination, non-collapse/fluency,
and teacher-output lexical overlap. Headline functional success requires the
first four axes. Comparisons use all 120 paired prompts and deterministic
bootstrap confidence intervals.

Public promotion requires all three ABI seeds to pass package integrity and
CPU/GPU execution, at least 114/120 label decisions, at least 15/20 functional
rows in every task for the median ABI seed, no worse median functional quality
than the teacher or either sequence control, at least 114/120
non-hallucinating and non-collapsed rows, deployed parameters no more than
1.25 times the labeled monolith, zero teacher/source parameters at execution,
and exact remove/reinstall restoration. A pass would authorize a hidden
replication only. It would not prove unrestricted English, global minimality,
arbitrary-domain extraction, or superiority to LoRA/distillation.
