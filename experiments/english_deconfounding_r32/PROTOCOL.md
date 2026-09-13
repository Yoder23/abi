# R32 counterbalanced normalization repair

R31 v4 failed its preregistered frozen-package replication at 130/144.  The
failure was concentrated in planning (8/12), reasoning (8/12), and abstention
(8/12).  Raw rows show a deterministic acquisition confound: three instruction
wordings cycled at the same period as generated detail categories.  In the
abstention source, for example, each instruction wording was paired with only
one of the three requested-fact categories.  The v4 shuffle broke that pairing.

This is one bounded data-normalization repair, not an architecture or
hyperparameter sweep.  The frozen R31 failure result has SHA-256
`5e447168c885649f710480d510e7b99785ce1c802386572dd53d0dcc12cc180f`.

## Acquisition

Use the unchanged Qwen2-7B-Instruct revision
`f2826a00ceef68f0f2b946d945ecc0477ce4450c` and the unchanged three generic
R31 system prompts.  Acquire exactly two accepted examples in each cell of a
3 instruction-wording by 3 detail-cycle matrix for every one of the 12
registered supplied-content contracts: 18 accepted rows per contract and 216
total.  No task name, label, oracle answer, or capability identifier is shown
to the teacher.  Each candidate may use the fixed three generic attempts;
each cell fails closed after six rejected candidates.  Preserve all attempts
and quarantines and account all teacher tokens, rendered tokens, bytes, time,
GPU memory, and CPU RSS.

## Single rescreen

Combine the 18 counterbalanced rows with the existing 48 R31 training rows per
contract.  Train exactly one package per contract with the unchanged R31
grounded-pointer architecture, optimizer, 533 exposures per row, and 384-action
ceiling.  Do not train a cascade or another width/depth/step variant.

Evaluate the resulting 12 packages on the exact R31 v4 fixture that exposed
the confound.  Runtime receives no teacher or reference outputs.  Pass requires
132/144 functional, at least 11/12 per contract, 12/12 abstention, 144/144
registered routes and contract audits, exact package hashes/installs, and
12/12 removal rejection.

A development pass only authorizes a new separately committed unseen-prompt
replication.  It cannot itself certify general English, autonomous labeling,
minimality, arbitrary domains, or superiority to LoRA/distillation.
