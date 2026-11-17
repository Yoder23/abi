# R109 — bounded resource-budget replication of R108

Drafted 2026-11-14; preregistered 2026-11-17 before any R109 model calls under the
user's explicit authorization to continue with
GPU settings unchanged, a realistic time budget, the same quality gates and
matched controls. This supersedes the prior requirement to resolve the GPU
operating constraint before a retry. It does NOT amend or complete R108.

## Only resource policy and evidence namespace change

Preserve the entire R108 packet, including its 13-row prefix and timeout. Use
`results/trained_grounding_r109/` for this new, write-once attempt. Re-run all
96 initial prompts from the beginning; do not splice old responses into it.
This is replication of an already frozen panel, not a fresh blind holdout:
13 source responses were previously observed. No model, prompts, scorer,
thresholds, routes, training, decoding settings or checkpoints are tuned from
those observations.

Import the unchanged R108 catalog, rendering, evaluator, qualification,
bootstrap and raw-validation functions directly. The population, 164-case
author fixture audit and historical matched-training audit must be byte-identical
to R108. Keep identifiers beginning `r108-` because the input identities are
unchanged. Retain Qwen3-4B-Instruct-2507 at
`cdbee75f17c01a7cc42f958dc650907174af0554` in BF16, the same native chat prompt,
and all four FP32 recipient checkpoints (parent, full-model, R99 LoRA and
adapter distillation). Use the same training-aligned recipient interface,
explicit trained routes, greedy decoding, seed 108001, four CPU threads, TF32
off, context 1024 and maximum 640 new tokens. No repeat penalty or output repair.

LoRA/distillation retain their original matched 192 training examples, 384
updates, sample order and 43,208 loss-token observations. Their trainable
parameter counts differ. The larger new control is a feasibility ceiling, NOT
their original teacher or a matched-size/training baseline. Zero new training,
copied recipient parameters, logits/activation stores or capability artifacts.

## Unchanged quality and completion gates

Each 48-row panel must score >=44/48 and >=5/6 in EVERY one of the eight tasks,
with zero collapse and zero generation-budget stops. At least 44/48 paired
counterfactuals must pass BOTH members, also >=5/6 pairs in every task.
Preserve all R104 conditions and R108's exact-case / negated-pending checks.
These are finite mechanical supplied-note obligations, not general semantics
or human English-fluency validation.

Run control initial and fresh-process replay first. Every scientific raw field
must replay exactly (wall times excluded). Only if the control qualifies may
all four recipients run, each once plus fresh-process replay. A control failure
closes the recipient gate. No best-of selection, partial pass or missing replay.
Use the unchanged 10,000-draw paired-cluster bootstrap and control-relative
noninferiority lower bound >= -0.10. No native-host or product promotion follows.

## Realistic finite resource budget

R108 measured 1,407.4623206000542 generation seconds over 13 completed prompts:
the linear 96-prompt projection is 10,393.56790596963 seconds (2.887 hours).
This is a planning extrapolation, not a measured complete-run latency or a
guarantee; prompt lengths differ. Register BEFORE new inference:

- **4 hours / 14,400 seconds per control process**, including loading and checks;
  about 38.5% above that generation-only projection.
- **1 hour / 3,600 seconds per recipient process**, reflecting their much
  smaller retained 87,710,428-parameter architecture; this is a conservative
  planning cap, not a matched throughput claim.
- **16.5 hours / 59,400 seconds for the controller**, including between-stage
  verification. Maximum model-process allowance is 57,600 seconds over ten
  processes. Maximum 960 new generation calls, inclusive of replay. Failure
  ends the run; no automatic restart or extra budget after observing outcomes.
- **1 GiB new evidence cap**, **10 GiB free-disk floor**. This explicitly changes
  R108's 20 GiB preparation reserve: the 8.06 GB source and all recipients are
  already cached and hash-verified; there are zero downloads, new weights or
  optimizer states. The November 14 preflight had approximately 13.5 GiB free;
  November 17 found zero free bytes. The user authorized retirement of only the
  two older `_public1105` checkouts after verifying retained artifacts and
  preserving all unique evidence. Record actual post-cleanup free space in the
  freeze's preflight observation; do not infer it from logical file sizes. Check storage
  before each stage and prompt; stop on the resource guard, without deleting data.

No GPU power, clock, cooling, driver or thermal setting may be changed. Record
read-only telemetry before/after every stage and preserve any operational
failure. November 14 preflight telemetry differed from R108 (105 W requested
limit, P0, no active throttling); November 17 reports 1,245 MHz at idle. The
agent made no setting change. Bind fresh telemetry before launch. Do not treat a
timing difference between attempts as a controlled hardware/performance result.
Automatic device state changes may occur; document them, do not tune them.

## Custody, reporting and failure ownership

Commit and publish the protocol, then the exact input freeze before inference.
Bind R108's frozen code/data, all unchanged source/recipient payload hashes, the
native host runtime files, R109 code/tests and this protocol. Current host HEAD
is documentation-only newer than R108; all inherited runtime hashes must match.
Record raw prompts/tokens/text, per-task outcomes, traces, CPU/wall times,
sampled RSS, CUDA allocated peak, source-cache reuse and the new-call budget.
Keep R108's acquisition/preparation cost separate from R109's costs.

The verifier recomputes all scores and rejects missing packages, hashes, rows,
replay, stale receipts or budgets. A resource stop remains incomplete; it cannot
be relabeled a quality failure or qualification. Report any agreement with the
old 13-row prefix descriptively, never as a replacement for the new full replay.

Control failure is an experimental prerequisite failure. Recipient low quality
is acquisition/recipient-suitability evidence, not automatically a native
LayerCake bug. Physical-route/execution discrepancies require separate
localization. No broad English, human validation, pure/minimal substrate,
ABI-superiority, CPU speed, host acceptance or full-moonshot claim is granted.
Publish the actual outcome and update the documentation; no automatic follow-on.
