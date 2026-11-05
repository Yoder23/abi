# R100: bounded acquisition-capacity control

Registered 2026-11-05 after R99-D1 and before any new teacher response or
optimizer step. R98/R99 and their failed generated-plan branch stay closed.
This is a conventional full-model sequence-distillation control, not a novel
ABI method, production core, or change to LayerCake runtime code.

## Measured reason

R99-D1 found only 7/24 ABI-plan, 12/24 route-adapter distillation and 13/24 LoRA
autonomous training-example passes. Their teacher-forced visible-response token
accuracies were 76.48%, 77.79% and 79.72%, versus 47.65% for the parent. Correct-
prefix fact-token accuracies were 97.03%, 97.03% and 98.51%. The small train/dev
response-accuracy gaps do not support treating unseen identifiers alone as the
dominant bottleneck. Do not introduce a lexical wrapper and claim it solves the
measured autonomous-realization problem.

## One intervention

Replace the restricted optimizer parameter scope with all parameters of the
same 87,710,428-parameter LayerCake parent. Everything else in the direct-response
distillation arm stays fixed: 192 normalized R99 training examples, seed 99001,
384 updates, batch two, context 1024, FP32 without scaling/autocast/TF32, AdamW
1e-4, weight decay .01, clipping 1.0, dropout disabled and maximum generation 640.
No extra data, extra updates, learning-rate sweep, early-best selection, resumed
checkpoint or source weights/activations/logits. Count the full parent, optimizer
enrollment, actual changed tensors, memory, wall time and token exposure.

The existing R99 distillation, LoRA and ABI-plan checkpoints are matched
acquisition controls: exactly the same training examples, teacher outputs,
parent, seed, example order and update count. They are not retrained. Different
trainable counts and FLOPs are explicit; this is not an equal-compute or
training-efficiency comparison. Include the unchanged parent.

## Prospective evaluation and source gate

Before acquisition, freeze 48 new development prompts (six per existing task),
with fresh identifiers, changed days, reordered notes and order seed 100002.
These are a resample of the bounded supplied-content family, not broad English.
Capture each source answer once from the same pinned Qwen2-7B-Instruct model
and system message, maximum 384 tokens, and apply the frozen R99 source-substring
normalizer. Reuse the 192 training source records exactly. The unchanged source
gate is >=216/240, >=75% per family, zero collapse and no truncation. A failed
source gate stops training; there is no source retry.

After the single full-scope training run, evaluate all five systems in fresh
processes on the new 48 prompts. Evaluate the full-scope control autonomously on
the same 24 balanced training examples selected before R99-D1. Keep existing
R99-D1 training outputs for the other systems; do not rerun or choose rows by
score. Report both raw and normalized teacher comparisons, paired prompt-level
10,000-resample bootstrap intervals and all collapse/coverage failures.

## Decision and scope

The useful-quality screen remains >=44/48, >=5/6 per task, zero collapse, and
normalized-teacher noninferiority lower 95% bound >=-0.10. This full-scope
baseline cannot promote an ABI product even if it passes. Training fit requires
>=22/24 and zero collapse. Call acquisition-scope sensitivity supported only if
the paired training-pass improvement over both restricted response-only controls
has a positive 95% lower bound. Report mixed or negative outcomes as such.

A failure does not prove LayerCake lacks capacity or that ABI is impossible:
the data, optimization budget and parent remain bounded. A successful full-scope
control would prioritize how ABI acquires/initializes the substrate, not a host
runtime rewrite or another generated-plan sweep. No automatic follow-on run.

Budget: two hours total, <=30 minutes optimization, <=8 GiB new disk, >=30 GiB
initial free disk. GPU acquisition/training; stop on resource, source, numerical,
hash or coverage failure. Preserve all rows and final checkpoint. New source cost
is 48 requests; inherited source accounting and the separate prior R99 48-query
development capture must both remain visible. No teacher at learner inference.
Human fluency, general English, signed host acceptance, purity, minimality,
three-seed replication and same-artifact CPU speed/TTFT/memory remain open.
