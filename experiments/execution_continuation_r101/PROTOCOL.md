# R101: execution consistency and continuation intervention

Registered 2026-11-05 before any R101 model execution. This is one no-training
diagnostic of preserved R100/R99 checkpoints. It is not an ABI invention,
English-quality certification, new data campaign, or LayerCake source change.

## Question and fixed population

R100 full-model training reached 20/48 development and 17/24 training passes.
Unfreezing alone did not solve the recipe. Before spending more training compute,
separate a train/inference execution mismatch from conditional reference fit
and sensitivity to the model's generated history. Data sufficiency, parent
suitability and optimization adequacy are not fully identified by this audit.

Use all 24 R99-D1 training examples and all 48 R100 development examples, without
score-based selection. These are known diagnostic populations, not a new hidden
quality test. Keep full-model R100, R99 LoRA, R99 adapter distillation and the
unchanged R76 parent. The generated-JSON-plan branch stays closed. Reuse the
exact R100 normalized source answers; no new source queries or optimizer steps.

## Stage A: execution contract

For each checkpoint choose the lowest ID in each task/split: 16 examples.
Compare all 50,257 logits at the first eight response prediction positions
(or the complete shorter target), using the identical gold token history:

1. unpadded full-sequence forward;
2. batch-two, right-padded-to-1024, training-style forward with attention mask
   and prompt lengths (second row is the next fixed catalog row);
3. prompt prefill plus incremental `decode_step`, feeding gold tokens.

Persist all three FP32 logit matrices as safetensors, token IDs and row metadata.
Recompute maximum error, tolerance violations and argmax disagreements from the
published tensors. Tolerance is `abs(other-reference) <= 1e-3 + 1e-4*abs(reference)`
for every element, plus zero argmax disagreements. Stop on any violation; do
not tune the tolerance. This flags a path inconsistency, NOT automatically a
native LayerCake bug: ABI loader/hooks, integration, underlying runtime and
floating-point effects would require subsequent localization. Passing only
supports equivalence on the measured histories and positions, not all hosts.

## Stage B: conditional fit and oracle-prefix intervention

On all 72 examples per checkpoint, record teacher-forced per-token argmax/NLL,
target IDs and response/fact masks. Exact-reference prediction is not fluent
English quality; alternative wording may be valid.

Generate three paired responses: no supplied answer prefix, the first floor(25%)
of reference tokens, and the first floor(50%). Feed those prefix tokens through
the same incremental path, then greedily generate the remainder. Total response
budget is 640 tokens INCLUDING supplied tokens, also respecting context 1024.
No teacher at inference. No logits processing, repetition guard or rescue
rerun. Persist supplied IDs, generated IDs, stop reason and every raw response.
The unassisted output IDs must exactly replay the preserved R100 development
rows and respective R100/R99-D1 training rows. Stop on mismatch, preserving it.

Report mechanical scores on joined responses as explicitly ORACLE-ASSISTED,
never as autonomous quality. The primary contrast instead tests required
literals wholly AFTER the supplied reference prefix. A literal is eligible
only if it occurs in the reference and no occurrence starts before the cut's
character boundary. Compare coverage of all these eligible literals in the
generated tail against their coverage in the unassisted output. Rows with no
eligible literal are excluded from that contrast only, with counts published.
This avoids credit for facts pasted into the prefix, but does not remove the
extra planning/style information that prefix assistance supplies.

Primary diagnostic contrast: full-model checkpoint, 50% prefix, R100 development
rows; paired prompt-level 10,000-resample percentile bootstrap, seed 101001.
A strictly positive 95% lower bound supports sensitivity to supplied response
history in this population. It does not prove exposure bias is the sole or
dominant cause, nor that the prefix is deployable. All other conditions/splits
are descriptive controls, not additional opportunities to declare a winner.
Report response/fact token accuracy and NLL separately, including parent and
matched LoRA/distillation controls. A null result does not exclude history drift.

## Custody, resource bounds and decisions

FP32 GPU execution, eval mode, seed 101001, four CPU threads, no TF32/autocast.
Four fresh checkpoint processes; at most 20 minutes per arm, 90 minutes total,
2 GiB new diagnostic evidence, at least 30 GiB initially free. No changes to
checkpoints, host, source text, historical decisions, quality gates or tokenizer.
Bind code, protocol, tests, source inputs and model hashes before execution.
Raw tensors and rows are required: missing/stale/nonfinite evidence fails closed.
Do not trust saved pass booleans. Preserve partial/error results; no retry into
an existing evidence path. A small operational repair requires additive custody.

New training/source/weight-import costs are zero. The inherited 87,710,428
parameters per checkpoint and all historical acquisition costs still count.
Store wall/CPU time, peak CUDA allocation, sampled RSS, versions and device.
No CPU speed advantage, compactness, semantic purity, ABI superiority, host
defect, human quality, independent hardware or full moonshot promotion follows.
After one complete audit, publish the outcome and identify what remains
unresolved; no automatic training follow-on or nearby prefix sweep.
