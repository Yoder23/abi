# Work package 2: frozen localization design, NOT a run authorization

Freeze together with R117 registration, before seeing any WP1 scores. WP1 must
pass its full absolute gate and replay. Failure closes WP2; no selecting only
successful families, retrospectively lowering thresholds, or relabeling R116.
Even on success require explicit execution authorization, complete checkpoint,
data, code and held-out manifests before the first optimizer step.

**Intervention.** Compare (O) independently correct plan vs (E) ABI-extracted
plan, expressed in the identical subject/event/modality schema. Use one
historical R76 parent initialization, the same installed route-local adapter
architecture and summarization route for both. Plans are TRAINING supervision,
never extra inference inputs. Same prompt-only inference, autonomous plan and
English generation with a fixed parser (malformed plans count as failures).
Do not substitute teacher wording for recipient output or feed an oracle prefix
at evaluation. Log plan correctness and delivered prose separately. Record-only
oracle control quantifies extraction loss before any training.

**Data and allocation.** Disjoint from WP1. Freeze 192 fixed training records,
one matched oracle/extracted pair per record, 48 training-fit probes, 48 new
counterfactual pairs for known-family generalization and a second 48-pair set
from eight completely held-out construction families (6 pairs/family). Both
arms see the same supplied notes and independently vetted English targets.
Quarantined/missing extracted plans receive a missing-plan marker, not a new
record or oracle backfill. Disclose its coverage; adjudicate the extracted
event fields separately from stylistic variation. Plan: subject, milestone,
asserted actuality vs nonactual modality; unknown never treated as factual.
Construct train-only teacher targets in a separate hash-bound dataset; never
train on WP1 evaluation rows. Prompts, targets, teacher revision, extraction
algorithm, schema and full deterministic scorer must be frozen and independently
traceable before execution. No validation/test labels in either generation path.

**Budgets.** Three paired seeds 117201/2/3, 384 updates/arm/seed, batch 2,
context 1024, fixed 3e-4 AdamW (betas .9/.999, eps 1e-8, weight decay .01),
FP32, gradient norm 1, fixed final step, greedy max 640 output tokens. No
sampling sweep, tuning or best checkpoint. Freeze identical train order per
pair. Report actual loss-bearing tokens, parameter-seconds, CPU/GPU time, bytes,
full parent footprint, and teacher preparation separately; matching updates is
not equal compute. Cap 1h/training worker, 12h total compute, >=10 GiB free;
if not feasible stop incomplete, don't reinterpret partial results.

**Decision tree (all three seeds separately).** Fit gate >=44/48 training
probes and >=5/6 per family; generalization >=44/48 in both panels and paired,
>=5/6 in every family, no truncation/collapse, no admitted unsupported
assertion, exact identifiers, complete supplied facts and locked English
quality/length/format gates as applicable. One independent replay of final
generations. Report paired cluster-bootstrap O-minus-E differences (10k draws,
seed 117201); call a differential supported only with strictly positive lower
95% bound. O fits but E fails with a supported gap: investigate extraction/
representation; both fail fit: inspect interface/optimization, not native host;
both fit yet fail withheld families: generalization; mixed-seed, ambiguous
or null contrasts: inconclusive, no causal promotion. Both pass: eligible
only to propose a new matched ABI/LoRA/distillation acquisition comparison.
This design cannot certify product fluency, faster training or host execution.
