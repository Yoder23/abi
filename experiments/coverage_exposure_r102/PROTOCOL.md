# R102: entity coverage versus optimization exposure

Registered 2026-11-10 after R101, before new source calls or training. This is a
bounded acquisition diagnostic using conventional LoRA and adapter sequence
distillation. Neither data augmentation nor extra updates are renamed ABI
novelty. The generated-plan and full-unfreezing branches remain closed.

## Reason and factors

R101 found measured execution-path agreement and a null primary correct-prefix
continuation contrast. Its full-model reference-token accuracy was 91.44% on
training versus 78.91% on development. That does not establish data insufficiency
or an optimization defect. R102 varies two factors in a fixed factorial design:

- Data policy: the exact 192 R99/R100 teacher targets, versus four invertible
  supplied-entity renamings per original row (768 synthetic rows). English
  wording, non-entity facts and labels are unchanged. Every transformed prompt
  and target has exactly the original recipient-token length. This tests entity
  coverage, NOT semantic/topic coverage, autonomous labeling or more teacher
  knowledge. Record synthetic bytes separately from teacher-imported bytes.
- Optimization exposure: fixed endpoints at 384 and 1,536 updates of each
  uninterrupted training trajectory. No best-checkpoint selection, learning-rate
  sweep, new optimizer, optimizer restart, or post-result extension.

Cross these factors with both original R99 methods: rank-32/alpha-64 conventional
LoRA and route-specific deep-adapter sequence distillation. Four trajectories,
eight predeclared endpoints, all from the same frozen 87,710,428-parameter R76
parent. No from-scratch English acquisition or full-model training. Unchanged
parent and retained R100 full-model checkpoint are evaluation-only references.

Use seed 99001, batch two, padded context 1024, response-only cross entropy,
AdamW lr 1e-4 / weight decay .01, gradient clipping 1.0, dropout disabled, FP32,
no autocast/scaler/TF32. Preserve the R99 sample-order algorithm. In augmented
training, choose alias (epoch mod 4); all four trajectories see the same original
row IDs in the same order. Prompt and loss-token counts match exactly per batch.
Within a method, trainable scope and padded work match at each endpoint; LoRA
and adapter FLOPs/parameter counts differ and must not be called equal compute.
Serialize merged LoRA endpoint states without mutating live optimizer parameters.

## Entity transformation and prospective evaluation

Derive aliases from the supplied NOTE/Person literals, not evaluation answers.
Replace only whole identifier occurrences, including request/fact co-reference;
invert the mapping to recover the original prompt and teacher target exactly.
Preserve recipient/coordinator suffix roles. Candidate sampling is seeded and
bounded to 4,096 attempts per row/alias. Accept on invertibility, unchanged
mechanical screen and exact recipient-token lengths only. Never filter on model
quality. Freeze all mappings and synthetic targets before GPU work.

Freeze 96 prospective development rows before source capture: 48 with fresh
identifiers of the original T-prefixed shape, 48 with shifted N102-prefixed
identifiers; six per task/stratum. They use the same bounded supplied-content
task family, not new broad English. Require no exact prompt/identifier overlap
with original, augmented or prior development populations. Evaluation answers
never enter training or alias selection.

Capture each of the 96 new source answers once from the existing pinned
Qwen2-7B-Instruct revision, same system message and greedy max 384 tokens as
R99/R100. Reuse the original 192 raw/normalized training answers unchanged.
Apply only the frozen R99 source-span normalizer. Source prerequisite: at least
260/288 mechanical passes, at least 27/36 per task, zero collapse/truncation.
Report raw and normalized source separately. Source failure stops all training;
no prompt/cap retry, deletion of bad examples or lowered threshold.

## Measurements and interpretation

After training, evaluate all eight endpoints and two retained references in fresh
processes on all 96 development rows. Evaluate all eight endpoints on the fixed
24 R99-D1 original training examples too. Greedy maximum 640 tokens, no oracle
answer prefix, guard, retrieval, postprocessing rescue or teacher at inference.
Record every output token/text, literal/format/collapse observation and resource
receipt. Retain every checkpoint, including failed endpoints.

Two predeclared diagnostic contrast families on the 96 paired prompts:

1. Entity-coverage effect: augmented minus repeated data at 1,536 updates.
2. Optimization-exposure effect: 1,536 minus 384 updates with repeated data.

For each family require a positive paired 97.5% percentile-bootstrap lower bound
in BOTH methods before labeling that factor supported. Resample the 48 base
template instances as clusters, keeping their two identifier-shape variants
together, rather than treating those variants as independent observations.
Use 10,000 resamples, seed 102001. This intersection rule with two contrast families is deliberately
conservative; other per-stratum, task and interaction contrasts are descriptive.
Report both 95% and 97.5% intervals and all raw deltas. Null results do not prove
absence of an effect. One seed does not establish universal causality.

The original autonomous screen remains unrelaxed: for BOTH 48-row strata,
>=44/48, >=5/6 per task, zero collapse and normalized-teacher noninferiority
95% lower bound >=-0.10. Training fit remains >=22/24 and zero collapse. Even a
screen pass is not human fluency, semantic entailment, an ABI invention, general
English, an installable capability, or a CPU-performance certificate. No new
candidate is promoted into the LayerCake product by this diagnostic.

## Custody, bounds and failure separation

Freeze code, tests, catalogs, schedules, all alias targets and inherited inputs
before source capture. Recompute claims from raw rows/hashes; missing/stale or
nonfinite evidence fails closed. Record exact raw/source token counts, unique
prompt/answer UTF-8 bytes, derived synthetic bytes, all inherited weight costs,
trainable/deployed parameters, optimizer exposure, wall/CPU hours, CUDA peak and
sampled RSS. No new teacher weights, logits or hidden activations are imported;
source inference uses the declared local GPU, no external hardware.

Budget: three hours total, <=30 minutes per training trajectory, 96 new teacher
requests, 6,144 total optimizer steps, <=6 GiB new evidence, >=30 GiB initial
free disk. Preserve partial failures. Stop on source, numerical, mapping,
coverage, hash or resource failure; no automatic retry or successor sweep.
Do not stop unrelated work. Publication follows the user's current privacy
choice; a local pre-measurement Git freeze is required either way.

ABI acquisition or augmentation failure is not a native LayerCake defect.
LayerCake source, prior checkpoints, canonical interfaces and historical gates
remain unchanged. Host attribution requires an independently localized defect.
Human quality, three-seed replication, independent hardware, broad English,
purity, minimality and same-artifact CPU speed/TTFT/memory remain open.
