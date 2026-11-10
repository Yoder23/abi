# R103: recipient and interface suitability

Registered 2026-11-10 after the negative R102 result, before any new model call.
One bounded, zero-training diagnostic. No weights, routes, decoding guards,
learning rates or historical gates are changed. Publication is authorized.

## Evidence-supported intervention

The R76 acquisition/screen used `prompt + newline`; R98-R102 used
`Task: <capability>` and a terminal `Response:` wrapper. Compare exactly these
two input interfaces on four unchanged research checkpoints: R76 parent,
R100 full-model control, and R99 FP32 LoRA and adapter-distillation controls.
Their parameters, tokenizer and fixed capability route do not change.
R99/R100 learners trained on the wrapped format; an asymmetric effect is possible.

The external positive control is pinned Qwen2-7B-Instruct revision
`f2826a00ceef68f0f2b946d945ecc0477ce4450c`, previously used as the teacher.
It must qualify on 48 new qualification inputs using its native chat template
and raw autonomous output: >=44/48, >=5/6 in each of eight tasks, zero collapse
and zero truncation. No normalization, answer selection, oracle prefixes or
teacher judging. If it fails, stop before recipient comparisons, preserve the
failure and report CONTROL_QUALIFICATION_FAILED. No substitute control or prompt
retry is authorized. "Qualified" means this bounded mechanical task screen,
not independently certified broad-English fluency or semantic correctness.

## Fixed comparison

If qualification passes, evaluate the control and four recipients on 48 separate
comparison inputs, each with both interfaces. Native means raw prompt plus one
newline for LayerCake research checkpoints, and the unchanged system/user chat
template for Qwen. Benchmark means the exact common Task/Response completion
string for every model, with no chat wrapper. These are within-model paired
interventions. Qwen-native versus recipient-native is a task positive control,
not an architecture-only, size-matched or compute-matched superiority comparison.

All 96 inputs are frozen before any outputs. They are disjoint entity/day/order
counterfactuals of the known eight R98 supplied-content templates, NOT unseen
semantic tasks or a broad-English generalization set. Every prompt explicitly
requests at most 120 words to align the input with the already bounded response
screen. This addresses measured teacher verbosity, applies to every arm and
interface, and is fixed before qualification. The historical screen (including
its 180-word bound) is unchanged. New results cannot be causally compared with
older prompt populations. Qualification and comparison identifiers are disjoint
from each other and all R98-R102 training/development identifiers.

Greedy generation, max 640 new tokens, no guard or cleanup beyond ordinary
special-token decoding and outer whitespace removal. Fail preflight if any input
plus that budget exceeds 1,024 tokens. Use local GPU: FP16 for Qwen, FP32 for the
87,710,428-parameter recipients, TF32 disabled, eval mode. Source parameter scale,
precision, prior acquisition and native templates differ and remain confounds
for between-model causal claims. No CPU performance certificate is inferred.

## Registered decisions

Two primary within-checkpoint contrasts on the same 48 comparison cases:
native minus benchmark for (1) the parent and (2) full-model control. A positive
97.5% paired percentile-bootstrap lower bound supports interface sensitivity for
that checkpoint only. Use 10,000 resamples, seed 103001. The two contrasts use
conservative 97.5% intervals; report 95% too. LoRA, adapter and Qwen contrasts
are prespecified descriptive controls, not a search for a winning prompt.

Report all ten cells, per-task screens, content omissions, format/contradiction,
repetition/collapse and truncation. A recipient screen additionally requires
>=44/48, >=5/6 per task, zero collapse/truncation and a 95% teacher-relative
paired lower bound >=-0.10 against the same comparison's Qwen native condition.
No checkpoint is a fluent-core product promotion, even if this screen passes.

If the native control also fails the comparison screen, transportability of its
qualification is not established and recipient-gap attribution is inconclusive.
If the control passes but recipients fail in both formats, this supports a
limitation of these checkpoint/acquisition/interface combinations on this task
family, not impossibility of the tasks or a proven native LayerCake defect.
Null interface contrasts do not prove all possible interfaces equivalent.

## Custody, execution and bounds

Freeze code/tests/protocol, both input sets, all four checkpoint/tokenizer bytes,
ABI loader code, external host source, pinned control inventory and versions.
Commit the successful freeze before model execution. Store exact prompt and
output token IDs/text, raw mechanical screens, stop reasons, per-call timing,
sampled RSS, peak CUDA allocation, source inventory and before/after file hashes.
Live recipient forwards must report only the requested cake and six matching
deep-adapter routes; mismatch stops as an unlocalized integration finding.

Report source requests/tokens/bytes, total inherited parameters, artifact bytes,
wall/CPU cost and GPU. New training steps, copied source parameters, imported
logits/activations and trained bridge parameters are zero. Source weights remain
only in the external control, never a recipient. Previous acquisition costs are
not erased. Historical LayerCake and ABI lineages remain separate.

Bounds: 90 minutes from pilot start, 25 minutes per model process, at most
144 control requests and 384 recipient requests (528 total), <=1 GiB new
evidence, >=20 GiB initially free. Stop on qualification, context, execution,
hash, numerical or resource failure. Preserve partial logs. No automatic retry,
training, model substitution, wrapper sweep or successor experiment.
