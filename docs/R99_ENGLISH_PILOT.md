# R99 grounded English acquisition pilot

Status: 2026-11-05. **No fluent-English core has been certified.**

## What changed and why

R98 and C1 could not qualify raw source emails. C1 resolved truncation but
seven emails remained overlong. R99 selects complete teacher sentences that
cover requested supplied facts, retaining exact source span offsets, greeting
and sign-off. It neither asks the teacher to rewrite nor invents target prose.
Other task targets stay unchanged. The rule minimizes only word count among
available sentence units, not English knowledge or global information content.

The [R99 protocol](../experiments/grounded_english_r99/PROTOCOL.md) freezes the
normalizer and 48 fresh development prompts before capture. All 192 training
rows remain present; controls receive the same normalized and raw archive.
The old development population is not reused for the new learner comparison.

## Current evidence

| Boundary | Observation | Interpretation |
| --- | --- | --- |
| R98 raw source | 206/240; 31 truncated | Failed acquisition prerequisite |
| C1 completed source | 226/240; emails 22/30 | Failed unchanged family gate |
| R99 normalized source | 230/240; emails 30/30; zero collapse | Passed mechanical data qualification, not semantic fluency |
| Initial R99 learner | One update, then non-finite FP16 gradients | Failed numerical implementation; no model-quality measurement |
| Two-batch numerical audit | Original failure reproduced; unit-scale FP16 and FP32 finite | Loss-scaling/precision cause isolated within this audit |
| R99-FP32 | All three arms completed; ABI plan 10/48, distillation 20/48, LoRA 27/48 | Failed acquisition pilot; no candidate promoted |

The [numerical audit](../experiments/grounded_english_r99/NUMERICAL_AUDIT.md)
reproduced 50,688 non-finite gradient elements in four abstention-route tensors.
It saved the tensors and recomputed their counts. It did not test fluency or
establish a LayerCake architectural regression. The failed launch is retained.

The [FP32 protocol](../experiments/grounded_english_r99_fp32/PROTOCOL.md)
restarts every arm from the unchanged parent: ABI plan, sequence distillation,
and conventional rank-32 LoRA. Same data, seed, sample order, 384 updates,
batch 2, 1024-token padding, optimizer and quality gates. No autocast, scaler
or TF32. No failed checkpoint is resumed. No additional sweep follows.

## Completed matched comparison

The 48 prospective prompts use the frozen mechanical screen, not human fluency
ratings. The required candidate score was at least 44/48, at least 5/6 in each
family, zero collapse and the registered paired-control gates.

| System | Mechanical passes | Collapse cases | Trained parameters | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| Raw teacher | 40/48 | 0 | Not trained | Not applicable |
| Normalized teacher targets | 43/48 | 0 | Not trained | Not applicable |
| Unchanged parent | 0/48 | 2 | 0 | 0 |
| Sequence distillation | 20/48 | 4 | 3,231,744 | 67.93 |
| Conventional LoRA | 27/48 | 0 | 2,359,296 | 75.73 |
| ABI generated plan | 10/48 | 5 | 3,231,744 | 68.47 |

All learners completed 384 finite-gradient FP32 updates. All deployed models
contain 87,710,428 parameters; LoRA was merged before fresh-process inference.
ABI plan consumed 125,316 loss-bearing tokens versus 43,208 for each response-
only control, despite equal padded positions and example order. Timings include
concurrent CPU review work and do not establish efficiency or speed dominance.

The candidate-minus-LoRA paired 95% bootstrap interval is **[-0.500, -0.229]**;
candidate-minus-distillation is **[-0.333, -0.104]**. The plan hypothesis is
rejected for this setting. A positive gain over the weak unchanged parent is
not sufficient, and the better LoRA control still fails the quality threshold.

Read the [decision](../results/grounded_english_r99_fp32/decision.json) and
[raw teacher comparison](../results/grounded_english_r99_fp32/raw_teacher_reference.json).
The final checkpoint hashes are:

- ABI plan: `1255ec333ff5fbc5f76bc83335a4b3b38aa374ce147fbd30bca839f3da912c1a`.
- Distillation: `859501c04cf7d82bbf07c23e765cebee1ef64ccb640e832553af5c40feeb8d0c`.
- LoRA: `f370385cac214f89760a1d9f98f568c6d1e20269933078260d07b569d02c9610`.

These are failed research candidates, not signed production English cores.

## Measured limiting factors and next decision

Required supplied facts were missing in 38/48 ABI-plan outputs, 27/48
distillation outputs and 16/48 LoRA outputs. The ABI arm also produced 22/48
invalid plan structures; LoRA produced none of the observed repetition collapses.
These counts overlap and are not an additive partition. Raw examples show
omitted coordinators/days, changed identifiers and repeated statements, not
just cosmetic format discrepancies.

Close this autoregressive-JSON-plan branch. Do not scale its steps or data
without a new, measured mechanism. The next bounded research question should
target reliable supplied-fact retention and autonomous realization, for example
a learned copy/grounding path that does not require serializing the whole plan.
That is a hypothesis, not an implemented fix or permission to reuse this
development set as a new hidden test. Any successor needs a new preregistration,
fresh evaluation and the same matched controls.

The source pipeline qualified its normalized data; this learner comparison
failed. Host mechanics passed and LayerCake source was unchanged, but that does
not establish unrestricted host English quality or rule out capacity limits.
An independent host regression is not demonstrated by this result. No CPU
dominance benchmark was promoted for these failed-quality checkpoints.

Across R98/C1 and the fresh R99 capture, the accounting retains 319 source
requests and 26,156 generated teacher tokens, including repeated prefixes.
The R99 corpus contains 81,992 raw target bytes and 63,260 normalized target
bytes. This reduction is not a minimum-English-information proof.

## Identities and reproduction

Use the [reproduction packet](R99_REVIEW.md) and machine-readable public review
inventory at `results/review_2026_11_05/r99_manifest.json` for artifact and
environment requirements. The annotated review tag does not certify the core.

- R99 preregistration: `03755a5`; source freeze: `3b37b6b`.
- R99 freeze SHA-256:
  `50ea05480885e4c330064560d5e66e959a778f2739c7241b7225dd9ee2bf8ed1`.
- FP32 preregistration: `a5b7a33`; input freeze: `21ee6b8`.
- FP32 freeze SHA-256:
  `c11004f4c58ea32cfcc949376ca6345651319483bde153bd12941a3b243f7ac9`.
- Parent: 87,710,428 parameters, tensor SHA-256
  `b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e`.
- Host Python sources: LayerCake `a24458818dc4489b4b977c49bc6ea5b8d089ef24`;
  later documentation-only commits do not change these bound source bytes.

Use `python -m experiments.grounded_english_r99.run verify-source` to recompute
normalization and source qualification. The FP32 `verify` command requires all
three final checkpoints, all training logs and all development outputs; missing
inputs fail closed. Do not rerun `freeze` or `pilot` into an existing output
directory. Preserve partial evidence on interruption.

Progress is recorded in `results/grounded_english_r99_fp32/pilot.stdout.log`;
errors in `pilot.stderr.log`; a terminal operational failure writes `stop.json`;
a complete comparison writes `decision.json` plus `raw_teacher_reference.json`.

## Claim ceiling and next gates

This remains a single-seed supplied-content pilot. The entire inherited parent
counts, so this is not a tiny independently deployable delta or a fact-free
English substrate. Source text selection is preprocessing; the learner uses a
distillation loss. Raw and normalized teacher comparisons must be distinguished.
Mechanical keyword/format screens cannot establish semantic entailment or human
fluency, and their known false positives remain disclosed.

Even a continuation-eligible pilot needs broader prospectively frozen tasks,
three paired seeds, independent human quality, immutable signed host acceptance,
and same-artifact CPU >=2x, throughput retention, TTFT, memory, persistence and
hostile tests. No old LayerCake speed or different ABI quality lineage can fill
those gates. `FULL_MOONSHOT_NOT_PROVEN` remains the controlling verdict.
