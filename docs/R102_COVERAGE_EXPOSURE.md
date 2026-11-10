# R102 entity coverage and optimization exposure

Status: 2026-11-10. Completed and strictly recomputed locally. Neither registered
factor met its support criterion, and no checkpoint passed the quality screen.
`FULL_MOONSHOT_NOT_PROVEN` remains controlling. Publication awaits the user's
current privacy preference; this document does not imply a new public release.

The initial corpus freeze stopped before any teacher call or optimizer step.
After 317 aliases, variant 1 of `r98-train-summarization-20` exhausted its
4,096-proposal bound. The [preserved rejection trace](../results/coverage_exposure_r102_preflight_v1/candidates.jsonl)
contains no jointly length-matched candidate. A read-only analysis found 46
compatible project-ID components and 69 compatible person-ID components in
that same pool; combining recorded proposals 40 and 144 preserves the exact
119 prompt / 44 target token counts and inverse text identity. This is a sampler
construction issue, not student failure, data insufficiency or host incapacity.
The [A1 construction amendment](../experiments/coverage_exposure_r102/PREFLIGHT_A1.md)
prospectively registers component-profile selection from the same bounded
proposal stream, with all token/identity constraints and scientific gates intact.
The repaired constructor subsequently produced all 768 exact-length,
inverse-verified aliases. The first successful input freeze is now complete;
source qualification began only after that freeze was committed. The original
failed attempt and every rejection remain unchanged.

## Local preregistration identity

- Original preregistration: `b647de0`; preserved preflight diagnosis: `1fcefbb`.
- A1 pre-execution amendment: `8549782`; successful input freeze: `935a41d`.
- [833-input manifest](../results/coverage_exposure_r102/freeze.json), SHA-256
  `81787684eb335149fc196e8ff6bcd9ada4c564e1ba82c4068fbf70c597360f25`.
- 768 synthetic targets; 401,848 unique synthetic prompt UTF-8 bytes and
  200,984 unique synthetic target UTF-8 bytes. These are derived data, not new
  teacher-generated knowledge. The inherited parent and teacher costs still count.
- Twenty focused tests passed before model execution, including the failed
  joint-pairing regression and non-mutating LoRA checkpoint serialization.
- These commits and the completed run are local; no new public release is claimed.

## Question

[R101](R101_EXECUTION_CONTINUATION.md) found measured execution-path agreement,
exact unassisted replay and a null primary answer-prefix intervention. The
full-model control fitted its training references more closely than development.
R102 tests whether entity coverage or greater fixed optimization exposure helps
under matched workloads. It does not assume the answer or blame LayerCake.

Read the [locked study protocol](../experiments/coverage_exposure_r102/PROTOCOL.md).

| Factor | Levels | Held constant |
| --- | --- | --- |
| Entity coverage | Exact 192 teacher examples; four invertible entity-renamed versions per original | Teacher wording and knowledge, labels, prompt/target token lengths |
| Optimization exposure | 384 and 1,536 updates | Parent, optimizer, learning rate, sample order, batch/context, precision |
| Method controls | Conventional LoRA; deep-adapter sequence distillation | Same data policy, source information and example schedule |

There are four training trajectories and eight predetermined checkpoints, not
an open sweep. The unchanged parent and retained R100 full model are additional
evaluation-only references. There is no new full-model training or generated-plan
candidate. Parameter counts and FLOPs differ between methods and are disclosed.

## Evaluation and claim limits

Ninety-six new development inputs cover the existing eight supplied-content
tasks, with 48 original-shape and 48 shifted-shape identifiers. A fixed source
gate precedes all training. Evaluation answers never enter training. The two
identifier variants of each base case remain together in cluster-bootstrap
comparisons; they are not treated as independent replicates.

These are identifier counterfactuals of the 48 R100 cases: non-identifier notes
and task requests are reused, not new semantic tasks. Prospective freezing
applies to the new inputs and measurements. Do not describe this as an unseen
semantic-task or broad-English holdout.

Source qualification completed at 282/288 normalized mechanical passes versus
272/288 raw passes, with zero collapse/truncation. Exactly 96 new source calls
used 15,724 input tokens and 7,622 generated tokens; source inference took
419.90 seconds. These calls supply evaluation references, not new training
answers. The original 192 training answers remain unchanged.

Both repeated-data 384-update checkpoints reproduce the original R99 LoRA and
distillation safetensors byte-for-byte. This implementation cross-check confirms
the low-exposure anchor; it is not a new quality or superiority result. The
local full review passed 266 default tests with four documented skips plus
16/16 exact historical Phase 7 tests (282 total).

The autonomous screen remains >=44/48, >=5/6 per task, zero collapse and
teacher-relative noninferiority, separately in both strata. These mechanical
proxies cannot certify fluency or semantic entailment. One seed and one bounded
task family cannot prove broad English, universal ABI superiority or a minimum
English core. Any positive result remains a conventional acquisition-control
result until it earns the separate product and scientific gates.

The budget is three hours, 96 new teacher evaluation calls, four trajectories
totalling 6,144 optimizer steps and at most 6 GiB new evidence. No threshold,
learning-rate or architecture sweep follows automatically. LayerCake source,
historical evidence and prior checkpoints remain unchanged.

## Completed result

All 6,144 optimizer updates were finite. Each trajectory consumed exactly
172,832 response-loss tokens with matched original-row order and prompt-token
counts. All eight checkpoints and ten complete development evaluations remain
preserved, including negative outcomes. There were 960 development generations
and 192 original-example generations; none used an oracle answer prefix.

| Method | Data policy | 384 updates: passes / 96 | 1,536 updates: passes / 96 |
| --- | --- | ---: | ---: |
| LoRA | Repeated original data | 51 | 46 |
| LoRA | Entity-renamed data | 14 | 34 |
| Adapter distillation | Repeated original data | 32 | 39 |
| Adapter distillation | Entity-renamed data | 10 | 24 |

The unchanged parent scored 0/96. The retained R100 full-model reference scored
53/96. The raw teacher scored 90/96 and the normalized teacher 95/96. The gate
still requires at least 44/48 **in each stratum**, plus every per-task,
collapse and teacher-relative constraint. None qualified. Do not compare these
96-row scores directly with previous 48-row scores as a causal improvement.

| Registered contrast | LoRA difference; 97.5% cluster interval | Adapter difference; 97.5% cluster interval |
| --- | --- | --- |
| Renamed minus repeated, 1,536 updates | -12.50 pp; [-26.04, +1.04] | -15.63 pp; [-27.08, -5.21] |
| 1,536 minus 384, repeated data | -5.21 pp; [-20.83, +10.42] | +7.29 pp; [-2.08, +17.71] |

Neither contrast has a positive lower bound in both methods. The particular
renaming recipe reduced the endpoint scores; this is not a general claim that
data diversity is harmful. More exposure did improve the augmented conditions
(14 -> 34 and 10 -> 24), but these prespecified descriptive comparisons do not
replace the primary tests or create a quality pass.

The [decision](../results/coverage_exposure_r102/decision.json) has SHA-256
`177117d4cb59e79139366cb09eda856c82dbf257c50fd973d5758effb7de9a4e`.
Its verdict is `BOUNDED_FACTORIAL_COMPLETED_NO_PRODUCT_PROMOTION`.

### Important measurement boundaries

- The augmentation changes identifier namespaces and can break the original
  shared numeric stem between project and person identifiers. Evaluation keeps
  that shared-stem structure. This is a compound renaming policy under a fixed
  budget, not an isolated increase in semantic English-data coverage.
- The 24-example panel is truly in-sample for repeated-data controls (13 -> 19
  LoRA passes; 12 -> 18 adapter passes). It is **not in-sample for augmentation**:
  all original identifiers were excluded from that training corpus. The frozen
  files call it `training_fit`; for augmented models its correct interpretation
  is original-format transfer/retention (2 -> 10 and 4 -> 6), not training fit.
  The historical field names and results are preserved, not silently rewritten.
- Evaluation answers are excluded from optimizer targets and alias construction.
  Source qualification reads the complete source ledger before training; this
  is not filesystem-isolated test custody or capability-blind host certification.
- One seed, known task templates and mechanical literal/format screens do not
  establish broad English, fluency, semantic entailment, latent purity or minimum
  information. Neither conventional control is relabeled as ABI novelty.

## Descriptive error inventory

The additive [post-run audit](../results/coverage_exposure_r102_error_audit_v1/summary.json)
uses no new model execution and changes no gate. At 1,536 updates, repeat-policy
LoRA and adapters missed required content in 43/96 and 52/96 rows respectively;
their augmented counterparts did so in 59/96 and 72/96. Both identifier and
non-identifier content (such as the planned day) are affected. Format,
contradiction and collapse failures also remain.

The audit found no wrong identifier exactly matching the recorded training-ID
set under the locked parser. Thus simple verbatim training-ID recall is not
established; this does not exclude subtler overfitting. A post-hoc identifier
replacement alone would leave many other screen failures. Counts overlap and
are not a replacement quality score. Audit SHA-256:
`deb8d1c5b1df12d49e6a232ecda1d48d3c04f92b07df22ed0b4ff53a957042dc`.

## Attribution and next decision

The source qualified, transformations were invertible, training executed, and
the early controls exactly reproduced R99. The acquisition/recipient learning
combination still failed functional grounding and adherence. No native
LayerCake runtime defect was localized, and no new LayerCake artifact is promoted.

The inherited recipient is the R76 research checkpoint, with the documented
[R47 reconstruction](../experiments/neural_host_reconstruction_r47/PROTOCOL.md)
and [R76 role-invariant acquisition](../experiments/role_invariant_choice_r76/PROTOCOL.md)
history. That provenance is not an independent general-English qualification
for this task. R102 did not compare it with a separately qualified English
recipient positive control. Recipient suitability, the input/learning interface,
and broader semantic-data coverage remain unresolved.

Do not extend the same alias/step-count sweep. The next proposal should first
test recipient/interface suitability against a qualified English positive
control and examine grounded content selection. No successor run is registered
or started by this result.

## Costs and local review

The four 1,536-update training endpoints recorded 297.74, 298.64, 283.08 and
280.38 seconds respectively (LoRA repeat/augment, adapter repeat/augment).
These are measured training-process walls, including checkpoint saves, not a
CPU-acquisition or equal-FLOP efficiency claim. Some repository tests ran
concurrently. Trainable counts were 2,359,296 for LoRA and 3,231,744 for adapters;
all 87,710,428 inherited/deployed parameters still count.

Peak CUDA allocation was 3,252,809,728 bytes for LoRA and 3,107,237,888 for
adapters; maximum sampled training RSS was 1,762,041,856 bytes, not an OS
high-water guarantee. The primary study directory contains 2,851,109,835 bytes.
The R98-R102 Qwen control series now totals 463 source requests and 37,569
generated tokens, including repetitions and old evaluation calls. Those totals
do not replace the older parent-acquisition information account. No new source
parameters, logits or hidden activations were imported.

Final local review passed 271 default tests with four documented skips and
16/16 exact historical Phase 7 tests: 287 total. The separately executed supported
CI surface passed 108 tests locally; this is not a GitHub Actions result. The
wheel and source archive were produced and passed archive-readability checks.
Documentation links and focused lint checks pass. Software checks do not promote
any failed scientific screen.

Read-only scientific verification:

```powershell
python -m experiments.coverage_exposure_r102.run verify
python scripts/run_review_tests.py
```

The raw rows, checkpoints, operation logs, failed preflight, and descriptive
audit are retained locally. Publication remains pending the user's privacy choice.

### Complete local inventory

The [review manifest](../results/review_2026_11_10/r102_manifest.json) inventories
782 repository files, including every owned frozen dependency, all new study
evidence and the final test receipts. It includes 14 LFS paths (12 unique payloads),
not just the eight new checkpoint paths. The listed file sizes total
4,642,747,745 bytes, counting repeated paths; this is not incremental disk use.
External LayerCake source bindings and the pinned source-tokenizer inventory are
recorded separately. Manifest SHA-256:
`45a03f18d0f567e8a5c06fc08f40c31cbc0adb6ce5baf0c3d0b7e9cf4e1a39be`.

This is a local custody inventory, not a claim that R102 has been uploaded,
reconstructed publicly or independently reproduced. Publication must include
the complete transitive payload set before a public-replay claim can be made.
