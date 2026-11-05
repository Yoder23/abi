# R101 execution and continuation diagnostic

Status: 2026-11-05. Completed and strictly recomputed. The measured execution
paths agree, but the primary continuation contrast is null. No candidate is
promoted. `FULL_MOONSHOT_NOT_PROVEN` remains controlling.

## Why this experiment

[R100](R100_ACQUISITION_CAPACITY.md) failed to qualify an English core even with
all model parameters trainable. R101 spends no additional training or teacher
compute. It asks whether train/inference execution disagrees and whether
reference-prefix assistance changes the remaining factual continuation.

Read the [locked protocol](../experiments/execution_continuation_r101/PROTOCOL.md).
All 24 existing training diagnostic rows and 48 R100 development rows are kept.
They are known diagnostic examples, not a fresh generalization certificate.
The four fixed checkpoints are the R100 full-model control, R99 LoRA, R99
adapter distillation, and unchanged R76 parent. The failed generated-plan
branch is not reopened.

## What distinguishes the measurements

| Measurement | What it can test | What it cannot establish |
| --- | --- | --- |
| Full, padded-batch and cached logits on identical histories | Measured execution-path consistency | Universal host correctness or an automatically localized host defect |
| Correct-prefix token predictions | Fit to a particular reference response | Autonomous fluency or correctness of all alternative phrasing |
| Unassisted live replay | Whether preserved output IDs reproduce | Independent hardware or new English capability |
| 25%/50% correct-prefix continuation | Sensitivity to response history | A deployable teacher-free capability or autonomous quality improvement |
| Remaining-literal contrast | Whether the generated tail covers facts not pasted into the prefix | Semantic entailment, naturalness, or complete removal of oracle assistance |

The primary contrast is the full-model checkpoint's 50% prefix condition on
the 48 R100 development rows. Paired 10,000-resample intervals are fixed in
advance. Other comparisons are descriptive. No outcome can promote an English
core or change the original 44/48 autonomous quality gate.

## Frozen review identity

- Preregistration commit: `7fc21ff`.
- Freeze commit: `da685e4`.
- [Input freeze](../results/execution_continuation_r101/freeze.json): 813 inputs,
  SHA-256 `4c23c21aa554558943153e24f8cd2745d00bc5b2c71828c39341d179415993a5`.
- No LayerCake source changes, optimizer steps, teacher queries, or new source
  weight/activation imports. Each inherited 87,710,428-parameter model still
  counts; this is not a compact artifact or low-information acquisition claim.
- GPU FP32, no TF32/autocast; four fresh checkpoint processes. Bounds: 90 minutes
  total, 20 minutes per arm, 2 GiB new evidence, 30 GiB initially free.

Read-only verification after completion:

```bash
python -m experiments.execution_continuation_r101.run verify
python -m pytest -q tests/test_execution_continuation_r101.py
```

Do not rerun `freeze`, `pilot` or `audit` into an existing evidence directory.
Missing raw logits, token rows, checkpoints, hashes or stale summaries fail
closed. The raw parity matrices are required Git LFS payloads, not optional
screenshots or trusted saved pass flags.

## Failure attribution

Execution disagreement would first be an ABI integration-path finding. It
requires localization across the ABI loader/hooks and underlying host before
blaming native LayerCake. Execution agreement would only exclude a measured
path mismatch as an explanation on those positions. Data coverage, optimizer
adequacy, parent suitability and broad English quality remain separate questions.

No new host, package-lifecycle, CPU-speed, memory, purity, minimality,
LoRA/distillation-superiority, human-quality or independent-hardware certificate
can be inferred from this diagnostic.

## Completed result

All four checkpoints passed both execution comparisons: zero tolerance violations
and zero argmax disagreements over 51,463,168 paired logit elements in total.
Maximum absolute error was 0.000366211. The 64 sampled histories cover eight
prediction positions each; this is not universal numerical equivalence.
All 288 unassisted output-token sequences replayed the historical observations
exactly. All 864 generation records and 288 conditional-token records are retained.

| Checkpoint | Unassisted development / 48 | 25% oracle prefix / 48 | 50% oracle prefix / 48 | Remaining-fact coverage: unassisted -> 50% assisted, eligible rows only |
| --- | ---: | ---: | ---: | --- |
| Full-model control | 20 | 28 | 34 | 24/34 -> 24/34 |
| LoRA | 20 | 22 | 35 | 21/34 -> 22/34 |
| Adapter distillation | 14 | 19 | 28 | 20/34 -> 22/34 |
| Unchanged parent | 0 | 0 | 5 | 0/34 -> 3/34 |

The oracle-prefix columns include supplied answer text and are NOT autonomous
quality. Fourteen development rows have no remaining eligible literal at 50%
and are excluded only from the remaining-fact contrast, not from other reports.
The primary full-model contrast is 0/34 net gain, 95% paired bootstrap interval
[-0.14706, +0.14706]. It does not support a positive remaining-fact rescue.
Neither does this null result exclude all forms of history drift. The increased
joined-response score cannot be advertised as successful acquisition.

| Checkpoint | Correct-prefix response-token accuracy: train | Development | Correct-prefix fact-token accuracy: train | Development |
| --- | ---: | ---: | ---: | ---: |
| Full-model control | 91.44% | 78.91% | 98.51% | 93.84% |
| LoRA | 79.72% | 77.56% | 98.51% | 97.36% |
| Adapter distillation | 77.79% | 75.51% | 97.03% | 96.83% |
| Unchanged parent | 47.65% | 50.41% | 67.82% | 79.40% |

The full-model control fits the training reference more closely than the
restricted controls without a comparable development gain. This is consistent
with limited generalization, but these different examples and single-reference
token metrics do not prove data insufficiency, overfitting as the sole cause,
fundamental model incapacity, or lack of fluency in every alternative response.

The [decision](../results/execution_continuation_r101/decision.json) is recomputed
from raw observations, not trusted status flags. SHA-256:
`c482c6511974c6180081112d0fc8e27aaa5ae1bb4bb4be6e69c28585956a08c6`.

### Costs and checks

The four measured GPU-audit processes took 158.19, 89.78, 117.53 and 57.77
seconds respectively (423.27 seconds total), excluding input checks performed
before each arm's timer and the separate CPU verifier. Peak CUDA allocation was
797,377,024 bytes per arm; maximum sampled process RSS was 1,120,624,640 bytes.
The complete diagnostic evidence occupies 310,674,878 bytes. No new optimization,
teacher requests, imported teacher weights or activations were used. These are
diagnostic costs, not an inference-speed or acquisition-efficiency benchmark;
repository checks ran concurrently during part of the audit.

Local review passed 246 default tests with four documented skips, plus 16/16
exact historical Phase 7 tests. The supported CI test list passed 83 tests,
including 26 R101 checks. Source distribution and wheel builds passed. The
package API remains version `0.4.0a1`; this research does not change it.

### Next discriminating question

Keep the cached host path unchanged absent new contrary evidence. Do not deploy
oracle answer prefixes or reopen the generated-plan branch. The next proposal
should separate acquisition-data coverage/generalization from optimization fit
under controlled information and compute budgets. Parent suitability remains
open and must be measured rather than blamed. No successor training run is
registered or authorized automatically by R101.

## Versioned public review

Research tag: `abi-research-r101-2026-11-05` (not a product release).
The [required artifact manifest](../results/review_2026_11_05/r101_manifest.json)
contains 35 assets totaling 1,714,292,449 bytes, including four inherited
checkpoints and all new raw parity tensors. Manifest SHA-256:
`eeca42d8c1550cd4ec6ec0f16a059470758d5f78e55d112c2f40f057d2d7602f`.

Use sibling public checkouts named `abi_release` and `layercake_release`.
The host checkout is `865a2b9e59e6221caa1185f9b64479d2077953f3`; its Python
bytes match the registered host binding. Pull required Git LFS objects:
the R76 parent, R99 LoRA/distillation, R100 full-model checkpoint, and four
R101 `parity.safetensors` files. Unresolved pointers fail verification.

At the exact tagged ABI commit, the [public-review helper](../scripts/review_r101_public.py)
checks identities, clean source trees, every manifest asset, all 813 frozen
inputs, strict recomputation and 43 focused diagnostic/control tests. Supply
the pinned manifest digest and exact peeled tag commit through its required
arguments; write its receipt outside either checkout. This is evidence
recomputation, not a repeat of GPU inference or independent hardware.

The original GPU execution and all prior checkpoints remain unchanged.
Later documentation may record public reconstruction without moving this tag.
