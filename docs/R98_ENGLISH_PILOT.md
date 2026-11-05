# R98 English acquisition pilot

Launched 2026-11-04; reviewed 2026-11-05. R98 is closed at a failed source
prerequisite, not a fluent-core or release certificate.

## Completed result

The frozen teacher produced 206/240 mechanical passes, including only 2/30
emails. Thirty emails and one tone-control response reached the 144-token cap;
the gate forbids any truncation. No student training or student generation
evaluation occurred. This is an ABI source-acquisition failure, not a measured
LayerCake language-quality failure.

The [raw responses](../results/plan_english_r98/source/rows.jsonl),
[source accounting](../results/plan_english_r98/source/result.json),
[stop record](../results/plan_english_r98/stop.json), and additive
[recomputed audit](../results/plan_english_r98_review/source_audit.json) are
preserved. Source inference took 843.919 seconds (877.394 seconds total),
generated 15,551 tokens, and peaked at 15,322,043,904 allocated CUDA bytes.
There was zero observed repetition collapse. No teacher logits, activations
or weights were imported.

[R98-C1](../experiments/plan_english_r98_completion/PROTOCOL.md) is a separately
frozen, one-pass completion-budget diagnostic. It does not change this result.
The evaluator's known lexical false positives remain disclosed and unchanged;
this corpus cannot certify semantic quality or fluency.

## Completed C1 result

All 31 completions reproduced their original token prefixes exactly and ended
without truncation. The corpus reached 226/240, but emails were 22/30, below
the unchanged 75% gate (at least 23). Seven emails exceeded 180 words and one
triggered the confirmation check. No student training was authorized.

The original orchestration also encountered an offline tokenizer metadata
lookup error. That error and its stop record remain unchanged. The additive
`python scripts/audit_r98_completion.py` uses the already-pinned local tokenizer
snapshot and recomputes the negative result, token/text identity, provenance and
information counts without new inference or altered thresholds. Its
[audit](../results/plan_english_r98_completion_review/audit.json) and 10 hostile
mutation tests preserve the distinction between operational and scientific
failure. Total source generation including repeated prefixes was 22,198 tokens;
the completed corpus contains 17,734 output tokens across 240 unique prompts.

The source-cap branch is closed. [R99](R99_ENGLISH_PILOT.md) tests a different
compiler-side normalization mechanism, not another source prompt or cap.

## What is being tested

The ABI candidate learns to generate a grounded content plan followed by a
free-token English response. Eight supplied-content task families are covered.
The controls are direct sequence distillation, conventional weight LoRA, and
the unchanged LayerCake parent. All trained arms use the same source examples,
parent weights, seed, example order, updates and padded token workload.
Trainable parameters and actual compute differ and are reported separately.

The inherited parent has **87,710,428 parameters**. Its complete state counts
toward the deployed footprint and imported-information budget. This pilot
does not extract a standalone 530,050-parameter English model from R97 and
does not inherit R97's reasoning or earlier LayerCake speed certificates.

The exact [protocol](../experiments/plan_english_r98/PROTOCOL.md) contains the
budgets, gates, attribution rules and stop conditions. A mechanical screen is
not a fluency or human-quality judgment. Broad natural-language evaluation,
three paired seeds, signed host acceptance and CPU certification remain open.

## Frozen identities

- Implementation/preregistration commit: `31fad09`.
- Input-freeze commit: `cb8bb19`.
- Freeze manifest SHA-256:
  `196d771f8533951fff93722784107d562cb4485218627472f7f0367b3de1ea35`.
- Parent tensor SHA-256:
  `b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e`.
- Unchanged LayerCake source: `a24458818dc4489b4b977c49bc6ea5b8d089ef24`.
- Teacher: Qwen2-7B-Instruct, revision
  `f2826a00ceef68f0f2b946d945ecc0477ce4450c`.
- Frozen population: 192 training and 48 development prompts.
- Training order: LoRA, distillation, ABI plan; one paired seed, 384 updates/arm.

## Historical launch evidence

The parent mechanics check passed on the RTX 3080 Laptop GPU with cached/full
logit maximum absolute difference `0.0000133514404296875`, below `0.001`.
That is an integration mechanics check, not language quality.

The first sandbox background launch exited before creating `launch.json` or
any scientific observations. Its two empty logs remain preserved. The second
launch started the six-hour-bounded orchestrator (PID recorded in
`results/plan_english_r98/launch.json`), passed parent mechanics, and began live
GPU source capture. It wrote each teacher response before the next request.

Read progress without modifying the experiment:

```powershell
Get-Content results/plan_english_r98/pilot.attempt2.stdout.log -Tail 10
Get-Content results/plan_english_r98/pilot.attempt2.stderr.log -Tail 10
Get-ChildItem results/plan_english_r98
```

The source prerequisite had to pass before any student training. A failure writes
`results/plan_english_r98/stop.json` and blocks subsequent stages. A completed
three-arm comparison writes `decision.json`. No automatic larger run follows.
Use `python -m experiments.plan_english_r98.run verify` only for completed
three-arm evidence; incomplete or failed-source evidence rejects certification.

Source capture runs in a separate process that exits before training and
student evaluation. The full teacher is absent from those inference processes.
No R98 learner checkpoint was produced. In general, publication and production
package acceptance require a separate reviewed result.

## Validation at launch

- 13 focused tests passed, including missing/changed input rejection,
  split isolation, control information equality, generated-plan boundaries,
  basic hostile output screens, and zero-function LoRA initialization.
- Changed Python files pass Ruff.
- Repository documentation/link checks pass.

The broader default suite was also run with CUDA hidden from the test process
to leave the GPU available for acquisition: **169 passed, 4 skipped, 2 failed,
16 setup errors**. The two failures were sandbox-denied atomic file replacements;
the complete ten-test `test_capability_pipeline.py` module then passed outside
the sandbox. The 16 setup errors are all historical Phase 7 exact-host bindings:
the old protocol requires package.py SHA-256
`405043c99e52817709aec21cbeb906a8de651f3e9d68850a5b1f18c1294be0dc`,
while the clean current LayerCake checkout contains
`48367e0469ca816cd5016e44750b2ad48b5fc3cab0e113615d96b71ac74009ef`.
The historical verifier correctly refuses substitution. R98 changes neither
file nor certificate. The full default suite is therefore **not green against
this sibling checkout**; use the historical exact host for those replay tests.
This is not a new LayerCake language-quality failure or an R98 result.

Additive review repair (2026-11-05): `python scripts/run_review_tests.py`
reconstructs 46 exact historical files in a temporary fixture using the pinned
LayerCake Git object. The unchanged Phase 7 test file then passes all 16 tests.
The remaining default tests pass 180 with four declared skips. No verifier,
source binding, historical test, or live sibling checkout is modified. This is
raw-evidence recomputation, not fresh model execution or independent hardware.

The controlling whole-moonshot verdict remains `FULL_MOONSHOT_NOT_PROVEN`.
