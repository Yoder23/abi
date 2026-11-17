# R109 — bounded trained-task replication

Status on 2026-11-17: **`CONTROL_QUALIFICATION_FAILED`**, fully executed and
replayed. No recipient, new training or native-host experiment ran.

## Result and interpretation

All 96 initial responses and 96 fresh-process replay responses completed. Every
scientific raw field replayed exactly; timing fields are excluded. The preserved
13-row R108 prefix also matches exactly, descriptively only. The panels passed
35/48 and 34/48, versus 44/48 required. Only 32/48 counterfactual pairs passed
both members, versus 44 required. Zero collapse and zero generation-budget stops.

| Task | Panel A / 6 | Panel B / 6 | Both members / 6 |
| --- | ---: | ---: | ---: |
| Summarization | 3 | 1 | 1 |
| Rewriting | 6 | 6 | 6 |
| Email drafting | 4 | 5 | 3 |
| Conversation | 4 | 4 | 4 |
| Format control | 6 | 6 | 6 |
| Tone control | 0 | 0 | 0 |
| Clarification | 6 | 6 | 6 |
| Abstention | 6 | 6 | 6 |

Every task requires at least 5/6 in both panels and paired outcomes. The
registered recipient gate therefore stayed closed. The historical matched
LoRA/distillation, parent and full-model checkpoints remain preserved and were
not executed. This is an experimental prerequisite failure, not a measurement
of ABI acquisition quality or a native LayerCake defect.

### Real adherence failures and evaluator scope errors are separate

The [descriptive inventory](../results/trained_grounding_r109/error_inventory.json)
finds 20 sentence-count failures (all 12 tone responses and eight summaries),
plus seven approval flags. Requested literal coverage is complete on all 96
rows; no unknown identifiers, extra numbers, unsupported days, collapse or
surface-form failures were flagged. These finite checks do not prove general
semantic correctness or English fluency.

The approval regex misreads negative/conditional scope: four flagged responses
say "Nothing is confirmed yet" (or the lowercase equivalent), and three describe
proceeding *once approval is granted*, not approval already granted. These are
author-observed evaluator limitations, not external human ratings. Preserve
every frozen score. The [post-result scope audit](../results/trained_grounding_r109/evaluator_scope_audit.json)
grants **every** approval-flagged row a pass as an explicitly optimistic
descriptive ceiling: 39/48 and 37/48 panels, 37/48 pairs. Even that ceiling fails
qualification; tone remains 0/6 and paired summaries 1/6. It is not a repaired
score, post-hoc promotion or permission to lower the gates.

The measured prerequisite now separates answer-length adherence from detector
scope errors. Do not interpret it as proof the source cannot speak fluent English,
that the recipients are incapable, or that native LayerCake is broken. No new
prompt/model sweep follows automatically. A successor needs a separate prospective
plan addressing evaluator validity and adherence without weakening quality.

### Recorded cost and custody

- 192 model calls, 17,600 generated tokens including replay; zero training or
  source parameters imported into recipients.
- 853.884 seconds of instrumented process wall time; 879.640 seconds of controller
  time through execution completion; 792.813 summed process CPU seconds.
- Peak sampled RSS 1,077,035,008 bytes; peak CUDA allocation 8,187,554,816 bytes.
- The 8,060,915,998-byte source snapshot was reused. Original R108 preparation
  costs are separate. No new weights, optimizer states or downloads were created.

Receipt process timing starts after imports and initial binding verification;
controller timing includes launches and between-stage checks but precedes the
final offline decision recomputation. Preflight/cache verification was not
separately timed as new acquisition. These are operational costs, not an
end-to-end training-efficiency or controlled inference-speed result.

The [decision](../results/trained_grounding_r109/decision.json) SHA-256 is
`57126b03256ca20623e6a527d9b2f38110f755ff003f0ccf170b3042e1762725`.
See the [review instructions](releases/R109_REVIEW.md) for immutable input
identities and public reconstruction. Full moonshot: **not proven**.

The [protocol](../experiments/trained_grounding_r109/PROTOCOL.md) authorizes one
resource-policy-only replication of [R108](R108_TRAINED_GROUNDING.md). The earlier
run remains incomplete: its 13 initial rows cannot substitute for full execution
or replay. R109 uses a separate, write-once evidence namespace. This is a reused
frozen panel, not a newly blind holdout.

## What stays fixed

- The 96 prompts (48 counterfactual pairs), eight tasks, exact scorer and gates.
- Qwen3-4B-Instruct-2507 at its pinned revision, native chat, BF16, greedy decoding.
- Four retained FP32 recipient checkpoints and their training-aligned interface.
- Matched historical LoRA/distillation data, updates, sample order and loss-token
  exposure; their trainable parameter counts are not equal.
- No training, weight changes, new extraction artifacts or native LayerCake edits.
- No GPU power, clock, driver, cooling or thermal-protection adjustments.

The larger control is a feasibility ceiling, not the historical training teacher
or a matched-size baseline. A control qualification failure prevents recipient
execution. A recipient failure does not establish a native LayerCake defect.

## Resource policy

R108's measured prefix projects roughly 2.9 hours per control panel. R109 allows
4 hours per control process, 1 hour per recipient process, and 16.5 hours total,
with at most 960 calls including fresh-process replay. It allows no extra budget
or automatic retry after failure. The evidence cap is 1 GiB; the free-disk floor
is 10 GiB because all source/recipient weights already exist locally and no
weights, optimizer states or downloads are created. Record actual observations,
not the planning projection, as measured costs.

The November 17 preflight found C: completely full. The user authorized removal
of two old public-review checkouts only after verifying retained artifacts and
preserving unique evidence. Root reconstruction receipts, the newer public
checkout, primary repositories and R108 are outside the deletion scope.

Cleanup completed on November 17: 12,309,544,960 free bytes were observed
(11.46 GiB). All 11,232 ABI and 5,362 LayerCake Git objects in the retired
checkouts exist in their respective primary repositories; both primary Git
integrity checks exited zero. The 7,754 remaining non-object files were hashed;
70 unmatched files (1,653,698 bytes) were moved to the local preservation
directory, not deleted. Seven root receipts/scripts remain byte-identical.
The initial duplicate LoRA checkpoint was separately SHA-256 verified before
removal to make room for the audit. The [cleanup summary](../results/review_2026_11_17/cleanup_summary.json)
records the local custody manifests; it is not scientific evidence of acquisition.

## Reproduction and claim limits

Recompute this completed packet with Python 3.10 and the pinned local environment:

```powershell
python -m experiments.trained_grounding_r109.run verify
python -m pytest -q tests/test_trained_grounding_r109.py tests/test_r109_completed_verifier.py
```

The pilot is write-once, not a resume command. Never run it over existing launch
evidence. The narrow mechanical screen does not establish broad
English fluency, human quality, a minimal/pure substrate, ABI superiority,
native-host acceptance, CPU performance, or either complete moonshot.
