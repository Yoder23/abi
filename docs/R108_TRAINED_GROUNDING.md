# R108: trained-task grounded generation

Status (2026-11-14): **`RESOURCE_BUDGET_STOP_NOT_QUALIFIED`**.
The registered attempt ended incomplete. Whole moonshot:
`FULL_MOONSHOT_NOT_PROVEN`.

## Observed outcome

The fixed 1,500-second control-process deadline expired after **13/96 initial
responses**, with **0 replay responses and 0 recipient executions**. No training
ran. The preserved prefix contains 1,343 output tokens / 4,846 UTF-8 bytes and
1,407.46 seconds of measured generation time. No partial pass rate, confidence
interval, or English qualification is promoted.

Under load, NVIDIA reported 210 MHz SM clocks, 405 MHz memory clocks, active
software power/thermal throttling, and a **requested 10 W power limit versus an
80 W default**. Actual reported board draw was about 24 W; the requested limit
is not a measured power-consumption value. The laptop reported plugged-in power
and 97% battery. GPU temperature was 51 C in the initial recorded observation;
the root cause of the throttling is not established. No power setting or thermal
protection was changed. This is a measured operating constraint plus the locked
time-budget stop, not proof that the hardware can never run the experiment.

Source preparation separately took 1,396.55 seconds for the pinned 8.06 GB
snapshot, verified against upstream Git/LFS identities. The cache is reusable.
Complete-run CPU time and peak-memory receipts are unavailable because the
process did not finish; live resource snapshots are not substitutes for them.

The frozen complete verifier **rejects** this interrupted attempt. An additive
observer validates its intact inputs, raw prefix, timeout, and hardware records;
it cannot produce a scientific qualification or promote an artifact. See the
[review packet](releases/R108_REVIEW.md).

Local validation: 49 focused interruption/raw-evidence checks; 417 supported
tests plus 16 exact historical-binding tests (four declared skips); 30 canonical
documents and tracked Markdown links checked. These verify implementation and
evidence handling, not English quality. No live replay or independent hardware
validation was completed in R108.

## Why this experiment

R107 tested an **untrained coherence route**, not whether acquired task routes
can generate grounded answers. R108 returns to autonomous generation on the
eight R99-trained task families with prospective counterfactual pairs.

One pinned Qwen3-4B-Instruct-2507 feasibility control must qualify before any
recipient comparison. It is not the teacher used to train the retained controls,
not a matched-size baseline, and not a new acquisition method. No model search,
new training, checkpoint modification, oracle answer prefix, or host edit occurs.

## Locked design

- 48 pairs / 96 prompts: six pairs per task, changing identifiers and weekdays.
- Each panel: >=44/48 overall and >=5/6 in every task, zero collapse/truncation.
- At least 44 pairs must pass both members, also >=5/6 per task.
- Unchanged R104 mechanical conditions plus stricter literal-case and explicit
  pending-status-negation checks; 164 author-specified fixtures, not human labels.
- Native source chat and one fixed training-aligned recipient interface.
- Retained parent, full-model, LoRA and adapter-distillation checkpoints;
  fresh-process replay; 90-minute inference budget; no training or rescue sweep.

The LoRA/distillation history is recomputed from its raw logs: 192 examples,
24 per task, 384 updates, identical sample order, and 43,208 loss-token
observations. Their trainable parameter counts differ (2,359,296 versus
3,231,744). This is matched data/exposure, not parameter-matched training.

## Claim boundary and failure ownership

A pass would qualify only these mechanical supplied-note obligations, not broad
English fluency, semantic purity, independent hardware, or human preference.
The screen has known role-binding, scope, and fluency blind spots. All raw
answers remain available for review; no selected examples replace the full set.

An external-control failure is an unmet experimental prerequisite. Poor recipient
answers diagnose acquisition/recipient suitability, not a native LayerCake bug.
A host attribution requires a separately isolated execution discrepancy. The
same-artifact CPU speed, TTFT, memory, packaging and portability gates remain
unmet for any unqualified external English core. No ABI superiority is presumed.

## Review entry points

- [Frozen protocol](../experiments/trained_grounding_r108/PROTOCOL.md)
- [Generation and verifier](../experiments/trained_grounding_r108/run.py)
- [Input/screen contract](../experiments/trained_grounding_r108/core.py)
- [Mechanical tests](../tests/test_trained_grounding_r108.py)
- [Completed-packet hostile tests](../tests/test_r108_completed_verifier.py)

Historical R98-R107 evidence and all recipient checkpoints remain unchanged.

## Evidence and next action

- [Interruption record](../results/trained_grounding_r108/interruption.json)
- [Original timeout](../results/trained_grounding_r108/stop.json)
- [All completed raw responses](../results/trained_grounding_r108/control/initial/raw.jsonl)
- [Under-load hardware observation](../results/trained_grounding_r108/resource_initial.json)
- [Final hardware observation](../results/trained_grounding_r108/resource_final.json)
- [Matched historical training audit](../results/trained_grounding_r108/matched_training_provenance.json)

First resolve or explain the GPU power/thermal operating mode. Any authorized
live retry must preserve this attempt in place and use a new evidence namespace,
with the same quality gates and a newly declared resource budget. Do not replace
the control, rewrite prompts, extend this expired deadline, or start a new
training sweep to evade the resource stop. Quality remains an open question.
