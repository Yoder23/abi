# R108: trained-task grounded generation

Status: preregistered, source preparation in progress (2026-11-14).
No inference result is claimed yet. Whole moonshot: `FULL_MOONSHOT_NOT_PROVEN`.

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
