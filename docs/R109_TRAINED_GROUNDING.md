# R109 — bounded trained-task replication

Status on 2026-11-17: preregistration/preflight; **not yet executed**.

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

After the freeze is published, execute with Python 3.10 and the pinned local
environment:

```powershell
python -m experiments.trained_grounding_r109.run pilot
python -m experiments.trained_grounding_r109.run verify
python -m pytest -q tests/test_trained_grounding_r109.py tests/test_trained_grounding_r108.py
```

The pilot is write-once, not a resume command. Never run it over existing launch
evidence. Public reconstruction instructions and the actual outcome will be
added after execution. The narrow mechanical screen does not establish broad
English fluency, human quality, a minimal/pure substrate, ABI superiority,
native-host acceptance, CPU performance, or either complete moonshot.
