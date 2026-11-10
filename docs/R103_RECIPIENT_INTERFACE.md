# R103 recipient and interface suitability

Status: preregistered 2026-11-10; no new model result yet.
`FULL_MOONSHOT_NOT_PROVEN` remains controlling.

R102 did not qualify any English core or support its two registered primary
factors. R103 tests one measured interface difference: the R76 acquisition
used raw prompts followed by a newline, whereas R98-R102 used a Task/Response
wrapper. There is no training, checkpoint selection, optimizer change or new
LayerCake runtime implementation.

Read the [locked protocol](../experiments/recipient_interface_r103/PROTOCOL.md).

## Fixed design

First qualify the pinned Qwen2-7B-Instruct control on 48 new supplied-content
inputs using native chat and raw autonomous outputs. Require at least 44/48,
at least 5/6 per task, zero collapse and zero truncation. Qualification failure
ends this run before recipient comparisons; no substitute model or retry.

If that prerequisite passes, compare both formats on a separate 48-input set:

| Checkpoint | Native interface | Benchmark interface |
| --- | --- | --- |
| Qwen positive control | System/user chat template | Plain Task/Response completion |
| R76 parent | Raw prompt plus newline | Plain Task/Response completion |
| R100 full-model control | Raw prompt plus newline | Plain Task/Response completion |
| R99 LoRA | Raw prompt plus newline | Plain Task/Response completion |
| R99 adapter distillation | Raw prompt plus newline | Plain Task/Response completion |

All recipients retain their exact weights, tokenizer and capability route.
Qwen is an external task-feasibility control, not a size-matched transformer
baseline, transferred English core or teacher resident inside LayerCake.
Differences in scale, precision and prior acquisition preclude architecture-only
causal claims. Within-checkpoint input-format comparisons are paired.

The two primary contrasts are native-minus-benchmark for the parent and the
full-model control, with 10,000-resample paired 97.5% intervals. Other checkpoints
are descriptive controls. A recipient must still pass every unchanged mechanical
quality condition and teacher-relative noninferiority; no screen alone promotes
a fluent product core, broad English or CPU performance.

## Limits and custody

The new inputs reuse known supplied-content task templates, with new identifiers,
day assignments and note order. They are not unseen semantic tasks. Every arm
receives the same explicit 120-word request, fixed before qualification to align
with the existing bounded-response screen. No source-span normalization, answer
prefix, output repair, retrieval or teacher scoring is used.

All input sets, scientific code, tests, source inventory and checkpoint dependencies
are frozen and committed before GPU work. Raw tokens/text, exact prompt rendering,
route traces and resource receipts are retained. The bound is 90 minutes, at most
144 control and 384 recipient generations, zero optimizer steps and 1 GiB new
evidence. Historical failures and unrelated running work are preserved.

Failure to qualify the control is a control/setup finding. If the control passes
but the recipients fail, the limitation belongs to those acquisition/checkpoint/
interface combinations until a separate test localizes a native LayerCake defect.
This diagnostic does not certify or disprove the underlying host architecture.

```powershell
python -m experiments.recipient_interface_r103.run verify
python -m pytest -q tests/test_recipient_interface_r103.py
```

The verifier is for completed evidence, not permission to overwrite a run.

Software preflight: 20 focused tests passed, and the full review passed 313
tests with four declared skips outside the Windows sandbox. The initial sandbox
run's two temporary-file rename permission failures remain in the
[operational record](../results/review_2026_11_10/r103_preflight_operations.md)
and separate failed JUnit receipt. No model-result claim follows from these checks.
