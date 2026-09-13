# R43 — Role-tagged shared English-core repair

Status: `PREREGISTERED` before implementation and execution.

R42 compressed twelve packages into one 295,006-parameter / 1,201,572-byte
shared core and scored 139/144 identically on CPU and CUDA, but correctly
failed because tone was 9/12. Three tone failures copied the numeric value
inside the message slot instead of the dedicated identifier; two planning
failures mixed goal, resource, and constraint spans. R42's anonymous slots are
the measured bottleneck.

## Frozen inputs and intervention

- R42 failed result SHA-256:
  `2d190f98b5d385241bda684fe1260e4dfe30b0a98a3d60c217f96b30d0ab24ba`.
- Every R42 source/result/component/fixture hash, all 282 selected teacher
  rows, the 144-row R40 development suite, model geometry, seed 42001,
  optimizer, batch 48, and 6,396 updates remain unchanged.
- The only material change is semantic role preservation inside each fixed
  slot: `slotN=<original_field_name>:<original_field_value>`. The absent slot
  is `unused:unused`; `task=<task>` remains unchanged.
- No new teacher calls, outputs, labels, manual targets, logits, activations,
  source weights, seed/step sweeps, or alternate prompts are allowed.
- Reuse the frozen R42 execution engine with only its canonical-prompt function
  replaced before selection/training/evaluation. Preserve the complete inner
  engine result and bind it into a separate R43 result.

## Gates

The complete R42 gate is unchanged: at least 132/144 functional on each device,
at least 10/12 in every task, exactly 12/12 abstention, 144/144 CPU/CUDA byte
identity, no representation failure or collapse, at most 450,000 active
parameters, archive size at most 20% of R41's twelve-package total, zero-state
at most 36/144 per device and at most 12/144 candidate matches, strict signed
packages, and exact remove/reinstall restoration.

Additionally, the role-tagged core must improve total score above R42's
139/144 and must score at least 10/12 for both tone and planning. All raw inner
artifacts and the outer evidence binding are immutable and fail closed.

A pass is a development result for a compact labeled supplied-content English
core. It is not a hidden replication, unrestricted English, autonomous
ontology discovery, arbitrary-domain extraction, global minimality, or
superiority to LoRA/distillation. The full ABI moonshot remains open.

