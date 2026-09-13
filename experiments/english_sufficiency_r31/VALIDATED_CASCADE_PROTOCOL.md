# R31 v3 validated capability cascade

The fixed 24-row, 48-row, and minimum-description package sets scored 116,
119, and 101 independently.  Their prospective functional union is 143/144,
with at least 11/12 in every capability and 12/12 abstention.

V3 must establish that result through live execution, not evidence replay.
For each prompt, execute the 24-row package first, the 48-row package only if
the first output fails, and the MDL package only if both fail.  Validation may
use only the prompt and generated output.  It must infer the registered
structural contract from prompt fields/instruction language and may not access
the teacher output or `oracle_task`.  The inferred contract is separately
audited against the oracle after selection.

All 36 signed packages are installed in three isolated LayerCake host
registries.  Exact route, archive hashes, attempt rows, selected output, package
removal, execution count, GPU time, RSS, and active parameter count are recorded.
Pass requires 132/144 overall, 11/12 per oracle task, 12/12 abstention, 144/144
runtime contract inference, 144/144 routes, and 36/36 removal rejection.

This is a disclosed public composition pilot.  It does not yet prove a hidden
replication, unrestricted English, autonomous semantic naming, minimum global
information, or superiority over LoRA/distillation.

