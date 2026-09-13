# R42 development diagnosis

Verdict: **FAIL**, with a strong compression result.

The single 295,006-parameter package scored 139/144 on CPU and 139/144 on
CUDA with 144/144 byte identity, zero representation errors, no collapsed
candidate outputs, and 12/12 abstentions. Its 1,201,572-byte archive is 9.75%
of R41's 12-package total (10.26x smaller). The signed all-zero state scored
0/144 on each device and matched no candidate outputs, establishing that the
learned state is causal.

The registered task floor failed: tone was 9/12 rather than 10/12. Planning
was exactly 10/12. The five failures are preserved in `evaluation.jsonl`.

The failures identify a concrete representational bottleneck. R42 replaced
task-specific field names with anonymous `slot0` through `slot3`. Three tone
failures copied the numeric identifier embedded in the message slot instead of
the dedicated identifier slot. Two planning failures mixed goal, resource,
and constraint spans. The shared model therefore lacks an invariant indication
of each slot's semantic role; aggregate training fit (99.22%) did not solve
unseen role binding.

The evidence supports one materially targeted successor: retain the same five
fixed fields and same source width, but prefix each slot value with its original
field-role label inside that slot. This adds no teacher rows or model capacity
and should improve role binding without sacrificing package size or execution
speed. Seed/step/data/prompt sweeps are not justified.

Result SHA-256:
`2d190f98b5d385241bda684fe1260e4dfe30b0a98a3d60c217f96b30d0ab24ba`.

The full ABI moonshot remains **OPEN**.

