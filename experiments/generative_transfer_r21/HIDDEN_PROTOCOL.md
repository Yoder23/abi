# R21 fresh hidden replication

The public R21 candidate and its 24 packages are frozen before this protocol.
The hidden split contains 120 new prompts: 20 for each of prose, summary,
email, bullets, clarification, and abstention. It uses new composed instruction
strings and new slot indices. A 256-bit seed is generated only after this code
is committed; the configuration commits its SHA-256 before the seed is
revealed.

The pinned unchanged source produces one response per hidden prompt on GPU.
It is not asked to label the prompts. The frozen ABI word classifier labels
instructions, and the exact frozen packages execute without training. Raw
sequence, labeled monolith, and ABI-factorized packages are compared across all
three frozen seeds on the same rows.

The hidden result passes only if:

- the labeler is correct on at least 114/120 prompts;
- every factorized seed scores at least 15/20 in every task;
- every factorized seed has at least 114/120 non-hallucinating and 114/120
  non-collapsed outputs;
- every factorized seed is no worse than the teacher, raw-sequence control, or
  labeled-monolith control on functional quality;
- every signed package is byte-identical to the public candidate;
- all 1,080 GPU rows and 54 task-covering CPU rows execute; and
- the source model is absent from package execution.

No retraining, model selection, threshold change, instruction edit, row
deletion, or second seed is allowed after reveal. Passing promotes only the
bounded six-task supplied-data transfer mechanism. General English and the ABI
moonshot remain open.
