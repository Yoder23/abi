# R32 source-interface repair v2

R32 source v1 failed after 21 accepted rows.  Its immutable receipt SHA-256 is
`6d7210e644ce2fb2bdbd94c2d831b031dbe1a2d2bb5b05574cf505540a1708c8`.
The raw rows isolate the failure: the v1 high offset produced eight-digit
values, and Qwen inserted thousands separators (for example `11,201,056`).
The prospective validator correctly rejected those outputs because the source
contained the exact unpunctuated value `11201056`.

V2 changes only the synthetic detail index range.  It uses indices beginning
at 600,000, disjoint from R31 training (400,000 family) and evaluation
(800,000 family) and within the numeric magnitude already exercised by R31.
The 3x3 counterbalance, two accepted rows per cell, source revision, system
prompts, validator, attempt budget, information accounting, and downstream
package protocol are unchanged.  No thousands separators are normalized or
silently accepted.  This is the sole source-interface repair.
