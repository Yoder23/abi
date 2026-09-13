# R35 typed sublexeme-slot representation

R34 failed closed before completing its development screen because a teacher
target joined and lowercased a supplied phrase (`Coordinator 35` to
`coordinator35`).  The immutable R34 failure record SHA-256 is
`72ccfc0358ad16a4f530847e1980c683737afdf95e8ddaf307b52e14e298c17c`.

R35 changes only the lossless action representation.  Every alphanumeric
lexeme is split at letter/digit boundaries before the frozen R34 dynamic-slot
rule is applied.  The neural decoder must still causally select every literal
and slot action.  Concatenation, whitespace, punctuation, and case remain
explicit output actions; the renderer performs no semantic generation.
Unknown literals, unavailable slots, excess slots, and non-lossless training
targets fail closed.

The data, twelve task boundaries, width-64 two-encoder/two-decoder
PortableTokenPlan, optimizer, 1,066 batches of 24 per task, maximum 384 output
actions, and zero-new-teacher budget are unchanged from R34.  The sole
development screen is the already disclosed R31 v4 fixture.

Pass requires at least 132/144 functional rows, at least 11/12 in every task,
12/12 abstention, 144/144 route and contract agreement, zero representation
failures, and lossless representation of every training target.  A pass only
authorizes prospective fresh replication and LayerCake packaging.  A failure
closes this representation.  This is not unrestricted English, autonomous
label discovery, arbitrary-domain extraction, global minimum information, or
LoRA/distillation superiority evidence.
