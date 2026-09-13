# R34 typed dynamic-slot representation

R33's counterfactual normalization failed its single development rescreen at
122/144.  Its immutable result SHA-256 is
`5ce858f6af067cae0a8af3a5a1a2f7a9100806d8f7abeefe5e4a714e494e5867`.
The reasoning package generated a stable fluent frame and exact unseen nonce on
12/12 rows but omitted the unseen item-number slot on 12/12.  Planning showed
the same source-position instability.  Training fit was effectively exact.

R34 changes representation, not data volume, width, depth, optimizer, or
training compute.  A typed dynamic-slot tokenizer maps each distinct source
lexeme that contains both letters and digits, or a numeric value greater than
99, to one of four position-stable slot actions.  Target occurrences of those
exact values use the same slot action.  The neural decoder still selects every
literal and slot action causally; slot realization only losslessly restores the
prompt value, analogous to a pointer tokenizer.  Unexpected pointer actions,
unknown literals, missing slots, and more than four slots fail closed.

Train one unchanged width-64, two-encoder/two-decoder PortableTokenPlan per
registered contract from the original 48 accepted R31 teacher rows.  Each gets
the exact R31 level-48 budget of 1,066 batches of 24 (25,584 sampled examples),
384 maximum actions, and no new teacher calls or tokens.  Save every state and
tokenizer hash.

The exact R31 v4 fixture is the sole development rescreen.  Pass requires at
least 132/144 functional, at least 11/12 per contract, 12/12 abstention, exact
registered routes/contracts, lossless target representation for every training
row, and zero unknown or unexpected pointer actions.  A pass only authorizes a
LayerCake host implementation and separately committed unseen replication; it
does not itself prove package transfer, unrestricted English, autonomous
labeling, minimum information, arbitrary domains, or LoRA/distillation
superiority.  A failure closes this representation.
