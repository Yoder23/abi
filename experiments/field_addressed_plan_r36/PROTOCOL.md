# R36 field-addressed neural plan

R35 proved that sublexeme splitting makes all 576 training targets losslessly
representable, but failed the disclosed screen at 99/144 with 22 unseen-source
representation failures.  Planning was 2/12 and reasoning 8/12 despite nearly
exact training fit.  The immutable R35 result SHA-256 is
`608283dffc03a577e25a005507e98c284a19435c9b58e8e455f0a559735fcea1`.

R36 materially changes the recipient representation.  Each supplied field is
encoded in a fixed, field-addressed block.  Content lexemes and numbers copied
from those fields are emitted as neural field-position actions; literals are
emitted from the learned fixed vocabulary.  Letter/digit boundaries remain
lossless.  The renderer only concatenates selected literal or source actions
and performs no task logic, factual lookup, template selection, inflection, or
semantic generation.  The instruction string is excluded after the external
router selects the task package, forcing the package to use supplied semantic
fields rather than the known instruction-paraphrase cycle.

For each task, select at most the 24 shortest source-accepted responses that
are exactly representable under this frozen scheme, breaking ties by record
ID.  Record every selected and quarantined row.  Train the unchanged width-64,
two-encoder/two-decoder PortableTokenPlan for 1,066 batches of 24, retaining
R34/R35's total exposure and optimizer.  No teacher calls, logits, hidden
activations, source weights, manual response templates, or evaluator-derived
targets are allowed.

The sole development screen is the disclosed R31 v4 fixture.  Pass requires
at least 132/144 functional rows, at least 11/12 in every task, 12/12
abstention, 144/144 route and contract agreement, zero representation errors,
and at least 12 selected source responses per task.  A pass authorizes only a
prospective fresh replication and LayerCake packaging.  A failure closes this
representation.  This is not unrestricted English, autonomous label
discovery, arbitrary-domain extraction, global minimality, or
LoRA/distillation superiority evidence.
