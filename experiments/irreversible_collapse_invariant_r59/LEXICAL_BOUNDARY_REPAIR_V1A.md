# R59 v1a lexical-boundary repair

R59 v1 produced a numeric disclosed-validation pass, but the protocol's
stronger preservation claim failed: the invariant rejected 22 token
transitions even though ordinary R55 had zero final collapsed validation
rows. The v1 result is therefore rejected and preserved.

The cause is an implementation error. Python's word regex treated the final
decoded BPE fragment as a complete lexical item. A later token can extend that
fragment, so the lexical count was not actually monotonic. The v1a repair
excludes the final regex match whenever it reaches the current decoded
prefix's end. That word becomes eligible only after a delimiter closes it.

Token-run accounting, the two locked thresholds, checkpoint, router, data,
evaluation gates, and all other logic remain unchanged. Validation is rerun in
a new immutable directory. It must additionally prove byte identity with
ordinary R55 for every R55 row whose final trajectory was non-collapsed. No
final-test is authorized unless that audit passes.
