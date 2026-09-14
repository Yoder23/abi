# R96 bounded bridge repair protocol

R95 correctly failed at 1,328/1,400. The miss was concentrated in structural
family 2 (310/350), and R91's training generator had an untested lexical-length
assumption: it emitted only three-letter plus six-digit labels, whereas R95 was
the first authorized candidate screen to use seven-digit labels.

R96 performs one bounded continuation of the same 530,050-parameter neural
joint-span bridge. It initializes from the immutable R91 checkpoint and trains
for 3,000 additional steps on the same 8,129 immutable teacher-labeled records.
Only meaning-preserving normalization changes: two- through five-letter stems,
four- through nine-digit identities, 32 relation paraphrases, eight neutral
interfaces, and all four candidate-list placements. The generator includes
ledger-like, hierarchy, entailment, mapping, and containment language but no
R95 prompt, output, source score, teacher call, or new factual claim.

The LayerCake parent and teacher-derived artifact remain immutable. The source
teacher is absent. No logits, hidden activations, source weights, templates,
rules, or lookup table enter the package. R95 may be used only as a disclosed
development check after the R96 checkpoint is frozen. Promotion requires a
later, separately preregistered prospective catalog whose rows are unavailable
during R96 training.
