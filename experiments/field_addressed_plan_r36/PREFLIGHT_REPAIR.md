# R36 zero-row preflight repair

The first training-only representation preflight stopped before selection,
training, evaluation, or result creation because Python `bytes` has no
`casefold` method.  The registered case-insensitive stop-word comparison is
implemented with `bytes.lower()` instead.  No protocol, data, selection,
model, seed, compute, fixture, threshold, or scientific gate changed.
