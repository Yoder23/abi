# R97 verifier transcription repair

The first strict-verifier invocation failed closed before producing a receipt
because its hard-coded result evidence digest contained one duplicated `e`.
The immutable result file, raw rows, aggregates, gates, and evidence digest were
not changed. This repair makes the verifier's expected constant exactly match
the digest stored in and recomputed from the result. No scientific check or
threshold was changed.

The second invocation failed identically on a duplicated `f` in the separately
copied source-evidence digest. That constant is corrected from the immutable
source result. Again, no receipt was emitted and no evidence or gate changed.
