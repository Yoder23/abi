# R21 self-hash assurance repair

The immutable registered-label control receipt contains the correct SHA-256 of
its canonical document before `evidence_sha256` was inserted. Its frozen
validator incorrectly recomputes the hash with `evidence_sha256` still present,
which cannot equal the stored pre-insertion hash.

This additive assurance repair changes only recomputation: remove exactly the
top-level `evidence_sha256` field, canonicalize the remaining document with the
already frozen function, and compare that digest to the stored value. It binds
the frozen control config, receipt, rows, and labeler byte-for-byte. It cannot
change labels, responses, models, training, evaluation, gates, or claims.

The repaired check must validate the existing artifact before training. The
same rule must validate any downstream wrapper and engine result. A positive
quality result still requires a separate fresh live execution verifier; this
repair does not authorize promotion by stored evidence alone.
