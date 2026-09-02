# R13 hostile audit v1 failure

The first hostile audit attempt on 2026-09-02 failed closed before any R13
seal. The case `hash_consistent_forged_source_row` was incorrectly accepted.

The mutation changed a source row's stored `canonical_prediction` to another
incorrect value, recomputed the raw-file SHA-256, updated the receipt, and
recomputed the receipt evidence hash. Aggregate canonical accuracy did not
change because both values were wrong. The verifier checked aggregate metrics
but did not require the stored canonical prediction to equal the argmax of the
stored canonical probability vector.

Repair requirement: recompute the canonical argmax for every source row. The
repaired verifier and a new hostile audit must be committed and executed under
a new immutable output name. The original run/package evidence is unchanged;
only the verifier was defective.

Status: `VERIFIER_AUDIT_FAILED_NO_R13_SEAL`.
