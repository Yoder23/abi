# R13 hostile audit v2 failure

The repaired verifier rejected the v1 hash-consistent forged source row. The
second audit attempt on 2026-09-02 then stopped at
`forged_scientific_boolean_ignored` because the audit incorrectly required the
entire verification certificate to remain byte-identical after changing and
rehashing the run receipt's untrusted `status` field.

The verifier did not consume the status field. It correctly recomputed the
same claim, capability count, package accuracies, and source/package agreement,
while binding the newly rehashed run-receipt identity into a different
certificate identity. The audit harness mistook that expected evidence-identity
change for a scientific-verdict change.

Repair requirement: compare recomputed scientific fields and the explicit
`stored_status_booleans_consumed` count, not receipt/certificate hashes. Run a
new immutable audit revision.

Status: `AUDIT_HARNESS_FAILED_NO_R13_SEAL`.
