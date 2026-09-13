# R59 verifier-interface repair

The first strict-verifier invocation failed before evidence loading because the
R59 screen module did not expose the candidate, metadata, router, and parent
count aliases expected by the shared fail-closed verifier. The aliases point
directly to the already frozen R55 constants. No evidence, generation,
threshold, metric, or decision logic changes. The failed invocation produced
no receipt.
