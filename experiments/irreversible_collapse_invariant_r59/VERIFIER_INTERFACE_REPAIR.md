# R59 verifier-interface repair

The first strict-verifier invocation failed before evidence loading because the
R59 screen module did not expose the candidate, metadata, router, and parent
count aliases expected by the shared fail-closed verifier. The aliases point
directly to the already frozen R55 constants. No evidence, generation,
threshold, metric, or decision logic changes. The failed invocation produced
no receipt.

A second pre-evidence invocation exposed two more interface defects: the shared
verifier also required the frozen R55 preflight function, and its live-replay
hook otherwise defaulted to the R49 generator. The repaired verifier aliases
the unchanged R55 preflight and explicitly supplies the R59 v1b generator with
fresh verification-only telemetry. This changes neither frozen evidence nor
the screened endpoint; it is required so a live verification actually replays
the claimed runtime instead of a parent runtime. Both failed invocations
terminated before reading or certifying evidence and produced no receipt.
