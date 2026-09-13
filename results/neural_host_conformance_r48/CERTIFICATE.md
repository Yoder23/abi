# R48 hidden-state router conformance certificate

Scientific verdict: **FAIL**.

R48 changed only the 7,690 task-classifier parameters of the R47 neural core.
Every transformer and task-cake tensor remained byte-identical.  The classifier
used 1,400 labeled search prompts, no response target, and no teacher execution.
It reached 1,400/1,400 on search but only 1,375/1,400 (98.21%) on the disjoint
validation prompts, below the preregistered 99% gate.  Its deterministic
rotated-label accuracy was 0/1,400.

Because the router prerequisite failed, the integrated generation rescreen was
not run and the candidate was not promoted.  This closes mean-hidden linear
classification as the routing interface for this lineage; it does not alter
the R47 neural acquisition evidence.

- checkpoint SHA-256: `33edc8c82a25cd389d78b0e6449b23ec89af3daba8139b7e94217cba99dce62b`
- evidence SHA-256: `1910d9baaaeb0055ad0b972cbddf2ff1afb6fe076bf9f487ccda2fe6e640947e`
- frozen non-classifier state unchanged: `true`

Full ABI moonshot: **OPEN**.
