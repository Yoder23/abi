# R44 prospective replication certificate

Verdict: **PASS — strictly verified frozen shared-core replication**

The exact R43 candidate package
`6ccb115de59e71b1c70efcfe2b5a1673b88d8f1478fb0059d8577ef5e2b6e6a4`
was not retrained or modified. It scored 141/144 on a newly materialized
144-prompt fixture on both CPU and CUDA. Every task scored at least 10/12;
abstention was 12/12. Route and contract checks were 288/288, candidate outputs
were non-collapsed 288/288, representation failures were zero, and CPU/CUDA
outputs were byte-identical 144/144.

The exact zero-state package scored 0/24 task probes and matched no candidate
output. Strict removal rejected generation, reinstall restored the output, and
a one-byte-corrupted archive was rejected.

The independent verifier ignored stored verdicts, reconstructed the complete
fixture, recomputed all 312 stored rows and aggregates, authenticated both
packages, compared candidate tensors to the frozen state, proved every control
tensor was zero, rejected the corrupt archive, then replayed all 288 candidate
rows through fresh CPU/CUDA hosts and repeated lifecycle restoration.

Result SHA-256:
`41f8625fba5b73538a572bb5bc9fe059da83f605b7933201909d76141176eec8`.
Evidence digest:
`6271e8e2e6963849a26123c7caacdc3d6f97b8426a23fed9595119bd1cb68a63`.

This certifies the 296,554-parameter / 1.21 MB shared core only for the bounded
twelve-task supplied-content interface. Teacher-relative quality on this new
fixture, unrestricted English, autonomous discovery, arbitrary domains,
global minimality, and LoRA/distillation superiority remain open. The full ABI
moonshot remains **OPEN**.

