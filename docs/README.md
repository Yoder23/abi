# ABI documentation

This is the documentation front door. Start with the path that matches your
goal rather than reading the campaign archive in filename order.

## Reading paths

| Audience | Read in this order |
| --- | --- |
| New user | [Project README](../README.md) -> [Getting started](GETTING_STARTED.md) -> [Architecture](architecture.md) |
| API integrator | [Getting started](GETTING_STARTED.md) -> [Architecture](architecture.md) -> [Repository map](repository-layout.md) |
| Scientific reviewer | [Project status](PROJECT_STATUS.md) -> [Claim ledger](../CLAIMS.md) -> [Proof ledger](ABI_PROOF_LEDGER.md) -> [Independent-review handoff](INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md) |
| Human rater/custodian | [Human-rating handoff](PHASE2_HUMAN_RATING_HANDOFF_V1.md) -> [Binding repair](../ABI_CAPABILITY_COMPILER_PHASE2_HUMAN_SCORING_BINDING_REPAIR_V2.json) |
| External operator | [Phase 8 reproduction](PHASE8_EXTERNAL_REPRODUCTION_V1.md) -> [Review packet](../review_packet/00_READ_ME_FIRST.md) |
| Contributor | [Repository map](repository-layout.md) -> [Contributing](../CONTRIBUTING.md) -> [Repository contract](../AGENTS.md) |

## Canonical current documents

- [Project status](PROJECT_STATUS.md) is the concise state dashboard.
- [Claims](../CLAIMS.md) defines what the evidence does and does not support.
- [Architecture](architecture.md) explains the compiler/host boundary.
- [Getting started](GETTING_STARTED.md) covers the installable alpha surface.
- [Repository map](repository-layout.md) separates supported code, research
  drivers, immutable evidence, and review material.
- [Technical claims](ABI_TECHNICAL_CLAIMS.md) and [final
  results](ABI_FINAL_RESULTS.md) contain the detailed R7 release record.
- [Research status](research-status.md) preserves the later R8-R23 and R97
  experimental interpretation.
- [Peer-review readiness](PEER_REVIEW_READINESS.md) lists the current candidate
  checks and the external gates that local development cannot complete.

## Evidence lineages

R7, V1089, R97, and R21-R23 answer different bounded questions. They are not
one final artifact and their measurements cannot be mixed to fill missing
gates. Each result document names its own source, package, host, rows,
verifier, and claim ceiling.

For the additive research sequence, use the result documents named
`R10_*` through `R23_*`. Negative results are intentionally retained.

## Source-of-truth order

1. immutable tag and hash-addressed payload;
2. preregistered protocol or contract;
3. raw observations;
4. fail-closed verifier and receipt;
5. current status/claim documentation; and
6. roadmap prose.

The current whole-moonshot verdict remains `FULL_MOONSHOT_NOT_PROVEN` until
external and scientific gates are actually completed.
