# ABI capability-compiler roadmap

Status date: 2026-08-04

The machine-readable contract
`ABI_CAPABILITY_COMPILER_CAMPAIGN_CONTRACT_V1.json` controls sequencing and
gates. `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V6.json` records live status.
Evidence gates, not experiment counts or dates, close phases.

| Phase | Objective | Status |
| ---: | --- | --- |
| 0 | Definitions, baselines, and preregistration | COMPLETE |
| 1 | Capability inventory and normalized acquisition IR | COMPLETE |
| 2 | Strong matched LoRA and distillation baselines | MACHINE COMPLETE; BLOCKED ON EXTERNAL HUMAN RATINGS |
| 3 | Causal teacher-to-target acquisition | INITIAL BRANCH FAILED; SEQUENCE SUCCESSOR PREREGISTERED; PHASE UNCERTIFIED |
| 4 | Sufficient-information Pareto frontier | LOCKED |
| 5 | Selective reconstruction and bounded exclusion | LOCKED |
| 6 | Composition, portability, and multi-source provenance | LOCKED |
| 7 | Integrated teacher-relative and systems certification | LOCKED |
| 8 | Independent hostile verification and release | LOCKED |

## Completed Phase 1

The sealed Phase 1 artifact contains 7,000 eligible English records, exactly
500 for each of 14 frozen capabilities. It preserves raw and normalized
material, exact source provenance, authoritative generated token IDs,
transformations, labels, hashes, and imported-information accounting.

Search, development, final, domain-isolation, and hostile suites are frozen and
disjoint under the preregistered exact and near-duplicate checks. Domain
reference inventories are evaluation-only. The original abstention shortfall,
the first GPU runtime-conformance failure, and the failed mathematics reference
outputs remain published negative evidence.

Phase 1 did not train or evaluate a student. It closes corpus normalization and
adequacy only.

## Phase 2 — machine campaign complete, human gate pending

All registered baseline training, three-seed reproduction, paired statistics,
checkpoint persistence, cold/warm runtime, and machine verification are done.
The next—and only—Phase 2 action is external:

1. Assign one complete blinded form to each of three independent human raters.
2. Keep the answer key and other raters' work hidden until all forms are locked.
3. Preserve completed forms separately from the immutable templates.
4. Compute the preregistered preference lower bound and rerun the hostile
   evidence verifier.
5. Issue a Phase 2 certificate only if every exit condition passes.

Phase 2 exits only when each mandatory baseline is credible, converged within
its locked budget, reproducible, and verifier-backed. No ABI-superiority claim
is possible in this phase.

## Later gates

The unavailable Phase 2 human ratings were deferred by explicit user
direction, not passed. The hash-bound Phase 3 A0-A4 development branch has
executed and failed its absolute gates. It cannot issue a certificate or open
Phase 4. A future Phase 3 successor must be separately preregistered and must
address the measured sequence-realization bottleneck; repeating the current
branch, adding data or steps, or sweeping nearby output-side cakes is barred.

- Phase 3 must show a causal ABI acquisition candidate beats parent,
  label-free, shuffled, bridge-only, and monolithic controls while passing
  autonomous quality and repetition gates.
- Phase 4 measures only a tested sufficient-information frontier with an
  adjacent lower failure.
- Phase 5 proves bounded exclusion and capability segregation.
- Phase 6 proves composition, exact artifact portability, provenance, and
  selected-only execution.
- Phase 7 certifies one teacher-free integrated candidate against its teacher
  and strongest matched baselines.
- Phase 8 recomputes everything in a clean environment and attacks evidence,
  accounting, labels, identities, teacher absence, and claim language.

## Repository boundary

ABI ends at independently validated acquisition artifacts and integration
requirements. Runtime implementation and product certification belong to the
separate [LayerCake repository](https://github.com/Yoder23/layercake). Detailed
LayerCake phase history is intentionally not duplicated here.

## Permanent stop rules

- Preserve negative evidence; never overwrite a failed phase with a successor.
- Do not repair or select against final-test failures.
- Do not launch a nearby sweep without a measured bottleneck and a bounded,
  preregistered repair.
- Do not call foreign-teacher acquisition lossless, ontology labeling
  exhaustive, or a tested information budget globally minimal.
- Do not start a locked phase early.
