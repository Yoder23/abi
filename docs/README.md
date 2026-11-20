# ABI documentation

This is the documentation front door. Start with the path that matches your
goal rather than reading the campaign archive in filename order.

## Reading paths

Latest: [R115 structural decoding](R115_STRUCTURAL_DECODING.md), complete bounded
mechanical pass. [Review and limitations](releases/R115_REVIEW.md): supported
sentence gain, exact replay, remaining semantic coverage gaps; no acquired core.

Preserved experiment: [R114 scope and sentence structure](R114_SENTENCE_STRUCTURE.md)
(complete, qualification failed). [Review guide](releases/R114_REVIEW.md) separates
sentence serialization, evaluator limitations and untested acquisition/host paths.

| Audience | Read in this order |
| --- | --- |
| New user | [Project README](../README.md) -> [Getting started](GETTING_STARTED.md) -> [Architecture](architecture.md) |
| API integrator | [Getting started](GETTING_STARTED.md) -> [Architecture](architecture.md) -> [Repository map](repository-layout.md) |
| Scientific reviewer | [Project status](PROJECT_STATUS.md) -> [Claim ledger](../CLAIMS.md) -> [Proof ledger](ABI_PROOF_LEDGER.md) -> [Independent-review handoff](INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md) |
| Human rater/custodian | [Human-rating handoff](PHASE2_HUMAN_RATING_HANDOFF_V1.md) -> [Binding repair](../ABI_CAPABILITY_COMPILER_PHASE2_HUMAN_SCORING_BINDING_REPAIR_V2.json) |
| External operator | [Phase 8 reproduction](PHASE8_EXTERNAL_REPRODUCTION_V1.md) -> [Review packet](../review_packet/00_READ_ME_FIRST.md) |
| Contributor | [Repository map](repository-layout.md) -> [Contributing](../CONTRIBUTING.md) -> [Repository contract](../AGENTS.md) |

## Canonical current documents

- [R115 structural decoding](R115_STRUCTURAL_DECODING.md): 45/48 constrained pairs,
  24/24 sentences, supported gain and exact replay. Mechanical qualification
  only; [review](releases/R115_REVIEW.md) documents unscored semantic limitations.

- [R114 scope and sentence structure](R114_SENTENCE_STRUCTURE.md): all 576 calls
  and exact replay; revision reaches 44/48 pairs but misses the tone floor;
  structured revision scores 39/48 and fails quality noninferiority.
  [Public evidence and download verification](releases/R114_PUBLICATION.md).

- [R113 storage-recovered replication](R113_STORAGE_RECOVERED_REPLICATION.md):
  all 576 calls and exact replay; feedback fails at 40/48 pairs versus 44.
  [Review guide](releases/R113_REVIEW.md) separates adherence and evaluator limits.

- [R112 count-feedback study](R112_COUNT_FEEDBACK.md): finite scope passed;
  source study stopped at the disk floor after 530/576 calls. See the
  [resource-incomplete review packet](releases/R112_REVIEW.md).
  [Verified cleanup and recovery](R112_CLEANUP.md): 61 duplicate copies removed;
  2,915 distinct historical target paths preserved with hard links and full R111 custody rechecked.

- [R111 scope/revision study](R111_SCOPE_REVISION.md): finite prerequisite passed
  800 new and 793 development cases; all source calls replay exactly. Sentence
  adherence improves, but revision qualification fails at 41/48 pairs versus 44.
  [Review guide](releases/R111_REVIEW.md) states the evidence boundary and costs.

- [R110 scope/adherence study](R110_SCOPE_ADHERENCE.md) completed: finite challenge
  600/600, but both source interfaces failed qualification with exact replay.
  [Review packet](releases/R110_REVIEW.md) and [verified cleanup](R110_CLEANUP.md)
  document the result, dependency custody and recovered storage. No product
  promotion, R109 change or native LayerCake result.

- [R109 trained-task replication](R109_TRAINED_GROUNDING.md) keeps R108's scientific
  gates and matched controls fixed under a new bounded resource policy. Its
  completed control failed qualification with exact replay; no recipient ran.
  [Review instructions](releases/R109_REVIEW.md) preserve that negative result.
  The [public reconstruction receipt](R109_PUBLIC_RECONSTRUCTION.md) confirms
  the published packet without promoting qualification.

- [R108 trained-task grounding](R108_TRAINED_GROUNDING.md) preserves an incomplete
  resource-budget stop, not an English qualification. Its
  [review packet](releases/R108_REVIEW.md) distinguishes interruption verification
  from the unchanged complete scientific verifier.
  The [public reconstruction receipt](R108_PUBLIC_RECONSTRUCTION.md) records
  `REPRODUCED_RESOURCE_STOP`, not a generation-quality pass.

- [R107 retention/intervention result](R107_RETENTION_INTERVENTION.md) separates
  predictive-loss interference, repetitive generation and untrained-route retention.
  Its [public reconstruction receipt](R107_PUBLIC_RECONSTRUCTION.md) verifies the
  published tagged packet without promoting it to English quality.

- [R106 grammar/interface diagnostic](R106_GRAMMAR_INTERFACE.md) failed its separate
  external grammatical-discrimination control; no recipient or training run.
- [R105 external scope falsification](R105_EVALUATOR_SCOPE.md) rejects a frozen
  conventional evaluator on pre-existing external gold labels; no acquisition ran.

- [R104 evaluator and control qualification](R104_EVALUATOR_CONTROL.md) preserves
  the failed prerequisite and additional evaluator limits; no recipient ran.
- [Weight-space research assessment](WEIGHT_SPACE_RESEARCH_2026_11.md) separates
  externally reported editing results from ABI evidence and future comparisons.

- [R103 recipient/interface suitability](R103_RECIPIENT_INTERFACE.md) is the
  completed control-qualification failure; recipient comparisons did not run.
- [R102 entity coverage versus optimization exposure](R102_COVERAGE_EXPOSURE.md)
  preserves the completed negative acquisition-control study, not an ABI novelty claim.
- [R101 execution and continuation diagnostic](R101_EXECUTION_CONTINUATION.md)
  tests runtime-path agreement and explicitly oracle-assisted continuation.
- [R100 capacity control and R99 fit audit](R100_ACQUISITION_CAPACITY.md)
  separate acquisition fit from autonomous generation and host attribution.
- [R98 result and history](R98_ENGLISH_PILOT.md) records the failed source
  prerequisite, exact identities, costs and preserved raw evidence.
- [R98-C1 protocol](../experiments/plan_english_r98_completion/PROTOCOL.md)
  defines the completed, failed one-pass completion-budget follow-up.
- [R99 English pilot](R99_ENGLISH_PILOT.md) follows source normalization,
  precision diagnosis and the matched ABI/LoRA/distillation comparison.
- [R99 reproduction packet](R99_REVIEW.md) lists exact artifacts, dependencies,
  verification commands and the expected negative result.
- [Versioning and review checks](VERSIONING.md) distinguishes API versions,
  immutable source snapshots and scientific certification.
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
