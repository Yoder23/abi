# ABI active mission

Status date: 2026-08-05

## State

The ABI capability-compiler moonshot is **OPEN**.

Phase 0 and Phase 1 are complete. Phase 2's entire preregistered machine
baseline campaign is complete, but Phase 2 remains blocked on the required
three independent blinded human-rating forms.

| Phase | State | Authority |
| --- | --- | --- |
| 0 — definitions and preregistration | COMPLETE | `ABI_CAPABILITY_COMPILER_PHASE0_CERTIFICATE_V1.json` |
| 1 — normalized acquisition IR | COMPLETE | `ABI_CAPABILITY_COMPILER_PHASE1_CERTIFICATE_V1.json` |
| 2 — matched LoRA and distillation baselines | MACHINE_EVIDENCE_COMPLETE_BLOCKED_EXTERNAL_HUMAN_RATINGS | `results/abi_capability_compiler_phase2/machine_evidence_v1.json` |
| 3 — causal teacher-to-target acquisition | TWO BRANCHES COMPLETE_FAILED; PHASE UNCERTIFIED | `results/abi_capability_compiler_phase3_sequence/conditional_decision_v1.json` |
| 4–8 | LOCKED | Campaign contract |

Phase 1 certifies a data artifact, not a model. It selected 7,000 normalized
English records—500 for each of 14 capabilities—with raw forms, authoritative
token IDs, provenance, labels, transformations, and content hashes. It also
freezes disjoint development, final, isolation, and hostile material.

The original source run failed the abstention floor and remains preserved. A
fresh, preregistered successor supplied 400 new passing abstention records; none
of the failed records were reclassified. Specialist inventories are
evaluation-only, and no specialist record is eligible for English training.

## Active objective: execute the preregistered shared-output Phase 3 successor

The user explicitly deferred the unavailable human raters. This does not
complete Phase 2. The corrected V4 A0-A4 development campaign has finished on
the certified Phase 1 IR without opening final material. It establishes a
development-only causal labeled teacher-payload signal, but the registered
candidate fails absolute quality and repetition safety. It cannot certify
Phase 3 or open Phase 4.

A0 scored 379/1,400 (27.07%, Wilson 95% CI 24.81%-29.46%) with 150
repetition collapses. Its paired pass-rate advantages over A1, A2, A3, and A4
were respectively 19.00, 7.71, 26.79, and 9.07 percentage points; every
stratified-bootstrap 95% lower bound was positive. Against T0, A0 was 61.29
points worse (95% CI -63.64 to -58.86). Two capabilities scored 0/100.

V6 tested a materially different 1,556,998-parameter continuous prompt encoder
and three rank-128 nonlinear pre-block residuals with all three host blocks
frozen. Its initial B0-B4 matrix is complete. B0 improved to 1,148/1,400, but
had 43 collapses, lost significantly to label-free B1, did not beat monolithic
B4, and remained significantly below T0. Every locked promotion family did
not pass, so the remaining seeds are prohibited and the branch is closed.

V8 is a read-only mechanistic diagnostic on the sealed B1 checkpoint. It
bypasses output cakes, the route embedding, or both in memory on the same 1,400
development prompts. It performs no training, persists no altered checkpoint,
cannot promote a result, and exists only to choose the component set for one
future preregistered train-from-initialization successor.

Attempt 1 stopped before evaluation or artifact creation on a task-cake
container index mismatch. V9 binds the one-line ModuleList repair and regression
test; no scientific design field changed.

The diagnostic is complete. R1 and R3 score 1,148/1,400; R2 scores
1,225/1,400 versus R0's 1,224/1,400. The route embedding is not measurably
responsible for quality, while output cakes contribute 5.43 points and add
collapse risk. No new training is authorized until the shared-output successor
is separately implemented, tested, parameter-accounted, and preregistered.

V11 is now preregistered. It retains the three shared rank-128 sequence
transforms, replaces six trainable English output cakes with one rank-64 shared
output residual, and adds a training-only penalty for wrong greedy predictions
that repeat a recent teacher token. Execute C0-C4 only at seed 104729.

The machine campaign contains all T0, L0, L1, D0, D1, and bounded D2 search
receipts; 15 persisted headline checkpoints; 1,400 prompts per seed; 10,000
paired bootstrap resamples per T0 comparison; one genuine-cold request per
system; and 20 warm observations per system. The final split was not accessed.

The first pack preflight failed before creating an artifact or training a
model because tokenizer base vocabulary size was incorrectly equated with the
model output-head size. The one allowed implementation repair now binds all
three authoritative quantities (32,000 base pieces, 32,011 runtime tokenizer
entries, and a 32,064-wide model vocabulary). The failure remains preserved.

Phase 2 must not claim that ABI has transferred capability or outperformed a
baseline. Its purpose is to establish credible competitors and a reproducible
comparison floor.

The compact students are fast but fail quality: D0/D1/D2 average 6.69%, 0.50%,
and 2.10% functional pass rates. L0 and L1 average 94.60% and 93.86% but retain
the 3.82B source model, and are slower than T0 on warm bytes/second. These are
baseline findings, not an ABI-candidate result.

## Stop boundaries

- Do not rerun or tune the completed A0-A4 or B0-B4 branches. Their remaining
  two seeds were not authorized after their first seeds failed locked gates.
- Do not launch a successor without a new preregistration addressing the
  measured routing/specialization bottleneck while preserving sequence
  conditioning, matched causal controls, and the final-data firewall.
- Do not call conditional Phase 3 evidence a certificate while Phase 2 lacks
  human ratings.
- Do not use final-test outputs for tuning, repair, source selection, or early
  stopping.
- Do not silently replace the certified Phase 1 artifact.
- Preserve failed baselines and underperforming configurations.
- Do not describe ABI as better than LoRA or distillation until the full
  matched evidence supports that bounded claim.
- Keep detailed LayerCake implementation and research status in the separate
  [LayerCake repository](https://github.com/Yoder23/layercake).

## Authoritative current documents

- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V11.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_SHARED_OUTPUT_PROTOCOL_V11.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_COMPONENT_DIAGNOSTIC_PROTOCOL_V8.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_COMPONENT_DIAGNOSTIC_REPAIR1_V9.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_COMPONENT_DIAGNOSTIC_REPORT_V3.md`
- `ABI_CAPABILITY_COMPILER_PHASE3_SEQUENCE_SUCCESSOR_PROTOCOL_V6.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_CONDITIONAL_OPEN_V1.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_PROTOCOL_V1.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_PAIRED_SAMPLER_AMENDMENT_V4.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_ANALYSIS_EMITTER_AMENDMENT_V5.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_MACHINE_REPORT_V1.md`
- `ABI_CAPABILITY_COMPILER_PHASE3_SEQUENCE_REPORT_V2.md`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_V1.md`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_CONTRACT_V1.json`
- `ABI_CAPABILITY_COMPILER_PHASE1_PROTOCOL_V1.json`
- `ABI_CAPABILITY_COMPILER_PHASE1_ABSTENTION_PROTOCOL_V2.json`
- `ABI_CAPABILITY_COMPILER_PHASE1_CERTIFICATE_V1.json`
- `ABI_CAPABILITY_COMPILER_PHASE2_PROTOCOL_V1.json`
- `ABI_CAPABILITY_COMPILER_PHASE2_PROTOCOL_REPAIR1_V2.json`
- `CURRENT_PROJECT_STATUS.md`
- `CLAIMS.md`
- `ROADMAP.md`
- `RESEARCH_STATUS_AND_GAPS.md`

Historical evidence remains authoritative only for its exact scope and never
overrides this phase boundary.
