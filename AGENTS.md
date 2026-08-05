# ABI repository working contract

## Repository scope

This repository owns foreign-teacher capability acquisition: source
qualification, record generation, semantic labeling, quarantine,
normalization, provenance, imported-information accounting, acquisition
experiments, and independently verifiable ABI artifacts.

It does not own the LayerCake runtime, cake installation, routing,
orchestration, or LayerCake performance claims. Those live in the separate
[LayerCake repository](https://github.com/Yoder23/layercake). Cross-repository
integration is discussed here only as an external interface and future
acceptance gate. Do not copy LayerCake code, certificates, or detailed research
history into this repository.

An ABI `.abix` or `.abicir` file is acquisition evidence. It is not a deployable
LayerCake cake.

## Current campaign state

Status date: 2026-08-05.

- Capability-compiler Phase 0 is **COMPLETE** under
  `ABI_CAPABILITY_COMPILER_PHASE0_CERTIFICATE_V1.json`.
- Capability-compiler Phase 1 is **COMPLETE** under
  `ABI_CAPABILITY_COMPILER_PHASE1_CERTIFICATE_V1.json`.
- Phase 2 machine evidence is **COMPLETE**, but Phase 2 is
  **BLOCKED_EXTERNAL_HUMAN_RATINGS**. The fail-closed report is
  `results/abi_capability_compiler_phase2/machine_evidence_v1.json`.
- All preregistered T0/L0/L1/D0/D1/D2 development, full-depth, three-seed,
  paired-statistics, genuine-cold, and 20-observation warm runs are complete.
  No ABI or LayerCake candidate was trained in Phase 2.
- The three blinded counterbalanced rating forms contain 7,000 pairs each and
  require 21,000 judgments from three independent people. Until those forms
  are completed and verified, Phase 2 has no final certificate.
- The user explicitly deferred the unavailable human ratings. The conditionally
  authorized Phase 3 A0-A4 development branch is **COMPLETE_FAILED** under
  `results/abi_capability_compiler_phase3/conditional_decision_v3.json`; this
  is not a Phase 2 pass or a Phase 3 certificate. Phase 4 remains locked.
- Corrected V4 evidence shows a causal labeled teacher-payload signal against
  all four matched controls, but A0 scored only 379/1,400 with 150 repetition
  collapses and is far below the locked absolute and teacher-relative gates.
- The ABI moonshot is **not complete**.

The machine-readable live state is
`ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V10.json`. The original campaign
contract and all historical evidence remain unchanged.

Phase 1 produced a 7,000-record normalized English acquisition IR with exactly
500 eligible records for each of 14 capabilities. It retains raw and normalized
forms, provenance, source identity, authoritative generated token IDs,
transformations, labels, and hashes. Search, development, final, isolation, and
hostile prompt families have zero detected exact overlap and zero detected
near-duplicate clusters under the preregistered check.

The first full extraction failed the abstention adequacy floor and remains
negative evidence. A separately preregistered, fresh abstention successor
passed. Failed V1 records were not relabeled. Four domain inventories are
evaluation-only and excluded from English acquisition. The mathematics source
reference set failed 0/100 and is preserved as negative evidence.

Phase 1 trained no LoRA, distilled student, or ABI candidate. It proves data
artifact suitability only. It does not prove fluent transfer, teacher-relative
quality, a minimum information budget, domain acquisition, or superiority over
LoRA or distillation.

## Active authorization

No further Phase 2 training is authorized. The A0-A4 and V6 B0-B4 branches are
closed. V6 B0 scored 1,148/1,400 with 43 collapses, lost to label-free B1 by
5.43 points, did not beat monolithic B4, and trailed T0 by 6.36 points. Seeds
130363 and 155921 are not authorized. Do not tune V6 or access final material.
Any successor requires a new preregistration tied to the measured
routing/specialization failure. Phase 2 human ratings remain deferred, Phase 3
is uncertified, and Phase 4 is locked.

The no-training V9 component diagnostic is complete. It found that bypassing
the route embedding changes B1 by only +1/1,400, while bypassing output cakes
costs 76/1,400. No altered checkpoint was persisted. No new training is
currently authorized.

V8 attempt 1 failed before evaluating any prompt because the diagnostic used a
ModuleDict-style string index on the host's ModuleList. V9 authorizes only the
integer-index repair and unchanged R1-R3 retry. The failure remains evidence.

## Permanent scientific rules

- Preserve every failed run, superseded protocol, and raw evidence artifact.
- Never edit a file bound by a certificate; create a versioned successor.
- Pin source and tokenizer revisions and hash the exact source manifest.
- Runtime-generated token IDs are authoritative. Character estimates are not
  token counts.
- Keep raw prompts and outputs alongside normalized material and retain
  record-level provenance and transformation history.
- Keep English, declared domains, mixed material, unknown material, conflicts,
  and spoof attempts distinct. Fail closed when destination is uncertain.
- Never call declared-ontology labeling exhaustive domain discovery.
- Never call foreign-teacher acquisition lossless. Reserve exact/lossless for
  byte-, archive-, manifest-, tensor-, or installed-payload identity.
- “Minimum” means the smallest passing preregistered tested budget paired with
  its adjacent lower failing budget, never a global minimum.
- Keep search, development, and final splits disjoint. Final data cannot select
  prompts, normalization, budgets, checkpoints, sources, or repairs.
- Account raw prompts, unique UTF-8 bytes, teacher outputs, authoritative
  teacher tokens, logits, activations, copied parameters, trainable parameters,
  disk, RAM, VRAM, wall time, inference time, hardware, and external cost.
- Name standard mechanisms honestly. Teacher-output learning is distillation;
  low-rank adaptation of a frozen base is LoRA.
- Do not claim ABI superiority until the mandatory matched campaign passes.
- Do not combine quality from one candidate with speed, memory, or isolation
  from another.
- A future external-host result must be certified on the same integrated
  candidate; it cannot inherit another repository’s evidence.

## Verification

```powershell
C:\Python310\python.exe -m abi.capability_compiler_phase1_certificate
C:\Python310\python.exe -m abi.capability_compiler_phase2_verify
C:\Python310\python.exe -m abi.capability_compiler_phase2_evidence
C:\Python310\python.exe -m pytest -q
```

The historical bounded reference must be reproduced only from its exact tagged
checkout; never weaken an identity verifier to make the current tree appear
equivalent.
