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

Status date: 2026-08-03.

- Capability-compiler Phase 0 is **COMPLETE** under
  `ABI_CAPABILITY_COMPILER_PHASE0_CERTIFICATE_V1.json`.
- Capability-compiler Phase 1 is **COMPLETE** under
  `ABI_CAPABILITY_COMPILER_PHASE1_CERTIFICATE_V1.json`.
- Phase 2 is **IN_PROGRESS_PREREGISTERED_REPAIR1** under
  `ABI_CAPABILITY_COMPILER_PHASE2_PROTOCOL_REPAIR1_V2.json`; the sole allowed
  implementation repair corrected tokenizer/model vocabulary accounting
  before any pack artifact or Phase 2 training result existed.
- Phases 3 through 8 are **LOCKED**.
- The ABI moonshot is **not complete**.

The machine-readable live state is
`ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V2.json`. The original campaign
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

Phase 2 may run only the hash-bound T0, L0, L1, D0, D1, and bounded D2 baseline
campaign in `ABI_CAPABILITY_COMPILER_PHASE2_PROTOCOL_REPAIR1_V2.json`. Do not start
Phase 3 ABI-candidate training until Phase 2 is certified. Final-test outputs
may not influence implementation, tuning, repair, stopping, or selection.

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
C:\Python310\python.exe -m pytest -q
```

The historical bounded reference must be reproduced only from its exact tagged
checkout; never weaken an identity verifier to make the current tree appear
equivalent.
