# ABI current project status

Status date: 2026-08-03

## Executive state

ABI is an open research campaign, not a completed English-acquisition product.
It has a working bounded reference release, a demonstrated small-scale causal
teacher-to-LayerCake signal, and a bounded pre-transfer labeling pass. It has
not yet produced a broadly fluent teacher-derived LayerCake English core.

The immediate research phase is **Phase 0 of the ABI capability-compiler
campaign: definitions, strong baselines, and preregistration**. Its protocol is
now preregistered and VERIFYING, but it has not yet been certified. Normalized
transfer-corpus construction follows as Phase 1;
the sufficient-information frontier is Phase 4 and is not yet authorized.

`ABI_CAPABILITY_COMPILER_CAMPAIGN_V1.md` and its machine-readable contract now
control future work. They do not alter any historical result.

## Product boundary

ABI owns:

- frozen-teacher surveying and source qualification;
- semantic destination labeling and ambiguity quarantine;
- English/domain segregation and provenance;
- normalization into training-ready acquisition material;
- imported-information and acquisition-cost accounting;
- tested information-budget and representation minimization; and
- independent qualification of artifacts before LayerCake consumption.

LayerCake owns:

- execution and generation;
- canonical core/cake interfaces;
- signed package installation and exact package transfer;
- immutable core and capability hosting;
- selected-only execution, routing, and orchestration; and
- integrated CPU/GPU quality, speed, TTFT, RSS, and hostile verification.

ABI `.abix` files contain acquisition material and are never deployable cakes.
Foreign-teacher acquisition is not called lossless. Lossless transfer is
reserved for exact LayerCake package bytes, manifests, tensors, and installed
payload identity.

ABI is being tested as a provenance-preserving capability compiler, not as a
renamed optimizer. Teacher-output learning remains distillation; low-rank
updates on a frozen base remain LoRA. ABI earns a distinct product claim only
if the same final LayerCake is quality-noninferior to strong matched baselines
while additionally proving teacher/base removal, segregation, bounded
exclusion, exact package operations, selected-only execution, and a measured
sufficient-information frontier.

## Evidence ladder

| Milestone | Status | What it establishes | What it does not establish |
| --- | --- | --- | --- |
| Historical bounded v47 reference | PASS on its locked synthetic suite | End-to-end teacher-free packaging, bounded capabilities, and a measured runtime | Broad English fluency; the later novel-form audit scored LayerCake 0/28 versus Phi-3 19/28 |
| V87 causal grammar transfer pilot | Causal signal PASS; formal pilot FAIL | Real teacher mappings beat unchanged-parent and shuffled-target controls | Reliable grammar transfer, deployability, broad English, or repetition safety |
| V89 pre-transfer labeling | Bounded PASS | Actual teacher records can be routed into English, four known domains, or quarantine under the locked ontology | Exhaustive domain discovery, arbitrary-record classification, normalization, sufficient data, or LayerCake quality |
| Compiler definitions and matched-baseline preregistration | VERIFYING | Protocol, environment, numeric gates, baselines, statistics, and stop rules are frozen in the working tree | No new training run is promotion-eligible until committed verification and a Phase 0 certificate pass |
| Normalized corpus and adequacy | LOCKED behind compiler Phase 0 | Not yet established | Training must not begin from raw or merely labeled records |
| Sufficient-information frontier | OPEN | Not yet established | No smallest fluent core or imported-information budget is known |
| Integrated ABI-derived LayerCake | OPEN | Not yet established | No ABI candidate inherits sealed LayerCake performance evidence |

## Current proven results

### Historical bounded reference

`ABI_MOONSHOT_CERTIFICATE_V2.json` remains valid for its exact frozen suite.
The later `ABI_POSTCERT_GENERALIZATION_AUDIT_DECISION.json` controls the broader
product conclusion: v47 is not a generally fluent English core.

### Causal transfer construct

`ABI_TEACHER_TO_LAYERCAKE_GRAMMAR_PILOT_V87_DECISION.json` records that the
real 160-row teacher mapping produced statistically positive paired likelihood
improvements over both the parent and matched shuffled controls. Autonomous
exact correction improved to 14/64 from 0/64 for both controls.

The same candidate achieved only 34/64 positive margins, below the locked
48/64 requirement, and produced 23/64 repetition collapses. The correct claim
is a causal transfer signal, not an operational transfer pass.

### Bounded labeling

`ABI_TEACHER_RECORD_LABELING_PHASE2_CERTIFICATE_V89.json` records a
source-record-disjoint 120-row GPU holdout:

- 117/120 exact destination-plus-domain routes;
- 100% English precision and zero specialist leakage into English;
- 90% English recall;
- 99.36% known-domain macro-F1;
- 97% known-record capability accuracy; and
- 100% quarantine recall across cross-domain, unknown-domain, and label-spoof
  families.

All raw metrics and gates were independently recomputed, and the hardened
verifier rejected five deliberate mutations. Per-class depth is 20, so this is
a bounded point-estimate pass rather than a population-wide guarantee.

## Current implementation

- `abi/grammar_transfer_pilot.py` implements the V87 real/shuffled causal
  pilot and held-out evaluation.
- `abi/teacher_record_labeling.py` implements the fail-closed deterministic
  risk screen plus frozen GPU semantic classifier.
- `abi/teacher_record_labeling_followup.py` builds source-record-disjoint V89
  holdouts.
- `abi/teacher_record_labeling_verifier.py` independently replays labels,
  metrics, gates, provenance, hashes, token accounting, and tamper rejection.
- The existing v3 segregated-bundle consumers enforce destination and purity
  metadata before materialization, but no current normalized successor corpus
  has passed adequacy for broad English acquisition.

The working research tree contains a large preserved experiment ledger and is
not represented here as a clean public release tag. Publication and repository
consolidation remain separate release-engineering work.

The historical `abi.moonshot_release verify` command is expected to fail closed
in this current tree when release-bound implementation files differ. Reproduce
v47 only from the exact clean
`abi-v47-bounded-reference-postcert-audit1` checkout; never relax its identity
checks to make later research code appear release-equivalent.

## Claims allowed now

- ABI has a reproducible bounded reference implementation.
- ABI has demonstrated a causal small-scale teacher-artifact-to-LayerCake
  signal.
- ABI has passed bounded pre-transfer labeling for the locked English,
  chemistry, civics, mathematics, Python, and quarantine ontology.
- Exact LayerCake-to-LayerCake package transfer remains available after an
  artifact has been created and certified by LayerCake.

## Claims not allowed now

- ABI has completed the English moonshot.
- V87 is a reliable or deployable transfer method.
- V89 discovers or correctly labels all capabilities in arbitrary weights.
- ABI can losslessly copy a foreign model.
- The smallest fluent English representation is known.
- A teacher-derived LayerCake matches teacher generation quality.
- A new ABI candidate inherits LayerCake's sealed speed, memory, TTFT,
  routing, or portability results.
- An English core is literally free of all world knowledge.
- ABI supersedes, dominates, or is universally better than LoRA or
  distillation.
- Any ABI-derived LayerCake is stronger than its teacher.

## Immediate next gate

Complete committed verification of
`ABI_CAPABILITY_COMPILER_PHASE0_PROTOCOL_V1.json`: bind the protocol,
environment, verifier, tests, and full-suite result to an implementation
commit, issue a Phase 0 certificate, and verify the resulting tree is clean.

Only after Phase 0 passes may normalized artifact work begin. That Phase 1
artifact must separately account for English and every selected domain:

1. immutable normalized record schema and transformation rules;
2. source revision, prompt, output, label, capability, provenance, and
   authoritative token preservation;
3. exact duplicate and near-duplicate accounting;
4. prompt-surface, behavior, and capability coverage;
5. contamination, cross-domain, unknown-domain, and conflict quarantine;
6. teacher-output correctness and completion qualification;
7. unique UTF-8 bytes, teacher tokens, disk, RAM, time, and hardware;
8. at least the preregistered minimum distinct passing records per required
   capability; and
9. an untouched natural generalization suite that cannot select normalization,
   budgets, checkpoints, or repairs.

Only after that artifact passes may the campaign run nested information
budgets and identify the smallest passing tested budget paired with the largest
lower failing budget.
