# ABI current project status

Status date: 2026-08-05

## Executive state

ABI is an open research campaign, not a completed English-acquisition product.
It has a working bounded reference release, a demonstrated small-scale causal
teacher-to-LayerCake signal, and a bounded pre-transfer labeling pass. It has
not yet produced a broadly fluent teacher-derived LayerCake English core.

Capability-compiler Phase 0 is **COMPLETE**, and Phase 1 is **COMPLETE**.
Phase 2's preregistered
machine campaign is complete and adversarially rehashed, but Phase 2 is
**BLOCKED_EXTERNAL_HUMAN_RATINGS**. Three independent people must complete the
21,000 blinded counterbalanced judgments before a final Phase 2 certificate can
exist. Phase 2 is not certified. The user explicitly deferred the currently
unavailable human raters.
Both conditionally authorized Phase 3 branches are now **COMPLETE_FAILED**,
without changing Phase 2's failed-closed status. The first found a causal
signal but failed quality. The sequence successor improved aggregate quality
substantially, but failed absolute, causal-control, teacher-relative, and
repetition gates. Phase 3 is not certified; Phases 4 through 8 remain locked.

`ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V8.json` is the live state. It
authorizes only the read-only R1-R3 component diagnostic against sealed B1;
training, altered-checkpoint persistence, final access, and promotion remain
prohibited.
`ABI_CAPABILITY_COMPILER_PHASE3_CONDITIONAL_OPEN_V1.json` and the versioned
Phase 3 protocols remain the historical authority for the narrow execution
exception. They did not alter any prior result or permit final-test use.

## Product boundary

ABI owns:

- frozen-teacher surveying and source qualification;
- semantic destination labeling and ambiguity quarantine;
- English/domain segregation and provenance;
- normalization into training-ready acquisition material;
- imported-information and acquisition-cost accounting;
- tested information-budget and representation minimization; and
- independent qualification of artifacts before LayerCake consumption.

Execution-host implementation and certification belong to the separate
[LayerCake repository](https://github.com/Yoder23/layercake). ABI records only
the external artifact and acceptance boundary; it does not duplicate
LayerCake's implementation or research history.

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
| Compiler definitions and matched-baseline preregistration | PASS | Phase 0 certificate binds the parent lineage, teacher, host, 11 systems, data boundaries, numeric gates, statistics, accounting, and stop rules | No LoRA, distillation, ABI acquisition, or superiority result |
| Normalized acquisition IR | PASS | 7,000 records; 500 per English capability; provenance, authoritative tokens, hashes, and disjoint frozen evaluation material | No model, fluent transfer, domain acquisition, or minimum-information result |
| Matched LoRA/distillation machine baselines | COMPLETE; human gate pending | All registered search/full/headline/runtime/statistical evidence is complete and content-verified | No ABI candidate, no teacher-free fluent model, and no Phase 2 certificate yet |
| Conditional Phase 3 A0-A4 branch | COMPLETE FAILED; causal signal | Labeled teacher payload and segmented routing beat four matched controls on paired development prompts | No fluent candidate, Phase 3 certificate, final-test access, or Phase 4 opening |
| Conditional Phase 3 B0-B4 sequence branch | COMPLETE FAILED; response/target sensitivity | Continuous sequence conditioning reaches 82.00%; correct responses and target pairing matter | Semantic routing causality, absolute gate pass, Phase 3 certificate, or final-test access |
| Sufficient-information frontier | LOCKED | Not yet established | No smallest fluent core or imported-information budget is known |
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
- `abi/capability_compiler_phase1_catalog.py` deterministically builds the
  frozen Phase 1 search, development, final, isolation, and hostile catalog.
- `abi/capability_compiler_phase1_extract.py` performs resumable, runtime-bound
  teacher extraction with authoritative token accounting.
- `abi/capability_compiler_phase1_ir.py` builds and verifies the normalized,
  content-addressed acquisition IR.
- `abi/capability_compiler_phase1_certificate.py` verifies the complete Phase
  1 seal and all bound evidence hashes.
- `abi/capability_compiler_phase2_evidence.py` rehashes every Phase 2 machine
  artifact and fails closed while independent human ratings are absent.
- `abi/capability_compiler_phase2_human_packet.py` deterministically creates
  three blinded, exactly counterbalanced rating forms and a separate key.
- `abi/capability_compiler_phase3.py` trains and evaluates the registered A0-A4
  frozen-host bridge matrix with authoritative exposure accounting.
- `abi/capability_compiler_phase3_analysis.py` verifies paired sample identity,
  computes Wilson intervals and 10,000-replicate stratified paired bootstraps,
  and emits the immutable failed-branch decision.
- `abi/capability_compiler_phase3_sequence_verifier.py` independently rehashes
  all V6 checkpoints, recomputes its decision byte-for-byte, and rejects
  rehashed teacher/source/isolation/promotion mutations.
- The existing v3 segregated-bundle consumers enforce destination and purity
  metadata before materialization. Phase 1 now supplies a normalized successor
  acquisition IR, but its sufficiency for broad English remains untested.

The large experiment ledger remains preserved in version control. Phase 1 has a
clean, locally verifiable certificate and is intended to be sealed by the
annotated tag `abi-capability-compiler-phase1-v1`. Publishing that local commit
and tag to the remote repository remains a separate explicit action.

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
- ABI capability-compiler Phase 0 governance and preregistration have passed.
- ABI capability-compiler Phase 1 has certified a 7,000-record normalized
  English acquisition IR with 500 records per capability, zero detected
  cross-split exact/near-duplicate collisions, and no specialist acquisition
  records.
- ABI Phase 2 has completed its preregistered machine baseline matrix. Strong
  source-retained adapters preserve quality but are not compact or teacher
  free; same-size teacher-free students are fast but fail quality severely.
- The corrected Phase 3 A0 candidate causally beats all four matched controls
  on the development suite, while failing absolute quality at 379/1,400 with
  150 repetition collapses and trailing T0 by 61.29 percentage points.
- The V6 sequence candidate B0 reaches 1,148/1,400, but has 43 collapses,
  loses significantly to label-free B1, does not beat monolithic B4, and
  trails T0 by 6.36 percentage points. Correct targets and response
  supervision are necessary; semantic routing is not established as causal.

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

Two gates remain unresolved. When raters become available, complete the Phase
2 packet without changing its forms or key. V6 B0-B4 is complete and failed;
its remaining seeds are prohibited. Do not rerun either failed branch or tune
V6 after observing results. A successor requires a new preregistration tied to
the measured routing/specialization failure. Final material remains unopened
and Phase 4 remains locked.

The full ABI suite currently reports 484 passed and one intentionally strict
external-control failure: the separate clean LayerCake repository is ahead of
the exact contract-bound control commit, which remains an ancestor. The
identity verifier was not weakened. See
`ABI_CAPABILITY_COMPILER_PHASE2_MACHINE_EVIDENCE_V1.md`.
