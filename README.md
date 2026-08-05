# ABI: provenance-preserving capability acquisition

ABI is the research and tooling layer for extracting measured capabilities from
frozen open-weight teachers, labeling and segregating those capabilities,
minimizing their tested information footprint, and preparing independently
validated capability-acquisition artifacts for external execution hosts.

Runtime hosting is outside this repository. The intended external consumer is
[LayerCake](https://github.com/Yoder23/layercake), maintained as a separate
codebase and evidence lineage. Source teachers and ABI `.abix`/`.abicir`
artifacts are acquisition material, not deployable cakes.

## Current status

**The ABI English-product moonshot is OPEN.**

Three facts define the current checkpoint:

1. The historical v47 bounded reference passes its exact synthetic catalog and
   packaging/runtime suite, but fails broad product interpretation: it scored
   0/28 on a frozen novel-form English audit while Phi-3 scored 19/28.
2. V87 demonstrates a real causal teacher-artifact-to-LayerCake signal against
   parent and shuffled controls, but its formal transfer-quality gate fails and
   it is not deployable.
3. V89 passes bounded pre-transfer labeling for English, chemistry, civics,
   mathematics, Python, and quarantine on a source-record-disjoint holdout.

Capability-compiler Phases 0 and 1 are **COMPLETE**. Phase 2 machine evidence is
complete, but Phase 2 is **BLOCKED_EXTERNAL_HUMAN_RATINGS**: the contract
requires three independent blinded counterbalanced ratings for each prompt.
The user deferred the unavailable human raters, so Phase 3 development is open
conditionally under a versioned governance exception. Phase 2 is still not
certified; Phase 3 cannot be promoted and Phase 4 remains locked. No current
ABI artifact is certified as a broadly fluent teacher-derived English core.

Read [CURRENT_PROJECT_STATUS.md](CURRENT_PROJECT_STATUS.md) before interpreting
any experiment or launching new work.

## Current evidence map

| Result | Status | Evidence |
| --- | --- | --- |
| Historical bounded end-to-end reference | PASS in its locked scope | `ABI_MOONSHOT_CERTIFICATE_V2.json` |
| Broad v47 English generalization | FAIL | `ABI_POSTCERT_GENERALIZATION_AUDIT_DECISION.json` |
| Small-scale causal teacher transfer | Signal demonstrated; formal pilot FAIL | `ABI_TEACHER_TO_LAYERCAKE_GRAMMAR_PILOT_V87_DECISION.json` |
| Pre-transfer English/domain labeling | Bounded PASS | `ABI_TEACHER_RECORD_LABELING_PHASE2_CERTIFICATE_V89.json` |
| Normalized English acquisition IR | PASS as a data artifact | `ABI_CAPABILITY_COMPILER_PHASE1_CERTIFICATE_V1.json` |
| Phase 2 matched-baseline machine campaign | COMPLETE; external human gate pending | `results/abi_capability_compiler_phase2/machine_evidence_v1.json` |
| Phase 2 human rating packet | READY; 21,000 independent judgments pending | `results/abi_capability_compiler_phase2/human_rating_packet_v1/manifest.json` |
| Conditional Phase 3 A0-A4 campaign | PREREGISTERED; development only | `ABI_CAPABILITY_COMPILER_PHASE3_PROTOCOL_V1.json` |
| Sufficient-information frontier | OPEN | Not yet measured |
| Integrated teacher-derived LayerCake | OPEN | Not yet produced |

## Phase 1 result

The sealed Phase 1 IR contains 7,000 English acquisition records—exactly 500
for each of 14 capabilities. Every record binds raw and normalized prompt and
output forms, the exact source revision, provenance, destination, capability,
authoritative generated token IDs, transformations, and content hashes.

The preregistered split audit found zero exact prompt overlap and zero detected
near-duplicate clusters across search, development, final, isolation, and
hostile families. Four 100-record domain inventories are retained only as
evaluation references; no specialist record is eligible for English training.

The first source extraction failed the abstention floor with 237 passing
records and 463 failures after its one allowed repair. That evidence remains
unchanged. A separately preregistered set of 400 fresh abstention prompts passed
400/400 and completed the inventory without reclassifying the failures. The
mathematics domain reference output set failed 0/100 and is also preserved as
negative, evaluation-only evidence.

Phase 1 performed no candidate training. It establishes artifact suitability,
not fluency, transfer, minimum information, specialist acquisition, or ABI
superiority.

## Phase 2 machine result

The preregistered matched-baseline campaign is complete on machine-verifiable
evidence. T0 scored 1,237/1,400 with 64 repetition collapses. Across three
seeds, L0 averaged 94.60% functional passes and L1 averaged 93.86%; both retain
the 3.82B source model at inference. L1 routed 4,200/4,200 prompts correctly.

The same-size teacher-free students are fast but unusable at this budget. D0,
D1, and D2 averaged 6.69%, 0.50%, and 2.10% functional passes, with severe
collapse. Their warm throughput is 9.83×, 5.94×, and 6.84× T0 in bytes/second,
but speed from a failed-quality checkpoint is not promotable. L0 and L1 retain
quality but reach only 0.84× and 0.83× T0 warm throughput.

No p95 or p99 is claimed from the 20 warm observations. Every cold result uses
one fresh process and one request, measuring load, first output, and total
latency from that same request. The final split was not accessed, and no ABI or
LayerCake candidate was trained.

Phase 2 is not complete until three independent people finish the blinded
rating forms. Conditional Phase 3 development may proceed, but it cannot be
certified and cannot open Phase 4 while those ratings remain deferred.

## V89 labeling result

The locked 120-row holdout contains 20 records each for English, chemistry,
civics, mathematics, Python, and quarantine. Its 128 underlying teacher record
IDs have zero overlap with the V88 development benchmark.

| Metric | Result |
| --- | ---: |
| Exact destination-plus-domain routing | 117/120, 97.5% |
| English precision | 100% |
| Specialist leakage into English | 0 |
| English recall | 90% |
| Known-domain macro-F1 | 99.36% |
| Capability accuracy | 97% |
| Quarantine recall | 100% |

All raw metrics and gates were independently recomputed. Deliberate label,
token-count, benchmark, source-identity, and implementation-lock mutations were
rejected. The result is bounded to its ontology and evidence depth; it is not
exhaustive domain discovery.

## The moonshot

The intended product workflow is:

```text
one or more pinned open-weight teachers
  -> qualified prompts and teacher outputs
  -> English / declared domain / quarantine labels
  -> normalized, provenance-bound acquisition inventories
  -> nested sufficient-information and representation budgets
  -> teacher-free LayerCake English core
  -> optional signed LayerCake domain packages
  -> LayerCake installation, sparse routing, and CPU/GPU recertification
```

The final system should support a minimal fluent English substrate, separately
selectable domain knowledge, multiple teachers with preserved provenance,
teacher-free inference, and the same canonical LayerCake interface. The same
integrated candidate—not a different checkpoint—must demonstrate generation
quality, isolation, speed, TTFT, memory, and reproducibility.

ABI is not being positioned as a novel distillation loss or LoRA variant. It
may use those mechanisms internally, but it must beat strong matched LoRA and
distillation systems on the complete LayerCake product contract: quality plus
teacher/base removal, provenance, segregation, bounded exclusion, exact
package operations, selected-only execution, and measured minimum passing
information. See [the capability-compiler campaign](ABI_CAPABILITY_COMPILER_CAMPAIGN_V1.md)
for the falsifiable definition and stop rules.

## What ABI does not claim

- Lossless copying of a foreign transformer
- Exhaustive discovery of every latent capability
- A globally smallest English representation
- Broad teacher-quality English in the current LayerCake candidate
- General mastery of Python, chemistry, civics, or mathematics
- Automatic compatibility with an untested model, tokenizer, host, or runtime
- Literal proof that an English neural representation contains no world facts
- Inheritance of the sealed LayerCake release's speed or quality evidence

## Historical bounded reference

The older v47 release remains reproducible for its exact locked scope from a
clean checkout of `abi-v47-bounded-reference-postcert-audit1`:

```powershell
C:\Python310\python.exe -m abi.moonshot_release verify
C:\Python310\python.exe -m abi.moonshot_release inspect
```

The current research working tree intentionally fails that historical
release's implementation-identity check because later ABI implementations are
present. Do not weaken the verifier or relabel the current tree as v47.

Generation requires the explicit `--allow-bounded-reference` flag because the
novel-form audit invalidated any general-purpose English interpretation.

## Rules for a new source or capability

1. Pin the immutable model and tokenizer revisions and hash every source
   weight.
2. Use a declared, user-governed ontology; never claim exhaustive discovery.
3. Retain authoritative generated token IDs, finish reasons, exact hardware,
   runtime, time, and memory accounting.
4. Label English, each selected domain, unknown material, cross-domain
   material, conflicts, and spoof attempts before composition.
5. Preserve raw prompts and outputs alongside normalized forms.
6. Keep search, validation, and final data disjoint.
7. Qualify teacher correctness, completion, diversity, provenance, and leakage
   before any candidate training.
8. Use nested data budgets and retain the adjacent lower failure.
9. Remove the teacher and all extraction material before final inference.
10. Rerun the same final integrated candidate through all quality, isolation,
    deployment, performance, and hostile-verification gates required by its
    target host.

## Documentation

- [Current project status](CURRENT_PROJECT_STATUS.md)
- [Capability-compiler campaign](ABI_CAPABILITY_COMPILER_CAMPAIGN_V1.md)
- [Machine-readable campaign contract](ABI_CAPABILITY_COMPILER_CAMPAIGN_CONTRACT_V1.json)
- [Phase 0 protocol](ABI_CAPABILITY_COMPILER_PHASE0_PROTOCOL_V1.json)
- [Phase 0 environment](ABI_CAPABILITY_COMPILER_PHASE0_ENVIRONMENT_V1.json)
- [Phase 0 certificate](ABI_CAPABILITY_COMPILER_PHASE0_CERTIFICATE_V1.json)
- [Historical campaign state V2](ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V2.json)
- [Historical campaign state V3](ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V3.json)
- [Current live campaign state V4](ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V4.json)
- [Phase 1 certificate](ABI_CAPABILITY_COMPILER_PHASE1_CERTIFICATE_V1.json)
- [Phase 1 protocol](ABI_CAPABILITY_COMPILER_PHASE1_PROTOCOL_V1.json)
- [Research-ledger policy](RESEARCH_LEDGER_POLICY.md)
- `ABI_LOCAL_RESEARCH_ARTIFACTS_MANIFEST_V1.json` for large local catalog identity
- [Phased research roadmap](ROADMAP.md)
- [Canonical claim ledger](CLAIMS.md)
- [Open scientific questions](RESEARCH_STATUS_AND_GAPS.md)
- [Active mission and stop rules](ACTIVE_MISSION.md)
- [Detailed bounded moonshot history](ABI_MOONSHOT.md)
- [LayerCake acquisition boundary](LAYERCAKE_ACQUISITION.md)
- [Universal-transfer limits](FORMAL_UNIVERSAL_TRANSFER.md)

Historical protocols and failed experiments remain evidence. They do not
override the current status or promote themselves through documentation.
