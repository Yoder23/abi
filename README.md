# ABI: capability acquisition for LayerCake

ABI is the research and tooling layer for extracting measured capabilities from
frozen open-weight teachers, labeling and segregating those capabilities,
minimizing their tested information footprint, and preparing independently
validated artifacts for LayerCake.

LayerCake is a separate repository and product. It owns execution, package
installation, exact package transfer, composition, routing, orchestration, and
integrated CPU/GPU performance. Source teachers and ABI `.abix` bundles are
never part of the deployed LayerCake product.

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

The active phase is **Phase 0 verification of the capability-compiler
campaign**. Its definitions, strong matched baseline specifications, numeric
gates, and stop rules are preregistered; certification is not yet complete.
Normalized transfer-corpus construction is Phase 1. No current ABI artifact is
certified as a broadly fluent teacher-derived LayerCake English core.

Read [CURRENT_PROJECT_STATUS.md](CURRENT_PROJECT_STATUS.md) before interpreting
any experiment or launching new work.

## Current evidence map

| Result | Status | Evidence |
| --- | --- | --- |
| Historical bounded end-to-end reference | PASS in its locked scope | `ABI_MOONSHOT_CERTIFICATE_V2.json` |
| Broad v47 English generalization | FAIL | `ABI_POSTCERT_GENERALIZATION_AUDIT_DECISION.json` |
| Small-scale causal teacher transfer | Signal demonstrated; formal pilot FAIL | `ABI_TEACHER_TO_LAYERCAKE_GRAMMAR_PILOT_V87_DECISION.json` |
| Pre-transfer English/domain labeling | Bounded PASS | `ABI_TEACHER_RECORD_LABELING_PHASE2_CERTIFICATE_V89.json` |
| Normalized broad-English artifact | OPEN | Not yet produced |
| Sufficient-information frontier | OPEN | Not yet measured |
| Integrated teacher-derived LayerCake | OPEN | Not yet produced |

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
   before LayerCake training.
8. Use nested data budgets and retain the adjacent lower failure.
9. Remove the teacher and all extraction material before final inference.
10. Rerun the same final integrated LayerCake candidate through quality,
    isolation, package, routing, CPU/GPU, TTFT, RSS, and hostile-verification
    gates.

## Documentation

- [Current project status](CURRENT_PROJECT_STATUS.md)
- [Capability-compiler campaign](ABI_CAPABILITY_COMPILER_CAMPAIGN_V1.md)
- [Machine-readable campaign contract](ABI_CAPABILITY_COMPILER_CAMPAIGN_CONTRACT_V1.json)
- [Phase 0 protocol](ABI_CAPABILITY_COMPILER_PHASE0_PROTOCOL_V1.json)
- [Phase 0 environment](ABI_CAPABILITY_COMPILER_PHASE0_ENVIRONMENT_V1.json)
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
