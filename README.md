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

Status date: 2026-08-07

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
The user deferred the unavailable human raters.

Phase 3 remains **UNCERTIFIED**. V34 produced the smallest preregistered exact
UTF-8 BPE representation pass (4,999 fixed actions); V37 then proved exact
LayerCake v3 sequence conformance on 16,800/16,800 prompt/output comparisons.
That opened neural acquisition, not a quality claim. V38's plain BPE candidate
and V41's label-aware/header-robust candidate both scored 0/1,400 on the locked
autonomous suite and each produced the wrong fact-free mode on 1,394 prompts.
V42 proved that V41's training-only capability head predicts all 1,400 original
held-out prompts as `fact_free_reasoning`; body-only routing reached 81.07%.
V45 subsequently solved that bounded routing failure at three fixed seeds,
scoring 1,400/1,400 on original prompts, bodies, and metadata rejection at each
seed. V47's integrated minimal bridge nevertheless scored only 760/1,400 with
20 collapses. V49 measured 90.56% action accuracy and 72.37% exact sequences on
the 7,000 acquisition records, isolating the remaining limit to capability
realization in the frozen generator/bridge rather than routing.
V50's non-promotional full-generator capacity control then scored 785/1,400
with 36 collapses even though routing remained perfect. V51 measured 96.53%
action accuracy and 79.14% exact acquisition sequences, below both fit gates;
the current topology/data branch is closed rather than promoted or swept.
V52 is now preregistered as one materially distinct resilience screen: the
qualified router selects among 14 independently trained, body-only capability
experts. Its placement compiler sends only eligible English acquisition records
to the core, maps known specialist selections to separate domain cakes, and
fails closed because no domain acquisition payload exists yet. This is an
authorized experiment, not a result.
Phase 4 remains locked. No current ABI artifact is certified as a broadly
fluent teacher-derived English core.

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
| Conditional Phase 3 A0-A4 campaign | COMPLETE; causal signal, branch FAIL | `results/abi_capability_compiler_phase3/conditional_decision_v3.json` |
| Phase 3 prompt-conditioned sequence successor | COMPLETE; branch FAIL | `results/abi_capability_compiler_phase3_sequence/conditional_decision_v1.json` |
| Phase 3 component diagnostic | COMPLETE; diagnostic only | `results/abi_capability_compiler_phase3_component_diagnostic/decision_v1.json` |
| Phase 3 shared-output successor | COMPLETE; causal/teacher-relative PASS, absolute branch FAIL | `results/abi_capability_compiler_phase3_shared_output/conditional_decision_v1.json` |
| V34 exact UTF-8 BPE representation | PASS representation only | `results/abi_capability_compiler_phase3_utf8_bpe/utf8_bpe_v34.json` |
| V37 LayerCake v3 tokenizer conformance | PASS, 16,800/16,800 | `results/abi_capability_compiler_phase3_external_tokenizer_conformance/conformance_v37.json` |
| V38 plain BPE neural acquisition | COMPLETE FAIL, 0/1,400 | `results/abi_capability_compiler_phase3_bpe_core/bpe_core_decision_v38.json` |
| V41 label-aware BPE acquisition | COMPLETE FAIL, 0/1,400 | `results/abi_capability_compiler_phase3_labeled_bpe_core/labeled_bpe_decision_v41.json` |
| V42 auxiliary-router generalization | COMPLETE FAIL; superseded router architecture | `results/abi_capability_compiler_phase3_auxiliary_router_diagnostic/diagnostic_v42.json` |
| V45 sparse router | THREE-SEED PASS, 4,200/4,200 per view | `ABI_CAPABILITY_COMPILER_PHASE3_SPARSE_ROUTER_REPLICATION_RESULT_V46.json` |
| V47 minimal integrated route bridge | COMPLETE FAIL, 760/1,400, 20 collapses | `ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_BRIDGE_RESULT_V48.json` |
| V49 routed training-fit attribution | COMPLETE; fit limited at 90.56% actions / 72.37% exact | `ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_BRIDGE_FIT_RESULT_V49.json` |
| V50 full-generator capacity upper bound | COMPLETE FAIL, 785/1,400, 36 collapses | `ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_CAPACITY_RESULT_V50.json` |
| V51 full-generator fit attribution | COMPLETE; fit limited at 96.53% actions / 79.14% exact | `ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_CAPACITY_FIT_RESULT_V51.json` |
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
rating forms. The authorized Phase 3 branch has finished without certification
and cannot open Phase 4 while its quality gates and the Phase 2 human gate are
unresolved.

## Conditional Phase 3 result

The newest controlling state is
`ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V53.json`. Exact representation and
external-host conformance are solved for the selected BPE surface; learned
English acquisition is not. V38 improved teacher-forced exact training
sequences to 85.23% but collapsed held-out capability routing. V41 used ABI's
14 labels, balanced header dropout, and causal-history corruption; its
training-only label head fit the acquisition sample but did not generalize to
the original held-out framing. V42 forbids an explicit routed host until ABI
can exceed 90% held-out router accuracy without changing LayerCake.

The next experiment must break acquisition-header/label correlation and improve
semantic body routing without adding teacher outputs. More steps alone,
tokenizer sweeps, a benchmark-specific header stripper, and a LayerCake v4
route are not authorized by the current evidence.

The corrected V4 campaign trained and evaluated A0 plus four controls on an
identical successful 28,000-record sequence. The first mixed-precision run was
invalidated because skipped optimizer steps advanced the sampler; V4 retries
the same batch and proves equal sequence identity (`b4ac23e8...`) across A0-A4.

| System | Functional passes | Rate | Repetition collapses |
| --- | ---: | ---: | ---: |
| A0 labeled, six routes | 379/1,400 | 27.07% | 150 |
| A1 label-free routing | 113/1,400 | 8.07% | 210 |
| A2 deranged targets | 271/1,400 | 19.36% | 216 |
| A3 no teacher payload | 4/1,400 | 0.29% | 634 |
| A4 monolithic route | 252/1,400 | 18.00% | 141 |
| T0 frozen teacher reference | 1,237/1,400 | 88.36% | 64 |

A0 beats every matched control with a positive paired stratified-bootstrap
95% lower bound, so the labels, actual teacher payload, and segmented routes
carry a causal signal in this bounded experiment. It nevertheless fails every
absolute quality family, teacher noninferiority, and zero-collapse gate. The
606,730-parameter output-side bridge lacks prompt-conditioned sequence
realization capacity. More data, steps, or nearby cake sweeps are prohibited;
any successor needs a separately preregistered architectural change. See
`ABI_CAPABILITY_COMPILER_PHASE3_MACHINE_REPORT_V1.md`.

The V6 sequence successor then added a continuous prompt encoder and three
rank-128 pre-block residuals. B0 reached 1,148/1,400 (82.00%) with 43
collapses, but B1 reached 1,224/1,400 and B4 reached 1,165/1,400. Paired
bootstraps put B0-B1 at -5.43 points (95% CI -7.50 to -3.43) and B0-B4 at
-1.21 (-3.14 to +0.71). Correct targets and teacher responses are causally
necessary, but semantic route specialization is not shown to cause the gain.
The remaining V6 seeds are prohibited. See
`ABI_CAPABILITY_COMPILER_PHASE3_SEQUENCE_REPORT_V2.md`.

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
- [Historical campaign state V4](ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V4.json)
- [Historical campaign state V5](ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V5.json)
- [Current live campaign state V69](ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V69.json)
- [Phase 3 sequence-successor protocol](ABI_CAPABILITY_COMPILER_PHASE3_SEQUENCE_SUCCESSOR_PROTOCOL_V6.json)
- [Conditional Phase 3 machine report](ABI_CAPABILITY_COMPILER_PHASE3_MACHINE_REPORT_V1.md)
- [Phase 3 sequence-successor report](ABI_CAPABILITY_COMPILER_PHASE3_SEQUENCE_REPORT_V2.md)
- [Phase 3 component diagnostic report](ABI_CAPABILITY_COMPILER_PHASE3_COMPONENT_DIAGNOSTIC_REPORT_V3.md)
- [Phase 3 shared-output report](ABI_CAPABILITY_COMPILER_PHASE3_SHARED_OUTPUT_REPORT_V4.md)
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
