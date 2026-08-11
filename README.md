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

Status date: 2026-08-11

**The ABI English-product moonshot is OPEN.**

The authoritative state is `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V631.json`.
The route-isolated English endpoint now passes all registered Phase 3 machine
gates: three paired training seeds and all A0-A4 causal controls, 1,393/1,400
selected-seed autonomous quality with zero collapse, exact routing, three
byte-identical host initializations, physical one-expert rank-16 execution,
persistent KV state, unchanged canonical host attachment, teacher absence, and
exact-lineage fully CPU runtime. The same checkpoint measures 12.835x median
UTF-8 throughput versus pinned optimized CPU Qwen (paired lower 95% 10.565x),
0.0254x TTFT, 0.7169x peak RSS, and 99.12% parent-speed retention.

Phase 3 is not unconditionally certified because Phase 2 is a prerequisite.
Its human packet has 21,000 blinded comparisons and 0 filled preferences.
Until three independent raters complete and lock those forms and Phase 2
passes verification, Phase 3 cannot receive an unconditional certificate.
Conditional Phase 4 research is user-authorized, but final access,
minimum-information certification, and superiority claims remain prohibited. See
`ABI_CAPABILITY_COMPILER_PHASE3_FINAL_CERTIFICATE_AUDIT_RESULT_V551.json`.

The Phase 2 external handoff is no longer underspecified. V554 preregisters
the scoring rule before any preference exists, requires three distinct-rater
and custodian attestations, locks all form hashes before unblinding, and
recomputes scores from raw locked forms. Fifteen hostile readiness tests pass.
Follow `docs/PHASE2_HUMAN_RATING_HANDOFF_V1.md`. This proves workflow
readiness, not completion of the missing human judgments. The complete
ABI-local suite passes 870/870; the known separate LayerCake exact-HEAD control
continues to fail closed because that clean checkout is ahead of its pinned
commit.

For actual raters, V555 adds one key-free resumable session per sealed form.
Progress is append-only and hash-chained, incomplete export is prohibited, and
all three production 7,000-row forms initialize independently without copying
the answer key. The focused rater-session suite passes 10/10. See
`docs/PHASE2_HUMAN_RATER_SESSION_V1.md`.

The user has conditionally opened Phase 4 research without waiving the pending
human gate. V558 completed the end-to-end container and checkpoint-lineage
audit. V565 then separated archive inventory from records actually consumed:
the endpoint uses 9,596 unique teacher attempts and 294,212 authoritative
output tokens; 5,000 unused targeted records remain counted in physical disk
footprint only. A smaller final payload cannot count as a
smaller substrate because V526 initializes from a full-V480 V484 bridge, and
the V463 core already inherits full Phase 1 data through V443 and V459. Phase
4 must rebuild every ABI data-dependent stage from the same fixed host at each
nested budget and compare matched LoRA and distillation frontiers. V559 permits
that protocol design but not training until it is sealed. V567 now freezes
balanced nested B10/B20/B40/B80/B100 consumed-information prefixes and the
three-seed adaptive order. V570 implemented the clean-start ABI arm. B10
(1,018 unique attempts, 31,434 teacher tokens) failed scientifically at
1,190/1,400 with 43 collapses. B20 (2,028 attempts, 62,417 tokens) also failed:
it improved to 1,251/1,400 and seven collapses and passed teacher
noninferiority, but still missed critical/per-capability quality and
zero-collapse; coherence fell from 11/100 to 5/100. B40 then reached
1,360/1,400, two collapses, all critical gates, and a +7.07-point paired lower
95% teacher-relative advantage. It still failed the per-capability gate because
format control was 86/100 (Wilson lower 0.7786), and its two collapses violate
zero-collapse. B80 is the first screening pass at 1,378/1,400 with zero
collapses, every absolute gate, and a +8.43-point paired lower-95
teacher-relative advantage. Routing and strong-path identity remained exact.
B40 is now the adjacent lower failure. The sealed adaptive order authorizes
only B40 and B80 at seeds 130363 and 155921 to test whether this boundary
reproduces. The first replication showed it does not yet: B40 passed at seed
130363 with 1,379/1,400 and zero collapses. B80 therefore cannot currently be
called the minimum passing budget. Its paired B80 run then failed at
1,359/1,400 because coherence reached only 78/100, despite zero collapses and
strong teacher-relative quality. The frontier is presently seed-sensitive and
non-monotonic. The two seed155921 runs must finish before any prospective
stabilization protocol is designed. B40 seed155921 also failed at 1,359/1,400
with one collapse, leaving B40 at FAIL/PASS/FAIL across the three seeds. Only
the final B80 seed155921 run remained in the sealed replication set. It passed
at 1,382/1,400 with zero collapses, leaving B80 PASS/FAIL/PASS. Therefore no
budget passes all three seeds, and the adjacent-lower failure is also not
all-seed. The ABI arm has no certified information frontier. Only a hash-bound
read-only seed-stability attribution is authorized before one prospective,
bounded stabilization protocol may be considered. V588 now provides that
hash-bound attribution: B80 coherence changes by +84, -6, and +13 points from
V463 to the final bridge across seeds, a 90-point effect range. B40's largest
final variation is format control at 13 points. Exactly one deterministic,
within-stratum exposure-balanced final-bridge stabilization is now sealed and
preflighted under V591/V592. Only B40/B80 at the three registered seeds may
run, with no new data, larger budget, threshold change, nearby sweep,
matched-baseline run, or final access authorized first. The first stabilized
pair is complete: at seed 104729, B40 fails at 1,360/1,400 with two collapses
and B80 passes at 1,383/1,400 with zero collapses. The boundary is reproduced
for one seed, not across all three. Seed 130363 rejects the intervention: B40
passes at 1,381/1,400 with zero collapses while B80 fails at 1,345/1,400 due
to coherence at 65/100. Only seed155921 remains for complete sealed-matrix
closure; no nearby stabilization sweep is authorized. That final pair leaves
the complete stabilized matrix unchanged from history: B40 is
FAIL/PASS/FAIL and B80 is PASS/FAIL/PASS. V606 therefore rejects random
within-stratum exposure as the instability cause. One prospective,
equal-weight consensus of aligned three-seed states at B40 and B80 may be
designed next without retraining or interpolation search. V608/V609 now seal
and preflight that exact one-third construction; only one B40 and one B80
consensus build/evaluation are authorized. Both fail: B40 scores 1,281/1,400
with one collapse and B80 scores 1,349/1,400 with zero collapse but misses
per-capability quality. Routing and strong-parent identity remain exact, so
V613 rejects equal-weight averaging as a functionally invalid merge of the
seed-specific weak-capability solutions. No averaging sweep is authorized;
only a read-only B80 parent/bridge compatibility attribution may be designed.
V615/V616 now seal that 3x3 matrix; only the six missing off-diagonal
evaluations are authorized, and no cell selection or promotion is permitted.
All six off-diagonal cells fail. V626 measures a 71.33-prompt diagonal pairing
advantage versus a 50-prompt bridge main-effect range and 22.33-prompt parent
range, proving strong parent-bridge co-adaptation. Any next ABI architecture
must hold one canonical host immutable and train only a self-contained
capability artifact against it.
The V628/V630 baseline stopped that training before compute was spent: the
proposed legacy host scored 0/1,400 with 1,373 repetition collapses, and its
61.7M architecture/checkpoint is not LayerCake's separate sealed 7.18M Phase 2
product core. This is an ABI host-lineage defect, not a LayerCake regression.
Phase 4 remains uncertified; only a hash-bound read-only audit of the actual
product handoff is authorized before more training.
Phase 2 and Phase 3
certificate statuses remain unchanged, final data remains unopened, and Phase
5 is locked.

### Preserved historical narrative through V463

The machine-readable live state is
`ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V463.json`. Phase 3 is **not
certified**: there is no qualified integrated teacher-derived English artifact.
V459 is the strongest strict development result at 1,304/1,400, but it has
five genuine V2 repetition collapses and misses four capability gates. V463 is
the zero-collapse Pareto result at 1,303/1,400 under historical V1 and
1,307/1,400 under prospective surface-equivalence V2; abstention, coherence,
fluent realization, and tone still fail. Neither checkpoint is promoted.

V447 prospectively repaired a repetition detector that falsely labeled normal
teacher prose as collapse. V461 prospectively added number-word/digit and the
explicitly requested abstention phrase as surface equivalents. Historical V1
evidence remains unchanged. The symmetric V461 audit moved the teacher from
1,237 to 1,308 and V459 from 1,304 to 1,309 while leaving catastrophic
controls at 4 and 10, proving both that the old evaluator was defective and
that the current LayerCake deficit remains real. Only one sparse weak-
capability residual on V463 using the already qualified 14-way router may be
reviewed next.

Phase 2 also remains externally blocked on 21,000 blinded judgments from three
independent human raters. ABI must not claim fluent transfer, minimum-
information English acquisition, Phase 3 completion, or superiority to LoRA
or distillation. New compute is limited by V463 and must preserve the final CPU
execution gates.

Three facts define the current checkpoint:

1. The historical v47 bounded reference passes its exact synthetic catalog and
   packaging/runtime suite, but fails broad product interpretation: it scored
   0/28 on a frozen novel-form English audit while Phi-3 scored 19/28.
2. V87 demonstrates a real causal teacher-artifact-to-LayerCake signal against
   parent and shuffled controls, but its formal transfer-quality gate fails and
   it is not deployable.
3. V93 verifies an 80,120,448-byte native causal teacher substrate: all 7,000
   provenance joins and offsets pass, and 167,616/167,616 independently
   recomputed fp16 scalars match exactly. V94 proves this signal is useful but
   insufficient, improving V75 by 13 paired passes while failing the absolute
   quality and repetition gates.

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
V52 tested one materially distinct resilience screen: 14 independently trained,
body-only capability experts behind the qualified router. It failed at
810/1,400 with 49 collapses. Hard isolation improved V50 by only 25 prompts;
preserved outputs instead expose held-out prompt-entity copying as the next
measured representation target. Its placement compiler still sends only
eligible English acquisition records to the core, maps known specialist
selections to separate domain cakes, and fails closed because no domain
acquisition payload exists yet.
The representation campaign subsequently qualified LayerCake v4 native actions,
verified a 208,647-state causal substrate, and tested one matched causal-state
candidate. V94 scored 837/1,400 with 55 collapses, two errors, zero coherence,
and zero fact-free reasoning. Its +0.43 to +1.50-point paired improvement over
V75 proves causal utility, not product adequacy. V97 measures 95.52% acquisition
action fit versus 72.85% held-out teacher-action fit, a 22.68-point gap. The
dominant current blocker is conditional generalization/data coverage, not
routing, token identity, host conformance, lexical geometry, or exposure-only
recovery. The next gate is a no-training acquisition-coverage audit.
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
| Phase 2 human rating packet and handoff | FAIL-CLOSED READY; 21,000 independent judgments pending | `ABI_CAPABILITY_COMPILER_PHASE2_HUMAN_RATING_READINESS_AUDIT_V554.json` |
| Phase 3 exact route-isolated endpoint | MACHINE EVIDENCE COMPLETE; prerequisite blocked | `ABI_CAPABILITY_COMPILER_PHASE3_FINAL_CERTIFICATE_AUDIT_RESULT_V551.json` |
| Phase 3 three-seed causal replication | PASS | `ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_ISOLATED_REPLICATION_RESULT_V538.json` |
| Phase 3 exact fully CPU runtime | PASS | `ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_ISOLATED_RUNTIME_COMPOSE_RESULT_V544.json` |
| Phase 3 three-host reproduction | PASS | `ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_ISOLATED_HOST_REPRODUCTION_RESULT_V547.json` |
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
| V52/V53 hard-routed resilience screen | COMPLETE FAIL, 810/1,400, 49 collapses | `ABI_CAPABILITY_COMPILER_PHASE3_RESILIENCE_RESULT_V53.json` |
| V54/V55 Unicode-safe BPE pointer screen | COMPLETE FAIL, 785/1,400, 40 collapses | `ABI_CAPABILITY_COMPILER_PHASE3_BPE_POINTER_RESULT_V55.json` |
| V73 LayerCake host-interface audit | BLOCKED old host; caused isolated v4 host work | `ABI_CAPABILITY_COMPILER_PHASE3_HOST_INTERFACE_AUDIT_RESULT_V73.json` |
| V75/V77 native-action output-only control | COMPLETE FAIL, 824/1,400, 38 collapses | `ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_NATIVE_CORE_RESULT_V77.json` |
| V83/V85 projected lexical substrate | EXTRACTED AND VERIFIED | `ABI_CAPABILITY_COMPILER_PHASE3_LEXICAL_SUBSTRATE_VERIFICATION_RESULT_V85.json` |
| V86/V87 bridge-only lexical screen | COMPLETE FAIL, 812/1,400, 76 collapses | `ABI_CAPABILITY_COMPILER_PHASE3_LEXICAL_SUBSTRATE_CORE_RESULT_V87.json` |
| V89/V91/V93 native causal substrate | FEASIBLE, EXTRACTED, HOSTILE-VERIFIED | `ABI_CAPABILITY_COMPILER_PHASE3_NATIVE_CAUSAL_VERIFICATION_RESULT_V93.json` |
| V94/V95 native causal core | COMPLETE FAIL, 837/1,400, 55 collapses; positive paired signal | `ABI_CAPABILITY_COMPILER_PHASE3_NATIVE_CAUSAL_CORE_RESULT_V95.json` |
| V96/V97 fit and prefix attribution | COMPLETE; conditional-generalization/data-coverage failure | `ABI_CAPABILITY_COMPILER_PHASE3_NATIVE_CAUSAL_FIT_RESULT_V97.json` |
| V98/V100/V101 acquisition coverage | COMPLETE; 94.79% source trigram / 73.26% target fourgram | `ABI_CAPABILITY_COMPILER_PHASE3_ACQUISITION_COVERAGE_RESULT_V101.json` |
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
rating forms. Phase 3's registered machine gates pass, but it cannot receive an
unconditional certificate or open Phase 4 while the Phase 2 predecessor gate
is unresolved.

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
