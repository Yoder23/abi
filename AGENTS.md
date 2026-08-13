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

Status date: 2026-08-13.

The controlling machine-readable state is
`ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V695.json`. Phase 3 has completed and
passed every currently required machine gate, but it is not unconditionally
certified because the controlling campaign contract makes it depend on Phase
2. Phase 2 has 21,000 immutable blinded rating rows and zero completed
preferences. Phase 4 is conditionally open only under the user's explicit
authorization; its final data and Phase 5 remain locked.

The external handoff is now operational and fail-closed under
`ABI_CAPABILITY_COMPILER_PHASE2_HUMAN_RATING_READINESS_AUDIT_V554.json`.
Scoring was frozen before any rating existed; 15 focused tests reject blank or
malformed forms, changed blinded content, duplicate identities, deficient
custody attestations, premature unblinding, post-lock edits, and altered score
manifests. This readiness pass is not a human-rating pass.

V555 additionally provides a key-free, one-form-per-person, resumable rater
session with a hash-chained progress log and immutable completion export. All
three production 7,000-row forms initialize separately without the answer key;
10 focused hostile tests pass. Use
`docs/PHASE2_HUMAN_RATER_SESSION_V1.md`. This removes hand-editing risk but does
not replace the three humans.

- Phase 0: **COMPLETE**.
- Phase 1: **COMPLETE**.
- Phase 2: **MACHINE EVIDENCE COMPLETE; BLOCKED ON 21,000 JUDGMENTS FROM THREE
  INDEPENDENT HUMAN RATERS (0/21,000 FILLED)**.
- Phase 3: **MACHINE EVIDENCE COMPLETE; BLOCKED BY THE PHASE 2 PREREQUISITE**.
- Phase 4: **CONDITIONALLY OPEN; EXACT V18 PRODUCT HANDOFF PASSES; STABLE
  INFORMATION FRONTIER AND MATCHED BASELINES REMAIN; NOT CERTIFIED**.
- Phases 5-8: **LOCKED**.

The user has explicitly authorized conditional Phase 4 research while the
external Phase 2 gate remains pending. V558 completed the read-only,
end-to-end imported-information lineage audit. It found 14,596 unique source
attempts and 450,660 unique authoritative teacher-output tokens across the
three inherited teacher artifacts. It also proved that a final-bridge-only
subset would be confounded: V526 starts from V484 after V484 consumed all of
V480, while V463 already inherits the full Phase 1 IR through three stages.
Every ABI data-dependent stage must therefore be rebuilt from the immutable
pre-V443 host for each nested budget. V565 prospectively refines the consumed
total to 9,596 unique attempts and 294,212 authoritative teacher-output tokens;
5,000 targeted records were physically archived but never consumed. V567
freezes nested B10/B20/B40/B80/B100 prefixes at 1,018/2,028/4,005/7,781/9,596
unique attempts. V569 authorizes implementation and protocol design only; it
does not waive Phase 2, unconditionally certify Phase 3, authorize training
before the training protocol is sealed, open Phase 5, or permit final access or
superiority claims. V570 subsequently sealed that ABI-arm protocol. The first
clean-start B10 run used 1,018 unique source attempts and failed at 1,190/1,400
with 43 V2 collapses. B20 used 2,028 attempts and failed at 1,251/1,400 with
seven collapses. It passed teacher noninferiority, but still failed the
critical/per-capability and zero-collapse gates; coherence fell from 11/100 to
5/100. Routing remained 1,400/1,400 and strong-path identity 1,000/1,000.
The B40 screen then reached 1,360/1,400 with two collapses, passed every
critical gate, and had a +7.07-point teacher-relative paired lower-95 bound.
It still failed per-capability quality because format control was 86/100
(Wilson lower 0.7786), and zero-collapse remained false. All three failures are
preserved. B80 then passed at 1,378/1,400 with zero collapses, every absolute
machine-quality gate, and a +8.43-point paired lower-95 teacher-relative
advantage. This establishes a screening boundary only. The frozen adaptive
order now authorizes exactly B40 and B80 at seeds 130363 and 155921. B40 passed
at seed 130363 with 1,379/1,400 and zero collapses, so it is not a reproduced
adjacent-lower failure and B80 cannot be called the minimum. The paired B80
seed130363 run failed at 1,359/1,400 because coherence was 78/100 while B40
passed. Finish the two seed155921 runs before any prospective stabilization
protocol. B40 seed155921 failed at 1,359/1,400 with one collapse, so B40 is
FAIL/PASS/FAIL across the three seeds. Only the registered B80 seed155921 run
remained before the ABI-arm replication decision. It passed at 1,382/1,400
with zero collapses, leaving B80 PASS/FAIL/PASS; B40 is FAIL/PASS/FAIL. No
budget passes all three seeds and no minimum can be claimed. Only a hash-bound
read-only attribution is authorized before at most one prospectively sealed,
evidence-supported stabilization attempt. V588 attributes the largest
instability to B80 coherence in the final bridge: intermediate-to-final deltas
are +84, -6, and +13 points. The single permitted design must remove random
within-stratum exposure imbalance without new data, parameters, steps, budget,
thresholds, or final access. V591/V592 now hash-bind and test that single
intervention. Exactly the registered B40/B80 runs at seeds 104729, 130363, and
155921 are authorized; no nearby sweep or other training is authorized. The
first stabilized pair, seed 104729, reproduces the intended boundary: B40
fails at 1,360/1,400 with two collapses and B80 passes every gate at
1,383/1,400 with zero collapses. Two paired seeds remain, so no stable frontier
or minimum is yet established. Seed 130363 rejects the stabilization: B40
still passes at 1,381/1,400 with zero collapses, while B80 fails at
1,345/1,400 because coherence is 65/100. Complete only the sealed seed155921
pair for matrix closure, then perform no nearby stabilization sweep. That pair
is now complete: B40 fails at 1,360/1,400 with one collapse and B80 passes at
1,383/1,400 with zero collapses. The full stabilized matrix is exactly the
historical FAIL/PASS/FAIL and PASS/FAIL/PASS topology. V606 therefore rejects
random exposure as the cause. No nearby sampler sweep is allowed. The only
authorized next design is one deterministic, equal-weight consensus of the
aligned three seed states at B40 and B80, without training, coefficient search,
new information, architecture growth, threshold changes, or final access.
V608/V609 now seal and preflight that exact construction: all parent, router,
and bridge tensor schemas align, and only one B40 and one B80 build/evaluation
are authorized. Both have now failed: B40 scores 1,281/1,400 with one collapse;
B80 scores 1,349/1,400 with zero collapse but misses per-capability quality.
Routing and strong-parent identity remain exact. V613 rejects arithmetic
averaging because schema-aligned seed states are not functionally linearly
mergeable. No coefficient or averaging sweep is allowed. A read-only B80
parent-versus-bridge compatibility matrix may be designed next to localize
parent, bridge, and co-adaptation effects before any new architecture.
V615/V616 now seal that 3x3 B80 matrix. The three diagonal cells already exist;
only six off-diagonal evaluations are authorized, with no training, new model
construction, or cell promotion. The matrix is now complete: all six
off-diagonal cells fail. Diagonal mean quality is 1,370.33 versus 1,299.00
off-diagonal, a 71.33-prompt advantage; bridge main-effect range is 50 and
parent range is 22.33. V626 proves strong parent-bridge co-adaptation. V628/V630
then measured the proposed pre-acquisition host before training. It scored
0/1,400 with 1,373 repetition collapses while remaining byte-immutable. The
measured 61,655,050-parameter legacy checkpoint is not the separate LayerCake
repository's sealed 7,176,097-parameter product checkpoint. This is an ABI
host-selection and lineage failure, not a LayerCake regression. Do not train
against the legacy host; only a read-only, hash-bound audit of the actual
LayerCake product handoff boundary is authorized next.

V632/V634 completed that audit. The exact successful Phase 3 endpoint is three
tensor components totaling 251,260,192 bytes plus a runtime guard, not one
signed package. None of LayerCake's declared v2, v5, or v16 handoffs accepts
those schemas unchanged. Do not relabel it. The next bounded work is a separate
generic LayerCake construct plus ABI-side exact namespaced packaging; no Phase
4 frontier training is authorized until package/runtime output identity passes.

V636 through V642 close that prerequisite without changing historical
evidence. The first signed v17 archive packaged the exact tensors but failed
strict conformance before execution because its residual key schema differed;
that package and failure are preserved. LayerCake independently certified the
exact-schema `lc-direct-neural-core/18` host. ABI then built a separately signed
253,117,431-byte v18 archive without training, reshaping, teacher loading, or
tensor-value changes. On the RTX 3080 Laptop GPU it matched the immutable V526
reference exactly on 1,400/1,400 prompts across output, original pre-guard
output, token IDs, capability route, physical residual route, task route, and
guard termination, with zero receiver learning. This proves product handoff,
not a Phase 4 minimum. The next bounded frontier action is the already frozen
B100 budget at the three registered seeds; final access and matched baselines
remain closed until that ABI arm is resolved.

B100 seed104729 then failed at 1,375/1,400: instruction following was 92/100,
below both critical thresholds, with eight exact identifiers shortened during
generation. The two replication seeds were canceled. A bound read-only audit
showed every B100 parent stage supplied 0.8x B80's observations per record, so
one exact exposure-normalization attempt was authorized. It also failed,
falling to 1,369/1,400; format control missed its Wilson gate at 91/100 and
instruction following missed the critical Wilson gate at 95/100. Zero
collapse, exact routing, exact strong-path identity, and teacher
noninferiority remained. Do not sweep steps or exposure. The shared-parent
frontier is closed. V657's five-fold capability-selected development bank
passed every absolute quality gate at 1,384/1,400 but retained one
clarification collapse. V661 applied the unchanged universal V2 guard to every
selected output: functional quality stayed 1,384/1,400, all 14 capability and
critical gates remained passing, and collapse fell from one to zero. This is
read-only design evidence, not a candidate or certificate. Exactly one
prospective physically selected, capability-isolated sparse adaptation with
one active path, the unchanged guard, and complete installed/active resource
accounting may be designed and preregistered. Training is prohibited until its
protocol and preflight are sealed. V664 then physically verified and trained
that 14-route design on the measured hard B80 seed. It improved coherence from
78/100 to 98/100 and retained zero collapse, exact routing, one active route,
and teacher noninferiority, but failed at 1,312/1,400 because format control
fell to 52/100, fluent realization to 85/100, and tone control to 86/100.
Uniform all-route continued training is closed; its remaining five runs are
canceled. Only one preregistered read-only acquisition-holdout attribution may
test a route-local inherited-versus-adapted acceptance rule. No route may be
selected from development scores or retrained first. V674 tested acquisition
response CE as that non-development signal and failed at 1,355/1,400: it
rejected the harmful format route but still selected harmful fluent-realization
and tone routes. Response-CE selection is closed. One new architecture may be
designed: exclude a deterministic validation partition before training, then
accept routes only by autonomous functional validation on those untouched
acquisition prompts. Seal its protocol and preflight before training; screen
the hard B80 seed first. V677 did so with a disjoint 4,470/1,130 split, but the
held-out Phase 1 acquisition prompts rated inherited and adapted coherence
71/71 each and therefore selected inherited. The frozen artifact then failed
development at 1,361/1,400 solely because coherence remained 78/100. Phase 1
acquisition prompts are closed as a sufficient validation distribution. V683
then preregistered the only allowed read-only audit of frozen targeted and
host-supervision evidence. The inherited coherence route passed 200/200 while
the adapted route passed 198/200, so the required strict improvement failed.
All three existing acceptance sources are now closed as discriminative
validation evidence. V686 preserves that negative result. Exactly one
prospective deterministic metamorphic coherence-validation suite may now be
designed from the frozen evaluator contract, frozen before inference, proven
hash-disjoint from every governed split, and used for one read-only comparison
of the exact inherited and already trained adapted routes. Training, teacher
loading, candidate construction, final access, and development-score selection
remain prohibited. Phase 4 is not certified.

V688-V693 then executed the one allowed prospective successor without model
leakage. A 400-prompt suite spanning ten new nonce namespaces and four event
families was frozen before inference and had zero exact prompt or evaluator
overlap with 38,780 governed records. Adaptation beat inherited coherence
157/400 to 96/400, with a paired-bootstrap lower-95 gain of 10.25 points,
one-sided McNemar p=2.52e-9, and strict gains in all ten namespaces. It still
failed decisively: adapted absolute quality was only 39.25% (Wilson lower
34.59%), and PREP/ACT/DONE regressed by six points. V694 rejects route
construction and closes this trained residual as a sufficient generalizing
coherence artifact. Only one read-only taxonomy of the frozen V693 failures
may proceed before another architecture is considered; no inference, training,
guard change, final access, or candidate construction is authorized first.

The exact route-isolated endpoint scored 1,393/1,400 with zero repetition
collapse at seed 240487 and passed all four same-lineage causal controls across
three paired training seeds. Its exact fully CPU runtime is 12.835x the pinned
optimized CPU Qwen baseline by median UTF-8 bytes/s, with a paired-bootstrap
lower 95% bound of 10.565x, 0.0254x TTFT, 0.7169x peak RSS, 99.12% parent-speed
retention, and 120/120 runtime-output identity. It reproduces byte-identically
across three host initializations. Physical one-expert execution, rank-16
activation, persistent KV state, the unchanged canonical host interface,
teacher absence, provenance/accounting, and hostile mutation rejection all
pass under `ABI_CAPABILITY_COMPILER_PHASE3_FINAL_CERTIFICATE_AUDIT_RESULT_V551.json`.

No further Phase 3 architecture, training, or runtime experiment is authorized.
The only phase-closing action is external completion of the three blinded
rating forms using `docs/PHASE2_HUMAN_RATER_SESSION_V1.md` and
`docs/PHASE2_HUMAN_RATING_HANDOFF_V1.md`, followed by the
existing Phase 2 verifier and the fail-closed
Phase 3 prerequisite audit. Do not call machine-evidence completion an
unconditional Phase 3 certificate, open Phase 4, access final material, claim a
minimum information budget, or claim ABI superiority over LoRA/distillation.

### Preserved historical snapshot through V463

Status date: 2026-08-10.

The controlling machine-readable state is now
`ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V463.json`. Phase 3 remains
**UNCERTIFIED**. V459 is the strongest strict-quality development checkpoint
at 1,304/1,400 but has five genuine V2 collapses. V463 is the zero-collapse
Pareto checkpoint at 1,303/1,400 under historical V1 and 1,307/1,400 under
prospective functional V2; abstention, coherence, fluent realization, and tone
still fail. Historical scores were not overwritten. V447 repetition and V461
surface-equivalence are prospective symmetric construct repairs, not gate
relaxations or promotions. Output pointers, identity residuals, copy-balanced
sweeps, token-substrate sweeps, and checkpoint selection are closed. Only one
separately preregistered sparse weak-capability residual on V463 may be
reviewed, using the already three-seed-qualified 14-way router. Final access
remains prohibited, Phase 2 still requires 21,000 judgments from three
independent human raters, and ABI superiority claims remain prohibited.

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
- The separate LayerCake repository qualified a construct-only signed direct
  neural English-core interface without changing its sealed release. That
  construct supplies no inherited quality or performance claim.
- ABI V23 exercised that interface with a 4.01M-parameter self-causal artifact
  and failed the absolute screen: 504/1,400, 77 collapses, zero generation
  errors. This is an ABI acquisition/representation failure, not a LayerCake
  regression. V23 and its remaining seeds/controls are closed.
- V34 selected a 4,999-action exact UTF-8 BPE representation and V37 proved
  exact `lc-direct-neural-core/3` tokenizer conformance on all 16,800 bound
  prompt/output fields. Those are representation and interface results only.
- V38's plain BPE candidate and V41's ABI-label-aware candidate each failed the
  autonomous screen at 0/1,400. Both emitted the wrong fact-free mode on
  1,394/1,400 prompts. Their checkpoints, raw outputs, and negative evidence
  are sealed.
- V42 shows the V41 auxiliary classifier is 100% accurate on sampled
  acquisition prompts, 7.14% on original held-out prompts, 81.07% on held-out
  bodies, and 73.21% with matched acquisition headers. It predicts all 1,400
  original held-out prompts as `fact_free_reasoning`. An explicit LayerCake v4
  routed host is not authorized.
- V45 supersedes that failed router architecture. Its sparse BPE-plus-character
  router passes 1,400/1,400 original prompts, bodies, and metadata rejections
  independently at each of three fixed seeds; every capability is 100/100.
- V47 bound the qualified router to V41 through only 2,688 trained route values.
  Routing remained 1,400/1,400, but generation scored 760/1,400 with 20
  collapses. V49 attributes the limit to generator/bridge fit: 90.56% action
  accuracy and 72.37% exact acquisition sequences.
- V50 tested the registered non-promotional full-generator capacity bound. With
  all 4,174,280 generator parameters trainable, routing remained 1,400/1,400
  but autonomous quality reached only 785/1,400 with 36 collapses. The paired
  candidate-minus-teacher interval was [-34.57, -30.00] percentage points.
- V51 confirms that V50 remained training-fit limited at 96.53% action accuracy
  and 79.14% exact acquisition sequences. All 1,400 deployed control prefixes
  encoded identically to the training path, so this is not a control-token
  evaluation mismatch. The current topology/data branch is closed.
- Phase 3 remains **not certified**. Routing is qualified; integrated English
  generation and LayerCake host certification are not.
- The ABI moonshot is **not complete**.

The earlier historical state at this point in the chronology was
`ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V254.json`. The original campaign
contract and all historical evidence remain unchanged.

V215-V230 opened one materially distinct progressive source-block replacement
route. The corrected 32,015-action design has a hostilely verified
196,899,840-parameter copied lexical/normalization substrate: all bf16 scalars
match the frozen source exactly, with zero logits, activations, teacher-forward
tokens, or complete source blocks in the artifact. The separate LayerCake v9
host construct removes the injected prompt/response BOS boundary and passes
637 tests plus its unchanged sealed verifier without importing ABI logic.

The first fused replacement cake failed its preregistered local gate at source
layer 0 after 256 steps. Mean relative RMSE was 0.30239 (passing the 0.85
limit), but mean output cosine was 0.96709 (below 0.98). The copied substrate
remained exact. Close fused-projection width, step, data, loss, seed, threshold,
and nearby variants. The only authorized successor is no-model feasibility for
a source-topology-preserving dual attention/MLP replacement with separate
residual paths. Phase 3 remains uncertified.

V231/V232 completed that no-model successor gate. The dual-path target keeps
the verified 196,899,840-parameter copied substrate, adds 94,384,128 trainable
replacement parameters, deploys 291,283,968 parameters in 582,567,936 fp16
payload bytes, and retains a theoretical 19.21x maximum-context source/target
MAC ratio with zero complete source blocks. The separate LayerCake repository
then construct-certified the generic `lc-direct-neural-core/10` host at commit
`065d19c`; all 641 tests and its unchanged sealed verifier pass. Neither result
is acquisition, quality, measured speed, or Phase 3 evidence. One separately
preregistered topology-aware local-fit screen may train the distinct attention
and MLP residual paths against frozen source-component targets. Final material
remains prohibited.

V233-V236 completed the one authorized dual-path local screen. Attempt V234
was terminated by its command envelope before producing an artifact and is
preserved; V235 authorized only a step-zero replay. At source layer 0 the
attention intermediate passed 0.98661 mean cosine, while the final block output
reached 0.97508, still below the unchanged 0.98 gate. Relative RMSE passed at
0.26453, all 196,899,840 copied parameters remained exact, and the negative
replacement artifact contains zero source blocks. The result improves over the
fused 0.96709 cosine but does not pass. Close dual-path loss, step, seed,
threshold, symmetric-width, and nearby variants. One read-only source-MLP
residual rank audit may now determine whether the measured post-attention miss
supports a materially asymmetric sparse-MLP successor. Training is prohibited
until that audit and a new no-model feasibility gate pass.

V237-V240 completed the read-only MLP residual rank audit after preserving one
pre-teacher-load missing-field failure. No tested expansion rank qualified:
rank 512 explained 97.749% of validation-centered residual energy, below the
fixed 98% floor. More importantly, the train-derived rank-192 oracle already
reached 0.99245 mean final cosine and 0.13025 relative RMSE on validation. The
measured blocker is therefore coefficient mapping/optimization, not a rank-192
output ceiling. Do not expand or sweep MLP rank. One no-model feasibility study
may retain rank 192 while making the training-derived residual mean and output
basis immutable imported substrate and training only the coefficient map.

V241/V242 passed that no-model gate: the basis-aligned target deploys
291,382,272 parameters in 582,764,544 fp16 bytes, imports 18,972,672 basis/mean
parameters, trains 75,509,760 attention/coefficient-map parameters, retains the
same active MAC count and 19.21x theoretical margin, and contains zero source
blocks. The separate LayerCake v11 generic host then passed 645 tests and its
unchanged sealed verifier at commit `cc6ccea`. Both results remain construct/
feasibility only. One fail-fast local GPU protocol may derive bases and means
from the fixed training records, store no raw activations, and directly
supervise coefficients under the unchanged final local gate.

V243/V244 completed that single local fit and failed narrowly at layer 0:
0.97879 mean final cosine versus the unchanged 0.98 floor, with 0.23357
relative RMSE and 0.34124 coefficient relative RMSE. The copied substrate
remained exact and the artifact contains zero source blocks. Close joint
basis-aligned loss, step, seed, and nearby variants. The measured next gate is
one read-only closed-form linear coefficient-map audit using the same
train-derived rank-192 basis and fixed records; it may not train a neural model
or authorize a host until its held-out oracle clears the unchanged local gate.

V245/V246 passed that read-only audit. A fixed 3,072-to-192 ridge map fitted on
31,001 training positions reached 0.98975 mean final cosine, 0.14451 relative
RMSE, and 0.11462 coefficient relative RMSE on the 28 validation records. It
stored no map, basis, or activations and performed no neural training. One
no-model direct-linear host feasibility gate is authorized; it must eliminate
the nonlinear MLP coefficient network rather than tune V244.

V247/V248 passed direct-linear accounting, and the separate LayerCake v12 host
passes 648 tests plus its unchanged sealed verifier. The target has 277,220,352
parameters, 554,440,704 fp16 bytes, 184,857,600 maximum-context active MACs,
and zero source blocks. One fail-fast local protocol may analytically extract
each layer's coefficient map/basis/mean and train only compact attention.

V249-V252 composed the independently passing components and failed honestly at
0.97845 mean cosine. The analytic map was fitted on teacher-attention features
but executed on compact-attention features. Close that mismatched composition.
One read-only replacement-conditioned replay may solve the identical fixed
ridge map on compact-attention features; rank, ridge, records, attention,
source targets, and gates remain frozen.

V253/V254 passed that replacement-conditioned integration at layer 0 with
0.98382 mean cosine and 0.19699 relative RMSE. It wrote no artifact and is not
Phase 3 evidence beyond the local gate. One sequential 32-layer protocol may
now derive each fixed analytic map on replacement-conditioned features and fit
only compact attention, failing fast under the unchanged gate.

V191-V214 tested the authorized weight-level structural route. The independent
LayerCake repository construct-certified `lc-direct-neural-core/7`; ABI did
not import its implementation. ABI's 14,654,784-parameter structural artifact
retains zero complete source blocks and passed hostile verification of every
selection and every fp16 scalar. Its matched conformance candidate failed at
858/1,400 with 42 collapses, zero coherence, zero fact-free reasoning, and a
teacher-relative interval of [-29.43, -24.71] points. The five-pass gain over
V184 has paired interval [-0.36, +1.14] points. Attribution shows 97.75%
acquisition action fit, 60.61% exact acquisition sequences, 83.44% held-out
action fit, and a 14.31-point gap. Training changed all but 23 deployed
entries, with global cosine 0.88756 to initialization. Close this branch and
all nearby structural selection, scale, step, loss, data, and seed variants.
Phase 3 remains uncertified.

V177-V190 tested one materially distinct decoder-only causal probability-field
route. The no-model gate passed at 14,000 records and 432,371 response
positions. GPU extraction produced a 56,320,238-byte top-32-plus-residual
artifact in 884.72 teacher-forward seconds with zero hidden activations and
zero copied source parameters. Hostile verification passed all 14,000 joins
and exactly recomputed 23,872 token IDs, 23,872 probabilities, and 746 residual
masses across 28 capability-stratified records. The separate generic LayerCake
v6 host construct passed 625 tests and its unchanged sealed verifier.

The trained 13,097,999-parameter C0 student nevertheless failed at 853/1,400
with 57 collapses, zero coherence, zero fact-free reasoning, and a paired
teacher-relative interval of [-29.79, -25.07] points. Read-only attribution
measured only 96.88% acquisition action fit, 58.41% acquisition sequence fit,
77.38% held-out teacher-forced action fit, and a 19.50-point generalization
gap. This branch and all nearby loss, step, capacity, tokenizer, data, and seed
variants are closed. Standard sequence, representation, activation, lexical,
causal-state, and logit distillation have not qualified the English core. The
only authorized next work is design of a no-model weight-level structural
extraction/compression feasibility gate with explicit source-block elimination.

V108-V144 tested one evidence-driven acquisition expansion without accessing
final material. A 9,800-probe Phase-1-task-family expansion produced 9,800
initial and 1,579 repair attempts on the pinned GPU teacher. It supplied at
least 520 eligible records for 13 capabilities but only 381 abstention records,
so it failed standalone and remains negative evidence. Combining its adequate
families with the independently passing V119 abstention source produced a
7,000-record, 500-per-capability IR. Hostile verification passed all records
with source/target maxima 192/197 and exact native target round trips.

The targeted combination raised source trigram coverage to 95.40% (PASS), but
target four-gram coverage reached only 76.84% (FAIL). Instruction-following
target coverage remained 45.97% for unigrams and 28.30% for four-grams. V143
then tested native token pointers without training: all acquisition and
held-out targets round-tripped, but only 20 of 1,346 missing instruction
actions were pointerable by exact native-token identity (1.49%). Contextual BPE
segmentation is the measured blocker. Do not request a native-token-pointer
host or train this path. Any successor must be a separately preregistered exact
byte-, lexeme-, or span-aware copy representation with matched capacity and the
targeted data. Phase 3 is still **UNCERTIFIED**.

V160-V176 tested the authorized exact-copy successor without weakening a gate.
The selective identifier-boundary representation ultimately passed every
representation gate at 4,999 fixed actions: exact reconstruction of all 14,000
acquisition and 1,400 development targets, source/target maxima 182/304,
59/100 instruction pointer exposure, and a 14,407,080-parameter matched plan.
The separate LayerCake v5 host construct passed 622 tests and its unchanged
sealed verifier.

That success did not transfer autonomous English. The only authorized V170
seed scored 898/1,400 with 55 collapses, zero coherence passes, zero fact-free-
reasoning passes, and a teacher-relative 95% interval of [-26.43, -21.93]
points. Repeated-span attribution failed, and the sealed checkpoint reached
only 95.97% teacher-forced action accuracy and 68.01% exact acquisition
sequences. V170 and all nearby tokenizer, pointer, step, capacity, and seed
variants are closed. No neural training is authorized; a future candidate must
be a materially distinct non-pointer acquisition architecture with an explicit
fit and generalization mechanism.

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

V56/V57 are complete. They select a per-record prompt/response pooled final-
hidden fp16 substrate: 14,000 vectors, width 3,072, and 86,016,000 tensor bytes.
V58 completed one exact GPU extraction after its no-model inventory passed. It
produced 14,000 fp16 vectors with an 86,016,000-byte tensor payload, stored no
logits or source parameters, and recorded 407.57 seconds of teacher-forward
time, 9.06 GB peak CUDA allocation, and 2.35 GB peak process RSS. V59's hostile
verification failed because singleton recomputation exceeded its preregistered
absolute-error tolerance; no verification result was written. V60 measured
0.05371 singleton maximum error but 0.999993 minimum cosine; exact original-
batch recomputation had zero error and 100% exact fp16 scalars across all 56
sampled vectors. V61 then passed every static, provenance, balance, accounting,
and exact original-batch check. The 86,016,000-byte substrate is verified, but
learned transfer is not. One representation-aligned candidate protocol may now
be designed against the existing same-size output-only control. V62 now seals
that single A0 screen: the exact V50 generator, seed, sample order, optimizer,
updates, router, and autonomous gates are retained; only fixed-projection
prompt/response cosine alignment to V61 is added. This is standard pooled
representation distillation. A0 completed and failed at 781/1,400 with 40
collapses, four fewer passes and four more collapses than matched V50. It still
scored 0/100 on coherence and fact-free reasoning. Do not sweep alignment
weights or run additional seeds. One no-training feasibility study may test a
materially different sequence/action-aligned causal representation with fixed
pre-storage compression. V64 passed: all 50,403 source and 49,029 target actions
map across 7,000 records, every target has causal predecessor states, and the
projected payload is 38,181,888 bytes. It records 14,338 source and 31,485
target boundary straddles. V66 now preregisters one exact GPU extraction after
a no-model inventory pass. It freezes the 3,072-to-192 projection, causal
predecessor indexing, ragged offsets, source identity, and 38,181,888-byte
activation payload. V66 completed in 413.29 teacher-forward seconds and wrote
99,432 vectors with exact source and record hashes, zero logits, and zero source
parameters. The output remains unverified and training-prohibited. One hostile
verification protocol may be designed. V68 passed all rows and offsets plus
exact original-batch recomputation of 64,512 scalars across two hash-picked
records per capability with zero error. V70 then tested one matched seed-240050
candidate with the exact V50 model, optimizer, order, updates, router,
tokenizer, and gates. Per-action causal alignment improved V50 by 9/1,400
passes (paired 95% interval +0.14 to +1.14 points), but still scored only
794/1,400 with 38 collapses, zero coherence, and zero fact-free reasoning. Its
teacher-relative interval was -33.93 to -29.29 points. V70 is closed: no weight
sweep or additional seed is authorized. The tested 4.17M-parameter
`lc-direct-neural-core/3` target is a construct-only interface, not LayerCake's
separately sealed English engine, so this is an ABI-plus-construct integration
failure rather than a LayerCake regression. One read-only host-interface
readiness audit is authorized before any further neural proposal.
V72 completed and passed its attribution rule without loading the teacher model
or training. All 14,000 acquisition texts encode/decode exactly with the native
teacher tokenizer, giving zero native-action boundary straddles, but none are
raw-piece concatenative. The tokenizer requires Sequence normalization,
TemplateProcessing, Sequence decoding, and byte fallback; the bound LayerCake
v3 construct machine-rejects that graph and remains construct-only with no
external English artifact. A collision-free 32,015-action PortableTokenPlan
would contain 14,575,440 parameters (3.492x V70). No further ABI training is
authorized until LayerCake separately preregisters and construct-certifies a
decoder-aware external-core interface.
LayerCake has now completed that independent prerequisite as the construct-only
`lc-direct-neural-core/4` interface at commit `86f81e6`; its 619 tests and
unchanged sealed verifier pass. ABI accepts the frozen interface under V74 but
inherits no English-quality or performance claim. One teacher-token-native
candidate may be separately preregistered.
V75 now preregisters exactly one seed-240075 teacher-token-native sequence-
distillation screen against that frozen v4 host. Its no-model inventory passes:
7,000 records, 32,015 fixed actions, 14,587,728 trainable parameters, maximum
143 source and 250 target actions, zero token-boundary straddles, and exact
reconstruction of every target. It stores no logits, activations, or source
parameters and may not load the teacher. Any initial-screen miss closes the
branch without nearby variants.
V75 attempt 1 was externally terminated by the 120-second command envelope
after 3,250 healthy successful updates and before artifact serialization. No
candidate directory or checkpoint exists. V76 preserves that failure and
authorizes one exact replay from step zero with a 240-second-or-longer command
envelope. No scientific field changes.
The exact replay completed all 4,000 updates and V75 failed the autonomous
screen at 824/1,400 with 38 collapses, five correctly rejected pointer actions,
zero coherence, and zero fact-free reasoning. Its teacher-relative interval is
-31.86 to -27.14 points. Native actions improve V70 by 30 passes and V50 by 39
with positive paired intervals, so the v3 mismatch was material but not
sufficient. The branch is closed. One read-only teacher-forced fit and error
attribution is authorized before any materially different proposal.
V78 preregisters that read-only attribution over every acquisition record and
every held-out teacher target. It may load only the frozen candidate, never the
teacher, and cannot train or alter an artifact.
V78 passed its attribution. V75 fits 95.51% of acquisition actions but only
64.99% of complete acquisition sequences; held-out teacher-forced fit is
72.92% of actions and 26.36% of sequences. The 22.60-point action gap proves a
dominant conditional-generalization failure alongside residual fit. Output-only
teacher-native training is closed. One no-training feasibility/accounting study
may inspect projected teacher input/output lexical substrate; no extraction or
training is yet authorized.
V80 preregisters that no-model feasibility study. It may inspect only indexed
tensor shapes/dtypes and fixed projection/accounting; it cannot read tensor
values, extract, or train.
V80 passed without reading tensor values. Both lexical tables are
[32,064,3,072] BF16. Projecting the first 32,011 rows into two 192-wide fp16
tables requires 24,584,448 bytes and 12,292,224 final imported parameters,
leaving 2,295,504 bridge/special parameters. This is 16x smaller than the two
source tables and retains zero source blocks. One chunked extraction protocol
may now be designed.
V82 preregisters exactly that two-table, first-32,011-row GPU extraction with
fixed projection, fixed host-compatible row norms, chunking, source hashes, and
24,584,448 expected tensor bytes. Training remains prohibited pending hostile
verification.
V82 completed in 0.287 extraction seconds after source-hash validation. It
wrote the exact 24,584,448-byte tensor payload (24,584,856-byte file) with
12,292,224 projected parameters, zero logits/activations/source blocks, and no
teacher inference. The artifact is unverified and training-prohibited.
V84 preregisters hostile static/norm verification plus exact original-partition
recomputation of eight hash-selected rows per table. Training remains closed.
V84 passed every static/norm gate and all 3,072 independently recomputed fp16
scalars matched exactly with zero error. The substrate is verified. One
bridge-only candidate protocol may now be designed with both imported tables
frozen and verified unchanged after training.
V86 now preregisters exactly one matched seed-240075 bridge-only screen. Its
inventory passes: 14,587,728 deployed parameters, 12,292,224 imported lexical
parameters, 12,293,760 frozen values including host-special table rows, and
2,293,968 trainable bridge/runtime parameters. It retains V75's data, sample
order, steps, native actions, router, and gates. Any miss closes the branch.
V86 completed and failed at 812/1,400 with 76 collapses, zero coherence, and
zero fact-free reasoning. The imported tables remained exact. Against V75 it
lost 12 passes with a paired interval spanning zero. It did reduce wall time by
17.9% and active parameter-seconds by 7.74x, exceeding the 5x compute target,
but not at matched quality. Lexical projection is closed; a future materially
distinct route must transfer deeper causal sequence state.
V88 preregisters a no-model exact-native-action and causal-predecessor
feasibility/accounting study for target states only. It cannot load the teacher,
extract, or train.
V88 passed: all 7,000 normalized response ID sequences and all 7,000 terminal
sequences match the authoritative Phase 1 evidence. Every one of the 208,647
native response actions has a causal predecessor state. A 192-wide fp16 target
state artifact is exactly 80,120,448 bytes plus 56,008 bytes of offsets, below
the locked 128 MiB ceiling. This is feasibility only. One separately
preregistered GPU extraction is authorized; training remains prohibited until
hostile verification passes.
V90 preregisters exactly one GPU extraction of those 208,647 predecessor
states through the frozen width-192 projection. It stores no logits, teacher
weights, terminal state, or source block. The expected tensor payload is
80,120,448 bytes. Training remains prohibited.
V90 completed all 7,000 records in 418.616 teacher-forward seconds. The
80,120,448-byte tensor payload contains 208,647 projected predecessor states,
zero logits, and zero copied source parameters. It is unverified and remains
training-prohibited pending a separately preregistered hostile verifier.
V92 preregisters that verifier. It checks all hashes, all 7,000 provenance
joins and offsets, tensor structure/finiteness/nondegeneracy, then reloads the
frozen teacher solely to recompute every action in two hash-selected records
per capability using their original batch partitions. Every sampled fp16
scalar must match exactly. Training remains prohibited.
V92 passed. All 7,000 provenance joins and offsets passed; all values are
finite and nondegenerate. The verifier recomputed 873 actions across 28
original batches and matched 167,616/167,616 fp16 scalars exactly with zero
maximum error. The artifact is verified. One matched V75 candidate protocol
may add only a fixed-weight target-state cosine objective; no run is authorized
until that protocol is sealed.
V94 preregisters exactly one matched seed-240075 LayerCake v4 screen. Relative
to output-only V75, its sole scientific change is an untuned 0.1 cosine loss
between each teacher-forced decoder position and its verified native causal
target. Data, order, optimizer, steps, architecture, router, and gates remain
matched. Any miss closes this branch without weight or architecture sweeps.
V94 completed and failed at 837/1,400 with 55 collapses, two generation errors,
zero coherence, and zero fact-free reasoning. It gains 15 prompts and loses two
versus V75; the paired improvement interval is +0.43 to +1.50 points. Thus the
verified causal signal is useful but grossly insufficient. Gold-prefix-only
alignment is closed. One read-only fit/prefix-divergence attribution may select
the next materially distinct branch; no training or weight sweep is authorized.
V96 preregisters that read-only attribution over all 7,000 acquisition targets,
all 1,400 held-out teacher targets, and the 1,400 already-produced autonomous
outputs. It cannot train or load the teacher.
V96 passed attribution. V94 fits 95.52% of acquisition actions but only 72.85%
of held-out teacher actions, a 22.68-point gap and slightly worse held-out fit
than V75. Autonomous outputs match 38.57% of teacher prefixes on average;
collapsed outputs match 11.53%. Because held-out teacher-forced fit is far below
95%, exposure recovery is not the dominant next branch. The measured blocker is
conditional generalization/data coverage. One no-training coverage audit may be
designed; V94 tuning and reruns remain prohibited.
V98 preregisters that no-model coverage audit over capability-conditioned
native source and teacher-target 1- through 4-grams, exact sequences, and
length support. It cannot train or load the teacher.
V98 attempt 1 failed before evidence access because it imported the V75 loader
for a V94 protocol. V100 preserves the failure and preregisters an exact replay
with only that import corrected; all scientific fields remain unchanged.
V100 passed and measures a material coverage gap. Source trigram coverage is
94.79%; target fourgram coverage is 73.26%. Instruction following is only
27.93% on target fourgrams, with abstention, clarification, conversation,
fact-free reasoning, and tone control between 53.03% and 61.05%. Only 16
held-out targets fall outside acquisition length ranges. The next gate is a
source-record-disjoint teacher-query expansion targeting the missing structures.
Retraining V94 on the unchanged 7,000 records remains prohibited.
Final material remains inaccessible, and Phase 4 is locked.

## Historical Phase 3 authorization log

V52 is preregistered under
`ABI_CAPABILITY_COMPILER_PHASE3_RESILIENCE_PROTOCOL_V52.json`. Its placement
compiler sends the 7,000 eligible English records only to the English core,
maps chemistry, civics, mathematics, and Python to separate domain-cake
destinations, blocks those domains because they currently have zero acquisition
records, and quarantines unknown selections. This is bounded diagnosis and
placement, not exhaustive knowledge discovery or domain transfer. Only E0 seed
240052 may train.

V52 E0 has now completed and failed. It improved V50 by only 25/1,400 while
adding 13 collapses. Coherence and fact-free reasoning remained 0/100, and
outputs substituted acquisition-family identifiers for held-out prompt
identifiers. Hard capability isolation is closed. The next bounded design may
combine V34/V37's Unicode-safe exact BPE surface with explicit prompt-pointer
targets; it must not rerun V24's obsolete representation or tune V52.

V54 is that separately preregistered screen under
`ABI_CAPABILITY_COMPILER_PHASE3_BPE_POINTER_PROTOCOL_V54.json`. Its preflight
passes exact reconstruction, length, parameter, teacher-information, and host-
immutability checks. Pointer coverage is zero for coherence and fact-free
reasoning, so the candidate must still pass the unchanged all-capability gates;
partial entity-copy improvements cannot promote it. Only P0 seed 240050 may
train.

V54 P0 has now completed and failed. Its aggregate exactly matches V50 at
785/1,400, adds four collapses, and fits only 96.22% of training actions and
78.11% of complete sequences. Close the limited pointer branch. Do not extract
new teacher signals yet: first preregister a read-only feasibility/accounting
study that compares information payload, storage, source inference, alignment,
and expected LayerCake bridge cost for logits and hidden representations.

V50/V51 are complete and closed. The non-promotional full-generator capacity
control scored 785/1,400 with 36 collapses despite perfect routing, and the
read-only fit attribution measured 96.53% action accuracy and 79.14% exact
training sequences. This is a valid negative result: all development control
prefixes reproduced the training tokenization exactly. Do not run nearby
optimization, seed, data, or budget variants. A future neural proposal must be
materially distinct and separately preregistered; Phase 3 remains uncertified
and Phase 4 remains locked.

V23 is complete failed under
`ABI_CAPABILITY_COMPILER_PHASE3_DIRECT_CORE_RESULT_V23.json`. Do not tune its
data, steps, seed, rank, or fixed-action representation. One materially
distinct pointer-supervised target-representation screen may be separately
preregistered because V23 never supervised the host plan's available prompt
pointers and its raw outputs lost prompt identities. It must use the same
fail-fast absolute gates; controls and remaining seeds are prohibited unless
the initial candidate passes every gate. Phase 2 human ratings remain deferred,
Phase 3 is uncertified, Phases 4 through 8 are locked, and final access and ABI
superiority claims remain prohibited.

V24 is now preregistered under
`ABI_CAPABILITY_COMPILER_PHASE3_POINTER_CORE_PROTOCOL_V24.json`. Its preflight
losslessly reconstructs all 7,000 targets and replaces 44,336 eligible fixed
actions with source-position actions without changing the 4,011,040-parameter
topology or 4,575-entry fixed vocabulary. Run only P0 seed 240017. If any
absolute gate fails, compute only the locked paired V24-minus-V23 diagnostic
and close the representation. Teacher comparison, controls, and remaining
seeds require the initial absolute pass.

V24 attempt 1 stopped before step 1 and created no output after deterministic
cuBLAS rejected a missing process environment setting. Repair V25 authorizes
exactly one unchanged retry with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. This is a
runtime-conformance repair, not a scientific result or design change.

V24 completed after that repair and is closed failed: 601/1,400, 139
collapses, and 31 Unicode generation errors. Against matched V23 it gained 97
passes (paired 95% CI +4.43 to +9.43 points) but added 62 collapses and lost 11
prompt-grounding passes. No controls, teacher comparison, or other seeds are
authorized. The invalid UTF-8 outputs expose a separate LayerCake action-surface
gap, but even granting all 31 errors a pass leaves ABI at 632/1,400. Record the
ABI failure and audit the host gap separately; do not conflate either owner.

The separate LayerCake repository has now construct-certified the Unicode-
atomic `lc-direct-neural-core/2` successor. That closes the bounded host UTF-8
gap but supplies no ABI quality or performance evidence. V24 remains failed.
No training is authorized. The read-only diagnostic is preregistered under
`ABI_CAPABILITY_COMPILER_PHASE3_FIT_DIAGNOSTIC_PROTOCOL_V26.json`. It binds the
sealed V23/V24 checkpoints, all representable training and development targets,
fixed/pointer action accuracy, and a deterministic autonomous training sample.
It may attribute failure ownership only; it cannot promote either checkpoint
or select a successor after observing the result.

V26 attempt 1 wrote no result and changed no checkpoint. Its reporter divided
by zero when a held-out capability had no representable target actions. The
failure is preserved under
`ABI_CAPABILITY_COMPILER_PHASE3_FIT_DIAGNOSTIC_PREFLIGHT_FAILURE_V26.json`.
Repair V27 authorizes one unchanged retry and changes only empty-stratum
aggregation to explicit null metrics.

That retry completed and independently recomputed exactly. Both checkpoints
are training-fit limited: V23 reaches 95.30% action accuracy and 49.43% exact
training sequences; V24 reaches 94.93% and 61.43%. V23 can represent only
657/1,400 development targets and V24 only 941/1,400; both reject all coherence
and fact-free-reasoning targets. V23/V24 remain closed. The next bounded work is
one no-training Unicode-atomic open-vocabulary representation bake-off. It must
reach exact representability on all 7,000 training and 1,400 development
teacher targets before any fit run is preregistered.

That bake-off is preregistered as V28. It compares deterministic 10%, 25%, and
50% Unicode-character fallback exposure behind source pointers with a pure
character control. It requires exact coverage of all 8,400 bound targets,
valid UTF-8 actions, length conformance, and at least 10 training actions for
every character needed by development outputs. It selects a representation
only; training and LayerCake host changes remain prohibited.

V28 completed with no qualifying representation. Every candidate covered all
7,000 training targets but only 1,394/1,400 development targets because `%`
and `/` never occur in the training output alphabet. Every candidate also had
at least one sequence above 320 actions. Do not add the observed characters as
post-hoc exceptions. One development-independent successor may combine the
complete printable-ASCII alphabet, training-observed non-ASCII scalars,
syntax-only conformance examples, source pointers, and compact lexeme actions.

V29 tested that successor and achieved exact representation of all 8,400 bound
targets with 4,634 fixed actions. It still failed because nine development
targets exceed 320 actions; training has zero such excesses. Do not raise the
limit. The next bounded work is read-only record-level length attribution.

V31 completed that attribution. All nine failures are instruction-following
records; character fallback adds 2,691 avoidable actions versus 2,101 actions
that must be removed. Every record can clear 320 through compact fallback, so
one training-derived sublexeme representation is authorized for preregistration.
Do not add development-derived whole words or raise the host limit.

V32 and V33 heuristic substring branches failed and are closed. V34 replaced
them with a genuine training-only UTF-8-concatenative BPE bake-off. The 4,996
budget is the smallest preregistered pass: 4,999 fixed actions, exact 8,400/
8,400 reconstruction, and maxima of 99 training and 317 development actions.
V37 later passed host conformance; V38/V41 neural failures and V42 attribution
are the controlling successors documented above.

No further Phase 2 training is authorized. The A0-A4 and V6 B0-B4 branches are
closed. V6 B0 scored 1,148/1,400 with 43 collapses, lost to label-free B1 by
5.43 points, did not beat monolithic B4, and trailed T0 by 6.36 points. Seeds
130363 and 155921 are not authorized. Do not tune V6 or access final material.
Any successor requires a new preregistration tied to the measured
routing/specialization failure. Phase 2 human ratings remain deferred, Phase 3
is uncertified, and Phase 4 is locked.

The no-training V9 component diagnostic is complete. It found that bypassing
the route embedding changes B1 by only +1/1,400, while bypassing output cakes
costs 76/1,400. No altered checkpoint was persisted.

V11 is complete and closed. C0 passes every paired causal comparison and the
locked teacher-relative aggregate margin, but fails per-capability, critical,
and zero-collapse gates. Remaining seeds are prohibited. Do not tune V11. No
new training is authorized until a self-prefix-recovery successor is separately
preregistered.

The V16 read-only failure-attribution screen is complete. The exact sealed
LayerCake identity and certificates pass, so there is no LayerCake regression.
The native parent has a measured scope gap on the broader ABI suite, but no
host representational ceiling has been proven. C0 carries a strong cached-
teacher signal against C3, while one wrong self-prefix token causes a large,
persistent recovery loss. One separately preregistered bounded self-prefix-
recovery bridge successor is authorized for design; the LayerCake repository
and V11 checkpoints remain immutable.

V17 tested that successor and is complete failed. Self-prefix S0 scored
1,185/1,400 with 64 collapses versus compute-matched S1's 1,208 and 56; the
paired 95% interval for S0-S1 is wholly negative. Treating every mismatch from
one cached teacher continuation as an error damaged valid fluent alternatives.
Do not tune V17 or run its remaining seeds. A materially distinct successor
may target only objectively invalid recent-token repeats under a new protocol.

V18 completed that materially distinct test and is also closed failed. S0
scored 1,213/1,400 with 52 collapses versus S1's 1,208 and 56, but both paired
quality intervals cross zero and collapse is not better than V11 C0's 51.
No remaining seeds or nearby recovery-loss variants are authorized. One
development-contaminated, permanently non-promotional oracle-fit capacity
control may be preregistered to separate host/bridge expressivity from
acquisition generalization.

V20/V21 completed that control. Directly fitting all 1,400 development pairs
produced 1,229 passes and 89 collapses; teacher-forced accuracy was 94.11% with
0.2631 NLL, missing both fit thresholds. The measured owner is integration-
bridge fit, optimization, or expressivity—not ABI extraction and not a sealed
LayerCake regression. Stop acquisition-data and recovery-loss experiments. One
materially expanded bridge oracle control may be preregistered, with added
inference parameters explicitly accounted and no inherited speed claim.

V22 completed that expanded control and failed. The 2,238,982-parameter bridge
scored 1,248/1,400 on the exact contaminated development pairs and produced 75
collapses, failing the 99% aggregate, 95%-per-capability, and zero-collapse
gates. The frozen LayerCake state remained exact. Do not run further ABI data,
labeling, recovery-loss, rank, or nearby bridge experiments. The next work is a
separately governed LayerCake integration-interface investigation. Phase 3 is
uncertified, Phase 4 is locked, and no speed or superiority claim is inherited.

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
