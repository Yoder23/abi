# ABI active mission

Status date: 2026-08-07

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
| 3 — causal teacher-to-target acquisition | UNCERTIFIED; V95 NATIVE CAUSAL CORE FAILED, BRANCH CLOSED | `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V114.json` |
| 4–8 | LOCKED | Campaign contract |

Phase 1 certifies a data artifact, not a model. It selected 7,000 normalized
English records—500 for each of 14 capabilities—with raw forms, authoritative
token IDs, provenance, labels, transformations, and content hashes. It also
freezes disjoint development, final, isolation, and hostile material.

The original source run failed the abstention floor and remains preserved. A
fresh, preregistered successor supplied 400 new passing abstention records; none
of the failed records were reclassified. Specialist inventories are
evaluation-only, and no specialist record is eligible for English training.

## Controlling Phase 3 objective: qualify integrated capability realization

V34 and V37 close the selected representation and host-tokenizer questions:
the 4,999-action UTF-8 BPE surface represents all bound targets and matches the
independent LayerCake v3 implementation on 16,800/16,800 comparisons.

The learned acquisition question remains open. V38's 4.17M-parameter plain BPE
candidate reached 95.13% teacher-forced action accuracy and 85.23% exact
training sequences, then scored 0/1,400 autonomously. V40 attributed primary
fit/state weakness plus a major header shift: capability-matched acquisition
headers raised a 280-prompt sample from 0% to 41.07%, but training replay was
only 85.71% exact.

V41 then tested ABI-native labels rather than plain sequence imitation. A
training-only 14-way capability head, balanced header dropout, and causal
history corruption did not improve the locked suite: 0/1,400, with 1,394 wrong
fact-free-mode outputs. V42 explains why. The classifier is 100% on sampled
acquisition prompts but predicts every original held-out prompt as
`fact_free_reasoning`; body-only accuracy is 81.07%, below the locked 90% route
gate, and coherence remains 0/100.

V45 has solved the bounded routing problem. Its sparse BPE-plus-character
router passed all original, body, metadata, and per-capability gates at three
fixed seeds: 4,200/4,200 observations for each view. V47 then tested whether a
minimal 2,688-value route bridge could make that state causal in the frozen V41
generator. The router stayed perfect, but autonomous quality reached only
760/1,400 with 20 collapses. V49 measured 90.56% teacher-forced action accuracy
and 72.37% exact acquisition sequences, proving a generator/bridge fit limit.

V50 then gave the entire existing 4,174,280-parameter generator topology a
non-promotional capacity test while holding the qualified router, data,
representation, and gates fixed. It reached only 785/1,400 with 36 collapses
and a paired teacher-relative interval of [-34.57, -30.00] points. V51 measured
96.53% action accuracy and 79.14% exact acquisition sequences, so even the full
generator remained fit limited. All 1,400 evaluation control prefixes matched
the training token path exactly. The current topology/data branch is closed.

V52 is complete failed. Fourteen body-only experts reached 810/1,400 with 49
collapses despite perfect routing; teacher-relative quality remained more than
28 points below the teacher at the upper end of the paired interval. Hard
isolation is therefore closed. The preserved outputs expose prompt-entity
copying as the next measured representation target. No neural run or LayerCake
host change is currently authorized. One Unicode-safe UTF-8-BPE pointer-
supervision screen was separately preregistered as V54. It also failed:
785/1,400, 40 collapses, 96.22% action fit, and 78.11% exact sequence fit. The
limited pointer policy is closed. No neural training, new teacher extraction,
or LayerCake change is authorized. The next bounded gate is a no-training
feasibility and accounting study for richer teacher signals. Phase 2's external
human-rating gate remains unresolved, so Phase 3 cannot receive a final
certificate and Phase 4 cannot open.

V56/V57 completed that feasibility gate and selected one 86,016,000-byte
per-record prompt/response pooled final-hidden substrate. V58 preregisters one
GPU extraction only. Its preflight discovered that independently tokenized
semantic prompt IDs are not literal subsequences of the contextual chat prompt;
the protocol therefore binds the unique semantic UTF-8 text span through the
frozen fast-tokenizer character offsets and fails closed on any straddle, gap,
duplicate span, or token-count change. Training remains prohibited until a
separate hostile verifier certifies the extracted artifact.

V58 has now extracted the preregistered substrate: 14,000 fp16 vectors, an
86,016,000-byte tensor payload, 792,572 source-forward tokens, zero stored
logits, and zero copied source parameters. The measured source-forward time was
407.57 seconds with 9.06 GB peak CUDA allocation and 2.35 GB peak process RSS.
The artifact remains unverified and training remains prohibited. V59 binds one
hostile verifier over every provenance row plus independent frozen-teacher
recomputation for two hash-selected records in each capability.

V59 failed at that recomputation gate: singleton forwards exceeded the frozen
0.0078125 absolute-error tolerance, so no verification result exists and
training remains prohibited. V60 is a non-promotional, read-only numerical
diagnostic over the identical 28 records. It compares singleton forwards with
the exact original 8-record extraction-batch context before any verifier repair
can be considered.

V60 found the failure mechanism: singleton recomputation had 0.05371 maximum
absolute error while retaining 0.999993 minimum cosine, whereas recomputation
under the exact original 8-record partition reproduced every sampled fp16
scalar exactly. V61 preserves the V59 failure and preregisters a verifier-only
repair requiring that zero-error original-batch evidence plus every static,
provenance, balance, and imported-information check.

V61 passed those gates. The exact 86,016,000-byte substrate is now verified and
contains no logits or copied source parameters. This is an artifact result, not
a learned-transfer result. The next bounded experiment is one representation-
aligned LayerCake candidate compared with the existing same-size, same-seed,
output-only V50 control. Training requires a new sealed protocol.

V62 now seals that experiment. It preserves V50's 4,174,280-parameter model,
seed 240050, balanced sample order, 4,000 updates, optimizer, router, tokenizer,
and gates. The only causal change is a fixed, discarded 3,072-to-192 projection
and two untuned 0.1-weight cosine losses over the verified prompt and response
vectors. Only A0 may run; any absolute miss closes the branch without a weight
sweep, additional seeds, or inherited runtime claim.

V62 completed and failed: 781/1,400 with 40 collapses versus matched V50's
785/1,400 with 36. Its paired difference is -0.29 points with a 95% interval
of [-0.64, 0.00], and coherence and fact-free reasoning remain 0/100. The
learned pooled alignment therefore does not cause better autonomous generation.
No weight sweep or additional seed is authorized. The next bounded question is
whether action-aligned causal prediction states are feasible at a controlled
payload; this must be answered without teacher extraction or model training.

V64 seals that feasibility study. It requires exact contextual token identity,
at least one teacher-token overlap for every LayerCake body/output action,
causal predecessor states for every target token, explicit boundary-straddle
counts, and a pre-storage 192-wide fp16 payload below 512 MiB. It cannot load
the teacher model, extract tensors, or train.

V64 passed all 7,000 records. It maps 50,403 source and 49,029 target actions,
uses causal predecessor states for every target, and requires a 38,181,888-byte
projected fp16 payload. It also exposes 14,338 source and 31,485 target actions
whose finer LayerCake boundary straddles a teacher token. No extraction is
authorized until an exact ragged-tensor protocol is separately sealed.

V66 now seals one exact GPU extraction. It freezes the projection, record order,
causal predecessor indices, ragged offsets, expected payload, and source model.
The resulting artifact remains unusable for training until hostile verification
is separately preregistered and passed.

V66 completed with 99,432 vectors and a 38,181,888-byte activation payload in
413.29 teacher-forward seconds. It stored no logits or source parameters. The
artifact is still training-prohibited until a hostile verifier checks every
ragged offset and provenance row and exactly recomputes a sealed stratified
sample under the original batch partition.

V68 now seals that verifier. It requires every provenance join and ragged offset
to match, both value tensors to be finite and nondegenerate, the exact source
manifest to reproduce, and every scalar for two hash-selected records per
capability to match under the original 8-record batch partition.

V68 passed. All 64,512 sampled fp16 scalars across 336 actions and 28 records
matched exactly with zero maximum error, after all 7,000 provenance rows and
ragged offsets passed. One matched candidate protocol may now be designed;
training remains unauthorized until that protocol is sealed.

V70 now seals one matched action-aligned candidate. It retains V50's exact
model, seed, optimizer, 4,000 updates, route, tokenizer, and gates; only masked
per-action source and causal-target cosine losses are added. Any gate miss
closes the branch without tuning or additional seeds.

V70 completed and failed. It reached 794/1,400 with 38 collapses, zero
coherence, and zero fact-free reasoning. The +9 passes over matched V50 are
statistically positive but materially inadequate; the teacher-relative 95%
interval remains -33.93 to -29.29 percentage points. The target alignment loss
also remained 0.7044, while 31,485/49,029 target actions share teacher states
across tokenizer-boundary straddles. No weight sweep, nearby variant, or
additional seed is authorized. The tested `lc-direct-neural-core/3` target is
only LayerCake's construct interface and carries no inherited English-quality
or performance claim. One no-training host-interface readiness audit must now
determine whether a causally aligned external artifact can reach a genuinely
qualified LayerCake host without misattributing host readiness to ABI.

V72 completed and passed its blocker-attribution rule. Every one of 14,000
bound acquisition texts round-trips exactly through the teacher tokenizer and
would have zero native-action straddles, but 0/14,000 are raw-piece
concatenative. The graph requires normalization, post-processing, decoding,
and byte fallback, all rejected by the frozen v3 host. V3 also remains
construct-only with no external English artifact. The next gate belongs to the
separate LayerCake repository: preregister and construct-certify a decoder-
aware external-core interface. ABI neural work remains closed until that
interface is frozen.

The LayerCake repository has now independently construct-certified
`lc-direct-neural-core/4` at commit `86f81e6`, including 619 tests, the
unchanged sealed verifier, CPU/CUDA package identity, persistent state, and
zero receiver learning. ABI V74 accepts that frozen interface only; it imports
no LayerCake English-quality or performance claim. One teacher-token-native
candidate protocol may now be designed.

V75 preregisters one seed-240075 teacher-token-native candidate. Its inventory
passes all 7,000 records with a 32,015-action collision-free vocabulary,
14,587,728 parameters, maximum 143 source and 250 target actions, exact target
reconstruction, and zero teacher-token boundary straddles. The run uses cached
teacher text only, loads no teacher weights, stores no logits or activations,
and receives no inherited quality claim. Any gate miss closes the branch.

V75 attempt 1 was healthy through 3,250 updates (loss 10.58 to 0.023) but the
120-second orchestration envelope terminated it before serialization. No
candidate directory or checkpoint was created. V76 authorizes one exact replay
from step zero with only the external command timeout enlarged; all scientific
fields and the immutable output path remain unchanged.

The exact replay completed and V75 failed: 824/1,400, 38 collapses, five
pointer-action errors, zero coherence, zero fact-free reasoning, and a
teacher-relative 95% interval of -31.86 to -27.14 points. It improved V70 by
30 passes and V50 by 39 with positive paired intervals, proving the tokenizer
confound was real but not sufficient. The branch is closed. One read-only
teacher-forced fit and autonomous-error attribution is the next gate.

V78 preregisters that read-only measurement over all 7,000 acquisition records
and all 1,400 held-out teacher targets. It loads only the frozen V75 checkpoint,
cannot train, and cannot load the source teacher.

V78 passed its attribution. Acquisition teacher-forced fit is 95.51% by action
but only 64.99% by exact sequence; held-out teacher-target fit falls to 72.92%
by action and 26.36% by sequence. The 22.60-point action gap establishes a
dominant held-out conditional-generalization failure plus residual training
fit. Output-only teacher-native training is closed. The next bounded question
is a no-training feasibility/accounting study for projected teacher input and
output lexical substrate.

V80 preregisters that no-model feasibility study. It may inspect only source
tensor shapes/dtypes and deterministic projection/payload accounting; tensor
values, extraction, training, teacher inference, and final material are closed.

V80 passed without reading tensor values. The two [32,064,3,072] BF16 lexical
tables can become two 192-wide fp16 tables over the first 32,011 actions in
24,584,448 bytes: 12,292,224 imported parameters, 2,295,504 remaining host
bridge/special parameters, 16x source-table compression, and zero source
blocks. A separately sealed extraction protocol is now required.

V82 preregisters exactly one chunked GPU extraction of the first 32,011 rows
from the two bound lexical tensors. Projection, row norms, source hashes,
chunking, dtype, and the 24,584,448-byte tensor payload are frozen. Training is
prohibited until a separate hostile verifier passes.

V82 completed and wrote 12,292,224 projected parameters in a 24,584,448-byte
tensor payload after validating both source-file hashes. It used no teacher
inference and retains no logits, activations, or source blocks. The artifact is
unverified and remains training-prohibited.

V84 preregisters hostile static, norm, and source-row recomputation checks over
the artifact. All 3,072 sampled fp16 scalars must match exactly. Training is
still prohibited.

V84 passed: all static and norm gates plus all 3,072 independently recomputed
fp16 scalars matched exactly with zero error. The substrate is verified. One
bridge-only candidate may now be preregistered with both imported tables frozen
and post-training identity required.

V86 preregisters one matched seed-240075 screen. Its inventory passes with
14,587,728 deployed parameters, 12,292,224 imported lexical parameters,
12,293,760 frozen table values, and only 2,293,968 trainable bridge/runtime
parameters. Data, sample order, native actions, router, steps, and gates match
V75. Any gate miss closes the branch.

V86 completed and failed at 812/1,400 with 76 collapses, zero coherence, and
zero fact-free reasoning. Both imported tables remained bit-identical. It cut
wall time by 17.9% and active parameter-seconds by 7.74x, exceeding the 5x
compute target, but failed matched quality and lost 12 passes versus V75 with a
paired interval spanning zero. Lexical projection is closed. Any next proposal
must transfer deeper causal sequence state rather than tune this branch.

V88 preregisters a no-model study of exact native response actions, terminal
identity, causal predecessor availability, and compressed target-state payload.
Teacher loading, extraction, training, and final material remain prohibited.

V88 passed every feasibility gate without loading the teacher. It establishes
7,000/7,000 native response sequences, 7,000/7,000 terminal sequences, and
208,647/208,647 causal predecessor states. A separately preregistered exact GPU
extraction is the only next authorized operation; training remains prohibited.

V90 preregisters that single GPU extraction. It binds the frozen teacher,
Phase 1 IR, exact native action order, fixed projection, 785,572 forward tokens,
and the 80,120,448-byte target-state payload. No training or final-test access
is authorized, and the output remains unusable until hostile verification.

V90 completed all 7,000 records. It stores 208,647 fixed-projection causal
predecessor states in an 80,120,448-byte tensor payload, with zero logits and
zero copied teacher parameters. Teacher forward time was 418.616 seconds at
1,876.59 tokens/s. The artifact is unverified and training-prohibited.

V92 preregisters hostile verification of every static/provenance/offset gate
and exact original-batch recomputation of all actions in 28 records, two per
capability. Every sampled fp16 scalar must match. Training remains prohibited.

V92 passed. All 7,000 provenance joins and offsets passed, and 167,616 of
167,616 sampled fp16 scalars across 873 actions matched exact original-batch
recomputation with zero maximum error. One matched V75 native-action candidate
may now be designed with target-state cosine alignment as its only change.

V94 binds that single candidate at the matched V75 seed, data order, optimizer,
4,000 updates, architecture, router, and autonomous gates. Its only scientific
change is the untuned 0.1 per-action causal-state cosine objective. Any miss
closes this branch without a weight or nearby architecture sweep.

V94 completed and failed at 837/1,400 with 55 collapses, two generation errors,
zero coherence, and zero fact-free reasoning. Its paired improvement over V75
is real (15 gains, two losses; +0.43 to +1.50 points), proving the state signal
is useful but insufficient. Gold-prefix-only alignment is closed; only one
read-only fit/prefix-divergence attribution is authorized next.

## Historical direct-core path

The separate LayerCake repository has qualified a construct-only signed direct
neural core interface. ABI V23 used that interface without copying LayerCake
code and failed its initial absolute screen at 504/1,400 with 77 repetition
collapses. Thirteen of fourteen capabilities missed the ordinary statistical
gate, and all three critical capabilities missed their stricter gates. The
teacher-relative comparison and causal controls were correctly not reached.

V23 is an ABI acquisition/representation failure, not a LayerCake regression.
Its fixed-action targets never supervised the available prompt-pointer actions;
raw outputs frequently replaced or duplicated prompt entities. The only open
experimental question is one separately preregistered pointer-supervised target
representation. No V23 tuning or rerun is authorized.

V24 binds that single question. Its inventory reconstructs all 7,000 training
targets byte-exactly while encoding 44,336 eligible target lexemes as source
pointers. Architecture, fixed vocabulary, seed, sampler, optimizer, steps, and
all absolute gates match V23. Only P0 seed 240017 may run. An absolute miss
authorizes only the registered paired V24-minus-V23 diagnostic before closure.

Attempt 1 stopped before training step 1 and created no artifact because the
process lacked the deterministic cuBLAS workspace setting. V25 preserves that
failure and permits one exact retry with `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
every scientific field remains unchanged.

V24 then completed and failed: 601/1,400, 139 collapses, and 31 Unicode
generation errors. Its +97 paired passes over V23 are statistically positive
but do not approach the absolute gates and come with worse collapse and prompt
grounding. V24, its controls, and remaining seeds are closed. A separate
LayerCake UTF-8 validity audit is required before another ABI architecture is
considered.

The separate LayerCake repository has completed that host-owned repair as the
construct-only `lc-direct-neural-core/2` interface. ABI inherits no quality or
performance from it, and V24 remains failed. The only active ABI work is the
V26/V27 attribution is complete. Both sealed checkpoints fail training fit;
V23 represents 657/1,400 held-out teacher targets and V24 represents 941/1,400.
Both reject all coherence and fact-free-reasoning targets. The next work is a
no-training Unicode-atomic open-vocabulary representation bake-off. No fit run
is authorized until one representation losslessly covers all 7,000 training
and 1,400 development targets.

V28 now binds that bake-off. It compares three deterministic character-
fallback exposure rates with a character-only control and cannot train a model
or change the LayerCake host.

V28 completed with no qualifier: all candidates miss six development targets
because the training alphabet lacks `%` and `/`, and all exceed the 320-action
limit on at least one record. The next successor must use a universal syntax
alphabet selected without development feedback and retain compact lexeme
actions. Training remains closed.

V29 fixed coverage without development-derived exceptions: 8,400/8,400 targets
are representable with 4,634 fixed actions. Nine development targets exceed
320 actions, so V29 fails and training remains closed. Measure those nine
records before proposing compression or any host limit change.

V31 attributed the nine failures to character fallback. V32/V33 heuristic
sublexeme designs failed and are closed. V34's genuine training-only UTF-8 BPE
passed exact reconstruction and the original limits at the smallest tested
4,996 budget. V37 later passed LayerCake host conformance; V38 and V41 then
failed neural acquisition as recorded in the controlling section above.

## Preserved historical Phase 3 path

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

V11 is complete. C0 scores 1,207/1,400 and passes all four causal comparisons
plus the locked teacher-relative aggregate noninferiority margin. It fails
absolute capability floors and has 51 collapses, 41 in email drafting.
Remaining seeds are prohibited. No V11 tuning is authorized.

V16 now attributes the failure without training. The exact LayerCake lineage
passes identity and certificate checks. C0 reaches 82.63% teacher-forced token
accuracy versus 27.61% for no-payload C3, proving that the cached ABI payload
is present. One C0-generated wrong prefix token leaves 16-token NLL 1.77x worse
than the clean prefix. The next branch must target this bridge state-recovery
failure and must not modify the sealed LayerCake host.

V17 completed that test and failed: S0 scored 1,185/1,400 with 64 collapses,
significantly below compute-matched S1 at 1,208/1,400 with 56. The result shows
that arbitrary disagreement with one cached teacher token is not a valid error
definition. V17 is closed. The only supported next recovery design is one that
corrupts objectively invalid recent-token repeats and keeps the same matched
control and immutable LayerCake boundary.

V18 tested recent-repeat-only recovery. It avoided V17's harm but did not
significantly beat S1 or V11 C0 and retained 52 collapses. The recovery-loss
family is closed. The active objective is now one oracle-fit capacity control
that is permanently barred from promotion and uses development data only to
decide whether the next production repair belongs to bridge expressivity or
ABI acquisition/generalization.

V20/V21 resolved that gate: the current 1.06M bridge misses even the
development-contaminated upper bound (1,229/1,400, 89 collapses, 94.11%
teacher-forced token accuracy). Further ABI data work is paused. The active
objective is one materially expanded, non-promotional bridge capacity control,
followed by explicit inference-cost measurement before any production retry.

V22 completed that final authorized ABI-side ownership control. Doubling the
sequence and shared-output ranks produced 1,248/1,400 with 75 collapses on the
exact contaminated development pairs. It failed the 99% aggregate,
95%-per-capability, and zero-collapse gates. The frozen LayerCake state hash
remained exact, so no regression is present; a fundamental host ceiling also
remains unproven. ABI extraction and labeling were not under test. Further ABI
experimentation is stopped until a separately governed LayerCake investigation
qualifies an integration interface. The expanded candidate inherits no host
speed, memory, TTFT, sparsity, or quality claim.

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

- Do not rerun or tune V23. A pointer-supervised successor must be a new,
  hash-bound representation experiment and must fail fast on the same absolute
  screen before any controls or additional seeds.

- Do not rerun or tune the completed A0-A4 or B0-B4 branches. Their remaining
  two seeds were not authorized after their first seeds failed locked gates.
- Do not launch an ABI successor until a separate LayerCake integration
  investigation qualifies the host interface and a fresh protocol binds that
  interface, matched controls, same-candidate runtime, and the final firewall.
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

- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V89.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_ACTION_ALIGNED_CORE_PROTOCOL_V70.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V88.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_ACTION_ALIGNED_VERIFICATION_RESULT_V69.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V87.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_ACTION_ALIGNED_VERIFIER_PROTOCOL_V68.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V86.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_ACTION_ALIGNED_EXTRACTION_RESULT_V67.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V85.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_ACTION_ALIGNED_EXTRACTION_PROTOCOL_V66.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V84.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_ACTION_ALIGNED_FEASIBILITY_RESULT_V65.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V83.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_ACTION_ALIGNED_FEASIBILITY_PROTOCOL_V64.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V82.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_REPRESENTATION_ALIGNED_CORE_RESULT_V63.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V81.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_REPRESENTATION_ALIGNED_CORE_PROTOCOL_V62.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V80.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_RESULT_V61.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V79.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_VERIFIER_REPAIR_V61.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V78.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_NUMERICS_PROTOCOL_V60.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_VERIFIER_FAILURE_V59.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V77.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_VERIFIER_PROTOCOL_V59.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V76.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_EXTRACTION_PROTOCOL_V58.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V75.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_RESULT_V57.json`
- `results/abi_capability_compiler_phase3_teacher_representation/feasibility_v56.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V73.json`
- `ABI_CAPABILITY_COMPILER_PLACEMENT_MANIFEST_V52.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_BPE_POINTER_RESULT_V55.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_BPE_POINTER_REPORT_V55.md`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V72.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_BPE_POINTER_PROTOCOL_V54.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V71.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_RESILIENCE_RESULT_V53.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_RESILIENCE_REPORT_V53.md`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V70.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_RESILIENCE_PROTOCOL_V52.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V69.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_CAPACITY_RESULT_V50.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_CAPACITY_FIT_RESULT_V51.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_CAPACITY_REPORT_V50.md`

The entries below are preserved historical authorities for their exact scopes;
they do not supersede V69.

- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V30.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_UNIVERSAL_SYNTAX_REPORT_V29.md`
- `results/abi_capability_compiler_phase3_universal_syntax/universal_syntax_v29.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V28.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_REPRESENTATION_BAKEOFF_REPORT_V28.md`
- `results/abi_capability_compiler_phase3_representation_bakeoff/representation_bakeoff_v28.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V27.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_REPRESENTATION_BAKEOFF_PROTOCOL_V28.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V26.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_FIT_DIAGNOSTIC_REPORT_V26.md`
- `results/abi_capability_compiler_phase3_fit_diagnostic/fit_decision_v26.json`
- `results/abi_capability_compiler_phase3_fit_diagnostic/fit_generalization_v26.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V25.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_FIT_DIAGNOSTIC_RUNTIME_REPAIR1_V27.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_FIT_DIAGNOSTIC_PREFLIGHT_FAILURE_V26.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V24.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_FIT_DIAGNOSTIC_PROTOCOL_V26.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V23.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V22.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_POINTER_CORE_RESULT_V24.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_POINTER_CORE_REPORT_V24.md`
- `results/abi_capability_compiler_phase3_pointer_core/pointer_core_decision_v24.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V21.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_POINTER_CORE_RUNTIME_REPAIR1_V25.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_POINTER_CORE_PREFLIGHT_FAILURE_V24.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V20.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_POINTER_CORE_PROTOCOL_V24.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_POINTER_CORE_INVENTORY_V24.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V19.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_DIRECT_CORE_RESULT_V23.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_DIRECT_CORE_REPORT_V23.md`
- `results/abi_capability_compiler_phase3_direct_core/direct_core_decision_v23_corrected_v1.json`
- `ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V17.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_EXPANDED_ORACLE_RESULT_V1.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_EXPANDED_ORACLE_REPORT_V9.md`
- `ABI_CAPABILITY_COMPILER_PHASE3_SHARED_OUTPUT_PROTOCOL_V11.json`
- `ABI_CAPABILITY_COMPILER_PHASE3_SHARED_OUTPUT_REPORT_V4.md`
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
