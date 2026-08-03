# ABI capability-compiler campaign v1

Status date: 2026-08-03
Status: controlling plan; Phase 0 is OPEN; no new training run is authorized

## Decision

ABI will be developed and evaluated as a **provenance-preserving capability
compiler**, not as a renamed fine-tuning or distillation algorithm.

Its intended job is to compile measured capabilities from one or more pinned
teachers into:

1. a minimal tested English substrate;
2. separately installable specialist artifacts;
3. explicit quarantine for unknown, mixed, disputed, or unsafe material; and
4. a teacher-free LayerCake deployment with exact package identity and
   selected-only execution.

ABI may use sequence targets, logits, activations, adapters, low-rank
factorizations, or ordinary supervised losses as internal compiler backends.
Those mechanisms retain their standard names. If the result is only a student
trained on teacher behavior, it is distillation. If the result is only a
low-rank update attached to a frozen base, it is LoRA. ABI is distinct only if
the integrated compiler contract below is actually demonstrated.

## What "supersedes LoRA and distillation" means

It does not mean that ABI must invent a new optimizer or win every metric in
every setting. It means that, for the LayerCake product objective, one
ABI-produced system must pass all common quality gates and provide capabilities
that the matched alternatives do not jointly provide:

- the source teacher and source base are absent at inference;
- the target architecture is native LayerCake rather than the source model
  plus attached updates;
- English and declared domains have record-level provenance and immutable,
  separate destinations;
- unselected capabilities are excluded under a bounded behavioral audit;
- a domain can be installed, removed, restored, and transferred exactly;
- only selected capabilities execute;
- the smallest passing **tested** imported-information budget is measured;
- the same final candidate passes quality, isolation, memory, TTFT, CPU/GPU,
  and reproducibility gates; and
- total deployed and active costs include every required base, router, bridge,
  tokenizer, cache, and runtime dependency.

Two comparative conclusions are allowed:

- **ABI product superiority:** the ABI LayerCake passes every hard product gate,
  is quality-noninferior to a strong matched baseline, and is strictly better
  on the preregistered deployment and capability-control endpoints.
- **ABI method superiority:** ABI also beats the matched alternative at equal
  imported-information and host-capacity budgets. This stronger claim requires
  its own paired statistical result.

No universal claim that ABI is "better than LoRA" or "better than
distillation" is permitted. If a matched conventional method wins the product
comparison, it becomes the preferred acquisition backend and the negative ABI
evidence is preserved.

## Prior-art boundary

The campaign treats the following as established baselines, not ABI novelty:

- [knowledge distillation](https://arxiv.org/abs/1503.02531) compresses teacher
  behavior into a deployable student;
- [LoRA](https://arxiv.org/abs/2106.09685) freezes a pretrained base and injects
  trainable low-rank updates;
- [AdapterFusion](https://arxiv.org/abs/2005.00247),
  [LoraHub](https://arxiv.org/abs/2307.13269), and
  [X-LoRA](https://arxiv.org/abs/2402.07148) establish adapter extraction,
  composition, and routing as prior art;
- [Self-MoE](https://arxiv.org/abs/2406.12034) establishes modular routed LoRA
  experts built from self-generated data; and
- [LESS](https://arxiv.org/abs/2402.04333) establishes targeted influential
  data selection, while
  [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301) shows that a
  smaller student can exceed a larger prompted teacher on bounded tasks.

The research hypothesis that remains is the integrated combination of
capability-level decomposition, reconstruction into a different native
architecture, measured sufficient information, bounded exclusion, exact
portable packages, and selected-only execution. Novelty is not claimed merely
because these properties appear together in a plan; it requires the campaign
result plus a separate professional prior-art review.

## "Stronger than the teacher" rule

Teacher improvement is a stretch result, not a substitute for faithful
acquisition. A final LayerCake may be called stronger than its teacher only on
a named, frozen evaluation scope when:

1. the paired confidence interval for the primary aggregate improvement is
   strictly above zero;
2. English fluency, grounding, adherence, abstention, safety, and every
   selected domain are noninferior under preregistered margins;
3. deterministic teacher-passing items do not regress beyond the locked
   tolerance;
4. no extra private data, evaluator leakage, or unaccounted source information
   explains the gain; and
5. the exact same checkpoint passes deployment and isolation gates.

The claim must always name the teacher, artifact revisions, suites, judge
protocol, margins, and confidence intervals. "Universally stronger than the
teacher" is prohibited.

## Mandatory competitors

Every promoted ABI route must face optimized, reproducible baselines:

| ID | Competitor | Required accounting and purpose |
| --- | --- | --- |
| T0 | Frozen teacher | Source-quality reference; all source/base bytes and inference costs count |
| L0 | Single-capability LoRA | Strong parameter-efficient adaptation baseline on the pinned source base |
| L1 | Routed or composed LoRA/adapters | Strong modular baseline; include base, router, every installed adapter, and active adapter cost |
| D0 | Sequence-distilled transformer student | Same teacher prompts and outputs; teacher absent at inference |
| D1 | Logit-distilled transformer student | Same prompt set and explicitly accounted logits; teacher absent at inference |
| D2 | Best justified distillation variant | Rationales, hidden states, or selected data are allowed only when identically accounted |
| A0 | ABI LayerCake | Full labeled, normalized, segregated capability-compiler route |
| A1 | ABI without destination labels | Tests whether labels create the claimed segregation benefit |
| A2 | ABI with shuffled targets | Tests causal dependence on teacher mapping |
| A3 | Bridge-only LayerCake | Tests whether the bridge relearned the capability |
| A4 | Monolithic unsegregated ABI | Tests whether packaging alone, rather than segregation, explains results |

The baseline implementation may use LoRA composition, adapter fusion, targeted
data selection, and modern distillation recipes. A weak baseline cannot support
an ABI-superiority claim.

## Fair-comparison views

Every headline result must be reported under all three views:

1. **Equal imported information:** identical source prompts and equivalent
   teacher-output/logit/activation budgets.
2. **Equal final deployment constraint:** matched total installed bytes, active
   bytes, peak memory, or latency envelope.
3. **Matched quality frontier:** cost required by each method to reach the same
   locked quality gate.

The ledger must separately record raw prompts, unique UTF-8 bytes, teacher
outputs, authoritative teacher tokens, logits, activations, source parameters,
copied parameters, trainable parameters, parameter-seconds, disk, RAM, VRAM,
wall time, energy when measurable, hardware, software revisions, and external
artifact-generation cost. "Small" cannot refer only to trainable parameters.

## Phased campaign

### Phase 0 - Definitions, baselines, and preregistration

Status: **OPEN**.

- Freeze the capability ontology, English boundary, quarantine policy, source
  identities, licenses, test splits, statistical methods, noninferiority
  margins, repetition rules, and resource accounting.
- Specify exact baseline implementations and optimization budgets.
- Freeze two scoreboards: common acquisition quality/cost and unique
  capability-control/deployment behavior.
- Require at least 100 distinct prompts for every promoted headline quality
  suite, paired prompt-level comparisons with 95% bootstrap confidence
  intervals, three paired training seeds or initializations, and at least 20
  repeated timing observations for every headline runtime configuration.
- Treat median bytes/second and characters/second as primary cross-model
  throughput metrics. Publish p95 only with at least 100 observations and p99
  only with at least 1,000; otherwise mark those statistics unsupported.
- Bind all future artifacts to hashes before results are observed.

Exit gate: a machine-readable protocol names every metric, threshold, split,
baseline, seed, stop rule, and artifact identity. Until then, no new training
run is promotion-eligible.

### Phase 1 - Capability inventory and normalized acquisition IR

Status: **LOCKED behind Phase 0**. V89 is supporting bounded labeling evidence,
not a Phase 1 pass.

- Convert raw teacher records into a versioned intermediate representation
  retaining raw and normalized forms, token IDs, provenance, destination,
  capability, license, transformation history, confidence, and rejection
  reason.
- Keep English, each domain, mixed content, unknown content, conflicts, and
  spoof attempts distinct.
- Resolve fact-free reasoning versus mathematics and factual domains.
- Qualify correctness, completion, diversity, deduplication, contamination,
  coverage, and at least 100 distinct passing records per required capability.

Exit gate: the immutable artifact passes adequacy and adversarial verification;
the untouched final suite has not influenced extraction or normalization.

### Phase 2 - Strong matched LoRA and distillation baselines

Status: **LOCKED behind Phase 1**.

- Train L0/L1 and D0/D1 plus only the strongest justified D2 under identical
  source prompts and explicit imported-information accounting.
- Tune each baseline on development data only and reproduce headline
  configurations across three paired seeds or initializations.
- Measure quality, leakage, composition interference, installed/active bytes,
  throughput, TTFT, memory, acquisition cost, removal, restoration, and
  portability.

Exit gate: each baseline is technically credible, converged within its locked
budget, and has raw evidence plus a verifier. ABI cannot claim superiority if a
required baseline is missing or intentionally under-tuned.

### Phase 3 - Causal teacher-to-LayerCake acquisition

Status: **LOCKED behind Phase 2**. V87 is a positive causal signal and formal
quality failure, not a Phase 3 pass.

- Train A0 alongside A1-A4 using shared artifact splits and matched host
  capacity.
- Require autonomous generation, not teacher-forced likelihood alone.
- Attribute failures separately to source adequacy, normalization, acquisition,
  bridge, LayerCake capacity, and decoding using the existing exact-host
  controls.

Exit gate: A0 beats parent, shuffled, bridge-only, label-free, and monolithic
controls with paired confidence intervals while passing the locked fluency,
grounding, adherence, coherence, abstention, and repetition gates.

### Phase 4 - Sufficient-information Pareto frontier

Status: **LOCKED behind Phase 3**.

- Run preregistered nested budgets for prompts, bytes, tokens, logits,
  activations, parameters, and bridge capacity.
- Keep evaluation, initialization, training policy, and architecture fixed
  within each comparison.
- Compare the ABI, LoRA, and distillation frontiers under the three fairness
  views.

Exit gate: the smallest passing tested ABI budget is paired with its adjacent
lower failure and reproduced across three seeds. No global-minimum claim is
allowed.

### Phase 5 - Selective reconstruction and bounded exclusion

Status: **LOCKED behind Phase 4**.

- Certify English-only behavior and abstention on withheld specialist probes.
- Install one domain at a time and prove selected recovery without core-byte
  mutation.
- Remove and exactly restore each domain; test cross-domain, conflicting,
  adversarial, and label-spoof cases.
- Compare residual unselected-domain behavior against the LoRA base and
  distilled student.

Exit gate: bounded exclusion, isolation, immutable core identity, package
  identity, and removal/restoration all pass on untouched evidence.

### Phase 6 - Composition, portability, and multi-source provenance

Status: **LOCKED behind Phase 5**.

- Compose multiple domains and, later, multiple pinned teachers without losing
  record-level provenance.
- Quarantine contradictions under a frozen policy.
- Prove exact transfer to independent compatible hosts and selected-only
  execution through physical traces.
- Measure interference and scaling against routed LoRA/adapters.

Exit gate: package bytes and installed payloads are exact, inactive packages
do not execute, quality does not collapse under composition, and deletion or
license lineage identifies every affected artifact.

### Phase 7 - Integrated teacher-relative and systems certification

Status: **LOCKED behind Phase 6**.

- Evaluate one final, teacher-free LayerCake checkpoint against T0, L1, and the
  best distilled student on the same functional and generative suite.
- Run CPU and GPU throughput, genuine cold TTFT, active memory, RSS, persistent
  state, and sparse-execution tests on that same checkpoint.
- Apply the product-superiority and optional teacher-improvement rules above.

Exit gate: all hard LayerCake integration gates pass, including the existing
  at-least-2x optimized CPU-transformer throughput requirement, without
  borrowing speed or quality from another lineage.

### Phase 8 - Independent hostile verification and release

Status: **LOCKED behind Phase 7**.

- Recompute all aggregates from raw evidence in a clean environment.
- Attack provenance, hashes, labels, package identity, teacher absence,
  routing, information accounting, final-set isolation, and claim language.
- Reproduce on independent hardware and seek external replication.
- Complete a separate prior-art and legal review before any patent or novelty
  representation.

Exit gate: a clean content-addressed release reproduces every promoted claim;
all negative evidence and limitations remain published.

## Hard decision rules

1. Stop a branch after its preregistered budget unless a profiler or causal
   ablation identifies a specific repair likely to improve the failed metric.
2. Never compensate for a weak ABI result with more teacher data, parameters,
   or compute without rerunning the matched baselines at the same budget.
3. If distillation matches ABI quality, cost, portability, isolation, and
   modularity, adopt distillation and narrow ABI to the proven labeling and
   packaging layer.
4. If routed LoRA wins and retaining its source base is acceptable for the
   product, use it. If base removal, bounded exclusion, or native LayerCake
   execution is required, report LoRA's quality win and its deployment-contract
   failure separately.
5. A quality gain caused by extra data, evaluator leakage, a larger host, or an
   uncounted base is not an ABI gain.
6. A result from one checkpoint cannot be combined with speed, memory, or
   isolation from another.
7. Preserve every negative result and superseded protocol.

## Immediate next action

Complete Phase 0 only: write the versioned evaluation protocol, choose the
exact optimized LoRA and distillation implementations, freeze the natural
quality/isolation suites, and set numeric noninferiority and superiority
margins. Then resume the already-required normalization and adequacy work as
Phase 1. Do not launch another teacher-to-LayerCake training variant first.
