# R112 — count-feedback revision and remaining scope constructions

Preregistered 2026-11-19. One bounded successor to preserved R111, authorized by
the user. All quality thresholds and GPU power/clock/thermal/driver settings stay
unchanged. No automatic successor, training, recipient or native-host change.

## Evidence and hypotheses

R111 completed with exact 384-call replay. Its revision improves sentence
adherence 7/24 to 15/24, but nine revised outputs still fail the sentence count;
revision quality is 41/48 paired against 44. Three scope-unresolved outputs remain
in each arm. Its decision SHA-256 is
`2a26b2ec4751e81835a7d0c47fa2aed9ad96786f7b710247a48516b52e7f22ce`.
Do not rewrite, rescore or retry R111. Its public manifest SHA-256 is
`13a93229d5d9f13f19df38a8eeb60017d57e6a6ebc42ccd226647304287e0f76`.

1. Can a separately frozen finite parser cover five observed construction gaps
   without accepting affirmative or ambiguous assertions?
2. Does supplying the deterministic sentence count of a draft improve one
   revision over the identical self-revision request without that count, and
   clear every unchanged absolute quality gate?

This tests a **conventional, counter-assisted two-call workflow**. Unlike R111,
the intervention explicitly receives one mechanical evaluator statistic. This
is not evaluator-independent generation, ABI novelty, learned English, single-pass
qualification, training efficiency, or superiority over LoRA/distillation.
Historical matched checkpoints remain preserved, not rerun. A later learner
comparison would need the same disclosed tool/data/call budget for every arm.

## Stage A: finite scope prerequisite

Implement a new parser version without changing historical evaluators. Preserve
every non-approval check and the sentence counter exactly. Parser input is text
only: no prompt IDs, reference labels, filenames, or known successful outputs.
Unresolved remains non-passing; unsafe dominates safe; negation and conditionals
remain local, never exemptions for later independent assertions.

Cover these observed families: bare nominal nonfinal predicates (`with
arrangements provisional`); `not yet been` perfect negation; approval-for-noun
phrases with a provisional modifier; procedural approval-process mentions without
a granted/confirmed assertion; and subject-to-change-until-approval conditions.
Add explicit negated-nonfinal, affirmative, mixed-scope, quoted and metalinguistic
counterexamples. Do not bless entire clauses merely because `pending` occurs.

Known development obligations comprise all 1,593 R111 scope cases (including its
793 development obligations), plus the five deduplicated unresolved R111 clauses
with inspected author labels. These are disclosed examples, not fresh validation
or independent human judgments. Preserve their source row/stage provenance.

Freeze the parser, generator, verifier, tests and complete input/dependency
closure before materializing a new 800-case challenge. The generator imports no
classifier and samples no cases based on scores. Eight classes named in
`design.contract()`, 100 globally unique cases each, all registered subfamilies
represented; expected labels arise from construction. Derive the public seed by
`design.challenge_seed(implementation_commit, SHA256(PROTOCOL.md))` and publish
the actual packet before scoring. Require every new and development case correct,
zero false-safe outcomes, and byte-identical fresh-process replay. The first
failure closes GPU execution: no repair, retry, seed selection or dropped class.
This is finite author-constructed validation, not general semantic certification.

## Stage B: matched revision contrast from a shared draft

Use the exact R111 source identity/settings: Qwen/Qwen3-4B-Instruct-2507 revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, BF16, native chat, greedy, seed108001,
four CPU threads, TF32 off, 640 new tokens, 2048-token total context envelope.
No downloads, weight changes or hardware-setting changes. Abort rather than
truncate an input. Reuse and hash-verify the pinned source cache.

Use 96 prompts from `design.population()` (seed112001), 48 counterfactual pairs,
six pairs per eight known tasks, new N112 identifiers/day instantiations. These
are not hidden semantic-family holdouts. Freeze all prompts and exact draft
input tokens before inference; all dynamic inputs must later be recomputable.

Two fresh processes, initial then replay. For each prompt in fixed order:

1. Generate one raw draft.
2. Generate the R111 unconditional revision from original request + complete
   decoded draft + exactly the R111 revision instruction.
3. Independently generate the count-feedback revision from that **same draft**
   and same instruction. Only on the 24 summary/tone inputs append exactly:
   `Observed sentence count in the draft: {count}. Required sentence count: two or three.`

The exact leading whitespace and rendering are in `design.py`. The count is
computed with the unchanged R104 mechanical sentence counter, not supplied by
the model. It is supplied regardless of whether the draft already meets the
range. No semantic, literal, scope, per-row pass/fail or reference-answer feedback;
no sentence plan, examples, safe phrase, output clipping/repair, forced EOS,
constrained decoding, adaptive retry or best-of choice. Each arm always emits
its own revision, even when worse. Never feed the baseline revision into the
feedback arm. Keep all raw outputs and tokens, complete conditioning, counts,
new scores and frozen R111 scores on **new outputs only**.

On the other 72 inputs, the two revision requests must be byte/token-identical
and greedy outputs must match exactly in each repeat. This is a negative control,
not 72 extra independent observations. Exactly 288 calls per worker, 576 total.
Each proposed workflow costs draft + its own revision; do not charge only the
second call or count the shared draft as free. Whole-study cost includes both
arms and replay. No new LoRA/distillation or LayerCake experiment is authorized.

## Unchanged analysis and decision gates

Use the exact R111 quality-gate dictionary: >=44/48 each panel and both-pass
pairs; >=5/6 every task per panel and paired; zero collapse/truncation; exact
literal case, grounded facts, requested format, 3–120 words, and 2–3 sentences
where requested. Any raw draft truncation/collapse prevents workflow qualification.
Never remove a task or reinterpret unresolved as passing.

Primary contrast: count-feedback minus ordinary revision sentence adherence on
24 sentence-limited rows, averaged within 12 counterfactual pair clusters.
10,000 paired bootstrap draws, seed108001, percentile 95% interval; lower bound
strictly >0 for a supported gain. All-task quality uses 48 clusters and lower
bound >=-0.10 for noninferiority. Both use the same frozen paired method as R111.
Unresolved counts are descriptive, never a substitute primary endpoint.

Promote a bounded count-assisted mechanical workflow only if every absolute
quality gate, scope prerequisite, 72 unchanged-input controls and exact scientific
replay pass. A supported-improvement verdict additionally requires the primary
gain and noninferiority. If absolute quality passes without supported gain,
report qualification only, not improvement. If only ordinary revision or draft
qualifies, report only that arm's narrow scope; never select outputs row by row.
Otherwise report control qualification failure. No broad-English or ABI/host
certificate follows from any outcome. Replay is not independent new data.

## Budgets, custody and stopping

Publish this registration before implementation freeze. Publish implementation,
then the actual challenge before scoring, then passing scope evidence before GPU
execution. Bind all scientific code/tests, registration, inputs, dependency and
source hashes, package versions and the full R111 public manifest closure.
The verifier must recompute from raw rows and reject missing/stale/unrecomputable
evidence. Keep write-once stage launches, operations, receipts and stop records.

Retain the existing 10 GiB free-space floor before every execution stage and
call, 1 GiB new-evidence cap, 1800-second scope budget, 14400 seconds per source
worker and 30600-second controller budget. Cap generation at 576 calls and
368640 output tokens. Record CPU/wall time, 50-ms sampled RSS, CUDA peak, input
and output bytes/tokens, both workflow costs and read-only GPU telemetry.
No deletion, automatic extension, download or hardware modification is authorized.

If storage is below the floor, low-footprint registration and an explicit
blocked-preflight receipt may be written, but do not run the scope challenge or
models, create model payloads, or claim a complete implementation freeze. Existing
experiments and checkpoints remain untouched. Restoring space is not permission
to skip implementation tests, publication, scope validation or any quality gate.

Resource stops are incomplete (`RESOURCE`), not quality failures. Scope failure
belongs to `EVALUATOR`; source/workflow failure to `EXPERIMENTAL_PREREQUISITE`.
ABI acquisition and native LayerCake are `NOT_TESTED`, with no English artifact.
Keep the full moonshot `NOT_PROVEN`; publish negative and blocked outcomes too.
