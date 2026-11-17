# R110 — evaluator scope and sentence-count adherence

Preregistered 2026-11-17. This request authorizes **registration, tests and
publication only**, not new model execution. The evaluator, challenge generator
and live driver are not yet implemented or execution-frozen. A subsequent
explicit execution request and the implementation freeze below are required.
Registration is not a scientific result. Never amend R109 or relabel it a pass.

## Motivation and questions

R109 completed 192 calls with exact scientific replay, but its panels scored
35/48 and 34/48 and only 32/48 paired outcomes passed against 44. Twenty raw
responses violated the requested two-to-three-sentence limit. Seven approval
flags included four negative-scope and three conditional-scope errors. Even
granting all seven flags a pass left only 37/48 paired outcomes. Preserve all
scores, the optimistic ceiling, raw outputs, checkpoints and release identities.

Ask two separate questions, in order:

1. Can a versioned approval-scope evaluator distinguish the registered syntactic
   cases without falsely accepting positive approval assertions?
2. On the same source, does one explicit, input-only response contract improve
   sentence-count adherence while preserving every other quality obligation?

This is prerequisite validity/interface work, not knowledge extraction, student
training, ABI novelty, a new English artifact or a LayerCake runtime experiment.

## Stage A — scope evaluator prerequisite

Leave R104, R108 and R109 code/data immutable. Implement a new evaluator module
under R110; never monkeypatch an old evaluator. Preserve R108's literal, case,
surface-form, numeric/day, word-limit, repetition, negated-pending and sentence-
count checks. Only approval-assertion scope classification may change.

The new classifier returns `NO_FORBIDDEN_ASSERTION`, `FORBIDDEN_ASSERTION` or
`UNRESOLVED`, with character spans and a syntactic reason for every relevant
clause. `UNRESOLVED` cannot pass. This finite approval-claim grammar is not a
general semantic or fluency judge. Unrecognized constructions within the
registered approval-claim vocabulary must be unresolved; recognizing every
possible English paraphrase is not claimed.

Required distinctions include: negation must apply to the approval claim itself;
conditional antecedents are not present-tense approval; a later assertion must
not be excused by an earlier conditional or negation; negated pending status and
guaranteed future approval remain forbidden; mixed/ambiguous scope is unresolved.
Do not exempt an entire answer on a keyword, remove a requirement, hard-code a
row ID/hash, or look up expected labels. Candidate scoring receives only the
response and ordinary task obligations, not test IDs, labels or file paths.

Development regressions: all 164 existing author fixtures retain their original
expected classifications, and all seven disclosed R109 scope cases must avoid
the false positive. Use those seven only as observed development cases, never
as fresh validation. Their historical R109 scores do not change.

Before challenge evaluation, publish **one** implementation freeze containing
the classifier, complete challenge grammar/generator, expected-label derivation,
case-balanced sampling rules, live driver, verifier and hostile tests. The
generator must encode expected labels from the construction, independently of
the classifier, and publish those rules before any challenge score. Its six
registered classes are direct assertions, scoped negations, conditionals,
conditionals followed by assertions, scope traps, and unresolved constructions.
Cross punctuation/conjunction boundaries, lower/title/upper case, subject forms,
formal-approval wording, and claim order. Put explicit unsafe claims after safe
clauses and vice versa. Include quoted/metalinguistic or ambiguous constructions
as unresolved, not automatic passes. No challenge family may be dropped.

Derive the challenge seed by the exact `design.challenge_seed` function from
the published implementation commit and registration SHA-256. Materialize and
publish 100 unique cases per class (600 total), their labels and hashes before
running the classifier on them. Require zero errors in **each** class, zero
false-safe assertions, and exact replay in a fresh process. An unresolved label
is an expected abstention, not an affirmative quality judgment. First challenge
failure stops R110 before GPU work; retain it and do not edit/retry the classifier
or draw another seed under this registration.

These are prospective, author-constructed syntactic tests, not an independently
blind or human-rated semantic validation. Public deterministic seeds are not
secret holdouts. No human-rating count may increase.

## Stage B — one fixed source-interface contrast

Only after Stage A passes and a separately authorized execution freeze exists:

- Use the exact R109 Qwen3-4B-Instruct-2507 snapshot and BF16/native-chat settings:
  revision `cdbee75f17c01a7cc42f958dc650907174af0554`, greedy generation,
  seed 108001, four CPU threads, TF32 off, context 1024, maximum 640 new tokens.
- Use the 96 inputs generated now by `design.population`: 48 counterfactual pairs,
  six pairs in each of eight tasks. These are new identifier/day instantiations
  of known templates, not new semantic tasks. Freeze both exact renderings and
  tokenizer token IDs before any inference. No response target is supplied.
- `plain` uses the existing native-chat user message. `response_contract` appends
  exactly the registered metadata block **only** for summarization and tone.
  It repeats existing required literals, pending status, two-to-three sentences,
  120-word cap and connected-prose format. It removes no note or requirement and
  supplies no answer, sentence plan, exemplar or teacher-generated content.
- The other 72 messages are byte-identical between interfaces and act as negative
  controls. Their output tokens must agree exactly across interfaces. Otherwise
  report an execution/conditioning discrepancy, not an interface benefit.
- Four fresh processes, fixed counterbalanced order: plain initial, contract
  initial, contract replay, plain replay. Each covers all 96 prompts in frozen
  order. Complete both interfaces and replays even if early quality looks poor;
  resource/integrity failure stops the study. No partial or best-of result.
- No output repair, sentence clipping, forced EOS, constrained decoder, extra
  turn, self-correction pass, new model, prompt sweep, training or receiver call.
  Raw outputs alone determine quality. Record old and new evaluator results on
  **every** new output; this separates scoring changes from interface changes.

## Unchanged quality gates and decision rule

Every interface independently needs at least 44/48 passes in **each** panel and
at least 5/6 for **every** task in each panel. At least 44/48 counterfactual pairs
must pass both members, with at least 5/6 pairs per task. Require zero collapse,
zero generation-budget stops and exact scientific replay (only timing excluded).
The sentence requirement remains two or three, the word cap remains 120, and
all literal/grounding/format/polarity obligations stay in force. Old-scoring
results are descriptive comparability data, never an edited historical score.

Primary interface endpoint: change in the unchanged sentence-count indicator
on the 24 sentence-limited rows (12 counterfactual-pair clusters). For each pair,
average the two within-row contract-minus-plain differences; bootstrap the 12
pair means with replacement for 10,000 draws, seed 108001, and report the mean
and two-sided percentile 95% interval. Support requires its lower bound > 0.
Do not use replay rows as additional independent samples. Report all-eight-task
quality effects with 48 paired clusters and the same bootstrap settings. For an
interface improvement claim, also retain the R109 noninferiority safeguard:
the all-task contract-minus-plain quality interval's lower bound must be >= -0.10.
This is a quality safeguard and descriptive effect, not a second superiority
endpoint. No endpoint substitution or selection after observing results.

An interface improvement requires that primary effect, the noninferiority
safeguard and all unchanged quality/replay/negative-control gates. If plain already qualifies without a
supported interface effect, report that narrow plain-control qualification only.
If contract qualifies without a supported effect, report the distinction and
do not promote an improvement claim. If neither qualifies, stop at prerequisite
failure. A bounded control qualification is not broad English or a recipient
artifact; no historical recipient gate is retroactively reopened.

Matched R99 LoRA/distillation histories, parent and full-model checkpoints remain
preserved. No new training, changed imported-information budget or student
comparison is authorized here. A future comparison must be prospectively
registered with those controls and exact accepted input/scoring conditions.

## Resource limits, custody and failure ownership

Offline evaluator validation: 1,800 seconds, zero model calls. If it fails, GPU
calls remain zero. Conditional source work: four processes at at most 14,400
seconds each, 57,600 seconds total process allowance, controller ceiling 59,400
seconds (16.5 hours), at most 384 generation calls and 245,760 generated tokens.
R109's two executions took 879.64 controller seconds; doubling that is roughly
29.3 minutes before contract overhead. This is a planning extrapolation, not a
guarantee or performance comparison. The conservative ceiling also allows the
previously observed slow hardware state; no automatic extension or retry.

GPU power, clock, cooling, driver and thermal settings remain untouched. Record
read-only telemetry. Require at least 10 GiB free before each stage/prompt and
at most 1 GiB new evidence. No downloads, new weights or optimizer states; use
verified caches. Abort if storage is inadequate, without deleting evidence.
Full source/tokenizer hashes must pass before any future inference. Record raw
prompts, input/output token IDs, outputs, evaluator spans and classifications,
actual calls, CPU/wall times, sampled RSS, CUDA peak and reuse of source artifacts.
Keep source preparation, live inference and offline verification costs separate.

The registration packet binds the design, thresholds, exact source input
renderings, tokenizer identity and the complete preserved R109 packet. A later
implementation/execution freeze must bind its own full dependency closure and
the registration, with input hashes rechecked before and after every process.
Missing packages, rows, hashes, replay or unrecomputable claims fail closed.
Every failure and scope limitation remains public, with immutable research tags.

Evaluator failure is evaluator failure. Source/adherence failure is an
experimental prerequisite failure. Neither tests ABI acquisition or native
LayerCake. No host code, checkpoint, interface or phase seal may change. No
English-core, minimum-information, source-removal, LoRA/distillation-superiority,
CPU/GPU-performance or full-moonshot certificate follows from this study.
