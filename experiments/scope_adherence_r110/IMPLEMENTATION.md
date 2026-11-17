# R110 execution implementation — prospective freeze

The user authorized implementation, a published freeze, and execution conditional
on the unchanged storage/evaluator gates on 2026-11-17. The earlier registration
and its `PROTOCOL.md`, code, tests, population and renderings remain immutable.
This document operationalizes that registration; it changes no hypothesis,
quality threshold, source, prompt, budget, endpoint or claim ceiling.

## Evaluator boundary

`evaluator.classify(response)` accepts text only. It has no access to row IDs,
expected labels, catalogs, filenames or hashes. It returns three-way approval
classification and exact character spans, clause spans and syntactic reasons.
The registered vocabulary is approval, approved, confirmed, granted, pending,
provisional, unconfirmed and awaiting. This is a finite grammar, not an English
semantic parser, a fluency judge or proof of exhaustive contradiction detection.

Clause boundaries are punctuation, newline, em dash, and the listed conjunctions
`and`, `but`, `however`, `although`, `so`. Conditional scope is deliberately not
carried across them, preserving historical conjunction-trap obligations.

| Construction | Treatment |
| --- | --- |
| Direct copular approval, granted/arrived approval, emphatic approval, status yes | Forbidden assertion |
| Guaranteed future approval; negated pending/provisional status | Forbidden assertion |
| Listed conditional + registered subject + approval predicate | Safe only within that span |
| Nothing/no-subject approval; claim-local not/not-yet/contracted negation | Safe within the matched span |
| Affirmative pending status, awaiting/depend-on approval, bare status labels | Safe within recognized forms only |
| Approved budget/amount/cost as a noun modifier | Not itself an approval assertion; other task checks remain active |
| Quoted/metalinguistic/uncertain/reported/nested-negative clauses | Unresolved; cannot pass |
| Unconsumed registered approval vocabulary | Unresolved; cannot pass |

An explicit forbidden span dominates unresolved/safe spans for the response;
otherwise unresolved dominates safe. There is no answer-wide exemption triggered
by a conditional, negation or the word pending. `screen` retains all R108
non-approval checks exactly and replaces only approval-scope classification.
Legacy approval evidence and the complete legacy screen remain descriptive
fields on new raw observations. No historical score is recomputed as a new pass.

## Label construction and sampling

`challenge.py` imports no classifier. Its labels follow construction:
direct claims, conditional-then-positive claims and scope traps are forbidden;
scoped negatives and conditionals are safe; quoted, metalinguistic, uncertain,
reported and nested-negative claims are unresolved. These are author labels,
not independent human adjudication. Conjunction cases follow the conservative
registered clause rule, not a claim about every linguistic interpretation.

The source lists the complete subfamilies before sampling. It crosses six
subjects, approved/confirmed predicates, punctuation/newline/conjunction
boundaries, case transformations, conditional introducers, formal-approval
wording and safe/unsafe claim order. Each subfamily's unique candidates are
sorted then shuffled by the content-derived seed. Round-robin sampling selects
exactly 100 unique texts per class and includes every subfamily. Global duplicate
texts, insufficient candidates or a missing class fail closed. No scorer output
participates in construction, sampling, labels or seed selection.

The seed is exactly the preregistered `design.challenge_seed` of the implementation
code commit and registration SHA-256. Structural generator unit tests may use
dummy commit identities but never score those generated cases. The only semantic
development tests before the freeze are explicit development examples, 164
historical fixtures and the seven disclosed R109 regressions. They are not fresh
validation. Materialize, commit and push the actual challenge before scoring it.
No failed challenge permits a repair, replacement seed or second attempt in R110.
The registered fresh-process replay verifies the same run; it is not a retry.

## Custody, execution and stopping

1. Commit the complete implementation and tests. `freeze` binds this commit, the
   immutable registration, its historical dependency closure, cached source
   inventory, package versions, and every implementation file. Publish the freeze.
2. `preflight` records free space without source execution. It is allowed to
   report an unmet floor without consuming a scientific attempt. It deletes
   nothing. The experiment cannot start below 10 GiB free or above 1 GiB evidence.
3. `materialize` requires the storage floor and a published implementation freeze.
   Commit/push its 600-case challenge packet before `scope`.
4. `scope` creates one write-once launch and two fresh child processes. Each
   scores 164 historical fixtures, seven known regressions and all 600 challenge
   cases. It checks storage before each item. Both raw streams, timing/storage
   receipts, child logs and complete/error records are retained. The total
   offline process allowance is 1,800 seconds. Verification rederives every
   classification and mismatch from raw text; stored pass fields are not trusted.
5. A failed scope challenge closes the GPU gate. Missing evidence, timeout or
   execution error is incomplete, never a passed evaluator. Publish the actual
   result. Do not proceed to the source study after a failed/incomplete prerequisite.
6. Only a complete scope pass permits `live`, after its evidence is committed and
   published. Four fresh source processes use the exact registered order, source,
   inputs and decoding. Check source/dependency hashes before and after every
   process and free space before every prompt. No early quality stopping or
   adaptive prompting. Each arm/repeat has a write-once directory and receipt.
7. Record complete decoded outputs and exact tokens; the scoring view uses only
   the historical `.strip()` normalization. No output rewriting or clipping.
   Recompute both scores, unchanged-input token agreement, paired bootstrap
   intervals and all absolute gates. Raw scientific replay excludes only resource
   observations (timing/free-space), not prompts, tokens, text, scores or identity.
8. Any post-launch resource/integrity error writes a stop record and closes that
   attempt. Do not silently resume, retry, extend budgets or overwrite evidence.

The live controller allows 59,400 seconds, four process caps of 14,400 seconds,
384 calls, 245,760 generated tokens, and no downloads/training/recipient execution.
It measures process/CPU/generation time, sampled RSS at 50 ms intervals, CUDA
peak allocation, input/output tokens and UTF-8 bytes. Source preparation remains
the reused historical cost, separate from new hashing and execution. Read-only
GPU telemetry brackets each source process. There are no hardware setters.

Launch commands check that the current commit and necessary artifacts are on
remote `master`; worker processes use offline model access. Verification can be
performed offline from exact dependencies. An implementation tag is not a
scientific result, and software tests are not challenge certification.

## Commands (ordered, not an automatic pipeline)

```powershell
python -m experiments.scope_adherence_r110.run freeze
# Commit/tag/push the freeze before any challenge.
python -m experiments.scope_adherence_r110.run preflight
python -m experiments.scope_adherence_r110.run materialize
# Commit/push challenge.json before scoring.
python -m experiments.scope_adherence_r110.run scope
# Stop if scope fails; otherwise publish its full evidence before live execution.
python -m experiments.scope_adherence_r110.run live
python -m experiments.scope_adherence_r110.run verify
```

If storage never reaches the floor, publish the freeze and blocked preflight;
report the experiment as unexecuted, not failed quality or a completed R110.
Evaluator failure belongs to the evaluator. Source-control/adherence failure
belongs to the experimental prerequisite. Neither measures ABI acquisition or
native LayerCake. Matched historical controls and LayerCake remain unchanged.
