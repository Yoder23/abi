# R114 review — scope and sentence-structured generation

Execution status: complete, **CONTROL_QUALIFICATION_FAILED**.
Whole moonshot: **FULL_MOONSHOT_NOT_PROVEN**.

## Frozen design and ownership

Read the [protocol](../../experiments/sentence_structure_r114/PROTOCOL.md),
[experiment guide](../R114_SENTENCE_STRUCTURE.md),
[implementation/input freeze](../../results/sentence_structure_r114/freeze.json),
and [finite scope result](../../results/sentence_structure_r114/scope/decision.json).

One bounded experiment: 96 fresh instances of known task templates, 48
counterfactual pairs, eight tasks, shared draft plus ordinary and sentence-array
revisions, two fresh worker processes and 576 total source calls. No independent
semantic-family holdout or general-English certificate. Ordinary revision and
structured revision are identical on 72 inputs; their greedy tokens must match.
Both workflows include the cost of their shared draft. Whole-study costs include
all arms and replay. Repeated deterministic outputs are not independent data.

The conventional codec is part of the workflow: parse the entire JSON response,
validate every string and join all unchanged sentences with one space. Invalid
serialization fails; no clipping, rewriting, retries, best-of or answer supplied.
Raw envelopes, delivered payloads, tokens and codec evidence are retained.
This does not establish native prose generation or ABI novelty.

All historical gates remain: 44/48 per panel and paired, 5/6 every task, zero
collapse/truncation, exact facts/literals/format, 3–120 words and two-to-three
sentences where required. Primary CI uses 12 paired sentence clusters; all-task
quality uses 48. Both retain 10,000 bootstrap draws, seed108001. Relative gains
cannot replace failed absolute qualification.

R113 remains closed and unchanged; R112 remains incomplete. Historical matched
LoRA/distillation checkpoints are preserved but not newly executed. Acquisition
and native LayerCake remain NOT_TESTED; no training, host change or English
artifact. A later learning comparison must match representation, information,
tool access and full call costs. GPU power/clock/thermal settings stay unchanged.

## Prerequisite and software evidence

The published finite scope challenge passes all 800 new construction cases and
2,411 disclosed development obligations, zero errors and false-safe outcomes,
with byte-identical fresh-process replay. This is finite author-constructed
validation, not general semantic certification or independent human judgments.
The ambiguous `as approval is confirmed` clause remains explicitly unresolved.

## Live-output limitations to retain

Both workers are complete and agree exactly on scientific fields. Strict raw
recomputation passes; every count below uses the frozen scores, not new labels.

| Arm | Panel A /48 | Panel B /48 | Paired /48 | Summary pairs /6 | Tone pairs /6 | Sentence adherence /24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Draft | 41 | 38 | 38 | 2 | 0 | 7 |
| Ordinary revision | 44 | 45 | 44 | 6 | 2 | 18 |
| Sentence-array revision | 41 | 40 | 39 | 2 | 1 | 12 |

Every other task has 6/6 paired passes in all arms. Ordinary revision reaches
the aggregate floors but misses the unchanged tone floor of 5/6. Aggregate-only
reporting would conceal that failure. All arms have zero collapse and truncation.
All 72 unchanged-input controls match within each worker. Replay compares tokens,
prompts, payloads, codec observations and scores exactly; timing/storage fields
differ, so the full source JSONL files are not byte-identical.

The primary structured-minus-ordinary sentence effect is -0.25, 95% CI
[-0.58333, +0.125]: no supported improvement. All-task row-quality difference is
-0.08333, CI [-0.16667, -0.01042], failing the -0.10 noninferiority margin. These
are row outcomes averaged within 12 and 48 paired clusters, respectively, not
confidence intervals for the both-pass counts. The registered 10,000 draws and
seed108001 are unchanged. No arm qualifies and no workflow is promoted.

The replayed outputs expose scope coverage gaps despite the finite
challenge pass. `r114-012-b` structured says `no changes are confirmed`, but the
classifier labels its `are confirmed` predicate forbidden. This is an apparent
quantified-negation false positive, not evidence that the source asserted
approval. `r114-011-a` structured (`provisional arrangements pending approval`),
`r114-012-a` structured (`no confirmation of approval has been issued`), and
`r114-043-a` revision (`this remains subject to final approval`) remain unresolved.
These qualitative observations are not new gold labels or corrected scores.
Keep the frozen decisions; general live-prose scope validity is not established.

Twelve structured responses (seven summary, five tone) contain one JSON string holding several
sentences, rather than the required two or three one-sentence strings. The
strict codec rejects these arrays and delivers no payload. Do not split the
strings after the fact, count their latent text as valid delivered output, or
change the registered codec. This is a source/interface serialization failure,
separate from evaluator coverage and unrelated to unexecuted ABI/LayerCake paths.
Ordinary revision has six genuine tone sentence violations plus one separate
unresolved-scope row. Resolving a scope concern would not erase those violations.
The [descriptive audit](../../results/review_2026_11_20/r114_error_audit.json)
retains every failed output and its original score.

## Measured costs and custody

The source controller took 2,746.422 seconds (45.77 minutes). The two workers
used 2,717.734 wall seconds and 2,508.422 CPU-process seconds. These are this
same-machine study's observed costs, not isolated runtime or training benchmarks.
Peak 50-ms-sampled process RSS was 4,642,574,336 bytes (4.324 GiB); peak CUDA
allocated memory was 8,344,542,720 bytes (7.771 GiB), not total GPU device usage.

All arms and replay together used 576 calls, 199,296 input tokens, 52,442 output
tokens, 934,470 input UTF-8 bytes, 190,182 raw output bytes and 183,694 delivered
payload bytes. Across initial and replay, workflow generation totals are 1,730.444 seconds
for draft plus ordinary revision and 1,734.237 for draft plus structured revision.
They each include the shared draft; do not sum them as disjoint study costs.
The complete existing source cache was reused, with zero new download bytes,
training steps or recipient calls. Scope and verification overhead is separate
in the operation/strict receipts. No GPU setting or resource floor was changed.

Decision SHA-256: `b18d9692e22ec6d8c8740ded5db54b4c838ac0d4a304fd10daa00e5a3c24b59c`.
The [frozen decision](../../results/sentence_structure_r114/live/decision.json)
binds both raw files, receipts, hardware observations and controller operations.
The [hostile audit](../../results/review_2026_11_20/r114_hostile.json) rejects all
14 actual-row mutations, including missing rows/tokens, forged prompts/scores/
payloads, invalid codec/cost/storage observations and reordered identities. This
is a same-agent targeted check, not independent blind review or semantic truth.

## Next decision point, not an authorized successor

The observed one-string arrays isolate a serialization-contract problem. A future
proposal could test a materially different structural decoder rather than
another wording/reminder sweep, but would need a new preregistration, unchanged
quality gates and full cost accounting. Any constrained interface must apply
equally to later learning controls and cannot supply semantic answers. In
parallel, scope validation needs prospective coverage of quantified negation and
the newly exposed noun/prepositional constructions. It must not relabel these
rows as a repaired R114 pass or claim broad validity from more authored fixtures.
Do not launch acquisition or alter LayerCake based on these prerequisite results.

## Software checks and identities

The combined offline suite passes 920 tests with four existing skips; the exact
historical-host fixture separately passes 16. Failed pre-freeze parser-regression
and incomplete test-fixture receipts are retained, not hidden. Tests and mutation
checks do not substitute for model quality or independent scientific review.
The post-run focused verifier/publication suite also passes all 91 tests in
92.65 seconds; see [its receipt](../../results/review_2026_11_20/r114-final-a1.xml).
Documentation link checks and scoped lint pass. GitHub software checks passed
on the preregistered implementation and scope publication; they are not a new
scientific execution on independent hardware.

Preregistration: `20bafbb8fce0279954ef8bd05125f1fed34bb224`.
Scientific source: `ab161c4f734799adae7e69d305a8a0aeb8bdbe94`.
Freeze SHA-256: `69ffc52a25b63bbbfb4173960bc1c8f3c9db6dc56ca08a0aac6a3012c92c1e01`.
Challenge SHA-256: `e855fb6ba904e51011ad0f3ec8322a5271c91d8f4f3c963305f50c885bb7afde`.
Scope raw SHA-256: `d96fe902d2bcd72d290507356970ab9b5e95214dcbd01fc1ee6f9d9e4a30da51`.
Implementation/input freeze and actual challenge were separately published before
scoring; the full passing scope packet was published before source execution.

## Recompute the completed result

Use the exact Git history, required Git LFS payloads, host dependencies and pinned
source cache identified in the final manifest. An evidence ZIP is not a
self-contained source-weight archive. Missing dependencies fail closed.

```powershell
python -m experiments.sentence_structure_r114.run verify
python scripts/check_docs.py
python scripts/run_review_tests.py
```

The review runner uses the exact historical host fixture for old Phase 7 checks,
not today's sibling files. Do not rerun write-once freeze/live stages over existing
evidence. No automatic successor or whole-moonshot promotion follows from this run.
