# R113 — storage-recovered count-feedback replication

Status: **CONTROL_QUALIFICATION_FAILED; complete exact replay** (2026-11-20).

All 576 new calls completed: 288 initial and 288 fresh-process replay. Scientific
fields match exactly; all 72 unchanged-input controls match in each worker. No
collapse or truncation occurred. No acquired English core or ABI/host breakthrough.

| Arm | Panel A / 48 | Panel B / 48 | Both-pass pairs / 48 | Sentence adherence / 24 |
| --- | ---: | ---: | ---: | ---: |
| Draft | 40 | 36 | 36 | 7 |
| Ordinary revision | 42 | 40 | 38 | 15 |
| Count-feedback revision | 43 | 43 | 40 | 18 |

The gate is 44/48 in each panel and paired, plus at least 5/6 in every task.
Feedback tone pairs score 3/6 and conversation 4/6. The primary sentence effect
is +0.125, paired 95% CI [-0.08333, +0.33333]: not supported. All-task quality
difference is +0.04167, CI [-0.01042, +0.10417]; relative noninferiority cannot
substitute for failed absolute gates.

[Decision](../results/count_feedback_r113/live/decision.json) SHA-256:
`6386bfb033dc421840758c794cdf655561e765dc84139ce73b643d0906b7c5e9`.
[Review guide](releases/R113_REVIEW.md).

The [post-result inventory](../results/review_2026_11_20/r113_error_audit.json)
records six feedback sentence failures, four unresolved scope outputs and one
forbidden flag (one row overlaps). The latter includes “Do not imply approval
has been granted” in `r112-031-b`: inspection suggests an imperative-scope false
positive, not an affirmative approval claim. Unresolved clauses concern waiting
for or requiring approval. These are disclosed evaluator limitations, not new
human ratings, corrected labels or rescored passes. Excess-sentence tone outputs
independently keep its task floor below 5/6.

Strict recomputation succeeds and the same-agent [raw-data hostile audit](../results/review_2026_11_20/r113_hostile.json)
rejects 14/14 mutations. This is not independent red-team review. Descriptively,
all 288 initial scientific rows also match R112's initial rows. That is continuity
of an exposed panel, not independent new data. R112 remains resource-incomplete.

Observed costs: source-process wall 2,514.64 seconds; controller loop 2,559.875
seconds (~42.66 minutes); CPU process time 2,345.156 seconds; 51,302 generated
tokens; peak sampled RSS 4,606,885,888 bytes; CUDA allocation 8,288,964,608 bytes.
These are not controlled speed comparisons. No downloads, training or recipients.

Next work requires a separate prospective design: falsify negated-instruction
and approval-prerequisite scope handling, then test a materially different
sentence-structured generation interface on fresh cases with unchanged gates.
Another scalar-count reminder, larger learner run or optimizer sweep is not
justified by this result. No automatic follow-on is authorized.

Source implementation: `5efaf9f`. Freeze SHA-256:
`54b113c52f9dbb1b3a344cee9a10567d3d46c157a9f06a0150c3e70144529bdc`.
The freeze binds 983 repository/host paths plus the complete pinned source cache.
Inherited scope recomputation and exact input-token identity pass. The freeze was
published at `abi-r113-implementation-2026-11-20` before launch.

Software checks: the final supported suite passes 829 tests with four existing
skips (including six publication checks), and all 16 historical Phase 7 tests pass in their exact
46-file fixture. The two direct Phase 7 invocations correctly rejected the newer
host's stale historical bindings; their failed receipts are preserved. No host
bytes, historical verifier or quality gate was changed. Focused execution tests
pass 133 checks; lint and documentation links pass. Receipts:
[final suite](../results/review_2026_11_20/r113-final-default-a1.xml) and
[exact historical fixture](../results/review_2026_11_20/r113-exact-a1-phase7.xml).
These are not model results.

[Protocol](../experiments/count_feedback_r113/PROTOCOL.md). This separately
authorized attempt repeats the frozen R112 comparison from the beginning in two
fresh processes. It does not resume R112 or splice its incomplete replay.
Prompts, model, evaluator, quality gates, GPU settings and budgets stay fixed.
The old panel is exposed; this is not fresh blind validation.

Storage cleanup recovered about 22.15 GiB without losing unique evidence:
1.45 GiB of verified public ZIP copies/disposable test fixtures were removed,
and two legacy token arrays were transparently compressed with identical hashes.
Source caches and Git were untouched. The local audit is outside both repos in
`storage_audit_20261120`; pre-inference telemetry, not this summary, controls launch.

R113 revalidated the inherited finite scope prerequisite, then performed 576
source calls including complete replay. Historical matched LoRA/distillation
checkpoints remain preserved. There is no training, recipient or native-host run.
Even a successful result could qualify only a conventional count-assisted workflow
on the known mechanical panel—not an English core, ABI novelty, or a breakthrough.
The whole-moonshot verdict remains `FULL_MOONSHOT_NOT_PROVEN`.

R112 remains [resource-incomplete](R112_COUNT_FEEDBACK.md). Every negative and
interrupted result stays independently addressable. No automatic successor.
