# R112 execution implementation

This additive implementation lives in `execution/` so the immutable top-level
registration source set is unchanged. The protocol, contract, prompt population,
known scope cases and original blocked preflights are not rewritten.

The finite text-only evaluator adds local recognition for the five disclosed
R111 construction gaps, with negative, mixed and ambiguous counterexamples.
Every non-approval observation and the sentence counter are unchanged. Unknown
scope is non-passing. The construction-labeled challenge generator imports no
classifier and performs no score-conditioned selection. Its 800 cases, 100 per
registered class with all subfamilies represented, are content-seeded from the
committed implementation and protocol hash. Development tests use disclosed
examples and check challenge structure only, not prospective challenge scores.

Before scoring, commit the entire implementation, write a single freeze binding
all source/tests, registration, full R111 manifest closure, source-cache hashes,
package versions and exact baseline input tokens, and publish it. Then materialize
and publish the actual challenge before its first scoring. All 1,598 development
obligations, 800 prospective cases and fresh-process byte-identical replay must
pass. The first failure closes GPU execution; no in-registration repairs/retries.

The two fresh source processes each execute 96 shared-draft triplets in fixed
order: draft, ordinary revision, count-feedback revision. Each revision always
uses the full original draft; neither receives the other revision. Count feedback
is exactly the registered sentence-count string on 24 summary/tone inputs only.
All 72 unchanged-input controls must match in prompt bytes, input tokens, output
tokens and complete decoded text. The verifier recomputes dynamic conditioning,
mechanical counts, old/new screens, termination, all absolute gates, paired
bootstrap contrasts, costs and exact scientific replay from all 576 raw rows.
Timing and storage observations alone are excluded from byte-exact scientific
replay; both workers' unchanged-input controls are checked independently.

Whole-study cost includes both revision arms and replay. Each workflow costs
draft plus its own revision; the shared draft is never called free. Record raw
input/output tokens and bytes, 50-ms sampled RSS, CUDA peak, CPU and wall time,
and read-only GPU telemetry. No training, downloads, recipient execution, host
change, source/quality/hardware-gate change or automatic follow-on is authorized.
All storage and time guards remain those in the protocol.

```powershell
python -m experiments.count_feedback_r112.execution.run freeze
# Commit/push implementation + freeze before materialization.
python -m experiments.count_feedback_r112.execution.run materialize
# Commit/push actual challenge before scoring.
python -m experiments.count_feedback_r112.execution.run scope
# Only after scope passes and its complete packet is committed/pushed:
python -m experiments.count_feedback_r112.execution.run live
python -m experiments.count_feedback_r112.execution.run verify
```

These are write-once stages. Missing, altered, incomplete or unrecomputable
evidence fails closed. Parser failure belongs to EVALUATOR; source/workflow
qualification failure to EXPERIMENTAL_PREREQUISITE; hard limits to RESOURCE.
ABI acquisition and native LayerCake remain NOT_TESTED. Even a passing workflow
is bounded mechanical qualification, not broad English or an ABI breakthrough.
