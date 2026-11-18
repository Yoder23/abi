# R111 — scope coverage and one unconditional revision

Preregistered 2026-11-18 before any new source execution. The user authorizes one
bounded study, preservation of R110, unchanged quality gates and GPU settings,
documentation and publication. No automatic successor or recipient run.

## Questions and evidence boundary

R110 passed its finite grammar challenge but left 14/24 plain/contract outputs
unresolved; sentence adherence was 6/24 and 8/24. Its final decision SHA-256 is
`b29a4855de20f1fd23d3712720492bfad2f15826e3273b773803b95c343c5fa2`.
Preserve all R110 code, outputs, scores, tags and package identities. Do not
retroactively rescore it into a pass.

1. Can a new finite scope grammar cover the observed construction gaps while
   rejecting affirmative, negated-status, mixed-scope and ambiguous controls?
2. Does exactly one unconditional source revision improve sentence adherence
   and meet the unchanged absolute quality gates on new known-template inputs?

This is a conventional self-revision workflow, not ABI novelty or acquisition.
It is not single-pass qualification, human fluency evaluation, native LayerCake
execution, or a superiority/efficiency comparison against LoRA or distillation.
Those historical matched controls and checkpoints remain untouched.

## Stage A: prospective finite scope prerequisite

Implement a new classifier; leave all old evaluators unchanged. Input is response
text only, not row IDs, labels, filenames, scores or reference answers. Preserve
all non-approval checks exactly. Unresolved always remains a non-pass.

New construction families: perfect negative predicates (including subjectless
conjuncts); contracted copulas; reduced conditionals; approval given/coming
through; approval-status and approval-for-noun pending forms; nominal/attributive
provisional status; and subject-to-final-approval predicates. Add en-dash clause
boundaries. Negation is local, conditionals cannot exempt later clauses, unsafe
claims dominate safe ones, and uncertain/quoted/metalinguistic clauses remain
unresolved. Never bless a response merely for containing `pending`.

Development obligations: all 164 old fixtures, seven R109 regressions, R110's
600 already-public challenge cases, and the disclosed R110 unresolved clauses
as known construction examples. These are not new validation or human labels.
The full generator and expected-label construction must be frozen before the
new challenge is materialized. It imports no classifier.

The new challenge has 100 globally unique cases in each of eight classes:
perfect negation, reduced/full new conditionals, nonfinal phrases, new affirmative
claims, negated-status traps, conditional-then-assertion, mixed safe/unsafe scope,
and unresolved ambiguity. Cross subjects, case, state words, clause boundaries
and claim order. Construction defines labels independently of scorer output.
The public seed is exactly `design.challenge_seed(implementation_commit,
SHA256(PROTOCOL.md))`. Publish the actual 800-case packet before scoring.
Require zero errors in every class and every development obligation, zero
false-safe assertions and byte-identical fresh-process replay. First failure
closes the GPU gate: no repair, retry, seed selection or dropped family.
This is author-constructed finite validation, not independent semantic proof.

## Stage B: fixed draft/revision comparison

Use the same pinned Qwen3-4B-Instruct-2507 revision, BF16, native chat, greedy
decoding, seed 108001, four CPU threads, TF32 off, and 640-new-token allowance.
Hardware power/clock/thermal settings remain unchanged. No downloads or training.

Use `design.population()` with seed 111001: 96 inputs, 48 counterfactual pairs,
six pairs in each of eight tasks. These are new identifier/day instantiations of
known templates, not hidden semantic families. Freeze the complete population,
baseline input token IDs, exact revision instruction and rendering code before
generation. The full user notes and requirements are retained.

Two fresh processes, initial then replay, each make 192 calls. For every prompt
in fixed order, first generate the native-chat draft, then ALWAYS generate one
revision using original request + entire decoded draft + the fixed instruction
in `design.py`. No scorer feedback, expected answer, sentence plan, few-shot
example or known-safe phrase is supplied. No branching on quality; no best-of
choice. The revision is the workflow's output even when worse than the draft.
Store both complete outputs and all tokens, prompts, scope spans and old scores.
No clipping, output rewriting, forced EOS, constrained decoding, retries or sweeps.

The **input envelope is 2048 tokens for both stages**, instead of R110's 1024,
solely to retain the full original request and draft plus all 640 output tokens.
This is a prospectively declared resource/interface change, not a quality-gate
change or inference-speed claim. Abort rather than truncate any context that
does not fit. This two-call workflow is not compute-matched to one draft; report
both costs. It cannot qualify a future recipient at the old one-call interface.

## Unchanged quality and analysis

Keep the exact R110 quality-gate dictionary: >=44/48 each panel and both-pass
pairs, >=5/6 each task per panel and pairs, zero collapse/truncation, literal
case, grounding, requested format, 3–120 words, and 2–3 sentences where required.
Exact scientific replay excludes only time/free-space observations. No copied
unchanged-input negative-control claim: unlike R110, every revision input changes.

Primary: revision-minus-draft sentence-count indicator on 24 sentence-limited
rows, averaged within 12 counterfactual pairs; 10,000 paired bootstrap draws,
seed 108001, two-sided percentile 95% CI, lower bound strictly >0 for supported
improvement. All-task quality uses 48 pairs and lower bound >=-0.10. Improvement
also requires every absolute final-output gate, no draft truncation/collapse,
and exact replay. Replay rows are not additional independent observations.
Unresolved-classification counts/effect are descriptive, not a replacement
primary endpoint. Report both R110 and new scoring on NEW outputs only.

If revision qualifies without a supported effect, report narrow two-call
mechanical qualification only. If draft alone qualifies, report only its own
scope. If neither qualifies, stop at prerequisite failure. No broad-English,
source-free core, ABI superiority or host certificate follows from any result.

## Execution, costs and stopping

Publish this preregistration, then complete implementation/tests and a full
freeze before the actual challenge. Publish passing scope evidence before GPU
execution. The freeze binds this code and tests, exact inputs, full R110 evidence
and inherited dependencies, source/tokenizer hashes and package versions. Verify
all hashes before/after workers. Missing/stale/raw-unrecomputable claims fail
closed. Write-once launch, raw rows, receipts, operation logs and stops.

Scope budget 1800 seconds. Each source worker <=14400 seconds; total controller
<=30600 seconds, <=384 calls and <=245760 generated tokens. At least 10 GiB free
before every stage/call; <=1 GiB new evidence. No deletions or automatic extension.
Resource errors are incomplete, not a quality failure. Record CPU/wall time,
50-ms sampled RSS, CUDA peak, all input/output tokens/bytes and read-only GPU
telemetry. Separate reused source preparation from execution and verification.
Cache vocabulary size once during validation while retaining every token-range
check; this is verifier bookkeeping, not a model/runtime speed claim.

Evaluator failure belongs to the evaluator. Source/adherence or revision failure
belongs to the experimental prerequisite. ABI acquisition and native LayerCake
are NOT TESTED. Preserve every outcome and publish the exact evidence and docs.
