# R105: external semantic-scope evaluator falsification

2026-11-11. **`EVALUATOR_CANDIDATE_REJECTED`**. No English control qualified;
no recipient, teacher-acquisition or LayerCake execution ran. The full moonshot
remains `FULL_MOONSHOT_NOT_PROVEN`.

## What this establishes

One frozen conventional NLI evaluator is unsuitable for the registered semantic-
scope screen. The result is reproducible, not an ABI breakthrough. Unlike R104's
author-written regression cases alone, the main challenge uses external gold
labels. It catches a major failure that a simple smoke test would miss.

The [protocol](../experiments/evaluator_scope_r105/PROTOCOL.md) and source were
committed before outputs. The [input freeze](../results/evaluator_scope_r105/freeze.json)
was committed at `65fd101`. One initial GPU pass and one fresh-process replay
used identical weights, tokenizer, batching and inputs. Their complete token,
logit and prediction files are byte-identical. No tuning, training, alternate
model or prompt retry followed the failure.

## Registered results

| External scope family | Correct / 200 | Required |
| --- | ---: | ---: |
| No negation | 193 | 190 |
| One negation outside relevant scope | 194 | 190 |
| One scoped negation | 4 | 190 |
| One scoped plus one unscoped negation | 5 | 190 |
| Two unscoped negations | 188 | 190 |
| Two scoped negations | 195 | 190 |

Overall: **779/1,200 (64.92%)**, below 1,140/1,200. The 95% bootstrap interval,
resampling 200 six-example groups, is **63.75%-65.92%**; its required lower
bound was 90%. No group had all six answers correct.

The separate approval-status panel scored **62/64**, below the required 64/64:
32/32 contradiction cases correct, but only 30/32 noncontradictions. For both
project names, it incorrectly treats an explicitly false approval claim as
contradicting the supplied pending-approval notes. These 64 expectations are
prospectively author-specified Boolean obligations, not independent human
fluency ratings.

The [descriptive paired audit](../results/evaluator_scope_r105/descriptive.json)
shows that each single-scoped family changes the gold answer relative to
no-negation in all 200 groups, while the evaluator keeps its no-negation answer
in 193/200. This is a measured failure pattern, not a claim about its hidden
reasoning mechanism. Example: not owning a dog does not entail not owning a
mammal, but the evaluator predicted entailment. Do not invert its labels on
these families after seeing this pattern; that would be post-hoc benchmark tuning.

## Sources and limits

Evaluator: [cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base),
revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`, 184,424,451 parameters,
GPU FP32, published three-class argmax. Its card declares SNLI/MultiNLI training.
This is an existing conventional model, not an ABI invention or English substrate.

External challenge: all six [ScoNe-NLI test files](https://github.com/selenashe/ScoNe/tree/b02532a2f4185c6118a57a148455e0750592d8c8/scone_nli/test),
revision `b02532a2f4185c6118a57a148455e0750592d8c8`. Gold labels are external and
pre-existing; no test examples were used for training here. Its scope contrasts
are described in the [ScoNe paper](https://aclanthology.org/2023.acl-short.154/).
The test files contain entailment/neutral labels only, so this panel alone does
not establish contradiction recall. ScoNe derives from earlier NLI materials:
source/lexical overlap and upstream exposure cannot be excluded. This is not a
secret contamination-free holdout. The upstream CC0 license is preserved.

Human campaign ratings remain 0/21,000. Neither external dataset authorship nor
this same-machine replay satisfies the campaign's human or independent-hardware
gates. R104 scores are not changed and its English-control gate remains closed.

## Measured cost and identity

Preparation downloaded 748,853,373 bytes (eight pinned evaluator files) in
46.89 seconds. Weights remain an evaluator dependency; **zero parameters** were
imported into LayerCake and no deployable capability artifact was created.

Each pass scored 1,264 pairs / 40,144 input tokens and retained 3,792 logits.
Initial/replay execution-body wall times were 6.71s/5.63s; CPU times 6.83s/6.19s.
These exclude library imports and preflight hashing and are not full acquisition
or CPU inference benchmarks. Peak CUDA allocation was 868,326,912 bytes for each;
sampled RSS was 1,060,245,504 and 1,060,548,608 bytes. Hardware: RTX3080 Laptop GPU.
Zero teacher calls, recipient calls, optimizer steps or new checkpoints.

- Freeze SHA-256: `893718abe540aca2335000636ac75f0f0b4376b86f6eef1bf7b97443a775e741`
- Decision SHA-256: `db2e63f37cde19b50adbd2400af57bd7115ab212d658ae594a86036dc2ea5273`
- Both raw-file SHA-256 values: `ce0db2ee33901f4de63d8eea790676882593e920dd43b21283f2930890b515a8`
- Model tensor SHA-256: `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`

## Audit failures preserved, not hidden

The [preflight repair](../experiments/evaluator_scope_r105/PREFLIGHT_A1.md)
records a nonexistent namespace initializer. No inference had run; the successful
freeze reused the exact unchanged catalog. The metadata 404/DNS failures are
also disclosed. No scientific settings changed.

The first completed-study hostile audit found that the frozen stage verifier
could run without the entire replay directory: it intentionally supports the
initial-only stage but was not sufficient as a completion attestation. The
[failed audit receipt](../results/review_2026_11_11/r105-completion-audit-v1.xml)
is preserved. The frozen runner, scores and outputs are unchanged.
Use the additive **full-study verifier** below. It requires initial and replay
raw data, receipts, operation logs, input identity and the existing decision,
then calls strict recomputation and compares the two raw files byte-for-byte.
It refuses to recreate missing completion evidence.

Final focused audit: 41 tests passed (16 preregistration plus 25 additive checks,
including 23 hostile cases and the preserved stage-verifier limitation).
The supported suite separately passed 338 default tests plus 16 exact historical
tests, with four declared skips. These are verification results, not quality passes.

## Recompute without running a model

```bash
python -m experiments.evaluator_scope_r105.run fetch-model
python -m experiments.evaluator_scope_r105.complete
python -m experiments.evaluator_scope_r105.describe
python -m pytest -q tests/test_evaluator_scope_r105.py tests/test_r105_completed_verifier.py
```

`fetch-model` restores only the pinned upstream evaluator files; the verifier
hashes them but does not execute inference. Never use `run verify` alone as a
complete-study attestation, and do not rerun `pilot` in its occupied directory.
The publication manifest lists all required repository assets plus the eight
external model-file hashes. No sibling LayerCake checkout is needed for R105.

The publication manifest binds 40 repository assets and eight upstream model
files. Its SHA-256 is
`1f651c13866164009630f523735a959e24284c230cc694b59e3a5de06a25e2f5`.
Public reconstruction remains a separate post-publication operation.

## What remains necessary

Do not install this rejected NLI model as the quality judge, invert labels,
lower thresholds, or begin another nearby evaluator/model sweep. A successor
must address the measured negation/proposition-scope failure on independent
challenge expectations before judging an English-control comparison. Grammar,
fluency, factual grounding and exact adherence need distinct measurements;
none can silently replace the locked joint acceptance gate.

This result does not localize an ABI acquisition or native LayerCake defect and
does not show that either project is obsolete. It prevents an unreliable judge
from manufacturing a positive result. The actual compact fluent-core acquisition
breakthrough, qualified control and recipient comparison remain unproved.
