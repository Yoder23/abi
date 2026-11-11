# R105: external-label semantic-scope evaluator falsification

Registered 2026-11-11 before model inference. The user asked to keep pushing
toward actual proof. This bounded prerequisite tests a materially different
evaluation mechanism, not another Qwen prompt, training sweep or regex patch.
R103/R104 remain immutable negative evidence; all English/control gates remain.

## Question and candidate

Can one frozen, conventional NLI cross-encoder distinguish semantic scope well
enough to be considered for future evaluation? Use only
`cross-encoder/nli-deberta-v3-base` revision
`6c749ce3425cd33b46d187e45b92bbf96ee12ec7`, its published tokenizer and label
order (contradiction, entailment, neutral), raw argmax of three logits, GPU FP32,
TF32 off, evaluation/inference mode, batch16, no truncation beyond512 tokens.
No learning, thresholds, ensemble, alternate model or retry after scores.

This is a third-party conventional evaluator, not ABI invention, an English
teacher, a deployable capability, a human rater or a LayerCake module.

## Frozen populations and limits

1. All 1,200 rows in the six ScoNe-NLI **test** files, 200 aligned contrast
   groups, at repository revision `b02532a2f4185c6118a57a148455e0750592d8c8`.
   Gold labels are external and pre-existing, not authored to fit our classifier.
   The test split is distinct from that repository's training split. It contains
   entailment/neutral only: it does NOT measure contradiction recall by itself.
2. 64 prospectively authored approval-scope cases: 32 noncontradictions and32
   contradictions, across two fictional project names. Include conditionals,
   negation, double negation, meeting-vs-approval scope, quoted/embedded claims,
   and assertions after conditionals. These formal expectations are author-
   specified, NOT independent human quality labels. They address the measured
   R104 scope errors without inserting fixes into a scorer after outputs.

The source model was trained on SNLI/MultiNLI; ScoNe derives from earlier NLI
materials. Lexical/source overlap and upstream exposure cannot be ruled out.
This is an external-label challenge, not a secret contamination-free benchmark,
new broad-English test or general semantic evaluator certification.

Both populations, all source bytes/license, model/config/tokenizer hashes,
code/tests/protocol and environment must be frozen and committed before inference.
Do not inspect predictions before that freeze or change labels afterward.

## Decisions

External scope gate: >=1,140/1,200 correct, >=190/200 in **each** of six families,
and a95% percentile cluster-bootstrap lower bound >=0.90. Resample the200
six-case groups10,000 times using seed105001. Report complete confusion counts,
per-family accuracy and all-six-correct groups. The group-level interval avoids
pretending correlated contrast examples are independent.

Approval-scope gate:64/64 Boolean contradiction decisions correct, including
32/32 for each class. No threshold tuning or majority/abstention fallback.
Both gates are required for `BOUNDED_EVALUATOR_CANDIDATE_PASSED`; otherwise
`EVALUATOR_CANDIDATE_REJECTED`. Even a pass does not qualify an English control
or certify fluency. It only supports a separately preregistered in-context
evaluation study. R104 is not rescored or promoted either way.

Run one fresh-process replay with the same inputs/batching regardless of outcome;
require raw logit/token/prediction JSONL to be byte-identical. Preserve resources
separately. Verification recomputes every prediction/aggregate from logits and
bound input/gold rows, checks all payload hashes and refuses missing/stale data.
Add hostile missing/hash/row/logit/label/decision/resource tests after completion.

## Budgets and attribution

At most1,264 pairs in the initial pass plus1,264 in replay;20min/process, no
other model inference. <=1GiB downloaded evaluator payload and <=50MiB evidence,
>=20GiB free initially. No paid API or external hardware, no source-teacher calls,
no recipient calls, no training, no model changes. Report downloaded bytes,
source licenses/revisions, parameters, input tokens, stored logits, GPU/RSS,
wall/CPU time and all failures. This evaluator is not imported into a student.

Failure belongs to evaluator suitability; it says nothing new about ABI
acquisition or native LayerCake quality/performance. LayerCake remains untouched.
Do not respond to failure with another model/threshold/prompt sweep. Publication
of bounded results remains authorized, without calling it a breakthrough.

## Primary sources

- [Model card](https://huggingface.co/cross-encoder/nli-deberta-v3-base):
  declared SNLI/MultiNLI training and three-class output; Apache2.0 metadata.
- [ScoNe repository](https://github.com/selenashe/ScoNe): externally authored
  contrast sets and field definitions, CC0 license.
- [ScoNe paper](https://aclanthology.org/2023.acl-short.154/): negation-scope
  benchmark, not our own new benchmark or proof of unrestricted reasoning.
