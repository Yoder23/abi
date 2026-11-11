# R106: separate grammatical discrimination from autonomous English generation

Registered 2026-11-11 before new model scoring. This is a zero-training diagnostic
after R105 rejected a semantic judge. It does not repair, replace or pass the
R103/R104 free-generation contract. Those 44/48 and 5/6 gates remain closed.
No new NLI model, prompt search, threshold sweep or historical rescoring.

## Question and measurement

Can the retained recipients discriminate grammatical English, and does the
Task/Response wrapper change that discrimination? This measures conditional
sentence likelihood, NOT autonomous generation, grounding, instruction following,
fluency, semantic-scope evaluation, extraction quality or an English product.
Every target token is supplied. A positive result cannot qualify the control for
the blocked free-generation matrix or substitute for any campaign gate.

Use four externally authored BLiMP paradigms, fixed before downloading predictions:
`anaphor_number_agreement`, `determiner_noun_agreement_1`,
`irregular_past_participle_verbs`, `regular_plural_subject_verb_agreement_1`.
Source revision: `3e56b06fcabca9b30822fc66435fca6b1aa40bb1` of
<https://github.com/alexwarstadt/blimp>. Preserve all four 1,000-row source files,
the README and CC-BY attribution. Select by ascending SHA-256 of
`r106:106001:<family>:<pairID>`: first 50 per family for qualification, next 100
for comparison. No filtering by scores, sentence lengths or ease. Fail on duplicate
IDs/pairs across the selected panels, missing fields or over-256-token sequences.
The benchmark is public and may have appeared in source pretraining; no hidden
generalization or new human-review claim is permitted.

## Fixed score and interface

One pinned Qwen2-7B-Instruct control (same revision and complete payload as R104),
and four unchanged research recipients: R76 parent, R100 full-model control,
R99 FP32 LoRA and adapter-distillation checkpoints. These controls have different
historical acquisition and parameter scales; they are not newly matched training.
No new weights, optimization, teacher outputs, logits imported into recipients,
or native LayerCake code changes.

User request: `Write one grammatical English sentence.` Native recipient prefix
is that request plus newline; native Qwen uses its existing system/user chat
template with system `Write English text as requested.` Benchmark prefix is
`Task: rewriting\nWrite one grammatical English sentence.\n\nResponse:\n` for all.
Recipient route is fixed to `rewriting` (index 6), an acquired task shared by the
retained controls; this is not a route search or a test of all installed cakes.
The task differs from their supplied-notes training distribution. Any failure
therefore concerns these checkpoint/context/route combinations, not host impossibility.

Tokenize prefix and candidate separately with no automatic special tokens, then
concatenate. Score every candidate token at its previous causal position. No EOS
score, truncation, length normalization, best-of selection or generated answer.
Primary score is sum of target logits minus log-sum-exp normalizers (float32
normalization, Python stable sum). A pair is correct only if good > bad; ties fail.
Record all IDs, target logits, normalizers, token counts and both strings.
Also report mean-token-score accuracy and shorter-token preference descriptively;
neither can rescue the primary score. Tokenizers/lengths differ across models.

Before model work require independent toy-probability, causal-shift, tie, swapped-
pair, malformed/nonfinite and missing-evidence tests. During every forward compare
per-token log-probabilities with independently computed negative cross-entropy
(absolute tolerance 3e-5). Store maximum disagreement; fail if exceeded.
These checks validate scoring mechanics, not general semantic judging.

## Control and conditional comparison

Control qualification uses only native Qwen on 200 pairs: at least 190/200 and
45/50 in EACH family. Run a fresh-process byte-identical raw replay regardless
of result. Failure ends R106 before any recipient calls. This is explicitly
`GRAMMATICAL_DISCRIMINATION_CONTROL`, not qualified fluent English.

If it passes, run all five checkpoints on the separate 400-pair comparison panel
under both fixed interfaces. Replay every cell in fresh processes. The comparison
native control must also reach 380/400 and 90/100 per family before relative
recipient attribution. The two primary contrasts are native-minus-benchmark for
parent and full-model control. Report paired 95% and 97.5% percentile intervals,
10,000 stratified-within-family resamples, seed 106001. Positive 97.5% lower bound
supports interface sensitivity for that checkpoint, conditional on these four
fixed paradigms only. Template/lexical dependence limits wider inference.
Report all ten cells; other interface and control-relative contrasts are descriptive.
A recipient's narrow discrimination screen requires the same 95%/90% floors and
a control-relative paired 95% lower bound >= -0.10. Never call it a product pass.

## Execution, custody and attribution

Freeze/commit protocol, scorer, tests, populations, external data, inherited ABI
loader/host source, all four checkpoints/tokenizers, Qwen payload and versions.
GPU FP16 Qwen; GPU FP32 recipients; batch one; TF32 disabled; eval/inference mode.
Physical recipient traces must show route 6 and its six adapters. Bind before
and after execution. Keep resource timings separate from deterministic raw rows.
Recompute aggregates from raw numeric evidence, not stored success flags.
Completed verification requires BOTH raw files, BOTH receipts and all successful
process operation records; it must reject absent replay directories, missing
packages/hashes/rows, stale decisions, unexpected extra stages and nonfinite values.

Bound: 90 minutes total pilot, 20 minutes/process, <=16,800 sentence forwards
(qualification plus comparison, including full replay), <=250 MiB new evidence,
>=20 GiB free initially. No new model downloads; small public data only. Stop on
control, route, identity, numerical, context, resource or replay failure, preserving
all partial evidence. No automatic successor or retry. Record inherited payload
bytes/parameters, scored input tokens/bytes, stored scalar logits/normalizers,
wall/CPU/GPU/RSS and zero acquisition/training costs separately.

R105 remains a rejected evaluator. R106 is a distinct diagnostic, not a way to
evade that result. ABI acquisition, native LayerCake defects, CPU advantage and
the full moonshot remain unproved by this experiment. Publication is authorized.

Primary reference: Warstadt et al. (2020),
[BLiMP](https://aclanthology.org/2020.tacl-1.25/). Its externally specified
minimal pairs permit direct probability ranking without a learned NLI judge.
