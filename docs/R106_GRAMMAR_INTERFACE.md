# R106: a separate grammatical-discrimination prerequisite

2026-11-11. **`GRAMMATICAL_DISCRIMINATION_CONTROL_FAILED`**.
Qwen scored 178/200 against the frozen 190/200 threshold. The experiment stopped
before its recipient/interface comparison. No English core, ABI acquisition,
native LayerCake defect or product certificate follows.

## What changed, and what did not

R105 rejected a conventional semantic judge. R106 instead uses external sentence-
pair labels and direct likelihood ranking: does the fixed model assign greater
conditional probability to the designated grammatical sentence? It does not
judge free-form text with that failed NLI model.

This is deliberately **not autonomous generation**. Every target token is
supplied, and likelihood is not the same thing as fluent behavior, grounding or
instruction adherence. R103/R104's 44/48 and 5/6 generation gates are unchanged
and still closed. Even a positive R106 result could not substitute for them.

The [protocol](../experiments/grammar_interface_r106/PROTOCOL.md), code, 26 scoring
tests and external data were committed at `1b7e64f`; the 742-input identity freeze
was committed at `519b73b`, before scoring. Four published BLiMP paradigms supply
50 qualification and 100 separate comparison pairs each, selected by a fixed
hash ordering. The 400 comparison pairs were not executed. No model/route/prompt
search, threshold adjustment or training occurred.

## Registered result

| Grammar family | Correct / 50 | Required |
| --- | ---: | ---: |
| Anaphor number agreement | 49 | 45 |
| Determiner-noun agreement | 47 | 45 |
| Irregular past participle verbs | 37 | 45 |
| Regular subject-verb agreement | 45 | 45 |

Total **178/200 (89%)**, below 190/200. Irregular past-participle cases also
miss their family floor. There are no ties. A fresh-process replay reproduced
all raw token IDs, selected logits and log-normalizers byte-for-byte.

The post-run [descriptive inventory](../results/grammar_interface_r106/descriptive.json)
lists every one of the 22 errors. Of these, 21 have equal token counts. Equal-
length cases score 171/192; unequal-length cases score 7/8. Per-token normalization
also scores 178/200 in aggregate; this descriptive alternative cannot rescue the
frozen sum-log-probability gate. These observations do not identify a causal
mechanism or prove that the model lacks fluent English.

## External labels have limits too

Data: [BLiMP source](https://github.com/alexwarstadt/blimp/tree/3e56b06fcabca9b30822fc66435fca6b1aa40bb1),
CC-BY, with its upstream README retained. The benchmark and likelihood-ranking
method are due to [Warstadt et al. (2020)](https://aclanthology.org/2020.tacl-1.25/),
not an ABI invention. These four grammar families are a subset of the benchmark;
the common instruction/chat prefixes also make this a conditional diagnostic,
not an official full BLiMP score. Public-data exposure in pretraining is unknown.

The error inventory reveals an apparent label reversal in upstream pair
`regular_plural_subject_verb_agreement_1:886`: the designated good sentence is
"Senators bothers Daniel." and the designated bad one is "Senators bother Daniel."
There are also potentially contestable language judgments, including singular
`themselves` and `shrank`/`shrunk` usage. We did not relabel, omit or rescore any
case. Crediting the single apparent reversal would give only a descriptive
179/200 ceiling, still below 190/200. This is not an independent adjudication of
all 200 labels; do not infer general human fluency from benchmark agreement.

The apparent reversal is preserved as a measurement limitation, not an excuse
to relax the gate. Campaign human ratings remain 0/21,000; independent hardware
and broad English validation are still open.

## Source identity, costs and boundaries

The source is the unchanged `Qwen/Qwen2-7B-Instruct` revision
`f2826a00ceef68f0f2b946d945ecc0477ce4450c`, 7,615,616,512 parameters in GPU FP16.
All ten local source files were hashed before and after execution. A separate
[public metadata cross-check](../results/grammar_interface_r106/upstream_weight_metadata.json)
matched the four tensor-shard sizes and SHA-256 values to the pinned Hugging Face
revision. No new model download or recipient parameter import occurred.

Each pass: 400 source-model scoring forwards, 12,956 input tokens, 2,556 scored
target tokens, 2,556 stored selected logits and 2,556 log-normalizers. The targets
are external benchmark text, not teacher-generated training data. Initial/replay
execution-body wall times were 58.82s/58.63s; process CPU times 67.25s/67.00s.
The two-process pilot measured 142.5s. Peak CUDA allocation was 15,316,812,288
bytes; sampled RSS was 2,130,075,648/2,132,402,176 bytes. These are diagnostic
costs, not CPU inference-speed, acquisition-efficiency or deployment benchmarks.

Zero recipient calls, optimizer steps, newly trained bridge parameters, new
checkpoints or autonomous generated outputs. The retained LoRA/distillation/
full-model/parent checkpoints and LayerCake source remain unchanged. Their
historical acquisition budgets are not erased; they produced no R106 observations.

## Integrity and reproducibility

- Freeze SHA-256: `fda0bebc109f707f430c94f6c284ce36ea7b5300eb6306ba86f5a160fc3f9e56`
- Decision SHA-256: `2c17e0e402c27ce51e83485a6374724d498d0b0f3c1ac5275bc16b09a22da9eb`
- Both raw-file hashes: `20ae0e78f36fb43b7f0b9471361dc93c23bbd99b27d75b9ec1db22e44efc2905`

The completed verifier requires both runs, process records, frozen identities,
external populations, all model payloads and the existing decision. It recomputes
scores from the numeric rows, rejects missing replay directories and does not
recreate a missing decision. Selected logits/normalizers are not a full-vocabulary
logit archive; their provenance is supported by the registered live executions
and exact replay, not inferred from a stored success flag.

All 52 focused checks passed: 26 preregistered mechanics tests and 26 additive
completed-evidence checks, including 25 hostile mutations. The raw-evidence
mutation tests hash the unchanged source payload once and reuse that validated
inventory; a separate test rejects absent source inventory. The supported suite
passed 364 default and 16 exact historical tests, with four declared skips.
The [operations note](../results/review_2026_11_11/r106_operations.md) preserves
the nonfatal Windows test-process shutdown warning.

```bash
python -m experiments.grammar_interface_r106.run verify
python -m experiments.grammar_interface_r106.describe
python -m pytest -q tests/test_grammar_interface_r106.py tests/test_r106_completed_verifier.py
```

These commands require the pinned source payload and sibling host/checkpoints;
they recompute evidence without model inference. Never rerun `pilot` in its
occupied directory. Publication/reconstruction is recorded separately below.

The publication inventory binds 581 repository assets, including four inherited
LFS checkpoint paths, plus 182 sibling-host source files and ten pinned Qwen
files. Manifest SHA-256:
`3e0b15e1368cd71e9e51cdd4e28f3a9aac973bc618ea67fd7ff5398288d30233`.
No new model checkpoint is published by R106.

## Next decision

Do not scale learners, choose an easier grammar subset, switch probability
normalization after the result, or treat a new evaluator as a qualified English
control. This diagnostic did not open either recipient comparison. The next
proposal must address actual task-response evaluation and the validity of its
expected labels, not silently replace free generation with token discrimination.
The compact fluent-core acquisition breakthrough remains unproved.
