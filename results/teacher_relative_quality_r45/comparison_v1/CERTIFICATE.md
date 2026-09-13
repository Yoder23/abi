# R45 Bounded Teacher-Relative Quality Certificate

Verdict: `PASS_R45_BOUNDED_TEACHER_RELATIVE_QUALITY`

The frozen 296,554-parameter shared LayerCake package from R43/R44 was not
retrained or regenerated. On the 144 prospectively frozen R44 prompts, its
already recorded outputs passed 141/144 functional checks. One live greedy
attempt from the pinned 7,615,616,512-parameter Qwen2-7B-Instruct source passed
94/144 under the same task-specific scorer. There were 49 candidate-only passes
and two teacher-only passes. Both systems produced 144/144 non-collapsed UTF-8
outputs.

The independent verifier recomputed all 144 paired rows, scores, hashes, task
totals, and information-accounting totals from raw evidence. It bound all
candidate outputs to the frozen R44 CPU observations and retokenized every
teacher output from the locally pinned source tokenizer.

Immutable evidence:

- `result.json`: `b651158946916be5a10cd41123d9bd36d948aab21014ff4371e07bd1740d0e26`
- `paired_quality.jsonl`: `035754c861b6fee0589ddaac4ec9fc9679b0929719da903a05f35854471c1e00`
- `strict_verification.json`: `7ca481ed20c71f153015fbad8580891f082f1fd5959d1202be7b3ab71ace6709`
- canonical evidence digest: `d001b465360f2941fa905763635a66b12e83c0ef63ca3ba3d2ae5c795171c9fe`

## Claim boundary

This certifies same-prompt functional quality only for the bounded R44
supplied-content interface. It does not certify human preference,
unrestricted English fluency, autonomous factual knowledge, arbitrary source
models, inference-speed superiority in this experiment, global minimality, or
superiority to LoRA or conventional distillation. The full ABI moonshot remains
`OPEN`.
