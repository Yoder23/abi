# R45 — Same-prompt teacher-relative quality

Status: `PREREGISTERED` before loading the source teacher.

## Frozen inputs

- R44 result SHA-256:
  `41f8625fba5b73538a572bb5bc9fe059da83f605b7933201909d76141176eec8`.
- R44 fixture SHA-256:
  `9409797e98fa771a3bbd62715bf33477f5db0db4e0c55b822b89805cb2ddf5c2`.
- R44 evaluation SHA-256:
  `85314b258b3f46edd42da24c717f1104ac79b525dba70aff1412d6184e3a3845`.
- Candidate package SHA-256:
  `6ccb115de59e71b1c70efcfe2b5a1673b88d8f1478fb0059d8577ef5e2b6e6a4`.
- Source: `Qwen/Qwen2-7B-Instruct`, revision
  `f2826a00ceef68f0f2b946d945ecc0477ce4450c`.
- System prompt SHA-256:
  `6d5f78c347dd548eb592b2d70c8b65a2066d75c9f379e77ca45472c87b682e08`.
- Greedy generation, one attempt, maximum 192 generated tokens. No retry,
  validator selection, prompt edit, package edit, or training is allowed.

## Comparison

Generate one live teacher response for every one of the 144 frozen R44 prompts.
Normalize only outer whitespace and CRLF to LF, exactly as R31 acquisition did.
Score teacher and the already frozen CPU package output with the same frozen
functional/grounding/adherence/non-collapse scorer. Pair by immutable record ID.

## Gates

1. Exactly 144 teacher calls and 144 paired rows; every source identity, prompt,
   system, rendering, output, and token count is recorded and hash-bound.
2. Candidate remains 141/144 and teacher is scored without retries.
3. Candidate total functional score is no lower than teacher.
4. Candidate-correct/teacher-wrong improvements are at least as numerous as
   teacher-correct/candidate-wrong regressions; candidate regressions are at
   most three.
5. Candidate remains at least 10/12 in every task and 12/12 on abstention.
6. Both systems have zero repetition-collapse outputs and all generated text
   is strict UTF-8.
7. Teacher load/generation time, input/output tokens and bytes, CPU RSS, CUDA
   memory, and source snapshot are recorded. The candidate is not retrained or
   regenerated in this experiment.

This is a same-prompt functional comparison only for the bounded R44 interface.
It is not a human-preference judgment, unrestricted English parity, an inference
speed comparison, arbitrary teacher transfer, global minimality, or superiority
to LoRA/distillation. Missing source weights, CUDA, rows, hashes, or outputs fail
closed. The full ABI moonshot remains open.

