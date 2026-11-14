# R108 — trained-task grounded generation qualification

Preregistered 2026-11-14. This is one user-authorized bounded successor to R107,
not a continuation of its untrained-coherence test or a training/model sweep.

## Question and scope

Can one fixed external control satisfy the existing mechanical obligations on
eight trained English task families, and can the frozen recipients do so through
their training-aligned interface? Generate answers autonomously. Do not supply
answer prefixes, plans, retrieval, candidate answers, or post-generation repairs.
These are fictional supplied-note tasks, NOT broad English or human fluency tests.

The preselected control is `Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, unquantized BF16 on CUDA. Its
[official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
reports instruction-following improvements. That motivates this one attempt,
not an assumption that it passes here. Greedy decoding differs from the card's
sampling recommendation. No second source, prompt rewrite, or rescue is allowed.

Keep the R76 parent, R100 full-model diagnostic, and R99 FP32 LoRA and
adapter-distillation final checkpoints unchanged. LoRA/distillation retain their
original matched data/update budgets; they were NOT trained by this new control.
The larger control is a feasibility ceiling, not a matched-size/training baseline.
No R99 plan branch reopening, new training, residual neutralization, or host edit.

## Frozen inputs, interfaces, and gates

Use 48 counterfactual pairs (six per task), made prospectively from the R104
qualification templates. Each pair differs only in note/person identifiers and
the supplied weekday. Both panels have novel identifiers, identical obligations,
and the same fact order. Run all 96 prompts in fixed interleaved order. Tasks:
summarization, rewriting, email drafting, conversation, format control, tone
control, clarification, and abstention. These are trained task families, not
training examples or proof of unrestricted generalization.

Recipients use only the R99 training-aligned `Task: ... / Response:` interface
and explicit registered task routes. Control uses its native chat template with
the unchanged R104 system message. Fixed greedy decoding, maximum 640 new tokens,
1024 total context allowance, no repetition penalty or truncation repair.

Keep the R104 screen as an unchanged necessary condition. Add case-sensitive
required-literal coverage and rejection of explicit negation of pending status
(`not/no longer pending/provisional/unconfirmed/awaiting`, including `n't`).
The existing unsupported-identifier/day checks reject counterfactual carryover.
This remains a conservative syntactic screen, with known scope/role-binding and
fluency blind spots; NOT a semantic entailment judge. Author fixtures establish
only the named obligations. No result can retroactively repair R103-R106.

Each 48-row panel must pass >=44/48 overall, >=5/6 for EVERY task, zero collapse,
zero generation-budget stops. Additionally >=44/48 pairs must pass BOTH members,
with >=5/6 successful pairs in EVERY task. Thresholds cannot be changed after
execution. Run the control and its fresh-process replay first; if it fails,
record the failure and DO NOT run the recipients. If it passes, run each of the
four recipients and fresh-process replay once. Replay must match every scientific
raw field exactly, excluding measured wall time. No partial best-of reporting.

Report all per-task counts, paired both-pass counts, counterfactual failure
types, collapse/termination, and paired cluster-bootstrap 95% intervals (10,000
draws, seed 108001, resample 48 pairs). A recipient mechanical noninferiority
screen additionally requires control-relative lower CI >= -0.10. Do not infer
method superiority from these small, previously trained checkpoint lineages.

## Resources, identity, and failure ownership

Source preparation: one pinned download <=9 GB, record bytes, time, public
revision and source-file hashes; require >=20 GiB remaining disk. Execution:
90 minutes total, 25 minutes/process, <=1 GiB new evidence. Four CPU threads,
fixed seed 108001, TF32 off; recipients FP32. Record process CPU/wall time,
sampled RSS (not claimed true peak), CUDA allocated peak, input/output IDs,
UTF-8 bytes, parameter count, hashes, environment and actual route traces.
No external hardware, teacher training calls, stored logits/activations, new
optimizer steps, copied parameters, bridge training, or deployed artifacts.

Freeze code/tests/protocol, prompts, evaluator fixtures, exact seven files per
recipient, native host runtime dependencies, and public source snapshot before
inference. Publish negative evidence as well as successes. Missing packages,
hashes, rows, replay, or raw accounting fail closed; saved pass flags are
recomputed. No overwritten evidence; interruptions get an explicit stop record.

Control failure = unmet experimental prerequisite, not ABI or LayerCake failure.
Recipient low quality = acquisition/recipient suitability finding; native host
fault requires a separate execution-conformance discrepancy. Route mismatch or
nonfinite execution = unlocalized integration failure until isolated. No new
CPU speed claim, host certification, core promotion, human evaluation, or full
moonshot claim follows. Next acquisition changes require the completed evidence.
