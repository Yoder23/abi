# R98: plan-supervised autonomous English pilot

Registered 2026-11-04 before new teacher capture or candidate training.

## Question and change from prior work

Can a teacher-derived, explicitly grounded content plan improve free-token
English realization in a compact existing LayerCake substrate compared with
direct sequence distillation and conventional LoRA?

R97 selected an answer span. R98 instead generates variable-length English
responses in eight families: summarization, rewriting, email, conversation,
format control, tone control, clarification, and abstention. The intervention
is a supplied-proposition plan learned as an intermediate autoregressive target.
At evaluation the candidate must generate its own plan and answer; it receives
no gold plan, teacher response, retrieval result, answer renderer, or evaluator.
Plans contain only literal facts supplied in the prompt. This is a controlled
test of ABI normalization, not autonomous semantic discovery. ABI still uses
ordinary supervised/distillation losses; novelty and superiority are hypotheses.

## Scope and matched controls

All three arms start from the exact R76 six-block LayerCake parent, including
its pretrained/imported information. Count its entire approximately 88M-parameter
state; a small trained delta is not a small independently deployable model.
The parent checkpoint SHA-256 is
`b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e`.
Source: unchanged cached Qwen2-7B-Instruct revision
`f2826a00ceef68f0f2b946d945ecc0477ce4450c`, CUDA FP16. No teacher weights,
logits, or activations are copied into the candidate. The inherited LayerCake
parent is a separately accounted imported substrate and may contain knowledge.
This experiment cannot prove a fact-free English core.

Arms:

1. ABI plan: existing route-local nonlinear adapters/cakes; plan then response.
2. Sequence distillation: identical parent and adapter/cake trainable scope;
   response only. Same source examples, plans available in its input archive.
3. LoRA: identical parent with frozen existing adapters/cakes; rank-32 deltas
   on each block's attention and MLP matrices; response only.
4. Unchanged parent: evaluation-only causal baseline.

The two response-only controls receive the same raw user prompt and source
answer, and have access to the same acquisition archive. The plan is a
deterministic duplication of supplied context, not extra teacher information.
All trained arms have seed 98001, identical sample order, 384 successful
updates, batch size 2, fixed padded context 768, AdamW 1e-4, weight decay
0.01, gradient clip 1, and FP16 autocast with FP32 master weights. Record real
loss-bearing tokens, parameter counts, GPU wall time, RAM, VRAM and parameter-
seconds. Equal padded workload is not equal FLOPs; no compute-superiority claim
is allowed from this pilot. LoRA's differing trainable count must be disclosed.
Training arm order is fixed by the registered seed. No tuning/retry/seed sweep.

## Evaluation and budgets

192 training prompts (24/family) and 48 development prompts (6/family), with
disjoint project/person identifiers, are frozen with evaluator and source code
before acquisition. The prompt population is controlled supplied-content
English, not a representative natural-language corpus. Teacher output maximum
144 tokens. Every source attempt, including failures, is retained. Candidate
generation maximum 384 tokens (includes generated plan); controls have the same
cap. Greedy generation with no lexical rewrite or repetition suppression.

Before training, the frozen teacher must meet the declared mechanical content
screen: >=90% overall, >=75% per family, zero repeated-substring collapse.
Missing/truncated/invalid training records fail the source prerequisite; no
selective dropping or prompt repair. Failure closes the current source interface.

The machine screen measures required supplied entities, unknown-identifier and
number leakage, length, requested surface form, and collapse. It does not
certify grammaticality, semantic entailment, human preference, or full fluency.
Report these as lexical/format proxies, never scientific English certification.
Keep all output text for separate blinded human assessment.

Pilot continuation requires >=90% overall, >=75% each family, zero collapse,
teacher noninferiority (paired 95% bootstrap lower bound >=-0.10), positive
paired gain over unchanged parent, and no lower point score than either trained
control. Package tensors must be finite and reload-exact. A machine pass only
opens a new prospectively sealed broader test and three paired seeds; it does
not promote a core. No held-out final data from earlier campaigns is consumed.

The experiment has a six-hour total wall limit, a 60-minute per-training-arm
limit, at most 20 GiB new disk, and requires >=30 GiB free disk before starting.
Preserve partial raw journals and unique checkpoint paths on any interruption.
An experiment budget stop is not a machine hard resource limit.

## Attribution and release boundary

- Source prerequisite fails: ABI source/acquisition interface; LayerCake untested.
- Parent fails mechanics before training: integration-boundary failure, pending
  isolated diagnosis; do not assert a LayerCake architectural defect.
- Parent mechanics pass, acquired generation fails: ABI acquisition/fit/
  generalization failure, not evidence of a LayerCake regression.
- ABI candidate and controls fail: retain all evidence; no automatic attribution
  to the host or permission for a larger data run.
- Native unchanged LayerCake regression requires an independently reproduced
  host test; no language-score result alone establishes it.

Use ABI's existing experimental loader against the exact pinned, clean sibling
LayerCake commit `a24458818dc4489b4b977c49bc6ea5b8d089ef24`. No LayerCake source
edits or release recertification occur here. The pilot checkpoints are ABI
research artifacts, not signed production cores. Canonical signed installation,
CPU >=2x optimized baseline, >=95% speed retention, TTFT/RSS, semantic isolation,
independent human quality, three seeds, and different hardware remain mandatory
before any product promotion. CPU timing here is diagnostic only.

## Run and evidence

`python -m experiments.plan_english_r98.run freeze` writes a once-only manifest
binding this protocol, code, source/parent identities, and catalogs.
`python -m experiments.plan_english_r98.run pilot` runs gated acquisition and
all matched arms, or preserves the first failed prerequisite. Use
`python -m experiments.plan_english_r98.run verify` to recompute completed
evidence; incomplete evidence must fail closed. Source execution and training
run in separate processes so the teacher cannot remain at student inference.

The whole-moonshot verdict remains `FULL_MOONSHOT_NOT_PROVEN`.
