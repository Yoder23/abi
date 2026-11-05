# R99: source-grounded normalization and matched English learning

Registered 2026-11-05 before new teacher capture or any student training.
R98 and R98-C1 remain failed source prerequisites. No old row, threshold,
evaluator, verifier or verdict is changed by this separately named experiment.

## Measured bottleneck and intervention

C1 preserved all 31 original token prefixes and removed truncation, but reached
226/240 mechanical passes. Emails were 22/30 versus a required 23: seven exceeded
180 words and one triggered the confirmation screen. Another token cap or prompt
sweep is not justified. The new intervention is compiler-side source-sentence
selection, not teacher prompting or extra gradient steps.

For emails only, retain complete source sentences covering the project ID,
coordinator, planned day and pending approval, plus the source greeting/sign-off.
A deterministic dynamic program minimizes word count over these source units.
Every emitted content span must be an exact teacher substring, with its offsets
recorded. Missing coverage preserves the original answer and its failure; no
invented words, synthetic answer renderer, selective row deletion or retry.
Other task outputs remain unchanged. The requested slots are supplied by the
public user prompt, not by hidden expected answers. The known lexical evaluator
remains unchanged, including its disclosed false positives.

This minimum is only over source sentence subsets for four literal constraints.
It is not global information minimality, semantic purity, or proof that removing
sentences preserves every implicature. The teacher may introduce unsupported
facts inside retained sentences; human semantic assessment remains open.

## Data and prospective boundary

Use all 192 R98 training requests and their C1 completed responses. No student
has trained on these data. Normalize their targets before any learner sees them.
Freeze 48 fresh development prompts before capture: new project/person IDs,
changed planned days and a new order, seed 99002, with the same eight controlled
supplied-content templates. None is an old R98 development prompt. This is a
prospective resample of this bounded family, not unrestricted English.

Capture each new development response once from the same pinned Qwen2-7B source
on GPU, with the unchanged R98 system message and a maximum of 384 tokens.
Normalize all 240 rows with the same frozen rule. Training receives only the 192
training rows. Raw and normalized teacher outputs and source span provenance
remain separately accessible; no development answers enter training targets.

The normalized acquisition corpus must satisfy the unchanged R98 mechanical
gate: >=216/240, >=75% per family, zero collapse and no source truncation. This
qualifies normalized data only; it does not retroactively qualify raw R98/C1.
If it fails, stop. No automatic normalizer, prompt, cap, data or optimizer sweep.

## Learners and budgets

Reuse the immutable R98 training/evaluation engine with a separate configuration
and output namespace. ABI-plan and response-only sequence distillation use the
same route-local adapter/cake scope; conventional LoRA uses rank-32 weight
deltas. All start from the exact 87,710,428-parameter R76 parent. All receive the
identical normalized target archive, raw source archive, supplied information,
sample order, seed 99001, 384 updates, batch 2, 1024 padded positions, optimizer
and 640-token generation cap. The unchanged parent is evaluation-only.

Record different trainable parameters, loss-bearing tokens and actual compute;
equal steps or padding are not equal FLOPs. ABI still uses a supervised
distillation loss. No superiority over distillation or LoRA is assumed.
The complete inherited parent is counted, not just the learned delta. No source
weights, activations or logits are copied, and no teacher is loaded at student
inference. Signed LayerCake core acceptance is not part of this pilot.

The source-normalized teacher comparison and the raw-teacher comparison must
both be reported. Continuation requires at least 44/48 candidate passes, at least
5/6 per family, zero collapse, valid autonomous generated plans, paired 95%
bootstrap lower bound >=-0.10 versus normalized teacher, positive lower bound
versus parent, and no lower point score than either trained control. A pass
opens a broader three-seed preregistration, not a fluent-core certificate.

Six hours total, one hour per training arm, 20 GiB new disk and >=30 GiB initial
free disk. GPU acquisition/training. Stop on first failed prerequisite, overflow,
non-finite optimization or budget exhaustion. Preserve partial journals and
checkpoints. Concurrent CPU test/build activity may affect costs; no inference
speed or training-efficiency claim may be drawn from these timings.

## Accounting, attribution and reproduction

Count all prior R98/C1 source requests (271), all repeated-prefix compute, new
48-request capture, unique source bytes, selected target bytes and token counts
separately. Preserve excluded text in the archive; omitted tokens are not a
claim that the teacher never supplied them. No English-core/domain purity is
proved because the inherited parent remains a separately counted substrate.

Source/normalization failure is ABI acquisition evidence. Learner failures are
ABI fit/generalization evidence after parent mechanics; they are not native
LayerCake regressions without an isolated reproduction. LayerCake source,
canonical ABI and historical release remain unchanged.

Commands: `python -m experiments.grounded_english_r99.run freeze`, then `pilot`.
Use `verify-source` for raw/normalized provenance and `verify` only after all
arms complete. Historical engine format names can still contain R98; the R99
freeze and output namespace are the controlling lineage. All product, CPU,
human-quality, purity, independent-hardware and full-moonshot gates remain open.
