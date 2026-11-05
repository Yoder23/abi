# R99-FP32: one numerical implementation repair, all matched arms restarted

Registered 2026-11-05 after the six-step R99-N1 audit. The original FP16 run
and its first successful update remain failed and immutable.

R99-N1 exactly reproduced the original first loss and sample IDs. At the next
minibatch the 65,536-scale FP16 condition produced 50,688 non-finite gradient
elements in the abstention route. Both unit-scale FP16 and FP32 produced finite
losses and gradients on both batches. This isolates a loss-scaling/precision
failure in the acquisition implementation; it proves no semantic or host defect.

Run exactly one FP32 restart of all three registered R99 arms from the unchanged
parent. No autocast, gradient scaler, lower precision or TF32 training. Keep the
same normalized source bytes, catalogs, seed 99001, sample order, optimizer,
learning rate, weight decay, clipping, 384 updates, batch 2, context 1024,
trainable scopes and 640-token generation cap. No failed FP16 checkpoint is
resumed. All comparisons come only from this new common-precision lineage.

This is a minimal fork of the frozen training loop. Evaluation, source
verification and decision logic use the unchanged R98/R99 engine with an
explicit separate configuration. The generated historical format names do not
imply a shared candidate identity; the output namespace and freeze bind it.

Preserve step-128/256/384 trainable snapshots and final states for all arms.
Every loss, gradient norm and exported tensor must be finite. Stop on the first
failure; no more precision, optimizer, data or architecture variants are
authorized by this repair. Total campaign deadline is the original R99 launch
plus six hours, not a new six-hour allowance. Each arm remains limited to one
hour, with <=20 GiB new disk. Record failed FP16 and numerical-audit costs
separately from these matched learning timings; no compute-superiority claim.

The same source gate and candidate continuation thresholds apply unchanged.
All raw/normalized source information is shared with LoRA and distillation.
Source normalization is supervised preprocessing and ABI learning uses a
distillation loss; superiority and novelty are not assumed. The 87,710,428-
parameter parent counts in full. Teacher-free inference, fixed artifact identity,
human fluency, semantic purity, broad generalization, three seeds, CPU >=2x,
TTFT/RSS, signed host acceptance and independent hardware remain separate gates.

Commands: `python -m experiments.grounded_english_r99_fp32.run freeze`, then
`pilot`. No new teacher queries occur. The full moonshot remains unproved.
