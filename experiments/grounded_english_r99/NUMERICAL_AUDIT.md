# R99-N1: bounded mixed-precision failure attribution

Registered 2026-11-05 after R99's normalized source passed 230/240 and the
distillation arm stopped after its first successful update with non-finite
gradients. Preserve that launch, source corpus, update and failure unchanged.

Reproduce only the first two registered distillation minibatches from the same
parent, sample order, targets, optimizer and seed. Compare exactly three
conditions: FP16 autocast with the original 65,536 loss scale; FP16 autocast
with unit loss scale; and FP32 without autocast or loss scaling. Each starts
from the same parent and performs at most two updates. No evaluation prompts,
quality scores, new teacher queries, data changes, learning-rate changes or
additional conditions. Do not publish these ephemeral states as candidates.

Record per-step loss, gradient norm, non-finite element counts and parameter
names, sample IDs, precision, scale and elapsed time. Save gradient tensors
for any non-finite event and bind their hashes. The original FP16 condition
must reproduce the first logged loss and sample IDs before an attribution.

If scaled FP16 fails while unit-scale FP16 and FP32 remain finite, classify a
loss-scaling/precision implementation failure, not an ABI semantic failure or
LayerCake architectural regression. If FP32 also fails, stop and diagnose;
do not automatically change the optimizer, parent or training data.

Only a successful numerical diagnosis may justify one separately frozen FP32
restart of all three matched R99 arms with the same 384-update budgets, source
archive, seed, sample order, hyperparameters and gates. This does not repair or
promote the failed FP16 run. Count failed/diagnostic costs separately and report
them in total acquisition cost. No precision or optimizer sweep follows.

The audit itself proves no English quality, training efficiency, deployment
speed or moonshot completion. Maximum runtime: ten minutes, maximum six total
updates, no saved trained candidate. CUDA only; the teacher is absent.
