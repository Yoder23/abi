# R63 preregistration: capability-blind prompt-invariance augmentation

R60 prospectively failed and R61 proved that route correction recovered zero
passes. R62 then showed that removing R60's artificial ordinal recovered some
quality but removing the full wrapper did not repair format control,
abstention, or reasoning. The limiting factor is therefore the learned
payload's prompt-surface dependence, not router accuracy alone.

R63 is one bounded training successor. It starts from the untouched R47 parent,
installs the same six-block rank-32 per-capability sparse adapter topology as
R55, and reuses the exact immutable broad and anchor English archives. Before
tokenization, a deterministic record-ID hash prepends one of eight
capability-blind views to each broad training prompt; the anchor remains
unmodified. One view is identity, five are natural request prefaces, and two
are neutral nonce metadata prefaces. No prefix names the capability, answer,
evaluator, route, R60 text, or R60 result.

Frozen training configuration otherwise matches R55: CUDA, seed 63,001, 6,000
successful steps, batch 8, anchor batch 4, balanced capabilities, learning
rates 2e-5/1e-4, classifier weight 0.25, prompt-overlap weight 1.0, parent-logit
preservation 0.5, weight decay 0.01, max tokens 256, overlength exclusion, and
autonomous prefix recovery 400/every-8/[8,16,32].

Teacher calls, new teacher outputs, R60 outputs, evaluator outcomes, source
logits, and source activations in training are zero. The augmentation receipt
must bind every original record ID and original/augmented prompt hash.

R60 is now a disclosed development screen only. R63 must materially improve
its aggregate, reach at least 65/100 in every capability, and eliminate final
collapse to authorize a new prospective split. It cannot promote on R60. A
failure closes this augmentation design; no prefix-family, rate, seed, or step
sweep is authorized. The full ABI moonshot remains OPEN.

