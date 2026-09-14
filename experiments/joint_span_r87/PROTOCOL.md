# R87 protocol: source-qualified joint neural span test

R86 was correctly closed before candidate training because its pinned source
missed one preregistered family floor by three rows. R87 carries the entirely
unobserved R86 candidate mechanism unchanged. To avoid iterating random source
catalogs, it reuses the seven lexical triples from the already source-qualified
R85 catalog while changing all 1,400 numeric identities and the subject prefix.
No R87 candidate may read or train on R85 validation rows.

The R85-measured changes remain: deterministic bijective lexical renaming of
each selected immutable R76 teacher record during training, and a compact
bidirectional bridge that scores start/end token pairs jointly over all spans
up to 16 tokens. The runtime has no rules, prompt templates, output lookup
table, source weights, source logits, or source activations.

Fresh prospective validation is frozen before source scoring or candidate
training: 1,400 distinct rows, 200 in each of seven families, numeric namespace
990000 through 991399, subject prefix `ENT`, and zero candidate observations.
The frozen Phi source must pass at least 1,330/1,400, at least 180/200 per
family, and have zero corrected ties under the frozen R81 prior-corrected
all-choice scorer. Source failure closes R87 without candidate training.

One CUDA candidate is authorized after a source pass: seed 87001, width 128,
two bidirectional Transformer encoder layers, four heads, feed-forward width
512, maximum prompt 256 tokens, maximum span 16 tokens, batch 32, 3,000
successful AdamW steps, learning rate 1e-3, and weight decay 0.01. All 8,129
immutable R76 records must be covered. Each optimizer example is one
deterministic bijective code renaming whose stems and numeric value are
disjoint from the R87 validation namespace. Only bridge parameters train.

Before candidate generation, every package, input, implementation, protocol,
and evidence hash is bound. Passing requires at least 1,330/1,400 exact
outputs, at least 180/200 per family, at least 95% source retention, at least a
0.50 paired gain over the unchanged R78 parent, at least a 0.50 paired gain
over a deterministic random bridge, no more than 500 random-bridge passes,
zero collapse, and physical execution of one capability cake, six deep
adapters, and one joint-span bridge.

Passing is still a bounded extractive transfer proof, not unrestricted English
or domain transfer, minimality, LoRA/distillation superiority, or the full ABI
moonshot. The full ABI moonshot remains `OPEN`.
