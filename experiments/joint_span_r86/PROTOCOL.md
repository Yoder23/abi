# R86 protocol: lexical-invariant joint neural span capability

R85 prospectively established a large causal transfer effect (954/1,400 versus
10/1,400 for its unchanged parent and 0/1,400 for a random bridge), but failed
its 1,330 quality gate. Its errors were balanced across destination display
positions. Most semantic errors selected the subject code, all 41 collapses
selected a whitespace-only tail, and some correct-role spans included adjacent
punctuation. These observations isolate surface-code shortcutting and global
length prediction as the next measured mechanisms.

R86 changes exactly those mechanisms. During training, each selected immutable
R76 teacher record receives a deterministic, bijective renaming of all four
nonce codes. The response is renamed by the same mapping. The transformation
adds no teacher claim or target information and the original teacher record is
retained as provenance. A compact bidirectional bridge then scores start and
end tokens jointly, normalizing over every physically valid span up to 16
tokens. Inference chooses one joint span; it has no rules, prompt templates,
output lookup table, source weights, source logits, or source activations.

Fresh prospective validation must be generated and hash-bound before source
scoring or candidate training:

- 1,400 distinct rows, 200 in each of seven premise families
- numeric namespace 980000 through 981399
- subject prefix `SUB`
- seven lexical triples disjoint from R76/R77/R85
- zero candidate observations at freeze

The frozen Phi source must pass at least 1,330/1,400, at least 180/200 per
family, and have zero corrected ties under the frozen R81 prior-corrected
all-choice scorer. Source failure closes R86 without candidate training.

One CUDA candidate is authorized after a source pass: seed 86001, width 128,
two bidirectional Transformer encoder layers, four heads, feed-forward width
512, maximum prompt 256 tokens, maximum span 16 tokens, batch 32, 3,000
successful AdamW steps, learning rate 1e-3, and weight decay 0.01. All 8,129
immutable R76 records must be covered. Every optimizer example is one
deterministic lexical renaming; generated training stems and numeric values are
forbidden from the prospective R86 namespace. Only bridge parameters train.

Before candidate generation, the exact bridge, metadata, parent, artifact,
catalog, source result, protocol, trainer, bridge implementation, and screen
are bound. Passing requires at least 1,330/1,400 exact outputs, at least 180/200
per family, at least 95% retention of source-passing rows, at least a 0.50
paired gain over the unchanged R78 parent, at least a 0.50 paired gain over a
deterministic random bridge, no more than 500 random-bridge passes, zero
collapse, and physical execution of exactly one capability cake, six deep
adapters, and one joint-span bridge.

Passing proves bounded extractive teacher-to-artifact-to-neural-package-to-
LayerCake transfer under fresh nonce identities. It does not prove unrestricted
English, domains, minimality, LoRA/distillation superiority, or the full ABI
moonshot. The full ABI moonshot remains `OPEN`.
