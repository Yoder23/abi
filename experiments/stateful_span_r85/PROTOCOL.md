# R85 protocol: stateful neural span capability

R83 and R84 prospectively rejected stateless token-mixture pointers. R84's
deployment-aligned pointer improved its R78 parent by only 23/1,400 and still
collapsed 36 times. The measured bottleneck is missing multi-step selection
state plus non-atomic realization.

R85 tests a different bounded capability ABI. A compact bidirectional neural
span bridge consumes frozen R78 hidden states, predicts a prompt start and
token length, and realizes only that learned span. The bridge is trained from
the immutable R76 teacher-labeled ABI artifact. It contains no rules, prompt
templates, source weights, source activations, source logits, or output lookup
table. R78 remains byte-identical and the source teacher is absent during
bridge training and inference. Span copying is deterministic only after the
neural bridge makes its selection.

Fresh prospective validation was generated before candidate training:

- catalog: `catalogs/stateful_span_validation_r85_v1.json`
- catalog SHA-256: `8ad899dfb15f120ae3ffdbde6b6d8dcb06bbb265883665bcecdeff048eaafc4c`
- rows: 1,400, 200 in each of seven families
- numeric namespace: 970000–971399
- subject prefix: `NODE`
- nonce triples are disjoint from R81
- candidate outputs observed before catalog freeze: zero

The frozen Phi source is scored first with the same prior-corrected all-choice
method used in R81. Source promotion requires at least 1,330/1,400, at least
180/200 per family, and zero corrected ties. Candidate training cannot start
until that source result is immutable and hash-bound.

If source scoring passes, one CUDA bridge is authorized: seed 85001, width 128,
two bidirectional Transformer encoder layers, four heads, feed-forward width
512, maximum span length 16, batch 32, 4,000 successful AdamW steps, learning
rate 1e-3, weight decay 0.01, uniform epoch-shuffled coverage of all 8,129 R76
records, and a 256-token maximum prompt. Only bridge parameters train. No
teacher, parent distillation, validation tuning, or nearby sweep is authorized.

Before any candidate generation, bridge weights, metadata, parent, source
artifact, protocol, screen, and fresh catalog are bound. Candidate gates are:
at least 1,330 exact autonomous span outputs, at least 180/200 per family, at
least 95% retention of source-passing rows, at least a 50-point paired gain over
the unchanged R78 autoregressive control, zero collapse, and physical execution
of one capability cake, six deep adapters, and one span bridge. Package-removed
and randomized-bridge causal controls are required after a quality pass.

Passing proves bounded extractive teacher-to-artifact-to-neural-package-to-
LayerCake transfer. It does not prove unrestricted English, unrestricted
domains, minimality, LoRA/distillation superiority, or the full ABI moonshot.
The full ABI moonshot remains `OPEN`.
