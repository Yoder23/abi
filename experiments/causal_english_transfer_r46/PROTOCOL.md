# R46 - Causal English transfer feasibility

Status: `PREREGISTERED` before candidate training or neural-only evaluation.

## Question

R46 asks whether an already acquired, semantically segregated foreign-teacher
archive can materially improve autonomous neural English generation in the
small sparse LayerCake Phase 2 core. It is an architecture feasibility test,
not a promotion or minimum-information certificate.

## Frozen inputs

- training acquisition archive:
  `results/abi_moonshot/segregated_acquisition_v3/phi3-broad-natural-conversation-complete-search-training-v3.abix`,
  SHA-256 `82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150`;
- disjoint evaluation archive:
  `results/abi_moonshot/segregated_acquisition_v3/phi3-v6-english-search-validation-survey.abix`,
  SHA-256 `b81477560f886ef79df08618956fb9f2ed721296fd5b97e1c6dec8798d68d4f1`;
- LayerCake Phase 2 parent checkpoint:
  SHA-256 `fc8d32c77bfa9d39152a6d32436c4dbbf767f9d1c88f3de17fec84a24bd3f781`;
- parent tokenizer:
  SHA-256 `d1c03801fd9559b8586490733a3788439ff230bf5f09666a0abb7ddb615e9f10`;
- exact parent architecture and sparse top-1 cake bank remain unchanged.

The training archive contains 24,421 labeled records and 2,784,714
authoritative Phi-3 teacher tokens across the 14 registered English-core
capabilities. The evaluation population contains 1,400 validation records,
100 per capability. Record-ID, prompt-hash, and output-hash overlap with the
training archive must each be zero.

## Frozen training and evaluation

- seed: 46,001;
- exactly 1,745 balanced steps, batch size 14 (one record per capability);
- all 7,176,097 LayerCake parameters are trainable for this feasibility upper
  bound; the deterministic English planner is never called for candidate,
  parent, or control generation;
- response-only next-token loss plus the existing 0.02 top-1 routing-balance
  objective;
- AdamW, learning rate 0.0001, weight decay 0.01, gradient clip 1.0;
- full available context up to 1,024 tokens; the five longer training records
  are excluded and identified by hash;
- generation is greedy with repetition penalty 1.1 and no-repeat four-token
  n-grams, at most 384 generated tokens and never beyond the model context;
- the terminator is the learned byte string `\n<|end|>` and is removed from
  rendered output;
- development evaluation uses the first 20 record IDs in lexical order per
  capability from the disjoint validation population: 280 prompts total;
- compare the trained candidate, the unchanged parent under the identical
  neural decoder, and the stored source response with the same frozen
  per-record evaluator;
- store every prompt/output hash, token sequence, score, collapse metric,
  latency, route evidence, and all information accounting.

## Feasibility gates

R46 passes only if strict raw recomputation establishes:

1. exactly 280 rows for each neural system and 280 paired source rows;
2. zero training/evaluation overlap by record ID, prompt hash, and output hash;
3. candidate functional accuracy at least 0.50 overall and at least 0.30 in
   every capability;
4. candidate-minus-parent paired functional improvement at least 0.20;
5. candidate-minus-source functional difference no worse than -0.25;
6. zero invalid UTF-8 outputs, at most two collapsed candidate outputs, and
   all eight physical sparse experts receive training assignments while at most
   one expert executes per token; and
7. the parent checkpoint/tokenizer and both source archives remain unchanged.

A pass authorizes a larger causal acquisition and fresh prospective
replication. It does not certify unrestricted English, teacher parity, human
quality, package installation, minimum information, arbitrary sources,
specialist-domain segregation, or superiority to LoRA/distillation. A failure
must be preserved and diagnosed before any materially different successor.
The full ABI moonshot remains `OPEN`.
