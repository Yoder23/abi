# R42 — Shared labeled English-core compression

Status: `PREREGISTERED` before implementation and execution.

## Frozen inputs

- ABI base commit: `04c986c`.
- LayerCake host commit: `f8a4569508f9f4a20e4babe3b4a942ac887da147`.
- R31 accepted rows:
  `5ff7691d67b0e8271439acfbfe25540c7d42c06918b58b9257873c6f7ca9f05d`.
- R38 accepted abstention rows:
  `9c9411aa10080aa65ce4088d99fb1306c709af0284084f50af03505ea3b57c04`.
- R36, R39, R40, and R41 result SHA-256 values respectively:
  `1830a0a1be9b520376ce228fd50514e79f9b3a8224d56f7a1b36e0338d03d5bd`,
  `eee143b1f35038ed518cbd4a9a8aa034da03e0799fea39a9498e76b7bed242f0`,
  `50c6a5aabb8e416ec7e959919591246eca1193b36c0c6b707ad0c217f7f324d6`,
  `d964ef7d591788211a6d54911b126c1e392a0b535fd2af8bc3027a6811f427c4`.
- The exact 282 R36/R39 selected record IDs are the only training targets.
- The 144 R40 prompts are the immutable development evaluation. This is not a
  new hidden replication and must be called development evidence.

## Architecture and information intervention

Replace twelve independent task models with one shared neural plan model. ABI
normalizes each already labeled prompt to the sorted schema
`slot0, slot1, slot2, slot3, task`; source field values use the task's sorted
original field order, absent fourth fields receive the fixed value `unused`,
and `task` contains the registered task label. The teacher output is unchanged.

The model remains one LayerCake `PortableTokenPlan`: width 64, four heads, two
encoder layers, two decoder layers, feed-forward width 192, pointer width 32,
zero dropout, 128 source positions, and 384 target positions. Train on CUDA
with AdamW (`lr=8e-4`, weight decay `0.01`), batch size 48, seed 42001, and
exactly 6,396 updates = 307,008 example exposures. This equals the aggregate
example exposures of the twelve R36/R39 models. No retry, seed sweep, teacher
call, new output, hidden activation, logit, source weight, or manual target is
allowed.

Package the resulting state and a declarative field tokenizer into one signed
non-executable `.cake`. Also package an all-zero state with the identical
tokenizer and geometry as the causal control. Execute both only through fresh
real LayerCake `DirectCakeHost` registries.

## Development gates

1. Exactly 282 frozen teacher outputs train the candidate; imported teacher
   bytes/tokens are fully counted and new teacher calls are zero.
2. Candidate CPU and CUDA functional score are each at least 132/144, every
   task is at least 10/12, abstention is 12/12, routes/contracts are 144/144,
   and representation failures are zero.
3. CPU and CUDA candidate output bytes are identical on 144/144 rows.
4. The candidate shows no empty-output or repeated-substring collapse.
5. The single candidate archive is no more than 20% of the total twelve R41
   candidate package bytes and has no more than 450,000 active parameters.
6. The zero-state control scores at most 36/144 functional, matches the
   candidate on at most 12/144 outputs, and is physically a different tensor
   payload/package hash.
7. Both packages pass strict signature/integrity verification. Removal of the
   candidate rejects generation and reinstall restores the first output.
8. Receiver training after packaging is zero; teacher/source is absent during
   LayerCake execution.
9. Raw training selection, training history, per-row CPU/CUDA/control outputs,
   package inventory, lifecycle rows, costs, and hashes are written once.

Missing inputs, hashes, rows, packages, CUDA, or recomputable evidence fail
closed. A pass is only a compact shared supplied-content English-core
development result. It is not unrestricted English, a hidden replication,
global minimality, arbitrary-domain extraction, or LoRA/distillation
superiority.

