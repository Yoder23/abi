# R30 v12 full-host learnability upper bound

V9 (frozen host), v10 (final sparse cakes), and v11 (depth-shared bridge)
scored 0, 11, and 21 of 144.  V12 is the final control on the existing 288-row
R30 corpus: it permits every parameter of the 61.7M-parameter LayerCake host to
update.  This deliberately violates the moonshot's immutable-core and compact
artifact requirements, so it is never promotion eligible.

V12 uses the same base checkpoint, tokenizer, cached validated responses,
task-route mapping, 2,400 balanced updates, batch size four, seed 30,812, and
learning rate 0.00002.  No source-teacher execution occurs.  A derived full
checkpoint is loaded into a fresh model for the same 144-row evaluation.

Interpretation is frozen:

- if the full host passes 132/144, the base architecture can acquire this
  corpus and compact bridging—not data—is the next bottleneck;
- if it fails, this 288-row acquisition corpus is insufficient for the host
  architecture and no nearby adapter sweep is authorized;
- either outcome remains a non-promotable sequence-training control and cannot
  establish ABI superiority, core immutability, or general English.

