# R30 v10 frozen-host sparse-cake adapter control

R30 v9 established that the unchanged LayerCake Phase 2 checkpoint scores
0/144 on the broader R30 supplied-content suite.  V10 asks a narrower and
properly separated question: can the cached, labeled teacher records conform
that host by changing only its physically routed low-rank task cakes?

The checkpoint, tokenizer, transformer, tied output embedding, task classifier,
and unused cakes are frozen.  Only the seven task-cake routes exercised by the
twelve R30 behaviors may update.  The source teacher is absent; training uses
the 288 cached and validator-normalized rows from v7.  Frozen transformer
features are computed once, then the selected rank-64 cakes receive 6,000
balanced token-batch updates of 128 target positions.  Seed is 30,610 and
AdamW learning rate is 0.002.

The artifact contains only the selected task-cake tensors and binds the base
checkpoint/tokenizer hashes.  A fresh LayerCake model must load the unchanged
base and then the artifact; evaluation may not reuse the training model.  One
row per task is regenerated after removing the artifact and must be byte-exact
with the frozen v9 baseline.

The public gate is 132/144, at least 10/12 in every task, 12/12 abstention,
exact frozen-state hashes before and after training, and 12/12 removal replay.
A pass would prove only a bounded LayerCake sparse-adapter control.  This is
closely related to adapter fine-tuning and therefore cannot establish ABI
superiority over LoRA or distillation.

