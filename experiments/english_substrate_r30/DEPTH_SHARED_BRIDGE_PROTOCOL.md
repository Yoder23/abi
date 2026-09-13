# R30 v11 depth-shared bridge control

V10 showed that 698,880 trainable parameters placed only in LayerCake's final
task cakes can fit cached features but cannot repair generation (11/144).  V11
tests the measured placement bottleneck with comparable capacity.

A single route-isolated residual contains seven rank-64 routes (689,664
parameters) and is physically reused before each of the three frozen
transformer blocks.  All LayerCake checkpoint tensors—including its existing
task cakes and output embedding—remain immutable.  Only the external residual
updates.  Training uses the same 288 cached, v7-validated responses; Qwen is
absent.  Seed 30,711, 2,400 updates, batch size four, learning rate 0.001, and
balanced route cycling are frozen before execution.

The residual tensor artifact binds the source rows, diagnosis, base checkpoint,
tokenizer, architecture, and route map.  Evaluation loads a fresh base and the
artifact.  The public gates remain 132/144, at least 10/12 per task, 12/12
abstention, exact frozen-state hashes, and causal removal on one row per task.

This is a bounded external-bridge control.  It is structurally adapter-like and
does not establish ABI superiority, general English, or minimality even if it
passes.  Failure closes nearby depth-shared rank/step/learning-rate sweeps.

The first launch failed on its first forward pass because CUDA autocast made
the residual delta fp16 while indexed placement expected the frozen fp32 hidden
dtype.  The failure is preserved.  The sole v11b repair casts that delta to the
hidden dtype before placement; no scientific setting changes.
