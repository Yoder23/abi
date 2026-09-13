# R30 v4 package-construction pilot

V4 consumes the immutable R30 v1 response corpus and the source-selected v3
instruction clusters. The Qwen source is absent during training and execution.
Rows are partitioned only by opaque instruction-cluster IDs. Task names and
oracle scores are joined after package generation.

Each cluster trains one small pointer-generative LayerCake cake. Nonce IDs and
large supplied numbers are dynamic pointer values and are not stored in the
fixed vocabulary. A deterministic, prompt-derived validator corrects only two
known source-error families: choosing the lower of two explicitly supplied
numbers, and applying an explicitly supplied fictional ordering rule. It does
not consume task labels or hidden answers.

This public pilot uses one seed and disclosed instruction paraphrases. It
passes only if routing is exact, all twelve packages install and execute with
the source absent, at least 132/144 held-out-content rows pass task-aware
functional scoring, each task passes at least 10/12, factual quarantine passes
12/12, every package removal is causal, and no source weights, logits, or
activations are copied into a package. A pass is not held-out English
certification and only authorizes a fresh-paraphrase replication.

