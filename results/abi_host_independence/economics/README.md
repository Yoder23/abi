# Reuse economics

The measured ABI path certifies each host once with 0 trainable parameters,
then installs one, two, or four capabilities by verification and loading. The
cumulative timings and adapter footprints are in `reuse_economics.json`.

No matched LoRA, sequence-distillation, or ordinary fine-tuning run used this
exact host/capability/task protocol. Combining older unmatched experiments
would manufacture a comparison, so this release makes no quantitative Pareto
superiority claim. What is proven is the ABI reuse mechanism: host certification
does not recur per capability and installation performs zero training and zero
calibration.
