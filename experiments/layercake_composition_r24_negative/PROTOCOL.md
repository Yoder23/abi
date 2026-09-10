# R24 additive negative-result verification

The original R24 result is immutable. This additive verifier reruns every
stored execution from the six frozen packages and certifies the negative
result without changing a package, model, prompt, seed, score, or gate.

The original lifecycle gate compared a restored package's output to the gold
answer. That confounds restoration identity with model task accuracy. This
verifier preserves that original failed gate exactly, and separately reports
the correctly scoped host lifecycle comparison: restored output versus the
same package's pre-removal output on the same prompt. The latter is diagnostic
only and cannot turn the overall R24 failure into a pass.

Passing this verifier means only that R24's negative result is exactly live-
reproducible and correctly decomposed. R24 remains failed unless all original
preregistered gates pass, which they did not.
