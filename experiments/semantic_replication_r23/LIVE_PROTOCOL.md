# R23 fresh live verification

The R23 positive stored result and strict recomputation are immutable before
this verifier is frozen. This verifier must regenerate all 1,080 package
outputs on GPU and 54 task-covering outputs on CPU from the exact 24 R21
package archives. It also removes and restores every package and submits one
targeted `tensors.safetensors` corruption per package to a fresh host.

The replay passes only when every regenerated output is byte-identical, every
CPU output matches GPU, every package is signed and bound to the declared
direct-decoder contract, every removal makes generation unavailable, every
restoration restores exact behavior, every targeted tensor mutation is
rejected, all gates recompute, source parameters and receiver training steps
are zero, and source-model software is absent from the execution process.

No candidate, scorer, package, threshold, or source-row change is authorized.
The claim ceiling remains the bounded six-task supplied-content mechanism.
