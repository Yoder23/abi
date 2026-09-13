# R83 failure certificate

R83 is closed and **not promoted**.

- Candidate checkpoint: `be62b3129b18c7f22816a2bd4e5f27abcc24526c04c99f7750cc8bd5246c5f76`
- Corrected pre-observation binding: `1c6b820daac9667aaea4bfd9970faff1dfa46189612147d8bfd281dc90763a7f`
- Screen result file: `9677ee87635131798619ee9c3544ca7bec3261d1b84d4ceddf648e283bb8364c`
- Raw 1,400-row evidence: `4d90c0b27c271fbf3656b89dc8e26f34b407393676dd8c250b9805e2ccc5ac2e`
- Recomputable evidence digest: `c940d483d323cf4052193d786ff271a32f794cc85741df52fe09c3872f58ba32`

The pinned source passed 1,382/1,400. The unchanged R47 parent and R83
candidate each passed 2/1,400, so the paired candidate-minus-parent gain was
exactly zero (95% bootstrap interval `[0.0, 0.0]`). Source retention was
2/1,382 (0.001447); every family floor failed; eight candidate rows collapsed.
Physical sparse execution, artifact immutability, and teacher absence passed.

The all-choice diagnostic passed only 309/1,400. On 971/1,400 rows the
candidate output was byte-identical to the parent. The training curve's small
selective pointer loss therefore did not represent the deployed autoregressive
mixture: selective training optimized gate/attention surrogate terms but
excluded the actual mixture NLL. R83 establishes that a pointer alone over R47
does not acquire this causal function.

The evidence-supported successor must address both measured deficits: begin
from the R78 deep-adapter parent, whose frozen conditional selector reached
683/1,400, and train the pointer through the exact deployed mixture likelihood.
This is a changed causal architecture/objective, not an R83 hyperparameter
sweep. The full ABI moonshot remains `OPEN`.
