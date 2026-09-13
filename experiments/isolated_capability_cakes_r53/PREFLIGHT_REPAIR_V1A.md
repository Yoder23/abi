# R53 preflight repair v1a

The first frozen-command launch on 2026-09-13 terminated before an output
directory was created and before any optimizer step.  The acquisition engine's
GPU-only parent-logit-preservation allowlist predated the newly added
`capability_cakes_classifier` scope and rejected it.

The additive repair admits exactly that scope to the existing CUDA-only
preservation objective and adds positive GPU and negative CPU tests.  It does
not alter inputs, seed, training data, targets, losses, rates, schedule, model
initialization, or decision thresholds.  The original failed invocation is
preserved in the task transcript; there is no partial checkpoint to resume.
