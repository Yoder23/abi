# R21 LayerCake manifest-contract repair

The second R21 launch completed all 24 frozen training/package jobs, then the
first real `DirectCakeHost.generate` call rejected the package before executing
an evaluation row. The `.cake` manifest declared `portable_decoder` and the
correct ABI, but its input contract omitted the host-required
`mode=direct_selected_portable_decoder` field.

This repair may add exactly that input-contract field. It may not change source
rows, labels, weights, architecture, optimizer, steps, seeds, exposure, quality
metrics, gates, host code, or LayerCake code. Because training histories and
timings existed only in the failed process, one unchanged retraining is
authorized to recover complete scientific accounting. The failed 24-package
inventory remains hash-bound negative operational evidence.

All replacement packages must be signed and must pass the real production host
install/generate/remove/restore/corruption path on GPU and CPU. A positive
quality result remains ineligible for promotion until a separate fresh live
verifier re-executes the frozen packages.
