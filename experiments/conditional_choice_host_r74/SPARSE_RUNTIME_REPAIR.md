# R74 physical-sparsity runtime repair

After the R74 weight checkpoint was sealed but before any candidate validation
generation, static inspection found that the historical deep-adapter hook
stacked all 14 adapters at each layer and then selected one. That is logically
top-1 but not physically sparse and does not meet the R74 contract.

The repaired inference path detects a uniform route (the required condition
for single-request inference and cached decoding), indexes the one adapter
module first, and executes only its norm/down/up tensors. Mixed-route training
batches retain the historical vectorized path. Each transformer block records
the exact adapter route it executed so validation can require six calls to the
reasoning adapter and no inactive adapter call, in addition to the one terminal
cake trace.

The R74 checkpoint is unchanged. This repair is frozen before R75 candidate
access and must pass numerical-equivalence tests against the historical formula
plus live per-block route telemetry. It does not alter or waive any quality,
retention, collapse, or source-comparison gate.
