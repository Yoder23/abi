# R23 live package-system binding repair

The first frozen live verifier failed before executing any candidate row
because it attempted to read package-system metadata from the R23 hidden result
instead of the separately frozen R21 public engine that owns the package
inventory. The failed attempt is immutable.

This repair may only join the `systems` object from that already bound public
engine into the already bound hidden result during live replay and strict
verification. It authorizes no candidate, package, scorer, source-row, gate,
or LayerCake change. The full 1,080-row GPU, 54-row CPU, and 24-package
lifecycle/corruption replay remains required.
