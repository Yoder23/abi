# R58 v1a serialization repair

The first R58 execution completed all 1,200 registered optimizer steps and
wrote a controller, then failed before metadata completion because the runner
looked for `manifest_sha256` at the extraction bundle's top level. That field
is not part of the bundle schema. The immutable partial output remains at
`results/sparse_termination_controller_r58/controller_v1/`; it has never been
screened and cannot be a candidate.

The v1a repair replaces only those two invalid dictionary lookups with the
already frozen broad and anchor manifest SHA-256 constants recorded by the R55
training manifest. Architecture, seed, inputs, row selection, sampler order,
steps, batches, optimizer, loss, threshold, and decision rules are unchanged.
The exact run is repeated into a new path. No validation or final-test output
was generated or inspected before this repair.
