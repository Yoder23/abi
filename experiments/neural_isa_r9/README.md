# R9 neural ISA

R9 tests the backend/compiler hypothesis exposed by R8. Its first experiment is
an intentionally capability-specific Pythia diagnostic. This is a search-space
divider, not a release claim.

```powershell
$config = "experiments/neural_isa_r9/configs/preregistered_v2.json"
$out = "results/neural_isa_r9/revision_002/capability_specific_pythia"
python -B -m experiments.neural_isa_r9.run_specific_diagnostic --config $config --output $out
python -B -m experiments.neural_isa_r9.verify_specific_diagnostic --config $config --run-dir $out --output "$out/verification.json"
```

Revision 001 is the preserved final-state-only negative. The run command
refuses an existing output directory. The verifier fails closed
on missing weights, raw rows, receipts, hashes, registered R8 artifacts, or
unrecomputable rows. Gate B remains closed unless the recomputed Gate A verdict
passes.
