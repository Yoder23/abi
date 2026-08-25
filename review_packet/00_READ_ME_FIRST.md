# Read me first

ABI's bounded result is a standalone capability-runtime, not a transplant into
foreign model weights. Start with the frozen candidate, causal audit, and final
raw recomputation:

1. `results/abi_final_validation/frozen_release_candidate.json`
2. `results/abi_final_validation/host_causality.json`
3. `results/abi_final_validation/headline_recomputation.json`
4. `results/abi_final_validation/release_certificate.json`

Run `abi-reproduce verify`, then follow `external_reproduction/README.md` on
independent hardware. Internal readiness is 18/18; external and human gates are
deliberately still open.
