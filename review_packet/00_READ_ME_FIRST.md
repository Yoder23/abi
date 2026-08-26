# Read me first

ABI's bounded result is a standalone capability-runtime, not a transplant into
foreign model weights. The former 18/18 packet is superseded. Start with the
repaired raw evidence:

1. `results/abi_final_validation_v2/strict_validation_r4_content_bound.json`
2. `results/abi_final_validation_v2/isolated_certification_strict_r4_content_bound/`
3. `results/abi_final_validation_v2/live_causality/`
4. `results/abi_final_validation_v2/live_isolation/`

Run `abi-reproduce verify`, then follow `external_reproduction/README.md` on
the published repaired archive. R3 public reconstruction passed but its blind
red-team failed. R4 public-manifest reconstruction and a fresh blind red-team
must pass before independent-hardware or human review opens.
