# Host causality

The neutral stub reproduced 5,043/5,043 promoted outputs. Zero, frozen-random,
and shuffled native state cannot affect semantics because `realize` accepts no
host hidden-state/logit input. Host substitution preserves 1,681/1,681
canonical outputs while native token-unit hashes differ.

Conclusion: host-model semantic causality is falsified. The valid claim is
standalone capability-runtime portability through tested codec/conformance
adapters. See `results/abi_final_validation/host_causality.json`.
