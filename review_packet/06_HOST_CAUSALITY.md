# Host causality

The repaired audit performs 3,072 new live executions—1,024 per host—under
real, neutral, zero, random, shuffled, host-removed, adapter-removed, and
capability-removed conditions. It reads neither stored matrix outputs nor
source-answer references. The real Qwen/Pythia states are computed from loaded
checkpoints. The frozen `realize` API has no host-state semantic input, so all
six positive conditions preserve output. Adapter removal fails realization and
capability removal fails generation.

Conclusion: host-model semantic causality is falsified for the tested runtime.
The valid claim is standalone capability-runtime portability through tested
codec/conformance adapters. See
`results/abi_final_validation_v2/live_causality/`.
