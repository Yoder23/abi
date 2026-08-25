# ABI final host-causality audit

Status: `PASS_WITH_CLAIM_NARROWED_TO_STANDALONE_CAPABILITY_RUNTIME`

## Falsification result

The neutral UTF-8 stub reproduced
5043/5043
promoted outputs. Zero, frozen-random, and shuffled host states did not degrade
behavior because no native hidden-state/logit channel enters the semantic
realization function. Canonical outputs remained identical for
1681/1681
cross-host tasks even when native token-unit representations differed.

Adapter removal failed closed for
3/3 hosts.
Capability removal failed closed for
12/12
host/capability cells and identical reinstall restored the locked behavior.

## Ownership and corrected claim

- Capability package: learned generation/routing computation and semantics.
- Generic ABI runtime: integrity, canonical typed state, lifecycle, and strict UTF-8.
- Named host: frozen checkpoint conformance probe and native tokenizer units.

The Qwen and Pythia base models do **not** generate or alter capability answers.
ABI therefore proves a standalone capability-runtime package executing through
tested host codec/conformance adapters. It does not prove hidden-state transfer,
base-weight transplantation, or causal use of foreign host-model computation.

Raw result: `results/abi_final_validation/host_causality.json`.
