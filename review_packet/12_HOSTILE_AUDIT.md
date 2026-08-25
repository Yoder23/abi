# Hostile audit

`results/abi_final_validation/hostile_release_verification.json` records
mutations of capability tensor payload, adapter, certification data, locks, ABI
version, host checkpoint binding, evaluator, decoding policy, teacher absence,
reveal state, runtime manifest, human packet, and external manifest. It also
tests four prohibited actions. Every case must fail closed.

Run `abi-reproduce hostile-audit`.
