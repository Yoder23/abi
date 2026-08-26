# Hostile audit

`abi_v2.strict_hostile` mutates only a disposable extracted release carrying an
explicit safety marker and no `.git` directory. It tests missing/corrupt
packages, missing raw causal files/rows/hashes, stale execution code, missing raw
mount evidence, missing raw reachable-filesystem inventories, fully rehashed
missing inventory rows, missing adapters, and stale certification bindings.
Every case must fail closed, and the exact release must pass again after every
restoration.
The former hostile audit remains historical but is not controlling.

The r4 pre-public disposable run rejects 17/17 mutations, reports no trusted
scientific-boolean dependency, and restores the exact baseline evidence digest.
Evidence: `results/abi_final_validation_v2/strict_hostile_pre_public_r4.json`.

Run `abi-reproduce hostile-audit`.
