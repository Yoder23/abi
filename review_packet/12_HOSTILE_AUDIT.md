# Hostile audit

`abi_v2.strict_hostile` mutates only a disposable extracted release carrying an
explicit safety marker and no `.git` directory. It tests missing/corrupt
packages, missing raw causal files/rows/hashes, stale execution code, missing raw
mount evidence, missing adapters, and stale certification bindings. Every case
must fail closed, and the exact release must pass again after every restoration.
The former hostile audit remains historical but is not controlling.

The pre-public extracted-archive run rejects 9/9 mutations, reports no trusted
scientific-boolean dependency, and restores the exact baseline evidence digest.
Evidence: `results/abi_final_validation_v2/strict_hostile_pre_public.json`.

Run `abi-reproduce hostile-audit`.
