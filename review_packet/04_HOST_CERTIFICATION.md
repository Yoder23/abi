# Host certification

Certification uses 128 deterministic domain-neutral UTF-8 records and 15
reference vectors. A private mount namespace contains only the exact generic
capsule plus the selected host; `/mnt/c` is replaced by private `tmpfs`.
Capability archives and success IDs are physically absent, not filtered by
path or name. Adapters have zero trainable parameters and optimizer steps.

Raw evidence: `results/abi_final_validation_v2/isolated_certification_strict/`.
Independent command: `abi-reproduce certify-hosts`.
