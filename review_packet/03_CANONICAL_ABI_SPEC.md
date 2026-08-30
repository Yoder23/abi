# Canonical ABI specification

The frozen specification is `abi_v2/canonical_spec.json`; the reference
implementation is `abi_v2/canonical.py`. The R7 candidate binds their exact
hashes in `results/abi_final_validation_v2/frozen_release_candidate_r7.json`.

The public reconstruction verifier must reject a changed spec, implementation,
adapter, package, source file, raw row, manifest, or archive. Stored scientific
status booleans are not authoritative; claims are recomputed from raw content.
