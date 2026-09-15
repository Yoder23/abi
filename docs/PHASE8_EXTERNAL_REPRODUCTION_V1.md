# Phase 8 external reproduction handoff

This handoff can close only the independent-operator and independent-hardware
portion of Phase 8. It cannot fill Phase 2's separate human-preference forms.

## Independence boundary

The operator must not be the ABI developer and must use CPU and CUDA hardware
whose captured fingerprint differs from the development fingerprint in the
released Phase 8 manifest. A second run on the development laptop is not an
independent reproduction. The operator must not change product bytes, prompts,
comparators, thresholds, schedules, or runtime code.

## Required inputs

1. Check out the ABI packet source commit recorded in the released readiness
   manifest and the exact LayerCake commit recorded there as sibling
   directories named `abi_release` and `layercake_release`.
2. Restore every manifest entry marked `tracked: false` at its exact relative
   path. In particular, the 253,216,208-byte signed English-core archive must
   hash to `acb787b3ffa0153c57d88cd37ba81c3f00b370d4ca4937e659cd4c775851f25d`.
3. Install the frozen Python environment described by the repositories. Do
   not install a teacher model; no teacher or source base is permitted during
   inference.

## Verify before execution

From `abi_release`, run:

```powershell
C:\Python310\python.exe -m abi.capability_compiler_phase8_release_readiness `
  --protocol ABI_CAPABILITY_COMPILER_PHASE8_RELEASE_READINESS_PROTOCOL_V1070.json `
  --verify-manifest results/abi_capability_compiler_phase8/readiness_v1073/manifest.json

C:\Python310\python.exe -m abi.capability_compiler_phase8_release_readiness `
  --protocol ABI_CAPABILITY_COMPILER_PHASE8_RELEASE_READINESS_PROTOCOL_V1070.json `
  --capture-hardware phase8_external/hardware.json
```

The manifest verification must pass. `phase8_external/hardware.json` must not
have the development fingerprint recorded in the manifest.

## Fresh runtime execution

The clean checkout must not contain the registered `allocation_bounded_verify_v1063`
CPU or CUDA result directories before execution. Run CPU first and stop on any
failure:

```powershell
C:\Python310\python.exe -m abi.capability_compiler_phase7_direct_artifact_runtime `
  --protocol ABI_CAPABILITY_COMPILER_PHASE7_ALLOCATION_BOUNDED_VERIFY_PROTOCOL_V1061.json `
  --device cpu `
  --output-dir results/abi_capability_compiler_phase7_integrated/allocation_bounded_verify_v1063/cpu
```

Only if CPU passes, run CUDA:

```powershell
C:\Python310\python.exe -m abi.capability_compiler_phase7_direct_artifact_runtime `
  --protocol ABI_CAPABILITY_COMPILER_PHASE7_ALLOCATION_BOUNDED_VERIFY_PROTOCOL_V1061.json `
  --device cuda `
  --output-dir results/abi_capability_compiler_phase7_integrated/allocation_bounded_verify_v1063/cuda
```

Do not rerun a failed device until the returned evidence has been reviewed. A
failure is Phase 8 evidence, not permission to tune the system.

## Return packet

Copy `PHASE8_EXTERNAL_OPERATOR_ATTESTATION_TEMPLATE_V1.json`, fill every field,
and sign it using the operator's independently controlled signing identity.
Return these immutable files:

- `phase8_external/hardware.json`
- completed and signed operator attestation
- CPU `result.json` and `observations.jsonl`
- CUDA `result.json` and `observations.jsonl`
- stdout/stderr logs and dependency inventory

The ABI custodian must hash-lock the returned packet before unblinding or
interpretation, preregister its read-only verification, recompute every Phase 7
gate from raw rows, run hostile mutations, and preserve failures. Only a fully
passing external packet can support a Phase 8 release certificate.
