# Independent audit preservation and response

This directory preserves the independent 2026-09-14 audit exactly as
received. The Markdown report, JSON summary, and four independent receipts
must not be edited in place. Their identities are pinned by
`REPLAY_SURFACE_REPAIR_MANIFEST_V1.json` and verified by the repository test.

The audit verdict is `FULL_MOONSHOT_NOT_PROVEN`. The repository accepts that
verdict. The response repairs reproducibility defects that can be repaired
without changing evidence:

- exact historical certificates and the V6 catalog are available at both the
  curated `evidence/current/` locations and the root-relative paths frozen by
  their original verifiers;
- the supported Phase 0 and Phase 6-8 verifier tests are part of the default
  collection surface;
- the locally retained R97 parent/tokenizer and V1089 Phase 7 handoff payloads
  are placed on the Git-LFS publication surface;
- the machine-readable proof ledger now includes the bounded R97 claim and
  the correct R13-B lineage for ABI-C3.

The response does not certify the full moonshot. The Phase 5 verifier still
fails closed because six exact baseline tensors were deleted during the
recorded 2026-08-29 cleanup. Human ratings, independent different-hardware
execution, one coherent final-artifact lineage, broad English, automatic
discovery, latent purity, global minimality, and fair universal comparison
with LoRA/distillation also remain open.

Run the review-surface checks with:

```powershell
C:\Python310\python.exe -m pytest -q
C:\Python310\python.exe -m pytest -q tests/test_independent_audit_replay_surface.py
C:\Python310\python.exe -m pytest -q tests/test_capability_compiler_phase6_verify.py
C:\Python310\python.exe -m pytest -q tests/test_capability_compiler_phase7_verify.py
C:\Python310\python.exe -m pytest -q tests/test_capability_compiler_phase8_release_readiness.py tests/test_capability_compiler_phase8_local_rehearsal_verify.py
```

Then follow the independent instructions in
[`../../docs/INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md`](../../docs/INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md).
