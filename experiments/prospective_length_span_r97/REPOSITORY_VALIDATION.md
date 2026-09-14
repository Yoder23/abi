# R97 repository validation

Validated: 2026-09-14

R97 is covered by a current-state test that invokes the fail-closed verifier
against all 1,400 bound raw rows. A second test independently validates the
hostile-audit and live-replay receipt hashes, rejection counts, and raw replay
digest.

Validation commands and outcomes:

- `python -m pytest -q`: 84 passed, 4 skipped, 0 failed.
- `python -m pytest -q tests/test_public_release.py tests/test_capability_pipeline.py tests/test_capability_segregation.py`: 23 passed.
- changed-test `ruff check`: passed.
- `python -m build`: source distribution and wheel built successfully.

The four skips are deliberate environment qualifications for positive R7
tests. R7 binds the exact published ABI transitive source tree and LayerCake
commit `a87a653dbdb1a4e5f713baf7bc508d508277e00d`. Current `master` contains
later additive ABI and LayerCake work and therefore must not replay R7 evidence
as though the execution tree were unchanged. Unexpected verifier failures are
not skipped; only the explicit stale-transitive-source condition is qualified.
The frozen R7 release remains reproducible through its definitive published
archive and manifests.

The stale V5 segregation implementation binding discovered during this pass
was not edited. It was superseded by
`ABI_CORE_DOMAIN_SEGREGATION_IMPLEMENTATION_CERTIFICATE_V6.json`, which binds
the current committed implementation while retaining the same bounded schema
and construction-only claim.

This validation does not expand R97 beyond bounded prospective two-hop nonce
reasoning transfer. The complete ABI moonshot remains open.
