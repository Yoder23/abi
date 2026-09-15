# Fresh-clone reproducibility audit at `bf674f8`

Date: 2026-09-14

External verdict: `REPOSITORY_NOT_REPRODUCIBLE`

Audited commit: `bf674f86f7180be54ec645237f093d8cf40046f5`

Original report: 17,680 bytes, SHA-256
`dcd530d61659c4dd2c64a68e0a9d2b75dbc6b8a12559b9cb085bf9471395e02e`

This file records the second independent review finding that triggered the
public-clone repair. It does not replace or soften the earlier full scientific
audit in this directory.

## Reproduced result

The reviewer cloned the exact commit into a new directory, pulled Git LFS,
verified a clean worktree, and obtained:

- default suite: 117 passed, 15 failed, 33 errors;
- R97 strict replay: passed from all 1,400 raw rows;
- R97 hostile replay: 12/12 mutations rejected;
- R7 public reconstruction: 17 focused tests passed;
- Phase 6 verifier tests: 17 errors;
- Phase 7 verifier tests: 16 errors;
- historical Phase 5 verifier: failed before reaching its six disclosed
  missing tensors; and
- Phase 8 named tests: 32 unit-level passes, but no integration replay of the
  actual 52-file V1089 manifest.

The reviewer found 54 absent paths on the active handoff surface: 42 files
referenced by certificates/results, six additional test/contract inputs, and
the six already disclosed deleted Phase 5 tensors. The missing historical
protocols and raw rows still existed only as ignored files in the development
checkout. Therefore the earlier local 161-test pass had consumed assets not
available to a public clone.

## Repair acceptance criteria

The successor handoff must:

1. publish the exact retained protocol, manifest, catalog, raw-row, and package
   bytes at their expected paths;
2. make the default suite pass using Git/LFS content only;
3. make Phase 4, Phase 6, and Phase 7 historical verification replay;
4. make Phase 5 fail first and specifically at the six disclosed tensors, or
   restore/rerun that phase without weakening its verifier;
5. add a real-manifest integration test covering every V1089 Phase 8 entry and
   both exact source commits;
6. preserve the human, different-hardware, broad-English, discovery, purity,
   minimality, fair-comparison, and coherent-final-artifact gates as open; and
7. undergo another sterile public-clone review before any repository-level
   reproducibility claim is accepted.

## Scientific boundary

This was a repository handoff failure. It does not invalidate the separately
reproduced bounded R7 and R97 results, and repairing it does not complete the
ABI moonshot. R7, V1089, and R97 remain separate artifact lineages.
