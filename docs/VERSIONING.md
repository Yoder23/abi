# Versions, review snapshots and scientific releases

| Identity | Current value | Scope |
| --- | --- | --- |
| Python package | `0.4.0a1` | Supported alpha compiler API |
| Controlling technical release | `abi-final-validation-v2-repaired-r7-2026-08-30` | Bounded R7 runtime/conformance proof |
| Latest public research review snapshot | `abi-research-r103-2026-11-10` | Raw control qualification failed; recipient comparison not run |
| Previous research snapshot | `abi-research-r102-2026-11-10` | Negative coverage/exposure factorial; no English-core promotion |
| Earlier diagnostic snapshot | `abi-research-r101-2026-11-05` | Execution/continuation diagnostic; no English-core promotion |
| Earlier capacity snapshot | `abi-research-r100-2026-11-05` | Preserved R99-D1/R100 negative capacity control and public reconstruction |
| Earlier research snapshot | `abi-research-2026-11-05` | Preserved R98/R99 research and public reconstruction |
| Default branch | `master` | Moving development and research history |

The package API has not changed in this research update, so its package version
stays unchanged. Experiment identifiers, canonical artifact/interface versions,
package versions and Git tags are different things. Never use the highest
experiment number as a measure of scientific progress.

Review tags are annotated and immutable. Do not replace a tag or silently
overwrite evidence under its old path. Verify remote availability before citing
a local tag as published:

```bash
git fetch origin --tags
git show --no-patch abi-research-r100-2026-11-05
git show --no-patch abi-research-r101-2026-11-05
git rev-parse 'abi-research-r101-2026-11-05^{commit}'
git lfs pull
git lfs fsck
```

## Review checks

The supported API checks in [Getting started](GETTING_STARTED.md) do not require
a sibling research host. Full research replay additionally requires published
Git LFS payloads and the LayerCake Git history named by each protocol.

```bash
python scripts/run_review_tests.py
python scripts/check_docs.py
python -m build
```

The review runner executes the default suite except the exact historical Phase 7
module, then executes that unchanged module against 46 isolated files. It reads
historical LayerCake bytes from commit
`662c5a9b7264a1a5478c9dfb656f35c450e2504f`, validates every registered binding,
and never checks out or modifies the live sibling. Missing Git objects, payloads
or hashes fail closed. Direct `pytest` against a newer sibling still rejects
stale Phase 7 bindings; this is intentional, not a gate to relax.

R99 snapshot local review: 203 tests pass with four declared skips, plus 16/16
historical Phase 7 tests in their exact-source fixture. See the
[default receipt](../results/review_2026_11_05/abi-seal-default.xml) and
[Phase 7 receipt](../results/review_2026_11_05/abi-seal-phase7.xml). These checks are
not live inference, human ratings or independent-hardware reproduction.

R100 local review: 220 default tests pass with four declared skips, plus 16/16
historical Phase 7 tests. See the [default receipt](../results/review_2026_11_05/r100-preflight-default.xml),
[Phase 7 receipt](../results/review_2026_11_05/r100-preflight-phase7.xml) and
[57-test supported CI surface](../results/review_2026_11_05/r100-ci-surface.xml).
The package version is unchanged because no supported API changed.

R102 local review: 271 default tests pass with four declared skips, plus 16/16
exact historical Phase 7 tests. The supported CI list passes 108 locally. The
annotated local tag points to `d5a8957b152ce883779436103bd61a7ba4cd2f35`; its
tag-object identity is `1ef35b43aa14c23e1f3836e07009dcf0bd5ee1b5`. See the
[R102 packet](R102_COVERAGE_EXPOSURE.md) for complete dependency custody and
the negative scientific result. The user subsequently authorized publication;
the same tag and complete payload set are now public. Original inventory fields
retain their historical local-only state. No package API or LayerCake source changed.

## Promotion policy

R103 is sealed at source commit `cc6b59cd8b0b72730a0ae847015c42a9c7a22ab5`,
annotated tag object `db1d4b999e6351b28e552115603516a3d3f1c1fc`. Its local
supported review passes 313 tests with four declared skips; 12 additive
completed-evidence checks pass separately. The [R103 packet](R103_RECIPIENT_INTERFACE.md)
records the 41/48 prerequisite failure and complete public manifest. No recipient
ran, no new model checkpoint was made, and the package API remains unchanged.

R101 local review: 246 default tests pass with four declared skips, plus 16/16
exact historical Phase 7 tests. The supported CI list passes 83. See the
[R101 packet](R101_EXECUTION_CONTINUATION.md), [default report](../results/review_2026_11_05/r101-default.xml),
[Phase 7 report](../results/review_2026_11_05/r101-phase7.xml) and
[CI-surface report](../results/review_2026_11_05/r101-ci-surface.xml).
These checks and the same-machine diagnostic do not fill independent external
or product-quality gates. No package version change is warranted.

- Change the package version for supported API changes and describe migration.
- Assign a new artifact/interface identity for incompatible execution contracts.
- Keep every research failure and preregister the next bounded intervention.
- Publish exact raw evidence and required artifacts before claiming public replay.
- Require scientific gates separately from tests, tags, builds and CI status.

R7, V1089, R97, R98/C1, R99, R100, R101, R102 and R103 retain separate claim scopes. The whole-moonshot verdict
is `FULL_MOONSHOT_NOT_PROVEN`. [LayerCake](https://github.com/Yoder23/layercake)
owns host certification; ABI does not inherit its runtime certificates.
