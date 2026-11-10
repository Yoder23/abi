# R103 software preflight observations

No R103 model call or training step occurred during these checks.

The focused R103 tests passed 20/20. The first full review inside the Windows
sandbox recorded 295 passing tests, four declared skips and two failures in
existing extraction-bundle tests. Both failed at `os.replace` with
`PermissionError: [WinError 5] Access is denied` inside sandbox-owned temporary
directories. The failed JUnit file is preserved as
`r103-preregistration-default.xml`.

The same review suite ran outside the sandbox without weakening or skipping
those tests: 297 default tests passed with four declared skips, followed by
16/16 historical Phase 7 tests in the exact-source fixture. The separate
`r103-preregistration-unsandboxed-*.xml` files preserve that result. These are
software/environment checks, not control qualification or English acquisition.
