# R104 preserved preflight operations

2026-11-10. No scientific gate or frozen historical file was changed.

- Focused preregistration: 25 R104 tests passed; focused Ruff and docs checks passed.
- First direct `python -m pytest -q` invocation in the sandbox: 320 passed,
  four skipped, two failures and 16 setup errors. The failures were sandbox
  WinError 5 temporary-file rename permissions in existing bundle tests. The
  errors were the historical Phase 7 verifier rejecting the current sibling
  LayerCake source instead of its pinned historical source. The original XML is
  `r104-preregistration-default.xml`. This was an incorrect harness invocation,
  not a reason to weaken tests or edit historical bindings.
- The supported `scripts/run_review_tests.py` ran outside that sandbox with its
  existing exact historical fixture: 322 default tests plus 16 historical tests
  passed, four skipped. Separate XML receipts use the
  `r104-preregistration-isolated` prefix. No test was deleted or weakened.
- Protocol commit: `bcc8c03`. Input freeze commit: `cd8c222`. The generic results
  ignore rule initially prevented staging the four evidence files; explicit
  `git add -f` of only those four files resolved publication, before model work.
- Freeze binds 729 files and a 155-case evaluator audit, SHA-256
  `b9af881a74405a7f9bdf0f3e10607e82d6614032ebae3dc5fceefe9e205d89cc`.
- The GPU pilot started only after the successful supported preflight and freeze
  commit. No model retry, checkpoint mutation or training occurred in preflight.

These are operational receipts, not quality or inference certificates.
