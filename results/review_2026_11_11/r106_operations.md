# R106 operational record

No scientific thresholds, historical scores or frozen model files were changed.

- Public BLiMP metadata resolved to `3e56b06fcabca9b30822fc66435fca6b1aa40bb1`.
  Four named source files and their attribution README were downloaded; deterministic
  selection produced 200 qualification and 400 unexecuted comparison pairs.
- Both preflight focused runs passed 26 tests. The second emitted a nonfatal
  `pyreadline3` Console destructor `OSError: [WinError -1073741816]` at process
  shutdown, after pytest completed successfully; process exit code was zero.
  No model call or test failure occurred in these preflights. Both XMLs remain.
- Protocol/data commit `1b7e64f`; model/input-freeze commit `519b73b`. The GPU
  pilot began only after both commits. It ran exactly two qualification processes,
  with 200 pairs/400 source forwards in each, and stopped at the failed control.
- Both model processes exited zero. The frozen raw files match byte-for-byte.
  Operation stdout/stderr, resource receipts, execution clock and decision remain
  under `results/grammar_interface_r106/`.
- A post-run public Hugging Face metadata read matched all four tensor shards
  to the locally hashed pinned source. This was not a new model run.
- Post-run focused audit: 52 passed. Supported review runner: 364 passed and
  four skipped in the default group; 16 passed against the exact historical
  Phase 7 file fixture. No historical verifier or binding was weakened.
- The descriptive label-anomaly note is additive. Upstream labels, selection,
  source scores and the failed qualification decision are unchanged.
